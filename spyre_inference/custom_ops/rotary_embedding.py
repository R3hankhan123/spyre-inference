# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre OOT replacement for RotaryEmbedding (CPU fallback).

This module provides Spyre-device-specific replacements for the rotary
positional embedding classes used in attention layers:

    - SpyreApplyRotaryEmb             — replaces ApplyRotaryEmb
      (vllm/model_executor/layers/rotary_embedding/common.py)
    - SpyreRotaryEmbedding            — replaces RotaryEmbedding
      (vllm/model_executor/layers/rotary_embedding/base.py)
    - SpyreYaRNScalingRotaryEmbedding — replaces YaRNScalingRotaryEmbedding
      (vllm/model_executor/layers/rotary_embedding/yarn_scaling_rope.py)

Architecture mirrors linear.py: a shared base class (SpyreRotaryEmbeddingBase)
holds the common CPU-fallback forward logic, and each OOT subclass inherits
from it via MRO alongside the upstream vLLM class.

Limitations:
    *) All ops run on CPU today: spyre lacks dynamic tensor indexing,
    such as `index_select`, `chunk`, last-dim slicing, etc.
    *) FP16 multiply diverges from CPU, which flips tokens under greedy decoding.
    Thus, for the moment, the multiplications in `SpyreApplyRotaryEmb` need to
    run on CPU.
    *) No promotion of the data types, as this is not yet supported in torch-spyre.

"""

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import (
    YaRNScalingRotaryEmbedding,
)

from .utils import convert

logger = init_logger(__name__)


@ApplyRotaryEmb.register_oot(name="ApplyRotaryEmb")
class SpyreApplyRotaryEmb(ApplyRotaryEmb):
    """Spyre OOT variant of ApplyRotaryEmb.

    Template: vllm/model_executor/layers/rotary_embedding/common.py
    Math is identical to `ApplyRotaryEmb.forward_static`; runs on CPU
    and restores the input device on the output.
    """

    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        device = x.device
        orig_dtype = x.dtype
        x_cpu = convert(x, device="cpu")
        cos_cpu = convert(cos, device="cpu")
        sin_cpu = convert(sin, device="cpu")

        # Run the rotation in cos/sin precision (float32 for YaRN,
        # float16 for plain RoPE) so that YaRN's fine-grained phase
        # differences between adjacent decode positions are preserved.
        compute_dtype = cos_cpu.dtype
        cos_cpu = cos_cpu.unsqueeze(-2)
        sin_cpu = sin_cpu.unsqueeze(-2)
        x_cpu = x_cpu.to(compute_dtype)

        if self.is_neox_style:
            x1, x2 = torch.chunk(x_cpu, 2, dim=-1)
        else:
            x1 = x_cpu[..., ::2]
            x2 = x_cpu[..., 1::2]

        o1 = x1 * cos_cpu - x2 * sin_cpu
        o2 = x2 * cos_cpu + x1 * sin_cpu

        if self.is_neox_style:
            output = torch.cat((o1, o2), dim=-1)
        else:
            output = torch.stack((o1, o2), dim=-1).flatten(-2)

        return convert(output, dtype=orig_dtype, device=device)


class SpyreRotaryEmbeddingBase:
    """Shared initialization and CPU-fallback forward for Spyre rotary embeddings.

    Mirrors `RotaryEmbedding.forward_static`. Runs on CPU today; the
    OOT structure is preserved for a later on-device migration.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        logger.debug("Building Spyre %s", self.__class__.__name__)

        # Recompute and keep the CPU cache in float32.  The upstream
        # __init__ converts the cache to model dtype (float16) which
        # loses too much precision for the trig values — especially
        # for YaRN's frequency-corrected cos/sin but also noticeable
        # for standard RoPE across many layers.
        # Use .clone() to ensure this is an independent copy, not an
        # alias of self.cos_sin_cache — important because the upstream
        # buffer may be moved to device by module.to(device) later,
        # which would silently invalidate index_select on it.
        # Plain attribute (not a buffer) so module.to(device) cannot
        # relocate it -- index_select requires a CPU-resident cache.
        self._cos_sin_cache_cpu = self.cos_sin_cache.clone().to(dtype=torch.float32, device="cpu")

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # positions arrive on spyre; query/key arrive on CPU (qkv split
        # in GraniteAttention runs on CPU). Restore their original
        # devices on the outputs so downstream layers see no shift.
        query_device = query.device
        query_dtype = query.dtype
        key_device = key.device if key is not None else None

        positions_cpu = convert(positions.flatten().long(), device="cpu")

        # Guard against out-of-bounds indexing
        max_cache_idx = self._cos_sin_cache_cpu.shape[0] - 1
        if positions_cpu.max() > max_cache_idx:
            logger.warning(
                "positions max=%d exceeds cos_sin_cache length=%d; clamping.",
                positions_cpu.max().item(),
                max_cache_idx + 1,
            )
            positions_cpu = positions_cpu.clamp(0, max_cache_idx)

        query_cpu = convert(query, device="cpu")
        key_cpu = convert(key, device="cpu") if key is not None else None

        num_tokens = positions_cpu.shape[0]

        cos_sin_cpu = self._cos_sin_cache_cpu.index_select(0, positions_cpu)
        cos_cpu, sin_cpu = cos_sin_cpu.chunk(2, dim=-1)

        query_cpu = self._rope(query_cpu, cos_cpu, sin_cpu, num_tokens)
        if key_cpu is not None:
            key_cpu = self._rope(key_cpu, cos_cpu, sin_cpu, num_tokens)

        return (
            convert(query_cpu, device=query_device),
            convert(key_cpu, device=key_device) if key_cpu is not None else None,
        )

    def _rope(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        x_shape = x.shape
        x = x.view(num_tokens, -1, self.head_size)

        if self.rotary_dim == self.head_size:
            return self.apply_rotary_emb(x, cos, sin).reshape(x_shape)

        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        x_rot = self.apply_rotary_emb(x_rot, cos, sin)
        return torch.cat((x_rot, x_pass), dim=-1).reshape(x_shape)


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(SpyreRotaryEmbeddingBase, RotaryEmbedding):
    """Spyre OOT variant of RotaryEmbedding (TP=1 only).

    Mirrors `RotaryEmbedding.forward_static`. Runs on CPU today; the
    OOT structure is preserved for a later on-device migration.
    """


@YaRNScalingRotaryEmbedding.register_oot(name="YaRNScalingRotaryEmbedding")
class SpyreYaRNScalingRotaryEmbedding(SpyreRotaryEmbeddingBase, YaRNScalingRotaryEmbedding):
    """Spyre OOT variant of YaRNScalingRotaryEmbedding (TP=1 only).

    YaRN (Yet another RoPE extensioN) uses frequency-domain interpolation
    with magnitude scaling to extend context length.  We explicitly
    recompute the cos/sin cache with YaRN's modified inv_freq and mscale
    rather than relying on MRO resolution, which could silently fall back
    to the plain RoPE cache.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Explicitly compute YaRN-scaled cache in float32 on CPU.
        inv_freq = convert(
            self._compute_inv_freq(self.scaling_factor),
            device="cpu",
            dtype=torch.float32,
        )

        num_positions = int(self.max_position_embeddings * self.scaling_factor)
        t = torch.arange(num_positions, dtype=torch.float32, device="cpu")

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale

        self._cos_sin_cache_cpu = torch.cat((cos, sin), dim=-1)
