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

"""Multi-LoRA linear layers on Spyre, as two dense matmuls and a token mask.

Upstream's punica kernels gather each token's adapter (bgmv) or group tokens by
adapter (sgmv). On Spyre the gather fails to compile for a single-slot source and
``_grouped_mm`` is 2-6x a base matmul and raises on skewed batches. Instead every
LoRA linear computes, inside the block graph::

    y += (x @ A_all * mask) @ B_all

``A_all`` ``[in, S*L*r]`` and ``B_all`` ``[S*L*r, out]`` hold all ``L`` adapter slots
of the layer's ``S`` output slices (``B_all`` block-diagonal over the slices), and
``mask`` ``[T, S*L*r]`` keeps, per token, only its slot's ``r`` columns of each slice.
The cost is flat in ``L*r`` (one stick at 4x16) and the graph is the same for every
mix of adapters, so it compiles once per token bucket.

Upstream keeps writing adapters into its host ``lora_{a,b}_stacked`` (``set_lora``
slices and zero-fills, which Spyre can't do in place); the layer then re-uploads its
packed ``A_all``/``B_all`` buffers whole. The mask is rebuilt on the host each step
by ``SpyrePunicaWrapper.update_metadata`` and copied into one device buffer per slice
count, which the graphs read rows ``[:T]`` of.
"""

from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.lora.layers import (
    BaseLayerWithLoRA,
    LogitsProcessorWithLoRA,
    LoRAMapping,
    VocabParallelEmbeddingWithLoRA,
)
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
from vllm.lora.utils import replace_submodule


class SpyrePunicaWrapper(PunicaWrapperBase):
    """Token-to-slot metadata for ``SpyreLinearLoRAMixin``, as device masks."""

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: torch.device | str,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_num_batched_tokens, max_batches, device)
        self._rank = 0
        self._col_slots: dict[int, torch.Tensor] = {}
        self._host_masks: dict[int, torch.Tensor] = {}
        self._masks: dict[int, torch.Tensor] = {}
        self._filled = 0
        self._last_indices: torch.Tensor | None = None

    def allocate(
        self,
        rows: int,
        num_slots: int,
        rank: int,
        slice_counts: set[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """One ``[rows, S*num_slots*rank]`` mask per slice count ``S``, zero (no LoRA)."""
        self._rank = rank
        slot_of_col = torch.arange(num_slots * rank) // rank
        for s in sorted(slice_counts):
            width = s * num_slots * rank
            self._col_slots[s] = slot_of_col.repeat(s)
            self._host_masks[s] = torch.zeros(rows, width, dtype=dtype)
            self._masks[s] = torch.zeros(rows, width, dtype=dtype, device=device)

    def mask(self, num_slices: int, num_tokens: int) -> torch.Tensor:
        return self._masks[num_slices][:num_tokens]

    def update_metadata(
        self,
        mapping: LoRAMapping,
        lora_index_to_id: list[int | None],
        max_loras: int,
        vocab_size: int,
        **kwargs: Any,
    ) -> None:
        self._update_base_metadata(mapping, lora_index_to_id, max_loras, vocab_size)
        self.is_prefill = mapping.is_prefill
        indices = self.token_lora_indices
        if self._last_indices is not None and torch.equal(indices, self._last_indices):
            return
        n = indices.shape[0]
        for s, host in self._host_masks.items():
            host[:n] = indices[:, None] == self._col_slots[s][None, :]
            host[n : self._filled] = 0
            rows = max(n, self._filled)
            self._masks[s][:rows].copy_(host[:rows])
        self._filled = n
        self._last_indices = indices.clone()

    def _unsupported(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Spyre applies LoRA in SpyreLinearLoRAMixin; only linear layers are supported."
        )

    add_shrink = _unsupported
    add_expand = _unsupported
    add_lora_embedding = _unsupported
    add_lora_linear = _unsupported
    add_lora_logits = _unsupported
    apply_lora_full_linear = _unsupported


def _pack(layer: BaseLinearLayerWithLoRA) -> tuple[torch.Tensor, torch.Tensor]:
    """``A_all [in, S*L*r]`` and block-diagonal ``B_all [S*L*r, out]`` from the host stacks."""
    num_slots, _, rank, in_features = layer.lora_a_stacked[0].shape
    block = num_slots * rank
    a_all = torch.cat([a[:, 0].reshape(block, in_features).t() for a in layer.lora_a_stacked], 1)
    b_all = a_all.new_zeros(len(layer.lora_b_stacked) * block, sum(layer.output_slices))
    offset = 0
    for s, (b, width) in enumerate(zip(layer.lora_b_stacked, layer.output_slices, strict=True)):
        assert b.shape[2] == width, (type(layer).__name__, b.shape, width)
        b_all[s * block : (s + 1) * block, offset : offset + width] = (
            b[:, 0].transpose(1, 2).reshape(block, width)
        )
        offset += width
    return a_all.contiguous(), b_all


class SpyreLinearLoRAMixin:
    """Mixed in before an upstream ``*LinearWithLoRA`` class by ``prepare_lora_for_spyre``."""

    _spyre_defer_sync = False

    def set_lora(self, index: int, lora_a: Any, lora_b: Any) -> None:
        # Upstream's set_lora resets the slot first; upload once, after the write.
        self._spyre_defer_sync = True
        try:
            super().set_lora(index, lora_a, lora_b)
        finally:
            self._spyre_defer_sync = False
        self._sync_spyre_lora()

    def reset_lora(self, index: int) -> None:
        super().reset_lora(index)
        if not self._spyre_defer_sync:
            self._sync_spyre_lora()

    def _sync_spyre_lora(self) -> None:
        a_all, b_all = _pack(self)
        self.spyre_lora_a.copy_(a_all)
        self.spyre_lora_b.copy_(b_all)

    def _apply_lora_to_output(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        original_shape = output.shape if output.ndim == 3 else None
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)
        mask = self.punica_wrapper.mask(len(self.lora_a_stacked), x.shape[0])
        output = output + (x @ self.spyre_lora_a * mask) @ self.spyre_lora_b
        if original_shape is not None:
            output = output.reshape(original_shape)
        return output


_SPYRE_CLASSES: dict[type, type] = {}


def _spyre_class(cls: type) -> type:
    if cls not in _SPYRE_CLASSES:
        _SPYRE_CLASSES[cls] = type(f"Spyre{cls.__name__}", (SpyreLinearLoRAMixin, cls), {})
    return _SPYRE_CLASSES[cls]


class _UnsupportedLoRAMixin:
    def set_lora(self, index: int, lora_a: Any, lora_b: Any) -> None:
        raise NotImplementedError(
            f"The adapter trains {type(self).__name__.removeprefix('Unsupported')}, which "
            "Spyre does not support; only linear-layer adapters are."
        )


def _unsupported_class(cls: type) -> type:
    if cls not in _SPYRE_CLASSES:
        _SPYRE_CLASSES[cls] = type(f"Unsupported{cls.__name__}", (_UnsupportedLoRAMixin, cls), {})
    return _SPYRE_CLASSES[cls]


def _check_no_padding(vllm_config: VllmConfig) -> None:
    hf_config = vllm_config.model_config.hf_config
    for cfg in (hf_config, getattr(hf_config, "text_config", None)):
        for attr in ("_spyre_orig_head_dim", "_spyre_orig_intermediate_size"):
            if cfg is not None and getattr(cfg, attr, None) is not None:
                # Adapters load into the leading rows/columns, but a padded head lays out
                # each head with its own trailing pad, so the deltas would land shifted.
                raise NotImplementedError(
                    f"LoRA is not supported on Spyre for a model whose {attr[12:]} is "
                    "padded for stick alignment."
                )


def prepare_lora_for_spyre(model: nn.Module, vllm_config: VllmConfig, device: torch.device) -> None:
    """Swap each LoRA linear to its Spyre class and size the wrapper's masks.

    Runs on the host model, after upstream wrapped the LoRA layers and before the
    device move, which carries the packed buffers registered here along.
    """
    _check_no_padding(vllm_config)
    layers: list[BaseLinearLayerWithLoRA] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, BaseLayerWithLoRA):
            continue
        if isinstance(module, (VocabParallelEmbeddingWithLoRA, LogitsProcessorWithLoRA)):
            # Wrapped whenever the model lists embed_tokens/lm_head as LoRA modules,
            # whatever the adapter trains. Run the plain layer; the manager keeps the
            # wrapper, so an adapter that does train it fails in set_lora.
            replace_submodule(model, name, module.base_layer)
            module.__class__ = _unsupported_class(type(module))
            continue
        if not isinstance(module, BaseLinearLayerWithLoRA):
            raise NotImplementedError(
                f"LoRA on {name} ({type(module).__name__}) is not supported on Spyre; "
                "only linear layers are."
            )
        module.__class__ = _spyre_class(type(module))
        a_all, b_all = _pack(module)
        module.register_buffer("spyre_lora_a", a_all, persistent=False)
        module.register_buffer("spyre_lora_b", b_all, persistent=False)
        layers.append(module)
    if not layers:
        return

    first = layers[0].lora_a_stacked[0]
    num_slots, _, rank, _ = first.shape
    compile_sizes = vllm_config.compilation_config.compile_sizes or []
    rows = max([vllm_config.scheduler_config.max_num_batched_tokens, *map(int, compile_sizes)])
    wrappers = {id(m.punica_wrapper): m.punica_wrapper for m in layers}
    for wrapper in wrappers.values():
        assert isinstance(wrapper, SpyrePunicaWrapper), type(wrapper)
        wrapper.allocate(
            rows,
            num_slots,
            rank,
            {len(m.lora_a_stacked) for m in layers if m.punica_wrapper is wrapper},
            first.dtype,
            device,
        )
