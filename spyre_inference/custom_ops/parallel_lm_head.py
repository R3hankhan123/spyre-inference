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

"""Spyre OOT replacement for ParallelLMHead.

Spyre Device Constraints:
    - Tensor Parallelism: TP>=1 supported with vocabulary sharding (each rank
      computes logits for its vocab partition)
    - Quantization: Fp8Config supported (resolves to UnquantizedEmbeddingMethod).
      Other quantization methods raise NotImplementedError. When the body's
      linears are FP8, ``install_fp8_lm_head`` quantizes the head too.
"""

from typing import cast

import torch
from torch.nn import Parameter
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from .fp8_linear_kernel import (
    FP8_E4M3FN_MAX,
    _dma_checkpoint_qfp8wt,
    _fp8_gemm_epilogue,
    _normalize_weight_scale,
    _spyre_fp8_kernel,
)
from .lazy_compile import CompileOutermost, maybe_compile
from .linear import SpyreTransposedWeightMethod

logger = init_logger(__name__)


class SpyreUnquantizedLMHeadMethod(
    CompileOutermost, SpyreTransposedWeightMethod, UnquantizedEmbeddingMethod
):
    """LM-head projection via the shared transposed-weight fast path.

    No graph encloses it: per-block wraps only the block ModuleList, and a whole-model
    compile intercepts ``__call__``, not ``compute_logits``. Compiled ``dynamic=False``,
    so rows must arrive padded onto one of the ``logits_row_buckets``.
    """

    WEIGHT_T_ATTR = "padded_weight_t"
    ROW_ALIGN = 64 * 32

    @maybe_compile
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return SpyreTransposedWeightMethod.apply(self, layer, x, bias)


class SpyreFp8LMHeadMethod(SpyreUnquantizedLMHeadMethod):
    """FP8 LM-head projection: ``padded_weight_t`` in QFP8WT, activations per token.

    The same ops as ``SpyreFp8LinearKernel.apply_weights``, compiled as the head's own
    graph like the fp16 projection.
    """

    @maybe_compile
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise NotImplementedError("SpyreFp8LMHeadMethod does not support embedding_bias.")
        scale_a = torch.ops.spyre.quantscalepertokenfp8(
            x,  # ty: ignore[invalid-argument-type]
            FP8_E4M3FN_MAX,  # ty: ignore[invalid-argument-type]
        )
        x_fp8 = torch.ops.spyre.quantize_fp8_with_scale(
            x,  # ty: ignore[invalid-argument-type]
            scale_a,  # ty: ignore[invalid-argument-type]
        )
        y = torch.ops.spyre.scaled_mm(
            x_fp8,  # ty: ignore[invalid-argument-type]
            layer.padded_weight_t,  # ty: ignore[invalid-argument-type]
            out_dtype=torch.float16,  # ty: ignore[invalid-argument-type]
        )
        out = _fp8_gemm_epilogue(y, scale_a, cast(torch.Tensor, layer.weight_scale), None)
        padding = cast(int, layer.spyre_row_padding)
        return out[..., :-padding] if padding else out


def install_fp8_lm_head(model: torch.nn.Module) -> None:
    """Quantize each ``SpyreParallelLMHead`` to FP8 when the body's linears are FP8.

    Upstream leaves the head unquantized for every FP8 config, so the checkpoint's fp16
    ``padded_weight_t`` is quantized here, one scale per vocab column. Runs before the
    device move, on the host copy: the result is DMA'd as QFP8WT, which ``.to`` skips.
    A tied head promoted at its first logits call stays fp16.
    """
    if not any(_spyre_fp8_kernel(m) is not None for m in model.modules()):
        return
    for module in model.modules():
        if not isinstance(module, SpyreParallelLMHead) or type(module.quant_method) is not (
            SpyreUnquantizedLMHeadMethod
        ):
            continue
        wt = cast(torch.Tensor, module.padded_weight_t).data.float()
        scale = (wt.abs().amax(dim=0) / FP8_E4M3FN_MAX).to(torch.float16)
        # The vocab pad columns are zero; any scale stores them as zero.
        scale[scale == 0] = 1.0
        w8 = (wt / scale.float()).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
        module.padded_weight_t = Parameter(_dma_checkpoint_qfp8wt(w8), requires_grad=False)
        module.weight_scale = Parameter(_normalize_weight_scale(w8, scale), requires_grad=False)
        # A class swap, not a new instance: keeps the compile mode sampled at construction,
        # when the vLLM config context was live.
        module.quant_method.__class__ = SpyreFp8LMHeadMethod
        logger.info("LM head %s projects in FP8", tuple(w8.shape))


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """Out-of-tree (OOT) ParallelLMHead implementation for IBM's Spyre device.

    The projection lives in `SpyreUnquantizedLMHeadMethod.apply`, reached via
    `LogitsProcessor._apply_head` -> `lm_head.quant_method.apply`. The base
    `ParallelLMHead.forward` raises and is unused.
    """

    def _apply(self, fn, recurse=True):
        # The GEMM reads `padded_weight_t`; once it exists `weight` is runtime-dead, so
        # skip moving it to device. Until then `weight` is still live: don't skip it.
        if not hasattr(self, "padded_weight_t"):
            return super()._apply(fn, recurse=recurse)
        weight = self._parameters.pop("weight", None)
        try:
            return super()._apply(fn, recurse=recurse)
        finally:
            if weight is not None:
                self._parameters["weight"] = weight

    def __init__(self, *args, **kwargs):
        # Pad the vocab so every TP shard is a whole number of 64-element sticks, which
        # lets the logits gather stay on device (see SpyreCommunicator.all_gather).
        kwargs["padding_size"] = 64 * get_tensor_model_parallel_world_size()
        super().__init__(*args, **kwargs)

        # Only UnquantizedEmbeddingMethod supported. Fp8Config resolves to it;
        # other quantization methods are rejected.
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support {type(self.quant_method).__name__}."
            )

        logger.debug("Building SpyreParallelLMHead with TP size %d ", self.tp_size)

        # Set the custom quantization method to route through spyre
        self.quant_method = SpyreUnquantizedLMHeadMethod()
