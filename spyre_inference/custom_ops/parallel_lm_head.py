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

Executes the lm_head matmul (hidden_states @ weight.T) on Spyre using
chunked computation to avoid hardware limits.

Architecture:
    - OOT Registration: @ParallelLMHead.register_oot() replaces upstream
      at instantiation
    - forward_oot(): Entry point for OOT dispatch, calls chunk.py helpers
    - quant_method override: SpyreUnquantizedLMHeadMethod.apply() routes
      through forward_oot() so LogitsProcessor uses the Spyre path

Spyre Device Constraints:
    - No Tensor Parallelism (TP) support: tp_size > 1 raises NotImplementedError
    - No quantization support: only UnquantizedEmbeddingMethod is replaced

References:
    - Upstream ParallelLMHead:
      vllm/model_executor/layers/vocab_parallel_embedding.py
    - Chunking logic: spyre_inference/custom_ops/chunk.py
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from .chunk import forward_lm_head_oot, setup_lm_head_padding

logger = init_logger(__name__)


class SpyreUnquantizedLMHeadMethod(UnquantizedEmbeddingMethod):
    """Routes ParallelLMHead logits through layer.forward_oot()."""

    def apply(self, layer, x, bias=None):
        return layer.forward_oot(x, bias)

    def process_weights_after_loading(self, layer):
        super().process_weights_after_loading(layer)
        setup_lm_head_padding(layer)


@ParallelLMHead.register_oot(name="ParallelLMHead")
class SpyreParallelLMHead(ParallelLMHead):
    """OOT ParallelLMHead that executes the lm_head matmul on Spyre.

    Weights reside on Spyre after model.to(spyre_device).
    The quant_method is replaced so that LogitsProcessor._get_logits()
    routes through forward_oot, which handles device conversion
    and runs F.linear on Spyre.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        quant_config = kwargs.get("quant_config")
        if quant_config is not None:
            raise NotImplementedError(
                "SpyreParallelLMHead does not support quantization "
                f"(quant_config={quant_config}). Only quant_config=None is supported."
            )

        if self.tp_size > 1:
            raise NotImplementedError(
                f"SpyreParallelLMHead does not support Tensor Parallelism "
                f"(tp_size={self.tp_size}). Only tp_size=1 is supported."
            )

        logger.debug("Building custom ParallelLMHead for Spyre")

        # Set the custom quantization method to route through spyre
        self.quant_method = SpyreUnquantizedLMHeadMethod()

    def forward_oot(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """OOT forward pass — lm_head matmul on Spyre.

        Called by SpyreUnquantizedLMHeadMethod.apply() from within
        LogitsProcessor._get_logits(). Converts x (arriving on cpu)
        to the weight device (residing on spyre), and runs chunked F.linear
        via forward_lm_head_oot.

        Args:
            x: Hidden states tensor [num_tokens, hidden_dim]
            bias: Optional bias tensor

        Returns:
            Logits tensor [num_tokens, vocab_size] on the input device
        """
        return forward_lm_head_oot(self, x, bias)
