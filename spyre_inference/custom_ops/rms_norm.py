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

"""Spyre-specific RMSNorm implementation using out-of-tree (OOT) registration.

This module provides a custom RMSNorm layer for IBM's Spyre device,
replacing the upstream vLLM implementation (vllm/model_executor/layers/layernorm.py)
when instantiated.

Architecture:
    - OOT Registration: @RMSNorm.register_oot() replaces upstream at instantiation
    - forward_oot(): Entry point for OOT dispatch, fully transparent to the outer
      torch.compile graph (no opaque custom-op boundary)

Spyre Device Constraints:
    - Computations performed in torch.float16
    - Epsilon as tensor: Instead of a scalar, a tensor is created via torch.full()

Limitations:
    Currently the implementation does NOT use dtype promotion, as this is not
    yet supported in torch-spyre.

References:
    - Upstream RMSNorm: vllm/model_executor/layers/layernorm.py
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

from .utils import convert

logger = init_logger(__name__)


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """Out-of-tree (OOT) RMSNorm implementation for IBM's Spyre device.

    This replaces the upstream vLLM RMSNorm (vllm/model_executor/layers/layernorm.py)
    when instantiated, providing Spyre-specific optimizations and device handling.

    Preserves input dtype and device.
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyreRMSNorm layer."""
        super().__init__(*args, **kwargs)

        logger.debug("Building custom RMS norm")

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Spyre-optimized RMS norm forward pass.

        Computation performed in original dtype (typically fp16) on Spyre device
        for maximum performance.

        Args:
            x: Input tensor [batch_size, hidden_size]
            residual: Optional residual tensor

        Returns:
            Normalized output, or (output, residual) tuple if residual provided
        """
        orig_dtype = x.dtype
        orig_device = x.device

        if self.variance_size_override is not None:
            raise NotImplementedError("variance_size_override not yet implemented")

        x = convert(x, device="cpu", dtype=torch.float32)

        if residual is not None:
            residual = convert(residual, device="cpu", dtype=torch.float32)
            x = x + residual
            residual_out = convert(x, device=orig_device, dtype=orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        if self.has_weight:
            weight = convert(self.weight.data, device="cpu", dtype=torch.float32)
            x = x * weight

        # Convert back to original dtype and device
        x = convert(x, device=orig_device, dtype=orig_dtype)

        if residual is None:
            return x
        return x, residual_out
