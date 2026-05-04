# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific linear layer implementations using out-of-tree (OOT) registration.

This module provides Spyre-device-specific replacements for the parallel linear
layer classes used inside MLP blocks:

    - SpyreMergedColumnParallelLinear  — replaces MergedColumnParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreQKVParallelLinear          — replaces QKVParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreRowParallelLinear          — replaces RowParallelLinear
      (vllm/model_executor/layers/linear.py)

Since tensor_parallel=1 is assumed, all classes are functionally equivalent
to F.linear(input, weight, bias) and share the same implementation pattern.

Spyre Device Constraints:
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.
    - Tensor parallelism: TP=1 assumed (single Spyre device)

References:
    - Upstream linear layers: vllm/model_executor/layers/linear.py
    - Pattern reference:      spyre_inference/custom_ops/rms_norm.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from .utils import convert

logger = init_logger(__name__)


class SpyreLinearBase:
    """Shared implementation for Spyre linear layers at TP=1."""

    def __init__(self, *args, **kwargs):
        """Common initialization for Spyre linear layers."""
        super().__init__(*args, **kwargs)
        if self.tp_size > 1:
            raise NotImplementedError(
                f"{self.__class__.__name__} only supports TP=1, got TP={self.tp_size}"
            )

        logger.debug("Building custom %s", self.__class__.__name__)

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        logger.warning_once(
            "%s: no dtype promotion (torch-spyre limitation),"
            "expect numerical differences to upstream vLLM.",
            self.__class__.__name__,
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward implementation transparent to torch.compile.

        Args:
            x: Input tensor of any device/dtype

        Returns:
            Output tensor on original device with original dtype
        """

        # Bias is fused into F.linear only when not skipping bias add
        bias = self.bias.data if (self.bias is not None and not self.skip_bias_add) else None

        output = F.linear(x, self.weight.data, bias)
        
        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(SpyreLinearBase, MergedColumnParallelLinear):
    """Spyre MergedColumnParallelLinear (TP=1 only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_: torch.Tensor):
        """Forward pass for PluggableLayer.

        Args:
            input_: Input tensor
        Returns:
            Tuple of (output, bias) if return_bias=True, else just output
        """
        return self._forward_impl(input_)


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(SpyreLinearBase, QKVParallelLinear):
    """Spyre QKVParallelLinear (TP=1 only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_: torch.Tensor):
        """Forward pass for PluggableLayer.

        Args:
            input_: Input tensor

        Returns:
            Tuple of (output, bias) if return_bias=True, else just output.
        """
        output_output_bias = self._forward_impl(input_)
        
        # D2H the output before downstream .split() — Spyre can't handle strided views
        if self.return_bias:
            output = convert(output_output_bias[0], device="cpu")
            return output, output_output_bias[1]
        else:
            output = convert(output, device="cpu")
            return output


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(SpyreLinearBase, RowParallelLinear):
    """Spyre RowParallelLinear (TP=1 only)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input_: torch.Tensor):
        """Forward pass for PluggableLayer.

        Args:
            input_: Input tensor

        Returns:
            Tuple of (output, bias) if return_bias=True, else just output.
        """
        return self._forward_impl(convert(input_, device=self.weight.device))
