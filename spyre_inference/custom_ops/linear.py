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

"""Spyre-specific linear layer implementations using out-of-tree (OOT) registration.

This module provides Spyre-device-specific replacements for the parallel linear
layer classes used inside MLP blocks:

    - SpyreMergedColumnParallelLinear  — replaces MergedColumnParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreQKVParallelLinear          — replaces QKVParallelLinear
      (vllm/model_executor/layers/linear.py)
    - SpyreRowParallelLinear          — replaces RowParallelLinear
      (vllm/model_executor/layers/linear.py)

At TP=1, the upstream forward() methods reduce to quant_method.apply() + bias
handling.  We inject a custom quant_method (SpyreUnquantizedLinearMethod) that
performs F.linear directly, QKV and RowParallel still override forward()
for device placement (D2H after GEMM, H2D before GEMM).

Spyre Device Constraints:
    - Computations performed in torch.float16:
      Input (dtype defined by model / user) converted to torch.float16 for
      operations on spyre and then converted back to original dtype for cpu.
    - Tensor parallelism: TP>=1 supported with all_reduce collectives

References:
    - Upstream linear layers:   vllm/model_executor/layers/linear.py
"""

import torch.nn.functional as F


from vllm.distributed import (
    split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger

from spyre_inference.distributed.spyre_communicator import (
    _spyre_collective_unsupported_message,
)
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)

from .utils import convert

logger = init_logger(__name__)


class SpyreUnquantizedLinearMethod(UnquantizedLinearMethod):
    """Spyre-specific linear method: F.linear without platform GEMM dispatch.

    Replaces the default UnquantizedLinearMethod so that upstream forward()
    methods work unchanged on Spyre at any TP size.

    - create_weights() is inherited — standard ModelWeightParameter works.
    - apply() does F.linear directly (no platform-specific GEMM dispatch).
    - process_weights_after_loading() is a no-op (skips CPU GEMM dispatch).
    """

    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight.data, bias)

    def process_weights_after_loading(self, layer):
        pass


class SpyreLinearBase:
    """Shared initialization for Spyre linear layers supporting TP>=1."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = SpyreUnquantizedLinearMethod()

        logger.debug(
            "Initialized %s with TP=%d, rank=%d",
            self.__class__.__name__,
            self.tp_size,
            self.tp_rank,
        )


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(SpyreLinearBase, MergedColumnParallelLinear):
    """Spyre MergedColumnParallelLinear with TP support.

    Supports TP>=1 with weight sharding along output dimension.
    Note: gather_output is not supported (all_gather not yet implemented).
    Outputs remain sharded across ranks.
    """

    def forward(self, input_):
        """Forward pass with TP support.

        At TP=1: Standard F.linear + bias handling
        At TP>1: Sharded F.linear per rank
        """
        bias = self.bias if not self.skip_bias_add else None

        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output and self.tp_size > 1:
            raise NotImplementedError(
                _spyre_collective_unsupported_message(
                    "allgather",
                    self.world_size,
                    blocker="libspyre_comms list-form allgather + torch-spyre",
                )
            )

        if not self.return_bias:
            return output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output_parallel, output_bias


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(SpyreLinearBase, QKVParallelLinear):
    """Spyre QKVParallelLinear with TP support.

    Supports TP>=1 with weight sharding for Q, K, V projections.
    Performs device transfers (D2H) after F.linear since downstream .split()
    cannot handle strided views on Spyre.
    """

    def forward(self, input_):
        """Forward pass with TP support and device handling.

        At TP=1: Standard F.linear + D2H transfer for CPU .split() compatibility
        At TP>1: Sharded F.linear per rank (outputs remain sharded) + D2H transfer
        """
        bias = self.bias if not self.skip_bias_add else None

        output_parallel = self.quant_method.apply(self, input_, bias)

        if self.gather_output and self.tp_size > 1:
            raise NotImplementedError(
                _spyre_collective_unsupported_message(
                    "allgather",
                    self.world_size,
                    blocker="libspyre_comms list-form allgather + torch-spyre",
                )
            )

        # D2H before downstream .split() — Spyre can't handle strided views
        output = convert(output_parallel, device="cpu")

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(SpyreLinearBase, RowParallelLinear):
    """Spyre RowParallelLinear with TP support.

    Supports TP>=1 with weight sharding along input dimension and
    all_reduce for aggregating results across ranks when reduce_results=True.

    RowParallelLinear is invoked from both attention and MLP layers
    """

    def forward(self, input_):
        """Forward pass with TP support and device handling.

        At TP=1: H2D transfer + F.linear + bias handling
        At TP>1: Input split (if needed) + H2D transfer + sharded F.linear + all_reduce
        """
        # H2D before matmul to ensure input is on Spyre device
        input_ = convert(input_, device=self.weight.device)

        if self.input_is_parallel:
            input_parallel = input_
        else:
            if self.tp_size > 1:
                split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
                input_parallel = split_input[self.tp_rank].contiguous()
            else:
                input_parallel = input_

        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output_parallel = self.quant_method.apply(self, input_parallel, bias_)

        if self.reduce_results and self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output = output_parallel

        if not self.return_bias:
            return output
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
