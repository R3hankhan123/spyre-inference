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

"""Spyre OOT registration for the fused QKV projection.

The fused QKV weight stays intact on-device.  In eager mode the model runner
compiles each attention module that owns a ``SpyreQKVParallelLinear`` child,
so the ``F.linear`` **and** the downstream ``qkv.split()`` both live inside
one compiled graph where indirect access handles the strided views.
In fullgraph mode the attention modules are already inside the whole-model
``torch.compile``, so no extra compilation is needed.
"""

from vllm.model_executor.layers.linear import QKVParallelLinear


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(QKVParallelLinear):
    """Out-of-tree (OOT) QKVParallelLinear for IBM's Spyre device.

    Asserts ``gather_output=False`` (all_gather not yet supported on Spyre).
    The actual compilation is done at the attention-module level by the model
    runner's ``_compile_qkv_attention_modules``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.gather_output, (
            f"{self.__class__.__name__} requires gather_output=False; "
            "all_gather is not yet supported on Spyre"
        )
