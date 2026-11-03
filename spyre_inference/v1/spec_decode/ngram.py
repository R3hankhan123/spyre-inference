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

"""Model-free proposers: upstream's n-gram and suffix decoding run unchanged on the host."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from spyre_inference.v1.spec_decode.base import SpyreSpecMethod

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class NgramMethod(SpyreSpecMethod):
    name = "ngram"

    @classmethod
    def matches(cls, spec: SpeculativeConfig) -> bool:
        return spec.method == "ngram"

    def before_warmup(self, worker: Any) -> None:
        # ngram's batch_propose saves and restores numba's thread count, which shares
        # libgomp with torch, and Dynamo guards on that count: align them before warmup.
        from numba import set_num_threads  # ty: ignore[unresolved-import]

        set_num_threads(torch.get_num_threads())


class SuffixMethod(SpyreSpecMethod):
    name = "suffix"

    @classmethod
    def matches(cls, spec: SpeculativeConfig) -> bool:
        return spec.method == "suffix"
