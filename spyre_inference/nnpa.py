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

"""The IBM Z Integrated Accelerator for AI (NNPA), reached through zDNN (csrc/cpu/nnpa_kernels.cpp).

Small layers at the edge of a step move here to free the card. Only s390x builds the
kernels; x86_64 and ppc64le always report 0 and keep every layer on the card.
"""

import functools
import platform

import torch

from spyre_inference.v1.sample.sampling_kernels import has_sampling_kernels


@functools.cache
def nnpa_max_dim() -> int:
    """The largest tensor dimension NNPA accepts; 0 when this host cannot offload to it."""
    if platform.machine() != "s390x" or not has_sampling_kernels():
        return 0
    try:
        return int(torch.ops._spyre_C.nnpa_max_dim_idx_size())  # ty: ignore[unresolved-attribute]
    except (AttributeError, RuntimeError):  # built without zdnn.h
        return 0
