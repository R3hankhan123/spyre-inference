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

"""Host stand-ins for the base proposer's input-preparation kernels, against their contracts."""

import torch

from spyre_inference.v1.spec_decode import eagle_kernels


def test_next_token_is_last_valid_sample_else_backup():
    sampled = torch.tensor([[5, -1, -1], [7, 8, 9], [-1, -1, -1], [4, 6, -1]])
    discard = torch.tensor([False, False, False, True])
    backup = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    next_ids = torch.empty(4, dtype=torch.int32)
    valid = torch.empty(4, dtype=torch.int32)

    eagle_kernels.eagle_prepare_next_token_padded(
        sampled, discard, backup, next_ids, valid, 100, 3, 4
    )

    assert next_ids.tolist() == [5, 9, 12, 13]
    assert valid.tolist() == [1, 3, 0, 0]


def test_sample_index_steps_back_over_rejected_drafts():
    cu_num_draft = torch.tensor([2, 4, 4])  # drafts per request: 2, 2, 0
    valid = torch.tensor([3, 1, 1], dtype=torch.int32)
    query_start_loc = torch.tensor([0, 3, 6, 7], dtype=torch.int32)
    indices = torch.empty(3, dtype=torch.int32)
    rejected = torch.empty(3, dtype=torch.int32)

    eagle_kernels.eagle_prepare_inputs_padded(
        cu_num_draft, valid, query_start_loc, indices, rejected, 3
    )

    assert rejected.tolist() == [0, 2, 0]
    assert indices.tolist() == [2, 3, 6]


def test_install_replaces_the_base_proposer_dispatchers():
    import vllm.v1.spec_decode.llm_base_proposer as base_proposer

    eagle_kernels.install()

    assert (
        base_proposer._eagle_prepare_inputs_padded.fn is eagle_kernels.eagle_prepare_inputs_padded
    )
    assert (
        base_proposer._eagle_prepare_next_token_padded.fn
        is eagle_kernels.eagle_prepare_next_token_padded
    )
    # Upstream registers warmups on the dispatchers when it builds the proposer.
    base_proposer._eagle_prepare_inputs_padded.register_warmup(max_batch_size=8)
