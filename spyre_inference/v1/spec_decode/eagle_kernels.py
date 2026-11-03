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

"""Host replacements for the Triton kernels upstream's draft-model proposer launches.

vLLM's ``llm_base_proposer`` prepares each draft step's inputs with two Triton kernels,
and the build has neither Triton nor the CPU backend's ``torch.ops._C`` stand-ins
(``VLLM_TARGET_DEVICE=empty``). Each function keeps its kernel's in-place output
contract. Batches are at most ``max_num_seqs`` rows.
"""

from __future__ import annotations

import torch


class _HostDispatch:
    """Stands in for upstream's Triton kernel dispatcher; there is nothing to warm up."""

    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def register_warmup(self, *args, **kwargs) -> None:
        pass


def eagle_prepare_inputs_padded(
    cu_num_draft_tokens: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_indices_to_sample: torch.Tensor,
    num_rejected_tokens: torch.Tensor,
    num_reqs: int,
) -> None:
    cu = cu_num_draft_tokens[:num_reqs].long()
    num_draft = torch.diff(cu, prepend=cu.new_zeros(1))
    valid = valid_sampled_tokens_count[:num_reqs].long()
    rejected = torch.where(num_draft > 0, num_draft + 1 - valid, torch.zeros_like(num_draft))
    last_token = query_start_loc[1 : num_reqs + 1].long() - 1
    token_indices_to_sample[:num_reqs] = (last_token - rejected).to(token_indices_to_sample.dtype)
    num_rejected_tokens[:num_reqs] = rejected.to(num_rejected_tokens.dtype)


def eagle_prepare_next_token_padded(
    sampled_token_ids: torch.Tensor,
    discard_request_mask: torch.Tensor,
    backup_next_token_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    valid_sampled_tokens_count: torch.Tensor,
    vocab_size: int,
    num_sampled_tokens_per_req: int,
    num_reqs: int,
    *args,
    **kwargs,
) -> None:
    tokens = sampled_token_ids[:num_reqs, :num_sampled_tokens_per_req].long()
    valid = (tokens != -1) & (tokens < vocab_size)
    count = valid.sum(dim=1)
    cols = torch.arange(tokens.shape[1]).expand_as(tokens)
    last_index = torch.where(valid, cols, torch.full_like(cols, -1)).amax(dim=1)
    last_token = tokens.gather(1, last_index.clamp(min=0).unsqueeze(1)).squeeze(1)
    discard = discard_request_mask[:num_reqs].bool()
    backup = backup_next_token_ids[:num_reqs].long()
    next_token_ids[:num_reqs] = torch.where(discard | (count == 0), backup, last_token).to(
        next_token_ids.dtype
    )
    valid_sampled_tokens_count[:num_reqs] = torch.where(discard, torch.zeros_like(count), count).to(
        valid_sampled_tokens_count.dtype
    )


def install() -> None:
    """Point the base proposer's input-preparation dispatchers at the host versions.

    Upstream binds each dispatcher at import time, so the module attribute is what
    gets replaced.
    """
    import vllm.v1.spec_decode.llm_base_proposer as base_proposer

    base_proposer._eagle_prepare_inputs_padded = _HostDispatch(eagle_prepare_inputs_padded)  # ty: ignore[invalid-assignment]
    base_proposer._eagle_prepare_next_token_padded = _HostDispatch(  # ty: ignore[invalid-assignment]
        eagle_prepare_next_token_padded
    )
