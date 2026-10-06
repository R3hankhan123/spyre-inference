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

"""Fused host sampling kernels ported from vLLM's CPU backend."""

import pytest
import torch

from spyre_inference.v1.sample import sampler, topk_topp_sampler
from spyre_inference.v1.sample.sampler import SpyreSampler
from spyre_inference.v1.sample.sampling_kernels import use_sampling_kernels
from spyre_inference.v1.sample.topk_topp_sampler import SpyreTopKTopPSampler


@pytest.fixture(autouse=True)
def _load_kernels() -> None:
    # Fail rather than silently test the fallback when the extension is not built.
    assert use_sampling_kernels(), "csrc/ sampling kernels are not built"


# Vocabs off a SIMD-width multiple exercise the kernels' remainder handling.
@pytest.mark.parametrize("rows", [1, 3, 16])
@pytest.mark.parametrize("vocab", [7, 32000, 128257])
def test_greedy_matches_argmax(rows: int, vocab: int) -> None:
    x = torch.randn(rows, vocab)
    top = x.max() + 1
    x[:, vocab // 2] = top
    x[:, vocab // 3] = top  # tie: argmax keeps the lowest index
    assert torch.equal(torch.ops._spyre_C.greedy_argmax(x), x.argmax(dim=-1))


def test_gumbel_matches_softmax_distribution() -> None:
    vocab, draws = 8, 200_000
    logp = torch.log_softmax(torch.randn(vocab) * 1.5, dim=-1)
    seeds = torch.randint(0, 2**31, (draws,), dtype=torch.long)
    out = torch.ops._spyre_C.fused_gumbel_argmax(logp.expand(draws, vocab), seeds)
    freq = torch.bincount(out, minlength=vocab).float() / draws
    assert (freq - logp.exp()).abs().max() < 0.01


def test_gumbel_reaches_large_vocab_tail() -> None:
    # A shared noise table limits each row to 2^20 noise windows, which makes
    # much of a large vocab unreachable and under-samples the tail.
    vocab, draws, chunk = 151_936, 8192, 256
    row = -1.1 * torch.log(torch.arange(1, vocab + 1, dtype=torch.float64))  # Zipf(1.1)
    p = torch.softmax(row, dim=-1)
    tail = p < 2.0**-20
    gen = torch.Generator().manual_seed(0)
    hits = 0
    for _ in range(draws // chunk):
        seeds = torch.randint(0, 2**62, (chunk,), dtype=torch.long, generator=gen)
        logits = row.float().expand(chunk, vocab).contiguous()
        hits += int(tail[torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds)].sum())
    expected = float(p[tail].sum())
    sigma = (expected * (1 - expected) / draws) ** 0.5
    assert abs(hits / draws - expected) < 4 * sigma


def test_gumbel_never_picks_masked_tokens() -> None:
    logits = torch.randn(64, 32000)
    logits[:, 100:] = float("-inf")
    seeds = torch.randint(0, 2**31, (64,), dtype=torch.long)
    assert (torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds) < 100).all()


def test_seeded_requests_are_reproducible() -> None:
    sampler = SpyreTopKTopPSampler("raw_logprobs", False)
    logits = torch.randn(4, 32000)

    def draw() -> torch.Tensor:
        gens = {i: torch.Generator().manual_seed(i) for i in (0, 2)}
        return sampler.forward_native(logits.clone(), gens, None, None)[0]

    a, b = draw(), draw()
    assert torch.equal(a[[0, 2]], b[[0, 2]])


def test_spyre_sampler_greedy_matches_stock() -> None:
    logits = torch.randn(8, 32000)
    assert torch.equal(SpyreSampler.greedy_sample(logits), logits.argmax(dim=-1))


def test_falls_back_to_torch_without_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sampler, "use_sampling_kernels", lambda: False)
    monkeypatch.setattr(topk_topp_sampler, "use_sampling_kernels", lambda: False)
    logits = torch.randn(4, 32000)
    assert torch.equal(SpyreSampler.greedy_sample(logits), logits.argmax(dim=-1))
    k = torch.full((4,), 5)
    topk = SpyreTopKTopPSampler("raw_logprobs", False)
    out = topk.forward_native(logits.clone(), {}, k, None)[0]
    assert (out.unsqueeze(1) == logits.topk(5).indices).any(dim=1).all()


def test_gumbel_strided_seeds_match_contiguous() -> None:
    logits = torch.randn(8, 32000)
    seeds = torch.randint(0, 2**31, (16,), dtype=torch.long)[::2]
    assert not seeds.is_contiguous()
    assert torch.equal(
        torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds),
        torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds.contiguous()),
    )
    with pytest.raises(RuntimeError, match="seeds must be int64"):
        torch.ops._spyre_C.fused_gumbel_argmax(logits, seeds.int())


def test_greedy_nan_matches_argmax() -> None:
    # torch.argmax treats NaN as the maximum and returns the first one.
    vocab = 32003  # off a SIMD-width multiple, so the last NaN lands in the scalar tail
    x = torch.randn(6, vocab)
    x[0, 100] = float("nan")  # before the real max
    x[0, 200] = 50.0
    x[1, 200] = 50.0
    x[1, 300] = float("nan")  # after the real max
    x[2, [7, 9000]] = float("nan")
    x[3, vocab - 1] = float("nan")
    x[4, 10] = float("inf")
    x[4, 20] = float("-inf")  # inf + -inf: NaN sum without a NaN logit
    assert torch.equal(torch.ops._spyre_C.greedy_argmax(x), x.argmax(dim=-1))
