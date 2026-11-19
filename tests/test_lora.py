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

"""spyre_inference/lora.py on the host: the token mask and the packed masked LoRA."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.lora.layers import LoRAMapping, VocabParallelEmbeddingWithLoRA

from spyre_inference.lora import (
    SpyreLinearLoRAMixin,
    SpyrePunicaWrapper,
    _pack,
    prepare_lora_for_spyre,
)

SLOTS, RANK, IN = 3, 4, 16
OUTS = (8, 4, 4)  # a QKV-like layer: three output slices
IDS = [11, None, 13]  # slot -> adapter id; slot 1 empty


class _UpstreamLike(nn.Module):
    """What the mixin builds on in BaseLinearLayerWithLoRA: the host stacks, and a
    set_lora that resets the slot and copies each adapter into its leading rows."""

    def __init__(self) -> None:
        super().__init__()
        self.lora_a_stacked = tuple(torch.zeros(SLOTS, 1, RANK, IN) for _ in OUTS)
        self.lora_b_stacked = tuple(torch.zeros(SLOTS, 1, o, RANK) for o in OUTS)
        self.output_slices = OUTS

    def reset_lora(self, index: int) -> None:
        for a, b in zip(self.lora_a_stacked, self.lora_b_stacked, strict=True):
            a[index] = 0
            b[index] = 0

    def set_lora(self, index, lora_a, lora_b) -> None:
        self.reset_lora(index)
        for i, (a, b) in enumerate(zip(lora_a, lora_b, strict=True)):
            self.lora_a_stacked[i][index, 0, : a.shape[0], : a.shape[1]].copy_(a)
            self.lora_b_stacked[i][index, 0, : b.shape[0], : b.shape[1]].copy_(b)


class _Layer(SpyreLinearLoRAMixin, _UpstreamLike):
    def __init__(self, wrapper: SpyrePunicaWrapper) -> None:
        super().__init__()
        a_all, b_all = _pack(self)
        self.register_buffer("spyre_lora_a", a_all, persistent=False)
        self.register_buffer("spyre_lora_b", b_all, persistent=False)
        self.punica_wrapper = wrapper


def _wrapper(slice_counts=(3,)) -> SpyrePunicaWrapper:
    wrapper = SpyrePunicaWrapper(16, 4, "cpu")
    wrapper.allocate(16, SLOTS, RANK, set(slice_counts), torch.float32, torch.device("cpu"))
    return wrapper


def _activate(wrapper: SpyrePunicaWrapper, token_ids: tuple[int, ...]) -> None:
    mapping = LoRAMapping(token_ids, tuple(dict.fromkeys(token_ids)), is_prefill=True)
    wrapper.update_metadata(mapping, IDS, SLOTS + 1, 32)


def _adapter(rank: int, gen: torch.Generator):
    return (
        [torch.randn(rank, IN, generator=gen) for _ in OUTS],
        [torch.randn(o, rank, generator=gen) for o in OUTS],
    )


def test_mask_keeps_each_tokens_slot_and_clears_rows_a_shorter_batch_left() -> None:
    wrapper = _wrapper(slice_counts=(1, 3))
    _activate(wrapper, (11, 0, 13, 11))
    mask = wrapper.mask(1, 6)
    slot_cols = [slice(s * RANK, (s + 1) * RANK) for s in range(SLOTS)]
    for row, slot in enumerate([0, None, 2, 0, None, None]):
        expected = torch.zeros(SLOTS * RANK)
        if slot is not None:
            expected[slot_cols[slot]] = 1
        assert torch.equal(mask[row], expected), row
    # Three slices repeat the per-slot pattern once per slice.
    assert torch.equal(wrapper.mask(3, 4), mask[:4].repeat(1, 3))

    _activate(wrapper, (13,))
    assert wrapper.mask(1, 1)[0, slot_cols[2]].eq(1).all()
    assert not wrapper.mask(1, 16)[1:].any()


def test_masked_lora_matches_a_per_token_reference() -> None:
    gen = torch.Generator().manual_seed(0)
    wrapper = _wrapper()
    layer = _Layer(wrapper)
    adapters = {0: _adapter(RANK, gen), 2: _adapter(2, gen)}  # slot 2 below max rank
    for slot, (a, b) in adapters.items():
        layer.set_lora(slot, a, b)

    token_slots = [0, None, 2, 0, 2]
    _activate(wrapper, tuple(11 if s == 0 else 13 if s == 2 else 0 for s in token_slots))
    x = torch.randn(8, IN, generator=gen)  # 5 tokens padded to a bucket of 8
    base = torch.randn(8, sum(OUTS), generator=gen)

    expected = base.clone()
    for t, slot in enumerate(token_slots):
        if slot is not None:
            a, b = adapters[slot]
            expected[t] += torch.cat([x[t] @ ai.T @ bi.T for ai, bi in zip(a, b, strict=True)])
    torch.testing.assert_close(layer._apply_lora_to_output(x, base), expected)


def test_set_lora_uploads_once_and_reset_clears_the_slot(monkeypatch) -> None:
    gen = torch.Generator().manual_seed(1)
    layer = _Layer(_wrapper())
    uploads = []
    sync = SpyreLinearLoRAMixin._sync_spyre_lora
    monkeypatch.setattr(
        SpyreLinearLoRAMixin, "_sync_spyre_lora", lambda self: (uploads.append(1), sync(self))
    )
    layer.set_lora(0, *_adapter(RANK, gen))
    assert len(uploads) == 1
    assert layer.spyre_lora_a.any()

    layer.reset_lora(0)
    assert len(uploads) == 2
    assert not layer.spyre_lora_a.any() and not layer.spyre_lora_b.any()


@pytest.mark.parametrize("attr", ["_spyre_orig_head_dim", "_spyre_orig_intermediate_size"])
def test_prepare_refuses_a_padded_model(attr: str) -> None:
    hf_config = SimpleNamespace(**{attr: 100})
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf_config))
    with pytest.raises(NotImplementedError, match="padded"):
        prepare_lora_for_spyre(nn.Module(), config, torch.device("cpu"))  # ty: ignore[invalid-argument-type]


def test_prepare_runs_the_plain_embedding_and_refuses_an_adapter_that_trains_it() -> None:
    wrapper = VocabParallelEmbeddingWithLoRA.__new__(VocabParallelEmbeddingWithLoRA)
    nn.Module.__init__(wrapper)
    wrapper.base_layer = nn.Embedding(8, 4)
    model = nn.Module()
    model.embed_tokens = wrapper
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace()))
    prepare_lora_for_spyre(model, config, torch.device("cpu"))  # ty: ignore[invalid-argument-type]

    assert model.embed_tokens is wrapper.base_layer
    with pytest.raises(NotImplementedError, match="VocabParallelEmbeddingWithLoRA"):
        wrapper.set_lora(0, torch.zeros(4, 8), torch.zeros(4, 4))
