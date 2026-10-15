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

"""Tests for the transposed-weight linear method (custom_ops/linear.py).

Most of these run on CPU (no Spyre device needed): `process_weights_after_loading`
is a pure host-side weight mutation. The generic `LinearBase` path (gate_up_proj,
down_proj, ...) is covered here; QKV and the LM head have their own tests
(test_mlp.py, test_parallel_lm_head.py).

A CPU-only test of `spyre_linear_t` proves nothing about how the card lowers its
matmul, and a 3-D @ 2-D matmul under torch.compile once came back wrong on Spyre
(torch-spyre#4155, since fixed). The batched case is checked in the on-card section
at the bottom.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available


def _make_mlp_module(hidden, inter, bias=False):
    """A minimal MLP: a fused gate_up_proj (MergedColumn) + a down_proj (Row)."""
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        RowParallelLinear,
    )

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = MergedColumnParallelLinear(
                input_size=hidden,
                output_sizes=[inter, inter],
                bias=bias,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="gate_up_proj",
            )
            self.down_proj = RowParallelLinear(
                input_size=inter,
                output_size=hidden,
                bias=bias,
                params_dtype=torch.float16,
                quant_config=None,
                disable_tp=True,
                prefix="down_proj",
            )

    return MLP()


def _bias(layer):
    return layer.bias.data if layer.bias is not None else None


def _forward(layer, x):
    """Run the transposed apply and drop the bias tuple."""
    return layer.quant_method.apply(layer, x, _bias(layer))


@pytest.mark.mlp
@pytest.mark.parametrize("num_tokens", [1, 7, 64])
@pytest.mark.parametrize("use_bias", [False, True])
def test_transposed_linear_matches_reference(tp_group, num_tokens, use_bias):
    """After processing weights, `x @ Wᵀ (+bias)` matches the upstream F.linear.

    The core correctness guard: a wrong transpose or axis swap would change the
    output here. Runs entirely on CPU.
    """
    from spyre_inference.custom_ops.linear import (
        SpyreMergedColumnParallelLinear,
        SpyreRowParallelLinear,
    )

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter, bias=use_bias)
    assert isinstance(mlp.gate_up_proj, SpyreMergedColumnParallelLinear)
    assert isinstance(mlp.down_proj, SpyreRowParallelLinear)
    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.weight.data.normal_(std=0.02)
        if layer.bias is not None:
            layer.bias.data.normal_(std=0.02)

    torch.manual_seed(1)
    x_gate = torch.randn(num_tokens, hidden, dtype=torch.float16)
    x_down = torch.randn(num_tokens, inter, dtype=torch.float16)

    # Capture the reference BEFORE process_weights_after_loading transposes.
    exp_gate = F.linear(x_gate, mlp.gate_up_proj.weight.data, _bias(mlp.gate_up_proj))
    exp_down = F.linear(x_down, mlp.down_proj.weight.data, _bias(mlp.down_proj))

    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.quant_method.process_weights_after_loading(layer)

    torch.testing.assert_close(
        _forward(mlp.gate_up_proj, x_gate).float(), exp_gate.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        _forward(mlp.down_proj, x_down).float(), exp_down.float(), atol=1e-2, rtol=1e-2
    )


@pytest.mark.mlp
def test_weight_stored_transposed(tp_group):
    """process_weights_after_loading replaces `weight` with `weightᵀ`, contiguous."""
    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)
    # torch.empty() leaves memory uninitialised (NaN in fp16 compares unequal to
    # itself); fill so the bit-for-bit check is meaningful.
    mlp.down_proj.weight.data.normal_(std=0.02)
    original = mlp.down_proj.weight.data.clone()
    orig_shape = original.shape  # [out, in] == [hidden, inter]

    mlp.down_proj.quant_method.process_weights_after_loading(mlp.down_proj)

    w = mlp.down_proj.weight
    assert w.shape == (orig_shape[1], orig_shape[0])  # [in, out]
    assert w.data.is_contiguous()
    torch.testing.assert_close(w.data, original.t(), atol=0.0, rtol=0.0)


@pytest.mark.mlp
def test_unquantized_layers_get_spyre_method(tp_group):
    """Unquantized linears get `SpyreUnquantizedLinearMethod` swapped in.

    The mixin only replaces the method when it is an `UnquantizedLinearMethod`
    (`quant_config=None` here), so a quantized layer would keep its own
    slow-but-correct F.linear method untouched.
    """
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from spyre_inference.custom_ops.linear import SpyreUnquantizedLinearMethod

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)

    assert isinstance(mlp.gate_up_proj.quant_method, SpyreUnquantizedLinearMethod)
    assert isinstance(mlp.down_proj.quant_method, SpyreUnquantizedLinearMethod)
    # SpyreUnquantizedLinearMethod is itself an UnquantizedLinearMethod subclass.
    assert isinstance(mlp.down_proj.quant_method, UnquantizedLinearMethod)


def _rows_reaching_gemm(gate_up, x, monkeypatch):
    """Rows that reach ``torch.matmul``.

    Short-row padding happens inside ``spyre_linear_t``, so ``apply`` still sees
    the real token count. On CPU ``out[:m]`` matches whether or not the pad fired.
    """
    from spyre_inference.custom_ops import linear as linear_mod

    seen = []
    real_matmul = linear_mod.torch.matmul

    def spy(activations, weight, *args, **kwargs):
        seen.append(activations.shape[0])
        return real_matmul(activations, weight, *args, **kwargs)

    monkeypatch.setattr(linear_mod.torch, "matmul", spy)
    out = _forward(gate_up, x)
    assert len(seen) == 1
    return seen[0], out


@pytest.mark.mlp
@pytest.mark.parametrize(
    ("num_tokens", "expected_rows"),
    [(1, 8), (4, 8), (7, 8), (8, 8), (9, 9), (64, 64)],
)
def test_short_rows_padded_on_gate_up(tp_group, monkeypatch, num_tokens, expected_rows):
    """A partial row block reaches the GEMM padded to `_PAD_ROWS`; a full one is untouched."""
    from spyre_inference.custom_ops.linear import (
        SpyrePaddedRowsLinearMethod,
        SpyreUnquantizedLinearMethod,
    )

    hidden, inter = 128, 256
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter)
    assert isinstance(mlp.gate_up_proj.quant_method, SpyrePaddedRowsLinearMethod)
    assert not isinstance(mlp.down_proj.quant_method, SpyrePaddedRowsLinearMethod)

    gate_up = mlp.gate_up_proj
    gate_up.weight.data.normal_(std=0.02)
    gate_up.quant_method.process_weights_after_loading(gate_up)

    torch.manual_seed(1)
    x = torch.randn(num_tokens, hidden, dtype=torch.float16)
    reference = SpyreUnquantizedLinearMethod().apply(gate_up, x, None)

    rows, out = _rows_reaching_gemm(gate_up, x, monkeypatch)
    assert rows == expected_rows
    assert out.shape == (num_tokens, 2 * inter)
    torch.testing.assert_close(out.float(), reference.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.mlp
def test_large_weights_not_padded(tp_group, monkeypatch):
    """The weight bound disables padding, since past it the pad rows cost more."""
    from spyre_inference.custom_ops import linear as linear_mod

    hidden, inter = 128, 256
    torch.manual_seed(0)
    gate_up = _make_mlp_module(hidden, inter).gate_up_proj
    gate_up.weight.data.normal_(std=0.02)
    gate_up.quant_method.process_weights_after_loading(gate_up)

    x = torch.randn(1, hidden, dtype=torch.float16)
    monkeypatch.setattr(linear_mod, "_MAX_PAD_WEIGHT", gate_up.weight.numel() - 1)
    rows, _ = _rows_reaching_gemm(gate_up, x, monkeypatch)
    assert rows == 1


# ---------------------------------------------------------------------------
# Numeric correctness on-card
# ---------------------------------------------------------------------------


@pytest.mark.mlp
@pytest.mark.parametrize(
    "batch,seq",
    [
        (1, 528),  # the production shape: one image, 24x22 patch grid
        (2, 64),  # batch > 1, so a fold that mishandles the leading dim is caught
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("mode", ["compile", "eager"])
def test_transposed_linear_3d_matches_reference_on_spyre(tp_group, batch, seq, use_bias, mode):
    """A 3-D `[B, T, hidden]` input through the layer on-card matches `F.linear`.

    Regression guard for torch-spyre#4155: a compiled 3-D @ 2-D matmul returned the
    right shape with uncorrelated values. It is fixed, so `spyre_linear_t` no longer
    folds to 2-D. The bug was compile-only, so the compile case is the one that
    guards it. Every other test here passes 2-D, as the decoder does; the Pixtral
    vision tower is the only 3-D caller.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    # hidden 1024 / 2x1536 reproduces the vision qkv_proj's 1024 -> 3072 GEMM.
    hidden, inter = 1024, 1536
    torch.manual_seed(0)
    mlp = _make_mlp_module(hidden, inter, bias=use_bias)
    layer = mlp.gate_up_proj
    layer.weight.data.normal_(std=0.02)
    if layer.bias is not None:
        layer.bias.data.normal_(std=0.02)

    torch.manual_seed(1)
    x = torch.randn(batch, seq, hidden, dtype=torch.float16)

    # Capture the reference BEFORE process_weights_after_loading transposes.
    # std-1 activations keep the output O(0.6), so atol=1e-2 is a real bound
    # rather than a tolerance wide enough to accept a wrong answer.
    expected = F.linear(x, layer.weight.data, _bias(layer))

    layer.quant_method.process_weights_after_loading(layer)
    layer = layer.to("spyre")

    def fn(x):
        return _forward(layer, x)

    if mode == "compile":
        torch._dynamo.reset()
        fn = torch.compile(fn, dynamic=False, backend="inductor")

    actual = fn(x.to("spyre"))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


def _row_parallel(monkeypatch, in_size, out_size, tp, bias=False):
    """A RowParallelLinear built as rank 0 of ``tp``, without a real TP group."""
    from vllm.model_executor.layers import linear as vllm_linear
    from vllm.model_executor.layers.linear import RowParallelLinear

    monkeypatch.setattr(vllm_linear, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(vllm_linear, "get_tensor_model_parallel_rank", lambda: 0)
    return RowParallelLinear(
        input_size=in_size,
        output_size=out_size,
        bias=bias,
        params_dtype=torch.float16,
        quant_config=None,
        prefix="proj",
    )


@pytest.mark.parametrize(
    ("in_size", "out_size", "tp", "gathers"),
    [
        (256, 256, 4, True),  # o_proj: gathering K moves half the all-reduce's bytes
        (1024, 256, 4, False),  # down_proj: K >= 2N, the all-reduce is cheaper
        (256, 256, 1, False),  # no TP, no collective to replace
    ],
)
def test_row_parallel_gathers_input_only_when_cheaper(
    tp_group, monkeypatch, in_size, out_size, tp, gathers
):
    layer = _row_parallel(monkeypatch, in_size, out_size, tp)
    assert layer.gathers_input is gathers


def test_long_row_parallel_input_is_gathered_not_reduced(tp_group, monkeypatch):
    """At and above the threshold the input is gathered and the full weight applied, with
    the whole bias on every rank; below it the layer all-reduces as upstream does."""
    from torch.nn.parameter import Parameter
    from vllm.model_executor.layers import linear as vllm_linear

    from spyre_inference.custom_ops import linear as linear_mod

    tp, k, n = 4, 256, 256
    layer = _row_parallel(monkeypatch, k, n, tp, bias=True)
    torch.manual_seed(0)
    layer.weight.data.normal_(std=0.02)
    layer.bias.data.normal_(std=0.02)
    layer.quant_method.process_weights_after_loading(layer)
    w_full = (torch.randn(k, n) * 0.02).to(torch.float16)
    layer.weight_full = Parameter(w_full, requires_grad=False)
    layer.weight_scale = Parameter(torch.ones(1, dtype=torch.float16), requires_grad=False)

    class _Kernel:
        def apply_fp8(self, x, weight, weight_scale, bias=None):
            out = (x.float() @ weight.float() * weight_scale.float()).to(x.dtype)
            return out if bias is None else out + bias

    monkeypatch.setattr(linear_mod, "_spyre_fp8_kernel", lambda module: _Kernel())
    reduced, gathered = [], []
    monkeypatch.setattr(
        vllm_linear, "tensor_model_parallel_all_reduce", lambda t: reduced.append(t) or t
    )

    rows = linear_mod._GATHER_INPUT_MIN_ROWS
    x_full = torch.randn(rows, k, dtype=torch.float16)

    def all_gather(t, dim=-1):
        gathered.append(t)
        return x_full  # rank 0's shard plus the other ranks', in rank order

    monkeypatch.setattr(linear_mod, "tensor_model_parallel_all_gather", all_gather)

    shard = x_full[:, : k // tp]
    out, _ = layer(shard)
    assert reduced == [] and gathered == [shard]
    expected = (x_full.float() @ w_full.float() + layer.bias.float()).to(torch.float16)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    gathered.clear()
    layer(shard[: rows - 1])
    assert gathered == [] and len(reduced) == 1
