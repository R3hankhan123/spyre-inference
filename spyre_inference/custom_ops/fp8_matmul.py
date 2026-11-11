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

"""Attention matmuls on ``spyre.scaled_bmm`` (batchmatmulfp8), for a float8 KV cache.

The kernels quantize Q once with fp8_quantize_q, then run Q·Kᵀ (fp8_qk_page) and P·V
(fp8_pv) inside each page loop. fp8_qk does both steps, for standalone use.

Both operands keep the batch dims the kernel holds them in: scaled_bmm takes
``[*batch, M, K] @ [*batch, K, N]``, so nothing is flattened.

batchmatmulfp8 reads its second operand as QFP8WT, while a cache page holds QFP8CH
bytes, so a page is requantized from its model-dtype cast. That is lossless: with
k_scale folded out, a page's values are exactly e4m3.

Those values reach 448, and batchmatmulfp8 writes fp16, so the other operand stays within
[-1, 1]: Q and P are each scaled per row to their own amax. A dot product
then stays under 448 * 128 = 57344 at head size 128, inside fp16. Scaling Q to 448 as
the FP8 linears do overflows to inf. The per-row scales are applied to the fp16 output,
since scaled_bmm's own scales are per-tensor; Q's is multiplied into the caller's softmax
scale first so the de-scaled scores never pass through an unscaled, wider intermediate.

Inside a for_each_tile body the quantizations need torch-spyre#5907 or #6013 (#5423); before it
the restickify planner finds no layout batchmatmulfp8 can read.

Off Spyre the same rounding is emulated, so CPU tests check the kernels' wiring.
"""

import torch

_FP8 = torch.float8_e4m3fn
_FP8_MAX = float(torch.finfo(_FP8).max)
# quantscalepertokenfp8's floor, so an all-zero (fully masked) row gets a finite scale.
_Q_SCALE_MIN = 2.0**-23


def _to_fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-_FP8_MAX, _FP8_MAX).to(_FP8).to(x.dtype)


def _quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``x`` scaled per row into [-1, 1] and quantized as batchmatmulfp8's first operand,
    and the per-row scale that undoes it. Both operands' rows go through
    quantize_fp8_with_scale: a bare qfp8ch leaves batchmatmulfp8 no layout it can read."""
    if x.device.type != "spyre":
        x_scale = x.float().abs().amax(dim=-1, keepdim=True).clamp(min=_Q_SCALE_MIN).to(x.dtype)
        return _to_fp8(x / x_scale), x_scale
    # scale_ub=1 makes the scale the row's amax.
    x_scale = torch.ops.spyre.quantscalepertokenfp8(x, 1.0)  # ty: ignore[invalid-argument-type]
    return torch.ops.spyre.quantize_fp8_with_scale(x, x_scale), x_scale  # ty: ignore[invalid-argument-type]


def _bmm(x8: torch.Tensor, w: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """``x8 @ w`` with ``w`` quantized as batchmatmulfp8's second operand (QFP8WT)."""
    if x8.device.type != "spyre":
        return torch.matmul(x8, _to_fp8(w))
    w8 = torch.ops.spyre.qfp8wt(w.contiguous())  # ty: ignore[invalid-argument-type]
    return torch.ops.spyre.scaled_bmm(x8, w8, out_dtype=out_dtype)  # ty: ignore[invalid-argument-type]


def fp8_quantize_q(q: torch.Tensor, scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """``q`` [*batch, M, D] quantized for fp8_qk_page, and the [*batch, M, 1] multiplier
    that de-scales its scores: the per-row scale times ``scale``.

    The kernels call this once, before the page loop: Q is the same on every page, and
    its row reduction inside the loop, multiplied into scores the loop then reshapes,
    fails inductor's pointwise/reduction split.
    """
    q8, q_scale = _quantize_rows(q)
    return q8, q_scale * scale


def fp8_qk_page(
    q8: torch.Tensor, q_mult: torch.Tensor, k_page: torch.Tensor, out_dtype: torch.dtype
) -> torch.Tensor:
    """``q @ k_pageᵀ`` from fp8_quantize_q's output: [*batch, M, N]."""
    return _bmm(q8, k_page.transpose(-2, -1), out_dtype) * q_mult


def fp8_qk(q: torch.Tensor, k_page: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """``(q @ k_pageᵀ) * scale``: [*batch, M, D] x [*batch, N, D] -> [*batch, M, N]."""
    q8, q_mult = fp8_quantize_q(q, scale)
    return fp8_qk_page(q8, q_mult, k_page, q.dtype)


def fp8_pv(
    probs: torch.Tensor, v_page: torch.Tensor, rows: tuple[int, ...] | None = None
) -> torch.Tensor:
    """``probs @ v_page``: [*batch, M, N] x [*batch, N, D] -> [*batch, M, D].

    ``rows`` is the [*batch, M, N] shape the bmm reads ``probs`` as, when ``probs`` holds
    its rows split across axes; the result comes back in ``probs``' own shape. The row
    scales are taken before that merge: a row reduction over a merged view of P, which
    the page loop computes, fails inductor's pointwise/reduction split.
    """
    p8, p_scale = _quantize_rows(probs)
    if rows is None:
        return _bmm(p8, v_page, probs.dtype) * p_scale
    out = _bmm(p8.reshape(rows), v_page, probs.dtype)
    return out.reshape(*probs.shape[:-1], v_page.shape[-1]) * p_scale
