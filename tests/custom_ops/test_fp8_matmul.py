"""The FP8 attention matmuls' CPU emulation: shapes and the fp16 range argument."""

import pytest
import torch

from spyre_inference.custom_ops.fp8_matmul import fp8_pv, fp8_qk

D, N = 128, 64


@pytest.mark.parametrize("batch", [(8,), (2, 8)], ids=["3d", "4d"])
def test_fp8_qk_and_pv_keep_the_batch_dims(batch):
    q = torch.randn(*batch, 4, D, dtype=torch.float16)
    k = torch.randn(*batch, N, D, dtype=torch.float16)
    p = torch.softmax(torch.randn(*batch, 4, N), -1).to(torch.float16)

    assert fp8_qk(q, k).shape == (*batch, 4, N)
    assert fp8_pv(p, k).shape == (*batch, 4, D)


def test_fp8_qk_stays_finite_at_the_e4m3_limit():
    """K pages hold e4m3 values up to 448 (k_scale folded out); Q scaled to [-1, 1] keeps
    the product in fp16, and the softmax scale applies before the large Q scale does."""
    q = torch.full((2, 4, D), 30.0, dtype=torch.float16)
    k = torch.full((2, N, D), 448.0, dtype=torch.float16)
    scale = D**-0.5 * 0.01  # softmax scale with a small k_scale folded in

    scores = fp8_qk(q, k, scale)

    assert torch.isfinite(scores).all()
    want = (q.float() @ k.float().transpose(-2, -1)) * scale
    torch.testing.assert_close(scores.float(), want, rtol=1e-2, atol=0)


def test_fp8_qk_all_zero_row_is_zero_not_nan():
    q = torch.randn(2, 4, D, dtype=torch.float16)
    q[0, 1] = 0
    k = torch.randn(2, N, D, dtype=torch.float16)

    scores = fp8_qk(q, k)

    assert torch.isfinite(scores).all()
    assert (scores[0, 1] == 0).all()
