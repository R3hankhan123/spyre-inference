"""Check that the FP8 attention matmuls (spyre.scaled_bmm) compile and match on Spyre.

Run on a card:  python fp8_attn_repro.py
Each case compiles alone and prints OK with its max error against the CPU emulation in
custom_ops/fp8_matmul.py, or the first line of its error. The for_each_tile cases need
torch-spyre#5907 or #6013.
"""

import traceback

import torch
import torch_spyre  # noqa: F401
from torch_spyre._inductor.wsr import for_each_tile

from spyre_inference.custom_ops.fp8_matmul import fp8_pv, fp8_qk

# N is the KV block size: batchmatmulfp8 reduces P·V over it, so it must fill FP8 sticks
# (128 elements), as the default block size does.
E, KV, M, D, N, PAGES = 2, 8, 4, 128, 128, 2
DEV = "spyre"


def run(name, fn, ref_fn, *args):
    torch._dynamo.reset()
    try:
        out = torch.compile(fn, dynamic=False, fullgraph=True)(*(a.to(DEV) for a in args))
        out = out.cpu().float()
        ref = ref_fn(*args).float()
        err = (out - ref).abs().max().item()
        rel = err / ref.abs().max().clamp(min=1e-6).item()
        bad = ~torch.isfinite(out)
        finite = ""
        if bad.any():
            first = tuple(bad.nonzero()[0].tolist())
            finite = f"  NON-FINITE {int(bad.sum())}/{out.numel()}, first at {first}"
        print(f"OK    {name}: {tuple(out.shape)} max abs err {err:.4g} (rel {rel:.2e}){finite}")
    except Exception as e:  # noqa: BLE001
        msg = str(e).strip().splitlines()
        print(f"FAIL  {name}: {msg[0] if msg else type(e).__name__}")
        traceback.print_exc(limit=0)


def make(*shape, scale=1.0):
    return torch.randn(*shape, dtype=torch.float16) * scale


def probs(*shape):
    return torch.softmax(torch.randn(*shape), -1).to(torch.float16)


def e4m3(*shape):
    """KV page values as the cache holds them: exactly e4m3, up to 448."""
    return make(*shape, scale=100.0).clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float16)


def row_scale(x):
    return torch.ops.spyre.quantscalepertokenfp8(x, 1.0)


def row_scale_ref(x):
    return x.float().abs().amax(-1, keepdim=True)


# 0. The per-row scale both operands are quantized with, which must equal each row's amax.
run("row scale 3-D", row_scale, row_scale_ref, make(KV, M, D))
run("row scale 4-D", row_scale, row_scale_ref, make(E, KV, M, D))

# 1-2. Q·Kᵀ and P·V as standalone graphs, 3-D (decode/prefill: [KV, rows, ...]).
run("qk 3-D", fp8_qk, fp8_qk, make(KV, M, D), e4m3(KV, N, D))
run("pv 3-D", fp8_pv, fp8_pv, probs(KV, M, N), e4m3(KV, N, D))

# 3-4. 4-D, the batched decode shape [entries, KV, q_per_kv, ...].
run("qk 4-D", fp8_qk, fp8_qk, make(E, KV, M, D), e4m3(E, KV, N, D))
run("pv 4-D", fp8_pv, fp8_pv, probs(E, KV, M, N), e4m3(E, KV, N, D))


# 5-7. Inside torch-spyre's tiled loop, one page per trip, as the kernels run them.
def tiled_qk(q, k_pages):
    def body(acc, tiles):
        k_page, q = tiles
        return acc + fp8_qk(q, k_page.squeeze(0)), None

    out, _ = for_each_tile(
        body,
        (k_pages, q),
        dims=(0, None),
        tile_size=1,
        init=torch.zeros(*q.shape[:-1], k_pages.shape[-2], dtype=q.dtype, device=q.device),
    )
    return out


def tiled_qk_ref(q, k_pages):
    return sum(fp8_qk(q, k_page) for k_page in k_pages)


def tiled_pv(scores_pages, v_pages):
    """P computed in the body from per-page scores, as the online softmax does."""

    def body(acc, tiles):
        scores, v_page = tiles
        scores = scores.squeeze(0)
        p = torch.exp(scores - scores.amax(-1, keepdim=True))
        return acc + fp8_pv(p, v_page.squeeze(0)), None

    out, _ = for_each_tile(
        body,
        (scores_pages, v_pages),
        dims=(0, 0),
        tile_size=1,
        init=torch.zeros(
            *scores_pages.shape[1:-1],
            v_pages.shape[-1],
            dtype=scores_pages.dtype,
            device=scores_pages.device,
        ),
    )
    return out


def tiled_pv_ref(scores_pages, v_pages):
    return sum(
        fp8_pv(torch.exp(s - s.amax(-1, keepdim=True)), v) for s, v in zip(scores_pages, v_pages)
    )


run("qk 3-D in for_each_tile", tiled_qk, tiled_qk_ref, make(KV, M, D), e4m3(PAGES, KV, N, D))
run("pv 3-D in for_each_tile", tiled_pv, tiled_pv_ref, make(PAGES, KV, M, N), e4m3(PAGES, KV, N, D))
run(
    "qk 4-D in for_each_tile",
    tiled_qk,
    tiled_qk_ref,
    make(E, KV, M, D),
    e4m3(PAGES, E, KV, N, D),
)
run(
    "pv 4-D in for_each_tile",
    tiled_pv,
    tiled_pv_ref,
    make(PAGES, E, KV, M, N),
    e4m3(PAGES, E, KV, N, D),
)


# 9-10. P·V in the loop with P quantized without a per-row scale (P is already in [0, 1]):
# isolates whether quantscalepertokenfp8 on an in-body value is what the scheduler rejects.
def _pv_no_row_scale(p, v_page, quantize):
    v8 = torch.ops.spyre.qfp8wt(v_page.contiguous())
    return torch.ops.spyre.scaled_bmm(quantize(p), v8, out_dtype=p.dtype)


def tiled_pv_with(quantize):
    def fn(scores_pages, v_pages):
        def body(acc, tiles):
            scores, v_page = tiles
            scores = scores.squeeze(0)
            p = torch.exp(scores - scores.amax(-1, keepdim=True))
            return acc + _pv_no_row_scale(p, v_page.squeeze(0), quantize), None

        out, _ = for_each_tile(
            body,
            (scores_pages, v_pages),
            dims=(0, 0),
            tile_size=1,
            init=torch.zeros(
                *scores_pages.shape[1:-1],
                v_pages.shape[-1],
                dtype=scores_pages.dtype,
                device=scores_pages.device,
            ),
        )
        return out

    return fn


def _one(p):
    return torch.ones((), dtype=p.dtype, device=p.device)


for label, quantize in (
    ("qfp8ch", lambda p: torch.ops.spyre.qfp8ch(p)),
    ("quantize_fp8_with_scale(1)", lambda p: torch.ops.spyre.quantize_fp8_with_scale(p, _one(p))),
):
    run(
        f"pv 3-D in for_each_tile, P via {label}",
        tiled_pv_with(quantize),
        tiled_pv_ref,
        make(PAGES, KV, M, N),
        e4m3(PAGES, KV, N, D),
    )
    run(
        f"pv 4-D in for_each_tile, P via {label}",
        tiled_pv_with(quantize),
        tiled_pv_ref,
        make(PAGES, E, KV, M, N),
        e4m3(PAGES, E, KV, N, D),
    )


# 13-14. P·V in the loop with P passed in per page rather than computed in the body: if
# this compiles where 7-12 do not, the scheduler rejects an in-body value feeding the bmm.
def tiled_pv_given_p(p_pages, v_pages):
    def body(acc, tiles):
        p, v_page = tiles
        return acc + fp8_pv(p.squeeze(0), v_page.squeeze(0)), None

    out, _ = for_each_tile(
        body,
        (p_pages, v_pages),
        dims=(0, 0),
        tile_size=1,
        init=torch.zeros(
            *p_pages.shape[1:-1], v_pages.shape[-1], dtype=p_pages.dtype, device=p_pages.device
        ),
    )
    return out


def tiled_pv_given_p_ref(p_pages, v_pages):
    return sum(fp8_pv(p, v) for p, v in zip(p_pages, v_pages))


run(
    "pv 3-D in for_each_tile, P given",
    tiled_pv_given_p,
    tiled_pv_given_p_ref,
    probs(PAGES, KV, M, N),
    e4m3(PAGES, KV, N, D),
)
run(
    "pv 4-D in for_each_tile, P given",
    tiled_pv_given_p,
    tiled_pv_given_p_ref,
    probs(PAGES, E, KV, M, N),
    e4m3(PAGES, E, KV, N, D),
)
