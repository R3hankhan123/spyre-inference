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

"""Bucketer for the attention kernel's compiled variants.

Dynamo specializes the per-sequence attention kernel on its padded query length,
compiling on first use, which would put a full Inductor compile in the serving
path. This module enumerates the query buckets a run can reach so warmup can record
them all up front. The page count is not bucketed: each sequence walks its own pages,
copied to the device with a declared page range, so one kernel per query bucket runs
every page count up to the context.

Separate from ``SpyreShapeBucketer``, which dispatches a single ``num_tokens``
int for the model graph.

The batched decode kernel runs all decode sequences of a step in one call and
specializes on its own key, ``(num_seqs, blocks_per_chunk, num_chunks)``, hence a
second bucket type and enumerator. Up to ``shared_slot_seqs`` sequences run on the
shared-slot kernel instead, whose key leaves ``num_seqs`` out.

Vocabulary:

- A *bucket* is one padded size a runtime length rounds up onto; the sorted list of
  them for one axis is that axis's *buckets*; the spacing between consecutive buckets
  is the *bucket step*.
- A *page* (also *block*) is one KV cache block: ``block_size`` tokens of one sequence.
- A *chunk* is what the batched decode kernel reads in one loop step: the next
  ``blocks_per_chunk`` pages of every sequence in the batch. ``_chunk_width`` picks
  ``blocks_per_chunk`` so that a chunk holds about 32 pages, one per core, and at least
  one page of each sequence. A batch runs as many chunks as its longest sequence needs;
  shorter sequences are masked past their end.
- A *block slot* is one of a chunk's pages. Each block slot keeps its own running
  softmax across chunks, and a sequence's block slots merge after the last one. The
  per-bucket kernels give every sequence ``blocks_per_chunk`` of them. The *shared-slot*
  kernel instead shares ``_SHARED_SLOTS`` of them among up to ``shared_slot_seqs``
  sequences, each step and by length, so it does not depend on the batch size. The
  per-sequence prefill kernel's counterpart is its *page group*, ``page_group`` pages of
  one sequence per loop step.
  Neither is related to vLLM's chunked prefill, which splits a prompt's tokens across
  scheduler steps.
- A *dynamic* kernel is compiled for a range of sizes (a ``DynamicRange``) instead of
  one, so one compiled kernel runs every page count or chunk count in that range.
"""

from __future__ import annotations

import bisect
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple, TypedDict

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv

from spyre_inference import envs
from spyre_inference.v1.worker.spyre_shape_bucketer import _align_up_pow2

logger = init_logger(__name__)

# Spacing of the default query buckets above the decode bucket, capped against
# max_num_batched_tokens. Every non-decode batch pads its query length up to a
# multiple of this.
_DEFAULT_QUERY_BUCKET_STEP = 512

# Cores available to split a gather's entry axis across.
_SPYRE_CORE_COUNT = 32

# Block slots per shared-slot chunk, two per core. Each chunk has a fixed cost, so a long
# context pays for its chunk count; 64 pages per chunk take half the chunks of 32.
_SHARED_SLOTS = 64

# The most chunks one dynamic shared-slot call runs, unless one max_model_len sequence needs
# more. Every launch fills one program correction per chunk the kernel could run, so a
# larger range costs every call; a batch past it splits into grouped calls. 16 chunks of 64
# slots cover the pages 32 chunks of 32 did.
_SHARED_SLOT_CALL_CHUNKS = 16


class DynamicRange(TypedDict):
    """The sizes one dynamic kernel is compiled for: ``min`` to ``max`` in steps of
    ``granularity``. A dict, since it is passed as is to torch-spyre's
    ``tensor.to("spyre", dynamic={dim: ...})``."""

    min: int
    max: int
    granularity: int


def _chunk_width(b_seqs: int) -> int:
    """The pages per sequence in one chunk: ``32 // b_seqs``, so that a chunk gives each
    of the 32 cores one page, but at least 1."""
    return max(1, _SPYRE_CORE_COUNT // b_seqs)


def share_block_slots(blocks: list[int], slots: int = _SHARED_SLOTS) -> tuple[int, list[int]]:
    """The fewest chunks C with ``sum(ceil(b / C)) <= slots``, and each sequence's block slots.

    Sequence i takes ``ceil(blocks[i] / C)`` consecutive block slots, so each of them reads at
    least one real block. Every sequence needs at least one block and one block slot.
    """
    assert blocks and min(blocks) >= 1 and len(blocks) <= slots, (blocks, slots)
    lo, hi = -(-sum(blocks) // slots), max(blocks)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(-(-b // mid) for b in blocks) <= slots:
            hi = mid
        else:
            lo = mid + 1
    return lo, [-(-b // lo) for b in blocks]


@dataclass(frozen=True)
class SpyreAttnBucket:
    """One recordable per-sequence attention kernel variant.

    Fields are the values the kernel specializes on, so a recorded bucket and a
    runtime dispatch reach the same Dynamo entry.
    """

    num_blocks: int
    padded_query_len: int


@dataclass(frozen=True)
class SpyreAttnBatchedDecodeBucket:
    """One recordable batched decode kernel variant.

    The kernel specializes on ``num_seqs``, ``blocks_per_chunk`` and
    ``num_chunks`` (derived from the logical block-axis tensor extent). A dynamic kernel
    (``dynamic_chunks``) runs every chunk count in its range; ``num_chunks`` is then the
    count warmup records it at. ``num_blocks`` is the page count the variant is recorded
    at.
    """

    num_seqs: int
    num_blocks: int
    blocks_per_chunk: int
    num_chunks: int
    dynamic_chunks: bool = False
    # A shared-slot kernel: num_seqs is its output row count, blocks_per_chunk its block slots,
    # and it serves every batch of up to num_seqs sequences.
    shared_slots: bool = False


class BatchedDecodePlan(NamedTuple):
    """How ``build()`` splits one batched decode call into chunks.

    ``num_blocks`` is the page count the call covers: the longest sequence's own page
    count for a dynamic kernel, otherwise ``num_chunks`` whole chunks. The call runs
    ``num_chunks`` chunks of ``blocks_per_chunk`` pages. ``chunk_range`` is the range of
    chunk counts a dynamic kernel runs, None for a fixed one.
    """

    num_blocks: int
    blocks_per_chunk: int
    num_chunks: int
    chunk_range: DynamicRange | None


class SharedSlotPlan(NamedTuple):
    """How ``build()`` splits one shared-slot call into chunks: its chunk count, the chunk
    range of the kernel that runs it, and each sequence's block slots."""

    num_chunks: int
    chunk_range: DynamicRange
    slots: list[int]


def _parse_buckets(raw: str | None) -> list[int] | None:
    """Parse a comma-separated env-var bucket list, or None when unset/empty."""
    if not raw:
        return None
    values = sorted({int(part) for part in raw.split(",") if part.strip()})
    if not values or values[0] < 1:
        raise ValueError(f"bucket entries must be >= 1, got {raw!r}")
    return values


def _powers_of_two_up_to(n: int, start: int = 1) -> tuple[int, ...]:
    """Powers of 2 in [start, n] (start rounded up to a power of 2), plus n itself."""
    if n < 1:
        return ()
    v = 1
    while v < start:
        v *= 2
    result = []
    while v < n:
        result.append(v)
        v *= 2
    result.append(n)
    return tuple(result)


def _min_num_kv_heads(vllm_config: VllmConfig) -> int:
    """The smallest KV head count any layer carries, after the TP split.

    ``get_num_kv_heads`` reports one model-wide number that a heterogeneous config can
    hide a smaller layer behind: gemma-4 reports 4 at TP2, its full-attention layers 1.
    """
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    arch = model_config.model_arch_config
    model_wide = model_config.get_num_kv_heads(parallel_config)
    overrides = arch.per_layer_overrides or ()
    return min(
        (model_config.get_num_kv_heads(parallel_config, arch[i]) for i in range(len(overrides))),
        default=model_wide,
    )


def _declared_page_range(max_pages: int, min_pages: int) -> DynamicRange | None:
    """Return the range of page counts one per-sequence kernel is compiled for.

    ``min_pages`` is the fewest pages a kernel can run. Returns None when ``max_pages`` is
    not above that minimum: the range would
    hold a single count, and every sequence then runs at ``max_pages``.

    Raises:
        ValueError: ``MAX_BUCKETS`` was exported below ``max_pages``.
    """
    from torch_spyre._inductor import config as ts_config
    from torch_spyre._inductor import pass_utils

    if not hasattr(pass_utils, "declared_dims"):
        raise RuntimeError(
            "This torch-spyre cannot run for_each_tile over a dynamic number of tiles "
            "(torch-spyre#6066), which the per-sequence attention kernels need. "
            "Upgrade torch-spyre."
        )
    # The page count is the only dynamic size, so it sets these torch-spyre limits unless
    # the user exported them.
    if "MIN_RESERVED_DIM_SIZE" not in os.environ:
        ts_config.min_reserved_dim_size = 1
    if "MAX_BUCKETS" not in os.environ:
        ts_config.max_buckets = max(ts_config.max_buckets, max_pages)
    min_pages = max(min_pages, ts_config.min_reserved_dim_size)
    # With min == max the size is a constant, which fails to trace.
    if max_pages <= min_pages:
        logger.info_once(
            "%d pages per sequence is not above the minimum of %d, so every sequence "
            "pads to %d pages and the per-sequence attention kernels use a static page count.",
            max_pages,
            min_pages,
            max_pages,
        )
        return None
    if max_pages > ts_config.max_buckets:
        raise ValueError(
            f"MAX_BUCKETS={ts_config.max_buckets} is below the {max_pages} pages one sequence "
            "can span, which one per-sequence attention kernel walks as many loop steps. "
            f"Unset MAX_BUCKETS or raise it to at least {max_pages}."
        )
    return {"min": min_pages, "max": max_pages, "granularity": 1}


# Prefill reads this many pages per loop step, so its page counts are multiples of it.
PREFILL_PAGE_GRANULARITY = 8


def _prefill_page_range(page_range: DynamicRange) -> DynamicRange:
    """Return ``page_range`` with its bounds rounded up to multiples of
    ``PREFILL_PAGE_GRANULARITY``.

    Returns ``page_range`` unchanged when the rounded range holds a single page count or is
    above an exported ``MAX_BUCKETS``.
    """
    from torch_spyre._inductor import config as ts_config

    g = PREFILL_PAGE_GRANULARITY
    min_pages = -(-page_range["min"] // g) * g
    max_pages = -(-page_range["max"] // g) * g
    if "MAX_BUCKETS" not in os.environ:
        ts_config.max_buckets = max(ts_config.max_buckets, max_pages)
    if not min_pages < max_pages <= ts_config.max_buckets:
        return page_range
    return {"min": min_pages, "max": max_pages, "granularity": g}


def _chunk_plans(
    chunks_by_seqs: dict[int, tuple[int, int]], page_range: DynamicRange | None
) -> tuple[dict[int, DynamicRange], dict[int, int]]:
    """How each num_seqs bucket runs a batched decode call.

    ``chunks_by_seqs`` maps each bucket to ``(min_chunks, max_chunks)``: the chunks its
    fewest and most pages need. Returns ``(chunk_ranges, static_chunks)``: a bucket in
    ``chunk_ranges`` takes every chunk count at run time, one included; one in
    ``static_chunks`` pads every call to that count, because a range with min == max does
    not compile (or, without a ``page_range``, nothing is dynamic).
    """
    from torch_spyre._inductor import config as ts_config

    ranges: dict[int, DynamicRange] = {}
    static: dict[int, int] = {}
    for num_seqs, (min_chunks, max_chunks) in chunks_by_seqs.items():
        min_chunks = max(min_chunks, ts_config.min_reserved_dim_size)
        if page_range is None or max_chunks <= min_chunks:
            static[num_seqs] = max_chunks
        else:
            ranges[num_seqs] = {"min": min_chunks, "max": max_chunks, "granularity": 1}
    return ranges, static


def _shared_slot_chunk_range(
    page_range: DynamicRange, num_seqs: int, min_chunks: int
) -> DynamicRange | None:
    """The chunk counts the dynamic shared-slot kernel runs: from ``min_chunks`` up to what
    ``num_seqs`` sequences of ``page_range["max"]`` pages need, capped at
    ``_SHARED_SLOT_CALL_CHUNKS`` or the count one such sequence needs.

    None, and logged, when the range would hold one count; batched decode then keeps one
    kernel per num_seqs bucket.
    """
    from torch_spyre._inductor import config as ts_config

    min_chunks = max(min_chunks, ts_config.min_reserved_dim_size)
    one_seq = -(-page_range["max"] // _SHARED_SLOTS)
    max_chunks = min(
        -(-page_range["max"] // (_SHARED_SLOTS // num_seqs)),
        max(_SHARED_SLOT_CALL_CHUNKS, one_seq),
    )
    if not min_chunks < max_chunks:
        logger.info_once(
            "No shared-slot batched decode: %d sequences of %d pages fit in %d chunks, not "
            "above the minimum of %d.",
            num_seqs,
            page_range["max"],
            max_chunks,
            min_chunks,
        )
        return None
    return {"min": min_chunks, "max": max_chunks, "granularity": 1}


def _resolve_buckets(
    raw: str | None, limit: int, name: str, default: Callable[[], list[int]]
) -> list[int]:
    """One axis's buckets: the env override clamped to ``limit``, else ``default()``.

    ``limit`` bounds the lengths the engine can schedule (max_model_len for kv,
    max_num_batched_tokens for query, max_num_seqs for the batch axis). Entries
    above it are unreachable, so they are dropped, and ``limit`` itself is added
    when missing.
    """
    buckets = _parse_buckets(raw)
    if buckets is None:
        return default()
    kept = [b for b in buckets if b <= limit]
    if len(kept) != len(buckets):
        logger.warning(
            "%s lists %s above the %d it must cover; dropping them as unreachable.",
            name,
            [b for b in buckets if b > limit],
            limit,
        )
    if not kept or kept[-1] < limit:
        logger.warning(
            "%s does not cover %d; appending it. Lengths in (%d, %d] would otherwise "
            "have no recorded bucket and would compile an attention kernel in the "
            "serving path.",
            name,
            limit,
            kept[-1] if kept else 0,
            limit,
        )
        kept.append(limit)
    return kept


class SpyreAttnBucketer:
    """Enumerates the attention variants to record, and rounds lengths onto them.

    Query lengths and batch sizes round *up* onto their buckets, matching
    ``SpyreShapeBucketer.find_bucket``; over-max returns None, and the caller falls back to
    compiling on demand. Page counts are not bucketed: a sequence runs at its own count,
    within the declared ``page_range``.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        declare_pages: bool = False,
        sliding_window: int | None = None,
    ) -> None:
        block_size = vllm_config.cache_config.block_size
        self.block_size = block_size
        max_model_len = vllm_config.model_config.max_model_len
        max_batched = vllm_config.scheduler_config.max_num_batched_tokens

        # A pooling request's query_len is its own context_len, so it can't
        # exceed max_model_len even when max_num_batched_tokens is larger (unlike
        # a decoder's chunked-prefill step). Without this cap, warmup could record
        # a query bucket no sequence can reach. Stick-aligned via _align_up_pow2 to match
        # the encoder shape ladder's padding (CLIP: 77 -> 128).
        if vllm_config.model_config.runner_type == "pooling":
            max_batched = min(max_batched, _align_up_pow2(max_model_len))

        # Default: powers of two up to max_num_seqs, the batch sizes the batched
        # decode kernel can be asked for.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._num_seqs_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_NUM_SEQS_BUCKETS,
            max_num_seqs,
            "SPYRE_ATTN_NUM_SEQS_BUCKETS",
            lambda: list(_powers_of_two_up_to(max_num_seqs)),
        )

        # Default: [1] (the decode-only batch, exempt from query padding by
        # build()) then multiples of a step up to max_num_batched_tokens, the
        # query lengths a prefill pads up to.
        step = min(_DEFAULT_QUERY_BUCKET_STEP, max_batched)
        self._query_buckets: list[int] = _resolve_buckets(
            envs.SPYRE_ATTN_QUERY_BUCKETS,
            max_batched,
            "SPYRE_ATTN_QUERY_BUCKETS",
            lambda: sorted({1, *range(step, max_batched + 1, step), max_batched}),
        )

        # The block table holds every page of a max_model_len sequence. A sliding window
        # walks only the pages its widest query's windows touch: ceil(span / B) + 1 for a
        # span of W + query_len - 1 positions that need not start on a page boundary.
        self._context_pages = cdiv(max_model_len, block_size)
        max_pages = self._context_pages
        if sliding_window is not None:
            span = sliding_window + self._query_buckets[-1] - 1
            max_pages = min(max_pages, cdiv(span, block_size) + 1)

        # One KV head on one page leaves torch-spyre's codegen no spare dim for the stick
        # dim, so the decode kernel will not compile there. A second, fully masked page
        # does.
        self._min_blocks = 2 if _min_num_kv_heads(vllm_config) == 1 else 1
        self._max_pages = max(max_pages, self._min_blocks)

        # The per-sequence kernels take the page count at run time, so they compile once
        # per query bucket. Without a range every sequence pads to max_pages.
        self._page_range: DynamicRange | None = None
        self._prefill_page_range: DynamicRange | None = None
        if declare_pages:
            self._page_range = _declared_page_range(self._max_pages, self._min_blocks)
        if self._page_range is not None:
            self._prefill_page_range = _prefill_page_range(self._page_range)

        # Batched decode gets one dynamic kernel per num_seqs bucket, for every chunk count.
        min_pages = self._page_range["min"] if self._page_range else self._max_pages
        chunks_by_seqs = {
            num_seqs: (
                self._chunks_needed(num_seqs, min_pages),
                self._chunks_needed(num_seqs, self._max_pages),
            )
            for num_seqs in self._num_seqs_buckets
        }
        self._chunk_ranges, self._static_chunks = _chunk_plans(chunks_by_seqs, self._page_range)

        # Up to this many decode sequences share a chunk's block slots by length, so none
        # of their batched kernels depends on the batch size. Each block slot reads one page
        # per chunk, so one KV head needs a second, masked chunk, as _chunks_needed does.
        self._shared_slot_seqs = min(_SHARED_SLOTS, self._num_seqs_buckets[-1])
        self._shared_slot_chunk_range: DynamicRange | None = None
        if self._page_range is not None:
            self._shared_slot_chunk_range = _shared_slot_chunk_range(
                self._page_range, self._shared_slot_seqs, self._min_blocks
            )
        if self._shared_slot_chunk_range is not None:
            logger.info(
                "SpyreAttnBucketer: batched decode of up to %d sequences shares the block "
                "slots of %d to %d chunks per call.",
                self._shared_slot_seqs,
                self._shared_slot_chunk_range["min"],
                self._shared_slot_chunk_range["max"],
            )

        logger.info(
            "SpyreAttnBucketer: %d query buckets [%d..%d], %d num_seqs buckets, pages %s",
            len(self._query_buckets),
            self._query_buckets[0],
            self._query_buckets[-1],
            len(self._num_seqs_buckets),
            self._page_range or f"static at {self._max_pages}",
        )

    @property
    def query_buckets(self) -> list[int]:
        return self._query_buckets

    @property
    def num_seqs_buckets(self) -> list[int]:
        return self._num_seqs_buckets

    @property
    def max_pages(self) -> int:
        """The most pages one sequence's kernel walks."""
        return self._max_pages

    def find_query_bucket(self, query_len: int) -> int | None:
        return self._round_up(query_len, self._query_buckets)

    def find_sequence_bucket(self, num_seqs: int) -> int | None:
        return self._round_up(num_seqs, self._num_seqs_buckets)

    @property
    def page_range(self) -> DynamicRange | None:
        """The ``min``, ``max`` and ``granularity`` of the page count one per-sequence
        kernel runs. None when every sequence runs at ``max_pages``."""
        return self._page_range

    @property
    def prefill_page_range(self) -> DynamicRange | None:
        """``page_range`` for a prefill sequence (query width > 1), whose page counts are
        multiples of ``PREFILL_PAGE_GRANULARITY``."""
        return self._prefill_page_range

    def find_pages(self, num_blocks: int, prefill: bool = False) -> int:
        """The page count a sequence of ``num_blocks`` pages runs at: its own, rounded up
        to the range granularity, or the range minimum if that is larger; ``max_pages``
        without a range."""
        assert num_blocks <= self._max_pages, (
            f"num_blocks={num_blocks} exceeds the {self._max_pages} pages a sequence can span"
        )
        page_range = self._prefill_page_range if prefill else self._page_range
        if page_range is None:
            return self._max_pages
        g = page_range["granularity"]
        return max(-(-num_blocks // g) * g, page_range["min"])

    def chunk_width(self, num_seqs_bucket: int) -> int:
        """Pages per sequence in one batched decode chunk: ``_chunk_width``, even past the
        longest context, since a narrower chunk is a gather torch-spyre may not lay out."""
        return _chunk_width(num_seqs_bucket)

    def _chunks_needed(self, num_seqs_bucket: int, num_blocks: int) -> int:
        """Chunks a call of ``num_blocks`` blocks needs. A one-block chunk takes a second,
        masked one where one KV head needs two pages."""
        width = self.chunk_width(num_seqs_bucket)
        return max(cdiv(num_blocks, width), self._min_blocks if width < self._min_blocks else 1)

    def batched_decode_plan(
        self, num_seqs_bucket: int, num_blocks: int
    ) -> BatchedDecodePlan | None:
        """How to run a batched decode call whose longest sequence has ``num_blocks`` pages.

        - If this bucket has a chunk range, it runs the bucket's dynamic kernel with just
          enough chunks for ``num_blocks``, and at least ``chunk_range["min"]``, one chunk
          included.
        - Otherwise, if the call fits in one chunk, it runs the one-chunk kernel.
        - Otherwise the call pads to the bucket's static chunk count.

        Returns None past ``max_pages``.
        """
        if num_blocks > self._max_pages:
            return None
        width = self.chunk_width(num_seqs_bucket)
        needed = self._chunks_needed(num_seqs_bucket, num_blocks)
        chunk_range = self._chunk_ranges.get(num_seqs_bucket)
        if chunk_range is not None:
            return BatchedDecodePlan(
                num_blocks, width, max(needed, chunk_range["min"]), chunk_range
            )
        if needed == 1:
            return BatchedDecodePlan(width, width, 1, None)
        static = self._static_chunks[num_seqs_bucket]
        return BatchedDecodePlan(static * width, width, static, None)

    @property
    def shared_slots(self) -> int:
        """The block slots of one shared-slot chunk."""
        return _SHARED_SLOTS

    @property
    def shared_slot_seqs(self) -> int | None:
        """The most decode sequences one shared-slot call serves, None without the kernel."""
        return None if self._shared_slot_chunk_range is None else self._shared_slot_seqs

    @property
    def shared_slot_chunk_range(self) -> DynamicRange | None:
        """The ``min``, ``max`` and ``granularity`` of the dynamic shared-slot kernel's chunks."""
        return self._shared_slot_chunk_range

    @property
    def splits_shared_slot_calls(self) -> bool:
        """Whether some batch of up to ``shared_slot_seqs`` sequences needs several calls."""
        if self._shared_slot_chunk_range is None:
            return False
        assert self._page_range is not None
        return self.shared_slot_plan([self._page_range["max"]] * self._shared_slot_seqs) is None

    def shared_slot_plan(self, blocks: list[int]) -> SharedSlotPlan | None:
        """The layout of a shared-slot call over sequences of ``blocks`` blocks each, or None
        without the shared-slot kernel, when the call has more than ``shared_slot_seqs``
        sequences, or when it needs more chunks than one call runs (see
        ``shared_slot_call_groups``).

        Every call takes the dynamic kernel, a single chunk included, at no fewer chunks
        than its range minimum.
        """
        chunk_range = self._shared_slot_chunk_range
        if chunk_range is None or len(blocks) > self._shared_slot_seqs:
            return None
        # A sequence with no blocks still takes a block slot, fully masked, so its row is
        # finite.
        num_chunks, slots = share_block_slots([max(1, b) for b in blocks])
        if num_chunks > chunk_range["max"]:
            return None
        return SharedSlotPlan(max(num_chunks, chunk_range["min"]), chunk_range, slots)

    def shared_slot_call_groups(self, blocks: list[int]) -> list[list[int]] | None:
        """The indices of ``blocks`` split into shared-slot calls, longest sequences first,
        each taking the first call it fits; None without the shared-slot kernel or when
        there are more than ``shared_slot_seqs`` sequences.

        A single sequence always fits one call, since the chunk range reaches what one
        max_model_len sequence needs.
        """
        if self._shared_slot_chunk_range is None or len(blocks) > self._shared_slot_seqs:
            return None
        groups: list[list[int]] = []
        for i in sorted(range(len(blocks)), key=lambda i: -blocks[i]):
            for group in groups:
                if self.shared_slot_plan([blocks[j] for j in (*group, i)]) is not None:
                    group.append(i)
                    break
            else:
                groups.append([i])
        return groups

    def min_real_query_len(self, padded_query_len: int) -> int:
        """Smallest runtime query_len that rounds up onto ``padded_query_len``."""
        idx = bisect.bisect_left(self._query_buckets, padded_query_len)
        return self._query_buckets[idx - 1] + 1 if idx else 1

    @staticmethod
    def _round_up(n: int, buckets: list[int]) -> int | None:
        idx = bisect.bisect_left(buckets, n)
        return buckets[idx] if idx < len(buckets) else None

    def variants(self) -> list[SpyreAttnBucket]:
        """Every variant worth recording, largest first: one per query bucket.

        Each is requested at ``max_pages``, or at the context its smallest real query
        needs if that is longer, and ``build()`` realizes its own page count: rounded into
        the range, or under a sliding window the pages the window touches. A query bucket
        no context up to max_model_len can hold is left out. The bound is on the
        *smallest real* query_len that reaches a bucket, not the bucket itself, since a
        2-token query on a 1-block sequence still dispatches to a large padded bucket;
        bounding by the bucket would prune that variant and put a compile back in the
        serving path.
        """
        out = []
        for padded_query_len in sorted(self._query_buckets, reverse=True):
            needed = cdiv(self.min_real_query_len(padded_query_len), self.block_size)
            if needed > self._context_pages:
                continue
            num_blocks = max(self._max_pages, needed)
            out.append(SpyreAttnBucket(num_blocks=num_blocks, padded_query_len=padded_query_len))
        return out

    def batched_decode_variants(self) -> list[SpyreAttnBatchedDecodeBucket]:
        """Every batched decode variant worth recording, largest first.

        Per num_seqs bucket: its dynamic kernel, recorded at ``max_pages``; without a chunk
        range, the one-chunk kernel and its static chunk count. With the shared-slot kernel,
        only the buckets above ``shared_slot_seqs``.
        """
        if not envs.SPYRE_BATCHED_DECODE:
            return []
        shared_slot_seqs = self.shared_slot_seqs or 0
        out: dict[tuple[int, int, int | None], SpyreAttnBatchedDecodeBucket] = {}
        for num_seqs in self._num_seqs_buckets:
            if num_seqs <= shared_slot_seqs:
                continue
            # The longest context first, so a dynamic kernel is recorded at max_pages.
            for num_blocks in (self._max_pages, self.chunk_width(num_seqs)):
                plan = self.batched_decode_plan(num_seqs, num_blocks)
                if plan is None:
                    continue
                dynamic = plan.chunk_range is not None
                bucket = SpyreAttnBatchedDecodeBucket(
                    num_seqs=num_seqs,
                    # Recorded at a context a sequence can have; the chunks still pad past it.
                    num_blocks=min(plan.num_blocks, self._max_pages),
                    blocks_per_chunk=plan.blocks_per_chunk,
                    num_chunks=plan.num_chunks,
                    dynamic_chunks=dynamic,
                )
                # One dynamic kernel runs every chunk count.
                key = (num_seqs, plan.blocks_per_chunk, None if dynamic else plan.num_chunks)
                out.setdefault(key, bucket)
        return sorted(out.values(), key=lambda b: (b.num_blocks, b.num_seqs), reverse=True)

    def shared_slot_variants(self) -> list[SpyreAttnBatchedDecodeBucket]:
        """The shared-slot kernel, recorded at ``shared_slot_seqs`` sequences filling its
        most chunks. Empty without the shared-slot kernel."""
        num_seqs = self.shared_slot_seqs
        if not envs.SPYRE_BATCHED_DECODE or num_seqs is None:
            return []
        assert self._page_range is not None and self._shared_slot_chunk_range is not None
        full_call_blocks = self._shared_slot_chunk_range["max"] * (_SHARED_SLOTS // num_seqs)
        num_blocks = min(self._page_range["max"], full_call_blocks)
        plan = self.shared_slot_plan([num_blocks] * num_seqs)
        assert plan is not None
        return [
            SpyreAttnBatchedDecodeBucket(
                num_seqs=num_seqs,
                num_blocks=num_blocks,
                blocks_per_chunk=_SHARED_SLOTS,
                num_chunks=plan.num_chunks,
                dynamic_chunks=True,
                shared_slots=True,
            )
        ]
