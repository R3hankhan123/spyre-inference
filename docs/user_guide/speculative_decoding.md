# Speculative decoding

Spyre supports four speculative-decoding methods. Each is a plug-in
(`spyre_inference/v1/spec_decode/`); verification is shared by all of them.

| Method | `--speculative-config` | Drafter | Limits |
|---|---|---|---|
| n-gram | `{"method": "ngram", "num_speculative_tokens": k}` | none (prompt lookup on the host) | — |
| suffix | `{"method": "suffix", "num_speculative_tokens": k}` | none (suffix tree on the host) | needs `arctic-inference` |
| Gemma-4 MTP | `{"method": "mtp", "model": "<gemma-4 assistant>", "num_speculative_tokens": k}` | Gemma-4 assistant checkpoint, reading the target's KV cache | Gemma-4 targets |
| XPress / DFlash | see [below](#xpress) | Qwen3 XPress checkpoint | TP=1, one request |

Any other method is rejected at startup with the list above.

**Shared verification.** A verification step packs `1 + k` tokens per running
sequence. The default compile sizes and attention query buckets add that width (see
[Configuration](configuration.md)), the scheduler pads every draft list to `k` so each
verification has one width, and the batched decode kernel serves it. Rejection
sampling runs on the fused C++ kernels in `csrc/cpu/spec_decode_utils.cpp`; build the
CPU extension before serving.

**Adding a method.** Subclass `SpyreSpecMethod` (`spec_decode/base.py`), override only
the hooks the method needs (every default is upstream's behaviour), and add it to
`registry.METHODS`. A drafter that attends through the target's attention builders
sets `draft_query_len` so warmup records the batched kernel at its width; one with KV
layers of its own sets `records_draft_attention`.

## Gemma-4 MTP

```bash
vllm serve google/gemma-4-26B-A4B-it -tp 4 --max-model-len 32768 --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' \
  --speculative-config '{"method": "mtp", "model": "<gemma-4-26B-A4B-it assistant>", "num_speculative_tokens": 4}'
```

The assistant is compiled per block like the target and drafts one token per request
per pass, at the decode buckets. Its context is extended to the target's
`--max-model-len`, and async scheduling is turned off: Spyre runs speculative decoding
synchronously. Under tensor parallelism the assistant's pre/post projections are held
whole on every rank, so drafting runs no collective.

## XPress

XPress runs a DFlash backbone once, refines a block of draft tokens, and lets
the target model verify the proposals. The experimental Spyre integration loads
the published `UIUC-SSAIL/Qwen3-8B-XPress-b16` checkpoint directly with
`Qwen/Qwen3-8B`; checkpoint conversion is not required.

Supported configuration: compiled FP16, unquantized weights, TP=1, PP=1,
the V1 runner, one active request, and prefix caching
disabled. The checkpoint fixes the block at one anchor plus 15 proposals.

### Run Qwen3-8B

```bash
export SPYRE_MODEL_RUNNER=v1
export SPYRE_ATTN_QUERY_BUCKETS=1,16,64
uv run --no-sync vllm serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --tokenizer-revision b968826d9c46dd6066d109eabc6255188de91218 \
  --dtype float16 --tensor-parallel-size 1 \
  --max-num-seqs 1 --max-model-len 512 --max-num-batched-tokens 64 \
  --no-enable-prefix-caching \
  --compilation-config '{"compile_sizes": [1, 16, 64]}' \
  --speculative-config '{"method": "dflash", "model": "UIUC-SSAIL/Qwen3-8B-XPress-b16", "revision": "f098ab5bbe4fce37c4a1093266bcd8ad15df5cb5", "num_speculative_tokens": 15, "xpress_topc": 128}'
```

Requests can use greedy (`temperature=0`) or stochastic target sampling,
including temperature, top-k and top-p. The drafter remains greedy;
vLLM treats its proposals as point-mass distributions during CPU rejection
sampling. The same speculative configuration works with `vllm.LLM`.
The end-to-end test and CI's public-model cache list pin the same model and
checkpoint revisions as this example; update all three together. A full
verification round needs at least 31 budgeted tokens because vLLM reserves
additional drafting slots; the adapter rejects a smaller budget.

### Refinement and compatibility

The refiner follows [merged vLLM XPress](https://github.com/vllm-project/vllm/pull/54448):
a fixed anchor, causal Jacobi updates, a block-wide hidden summary, and candidate
sets fixed across passes. Target input embeddings and output-head weights are
shared separately. Accepted target states populate the draft KV cache; rejected
and padded rows write only the reserved null page.

`xpress_topc` overrides the checkpoint field. Spyre defaults to 128 when neither
supplies a value; explicit 512 retains the upstream default and 0 enables
full-vocabulary refinement. A smaller shortlist can change acceptance. The
checkpoint selects the pass count, with six as the default.

Replaying a seeded request with the same configuration is deterministic.
Changing between ordinary decoding, DFlash and XPress changes RNG consumption;
the same seed need not produce identical tokens across these modes.

The pinned vLLM 0.32 lacks native XPress registration, so this adapter uses its V1
DFlash classes with a configuration bridge. The merged GPU speculator uses V2.
Initial selection and per-pass argmax run on CPU; readout and refinement run on
Spyre. Batched XPress and device candidate selection are follow-up work.

Spyre's V2 runner still lacks model loading, compilation and warmup. After that
runner is operational, XPress needs its auxiliary-state, draft-context and
CPU-sampling hooks ported to V2's speculator interface. See
[the V2 migration issue](https://github.com/torch-spyre/spyre-inference/issues/906).

### Validation

```bash
uv run --no-sync pytest tests/spec_decode -m 'not upstream'
SPYRE_XPRESS_TEST_TOPC=128 uv run --no-sync pytest \
  tests/e2e/test_speculative_decoding.py -m model_quality -s
```

PR regression jobs prefetch both pinned models before running the quality gate.
On Spyre, missing checkpoints fail the tests instead of skipping coverage.

CPU recurrence, sampling and lifecycle tests pass. At C128, device validation
passes seven reference cases and 63 strict greedy token comparisons across
scripted rejection, full acceptance, DFlash and XPress. Stopping, cancellation
and cache reuse checks also pass, including exact controlled-cache comparisons.

A known numerical limitation remains: nearly tied FP16 target scores can select
different greedy tokens between ordinary decoding and block verification. The
counting prompt avoids the previously observed Spain/Germany tie; the underlying
numerical difference is not fixed. Repeat device validation with C512 when
qualifying that configuration.

Seeded chunked prefill preserves CPU RNG state when discarding intermediate
samples. At C128, stochastic device validation passes 24 generations across
ordinary decoding, DFlash and XPress: temperature, top-k, top-p, and combined
filtering near the context limit, each repeated with exact token and cache replay.

The 503-token case starts too close to the context limit to draft. A separate
495-token case verifies that a previously nonempty draft is cleared at the limit
and that the next target step has no scheduled draft tokens. Validation timings
are collected by the test worker; the serving loop has no timing counters.
