# WP45: a serving loop and where it loses

`hipml serve` is an HTTP server with continuous batching, MTP speculation for a
lone request, and a hybrid (KV + SSM) prefix cache. This note records the
design, the first cross-engine measurements against llama-server, and the
places the numbers do not meet the targets.

## Server design

One process, one GPU, one scheduler thread. HTTP connection threads parse the
request, tokenize it and stream chunks back over a per-request `mpsc` channel;
only the scheduler touches the GPU, so there are no device locks and every
kernel launch sees the same stream.

* `POST /v1/chat/completions` (OpenAI chunks, SSE when `stream` is true),
  `POST /completion` (llama-style: string or raw token ids, `content`/`tokens`/
  `timings`, SSE), `GET /health`, `GET /metrics`, `GET /props`,
  `POST /tokenize`. HTTP/1.1 over `std::net::TcpListener`; no external crates.
* **Lone request**: chunked prefill on `GpuEngine`, then MTP draft+verify steps.
* **Concurrent requests**: a `BatchEngine` slot per request, one batched weight
  pass per decode step. A request that arrives while another is decoding is
  prefilled in `--chunk` tiles, one tile between decode steps.
* **Handoff**: the prefilled single-sequence KV rows and SSM state are copied
  device-to-device into the slot (`batch/adopt.rs`), so the first decoded token
  is the same token the single engine would have sampled.
* **Slots must stay dense.** The batched KV/attention/recurrent kernels use
  `blockIdx.y` as both the row index and the physical slot. New requests take
  the next index; release swaps the last slot's state down with a device copy.
  The scheduler releases retirees high-to-low so a retiring request never moves.
* **Prefix cache**: capture at rendered message boundaries, keyed by the exact
  token prefix; restore copies the 16 full-attention KV slices and the 48 SSM
  recurrent states. Capture and restore are queued on the engine stream and
  synchronised once. A hit covers the system turn; the remaining user turn is
  prefilled normally. Cache hits skip MTP, because the drafter's KV cannot
  cover a restored prefix.

## Measurement method

Same box (Radeon AI PRO R9700, 32 GiB), same GGUF (`Qwen3.8-27B-Q4_K_M`), q8 KV
on both engines, greedy, 64 new tokens per request, a ~2.4K-token system prompt
shared by every conversation, and distinct user turns. Three waves per cell
after one warm wave; TTFT is the first streamed chunk, decode is over tokens
2..64. `bench/serve/serve_bench.py` runs one engine/cell per invocation and
stores `bench/serve/results/<engine>_<cell>.json`.

| cell | streams | ctx per slot | hipml | llama |
| --- | ---: | ---: | --- | --- |
| a | 1 | 32768 | `--slots 1`, `cache_prompt:false` (MTP path) | `-c 32768 --parallel 1` |
| b | 8 | 16384 | `--slots 8 --ctx 16384` | `-c 131072 --parallel 8` |
| c | 16 | 8192 | `--slots 16 --ctx 8192` | `-c 131072 --parallel 16` |

llama-server's `-c` is the total KV budget and `--parallel` divides it, so the
script passes `ctx * streams`. Cell a keeps 32K. Cell b cannot use 32K per slot
on hipml: 8 slots plus the prefill engine and weights measured 32.2 GiB of the
31.9 GiB usable and paged. 16 x 32K is impossible for either engine.

## Results

| cell | engine | TTFT p50 | TTFT p95 | per-stream tok/s | aggregate tok/s | peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| a | hipml | 2955 ms | 2963 ms | **38.3** | 38.3 | 21.6 GiB |
| a | llama | 2182 ms | 2194 ms | 30.0 | 30.0 | 17.4 GiB |
| b | hipml | 1004 ms | 1715 ms | 10.1 | 70.6 | 25.4 GiB |
| b | llama | 661 ms | 1345 ms | 12.3 | **92.3** | 22.0 GiB |
| c | hipml | 1960 ms | 4086 ms | 5.2 | 71.1 | 27.1 GiB |
| c | llama | 1338 ms | 2951 ms | 13.6 | **211.3** | 23.1 GiB |

Cell a is the one win: the speculative decoder is 1.28x llama's single-stream
rate. Cold TTFT is 35% slower because hipml prefills at ~820 tok/s against
llama's ~1110 tok/s.

Cell b and c are losses. At 8 streams hipml reaches 76% of llama's aggregate
and at 16 streams 34%. Hipml's aggregate is flat (71 tok/s at both 8 and 16
slots) while llama's doubles: hipml's step is limited by per-slot work
(attention over the slot's context and the SSM recurrence), so adding slots
divides the same aggregate; llama still has weight-pass headroom.

## Prefix cache

With the shared system prompt, `hipml_prefix_hit_rate` was **1.0** after the
first (warm) request: cell b 31/31 lookups hit, cell c 63/63, restoring 2396
tokens per request. Average restore was 1.8 ms (0.116 s over 63 restores) and a
2.4K capture was ~6 ms. Cold TTFT for the same prompt is 2.96 s; a hit removes
the 2396-token prefill.

The hit does not make the fleet fast, because every request still has to
prefill its own user turn through the single prefill engine, and that tile is
scheduled between batch steps:

| cell | interleaved tiles | stall total | stall max |
| --- | ---: | ---: | ---: |
| b | 21 | 3.36 s | 161 ms |
| c | 45 | 7.22 s | 164 ms |

A 26-token tail tile still streams the whole weight set once and measured
~160 ms, against a ~30 ms roofline; it also holds the batched step off the GPU
for that time. That is why per-stream decode halves from 10.1 to 5.2 tok/s
between b and c while the aggregate stays flat. The answer to "does
interleaved prefill wreck decode": it costs 3.4 s / 7.2 s of measured stall in
these waves, and it is the largest term in cell c's TTFT tail (p95 4.1 s vs
llama's 3.0 s).

## Acceptance

* Single-stream chat **38.3 tok/s** at 32K with the 2.4K prompt, short of the
  >=40 target; on a 2425-token prompt with 16 generated tokens the same server
  measured 42.4 tok/s. Both are MTP.
* **Identical to CLI**: yes on the check that matters: same ids file, same
  ctx/chunk, server and `hipml spec-bench` produced **16/16 identical tokens**
  (server 42.4 tok/s, CLI 39.5).
* 8-concurrent aggregate **70.6 vs 92.3** llama: target missed.
* Shared-prefix TTFT **1.00/1.96 s vs 0.66/1.34 s** (p50) and **1.72/4.09 s vs
  1.35/2.95 s** (p95): target missed.
* Interleaved prefill is measured above rather than assumed; it is the main
  reason the concurrency cells lose.

## What a user would feel

A single chat with this model is the good case: it streams noticeably faster
than llama-server and the first question after a shared system prompt is
served from cache in about a fifth of the cold time. Put eight or sixteen
agents behind it and the experience changes: each reply streams at roughly half
to a quarter of llama-server's rate, a question that arrives while others are
decoding waits one to four seconds for its first token, and the machine's total
output stops growing after about eight streams even though every extra agent
gets a slot. The shared system prompt saves the big prefill, but not the small
tail or the stall it causes.

## Not verified / next steps

* No token-identity check for the **batch** handoff (adopt path) against the
  single engine; only the speculative path was compared token-for-token.
* Batch cells run with MTP resident but unused. MTP + a mixed batch was not
  exercised.
* Interleaved prefill was not timed with MTP as the concurrent decoder (batch
  cells never use speculation).
* llama's per-request prefix reuse is inferred from TTFT, not from a counter;
  its `cache_prompt` accounting was not exported by the build used here.
* Peak VRAM includes model load transients; it is a process-dedicated peak,
  not steady-state after a GC.
* Next step, in order: prefill multiple cache-hit tails in one pass (they are
  independent sequences, so the single engine cannot do it today); make the
  small-M prefill tile cheaper than a full weight pass; then retest cells b/c.

## WP46: where the fleet aggregate actually stops

The exact-prefill merge (`feat/exact`) is in this tree: `oracle-gate` returns
20/20 ids on `p2_math_169` (step 19 = 41020). Cell a is unchanged, so the fix
did not move the single-stream path.

Two concurrency changes were measured on that tree.

**Finishing a cache-hit tail inside the batched step does not pay.** A 26-token
tail was fed one row per batched step instead of taking the prefill engine for
one tile. It removes the exclusive stall, but a tail then costs 26 weight
passes instead of one: cell b TTFT p50 1.0 -> 2.7 s and aggregate 70.6 -> 61.8
tok/s. It was reverted rather than tuned.

**The batched decode step was the other half of the loss.** At B>=4 the exact
`_b` GEMV gives way to the WMMA skinny tile by default in `serve`
(`HIPML_BATCH_SKINNY`; B<4 and the lone MTP request keep the exact path, so
cell a and the one-request token check are unchanged). Same box, same prompts,
q8 KV:

| cell | slots | engine | TTFT p50/p95 ms | per-stream tok/s | aggregate tok/s |
|---|---:|---|---:|---:|---:|
| a | 1 | hipml before | 2955 / 2963 | 38.3 | 38.3 |
| a | 1 | hipml after | 2950 / 2967 | 38.2 | 38.2 |
| a | 1 | llama | 2182 / 2194 | 30.0 | 30.0 |
| b | 8 | hipml before | 1004 / 1715 | 10.1 | 70.6 |
| b | 8 | hipml after | 985 / 1633 | 15.3 | **101.5** |
| b | 8 | llama | 661 / 1345 | 12.3 | 92.3 |
| c | 16 | hipml before | 1960 / 4086 | 5.2 | 71.1 |
| c | 16 | hipml after | 1877 / 3435 | 11.9 | 138.6 |
| c | 16 | llama | 1338 / 2951 | 13.6 | 211.3 |

b now clears llama on aggregate and per-stream. c's per-stream is inside the
20% band (11.9 vs 13.6) but its aggregate is not.

The remaining c gap is the tail prefill, measured rather than inferred: 45
interleaved tiles, **7.19 s of exclusive GPU time**, max stall 163 ms, against
a 7.27 s median wave window. The decode steps themselves are already close to
llama (11.9 vs 13.6 per stream). Removing the 2.4 s per wave projects
`1008 / (7.27 - 2.40) = 207` tok/s, i.e. the aggregate gap is the serialized
tail, not the batch decoder. The next step is a true multi-row prefill for the
tails (one projection pass over N independent suffixes) — the scheduler has no
such path today; the single-sequence prefill engine cannot express it.

## WP47: one weight pass for N prefix-hit tails

The WP46 next step is implemented: a prefix-hit request whose remaining suffix
is short no longer takes the prefill engine for its own tile. The scheduler
gathers every such tail that is ready at admission, copies each restored prefix
into a *quiet* batch slot (KV + SSM state, no `active` entry), and runs all of
their suffixes through one `BatchEngine::prefill_rows` call.

**Row layout.** Rows are concatenated tail by tail, so a slot's rows are
contiguous. The projections and the FFN are the ordinary tiled prefill GEMMs
over `M` rows. The per-sequence work stays per sequence:

* full attention: one `chunk_front_rows` + `kv_encode_bulk_rows` +
  `chunk_attention_rows` chain per tail (the slot's cache and its own position
  base), then `apply_output_chunk` over all rows;
* SSM: gates, L2 norm and gated RMS run once over `M` rows; the causal conv and
  the delta-net recurrence launch once per tail against that slot's window and
  matrix, using the same `ssm_conv1d_chunk_f32` / `gdn_rows_f32` kernels the
  single-sequence chunk path uses.

**Measured cost.** `rows-check` isolates the block: 16 tails of 12-14 tokens
(207 rows) take 237 ms, 16 tails of 17-19 (287 rows) take 355 ms, and one
6-row tail takes 149 ms. The fixed term is one 17 GB weight stream (~150 ms);
the marginal row is ~0.9 ms. The server counters agree: cell c ran 7 blocks,
1515 tail rows, 2.26 s of row-block time (36 ms per tail, 24 rows/tail); cell b
ran 7 blocks, 752 rows, 1.47 s (61 ms per tail).

**Gather wait.** `HIPML_ROW_GATHER_MS` bounds how long the scheduler waits for
a companion tail when the queue is empty (default 4 ms). Cell c is bursty: all
16 requests are enqueued before the first gather, so the wait never fires and
0/4/5/10 ms are within run-to-run variance (aggregate 214.4-217.0, TTFT p50
731-756 ms). It is kept for trickle arrivals, which this benchmark does not
produce.

| cell | slots | engine | TTFT p50/p95 ms | per-stream tok/s | aggregate tok/s |
|---|---:|---|---:|---:|---:|
| a | 1 | hipml before | 2950 / 2967 | 38.2 | 38.2 |
| a | 1 | hipml after | 2961 / 2978 | 38.1 | 38.1 |
| a | 1 | llama | 2190 / 2199 | 30.1 | 30.1 |
| b | 8 | hipml before | 985 / 1633 | 15.3 | 101.5 |
| b | 8 | hipml after | **491 / 492** | **17.5** | **129.5** |
| b | 8 | llama | 637 / 1319 | 12.0 | 95.3 |
| c | 16 | hipml before | 1877 / 3435 | 11.9 | 138.6 |
| c | 16 | hipml after | **756 / 758** | **15.2** | **214.4** |
| c | 16 | llama | 1380 / 2666 | 14.1 | 215.8 |

Interleaved prefill disappears as a cost: cell c reports 0 interleaved tiles
and 0 interleave stall (before: 45 tiles, 7.19 s).

**Token and state checks.** `hipml rows-check --gguf ... --slots 16 --prefix 96
--tail 12 --decode 4` gives every slot a distinct 96-token prefix and a
12-14-token tail, restores them into quiet slots, and compares the row block
against the same tails run one at a time through `prefill_tile`: all 16 first
ids and 48 subsequent batched-decoded ids match, and the **whole carried SSM
state (39,223,296 words: conv window + delta-net matrix per layer) is
bit-identical**. At 287 rows the ids still match but the state differs after
layer 0 in the last bits: the FFN split-K heuristic (`prefill_should_split`)
turns off once the row count crosses 256, so the wider tile accumulates in a
different order. That is the same caveat any longer prefill tile has; the
recurrence kernel and its per-slot arithmetic order are unchanged.

End to end on the server (`bench/serve/tail_check.py`): a 169-token prompt
prefilled normally and the same prompt with a 26-token tail served by the row
block emit the **same 20/20 ids** (step 19 = 41020), with
`hipml_tail_row_blocks_total 1`, `hipml_tail_rows_total 26`, 150 ms.

`oracle-gate --gguf ... --ids-file bench/fixtures/p2_math_169.ids --chunk 2048`
still returns **20/20 ids (step 19 = 41020)**, and `parity --deep` passes 56/56
(1 skipped). `cargo test --workspace` and `cargo check --workspace --features
hip` are clean.

**Not verified.**

* Tails longer than 64 tokens, or shorter than `d_conv-1` (3), still take the
  tiled prefill path; the crossover was not swept.
* Row blocks above 256 rows are token-identical but not state-bit-identical
  (above). Forcing the split policy or splitting the block at 256 rows would
  restore bit identity at the cost of another weight pass; not measured.
* The gather wait under genuinely trickle-arriving requests is unmeasured; the
  burst cells cannot exercise it.
* MTP was resident but unused in the b/c cells; a row-block tail during a lone
  speculative request was not run.
* The row block holds the GPU for ~0.15-0.36 s; cell-level TTFT shows it is
  paid once per wave, but it is still an exclusive stall for the decoding
  batch.

**Next step.** Make the FFN split-K choice depend on the block's *per-tail*
row count rather than `M` (or split the block at 256 rows) so the large-block
path is state-bit-identical too, then retest cells b/c and sweep the tail-length
crossover.
