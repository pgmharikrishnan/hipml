# WP45: a serving loop and where it loses

> *Engineering log from development; internal work-package IDs and paths have been sanitized or removed where they were machine-specific. Current numbers: [docs/benchmarks.md](../benchmarks.md).*

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

## WP51: making the server safe to leave running

Review 4 found the server fast but not safe: one bad request took the process
down, a burst queued without limit, `temperature` was silently ignored, the
batched path was not bit-exact at four or more slots, and the first request paid
a ~5 s first-touch cost. This section is what changed and what it cost.

### 1. A failure fails one job

Before: every `?` in the scheduler loop returned to `sched::run`, which returned
to `serve::run`, which exited 1 — so an allocation failure, a transient HIP
error or one slot's bad state dropped every in-flight stream.

Now every fallible step has a scope (`serve/sched/fail.rs`):

| scope | sites | what happens |
|---|---|---|
| one job | admit, prefill tile, spec seed, spec step, adopt, activate, a corrupt id | that job gets an error event, its slot is freed, its prefix-cache entries are dropped (`prefix_invalidate`), the engine keeps serving |
| one batch | the shared weight pass (`BatchEngine::forward`) | no slot can be blamed for a launch that covers every slot: every job in the batch fails, the batch engine is reset (zeroed SSM state, empty slot set), and the next admission starts clean |
| the device | a reset that itself fails, or three resets with no successful step between them | every stream is drained with an error, the scheduler waits up to 2 s for the connections to close (`admit::await_close`), then the process exits 1 so a supervisor restarts it |

The per-slot scope is real, not a wrapper: after the shared pass every id is
checked against the vocabulary, and an out-of-vocabulary id fails **that slot's
job** while the other slots' tokens — already computed in the same pass — are
delivered unchanged. `HIPML_FAULT_SLOT=n` (with `HIPML_FAULT_STEP`, default 4)
injects exactly that failure so the probe can prove it.

`begin` now hands the job back with its error (`Result<Prefill, (Job, String)>`)
instead of dropping it, so a rejected request can still be answered and its
prefix entries invalidated.

**Evidence** (`bench/serve/wp51_probe.py --plan isolation`): four concurrent
streams, 24 tokens each, first a no-fault run and then the same four with slot
1's step injected to fail.

| run | streams | tokens per stream | errors | resets |
|---|---|---:|---|---:|
| no fault | 4 | 24, 24, 24, 24 | 0 | 0 |
| slot 1 faulted | 4 | 24, 24, 24, **3** | 1 (the 3-token stream) | **0** |

The three intact streams' ids are **identical to the no-fault run, token for
token**. `/metrics` shows `hipml_slot_faults_total 1`, `hipml_job_errors_total
1`, `hipml_batch_resets_total 0`; the process stays healthy and a request after
the fault returns 8/8 tokens. The 4x24 wave took 1.97 s.

**The other two scopes** (`--plan scopes`, which injects into the shared pass):

| injection | result |
|---|---|
| one pass fails with four slots active (`HIPML_FAULT_BATCH=30`) | all four jobs in that pass get an error event, `hipml_batch_resets_total 1`, `hipml_job_errors_total 4`; the next request in the same process returns 8/8 tokens |
| every pass fails (`HIPML_FAULT_BATCH_ALL=1`) | three resets with no step between them, all four streams get an error event, then the process exits **1** |

### 2. Backpressure

The real limit is VRAM slots, not threads: a slot costs its KV cache plus its
SSM recurrent state (hundreds of MiB at 8K ctx), while a queued request costs
only its token ids on the host (4 bytes a token). So the queue is bounded as a
small multiple of the slots and the *connection* bound is separate and looser:

* `inflight = slots + queue_max`, with `queue_max` = `--max-queue` (default
  `2 x slots`, i.e. 32 for 16 slots). This is the admission bound.
* `--max-conns` (default 64) bounds connection threads, so a flood of idle
  sockets cannot exhaust memory either.
* A generation request past `inflight` is answered `503 Service Unavailable`
  with `Retry-After: 1` and a JSON error, **before** it is tokenized or queued.
  `/health`, `/metrics`, `/props` and `/tokenize` are never refused.

`/metrics` exposes the bound (`hipml_inflight_max`, `hipml_queue_max`,
`hipml_connections_max`) and the live values, plus
`hipml_requests_rejected_busy_total` and `hipml_connections_rejected_total`.

**Measured** (`--plan backpressure`, 40 concurrent requests against 16 slots):

| queue bound | streams | 503s | other | hangs | wall |
|---|---:|---:|---:|---:|---:|
| `--max-queue 4` (capacity 20) | 20 | 20 | 0 | 0 | 5.1 s |
| default 32 (capacity 48) | 40 | 0 | 0 | 0 | 9.7 s |

Every one of the 40 requests was answered, each 503 carried `Retry-After: 1`
and a JSON body, `hipml_requests_rejected_busy_total` counted exactly the 20
refusals, and `hipml_inflight` was back to 0 afterwards. No request hung past
the 180 s read timeout (the slowest finished in 9.7 s).

### 3. Sampling honesty

Chosen: **refuse, do not fake.** A request with `temperature > 0` or `n > 1` is
a `400` whose message says the server is greedy-only. The alternative —
implementing a sampler — would put a second, non-argmax decode path beside the
exactness claims this project is built on, and the batched kernels sample per
slot with `argmax` only; a half-implemented sampler is worse than an honest
refusal.

`top_p` and `seed` are accepted and are no-ops *by arithmetic*: under greedy
decode the argmax is inside every nucleus, and a seed has nothing to seed.
`temperature: 0` and an absent temperature are both accepted and produce the
same ids.

### 4. Batched exactness

The serving default is still the WMMA tile at B>=4 (`HIPML_BATCH_SKINNY=1`),
because the exact `_b` path costs the throughput the fleet numbers depend on
(measured below). The divergence is now impossible to miss:

* every generation response carries `X-Hipml-Batch-Exact: 0` (with
  `X-Hipml-Batch-Tile-Min: 4`) when the tile can be used, and `1` when the exact
  path is the only one;
* `/metrics` has `hipml_batch_exact_path` (0/1) and
  `hipml_tile_steps_total`, the number of decode steps that actually ran
  through the tile;
* `HIPML_BATCH_LEGACY=1` makes the whole server exact.

**Cost of exact** (`--plan exact`, 8 and 4 slots, 400-token prompts, 64 tokens,
one warm wave):

| slots | tile aggregate tok/s | exact aggregate tok/s | cost of exact |
|---:|---:|---:|---:|
| 4 | 67.7 | 66.3 | 2% |
| 8 | 107.1 | 75.3 | **30%** |

(400-token distinct prompts, cache off, 64 tokens, one warm wave; the same
binary with only `HIPML_BATCH_LEGACY` changed.) At B=4 the tile barely wins, so
exactness is nearly free there; the 8-slot fleet case is where it costs. That is
why the tile stays the default and the divergence is advertised instead: on the
published protocol the same build still gives **129.8 tok/s at 8 slots** (cell
b, 0 errors) and **214.9 tok/s at 16** (cell c), so forcing exact would drop the
8-slot number from 130 to about 90 and break the fleet claim.

Header and gauges, measured (`--plan headers`): a default reply carries
`X-Hipml-Batch-Exact: 0` and `X-Hipml-Batch-Tile-Min: 4`; four concurrent
streams push `hipml_tile_steps_total` to 4; with `HIPML_BATCH_LEGACY=1` the
header is `1` and the gauge `hipml_batch_exact_path` is 1.

### 5. Warm-up and supervisor

`serve` now warms three paths at load (`serve/sched/warm.rs`): one full
`--chunk` prefill tile (every weight page and every prefill kernel), one batched
step on two synthetic slots (the batch kernels), and one drafter step. Then it
resets the state. `HIPML_NO_WARMUP=1` skips it, which is how the cost is
measured.

| first request | load | TTFT, 200-token prompt | TTFT, 2000-token prompt |
|---|---:|---:|---:|
| before WP51 | 8.6 s | 4797 ms | 1869 ms |
| no warm-up (`HIPML_NO_WARMUP=1`) | 12.8 s | 238 ms | 1861 ms |
| warm-up (default) | 13.0 s (269 ms of it warm-up) | **210 ms** | 1870 ms |

**The ~5 s was not first touch.** The listener used to start accepting before
the scheduler had built its engines, so `/health` answered ~4.4 s early and the
first request queued behind `GpuEngine`/`BatchEngine` construction (26 GB of
device buffers). The fix that mattered is a readiness gate: the port is bound
while the engines warm up, but nothing is accepted until the scheduler signals
ready, so `/health` means "warm and serving". That is why load time grew by the
same 4.4 s the first request used to pay. The warm-up itself (one 256-token
prefill tile, one two-slot batch step, one drafter step) costs 269 ms of load
and removes the remaining first-launch cost; the 28 ms difference between warm
and cold is inside run-to-run noise at this prompt size.

`scripts/hipml-serve.ps1` is the supervisor: start, log to a timestamped file,
wait for exit, relaunch with a 2 s → 60 s backoff (reset after 60 s of uptime),
stop the child on Ctrl+C. A context loss now costs one restart, not an outage.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\hipml-serve.ps1 -Slots 16 -Ctx 32768
```

Tested end to end (`bench/serve/wp51_supervisor_test.ps1`): the child was killed
(pid 14800, the same thing a context loss looks like from outside), the loop
relaunched it as pid 21312 within the 2 s backoff, `/health` answered, and a
`/completion` after the restart returned HTTP 200 — one restart, no outage.

### Regression gates

| gate | result |
|---|---|
| `oracle-gate --ids-file p2_math_169.ids --chunk 2048` | PASS, 20/20 ids identical |
| `parity --deep` | 56 passed, 0 failed, 1 skipped |
| `cargo test -p hipml-core --test tokenizer_fixture` | 2 passed |
| `cargo test --workspace` | all suites pass |
| serve_bench cell b (8 slots) | 129.8 tok/s aggregate, TTFT p50 492 ms, 0 errors, 26.3 GiB |
| serve_bench cell c (16 slots) | 214.9 tok/s aggregate, TTFT p50 753 ms, 0 errors, 28.3 GiB |
| single-stream MTP | 39.0 tok/s p50, TTFT 251 ms |
| review 4 crash plan | every malformed input still a 400 (one 100 KB request line is a connection reset), server healthy throughout |

### What is still not done

* The batch-wide reset is the honest scope for a shared weight pass: a launch
  that fails cannot be pinned to one slot, so those jobs fail together. Only
  per-slot failures (the common case) are isolated.
* The fault hooks (`HIPML_FAULT_SLOT`, `HIPML_FAULT_BATCH`,
  `HIPML_FAULT_BATCH_ALL`) inject the failure at the scheduler's decision
  points, not inside a kernel: the recovery path is the real one, the failing
  HIP call is not.
* A multi-hour soak was not run; the probes run minutes, not hours.
* The supervisor is a restart loop, not a health monitor: it does not restart a
  server that is up but wedged (a step that never returns). A watchdog on
  `hipml_decode_steps_total` would be the next step.
* A prompt longer than the context still answers 500 rather than 400 (review 4
  item 24); it is unchanged because the 500 comes from the generic job-error
  path, and separating client errors from device errors there is not a
  one-liner.
