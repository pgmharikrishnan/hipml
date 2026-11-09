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
