# Review 4: final audit of the server, batch engine and MTP verify

Base: `a9523ef` (feat/tail-prefill). Fixes are on `feat/review4`. The box is
an R9700 (gfx1201, 32 GB) running Qwen3.8-27B Q4_K_M with q8 KV. All GPU work
ran under the `review4` lock in five holds, each under 20 minutes, on port
8093 only. The probe client is `bench/serve/review4_probe.py`. Its plans are
`exact`, `crash`, `stress` and `recheck`.

## 1. The three claims

| Claim | Result | Numbers |
|---|---|---|
| Server, 8 slots: 129.5 tok/s aggregate | **Reproduced** | p50 129.3 (best 129.7); TTFT p50/p95 491/495 ms (claimed 491/492); 17.5 tok/s per stream; 0 errors in 24 requests; peak VRAM 26.3 GiB |
| 100K prefill: 928 tok/s, 107.8 s (WP42 fix) | **Reproduced** | 107.99 s TTFT, 926 tok/s (0.2% slower, within noise); decode 23.4 tok/s; llama.cpp's documented time is 120.9 s |
| MTP is exact under the server path | **Refuted at `a9523ef`** for drafts ≥ 2–3. **Holds after `b047930`** | see below |

**Caveat on claim 1.** This is a warm-cache, short-reply benchmark:

- 2,396 of the 2,422 prompt tokens were prefix-cache hits;
- every reply was 64 tokens, with `ignore_eos`.

It measures decode throughput, not prefill. It does not represent a cold
multi-user workload.

**Claim 3 in detail.** The test was `p2_math_169`, 128 tokens, compared with
plain greedy decode on the same server.

- **drafts=1:** exact over 128 and 343 tokens.
- **drafts=3:** diverged at token 20. Plain decode gave `[41020, 321, 1067…]`; drafts=3 gave `[41020, 11, 1179, 1067…]`.
- **CLI `spec-bench`:** exact in f32 and q8, at ctx 313 and 512. The bug is server-only.
- **Cause:** `serve` sets `HIPML_BATCH_SKINNY=1` for the whole process (`serve/mod.rs:153`). An MTP verify with 3 drafts is a 4-column `gemv_batched`. At B ≥ 4 the flag selects the WMMA tile, which is not bit-identical to the one-column launch. So the claim in `spec.rs:13-16` ("every verify column is bit-identical") was false in the server.
- **Fix (`b047930`):** `forward_columns` holds a thread-local `exact_scope()` unless skinny was requested explicitly. `graph_launch.rs:93` honours it.
- **After the fix:** drafts=2 and drafts=3 match plain decode for 128 tokens and for 343 tokens, the latter filling the context to its last row.

## 2. Findings, most severe first

Line numbers refer to `feat/review4` HEAD.

### Critical / High: fixed

1. **Deeply nested JSON kills the whole process.**
   - `json.rs` recursed without a limit, so `[[[[…` of about 100K levels overflowed the stack and aborted every slot.
   - Now limited to `MAX_DEPTH = 64` (`serve/json.rs:80`), and the request gets a 400. Lone low surrogates are also rejected.
   - Commit `d0d7610`.
2. **MTP near a full context kills the server.**
   - The verify appends `1 + drafts` KV rows, and nothing checked the row index on the device.
   - With drafts=3 near `n_ctx`, the scheduler failed with "position 512 beyond quantized cache 512", which exits the process (see High #5).
   - `spec.rs:147-150` now clamps the drafts to the room left. Fewer drafts cannot change the ids. Now 343 of 343 tokens complete.
   - Commit `6c3bc53`.
3. **The server's drafts=3 output is not greedy-exact.** See claim 3. Fixed in commit `b047930`.
4. **`/completion` with token ids passed out-of-vocabulary ids to the device.** `serve/chat.rs` now rejects any id ≥ vocab with a 400. Commit `bd8c39f`.

### High: unfixed

5. **One GPU error takes down the whole server.**
   - Every `?` in the scheduler loop returns to `sched::run`, which leads to `serve: scheduler: …` and process exit 1 (`serve/mod.rs:248-253`).
   - The sites are in `serve/sched/run.rs` at lines 45, 52, 207, 301-303, 321, 345, 359, 368 and 420. They cover alloc, prefill tile, batch step, spec step and snapshot.
   - So one bad request, OOM or transient HIP error drops every in-flight stream.
   - There is no supervisor and no restart.
   - This needs per-job error isolation (fail the job, reset its slot, keep the engine), plus an external restart. It is not a small fix.

### Medium: fixed

6. **No cancellation.**
   - A client that disconnected kept its slot until `max_tokens`, whether it was streaming, non-streaming, queued or mid-prefill.
   - Now `Job.cancel` is checked between tiles and steps, and `api.rs` peeks the socket every 250 ms of wall time.
   - Measured: a streaming disconnect frees its slot in 0.28 s; a non-streaming one goes idle in 0.54 s.
   - Commits `761476a` and `7743844`.
7. **Stop strings were ignored and leaked.**
   - Only single-token stop strings worked, and the stop text was sent to the client.
   - Text-level matching now cuts the output exactly before the stop and accepts a string or an array.
   - Commit `761476a`.
8. **Prefill errors were swallowed, and the spec path reported the wrong `finish_reason`.** Fixed in commit `761476a`.
9. **A malformed request got a silent close.**
   - Bad framing, a huge `Content-Length`, a chunked body or a 100 KB request line got a close or a 404.
   - They now get a 400 with a message, a 30 s read timeout and limits on line, header and body size.
   - Commit `dd66bda`.
10. **Prefix cuts were unbounded.**
    - Every message boundary was a cut, and each cut ends a prefill tile (a full ~150 ms weight pass) and captures state.
    - A 200-message chat made 201 captures, took 30.1 s to first token and flushed the prefix cache.
    - Cuts are now limited to the last two boundaries (`serve/chat.rs`). TTFT is now 4.4 s for 200 messages (was 30.1 s) and 0.92 s for 20 (was 3.5 s).
    - Commit `c8e7752`.
11. **A system prompt with trailing whitespace was never cached.**
    - The template trims the turn, so `end_of_turn` never found the boundary.
    - Fixed by trimming in `end_of_turn`, with a regression test. A second request now hits 763 cached tokens, and its TTFT drops from 4.97 s to 0.45 s.
    - Commit `c8e7752`.
12. **Metrics were wrong.**
    - `prompt_tokens_total` was 0 under mixed traffic (now 7032 in the same test).
    - The active gauge ignored prefills, so a cold 12K prefill looked idle.
    - Commit `af7ee37`.

### Medium: unfixed

13. **No backpressure.** The job queue is an unbounded `VecDeque` (`serve/sched/run.rs:61`), and every connection gets a thread (`serve/http.rs:41-49`). A burst queues without limit instead of returning 503.
14. **Sampling parameters are silently ignored.** `temperature`, `top_p`, `seed` and `n` are accepted and dropped; every reply is greedy. A client asking for temperature 0.8 gets greedy output and no warning.
15. **Thinking mode is hard-coded.** `enable_thinking: true` and `reasoning_effort: "xhigh"` are hard-coded (`serve/chat.rs:90-91`), and request fields cannot override them.
16. **Batched decode at B ≥ 4 is not bit-identical.**
    - With 4 or more active slots it uses the skinny WMMA path (`serve/mod.rs:153`).
    - No divergence was seen: all 5 mixed requests matched their solo output over 32 tokens, as did the batch-bench check in the docs.
    - Exactness is still not guaranteed, and the docs should say so.
17. **BPE is O(n²) per word** (`hipml-core/src/tokenizer.rs:250`). A long run of characters with no whitespace is slow to tokenize on the connection thread.

### Low: unfixed

18. **`prefix_lookups` counts only hits** (`serve/sched.rs:186`), so the hit rate reads 100%. Lookups and misses are also counted twice when a job is re-queued (`run.rs:179`).
19. **A failed `prefix_capture` fails the request** (`serve/sched.rs:287`). It is only an optimisation and could be skipped.
20. **`spec_drafted` counts `job.drafts`, not the clamped count** (`run.rs:371`), so it overcounts near the end of the context.
21. **`token_bytes` panics on an id ≥ vocab** (`tokenizer.rs:280`). It runs on the connection thread only, and ids are now validated upstream.
22. **A half-closed client looks gone.** `peer_closed` treats a client that shut down its write side as disconnected.
23. **No CORS or `OPTIONS` handling.**
24. **A prompt longer than the context returns 500,** not 400 (`serve/sched.rs:170`).
25. **The first request after load takes ~5 s longer** (warm-up). Adding a warm-up step to `serve` would hide it.

Checked and found correct: the prefix-cache key (exact token-prefix compare), the UTF-8 tail flush, the `max_tokens` boundary, and the HTTP/1.1 framing of SSE chunks.

## 3. Stress results (fixed tree)

| Scenario | Result |
|---|---|
| `max_tokens` exact | completion with n=1 and n=17 gives exactly n tokens, `stop=false`; chat with max 9 gives `completion_tokens` 9, `finish_reason: "length"` |
| Multi-token stop string `"hit/miss and"` | output cut exactly before it, `stop=true` |
| Client disconnects mid-stream | slot freed in 0.28 s, counted as cancelled |
| Client disconnects during a non-streaming request | base: never detected until generation ended; fixed: idle in 0.54 s, `cancelled 1` |
| Cancel during a 12K-token prefill | the short request queued behind it was served in 0.48 s |
| Two identical prompts at the same instant | both 200, both match their solo outputs; no in-flight dedupe, so neither hits the cache |
| Same prefix, different tails, concurrent | both 200 and exact; the second request hits 763 cached tokens (TTFT 4.97 → 0.45 s) |
| Mixed concurrent requests (120/400/900/1500/2200 tokens + 12K) on 4 slots | all 200; all 5 short ones match their solo output over 32 tokens; the 12K one took 21 s; healthy afterwards |
| 200-message chat (4,662 tokens) | TTFT 30.1 s → 4.4 s |
| 8,193- and 8,300-token prompts | work; the server's 256-token tiles avoid the split-scratch bug, which affects only the CLI with large chunks |
| Deeply nested JSON | base: **process abort**; fixed: 400 |
| Token id 10^7 | 400 |
| Prompt longer than ctx | 500 with a message (should be 400) |
| `n_predict` larger than the remaining room | silently clamped |
| Huge Content-Length / chunked body / 100 KB request line | 400 (base: silent close or 404) |
| Lone surrogate `\udc00` | 400 |
| drafts=3, reply filling ctx to the last row | base: **scheduler dies**; fixed: 343/343, exact |

## 4. Honest claim

hipml is ready today for one job: serving this one model on this one card to a
handful of users or agents who want deterministic (greedy) answers. In that
role it beats llama.cpp:

- a 100K-token document reads in 108 s, against llama.cpp's 121 s;
- eight concurrent streams of short replies over a shared prompt total about 129 tokens per second, against 95.

Its outputs match plain decoding token for token, but only with the fixes in
this review. Before them, the speculative path changed answers in the server.
Keep llama.cpp for:

- anything that needs temperature or sampling, which hipml silently ignores;
- sixteen or more concurrent users, where the two are about even;
- images;
- non-English text until WP48 lands;
- anything that must stay up unattended. Today, one GPU error anywhere in
  hipml ends every conversation and the process exits.

Move everything over only when:

- a failure kills only its own request, and something restarts the server;
- a full queue answers "busy" instead of growing without limit;
- sampling either works or is refused with an error;
- the batch path's exactness is documented or made the default;
- WP48 is merged;
- a multi-hour soak test passes with no restarts.

## 5. What I would not ship

- **`a9523ef` as is.** A single nested-JSON request, or an MTP request with
  drafts=3 that fills the context, kills the whole process. Its drafts=3
  output is also not the greedy output. Ship the `feat/review4` fixes, or none
  of the MTP server path.
- **Unattended production without a supervisor**, while scheduler errors are
  fatal (finding 5).
- **Any public endpoint without a queue bound and 503** (finding 13).
- **Any claim that "temperature works"** (finding 14), or that batched
  serving is bit-exact at four or more slots (finding 16).
