# Serving eight sessions on one 32 GB card: 125 tok/s, a 16-slot tie, and a cache that could not keep its promise

*Continuous batching beats llama-server by 32% at eight slots and ties it at sixteen, with better tail latency; the hybrid prefix cache is where the release run found — and then fixed — a real defect.*

At eight slots the server returns 125.25 tok/s aggregate against llama-server's 94.75. The median time to first token is 518 ms against 497; the 95th percentile is 526 against 1,377. At sixteen slots the aggregate is a tie: 212.14 against 214.41. Latency is not: p50 779 against 1,237 ms, p95 784 against 2,682.

## The three cells

Same model as [article 01](01-beating-llama-cpp-prefill.md), Qwen3.8-27B Q4_K_M, q8 KV on both engines, greedy, 64 new tokens per request, a 2,423-token shared prompt; 50 requests in a, 48 in b and c.

| cell | slots | engine | requests | errors | TTFT p50 / p95 | aggregate tok/s | per-stream tok/s | peak GiB |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |
| a | 1 | hipml | 50 | 0 | 2,999 / 3,031 | 41.19 | 41.19 | 20.38 |
| a | 1 | llama | 50 | 0 | 2,240 / 2,326 | 31.58 | 31.58 | 16.65 |
| b | 8 | hipml | 48 | 0 | 518 / 526 | 125.25 | 16.92 | 24.36 |
| b | 8 | llama | 48 | 0 | 497 / 1,377 | 94.75 | 12.12 | 21.24 |
| c | 16 | hipml | 48 | 0 | 779 / 784 | 212.14 | 15.01 | 26.28 |
| c | 16 | llama | 48 | 0 | 1,237 / 2,682 | 214.41 | 14.17 | 22.40 |

Cell a is the only one that uses MTP: 41.19 against 31.58, +30.4% from speculation. Cells b and c are batched decode.

## One scheduler, N slots

One process owns the GPU: connection threads parse, tokenise and stream; one scheduler thread issues every kernel, so every launch sees the same stream.

Concurrent requests take a slot each. A decode step is one batched weight pass over all active slots; a request that arrives mid-flight is prefilled in 256-token tiles, then its KV rows and recurrent state copied into the slot. Slots stay dense: the batched kernels use `blockIdx.y` as both row index and physical slot, so a release swaps the last slot's state down rather than leaving a hole.

Admission is backpressure bounded by slots, not threads. A slot costs its KV cache plus recurrent state — hundreds of MiB at 8K — while a queued request costs token ids. The limit is `slots + max_queue`, defaulting to 3× slots; a request past it gets a 503 with `Retry-After: 1` before tokenisation. `--max-conns` bounds idle sockets separately.

Cancellation is checked between tiles and steps; the HTTP layer peeks the socket every 250 ms, so a disconnect frees its slot rather than holding it to `max_tokens`.

Failures have three scopes. A per-job failure — a bad id, an allocation, a speculative step — fails that job, frees its slot and invalidates its prefix entries, leaving the engine serving. A failure in the shared weight pass cannot be pinned to one slot, so the whole batch fails and the engine is reset. A device-level failure drains every stream and exits 1 so a supervisor can restart. An injected slot fault left the other streams token-identical.

## A prefix cache for a hybrid model

The cache captures state at rendered message boundaries, keyed by the exact token prefix: 16 full-attention KV slices plus 48 SSM recurrent states, both required. A KV-only cache is wrong here: the 48 gated-delta-net layers do not reconstruct history from attention rows; their conv and recurrent buffers are the history, compressed. Restoring KV rows with the recurrent state at zero yields a plausible token from a model that never saw the prefix. In b and c a hit restores 2,396 of the 2,423-token shared prompt.

Hits skip MTP because the drafter's KV cannot cover a restored prefix; cells b and c are pure batched decode.

### The cache defect: a hit that was not the hit asked for

With the default 1 GiB cap, an 8,192-token snapshot does not fit: 1,174.6 MiB — 149.6 MiB SSM state across 48 layers, 1,024.0 MiB KV across 16, 0.9 MiB logits. Capture stored nothing and returned no error; the bench printed the second entry as 0.0 MiB and the cache held only the 4,096-token entry.

The exact-repeat request asked for 8,192 tokens, got a 4,096-token hit, and the benchmark used that entry's logits as if they followed the full prompt, decoding from the wrong position with different ids. The pre-fix output is `bench/release/cache/cache_8192_default_cap.txt`. With `--bytes-mib 4096` the same case was already exact on the pre-fix build: a hit at 8,192 tokens, 4.7 ms against 6,786.4 ms cold, 1,448.7×.

The serving path uses restored logits only for a whole-prompt hit and otherwise prefills the remainder, so no client got a wrong answer. It is still a cache-contract failure, now fixed. Capture refuses an entry it cannot store and counts the refusal instead of storing a 0-byte one; lookup verifies the hit length equals the requested prefix before using its logits as end-of-prompt; an oversized snapshot is a miss, not a truncated hit. On the patched build, the default 1 GiB cap refuses the 1,174.6 MiB snapshot and falls back to a full prefill — 6,766.2 ms against 6,691.7 ms cold, token-identical ids — while `--bytes-mib 4096` keeps the 4.6 ms exact hit at 1,460×. `cache-gate` replays the old consumer and requires it to diverge: it does, at id 0, while the fixed fallback matches the uncached ids. A cache that cannot store an entry must not claim it; a caller that needs an exact hit must check the length.

## What it costs, and where it stops

Eight slots are the win: +32.2% aggregate and +39.6% per-stream over llama-server. p50 is 4% behind llama's, p95 2.6× better. Sixteen slots are the tie, −1.1% aggregate, even though p50 is 1.6× and p95 is 3.4× better. llama-server's aggregate grows faster with slots.

Cell a is the mirror image. MTP lifts the aggregate by 30.4%, but cold TTFT is 2,999 ms against 2,240: llama-server prefills the 2,423-token prompt about 25% sooner. The server prefills in 256-token tiles, article 01's matrix in 2,048-token chunks. It is visible in TTFT, not proof of cause.

Peak VRAM is 3.1 to 3.9 GiB higher in every cell: 20.38 against 16.65 at one slot, 24.36 against 21.24 at eight, 26.28 against 22.40 at sixteen.

Bit-exact batching costs throughput: 107.46 tok/s on the exact path against 149.43 on the opt-in WMMA tile at eight slots — about 28%; the server's 8-slot probe measured 30% on an earlier Qwen3.8-27B quantisation (107.1 against 75.3). The server defaults to the tile at four or more slots and advertises `X-Hipml-Batch-Exact: 0`; `HIPML_BATCH_LEGACY=1` forces the exact path.

Three limits are hard. The server is greedy only: `temperature > 0` or `n > 1` is a 400, and `top_p` and `seed` are accepted no-ops. It has no authentication and binds `127.0.0.1` by default: not an internet service. A device-level failure exits the process; the supervisor restarts it as a restart loop, not a health monitor, so a wedged server stays up.

Not measured: 32 slots, another context split, batch depth above zero, or a multi-hour soak. The cells are repeated waves of 64-token replies over a shared prompt: decode throughput and cache-hit latency, not cold multi-tenant prefill.

## Where the numbers come from

- The three cells, requests, errors, TTFT, throughput and peaks: [`../release-run.md`](../release-run.md), "Server, 1 / 8 / 16 slots".
- Raw per-cell server results and commands: [`../../bench/release/server/hipml_b/result.json`](../../bench/release/server/hipml_b/result.json) and the sibling directories [`hipml_a`](../../bench/release/server/hipml_a/result.json), [`hipml_c`](../../bench/release/server/hipml_c/result.json), [`llama_a`](../../bench/release/server/llama_a/result.json), [`llama_b`](../../bench/release/server/llama_b/result.json) and [`llama_c`](../../bench/release/server/llama_c/result.json).
- Continuous batching, slot adoption, admission, cancellation and failure scopes: [`../notes/wp45-51-serve.md`](../notes/wp45-51-serve.md).
- The per-slot fault test and the server review: [`../notes/review4-server-audit.md`](../notes/review4-server-audit.md).
- The cache snapshot layout, the default-cap failure it used to have, the fix and the 4 GiB control: [`../../bench/release/cache/cache_8192_default_cap.txt`](../../bench/release/cache/cache_8192_default_cap.txt) (pre-fix), [`../../bench/release/cache/cache_8192_default_cap_fixed.txt`](../../bench/release/cache/cache_8192_default_cap_fixed.txt), [`../../bench/release/cache/cache_8192_4gib_fixed.txt`](../../bench/release/cache/cache_8192_4gib_fixed.txt) and [`../../bench/release/cache/cache_gate.txt`](../../bench/release/cache/cache_gate.txt).
- The cache's insert bound and the capture/restore code: [`../../crates/hipml-core/src/prefix_cache.rs`](../../crates/hipml-core/src/prefix_cache.rs) and [`../../crates/hipml-model/src/gpu/prefix.rs`](../../crates/hipml-model/src/gpu/prefix.rs).
- The exact/skinny batch cost: [`../../bench/release/batch/exact_throughput.txt`](../../bench/release/batch/exact_throughput.txt).
- Greedy-only, no-auth and loopback defaults: [`../server.md`](../server.md).
