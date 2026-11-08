# The GPU graph: one token end to end

WP13 wired the pieces from WP3–WP14 into one decode graph and measured it; WP14
closed the host-side overhead and wired the tiled prefill GEMM. The CPU
reference from `feat/graph` stays the oracle: `--cpu` runs it, `--device` runs
the graph below, and both print the same ids at short context.

## Shape of the decode graph

```text
x = embed(token)                              embed_row_q4_k: device row dequant
for layer 0..63:
    h  = rmsnorm(x, attn_norm)                rmsnorm_f32
    b  = full_attention(h) | gated_delta_net(h)
    x1 = x + b                                residual_add_f32
    t  = rmsnorm(x1, post_attention_norm)     rmsnorm_f32
    g  = W_gate t,  u = W_up t                gemv_q4_k / gemv_q6_k / gemv_q8_0
    a  = silu(g) * u                          silu_mul_f32
    x  = x1 + W_down a                        gemv_q4_k / _q6_k / _q5_k
logits = W_output rmsnorm(x, output_norm)    gemv_q6_k
id = argmax(logits)                           argmax_f32: two launches, one u32 back
```

Layers 3, 7, …, 63 are full attention; the other 48 are gated delta net. That
dispatch is `Config::is_recr`, the same mask the CPU reference reads.

Every weight matvec streams the loader's resident `DeviceTensor` bytes through
the quantized GEMV, one entry per block format: Q4_K, Q6_K, Q8_0 and Q5_K.
Nothing is dequantized on the host and no tensor is uploaded twice. `blk.51/55.
ffn_down` are the two Q5_K tensors; they used to be dequantized once into f32
(712 MB a token instead of 123 MB) and WP16 gave them `gemv_q5_k`.

Launches are chained on one stream and synchronized once per token, before the
sampled id is read back. The two host steps WP13 shipped — the embedding row and
the 248,320-way argmax — are gone:

| step | before | now |
| --- | --- | --- |
| embedding | `dequant_q4_k_f32` + blocking copy + host row write | `dequant_q4_k_f32` straight into `x` (no copy) |
| argmax | 993 KB copy back + host walk | `argmax_f32` (64 partial blocks + fold), 4 bytes back |
| KV append (q8/q4) | read k/v back, host `kv::quantize`, 8 uploads | `kv_encode_q8` / `kv_encode_q4`, one launch |

### Full attention

`AttnLayer` borrows the four resident projections (`new_resident`) instead of
uploading host `Mat`s. Its f32 KV path (mrope → head-RMS → q-gate → kv-append →
split-K attention) is the short-context path and the one the token check uses.

### Quantized KV at depth

`--kv q8` (default for `gpu-bench --depth`) swaps the cache for the `feat/kv-100k`
format: `[n_kv][seq][head_dim/32]` q8_0 blocks, K and V separate. The layer
computes q/k/v on device, `kv_encode_q8` quantizes the 2048-wide row straight
into the cache at `pos` (one launch, no host round-trip), and the decode pair
(`attn_decode_kv_q8` + reduce) runs over the split-K grid. q4_0 is selectable
and halves the cache again.

The encoder is bit-exact against `hipml_core::kv::quantize`: `hipml parity
kv_encode` compares the decoded bytes for both formats and reports
`max_abs 0.00e0`.

At 100K the q8 cache is **3.243 GiB (3.48 GB)** for the 16 attention layers,
against 13.1 GiB for the f32 path — which is why the f32 path is not used there.

### Gated delta net

`SsmGpuLayer::new_resident` borrows the five resident projections (qkv, gate,
alpha, beta, out) and keeps the small f32 vectors (conv taps, dt bias, A, norm)
as device buffers. The kernel chain is WP11's: gates → conv1d+silu → L2 →
gated delta recurrence → gated RMS. State is 156.9 MB per sequence and is not
prefix-cacheable, so prefill and decode cost the same per token.

## Chunked prefill

`--chunk N` (and `gpu-bench --prefill N`, which defaults to 256) runs the prompt
as tiles of N tokens instead of N decode steps. Per tile:

```text
xc  = embed(tokens)                     N row dequants into [N][n_embd]
per layer:
  tc   = rmsnorm_rows(xc, attn_norm)    one block per row
  block: projections as prefill GEMMs   prefill_q4_k / _q6_k / _q8_0
         + the sequential part per token (conv, GDN, rope, KV append, attention)
  x1c  = xc + block;  tc = rmsnorm_rows(x1c, post_attention_norm)
  act  = silu(gemm(gate, tc)) * gemm(up, tc)
  xc   = x1c + gemm(down, act)
logits = lm_head(rmsnorm(last row))
```

A tile streams the 15.89 GiB of weights **once** instead of once per token; the
gated-delta-net recurrence, the conv window, rope, the cache append and causal
attention stay per token because they are genuinely sequential in time. The
projections use the tiled GEMM, which accumulates a 32-value k-tile at a time
instead of the GEMV's 2,048-value shared tile, so a tile can differ from the
token loop in the last bits — measured: the 20 ids are identical, and the token
loop remains the default (`--chunk` opts in) and the parity reference.

## Commands

```powershell
# correctness: same ids as the CPU oracle, plus first-step logit distance
hipml generate --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf `
  --ids 760,6511,314,9338,369 --tokens 20 --device
hipml generate ... --device --compare-cpu        # loads the CPU oracle too
hipml generate ... --device --chunk 256          # chunked prompt tile

# the oracle itself, no GPU build needed
hipml generate ... --cpu

# speed: 512-token prefill through the chunked path (and through the token loop)
hipml gpu-bench --gguf ... --prefill 512
hipml gpu-bench --gguf ... --prefill 8192
# speed: decode at depth, caches pre-filled with valid blocks
hipml gpu-bench --gguf ... --depth 20000  --kv q8 --tokens 8
hipml gpu-bench --gguf ... --depth 100000 --kv q8 --tokens 8
# where the non-weight time goes
hipml gpu-profile --gguf ... --reps 4000
hipml gpu-profile --gguf ... --trace        # per-stage GPU time and GB/s
# launch cost vs per-byte cost, and the A/Bs
hipml gpu-sweep --gguf ... --gemv           # real GEMV spans, per dtype
hipml gpu-sweep --gguf ... --pair           # two-stream overlap
hipml gpu-sweep --gguf ... --variant        # 1-row vs 2-row warps, bit-exact
hipml bench-kv --depths 100000 --splits 128 # KV split count
```

## Measured, AMD Radeon AI PRO R9700 (gfx1201)

Resident load: **17,192,421,376 B in 851 tensors** (16.01 GiB), upload 5.3 s.
Graph streamed per token: **17,065,064,448 B (15.89 GiB)** — every layer's
projections plus lm_head; the remainder of the resident set is the embedding
table (one row is touched per token) and the f32 norm vectors.

### Acceptance 1 — ids

Prompt `760,6511,314,9338,369`, 20 greedy tokens:

| source | ids |
| --- | --- |
| device (GPU graph) | `11751 13 198 760 6511 314 14898 369 21047 13 198 760 6511 314 17163 369 23327 13 198 760` |
| device, chunked prefill | identical |
| CPU reference | identical |
| llama.cpp | identical (`docs/forward-parity.md`) |

First-step logits against the CPU oracle: **max |d| = 4.77e-5** at index 233546;
device and CPU argmax both 11751.

### Acceptance 2 — speed

| measurement | WP13 | WP14 | WP16 | WP21 |
| --- | --- | --- | --- | --- |
| short-context decode, tokens 2..20 | 27.9 tok/s (35.9 ms) | 28.0 tok/s (35.7 ms) | **29.9 tok/s (33.5 ms)** | 30.0 tok/s (33.4 ms) |
| 512-token prefill | 22.99 s, 22.27 tok/s (token loop) | **5.35 s, 95.8 tok/s** (chunk 256) | 5.35 s |
| 8,192-token prefill | — | **89.1 s, 91.9 tok/s** | unchanged |
| decode at depth 20,000, q8 KV | 21.21 tok/s (47.1 ms) | 26.5 tok/s (37.7 ms) | **28.0 tok/s (35.7 ms)** |
| decode at depth 100,000, q8 KV | 17.58 tok/s (56.9 ms) | 21.0 tok/s (47.7 ms) | 21.8 tok/s (45.9 ms) | 21.5 tok/s (46.5 ms) |

WP16 measured every stage of a token with HIP events (`gpu-profile --trace`)
instead of attributing the remainder to "DRAM ramp". It found that the gap is
*inside* the kernels, not between launches, and that it scales with how much
parallelism each kernel can put in flight — so the two changes that moved the
number were a wider grid (one row per warp) and 6x fewer bytes for the Q5_K
fallback. Two streaks did not pay and are recorded below so they are not retried.

llama.cpp HIP on the same box: 31.4 tok/s short context, 21.86 tok/s at a
97,387-token prompt.

### Where a short-context token's 33.5 ms goes

WP14 attributed the remainder to "DRAM ramp" from a launch-loop measurement.
WP16 timed the graph itself instead: `gpu-profile --trace` records a HIP event at
every stage boundary, synchronizes once, and folds the spans (plus the streamed
bytes each stage owns). Summed over four tokens:

| stage | ms/token | weight GB | GB/s |
| --- | ---: | ---: | ---: |
| ffn_gate (Q4_K) | 5.61 | 3.23 | 576 |
| ffn_up (Q6_K) | 6.62 | 3.62 | 547 |
| silu + ffn_down (Q4_K + 2x Q5_K) | 6.62 | 3.30 | 498 |
| GDN input projections (qkv, gate, alpha, beta) | 5.69 | 2.65 | 466 |
| GDN output projection | 3.46 | 1.33 | 385 |
| 16 full-attention blocks | 2.97 | 1.30 | 436 |
| lm_head (Q6_K) | 1.68 | 1.04 | 622 |
| GDN recurrent chain (7 kernels x 48 layers) | 1.15 | — | — |
| residual/norm/embed/argmax | 0.25 | — | — |
| **total** | **34.06** | **16.47** | **504** |

(The trace's own event recording costs about 0.5 ms a token, which is why the
sum is 34.06 and `generate` measures 33.5.)

What that says, and what it rules out:

- **The loss is inside the kernels.** In the same 399-tensor stream, the sum of
  event-measured kernel spans is 33.4 ms of a 34.6 ms wall — 1.26 ms of idle
  between launches (3 us each), not the 6.4 ms an empty-kernel loop predicted.
  A pure stream-read over the same bytes shows the same shape: 477 GB/s with one
  launch per tensor, **636 GB/s** with the same bytes in one launch, and only
  1.26 ms of inter-launch idle. So there is nothing to win by submitting faster;
  the ramp is the first and last few microseconds of each kernel's own memory
  pipeline, and it is proportional to how much work each thread has queued.
- **Concurrency loses.** Two independent weight streams on two HIP streams is
  0.74x (`gpu-sweep --pair`: 500 GB/s sequential, 370 GB/s overlapped). Their
  access patterns evict each other's DRAM pages. This kills both the
  prefetch-stream and two-stream-pipelining ideas on this card.
- **Parallelism helps, up to a point.** ffn_down is 5120 rows = 320 blocks at 16
  rows/block, and it streamed at 498 GB/s while lm_head (248,320 rows = 15,520
  blocks) streamed at 622. One row per warp doubles the grid with identical
  arithmetic: down 498 -> 565, gate 566 -> 585, up 570 -> 581. That is the whole
  of the row-grain change.
- **Bytes are the other lever.** The two Q5_K `ffn_down` tensors were served
  from an f32 copy: 712 MB a token at ~350 GB/s, the slowest bytes in the
  stream. `gemv_q5_k` reads the blocks directly, so the same stage now moves
  3.30 GB instead of 3.89 GB and takes 6.62 ms instead of 7.51 (498 GB/s over
  the smaller byte count; -0.9 ms).

Roofline check: at lm_head's 622 GB/s every weight stage would take
16.47/0.622 = 26.5 ms, plus 1.15 ms of recurrence and 0.25 ms of elementwise =
**27.9 ms, 35.8 tok/s**. The 33.5 ms actually measured is that ceiling plus the
per-kernel ramp on fifteen separate weight streams.

At depth the same base applies, plus the KV stream:

| depth | KV bytes/token | at 632 GB/s | measured attention | total |
| --- | ---: | ---: | ---: | ---: |
| 20,000 | 0.696 GB | 1.1 ms | 2.2 ms | 35.7 ms |
| 100,000 | 3.48 GB | 5.5 ms | 12.5 ms | 45.9 ms |

The quantized-KV attention kernel runs at 282 GB/s and is not block-limited:
`bench-kv --splits` measures 14.3 ms at 32 splits, 12.34 at 64, 12.98 at 128,
12.49 at 256 and 12.58 at 512 for the same 3.24 GiB. It is the per-row work
(stage to shared, one dot per query head, one rescale per tile) that sets its
pace, so the remaining 12 ms at 100K needs a different kernel, not a wider grid.

### Driver-level submission (WP21)

The previous section attributed the 33.5 ms to "per-kernel memory ramp across
fifteen separate weight streams" and implied that submitting faster would
reclaim it. WP21 tested that directly, with the runtime rather than the kernels:
stream capture into a HIP graph, one kernel over a device-side job list, a
second stream, split-K, and cache hints on the weight loads.

The capture experiment needs one thing from the graph before it is reusable: the
step scalars. `pos`, `seq`, the token id and the four M-RoPE positions moved into
a 32-byte device buffer (`Launcher::step`), and `mrope_f32`, `kv_append_f32`,
`attn_decode_f32`, `attn_decode_split_f32`, `attn_decode_kv` and `kv_encode` read
them there. The split count is fixed per layer from `n_ctx` instead of per step,
because the grid is part of a captured graph. With that, one graph decodes every
step: `hipml generate --device --graph` prints the same twenty ids as the eager
path, and `gpu-bench --graph` the same argmax, at depth 0 and at 100K.

| approach | effective GB/s | ms/token | tok/s | submit |
| --- | ---: | ---: | ---: | ---: |
| eager, 1,299 launches | 494 | 33.37 | 29.97 | 2.35 ms of launch API (1.81 us/launch) |
| capture + `hipGraphLaunch` | 494 | 33.39 | 29.95 | 1 call |
| one kernel over an ffn gate+up job list | 562 vs 568 | — | — | 64 vs 128 launches |
| split-K x2 / x4 / x8 on ffn_down | 546 / 551 / 531 vs 559 | — | — | 120 vs 60 launches |
| sc1 streaming loads (`nontemporal`) | 491 vs 559 | — | — | — |
| two weight streams (`gpu-sweep --pair`, WP16) | 370 vs 500 | — | — | — |
| `hipMemAdvise` / `hipMemPrefetchAsync` | not applicable | | | |

**Submission is not on the critical path.** 1,299 launches at 1.81 us are 2.35 ms
of API time against 33.3 ms of GPU work. In the real loop the host cannot run
ahead anyway: it needs the sampled id back before it can issue the next token, so
it only ever has one step queued, the command queue fills, and the driver
throttles the issuing thread to the device's own pace (`gpu-bench --graph-probe`
sees 31.3 ms of blocked host per token, graph or not). A graph launch therefore
changes the wall clock by nothing (0.99-1.00x), and the 1.6 ms of "idle between
launches" the event pass reports is GPU-side drain at each kernel boundary, which
capture does not remove.

**The fifteen weight streams are not paying for being fifteen.** The same 6.85 GB
of ffn gate+up bytes stream at 568 GB/s as 128 launches and 562 GB/s as 64
launches of one job-list kernel — and 561 GB/s as a single launch over all 96
jobs, which is not schedulable but bounds the idea. Splitting k to widen the
grid (546/551/531 GB/s at 2/4/8 splits, plus an ordered reduce) and marking the
payload loads streaming (491 vs 559, sc1 costs 12% — the L2 was helping) are both
worse. The equivalent change that did pay earlier was row-grain, not launch-grain.

That leaves the honest attribution of a 33.3 ms short-context token:

| component | ms | note |
| --- | ---: | --- |
| weight roofline at lm_head's 622 GB/s | 26.5 | 16.47 GB |
| the other fourteen shapes at 556 GB/s average | +3.1 | 493-577 GB/s per share, insensitive to launch count, grid shape, split-K, load width and cache policy |
| GDN recurrent chain + elementwise + attention compute | +1.7 | 1.18 + 0.25 + ~0.3 |
| kernel-boundary drain that survives capture | +2.0 | residual |
| **measured** | **33.4** | 30.0 tok/s |

At depth the same 33.3 ms is the base: 46.5 = 33.3 + 5.5 KV roofline + 6.9
attention excess + 0.8. Run-to-run scatter on the decode numbers is ±0.2%.

Instruments, all on the resident weights: `gpu-sweep --merge` (job-list kernel,
three launch shapes over the same ffn bytes, with the merged rows checked against
the standalone entries), `gpu-sweep --split` (r1 vs sc1 vs 1/2/4/8 k-splits, with
the relative difference against the unsplit kernel), `gpu-sweep --pair` (two
streams) and `gpu-bench --graph-probe` (eager vs captured step).

Two bugs fell out of the capture work and are fixed: the quantized-KV decode
passed the position count where the cache stride belongs, so every KV head but
the first read rows at the wrong offset whenever `n_ctx > pos + 1` (invisible to
`parity --deep`, which uses `seq == stride`); and a capture warm-up token must
not run, because it appends to every layer's cache at the current position.

### Prefill

| tokens | chunk | wall | tok/s | ms/token |
| --- | ---: | ---: | ---: | ---: |
| 512 | 128 | 6.92 s | 74.0 | 13.5 |
| 512 | 256 | 5.34 s | **95.9** | 10.4 |
| 8,192 | 256 | 89.1 s | **91.9** | 10.9 |

The per-token part of a chunk is what dominates now: each token still launches
48 × 7 gated-delta-net kernels and 16 × 8 attention kernels, about 464 launches
a token, and those small kernels are ~2.4 s of the 2.7 s a 256-token chunk takes.

## What runs where

| stage | device | host |
| --- | --- | --- |
| embedding row dequant | `dequant_q4_k_f32` into `x` | — |
| norms, residual, SwiGLU | kernels | — |
| all weight matvecs | quantized GEMV over resident bytes | — |
| all weight GEMMs (prefill) | tiled prefill GEMM over resident bytes | — |
| full attention (short) | mrope/head-norm/q-gate/append/split-K | — |
| full attention (deep) | `kv_encode_q8` + quantized-KV decode pair | — |
| gated delta net | conv/L2/recurrence/gated-RMS | — |
| lm_head | resident Q6_K GEMV | — |
| sampling | `argmax_f32` | — |

Nothing on the decode path round-trips to the host except the 4-byte id.

## Unverified / next

- Two-stream overlap, a prefetch stream, and splitting more than 64 KV splits
  were measured and rejected (see above); they are recorded here so they are
  not re-attempted.
- Deep-path **token** parity is not asserted: q8 KV changes the attention values
  by design. `hipml parity --deep` verifies the kernel itself against the
  reference at `s=100000`; `hipml parity kv_encode` verifies the encoder is
  bit-exact. The graph's short-context tokens verify the rest.
- Chunked prefill requires `--kv f32`; the quantized caches are the decode path.
- The 8,192-token prefill is a real prefill; 100K prefill is not measured
  (extrapolated: 100K × 10.9 ms ≈ 18 minutes at chunk 256).
- The 1-row-per-warp entries are only used by the decode path; `bench` and the
  parity cases still exercise the 2-row entry, and `gpu-sweep --variant`
  checks that the two write identical bytes.
- No fp16 KV path.
- **Not launch-bound, not graph-bound, not grid-bound.** Measured in WP21 and
  left here so they are not re-attempted: stream capture (0.998x), a job-list
  kernel over the independent projections (561 vs 568 GB/s), split-K to widen
  the grid (546-551 vs 559), sc1/streaming weight loads (491 vs 559), two
  streams (0.74x, WP16). Reusable capture does work and is kept behind
  `--graph`; it is worth 0.06 ms, not 3 ms.
- **Next step, in order of size:** (1) the quantized-KV attention kernel at
  282 GB/s of 632, worth ~5 ms at 100K — it needs the stage/dot/rescale work
  reorganized (more than one query head's worth of rows per staged tile, or
  f32 scores over a shorter row), not more blocks; (2) the GDN output
  projection at 385 GB/s and the GDN input projections at 466 — both are Q8_0,
  whose per-row kernel has 5.2 KB per block and a two-stage reduction, so it is
  the last format without a row-tile entry and worth ~1 ms of the 3.1;
  (3) chunk variants of the conv/GDN kernels so a prefill tile runs one kernel
  per layer instead of 464 launches (the 10.4 ms/token prefill floor).
