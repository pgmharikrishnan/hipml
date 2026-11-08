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

The prompt is a tile of `--chunk N` tokens (512 by default) instead of N decode
steps. Per tile:

```text
xc  = embed(tokens)                     N row dequants into [N][n_embd]
per layer:
  tc   = rmsnorm_rows(xc, attn_norm)    one block per row
  block: projections as prefill GEMMs   prefill_q4_k_h / _q6_k_h / _q8_0_h
         + one GDN chain for the tile (6 launches)
         or one attention chain for the tile (4 launches + output GEMM)
  x1c  = xc + block;  tc = rmsnorm_rows(x1c, post_attention_norm)
  act  = silu(gemm(gate, tc)) * gemm(up, tc)
  xc   = x1c + gemm(down, act)
logits = lm_head(rmsnorm(last row))
```

A tile streams the 15.89 GiB of weights **once** instead of once per token. The
projections use the fp16 tiled GEMM (half in shared, `v_dot2_f32_f16`, f32
accumulate); `HIPML_PREFILL_F32=1` selects the f32 kernels. The GDN recurrence
and causal attention are still sequential in time, but they are now one launch
per stage per tile rather than one per token: the conv window is recomputed in
parallel from the tile's rows, the delta-net state stays in registers, and the
attention kernel is a matrix-unit flash kernel: one block per (KV head, 32
queries, half of the head width), two warps per query head (16 queries each),
K staged in full and V staged transposed, `wmma_f32_16x16x16_f16` for both
QK^T and PV. `HIPML_ATTN_F32=1` selects the old f32 tiled kernel for A/B, and
`HIPML_TOKEN_STEPS=1` keeps the per-token chain.

Numerics: fp16 operands and a 32-value k-tile accumulation mean a tile is not
bit-identical to the token loop. Measured: the 20 acceptance ids are identical;
the first-step logits and the kernel-level fp16 budgets are in
`docs/benchmarks.md`.

## Commands

```powershell
# correctness: same ids as the CPU oracle, plus first-step logit distance
hipml generate --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf `
  --ids 760,6511,314,9338,369 --tokens 20 --device
hipml generate ... --device --compare-cpu        # loads the CPU oracle too
hipml generate ... --device                     # chunked prefill (chunk 512)
hipml generate ... --device --chunk 256          # explicit tile
hipml generate ... --device --chunk 0            # token-loop prefill

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

| measurement | WP13 | WP14 | WP16 | WP19 | WP20 | WP21 |
| --- | --- | --- | --- | --- | --- | --- |
| short-context decode, tokens 2..20 | 27.9 tok/s (35.9 ms) | 28.0 tok/s (35.7 ms) | 29.9 tok/s (33.5 ms) | 28.35 tok/s (35.28 ms) | 28.3 tok/s | **30.0 tok/s (33.4 ms)** |
| 512-token prefill | 22.99 s, 22.27 tok/s (token loop) | 5.35 s, 95.8 tok/s (chunk 256) | 5.35 s | 0.50 s, 995–1021 tok/s | 0.40 s, 1266 tok/s* | — |
| 8,192-token prefill | — | 89.1 s, 91.9 tok/s | 9.93 s, 825.1 tok/s | 11.41 s, 717.7 tok/s | 6.82 s, 1201.0 tok/s* | — |
| decode at depth 20,000, q8 KV | 21.21 tok/s (47.1 ms) | 26.5 tok/s (37.7 ms) | **28.0 tok/s (35.7 ms)** | unchanged | unchanged | — |
| decode at depth 100,000, q8 KV | 17.58 tok/s (56.9 ms) | 21.0 tok/s (47.7 ms) | 21.8 tok/s (45.9 ms) | unchanged | 21.1 tok/s (47.5 ms) | 21.5 tok/s (46.5 ms) |
| decode at depth 32,768, q8 / q4 KV | — | — | — | — | 25.4 / 25.0 tok/s | — |

\* WP20 prefill used the WP17 prefill kernel, which skipped half of every GEMM
(see `docs/benchmarks.md`); WP19 is the same kernel with the accumulator index
fixed. The merged tree uses the WP19 prefill GEMM with the WP20 matrix-unit
attention kernel, so its prefill is measured fresh in `bench/RESULTS.md`.

llama.cpp HIP on the same box: 31.4 tok/s short context, 21.86 tok/s at a
97,387-token prompt, pp512 **1256.5 tok/s**, 97K prefill **844.9 tok/s**.

The 20-id acceptance string is unchanged by WP15 (fp16 projections, chunked
recurrence and chunked attention) and by WP19 (the accumulator-index fix):
`11751 13 198 760 6511 314 14898 369 21047 13 198 760 6511 314 17163 369 23327
13 198 760`. That prompt is five tokens, which is why it never exercised the
WP19 bug; the 512-token bench prompt's last id does — it is back to **489**, the
pre-WP17 value (the buggy kernel read 122271).

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
| 512 | 128 | 5.18 s | 98.9 | 10.1 |
| 512 | 256 | 1.43 s | 357.1 | 2.80 |
| 512 | 512 | **0.50–0.51 s** | **995–1021** | 1.01 |
| 8,192 | 512 | **11.41 s** | **717.7** | 1.39 |

The 512 and 8,192 rows are WP19: the same kernel as WP17 with the accumulator
index fixed (WP17 skipped half the MACs, see `docs/benchmarks.md`). WP17 reported
0.44 s / 1153.8 and 9.93 s / 825.1 for a kernel that produced zeros for the
second 16 tokens of every 32-token warp span. WP20 replaced the attention stage
in the 8K row (3.58 s → 0.78 s) and reported 6.82 s / 1201 tok/s; the merged
tree has both changes and is re-measured in `bench/RESULTS.md`.

The chunk graph is the device default (`--chunk N`, 512 unless overridden;
`--chunk 0` and `--compare-cpu` keep the token loop). Four changes from WP14,
the fourth being WP17:

1. **Matrix-unit projections (WP17, corrected in WP19).** The prefill GEMM runs
   `wmma_f32_16x16x16_f16` on the gfx12 matrix unit — 256 tokens x 128 rows per
   block, 512 threads, eight 16x16x16 f32 accumulators per warp, weights
   dequantised into shared as half, f32 epilogue written straight to `y`. The
   ffn shapes measure 65–80 TFLOPS (68–83% of the quoted 95.7, 35–43% of the
   175 TFLOPS the instruction itself measures in a register-only probe), against
   22.9 for the dot2 path. `--prefill-dot2` / `HIPML_PREFILL_DOT2` and
   `--prefill-f32` / `HIPML_PREFILL_F32` are the no-rebuild A/B switches; the
   fragment layouts are measured by `scripts/probe_wmma_layout.cpp`.
2. **fp16 dot2 projections (WP15, previous path).** The tiled GEMM holds half in
   shared and uses `v_dot2_f32_f16` with f32 accumulators: 21–27 TFLOPS against
   15–17 for the f32 kernel, i.e. 44–57% of the quoted fp32 peak and 66% of the
   41.4 TFLOPS VALU ceiling measured for this instruction mix.
3. **Chunked GDN.** `steps_chunk` runs gates, conv, L2, the delta recurrence
   and the gated RMS once per layer per tile. The recurrence keeps each head's
   128×128 state in registers across the tile, so the state is read once and
   written once instead of per token.
4. **Chunked attention.** q-gate split, norm+rope, KV append and causal
   attention are four tile-wide launches; the attention kernel is 64 queries
   per block (4 per warp) with shuffle reductions, and hides future rows with
   the causal loop bound instead of a branch.

Launch count per 512-token prompt:

| stage | WP14 (per token) | WP15 (per tile) |
| --- | ---: | ---: |
| GDN recurrent chain | 48 × 7 × 512 = 172,032 | 48 × 6 = 288 |
| attention chain | 16 × 8 × 512 = 65,536 | 16 × 6 = 96 |
| everything else | ~1 launch/token/op | count in `PrefillReport` |
| **total** | **≈ 237,600 (464/token)** | **≈ 1,700 (3.3/token)** |

Where a 512-token, chunk-512 prompt's 0.51 s goes (`gpu-bench --prefill-split`):

| stage | ms | note |
| --- | ---: | --- |
| ffn GEMMs + SwiGLU | 250.6 | 17.5 TFLOP at ~70 TFLOPS |
| ssm GEMMs | 56.2 | qkv/gate/out, 5.7 TFLOP |
| attention GEMMs | 20.5 | q/k/v/o |
| elementwise | 12.5 | norms, residuals |
| sequential GDN | 119.4 | 48 layers × 6 launches, state in registers |
| sequential attention | 40.8 | 16 layers × 4 launches + output |
| embed + final head | 2.7 | embedding rows, lm_head on the last row |

At 8K the same breakdown is attention 3.78 s, ffn 4.09 s, GDN 1.97 s, ssm GEMMs
0.93 s, attention GEMMs 0.34 s, elementwise 0.25 s. The GEMM terms moved 3.6×;
the sequential ones did not, and attention is now the largest single term at 33%.

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
  (extrapolated from the 8K rate and the T² attention term, not timed).
- Q5_K still has no GEMV kernel: two decode tensors take the f32 GEMV. Prefill
  uses a half copy made at load.
- No fp16 KV path.
- **fp16 numerics.** The projections quantise x and w to fp16 and accumulate in
  f32. The 20 greedy ids above are identical, but this is a numerical change,
  not a bit-exact one: `hipml parity` compares the fp16 kernels at ~1e-3
  relative, and the synthetic 512-token bench prompt samples a different first
  id than the all-f32 path did (489 vs 220). The real-prompt ids are the
  acceptance check; the bench prompt is not.
- **Attention was the 8K limit; the matrix unit took it (WP20).** The f32 tile
  kernel reduced every (query, key) score with a five-step 32-lane shuffle chain
  and carried f32: 3.58 s of 9.93 s at 3.7 TFLOPS, latency-bound rather than
  FLOP-bound. `attn_chunk_wmma_f16` moves QK^T and PV to
  `wmma_f32_16x16x16_f16` and leaves one `xor 16` per 16x16 score tile: the
  attention stage is now **779 ms (16.9 TFLOPS)**. Two register-pressure facts
  shaped the kernel and are worth keeping: a warp holding both 16 Q fragments
  and 16 output fragments needs ~240 VGPRs and the compiler spills the output
  fragments to scratch *inside the key loop*, and the sparse-tile role split
  does not help because both live ranges span the loop. Splitting the head width
  across `blockIdx.z` (8 output fragments per warp) is what removed the spill.
- **The GEMM's remaining gap** (WP19) is **not** the dequant (removing it
  entirely is worth 2%), LDS fragment traffic (worth 0%) or occupancy (the
  register-only probe hits 174–176 TFLOPS at 4–32 warps per block). It is that
  the k loop is two serialized phases: staging 42% of the time (global loads +
  dequant + shared stores) and the multiply 58%, with two `__syncthreads` per
  k-tile. Pure matrix work with the same structure measures 131.5 TFLOPS; the
  175 TFLOPS instruction rate needs staging loads in flight one tile ahead,
  which costs 20–40 VGPRs the 512-thread, 8-accumulator shape does not have.
  Grid tails are the remaining 5–19%: at 512 tokens `ffn_gate` launches 80
  blocks on 64 CUs.
- The 1-row-per-warp entries are only used by the decode path; `bench` and the
  parity cases still exercise the 2-row entry, and `gpu-sweep --variant`
  checks that the two write identical bytes.
- No fp16 KV path.
- **fp16 numerics.** The projections quantise x and w to fp16 and accumulate in
  f32. The 20 greedy ids are identical, but this is a numerical change, not a
  bit-exact one: `hipml parity` compares the fp16 kernels at ~1e-3 relative.
  The real-prompt ids are the acceptance check.
- **Not launch-bound, not graph-bound, not grid-bound.** Measured in WP21 and
  left here so they are not re-attempted: stream capture (0.998x), a job-list
  kernel over the independent projections (561 vs 568 GB/s), split-K to widen
  the grid (546-551 vs 559), sc1/streaming weight loads (491 vs 559), two
  streams (0.74x, WP16). Reusable capture does work and is kept behind
  `--graph`; it is worth 0.06 ms, not 3 ms.
- **Next step, in order of size:** (1) the quantized-KV attention kernel at
  282 GB/s of 632, worth ~5 ms at 100K; (2) the GDN output projection at
  385 GB/s and the GDN input projections at 466, both Q8_0, worth ~1 ms of the
  3.1; (3) a 6-accumulator / 384-thread prefill GEMM shape that frees 16 VGPRs
  for the register-prefetch pipeline; (4) chunk variants of the conv/GDN
  kernels so a prefill tile runs one kernel per layer instead of 464 launches.
