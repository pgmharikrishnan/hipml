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
    x  = x1 + W_down a                        gemv (Q5_K rows: f32 fallback)
logits = W_output rmsnorm(x, output_norm)    gemv_q6_k
id = argmax(logits)                           argmax_f32: two launches, one u32 back
```

Layers 3, 7, …, 63 are full attention; the other 48 are gated delta net. That
dispatch is `Config::is_recr`, the same mask the CPU reference reads.

Every weight matvec streams the loader's resident `DeviceTensor` bytes through
the quantized GEMV. Nothing is dequantized on the host and no tensor is uploaded
twice. The only exception is Q5_K: Qwen3.8-27B stores `blk.51.ffn_down` and
`blk.55.ffn_down` as Q5_K, which has a dequant reference but no GEMV kernel, so
those two matrices are dequantized once at graph build into f32 and served by
`gemv_f32` (about 0.5 GiB extra VRAM).

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
attention kernel tiles 32 queries per block. `HIPML_TOKEN_STEPS=1` keeps the
per-token chain as an A/B reference.

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

| measurement | WP13 | WP14 | WP15 |
| --- | --- | --- | --- |
| short-context decode, tokens 2..20 | 27.9 tok/s (35.9 ms) | 28.0 tok/s (35.7 ms) | **28.3 tok/s (35.3 ms)** |
| 512-token prefill | 22.99 s, 22.27 tok/s (token loop) | 5.35 s, 95.8 tok/s (chunk 256) | **0.45 s, 1144.7 tok/s** (chunk 512, matrix unit) |
| 8,192-token prefill | — | 89.1 s, 91.9 tok/s | **10.67 s, 767.7 tok/s** (chunk 512, matrix unit) |
| decode at depth 20,000, q8 KV | 21.21 tok/s (47.1 ms) | 26.5 tok/s (37.7 ms) | unchanged |
| decode at depth 100,000, q8 KV | 17.58 tok/s (56.9 ms) | 21.0 tok/s (47.7 ms) | unchanged |

llama.cpp HIP on the same box: 31.4 tok/s short context, 21.86 tok/s at a
97,387-token prompt, pp512 **1256.5 tok/s**, 97K prefill **844.9 tok/s**.

The 20-id acceptance string is unchanged by WP15 (fp16 projections, chunked
recurrence and chunked attention): `11751 13 198 760 6511 314 14898 369 21047
13 198 760 6511 314 17163 369 23327 13 198 760`.

### Where a short-context token's 35.7 ms goes

Measured with `hipml gpu-profile` (4,000-rep launch loop, 274-tensor stream test,
and the graph itself):

| component | ms | note |
| --- | ---: | --- |
| weight bytes at the card's 632 GB/s | 27.00 | 15.89 GiB, the roofline |
| launch submit | 2.06 | 1,299 launches at 1.59 us (handle cached) |
| host steps | 0.20 | embedding 0.15, id readback < 0.05 |
| DRAM utilisation loss | 6.4 | 1,299 short dependent kernels, not one sweep |

The last row is the whole remaining gap and it is measured, not inferred: the
same resident tensors streamed back to back, one launch each, reach **411 GB/s**
(11.92 GB in 28.998 ms over 274 GEMVs), while the same kernels repeated over a
resident 64 MiB chunk reach 605 GB/s (`decode-bench --reps 11`). Each kernel's
ramp and tail costs a few microseconds and there are 1,299 of them. Note also
that the entry-point lookup is free now (0.010 us, cached in `Module`) — an
earlier 4.0 us/launch reading was the first-launch code-object residency, not
the lookup, which is why caching handles bought 0.1 ms and not 3 ms.

At depth the same 35.7 ms base applies, plus the KV stream:

| depth | KV bytes/token | at 632 GB/s | measured attention | total |
| --- | ---: | ---: | ---: | ---: |
| 20,000 | 0.696 GB | 1.1 ms | 2.1 ms | 37.7 ms |
| 100,000 | 3.48 GB | 5.5 ms | 12.1 ms | 47.7 ms |

So at 100K the quantized-KV attention kernel runs at ~280 GB/s of a 632 GB/s
roofline; that, not the host, is the remaining depth gap (the 16 host round-trips
that WP13 paid are gone: 56.9 → 47.7 ms).

### Prefill

| tokens | chunk | wall | tok/s | ms/token |
| --- | ---: | ---: | ---: | ---: |
| 512 | 128 | 5.18 s | 98.9 | 10.1 |
| 512 | 256 | 1.43 s | 357.1 | 2.80 |
| 512 | 512 | **0.45 s** | **1144.7** | 0.87 |
| 8,192 | 512 | **10.67 s** | **767.7** | 1.30 |

The chunk graph is the device default (`--chunk N`, 512 unless overridden;
`--chunk 0` and `--compare-cpu` keep the token loop). Four changes from WP14,
the fourth being WP17:

1. **Matrix-unit projections (WP17).** The prefill GEMM runs
   `wmma_f32_16x16x16_f16` on the gfx12 matrix unit — 256 tokens x 128 rows per
   block, 512 threads, eight 16x16x16 f32 accumulators per warp, weights
   dequantised into shared as half, f32 epilogue written straight to `y`. The
   ffn shapes measure 81–109 TFLOPS (85–114% of the quoted 95.7, 44–59% of the
   185 TFLOPS the instruction itself measures in a register-only probe), and the
   whole quantized tensor stream runs at 79.74 TFLOPS against 22.91 for the
   dot2 path. `--prefill-dot2` / `HIPML_PREFILL_DOT2` and `--prefill-f32` /
   `HIPML_PREFILL_F32` are the no-rebuild A/B switches; the fragment layouts are
   measured by `scripts/probe_wmma_layout.cpp`.
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

Where a 512-token, chunk-512 prompt's 0.45 s goes (`gpu-bench --prefill-split`):

| stage | ms | note |
| --- | ---: | --- |
| ffn GEMMs + SwiGLU | 200 | 17.5 TFLOP at ~88 TFLOPS |
| ssm GEMMs | 46 | qkv/gate/out, 5.7 TFLOP |
| attention GEMMs | 17 | q/k/v/o |
| elementwise | 15 | norms, residuals |
| sequential GDN | 113 | 48 layers × 6 launches, state in registers |
| sequential attention | 42 | 16 layers × 4 launches + output |
| final head | 2 | lm_head on the last row |

At 8K the same breakdown is ffn 3.22 s, attention 4.32 s, GDN 1.82 s, ssm GEMMs
0.74 s. The GEMM terms moved 3.5×; the sequential ones did not, and attention is
now the largest single term at 40%.

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
- **Attention is now the 8K limit.** 4.32 s of 10.67 s, ~3 TFLOPS: the tile
  kernel reduces each (query, key) score with a 5-step 32-lane shuffle chain and
  carries f32, so it is latency-bound, not FLOP-bound. Doubling the query tile
  (32 → 64 tokens) to halve the KV re-reads changed nothing (4294 → 4316 ms), and
  8 queries per warp (more ILP, ~160 VGPR) made it worse (5098 ms), which rules
  out both KV bandwidth and ILP as the lever. The lever is the matrix unit: 16
  queries x 16 keys per `wmma` puts the score in registers with no cross-lane
  reduction. Shape: block = 16 queries x 256 dims, 4 warps (one per 64-dim
  slice), S reduced through shared, P in shared as half, V stored transposed
  (dim-major) for the P·V operand, online max/sum per query row. That is the
  next kernel and it is not implemented.
- **The GEMM's remaining gap** to its 185 TFLOPS instruction ceiling (79.74
  whole-model, 43%) is the dequant (~12% of the instruction count), LDS tile
  traffic, and grid tails: at 512 tokens `ffn_gate` launches 80 blocks on 64 CUs
  and loses ~25%.
- **Next step, in order of size:** (1) the flash-shaped WMMA attention above
  (8K: expect 4.3 s → ~0.6 s, i.e. ~1150 tok/s); (2) an 8-queries-per-warp or
  split-K variant of the prefill GEMM for shapes with fewer than ~2 waves;
  (3) a fused layer kernel or two-stream scheduling to remove the 6.4 ms of DRAM
  ramp from 1,299 short decode kernels, still what stands between 35.3 ms and
  llama.cpp's 31.8 ms.
