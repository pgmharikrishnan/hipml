# Full attention on the GPU

The 16 full-attention blocks of Qwen3.8-27B run on the device as one layer
function over resident quantized weights. The CPU reference is
`crates/hipml-core/src/attention.rs` / `model.rs`; this note records what was
checked and what it costs.

```
hipml attn-parity --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf
hipml attn-bench  --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf --depths 0,8192,100000
```

Device: AMD Radeon AI PRO R9700 (gfx1201), 31.9 GiB. Layer 3:
24 query / 4 KV heads, head_dim 256, M-RoPE `dimension_count 64`,
sections `[11, 11, 10, 0]`, base 1e7. Weights: `attn_q` Q4_K 12288x5120,
`attn_k`/`attn_v` Q8_0 1024x5120, `attn_output` Q8_0 5120x6144.

## Kernels

| op | entry | note |
|---|---|---|
| per-head q/k RMS norm | `head_rmsnorm_f32` | head_dim 256, one block per head, in-place safe |
| M-RoPE | `mrope_f32` | NeoX pairs over the first 64 channels; section map is the reference's |
| q/gate split | `qg_split_f32` | `[q\|gate]` per head from `attn_q` |
| q gate | `q_gate_f32` | `o *= sigmoid(gate)` |
| KV append | `kv_append_f32` | position-major `[n_ctx][n_kv*head_dim]` |
| decode attention | `attn_decode_split_f32` + `attn_reduce_f32` | split-K over sequence slices, one warp per query head of a KV group |
| fallback / parity | `attn_decode_f32` | one block per head; grew `head_stride`/`row_stride` so it serves both cache layouts |

The base branch has no quantized-KV path (that kernel lives on `feat/kv-100k`,
not merged). The f32 cache therefore uses the split-K sibling of that kernel:
same grid shape and shared K/V tile, natural-log softmax. The old
`attn_decode_f32` is kept for parity and for head shapes the split kernel does
not cover (`head_dim % 32 != 0`).

## Parity vs the CPU reference

Tolerance is `max_abs <= 1e-4` **or** `max_rel <= 1e-3` (the harness' rule).
Large relative errors are expected where the expected value is near zero.

| case | max_abs | max_rel | verdict |
|---|---:|---:|---|
| q norm + M-RoPE, 24 heads | 1.19e-6 | 3.59e-6 | PASS |
| k norm + M-RoPE, 4 heads | 9.54e-7 | 8.41e-7 | PASS |
| one layer, depth 1 | 2.48e-5 | 1.79e-2 | PASS |
| one layer, depth 1000 | 4.17e-6 | 2.06e-2 | PASS |

Single-op parity (same run): `mrope_f32` 2.38e-7, `head_rmsnorm_f32` 3.58e-7,
`qg_split_f32` exact, `q_gate_f32` 2.98e-8, `kv_append_f32` exact,
`attn_decode_pm_f32` 1.34e-7, `attn_decode_split_f32` 1.49e-7.

Worst error overall is the depth-1 layer at **2.48e-5 absolute**.

## Per-layer time and the 16-layer projection

`reps=7`, one resident layer (weights + zeroed cache), sequence length
`depth + 1`:

| depth | layer ms | 16 layers ms | effective weight GB/s |
|---:|---:|---:|---:|
| 0 | 0.189 | 3.03 | 422.7 |
| 8192 | 0.441 | 7.05 | 181.4 |
| 100000 | 2.776 | 44.41 | 28.8 |

Weights are 80.0 MB per layer (1279.3 MB for 16). Against the whole-model
weight-read time of **27.19 ms** per decode step: at depth 0 the attention
weights are ~3.0 ms, so depth-0 attention is a small addend. At 100K the KV
read dominates and the 16 blocks cost **~44 ms**, more than the model's entire
weight stream. (The first version of this used `attn_decode_f32` directly and
measured 100.3 ms/layer at 100K; split-K is 36x faster.)

## KV cache VRAM

Reference layout is `[n_ctx][n_kv*head_dim]` f32 for **both** K and V:
4 KV heads x 256 x 4 B = 4 KiB per array, 8 KiB per layer per token.

| item | value |
|---|---|
| one layer at 100,001 positions | 819.2 MB |
| 16 layers at 100K | 13.11 GB |
| per token, 16 layers | 128 KiB |

The brief expected 64 KiB/token = 6.55 GB, which is the K cache alone; the
reference stores V at the same f32 width, hence 2x that number.

## Not verified

- Only layer 3 was loaded; the other 15 blocks are assumed to share its shapes
  and quantization types.
- Parity is on one token at a time. Prefill (batch > 1) and the full-model
  logits are not compared.
- The split kernel is exercised at head_dim 256 only; the fallback path is
  compile-time present but not parity-tested on this GPU.
- Q5_K attention weights would be rejected (no GEMV kernel); this model has
  none.

## Next step

Wire `AttnLayer` into the whole-model GPU graph dispatcher, then decide whether
100K contexts justify porting the quantized-KV kernel (13.11 GB is most of the
card). Tuning `SPLIT_TILE` / split count is the cheap way to close the remaining
gap to the ~600 GB/s roofline at depth.

## Chunk attention on the matrix unit (WP20)

`attn_chunk_wmma_f16` replaces `attn_chunk_tile_f32` for the chunked-prefill
path (`HIPML_ATTN_F32=1` keeps the old kernel for A/B). The f32 kernel reduced
every (query, key) score with a five-step 32-lane shuffle chain; at 8K that was
3.58 s of a 9.93 s prompt, 3.7 TFLOPS of the matrix unit's measured 171-188.

Geometry: one block per (KV head, 32 queries, half of the head width), two
warps per query head over 16-query fragments. K is staged in full and V only for
the block's half, both to shared, shared by all six query heads of the GQA
group. `blockIdx.z` selects the half: a warp that holds both the 16 Q fragments
(64 VGPRs) and 16 output fragments (128 VGPRs) needs ~240 VGPRs, and the
compiler responds by spilling the output fragments to scratch *inside the key
loop* (four `scratch_store_b128`/`scratch_load_b128` pairs per tile in the
generated assembly). Eight output fragments fit: 192 VGPRs with no scratch in
the loop, one block per CU.

The score tile is computed transposed, `S^T = K Q^T`, which is what makes both
cache tiles stage row-major and the softmax one shuffle. Two layout facts cost
two debugging rounds and are recorded here because both make the kernel compute
plausible-looking garbage rather than fail:

- The matrix unit's **B operand is row-major by output column** (`B_hw[n][k]`,
  as `prefill_wmma.hip` feeds its weights), so Q needs no transpose and V must
  be staged transposed. The first version fed both transposed.
- A **cache row is the global position**. The chunk appends at `pos0`, so the
  causal mask compares the row index against `pos0 + chunk_row`; adding `pos0`
  to the key side as well silently restricted every chunk after the first to
  its own 32 rows.
- P's A-fragment layout is bit-for-bit the score accumulator's, so the
  probability tile goes from the softmax registers into PV with no shared
  round trip.

`hipml attn-parity` now runs the kernel against `attn_causal_f32` on a 37-query
chunk at position 29 (`chunk attention 37@29`, max |d| 8.35e-4, tolerance 2e-3).

| 8K prefill | wall | attention stage (`--prefill-split`) | TFLOPS |
| --- | ---: | ---: | ---: |
| f32 tiled kernel | 9.70 s (845 tok/s) | 3598 ms | 3.7 |
| `attn_chunk_wmma_f16` | **6.82 s (1201 tok/s)** | **779 ms** | **16.9** |

The attention stage includes the layer's output GEMM (~50 ms at 8K), so the
kernel itself is ~730 ms. The remaining gap to the 171 TFLOPS instruction
ceiling is the block's KV re-read (32-query tiles re-read the prefix; the layer
moves ~12.6 GB of K/V because the d-half split reads K twice) and the 8-row
PV/16-key tile's ratio of staging bytes to matrix work.

### Reading the quantized cache instead of the mirror (WP25)

The chunk kernel's staging pass now takes a cache dtype. `aw_ld4` turns a
4-value group of a q8_0 (34 B) or q4_0 (18 B) block into the same half4 the f32
mirror writes, so QK^T, the causal mask, the softmax and PV are untouched; the
staging loops are compiled once per dtype and dispatched once per tile, because
a dtype branch inside the load cost the f32 mirror 1.5% at 32K. A q8 row is 272 B
at head_dim 256 instead of 1 KB, and there is no 12 GiB mirror.

With the q8 staging in place the mirror stopped paying for itself: 8K prefills
at 998 tok/s without it (993 through the mirror, which reads 4x the bytes) and
32K is level (867 vs 871). A 32K mirror costs 4.3 GiB and 100K's would be
13.4 GiB, so quantized engines never allocate one; `--kv f32` keeps the f32
cache and the token loop as the parity reference. 100K prefill went
116 -> 637 tok/s (TTFT 860.8 -> 156.9 s) at 20,995 MiB peak, and the mirror
removal takes 1.1 GiB off 8K and 4.0 GiB off 32K.

The staged tile traffic is `3 * row_bytes` per key position (K twice, once per
d half, and V once): 8.16 TB at 100K, i.e. 2,345x the 3.48 GB unique cache, at
an effective 124-137 GB/s. That is well under the 640 GB/s spec, so the deep
kernel is matrix-issue bound (~50 TFLOPS), not HBM bound.

## Decode at 100K: what the 12 ms is bound by

`hipml bench-kv --depths 100000` on the q8 cache, 24 heads x 16 layers,
`--splits` overriding the engine's 64. Splits move nothing, so the kernel is
not parallelism- or combine-bound:

| splits | q8 (f32 staging) | q8s (q8 staging) | q8g (no staging) | q4 (f32 staging) |
| ---: | ---: | ---: | ---: | ---: |
| 64 | **12.85 ms** | 58.0 ms | 34.7 ms | 15.1 ms |
| 128 | 13.76 ms | 56.7 ms | 33.9 ms | 15.0 ms |
| 256 | 13.69 ms | 55.4 ms | 32.6 ms | 14.4 ms |

Three alternatives were built to locate the limit and all lost:

- **q8s** stages the 272 B quantized rows and dequantizes inside each of the six
  query-head warps: 4.5x slower. Staging f32 once is cheaper than six
  dequantizations per row.
- **q8g** drops shared staging entirely and lets each warp read its own copy
  from global: 2.7x slower.
- **q4** halves the cache bytes (1.72 GiB vs 3.24) and is *slower* than q8.

If bytes, shared traffic, and split count all fail to move it, the kernel is
issue/latency-bound in the six-warp score/PV path: per cache row each warp does
two 128-bit K loads, eight FMAs, five bpermute+add butterfly steps, one exp2,
two V loads and eight more FMAs, ~35-40 instructions per row per warp, and the
dynamic count agrees with the measured ~5 cycles per (KV head, row) per CU. A
matrix-unit decode kernel (16 keys against the six GQA heads padded to 16 rows)
is the way out, not more splits.

## Decode kernel rewrite: one warp per KV head (WP31)

The staged kernel above reads the cache through shared memory with six warps per
row (one per query head) and measures 12.58 ms for the 3.242 GiB q8 cache at
100K (276.7 GB/s at 256 splits). A phase-attribution kernel
(`attn_decode_kv_lab`, entries `lab_*`) compiles phases out so one 100K pass
says where the time goes (16 layers, 64 splits):

| phases compiled in | us | GB/s |
| --- | ---: | ---: |
| stage K+V, score, PV (full) | 12429 | 280.1 |
| stage K, score, no V | 9237 | 376.9 |
| stage V, no score, PV | 4892 | 711.7 |
| stage K+V, no compute | 1730 | 2013.0 |
| score+PV, no staging | 11191 | 311.1 |

Staging is 1.7 ms and overlaps with compute (full is 0.5 ms under the sum of
the two floors). The score chain — two K dword loads, eight FMAs and a
five-step `ds_bpermute` butterfly per row and head — is ~8.4 ms of the 12.4 ms
and PV ~4.0 ms. The instruction count, not the cache read, is the limiter.

The new kernel (`attn_decode_kv_w.hip`) gives one warp a whole (KV head, key
range) task and keeps the six query heads of the GQA group in it: `qv[6][8]`
and `acc[6][8]`, eight components per lane, one 8-deep dot chain per head
reduced by the same five butterfly steps, the same online softmax and the same
split/reduce ABI. Two shapes exist: `w2`/`w4` (whole group, 2/4 rows per tile)
and `s1`/`s2`/`s4` (three heads per warp, two warps per task sharing the row
through L1). 100K, 16 layers, 256 splits:

| entry | warps/task | VGPRs | us | GB/s |
| --- | ---: | ---: | ---: | ---: |
| `attn_decode_kv_q8` (staged) | 6 | 192 | 12583 | 276.7 |
| `attn_decode_kv_s1` | 6 | 101 | 10375 | 335.6 |
| `attn_decode_kv_s2` | 3 | 110 | 9462 | 367.9 |
| `attn_decode_kv_s4` | 2 | 134 | 8815 | 395.0 |
| `attn_decode_kv_w4` | 1 | 192 | - | 324 |
| `attn_decode_kv_w2` (shipped) | 1 | 192 | 8323 | 418.3 |

Fewer warps per task wins: the whole group in one warp removes the duplicate
K/V reads, and the L1 sharing between the two `s4` warps does not pay for the
extra issue slots. At 100K, 384 splits are level with 256 (397 GB/s for `s4`)
and 128 is worse (310 for `w2`). The engine therefore uses one split per 64
positions, capped at 256: at 100K that is 256 (the 100K optimum), but at 8K it
is 128 instead of the 32 that `seq/256` produced. That matters — `bench-kv
--kernels 4` at 8K measures 2.90 ms with 32 splits, 1.59 with 64 and 1.12 with
the 128 cap; the engine decode recovers from 27.46 to 28.34 tok/s. At 16K the
256 cap wins (1.72 vs 2.98 ms for 64).

**An earlier `s4` reading of 698 GB/s was a measurement bug, not a kernel.**
The launcher sized the grid as `ceil(tasks/4)` blocks for a kernel that needs
`ceil(group/3)` warps per task, so every `s*` run above group 3 launched only
the first half of its warps and read half the cache. The fix launches
`ceil(tasks * wpr / 4)` blocks; `s4` then measures 395 GB/s and `w2` (one warp
per task, never affected) is the fastest entry. The same bug made
`attn_decode_kv_q8_hd256_s1000` and `_s4096` fail parity at max abs 0.103 and
0.052; after the fix they are 1.27e-7 and 1.68e-7 and the whole KV table plus
both 100K deep cases is green.

Engine decode at 50 tokens (eager, q8 cache, unfused loop):

| depth | baseline e9705d2 | shipped (w2) | delta |
| ---: | ---: | ---: | ---: |
| 0 | 33.54 ms (29.82) | 33.70 ms (29.68) | level |
| 32,768 | 37.89 ms (26.39) | 37.59 ms (26.60) | +0.8% |
| 100,000 | 46.59 ms (21.47) | 42.15 ms (23.73) | +9.5% |

Depth 0 is level (33.4-33.7 ms across runs of either tree) and 32K is slightly
better. Attention at 100K fell from 12.3-12.6 ms to 8.3 ms, which is the whole
4.4 ms of the step; the weight stream (27.2 ms at 632 GB/s) and the rest of the
step are unchanged. **The 24.5 tok/s target was not reached**: 100K needs
40.8 ms and the step is 42.15 ms, so the remaining gap is 1.35 ms of attention
(16%). With f32 dot math the score phase is the floor; the way past it is a
matrix-unit score (16 keys x 16 heads) with an f16 dot, which the 1e-4 parity
tolerance does not currently allow, or a whole-group kernel at one row per tile
to cut the 192 VGPRs and raise occupancy.

Not measured here: the `s1`/`s2`/`s4` entries stay in the manifest for A/B but
are not the shipped path, and the graph path (`--graph`) was not re-timed after
the kernel change.

The fused residual+rmsnorm and SwiGLU epilogue were built and measured before
being reverted: 42.478 ms vs 42.542 unfused at 100K (+0.06 ms, noise) and
34.078 vs 33.566 ms at depth 0 (-0.51 ms, a regression). The ~190 launches it
removes are not on the critical path at 2-3 us each, and the fused kernels cost
more per launch than the pairs they replace, so the decode loop keeps the
unfused launches.

`hipml attn-parity` failed `layer depth 1` (max abs 8.0) and `layer depth 1000`
(0.94) on every tree, including the base commit: the harness drove
`AttnLayer::forward` without publishing the step buffer, so the decode kernels
read the sequence length left behind by the norm+rope case (38) and depth 1
attended 38 rows of a one-row f32 cache. `forward` now publishes the step (the
engine already did, which is why the token loop was unaffected); the two cases
pass at 2.48e-5 and 4.17e-6.

## KV layout: f32 for the chunk, q8 for depth

The chunk path appends f32 rows because the attention kernel stages the cache
directly; the decode path uses q8 blocks written by `kv_encode_q8`. Measured
VRAM for one model:

| context | f32 K+V (both paths) | q8 K+V | q4 K+V |
| ---: | ---: | ---: | ---: |
| 8,192 | 1.00 GiB | 0.27 GiB | 0.14 GiB |
| 32,768 | 4.00 GiB | 1.06 GiB | 0.56 GiB |
| 100,000 | 12.50 GiB | 3.24 GiB | 1.72 GiB |

A single q8 layout would save 1.7-9 GiB at those depths, but the chunk path is
not bandwidth-bound in the same way: at 8K the attention stage moves ~12.6 GB
of f32 K/V in 779 ms, so a q8 cache would cut maybe 0.2 s of a 6.8 s prompt
while adding a dequant per staged value and a device-side chunk encoder. The
prefill trade is small; the depth trade (12.5 GiB vs 3.24) is the reason the two
layouts persist.

One layout finding does survive: the PV operand wants V **dim-major**, so the
cache's key-major V is transposed through shared every tile. A dim-major f32 V
cache would remove that staging pass; it was not measured because it touches
every f32-cache reader (append, the chunk kernels and the decode fallback) for
an estimated ~5% of the attention stage.
