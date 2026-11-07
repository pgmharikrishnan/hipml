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
