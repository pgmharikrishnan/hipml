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

| measurement | WP13 | WP14 |
| --- | --- | --- |
| short-context decode, tokens 2..20 | 27.9 tok/s (35.9 ms) | **28.0 tok/s (35.7 ms)** |
| 512-token prefill | 22.99 s, 22.27 tok/s (token loop) | **5.35 s, 95.8 tok/s** (chunk 256) |
| 8,192-token prefill | — | **89.1 s, 91.9 tok/s** |
| decode at depth 20,000, q8 KV | 21.21 tok/s (47.1 ms) | **26.3 tok/s (38.0 ms)** |
| decode at depth 100,000, q8 KV | 17.58 tok/s (56.9 ms) | **20.7 tok/s (48.2 ms)** |

llama.cpp HIP on the same box: 31.4 tok/s short context, 21.86 tok/s at a
97,387-token prompt.

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
| 20,000 | 0.696 GB | 1.1 ms | 2.3 ms | 38.0 ms |
| 100,000 | 3.48 GB | 5.5 ms | 12.5 ms | 48.2 ms |

So at 100K the quantized-KV attention kernel runs at ~280 GB/s of a 632 GB/s
roofline; that, not the host, is the remaining depth gap (the 16 host round-trips
that WP13 paid are gone: 56.9 → 48.2 ms).

### Prefill

| tokens | chunk | wall | tok/s | ms/token |
| --- | ---: | ---: | ---: | ---: |
| 512 | 128 | 6.92 s | 74.0 | 13.5 |
| 512 | 256 | 5.35 s | **95.8** | 10.4 |
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

- Deep-path **token** parity is not asserted: q8 KV changes the attention values
  by design. `hipml parity --deep` verifies the kernel itself against the
  reference at `s=100000`; `hipml parity kv_encode` verifies the encoder is
  bit-exact. The graph's short-context tokens verify the rest.
- Chunked prefill requires `--kv f32`; the quantized caches are the decode path.
- The 8,192-token prefill is a real prefill; 100K prefill is not measured
  (extrapolated: 100K × 10.9 ms ≈ 18 minutes at chunk 256).
- Q5_K has no GEMV kernel; two tensors take the f32 fallback.
- No fp16 KV path.
- **Next step, in order of size:** (1) chunk variants of the conv/GDN kernels so
  a tile runs one kernel per layer instead of 464 launches (the 10.4 ms/token
  prefill floor); (2) a fused layer kernel or two-stream scheduling to remove
  the 6.4 ms of DRAM ramp from 1,299 short kernels — that is what stands between
  35.7 ms and llama.cpp's 31.8 ms; (3) the quantized-KV attention kernel at
  ~280 GB/s of 632, worth ~7 ms at 100K.
