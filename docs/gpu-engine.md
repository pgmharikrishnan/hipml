# The GPU graph: one token end to end

WP13 wires the pieces from WP3–WP14 into one decode graph and measures it. The
CPU reference from `feat/graph` stays the oracle: `--cpu` runs it, `--device`
runs the graph below, and both print the same ids at short context.

## Shape of the graph

```text
x = embed(token)                              dequant_q4_k_f32 on device, row read back
for layer 0..63:
    h  = rmsnorm(x, attn_norm)                rmsnorm_f32
    b  = full_attention(h) | gated_delta_net(h)
    x1 = x + b                                residual_add_f32
    t  = rmsnorm(x1, post_attention_norm)     rmsnorm_f32
    g  = W_gate t,  u = W_up t                gemv_q4_k / gemv_q6_k / gemv_q8_0
    a  = silu(g) * u                          silu_mul_f32
    x  = x1 + W_down a                        gemv (Q5_K rows: f32 fallback)
logits = W_output rmsnorm(x, output_norm)    gemv_q6_k
id = argmax(logits)                          host, 248320 values
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
logits are read back.

### Full attention

`AttnLayer` borrows the four resident projections (`new_resident`) instead of
uploading host `Mat`s. Its f32 KV path (mrope → head-RMS → q-gate → kv-append →
split-K attention) is the short-context path and the one the token check uses.

### Quantized KV at depth

`--kv q8` (default for `gpu-bench --depth`) swaps the cache for the `feat/kv-100k`
format: `[n_kv][seq][head_dim/32]` q8_0 blocks, K and V separate. The layer
computes q/k/v on device, the 2048-wide k/v row is read back, quantized on the
host with `hipml_core::kv::quantize`, appended at `pos`, and the decode pair
(`attn_decode_kv_q8` + reduce) runs over the split-K grid. q4_0 is selectable and
halves the cache again.

At 100K the q8 cache is **3.243 GiB (3.48 GB)** for the 16 attention layers,
against 13.1 GiB for the f32 path — which is why the f32 path is not used there.

### Gated delta net

`SsmGpuLayer::new_resident` borrows the five resident projections (qkv, gate,
alpha, beta, out) and keeps the small f32 vectors (conv taps, dt bias, A, norm)
as device buffers. The kernel chain is WP11's: gates → conv1d+silu → L2 →
gated delta recurrence → gated RMS. State is 156.9 MB per sequence and is not
prefix-cacheable, so prefill and decode cost the same per token.

## Commands

```powershell
# correctness: same ids as the CPU oracle, plus first-step logit distance
hipml generate --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf `
  --ids 760,6511,314,9338,369 --tokens 20 --device
hipml generate ... --device --compare-cpu        # loads the CPU oracle too

# the oracle itself, no GPU build needed
hipml generate ... --cpu

# speed: real 512-token prefill through the same token loop
hipml gpu-bench --gguf ... --prefill 512
# speed: decode at depth, caches pre-filled with valid blocks
hipml gpu-bench --gguf ... --depth 20000  --kv q8 --tokens 8
hipml gpu-bench --gguf ... --depth 100000 --kv q8 --tokens 8
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
| CPU reference | identical |
| llama.cpp | identical (`docs/forward-parity.md`) |

First-step logits against the CPU oracle: **max |d| = 4.77e-5** at index 233546;
device and CPU argmax both 11751.

### Acceptance 2 — speed

| measurement | result |
| --- | --- |
| short-context decode, tokens 2..20 | **27.9 tok/s** (19 steps, 681.5 ms, 35.9 ms/token) |
| short-context prefill (5-token prompt, same loop) | 28.2 tok/s |
| 512-token prefill (token loop) | **22.99 s, 22.27 tok/s** (44.9 ms/token) |
| decode at depth 20,000, q8 KV | **21.21 tok/s** (47.1 ms/token) |
| decode at depth 100,000, q8 KV | **17.58 tok/s** (56.9 ms/token) |

The short-context number is the weight stream plus launch overhead: 15.89 GiB at
608.9 GB/s is 28.0 ms, so 35.9 ms/token is 78% of the streaming roofline. The
gap at depth is the 16 quantized-KV attention passes (12.4 ms at 100K measured in
WP14) plus 16 host round-trips to quantize the new K/V row.

The 512-token prefill is the **token loop**, not the tiled prefill GEMM from
WP14. WP14 measured 1.89 s cold TTFT for the same 512 tokens with the tiled
GEMM; wiring that GEMM into this graph is the next step, so the 22.99 s here is
the honest number for the graph as shipped.

### At 100K

A 100K token prefill would be 100K decode steps (about an hour), so the depth
numbers come from pre-filling the caches with format-valid blocks and timing
decode at that depth; the SSM step cost is independent of the values. Prefill at
100K is therefore **not measured**. The measured prefill points are 5 tokens
(28.2 tok/s) and 512 tokens (22.3 tok/s), and the per-token cost is flat because
the recurrent state is per-token, so the extrapolation is 100K × 44.9 ms ≈ 75
minutes for a 100K prefill through the token loop.

## What runs where

| stage | device | host |
| --- | --- | --- |
| embedding row dequant | `dequant_q4_k_f32` | 20-block row copy (2.8 KiB) |
| norms, residual, SwiGLU | kernels | — |
| all weight matvecs | quantized GEMV over resident bytes | — |
| full attention (short) | mrope/head-norm/q-gate/append/split-K | — |
| full attention (deep) | quantized-KV decode pair | read k/v row, quantize, append |
| gated delta net | conv/L2/recurrence/gated-RMS | — |
| lm_head | resident Q6_K GEMV | argmax over 248320 logits |

The two host steps are deliberate: embedding needs one row (choosing a row on
device would need a gather kernel), and quantizing the new K/V row reuses the
exactly-tested `kv::quantize` rather than shipping an untested encode kernel.

## Unverified / next

- Deep-path **token** parity is not asserted: q8 KV changes the attention values
  by design. `hipml parity --deep` verifies the kernel itself against the
  reference at `s=100000`; the graph's short-context tokens verify the rest.
- The tiled prefill GEMM is not wired into the graph (see above).
- Q5_K has no GEMV kernel; two tensors take the f32 fallback.
- No fp16 KV path and no device-side KV encode kernel.
- The 100K decode includes 16 host syncs for K/V quantization; a device encode
  kernel would remove them and roughly halve the depth gap.
