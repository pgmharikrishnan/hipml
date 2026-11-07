# Speed: measured against the roofline

The only honest way to judge an inference stack is against the ceiling its
hardware imposes, not against another stack. Two ceilings matter and they are
different for the two phases of a token.

**Decode is bandwidth-bound.** One token streams the whole weight set through the
memory bus, so

```
decode tokens/s ≤ achieved bandwidth / weight bytes
```

**Prefill is compute-bound.** It runs the whole prompt through the weights, so

```
prefill tokens/s ≤ fp16 peak / (2 × parameters)
```

Radeon AI PRO R9700: 32 GB GDDR6 on a 256-bit bus at 20 Gbps → **640 GB/s** spec,
of which a hand-written HIP GEMV already achieves **633.6 GB/s** (99%), and
**95.7 TFLOPS** fp16 dense. Using the measured bandwidth rather than the spec
keeps the decode ceiling honest.

## The model under test

`bartowski/NousResearch_Qwen3.8-27B-GGUF`, Q4_K_M: 16.01 GiB, 26.90 B parameters,
`qwen3_5`. Decode ceiling **36.9 tok/s**, prefill ceiling **1779 tok/s**.

Two reference points measured on the same box in the same conditions:

| Model | Weights | Decode ceiling |
| --- | ---: | ---: |
| Qwen3.8-27B Q4_K_M | 16.01 GiB | 36.9 tok/s |
| Swift-1.5-27B Q4_K_M | 16.23 GiB | 36.4 tok/s |
| Ripple-65B IQ3_XXS | 26.04 GiB | 22.7 tok/s |

## Measured

`llama-bench -ngl 99 -p 512 -n 128 -r 3`, Vulkan b12126 versus ROCm/HIP b11820.

| Backend | Test | Measured | Ceiling | % of ceiling |
| --- | --- | ---: | ---: | ---: |
| Vulkan | pp512 | 974.8 t/s | 1779 | 54.8% |
| **HIP** | pp512 | **1256.5 t/s** | 1779 | **70.6%** |
| Vulkan | tg128 | **32.06 t/s** | 36.9 | **87.0%** |
| HIP | tg128 | 31.30 t/s | 36.9 | 84.9% |
| Vulkan | pp512 @ d8192 | 885.8 t/s | 1779 | 49.8% |
| HIP | pp512 @ d8192 | 1144.2 t/s | 1779 | 64.3% |
| Vulkan | tg128 @ d8192 | 31.14 t/s | 36.9 | 84.5% |
| HIP | tg128 @ d8192 | 30.27 t/s | 36.9 | 82.1% |

## What it says

**Decode is nearly finished.** Both backends land at 82–88% of the memory
roofline, so the remaining headroom on a dense model is about 15%, and it is
shared by every engine on this hardware — including a custom one. Vulkan is
marginally ahead (32.06 vs 31.30, +2.4%), which contradicts the widely repeated
claim that Vulkan is ~20% faster at token generation on RDNA4 Windows.

**Prefill is where the difference is.** HIP reaches 70.6% of fp16 peak against
Vulkan's 54.8% — a +29% gap that shows up as a shorter time-to-first-token on
long prompts, and roughly halves at depth (64.3% vs 49.8% at 8K).

**Depth costs about 3% of decode and 9% of prefill** at 8K. Cheap, as long as the
cache stays in VRAM.

**The engine opportunity is measurable and small on a mature architecture.** Ripple
sits at 70.6% of its own decode ceiling (16.0 of 22.7 tok/s) while Qwen3.8-27B and Swift
sit at 87–88%. That 17-point gap is the entire case for a custom engine on this
card: 16 → ~19.5 tok/s, not a doubling. It exists because `ricursive` has no
Flash Attention path, and it disappears as soon as llama.cpp gains one.

## Quantized GEMV: Q4_K and Q6_K

The fp32 GEMV above measures the card, not the model. Qwen3.8-27B is Q4_K_M, so a
decode step streams **Q4_K, Q6_K, Q8_0 and Q5_K** blocks. These kernels
dequantise in registers and never materialise the fp32 matrix.

The first version used one block per output row. At 4-bit weights x is 5-8x the
weight bytes, so every row re-read x from L2 and the kernel topped out at
373-483 GB/s. The fix is the row-tile structure from `gemm_f32`: one block owns
16 output rows, each warp owns 2 rows, and the x k-tile is staged in shared once
per tile. A shared x value is applied to both of the warp's rows before it is
discarded, so the weight stream, not x, sets the pace. The tile is de-interleaved
by 512-value group (`sx[g*512 + comp*32 + lane]`) so the 16 scalar reads hit
distinct banks. Q4_K lanes take 16 columns (one scale/min per half sub-block);
Q6_K lanes take 8 (one scale per 8 of a 16-value sub-block). Two barriers per
k-tile, one warp shuffle reduction per row, no reordering of the maths.

### Single tensor, Qwen3.8-27B shapes

`hipml bench` (the quant rows are the tail of the table):

| Weight | Shape `[m,k]` | Type | Bytes | us | GB/s | % of 631 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ffn_up | 17408x5120 | Q6_K | 73,113,600 | 110.31 | 662.8 | 105.0% |
| ffn_gate | 5120x17408 | Q4_K | 50,135,040 | 77.15 | 649.8 | 103.0% |
| ffn_down | 5120x17408 | Q6_K | 73,113,600 | 117.27 | 623.5 | 98.8% |
| ffn_down | 17408x5120 | Q4_K | 50,135,040 | 70.61 | 710.1 | 112.5% |

The same run's fp32 `gemv_f32` was **634.2 GB/s** (839 MB), so the reference
ceiling stays 631 GB/s. The >100% rows are the 50-73 MB tensors fitting partly in
the 64 MB Infinity Cache across the 51 benchmark reps; the decode stream below
is the conservative number.

### Tile tuning

All rows are GB/s at the four shapes above, in the same order. The best
compromise is the last row: Q6_K regresses when its lane widens to 16 values at
k=17408, so the two formats use different lane widths.

| Config | ffn_up Q6_K | ffn_gate Q4_K | ffn_down Q6_K | ffn_down Q4_K |
| --- | ---: | ---: | ---: | ---: |
| per-row block (old) | 442.5 | 443.2 | 483.5 | 372.7 |
| 1 warp/row, 8-value, K=512 | 541.2 | 395.9 | 511.7 | 423.9 |
| 2 rows/warp, 8-value, K=512 | 631.6 | 417.5 | 583.6 | 495.4 |
| 2 rows/warp, 8-value, K=1024 | 654.0 | 454.0 | 608.8 | 534.1 |
| 2 rows/warp, 8-value, K=2048 | 662.5 | 476.5 | 624.0 | 551.7 |
| 2 rows/warp, 8-value, K=4096 | 664.0 | 497.2 | 602.8 | 560.9 |
| 4 rows/warp, 8-value, K=2048 | 504.6 | 456.9 | 417.1 | 549.5 |
| 2 rows/warp, 16-value both, K=2048 | 654.9 | 642.7 | 544.5 | 706.7 |
| **2 rows/warp, Q4 16-val / Q6 8-val, K=2048** | **662.9** | **648.8** | **623.1** | **709.2** |

Four rows per warp loses Q6_K (more registers and fewer rows in flight), and
16-value Q6_K lanes lose 80 GB/s at k=17408; Q4_K wants the wider lane because
its 6-bit header is a larger share of its weight bytes.

### Decode step: stream the model once

`hipml decode-bench` reads every Q4_K/Q6_K/Q8_0 tensor in 64 MiB row-aligned
chunks, uploads each chunk, and launches the matching kernel. It never holds the
17 GB model in VRAM (or host RAM).

| Pass | Bytes | GB/s | implied tok/s |
| --- | ---: | ---: | ---: |
| single pass, one launch + sync per chunk (`reps=1`) | 16,964,915,200 | 259.7 | 15.31 |
| kernel-only, 11 launches per chunk (`--reps 11`) | 16,964,915,200 | 604.3 | 35.62 |

Per type at `--reps 11`: Q4_K **612.1**, Q6_K **557.0**, Q8_0 **708.5**. The
first row is a hard lower bound for this harness (it syncs per 64 MiB chunk, not
per token); the second amortises launch/sync latency and is the comparable
number. Including all 17.192 GB of tensor bytes at the same bandwidth gives
**35.15 tok/s**; `llama.cpp` HIP measures **31.4 tok/s** on the same file.

### Commands

On the GPU box, from `C:\hipml-q4k`:

```
set HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
set HIPML_ARCH=gfx1201
set PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
cargo build --release -p hipml-cli --features hip
target\release\hipml.exe parity
target\release\hipml.exe bench
target\release\hipml.exe decode-bench C:\models\Qwen3.8-27B-Q4_K_M.gguf -v
target\release\hipml.exe decode-bench C:\models\Qwen3.8-27B-Q4_K_M.gguf -v --reps 11
target\release\hipml.exe gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf
```

Parity for the three kernels (reference vs device, `max_abs` / `max_rel`):

| Case | 64x512 | 17408x5120 |
| --- | --- | --- |
| `gemv_q4_k` | 5.48e-6 / 3.49e-5 | 1.37e-4 / 9.20e-3 |
| `gemv_q6_k` | 4.20e-5 / 1.44e-5 | 1.10e-3 / 5.40e-2 |
| `gemv_q8_0` | 2.38e-6 / 4.27e-6 | — |

The 5k-term rows are compared against a sequential f32 reference: an output near
zero from 5120 terms can legitimately move by ~1e-3 through cancellation. The
hand-computed block vectors are exact (`assert_eq!`), so a layout misread cannot
hide inside that budget. `13 passed, 0 failed, 1 skipped` (the host-only top-p
case).

## Method notes

- Bandwidth ceiling uses the measured 633.6 GB/s, not the 640 GB/s spec, so the
  percentages are not flattered.
- The fp16 peak of 95.7 TFLOPS is the vendor's dense figure; 70% of it under HIP
  is a realistic matmul efficiency, not a shortfall to chase.
- Neither ceiling accounts for KV-cache traffic or launch overhead, so no
  measurement should be expected to reach 100%.
- Numbers are `llama-bench` medians of three repetitions, same model file, same
  box, sequentially run.
