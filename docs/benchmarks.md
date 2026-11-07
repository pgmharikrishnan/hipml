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
decode step streams **Q4_K, Q6_K, Q8_0 and Q5_K** blocks. These kernels dequantise
in registers and never materialise the fp32 matrix.

`gemv_q4_k` and `gemv_q6_k` use one block per output row and 256 threads. A thread
owns 8 consecutive columns, so x is read as two `float4`s, the 8 quant bytes are
one 64-bit (Q4_K) or eight 16-bit (Q6_K) loads, and one scale/min pair covers the
chunk. Q4_K's 6-bit header and Q6_K's four 2-bit `qh` planes are unpacked exactly
as in the hand-computed vectors in `crates/hipml-core/tests/quant_reference.rs`.

### Single tensor, Qwen3.8-27B shapes

`hipml bench` (the quant rows are the tail of the table):

| Weight | Shape `[m,k]` | Type | Bytes | us | GB/s | % of 631 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ffn_up | 17408x5120 | Q6_K | 73,113,600 | 160.59 | 455.3 | 72.2% |
| ffn_gate | 5120x17408 | Q4_K | 50,135,040 | 112.32 | 446.4 | 70.7% |
| ffn_down | 5120x17408 | Q6_K | 73,113,600 | 152.05 | 480.8 | 76.2% |
| ffn_down | 17408x5120 | Q4_K | 50,135,040 | 133.37 | 375.9 | 59.6% |

The same run's fp32 `gemv_f32` was **618.8 GB/s** (839 MB), so 631 GB/s remains
the reference ceiling.

### Decode step: stream the model once

`hipml decode-bench` reads every Q4_K/Q6_K/Q8_0 tensor in 64 MiB row-aligned
chunks, uploads each chunk, and launches the matching kernel. It never holds the
17 GB model in VRAM (or host RAM).

| Pass | Bytes | GB/s | implied tok/s |
| --- | ---: | ---: | ---: |
| single pass, one launch + sync per chunk (`reps=1`) | 16,964,915,200 | 217.8 | 12.84 |
| kernel-only, 11 launches per chunk (`--reps 11`) | 16,964,915,200 | 393.6 | 23.20 |

The second row amortises launch/sync latency (the first is a hard lower bound for
this harness, not for a real engine, which enqueues all tensors and syncs once).
Some small tensors become cache-resident across the 11 reps, so 23.2 tok/s is an
optimistic kernel-side estimate. `llama.cpp` HIP measures **31.4 tok/s** on the
same 17.19 GB. Including the 227 MB of F32 tensors at the same bandwidth gives
22.89 tok/s, so the miss is the kernel, not the accounting.

### Why not 95% of bandwidth

All four Qwen3.8-27B-shaped cases move the same activation volume: `m*k*4 = 356 MB` of
x re-reads per tensor, against only 50–73 MB of weights. With one block per row
and no shared-memory staging, those reads are served from L2/Infinity Cache. The
measured total (x + weights) is 406–429 MB in 112–161 us, i.e. **~2.7–3.6 TB/s of
L2 traffic**; the weight stream alone would take 80–116 us at 631 GB/s. The fp32
GEMV does not hit this because its x:weight ratio is 1:1; the quantized ratio is
5–8:1. Reaching 95% needs multiple rows per block reusing x from registers or
shared memory, which the one-block-per-row specification excludes.

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

Parity for the two formats (reference vs device, `max_abs` / `max_rel`):

| Case | 64x512 | 17408x5120 |
| --- | --- | --- |
| `gemv_q4_k` | 6.68e-6 / 5.21e-5 | 1.41e-4 / 8.21e-3 |
| `gemv_q6_k` | 3.43e-5 / 6.46e-6 | 1.13e-3 / 2.26e-2 |

The 5k-term rows are compared against a sequential f32 reference: an output near
zero from 5120 terms can legitimately move by ~1e-3 through cancellation. The
hand-computed block vectors are exact (`assert_eq!`), so a layout misread cannot
hide inside that budget. `12 passed, 0 failed, 1 skipped` (the host-only top-p
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
