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

### Qwen3.8-27B hybrid KV traffic roofline

Qwen3.8-27B has 16 full-attention layers (every fourth layer), not 64; its
remaining 48 layers use fixed-size recurrent state. Each attention layer has
24 query heads, 4 KV heads and head width 256. At 32 values per block, the
16-layer KV traffic per cached token is 65,536 bytes for f16, 34,816 bytes
for q8 (2-byte scale plus 32 bytes), and 18,432 bytes for q4 (2-byte scale
plus 16 bytes). Query gating and per-head query/key RMS normalization are
separate operations.

The following is an arithmetic ceiling, **not a benchmark**. It assumes
17.19 GB of weights and one read of each KV element per decode step, at the
measured 631 GB/s. It excludes recurrent state, activations, compute, and
launch overhead; repeated KV reads for GQA can lower achievable throughput.

| Context tokens | f16 tok/s | q8 tok/s | q4 tok/s |
| ---: | ---: | ---: | ---: |
| 0 | 36.71 | 36.71 | 36.71 |
| 8,192 | 35.60 | 36.11 | 36.39 |
| 32,768 | 32.63 | 34.42 | 35.46 |
| 100,000 | 26.58 | 30.52 | 33.15 |
| 131,072 | 24.48 | 29.01 | 32.18 |

At 100,000 tokens, f16 KV is 6.5536 GB: 38.1% of the weight bytes, or
27.6% of combined weight and KV bytes. Weights plus f16 KV total 23.7436 GB,
before activations and recurrent state; quantization is not a KV capacity
requirement at this depth. The q8 ceiling rises by 3.94 tok/s at 100,000
tokens and 0.51 tok/s at 8,192; q4 rises by 6.57 and 0.79 tok/s respectively.

Formula: `631e9 / (17.19e9 + depth * bytes_per_token)`. These figures
are distinct from the measurements below, which use 633.6 GB/s.

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

## Method notes

- Bandwidth ceiling uses the measured 633.6 GB/s, not the 640 GB/s spec, so the
  percentages are not flattered.
- The fp16 peak of 95.7 TFLOPS is the vendor's dense figure; 70% of it under HIP
  is a realistic matmul efficiency, not a shortfall to chase.
- Neither ceiling accounts for KV-cache traffic or launch overhead, so no
  measurement should be expected to reach 100%.
- Numbers are `llama-bench` medians of three repetitions, same model file, same
  box, sequentially run.
