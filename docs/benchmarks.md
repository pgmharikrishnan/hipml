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

## Quantized KV at depth (measured)

The formats, the reference and the kernels are in `hipml-core/src/kv.rs`,
`ops::attn_decode_kv` and `kernels/attn_decode_kv.hip`. The block layouts are
ggml's, so a cache packed here is the same bytes `llama.cpp -ctk/-ctv` holds:

| dtype | block | bytes/value | Qwen3.8-27B bytes/token | 100K cache | 128K cache |
| --- | --- | ---: | ---: | ---: | ---: |
| f16 | — | 2.0 | 65,536 | 6.5536 GB | 8.3886 GB |
| q8_0 | 2 B f16 scale + 32 i8 = 34 B | 1.0625 | 34,816 | 3.4816 GB | 4.5639 GB |
| q4_0 | 2 B f16 scale + 16 B packed = 18 B | 0.5625 | 18,432 | 1.8432 GB | 2.4159 GB |

Qwen3.8-27B bytes/token = `2 (K,V) * 4 kv heads * 16 full-attention layers * 256`.
Parity is `Tol::DEFAULT` (abs 1e-4, rel 1e-3); the worst absolute difference
across the 14 quantized-attention cases, including the 100,000-token one, was
**2.4e-7**. Both sides read the same encoded bytes, so the comparison measures
accumulation order, not the format: the quantization error cancels.

### Decode attention: depth, one head and the whole step

`hipml bench-kv --depths 1024,8192,32768,100000 --reps 5`, gfx1201, one process
per row set. "Step" is one pass over 16 layers at that depth, i.e. the attention
work of one 100K decode token. `share` and `tok/s` count the measured weight
stream (632.0 GB/s, 27.20 ms for 17.19 GB) plus attention; they exclude every
other layer op, so they are an attention-only model of a decode step.

| dtype | heads | layers | depth | splits | µs/step | KV GiB | KV GB/s | share | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q8 | 1 | 1 | 1,024 | 16 | 200.2 | 0.001 | 2.8 | — | — |
| q8 | 1 | 1 | 8,192 | 128 | 238.7 | 0.004 | 18.7 | — | — |
| q8 | 1 | 1 | 32,768 | 256 | 473.5 | 0.017 | 37.6 | — | — |
| q8 | 1 | 1 | 100,000 | 256 | 545.3 | 0.051 | 99.8 | — | — |
| q8 | 24 | 16 | 1,024 | 16 | 634.3 | 0.033 | 56.2 | 2.3% | 35.93 |
| q8 | 24 | 16 | 8,192 | 64 | 2,005.5 | 0.266 | 142.2 | 6.9% | 34.24 |
| q8 | 24 | 16 | 32,768 | 64 | 7,425.6 | 1.062 | 153.6 | 21.4% | 28.88 |
| **q8** | **24** | **16** | **100,000** | **64** | **21,869.0** | **3.242** | **159.2** | **44.6%** | **20.38** |
| q8 | 24 | 16 | 131,072 | 64 | 28,284.8 | 4.250 | 161.3 | 51.0% | 18.03 |
| q4 | 1 | 1 | 1,024 | 16 | 103.3 | 0.000 | 2.9 | — | — |
| q4 | 1 | 1 | 8,192 | 128 | 115.9 | 0.002 | 20.4 | — | — |
| q4 | 1 | 1 | 32,768 | 256 | 433.5 | 0.009 | 21.8 | — | — |
| q4 | 1 | 1 | 100,000 | 256 | 838.1 | 0.027 | 34.4 | — | — |
| q4 | 24 | 16 | 1,024 | 16 | 650.7 | 0.018 | 29.0 | 2.3% | 35.91 |
| q4 | 24 | 16 | 8,192 | 64 | 2,063.1 | 0.141 | 73.2 | 7.1% | 34.18 |
| q4 | 24 | 16 | 32,768 | 64 | 7,522.6 | 0.562 | 80.3 | 21.7% | 28.80 |
| q4 | 24 | 16 | 100,000 | 64 | 22,248.2 | 1.717 | 82.8 | 45.0% | 20.22 |
| q4 | 24 | 16 | 131,072 | 64 | 28,745.2 | 2.250 | 84.0 | 51.4% | 17.89 |

The one-head rows are launch- and parallelism-bound (16 to 256 warps on a 64-CU
card) and are not a bandwidth statement; they are here because the brief asks
for them. The 24-head rows are the meaningful measurement.

**The kernel does not reach the 300 GB/s acceptance target.** The best deep
number is 159–162 GB/s of unique KV traffic (q8, 100K and 128K; the range is
repeated runs of the same binary), 25% of the card's 632 GB/s. q4 is limited by
the same thing at half the bytes: 83–84 GB/s at 100K because it takes the same
22 ms as q8 on half the data, so the loop is bounded by work per cached value,
not by DRAM. Getting to 300 needs fewer instructions per value —
wider shared reads (`float2`/`float4` per lane), or an MMA path for the score —
not more parallelism.

What the measurements did establish:

- **GQA duplication was the first wall.** With one block per query head the six
  heads sharing a KV head each pulled the row from DRAM; the kernel measured
  108 GB/s of unique traffic while DRAM was already at ~648 GB/s, i.e. six
  reads of every byte. One block per (KV head, split) with the six heads as
  warps, reading a dequantized shared tile, moved it to 159 GB/s.
- **Branching around each block load cost more than the loads.** The SASS for
  `(c < nb) ? load : 0` was a branch and an `s_wait_loadcnt 0` per 32-value
  block, eight serialised round trips per row. Unconditional loads with a
  masked value fixed that; it was worth 40%.
- **`expf` was not free.** gfx12 has `v_exp_f32`; `exp2(x * log2(e))` is the
  same probability and removed a forty-instruction range reduction from the
  inner loop.

### Depth against the ceiling

Ceiling is `632.5 GB/s / (17.19 GB + depth * bytes_per_token)`, the same
arithmetic as the roofline table above, restated at the measured bandwidth.
"Measured" rows substitute the measured attention time into the step; f16 has no
fast hipml kernel (the f32 `attn_decode_f32` reaches 6.8 GB/s), so its attention
column is calculated, not measured, and is labelled as such.

| depth | dtype | KV GB | ceiling tok/s (calc) | attn ms/step | step ms | tok/s | share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | f16 | 0.537 | 35.68 | calc | 28.01 | 35.68 | calc |
| 8,192 | q8 | 0.285 | 36.16 | 2.01 | 29.21 | 34.24 | 6.9% |
| 8,192 | q4 | 0.151 | 36.39 | 2.06 | 29.26 | 34.18 | 7.1% |
| 32,768 | f16 | 2.148 | 32.73 | calc | 30.55 | 32.73 | calc |
| 32,768 | q8 | 1.141 | 34.54 | 7.43 | 34.63 | 28.88 | 21.4% |
| 32,768 | q4 | 0.604 | 35.51 | 7.52 | 34.72 | 28.80 | 21.7% |
| 100,000 | f16 | 6.554 | 26.68 | calc | 37.48 | 26.68 | calc |
| 100,000 | q8 | 3.482 | 30.62 | 21.87 | 49.07 | 20.38 | 44.6% |
| 100,000 | q4 | 1.843 | 33.21 | 22.25 | 49.45 | 20.22 | 45.0% |
| 131,072 | f16 | 8.389 | 24.63 | calc | 40.61 | 24.63 | calc |
| 131,072 | q8 | 4.564 | 29.06 | 28.28 | 55.48 | 18.03 | 51.0% |
| 131,072 | q4 | 2.416 | 32.34 | 28.75 | 55.95 | 17.89 | 51.4% |

**Quantization buys nothing until attention is fast enough to notice.** At 100K
the q8 cache halves KV bytes and the ceiling rises 3.94 tok/s, but the measured
step is 49.1 ms against 49.5 ms for q4 — the two formats are within 1% of each
other because the kernel is not reading-limited. The lever is only a lever once
the kernel is.

### VRAM

`hipml vram` calls `hipMemGetInfo`; on the benchmark box, with nothing else
resident, the card reports **31.86 GiB total, 31.71 GiB free**. Weights are
17.19 GB (16.01 GiB). f16 KV fits at 100K: 6.5536 GB of cache gives
23.7436 GB (22.11 GiB), leaving 8.1 GiB for activations and recurrent state.

| depth | f16 total | q8 total | q4 total |
| ---: | ---: | ---: | ---: |
| 100,000 | 23.74 GB / 22.11 GiB | 20.67 GB / 19.25 GiB | 19.03 GB / 17.73 GiB |
| 131,072 | 25.58 GB / 23.82 GiB | 21.75 GB / 20.26 GiB | 19.61 GB / 18.26 GiB |
| 262,144 | 34.37 GB / 32.01 GiB | 24.67 GB / 22.98 GiB | 20.88 GB / 19.44 GiB |

At 262,144 tokens f16 KV is exactly 16 GiB, so the total is 32.01 GiB against
31.86 GiB of card: it does not fit, by 0.15 GiB, before any activation. That is
the only depth in this table where quantization is a capacity requirement, and
it is 256K, not 100K.

### llama.cpp on the same box, 100K depth

`llama-server.exe` from `C:\llama.cpp\rocm-gfx120x`, `-m
C:\models\Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -c 102400 -np 1 --port 8092`, prompt
built from `C:\models\calib_longctx.txt` to 97,932 tokens, `n_predict 128`,
`temperature 0`, `cache_prompt false`, `ignore_eos true`.

| KV | prompt tokens | prefill s | prefill tok/s | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| f16 (default) | 97,932 | 115.91 | 844.9 | 21.83 |
| q8_0 / q8_0 | 97,932 | 118.02 | 829.8 | 22.32 |

q8 KV is **+2.3% decode** and **−1.8% prefill** at this depth. The traffic model
predicts ~11%: 3.07 GB less KV per token at 632 GB/s is 4.9 ms off a 45.8 ms
step. llama.cpp does not get it, which is consistent with the quantized K/V
being dequantized into the attention operand path rather than consumed in place.

The two numbers put the hipml attention kernel in context: 21.9 ms of attention
is 44.6% of a decode step, while llama.cpp's *entire* step at the same depth is
44.8 ms. Its attention is therefore a few milliseconds at most, and this kernel
is roughly four times more expensive than the one it would replace. That is the
gap to close before quantized KV matters.

## Method notes

- **Decimal GB, except VRAM.** Sizes are `10^9` bytes; `hipMemGetInfo` is quoted
  in GiB because that is what the driver returns.
- **KV bench cache is synthetic but format-valid.** `kv::synthetic_blocks` fills
  every 32-value block with a real f16 scale and a real payload, so the kernel's
  dequantize path runs at full cost without a 3.5 GB host encode per case. The
  100K *correctness* case is `hipml parity --deep`, which encodes a real cache
  and compares against the reference at 100,000 tokens.
- **Attention share** is `attn / (measured weight stream + attn)`, where the
  weight stream is the 838 MB `gemv` case's measured 632 GB/s applied to 17.19 GB.
  It excludes every other layer op, so it is an upper bound on what a real step
  would spend in attention.
- **Repeats.** `bench-kv` warms the card with the weight stream, then runs each
  row for at least 5 passes and enough passes to reach ~200 ms; the reported
  time is the mean of the timed passes after a warm pass.
- **llama.cpp** numbers are one request each on a freshly started server, so the
  prefill is a cold prefill at the stated depth and the decode rate is from the
  128 generated tokens of that same request.
- **Bandwidth is re-measured per run.** The weight stream reads 631.0–633.6 GB/s
  across runs on this box; each table quotes the value from its own run, so a
  tenth of a GB/s between sections is measurement scatter, not a different card.
- Bandwidth ceiling uses the measured 633.6 GB/s, not the 640 GB/s spec, so the
  percentages are not flattered.
- The fp16 peak of 95.7 TFLOPS is the vendor's dense figure; 70% of it under HIP
  is a realistic matmul efficiency, not a shortfall to chase.
- Neither ceiling accounts for KV-cache traffic or launch overhead, so no
  measurement should be expected to reach 100%.
- Numbers are `llama-bench` medians of three repetitions, same model file, same
  box, sequentially run.
