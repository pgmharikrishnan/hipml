# Speed: measured against the roofline

> *Engineering log from development; internal work-package IDs and paths have been sanitized or removed where they were machine-specific. Current numbers: [docs/benchmarks.md](../benchmarks.md).*

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

Qwen3.8-27B Q4_K_M: 16.01 GiB, 26.90 B parameters,
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
sits at 70.6% of its own decode ceiling (16.0 of 22.7 tok/s) while Qwen3.8 and Swift
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

The same run's fp32 `gemv_f32` was **634.2-648.0 GB/s** (839 MB), so the reference
ceiling stays 631 GB/s. Repeat runs are within ~1% except Q6_K ffn_down, which
ranged 596-624 GB/s across three runs. The >100% rows are the 50-73 MB tensors
fitting partly in the 64 MB Infinity Cache across the 51 benchmark reps; the
decode stream below is the conservative number.

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
| kernel-only, 11 launches per chunk (`--reps 11`) | 16,964,915,200 | 605.2 | 35.68 |

Per type at `--reps 11`: Q4_K **613.3**, Q6_K **557.7**, Q8_0 **707.6** (repeat
runs: 604.3-605.2 GB/s total). The
first row is a hard lower bound for this harness (it syncs per 64 MiB chunk, not
per token); the second amortises launch/sync latency and is the comparable
number. Including all 17.192 GB of tensor bytes at the same bandwidth gives
**35.20 tok/s**; `llama.cpp` HIP measures **31.4 tok/s** on the same file.

### Commands

From the repo root, with `HIP_PATH` and `HIPML_GGUF` set (see
`docs/usage.md`):

```
set HIPML_ARCH=gfx1201
cargo build --release -p hipml-cli --features hip
target\release\hipml.exe parity
target\release\hipml.exe bench
target\release\hipml.exe decode-bench %HIPML_GGUF% -v
target\release\hipml.exe decode-bench %HIPML_GGUF% -v --reps 11
target\release\hipml.exe gguf %HIPML_GGUF%
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

## Prefill: tiled GEMM over quantized weights

Prefill is compute-bound. One 26.9 B-parameter token costs 53.8 GFLOP, so the
two arithmetic ceilings for this card are:

| Compute | Peak | Tokens/s ceiling | % of it that would match llama.cpp |
| --- | ---: | ---: | ---: |
| fp32 FMA (dense) | 47.8 TFLOPS | **888 tok/s** | 141% — impossible |
| fp16 FMA (dense) | 95.7 TFLOPS | **1779 tok/s** | 70.6% |

`llama.cpp` HIP measures pp512 **1256.5 tok/s**, which needs 67.6 TFLOPS. That
is above the fp32 peak, so an fp32 prefill cannot match it by construction, and
it needs the matrix unit — which is what WP17 added (below, and measured against
the instruction's own ceiling as well as the two vendor numbers).

### WP32: the occupancy shape, and the fp16 activation tile

The WP17 shape spends 115 VGPRs on 8 accumulators per warp and runs one
512-thread block per CU; the staging phase of a k-tile is serialized against the
multiply, so a third of the kernel is a memory-latency phase with only 16 warps
to hide it. `prefill_wmma2.hip` trades warp tile for reuse: **256 tokens × 128
rows per block, 256 threads, 64×64 per warp, 16 accumulators**. A 64×64 warp
tile is 4× the reuse of 32×64 (the dequant is paid once per 64 tokens instead of
32) and the block still owns 32768 output elements, which is exactly the
register file (128 f32 accumulators/thread + 62 for the rest, 190 VGPRs,
`__launch_bounds__(256, 1)`, no spills). **32768 outputs is the largest f32 tile
this register file can hold**, so the shape is at its own ceiling.

Because the shape is register-saturated, the activation tile is the one stream
that can still be halved: `prefill_xhalf` converts it once per projection and
the `_w2h` entries read fp16. The conversion is the same `floats2half2_rn` the
kernel already applies while staging, so the GEMM output is **bit-identical**
(`scripts/probe_wmma_cmp.cpp`: 0 differing bits over q4k/q6k/q8/half at
t=200/333/256 and 17408×5120).

#### Shape and pipeline sweep

`scripts/probe_wmma2.cpp`, M=2048, fp16 activation, `--reps 6` (µs includes one
cold launch). The shipped shape is the E0 row.

| lever | tile | threads | VGPR (spills) | q6k TFLOPS | q4k TFLOPS | verdict |
| --- | --- | ---: | --- | ---: | ---: | --- |
| v1 (shipped, WP17) | 256×128, 8 acc | 512 | 115 (0) | 66–76 | 68–76 | reference |
| E0 k-tile 32 | 256×128, 16 acc | 256 | 190 (0) | **95.0** | **108.9** | **kept, default** |
| E0 + 512 threads / 8 acc | 256×128, 8 acc | 512 | 173 (—) | does not launch (>128 VGPR) | | rejected |
| E0 k-tile 64 | 256×128, 16 acc | 256 | 216 (0) | 87.2 | 94.1 | −10%, rejected |
| E0 PF=1, double buffer | 256×128, 16 acc | 256 | 216 (0) | 79.7 | 87.8 | −16%, rejected |
| E0 PF=2, x+w in registers | 256×128, 16 acc | 256 | 256 (43) | 25 | — | rejected |
| E0 PF=3, x only in registers | 256×128, 16 acc | 256 | 256 (0) | 80.8 | 82.3 | −15%, rejected |
| F0 half-height | 256×64, 16 acc | 128 | 190 (0) | 80.2 | 87.5 | −18%, rejected |

PF=3 fits (the fp16 x array is 16 VGPRs, so it does not spill) and still loses:
the prefetched load is consumed right after the multiply, and with 8 warps there
is nothing left to cover that stall. The 512-thread variants cannot hold the
16-accumulator warp tile at all (128 VGPR cap). The half-height block fits two
blocks per CU but halves the reuse, which costs more than the extra block buys.

The staging/multiply split at E0, M=2048 Q6_K: 3781 µs base → 2345 µs with
`PWW2_PROBE_NO_STAGE` (**155.7 TFLOPS**), i.e. **staging 38%, multiply 62%**.
Removing the dequant ALU changes nothing, so the staging cost is load latency,
not arithmetic — and the same measurement says a perfect overlap would be worth
at most 1.6×. Every overlap attempt above lost to register pressure instead.

#### At the engine's actual tile (2048-token chunks)

| shape | tokens | blocks | waves | q6k TFLOPS | q4k TFLOPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17408×5120 (ffn up/down) | 512 | 272 | 8.5 | 81.7 | 94.8 |
| 17408×5120 | 2048 | 1088 | 34 | 95.0 | 108.9 |
| 17408×5120 | 8192 | 4352 | 136 | 94.0 | 108.5 |
| 5120×17408 (ffn gate) | 512 | 80 | 2.5 | 73.0 | 67.8 |
| 5120×17408 | 2048 | 320 | 10 | 95.7 | 107.0 |
| 5120×17408 | 8192 | 1280 | 40 | 69.1 | 106.2 |

The last row is a probe artifact, not an engine case: at 8192 tokens that
tensor's fp16 activation tile is 285 MB, which no longer fits L2 with the weight
stream, and the k=17408 Q6_K sweep falls to 69 TFLOPS. The engine chunks at
2048, where the same shape measures 95.7.

#### The fp16 activation copy, applied to every projection

`prefill_xhalf` + the `_w2h` entries now cover the ffn, the SSM projections
(`wqkv`, `wgate`, and `out`) and attention q/k/v. One conversion per projection,
shared by every output row tile of that projection. Whole-model effect at 8K:

| stage (8192 tokens) | v2, fp16 ffn only | v2 + ssm/attn fp16 |
| --- | ---: | ---: |
| gemm ffn | 3199 | 3226 |
| gemm ssm | 920 | **738** |
| gemm attn | 284 | **213** |
| seq ssm | 1519 | 1486 |
| seq attn | 814 | 845 |
| elementwise | 194 | 186 |
| total prefill | 6.83 s | **6.61 s** |

(The shipped v1 kernel measures 7.30 s at the same prompt; the stage ledger is
only kept on the v2 path.) The x-half path is bit-identical, so it is not an
accuracy trade.

#### Results

| metric | shipped | WP32 | target |
| --- | ---: | ---: | ---: |
| FFN GEMM, 8K shapes (Q4_K / Q6_K) | 68–76 / 66–76 TFLOPS | **108.9 / 95.0** | ≥100 |
| 512-token prefill | 1081 tok/s | **1181 tok/s** | ≥1300 |
| 8K TTFT | 7.30 s | **6.61 s** | ≤6.0 s |
| 32K prefill | 32.47 s (1009 tok/s) | **31.49 s** (1041 tok/s) | no regression |
| 100K prefill | 149.75 s (684 tok/s) | **140.30 s** (730 tok/s) | no regression |

Q4_K clears 100 TFLOPS; Q6_K sits at 95, which is 61% of the staging-free
ceiling and 54% of the register-only WMMA probe (174–188). The remaining TTFT
and 512-token gaps are not in this kernel any more: at 8K the ffn is 3226 ms of
6610, and `seq ssm` + `seq attn` (2331 ms) plus the residual/rmsnorm pass
(186 ms) are untouched by it. At 512 tokens the ffn GEMMs are wave-starved
(2.5 waves for `ffn_gate`), which is what split-K would address and this shape
cannot.

### WP19: the accumulator index, and what the loop actually costs

The WP17 kernel was computing **half of every GEMM**. The multiply wrote
`acc[i * PWW_ACC + j]` while the epilogue read `acc[i * PWW_NRG + j]`; with two
token tiles (16 tokens) and four row tiles per warp those disagree for `i = 1`,
the out-of-range stores were dead-code eliminated (`private_segment_fixed_size`
stayed 0, and the loop body assembled **8** `v_wmma` per k-tile instead of 16),
and every warp produced tokens 0–15 of its 32 and left 16–31 at zero. Parity
could not see it: every case used at most four tokens. The acceptance prompt is
five tokens, so it could not see it either.

The tell was the arithmetic: fixing the index doubles the MACs and the ffn
kernels got **only 19% slower** (1002.8 → 1204 µs at 17408×5120, M=512), i.e.
the extra 50% of matrix work hid entirely inside stalls. Two consequences:

- the WP17 figures were inflated. "91 TFLOPS" at M=512 was **45 real TFLOPS**,
  and the whole-model "79.74 TFLOPS / 1568 tok/s" was measured on a kernel that
  skipped half the MACs and wrote zeros for half the activation rows.
- the first-token id of the synthetic 512-token prompt returns to **489**, the
  value documented before WP17 (it read 122271 with the bug). The 20-id
  acceptance string is unchanged, because that prompt is five tokens.

`prefill_q4_k` and `prefill_q6_k` now have a **t=40** parity case (multi-token
coverage; a zeroed token row is O(1) and caught) and the suite is green.

#### Where the loop's time goes (Q6_K 17408×5120, M=512)

Probe switches in the kernel (`PWW_PROBE_NO_STAGE`, `PWW_PROBE_NO_FRAG`,
`PWW_PROBE_NO_COMPUTE`) split the k loop into its phases, via
`scripts/probe_wmma_gemm.cpp` (the kernel source, no engine, 20 launches):

| variant | µs | TFLOPS | what it says |
| --- | ---: | ---: | --- |
| base (index fixed) | 1204 | 75.8 | reference |
| no staging (WMMA + barriers only) | 694 | **131.5** | ceiling of this structure |
| fragments read once, staging kept | 1213 | 75.2 | LDS fragment reads are free |
| no matrix work | 92 | — | staging alone, ~8% (mostly DCE'd) |
| already-half weights (no dequant) | 1182 | 77.2 | the dequant ALU is ~2% |

So **time = staging + compute, serialized**: 510 µs of staging (42%) sits in a
phase that cannot overlap the multiply because the two `__syncthreads` per k-tile
align all 16 warps. The staging phase is not bytes-bound: halving the x stream
(x already fp16) changes nothing at M=512, and removing the dequant ALU changes
nothing. It is the load latency of a phase in which every warp of the only
resident block waits.

#### Levers tried, each measured alone

Same shape, M=512 unless noted; a lever is only kept if it beats the base:

| lever | µs | TFLOPS | verdict |
| --- | ---: | ---: | --- |
| K-tile 64 (half the barriers, 54 KB LDS) | 1433 | 63.7 | −16%, rejected |
| double buffer, stage-then-compute, 1 barrier | 1322 | 69.0 | −9%, rejected |
| touch-prefetch the next tile into L2 (1 load/thread) | 1674 | 54.5 | −28%, rejected |
| non-temporal weight loads (`TH_LOAD_NT`) | 3359 | 27.2 | −64%, rejected (per-byte NT kills coalescing) |
| 256 threads, 128×128 block | 1951 | 46.8 | −38%, rejected |
| 384 threads, 384×64 block | 2027 | 45.0 | −41%, rejected |
| 128×256 block (256 rows per dequant) | 1851 | 49.3 | −35%, rejected |
| x already fp16, M=512 | 1209 | 75.5 | neutral |
| x already fp16, M=2048 | 4102 | 89.0 | **+13%** |
| x already fp16, Q4_K M=2048 | 4120 | 88.6 | **+13%** (78.0 base) |
| x already fp16, Q8_0 M=2048 | 3774 | 96.7 | best measured |

The x-fp16 variant (`prefill_*_wh` + `prefill_xhalf`) halves the activation
re-read (`tokens × k × 4 × m/RT` bytes, 1.43 GB per launch at M=512) and gains
13% at M=2048; it is neutral at M=512, which is what the engine runs, so it is
not the default. The last two failures have one cause: overlapping staging with
the multiply needs the next tile's loads in flight in registers, and the
512-thread/8-accumulator shape has 126 of its 128 usable VGPRs already
committed. Register-prefetch builds land at 143–169 VGPR and do not launch.

The register-only instruction probe now also sweeps block size: 174–176 TFLOPS
at 4, 8, 16 and 32 warps per block, so occupancy is not the remaining gap either.

### Kernel: fp16 shared tile, matrix unit (WP17)

The shipped prefill kernel runs the inner product on the **matrix unit**:
`wmma_f32_16x16x16_f16` (the gfx12 wave32 form exposed by
`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`). One block owns **256 prompt
tokens × 128 output rows**, 512 threads, eight 16x16x16 accumulators per warp
(each warp: 32 tokens × 64 rows). The weight tile is dequantised into shared as
half exactly as before, the accumulator is f32, and the epilogue writes f32
straight to `y` — no shared transpose.

The fragment layouts were **measured, not assumed** (`scripts/probe_wmma_layout.cpp`
reads the accumulator back for A×I, for A = row indices and for B = column
indices; `scripts/probe_wmma_ceiling.cpp` is the register-only rate probe):

```
operands A/B: lane l, element e -> index l % 16, k = 8*(l / 16) + e
accumulator:  lane l, element e -> col l % 16, row 8*(l / 16) + e
```

The natural guess — adjacent lane pairs per row — is wrong for both operands and
for the accumulator, and a one-warp probe read through an assumed accumulator map
appeared to confirm it; that cost the first working GEMM its 15 TFLOPS. The tile
geometry is the other half of the win: with 32 tokens per block the shared weight
tile is dequantised once per 32 prompt tokens and the k loop is a chain of
dependent global loads, which measured 10–15 TFLOPS. 256 tokens per block makes
the dequant 8x cheaper per FLOP. A sweep over (tokens/block, rows/block,
tokens/warp, rows/warp) at the ffn shapes is what picked 256×128.

### Kernel: fp16 shared tile, dot2 accumulate (previous path)

One block owns **256 prompt tokens × 128 output rows**. The weight tile is
dequantised ONCE into shared memory (Q4_K/Q6_K/Q8_0, 32-value k-tiles) and
multiplied against all 256 tokens, which is the entire win over 256 GEMVs: the
dequantisation cost is paid once per 256 tokens instead of once per token.
Each of 512 threads holds 8 tokens × 8 rows = 64 f32 accumulators.

Two things changed from the first (WP13) kernel:

1. **Shared holds half, not float.** The x tile and the dequantised weight tile
   are `__half2`, so an LDS dword carries two k values and bytes per FMA halve.
2. **`v_dot2_f32_f16` in the inner loop** (`amd_mixed_dot`): two fp16 products
   per instruction with an f32 accumulator. Per 16 half2 k-steps a thread reads
   8 x-half2 + 8 w-half2 (64 B) and issues 64 dot2 (128 FMAs) — 4 FLOP/byte,
   which matches a 128 B/cycle LDS at the packed-fp16 rate.

A thread's tokens are strided by 32 (`tg + 32*tt`) and the shared row stride is
17 half2 (68 B), coprime with the 32 banks, so a warp reads 32 banks; weight
rows are a broadcast. 119 VGPRs, no spills, 58 KB LDS per block (17.4 KB x tile,
8.7 KB weight tile, 32 KB epilogue transpose — reusing the tile regions was not
possible, the static allocation is the sum).

The first version used a 4×8 f32 microtile (128 tokens × 64 rows) and measured
**10.1–11.0 TFLOPS**; the 8×8 f32 tile measured **15.1–16.9**; the fp16 tile
measures **21.2–27.3**, 1.6× the f32 kernel. At M=2048 the per-token cost is
flat, so the tile is compute-bound rather than launch-bound.

### What the ceiling actually is

The 47.8 / 95.7 TFLOPS figures are vendor dense numbers (WMMA-class for fp16).
A register-only throughput probe on this card — 8 independent accumulators,
256-thread blocks, no shared memory, loop kept alive with inline asm — measures:

| Instruction | Measured | % of the quoted peak |
| --- | ---: | ---: |
| `v_fma_f32` | 17.5 TFLOPS | 37% of 47.8 |
| `v_dot2_f32_f16` | 41.4 TFLOPS | 43% of 95.7 |
| `v_pk_fma_f16` | 41.1 TFLOPS | 43% of 95.7 |
| `wmma_f32_16x16x16_f16` | **171–188 TFLOPS** | 180–196% of 95.7 |

So **41.4 TFLOPS is the VALU ceiling** for the dot2 inner loop on this card, and
the dot2 GEMM at 27.3 TFLOPS was at **66% of it**. The matrix unit is a different
device entirely: the same probe shape (8 independent accumulators, 256-thread
blocks, no shared memory, 1024–4096 blocks) measures **171–188 TFLOPS** for
`wmma_f32_16x16x16_f16`, roughly twice the 95.7 TFLOPS the vendor quotes for
dense fp16. The quoted number is therefore not this instruction's ceiling, and a
GEMM cannot be scored against it as if it were. The ceilings this work is
measured against are:

| Ceiling | TFLOPS | how |
| --- | ---: | --- |
| quoted vendor fp16 dense | 95.7 | vendor spec, exceeded by the instruction itself |
| measured `v_dot2_f32_f16` VALU | 41.4 | register-only probe, WP15 |
| measured `wmma` matrix unit | 171–188 | register-only probe, `scripts/probe_wmma_ceiling.cpp` |

### Qwen3.8-27B ffn shapes

`hipml prefill-bench --reps 5`, 5 launches averaged per row after one warm-up.
Default is the fp16 kernels; `--f32` measures the previous path.

| Weight | Shape `[m,k]` | Type | M | wmma us | TFLOPS | % 95.7 | % 185 (measured) | WP17 (index bug) TFLOPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ffn_up | 17408×5120 | Q6_K | 512 | 1280.0 | 71.30 | 74.5% | 38.5% | 89.80 |
| ffn_up | 17408×5120 | Q6_K | 2048 | 4585.8 | 79.61 | 83.2% | 43.0% | 88.98 |
| ffn_gate | 5120×17408 | Q4_K | 512 | 1403.1 | 65.05 | 68.0% | 35.2% | 81.41 |
| ffn_gate | 5120×17408 | Q4_K | 2048 | 4666.6 | 78.23 | 81.7% | 42.3% | 108.66 |
| ffn_down | 17408×5120 | Q6_K | 512 | 1283.3 | 71.12 | 74.3% | 38.4% | 89.86 |
| ffn_down | 17408×5120 | Q6_K | 2048 | 4579.0 | 79.73 | 83.3% | 43.1% | 87.54 |

The last column is the same kernel before the WP19 index fix, i.e. doing half the
MACs: every row was inflated ~2×, and the ffn shapes are really 65–80 TFLOPS.
The M=512 `ffn_gate` row is the weakest: 2 token tiles × 40 row tiles = 80
blocks is 1.25 waves on 64 CUs, so the tail costs ~19% (65.05 vs 78.23 at 2048).

`tok/s-eq` (omitted) is M tokens through that one tensor per second; it is not
comparable to the whole-model ceilings. `--dot2` and `--f32` rerun the A/B paths.

### Whole model, one M=512 chunk

`hipml prefill-model Qwen3.8-27B-Q4_K_M.gguf --tokens 512` reads every quantized
tensor once in 64 MiB row-aligned chunks and times only the kernel launches
(file read and uploads excluded, same method as `decode-bench`). 399 tensors run:
Q4_K, Q6_K, Q8_0, plus `output.weight`; the token embedding is a lookup and is
not a GEMM. The two Q5_K `ffn_down` tensors are excluded from this harness (its
filter is Q4_K/Q6_K/Q8_0); in the graph they stream from a half copy made at
load time (`prefill_halfw_h`), described below.

| Pass | Tensors | Bytes | ms | TFLOP | TFLOPS | implied tok/s | % 95.7 | % 185 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cold (all blocks) | 399 | 16.250 GB | 419.2 | 26.0 | 62.10 | **1221.4** | 64.9% | 33.6% |
| warm (16 attention blocks skipped) | 289 | 12.444 GB | 313.3 | 20.1 | 64.20 | **1634.1** | 67.1% | 34.7% |

(WP17 measured 79.74 / 80.46 TFLOPS here, on a kernel that skipped half the
MACs; the 2.7× over the dot2 whole-model rate 22.91 is the honest multiple.)
The gap to the 175 TFLOPS instruction ceiling is the serialized staging phase
(see WP19 above), the dequant and the 80–680 block grids: the chunk is 512
tokens, so each weight tile is read 2× and small projections launch fewer blocks
than a wave.

The whole-model TFLOPS is below the single-tensor 80–109 because a 64 MiB chunk
is a few hundred blocks, not a full tensor, so launch and tail effects show up.

**The sum is compute-bound, not weight-read-bound.** 26.0 TFLOP at 79.74 TFLOPS
takes 326 ms; reading the same 16.25 GB at the measured 631 GB/s takes 25.8 ms.
Arithmetic intensity is 1602 FLOP/byte against an fp32 balance point of 76
FLOP/byte, so the weight stream is 21× from being the limit at M=512; at M=2048
it is 84×.

Two Q5_K tensors exist — `blk.51.ffn_down.weight` and `blk.55.ffn_down.weight`,
5120×17408, 61,276,160 B each — and had no GEMV when this was measured (WP16
added `gemv_q5_k`; the decode step used an f32 copy before that). They are
0.123 GB, 0.7% of the quantized bytes and 0.66% of model FLOPs. They are
reported, not silently dropped. WP20 measured the prefill GEMM's instruction
ceiling; the merged tree reads Q5_K directly with `gemv_q5_k`.

### Cached input: cold versus warm TTFT

The reference KV cache (`hipml-core/src/cache.rs`) is `[n_kv, capacity, head_dim]`
per layer with an append API, and `prefix_match` returns the common prefix of the
cached tokens and the prompt. The model has **16 full-attention blocks** (blk.3,
blk.7 … blk.63) and **48 SSM blocks**. Only the attention blocks have a
positional KV cache. The SSM blocks carry a fixed-size recurrent state that
cannot be sliced by prefix, so the warm pass re-runs all 48 of them for the whole
prompt. Nothing else is faked: the warm pass actually launches those GEMMs.

| Scenario | Cached tokens | TTFT | Implied tok/s | Speed-up |
| --- | ---: | ---: | ---: | ---: |
| cold, cache empty | 0 | 326.4 ms | 1568.5 | 1.00× |
| warm, same 512-token prompt | 512 | 250.0 ms | 2048.2 | **1.31×** |

The cache is 67.1 MB at 512 tokens (16 layers × 2 × 4 KV heads × 512 × 256 × 4 B).
Warm is 12.444 GB / 16.250 GB = 76.6% of the cold weight stream, i.e. the 48 SSM
blocks dominate and cap the reuse win at 1.38×. That harness caches only the
attention KV and re-runs every SSM block, so it is the lower bound, not what the
engine now does. Attention numerics are not measured in it either: the KV
contents are placeholders, and the claim is the prefix match plus the skipped
GEMM work.

### Cached prefill with SSM state (WP22)

The engine now snapshots the *whole* prefix state: the 48 SSM layers' conv window
and delta-net matrix, plus the 16 f32 attention layers' KV rows, keyed by the exact
token prefix. `hipml-core/src/prefix_cache.rs` holds the bounded LRU metadata;
`hipml-model/src/gpu/prefix.rs` copies the state device-to-device. A hit restores
and prefills only the unshared suffix, so the recurrent half is no longer
recomputed.

One entry at `N` tokens is **149.6 MiB of SSM state** (48 × 3.27 MB, independent
of `N`) + **0.125 MiB per token of f32 KV** (16 layers × 2 × 1024 values × 4 B) +
0.95 MiB logits. At 1,024 tokens that is 278.6 MiB; at 2,048, 406.6 MiB. The
cache is bounded in entries and bytes and evicts LRU, so the SSM term is what a
capacity budget is really buying.

`hipml cache-bench --gguf ... --len 2048 --shared 1024 --chunk 512` builds P and Q
from a real tokenized system prefix and two different questions, and an unrelated
R. Cold is a fresh state; warm-exact restores P's end snapshot and takes the
cached logits; warm-shared restores the shared boundary and prefills only Q's
suffix.

| case | cache | matched | computed | TTFT | vs cold | prefill skipped | ids |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| cold P (2048) | miss | 0 | 2048 | 1801.5 ms | 1.00× | 0% | — |
| **warm exact P** | hit | 2048 | 0 | **7.6 ms** | **236.7×** | **100%** | identical |
| cold Q (2048) | miss | 0 | 2048 | 1804.8 ms | 1.00× | 0% | — |
| **warm shared Q** (S=1024) | hit | 1024 | 1024 | **958.2 ms** | **1.88×** | **50%** | identical |
| cold R (2048) | miss | 0 | 2048 | 1811.0 ms | 1.00× | 0% | — |

The same pair with a 1,536-token shared prefix (75%):

| case | matched | computed | TTFT | vs cold | prefill skipped |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold Q (2048) | 0 | 2048 | 1811.2 ms | 1.00× | 0% |
| **warm shared Q** | 1536 | 512 | **495.1 ms** | **3.66×** | **75%** |

Restore alone (lookup + 128 device copies + argmax) is **5.3 ms at 512 tokens**
and **6.4–8.1 ms at 2,048**. The 20 generated ids are identical to the cold path
in every case; the exact repeat's first id comes from the cached logits, the
remaining 19 from the restored state. Exact repeat at 512 tokens measured
5.5 ms against 493.5 ms cold (89×).

The wall-time reduction is 2–3 points below the token fraction (46.9% vs 50%,
72.7% vs 75%). That is real, not noise: the tiles that remain are the *later*
tiles, whose causal attention runs over a longer context than the tiles that were
skipped, and the restore itself costs 6–8 ms. The work reduction is exactly the
shared fraction: 1,024 of 2,048 tokens are not computed.

The 3-request sequence with a 2-entry / 1 GiB bound: seed P (two entries, 685.1
MiB) → exact repeat **hit** @2048 → shared Q **hit** @1024 → unrelated R **miss**
(inserted, **1 eviction**). The evicted entry is P@2048, the LRU victim, so a
repeat of P afterwards is only a *partial* hit at 1,024 tokens. Counters:
3 hits, 1 miss, 1 eviction, 0 too-large. The f32 KV rows are the chunked-prefill
path; q8/q4 decode caches are not snapshot targets yet.

### Parity

Reference is `ops::gemm_q4_k/q6_k/q8_0` (dequantise a row, dot every token, same
k order as the kernel). `hipml parity` now compares the **shipped fp16 kernels**
(`HIPML_PREFILL_F32=1` switches back to the f32 rows). fp16 operands cost ~1e-3
relative: 2e-2 abs / 5e-2 rel for the single-tile cases, 1e-1 abs / 5e-1 rel for
the multi-token case (a near-zero output lands on one of 2,560 values and moves
the ratio by an order of magnitude) and 2.5e-1 abs / 5e-2 rel at 17408×5120,
where a near-zero output from 5120 products can move by ~0.1 (still 1e-3 of the
output scale). Every budget is far below a layout bug, which moves values by
O(1) on O(100) outputs.

| Case (fp16) | max_abs | max_rel |
| --- | ---: | ---: |
| `prefill_q4_k` 64×512, t=4 | 4.45e-3 | 1.02e-1 |
| `prefill_q6_k` 64×512, t=4 | 3.15e-2 | 4.44e-2 |
| `prefill_q8_0` 64×512, t=4 | 1.24e-3 | 7.23e-2 |
| `prefill_q4_k` 64×512, t=40 | 5.73e-3 | 4.51e-1 |
| `prefill_q6_k` 64×512, t=40 | 3.09e-2 | 1.28e-1 |
| `prefill_q6_k` 17408×5120, t=2 | 1.12e-1 | 2.40e1 |

The t=40 cases are the WP19 regression test: with the accumulator index wrong
they read O(1) maxima because 24 of the 40 token rows come back zero.

`hipml parity` → `46 passed, 0 failed, 1 skipped` (the host-only top-p case).
The f32 cases still pass at their original budgets when `HIPML_PREFILL_F32=1`.

### Commands

From the repo root, with `HIP_PATH` and `HIPML_GGUF` set (see
`docs/usage.md`):

```
set HIPML_ARCH=gfx1201
cargo build --release -p hipml-cli --features hip
target\release\hipml.exe parity
target\release\hipml.exe prefill-bench
target\release\hipml.exe prefill-model %HIPML_GGUF% --tokens 512
```
## Quantized KV at depth (measured)

The formats, the reference and the kernels are in `hipml-core/src/kv.rs`,
`ops::attn_decode_kv` and `kernels/attn_decode_kv.hip`. The block layouts are
ggml's, so a cache packed here is the same bytes `llama.cpp -ctk/-ctv` holds:

| dtype | block | bytes/value | Qwen3.8 bytes/token | 100K cache | 128K cache |
| --- | --- | ---: | ---: | ---: | ---: |
| f16 | — | 2.0 | 65,536 | 6.5536 GB | 8.3886 GB |
| q8_0 | 2 B f16 scale + 32 i8 = 34 B | 1.0625 | 34,816 | 3.4816 GB | 4.5639 GB |
| q4_0 | 2 B f16 scale + 16 B packed = 18 B | 0.5625 | 18,432 | 1.8432 GB | 2.4159 GB |

Qwen3.8 bytes/token = `2 (K,V) * 4 kv heads * 16 full-attention layers * 256`.
Parity is `Tol::DEFAULT` (abs 1e-4, rel 1e-3); the worst absolute difference
across the 14 quantized-attention cases, including the 100,000-token one, was
**1.6e-7**. Both sides read the same encoded bytes, so the comparison measures
accumulation order, not the format: the quantization error cancels.

### Decode attention: depth, one head and the whole step

`hipml bench-kv --depths 1024,8192,32768,100000 --reps 5`, gfx1201, one process
per row set. "Step" is one pass over 16 layers at that depth, i.e. the attention
work of one 100K decode token. `share` and `tok/s` count the measured weight
stream (632.0 GB/s, 27.20 ms for 17.19 GB) plus attention; they exclude every
other layer op, so they are an attention-only model of a decode step.

| dtype | heads | layers | depth | splits | µs/step | KV GiB | KV GB/s | share | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q8 | 1 | 1 | 1,024 | 16 | 72.1 | 0.001 | 7.7 | — | — |
| q8 | 1 | 1 | 8,192 | 128 | 85.5 | 0.004 | 52.1 | — | — |
| q8 | 1 | 1 | 32,768 | 256 | 125.1 | 0.017 | 142.5 | — | — |
| q8 | 1 | 1 | 100,000 | 256 | 368.0 | 0.051 | 147.8 | — | — |
| q8 | 1 | 1 | 131,072 | 256 | 265.6 | 0.066 | 268.5 | — | — |
| q8 | 24 | 16 | 1,024 | 16 | 501.1 | 0.033 | 71.1 | 1.8% | 36.15 |
| q8 | 24 | 16 | 8,192 | 64 | 1,228.2 | 0.266 | 232.2 | 4.3% | 35.23 |
| q8 | 24 | 16 | 32,768 | 64 | 4,211.8 | 1.062 | 270.9 | 13.4% | 31.88 |
| **q8** | **24** | **16** | **100,000** | **64** | **12,401.5** | **3.242** | **280.7** | **31.3%** | **25.28** |
| q8 | 24 | 16 | 131,072 | 64 | 16,115.0 | 4.250 | 283.2 | 37.2% | 23.11 |
| q4 | 1 | 1 | 1,024 | 16 | 40.2 | 0.000 | 7.3 | — | — |
| q4 | 1 | 1 | 8,192 | 128 | 37.9 | 0.002 | 62.3 | — | — |
| q4 | 1 | 1 | 32,768 | 256 | 176.6 | 0.009 | 53.4 | — | — |
| q4 | 1 | 1 | 100,000 | 256 | 325.7 | 0.027 | 88.4 | — | — |
| q4 | 1 | 1 | 131,072 | 256 | 388.5 | 0.035 | 97.2 | — | — |
| q4 | 24 | 16 | 1,024 | 16 | 284.5 | 0.018 | 66.3 | 1.0% | 36.44 |
| q4 | 24 | 16 | 8,192 | 64 | 1,427.1 | 0.141 | 105.8 | 5.0% | 34.98 |
| q4 | 24 | 16 | 32,768 | 64 | 4,995.4 | 0.562 | 120.9 | 15.5% | 31.10 |
| q4 | 24 | 16 | 100,000 | 64 | 14,615.1 | 1.717 | 126.1 | 35.0% | 23.94 |
| q4 | 24 | 16 | 131,072 | 64 | 18,983.9 | 2.250 | 127.3 | 41.1% | 21.67 |

The one-head rows are launch- and parallelism-bound (16 to 256 warps on a 32-CU
card) and are not a bandwidth statement; they are here because the brief asks
for them. The 24-head rows are the meaningful measurement.

**The 100K path now runs at 281 GB/s, 89% of the card's 632 GB/s stream.** q8 reads
3.242 GiB in 12.40 ms; the 24-head × 16-layer attention term is 12.4 ms of a
39.6 ms decode step, so a hipml decode models at 25.3 tok/s against llama.cpp
HIP's measured 21.86 tok/s at 97,387 tokens. q4 no longer wins on time: it reads
half the bytes in the same 14.6 ms, because the instruction count per cached
value is dtype-independent, so the smaller cache buys DRAM margin rather than
wall time (126 GB/s of unique traffic). The remaining gap to the stream is the
compute pass itself: an ablation with staging removed still measured 11.0 ms at
100K.

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
- **The loop was 243 instructions per cached value, not 67.** The disassembly
  showed 16 scalar shared loads plus address math, and a `__shfl_xor_sync`
  expansion of ~12 instructions per butterfly step — 60 of the 243. Giving each
  lane eight components as two `float4` chunks (LDS.128, conflict-free), folding
  the five butterfly steps into `__builtin_amdgcn_ds_bpermute` with the byte
  offsets hoisted, and moving the rescale to once per tile took the 100K step
  from 21.6 ms to 15.5 ms.
- **Staging, not the loop, was then the wall: 43 instructions per dequantized
  value.** Each value recomputed a 64-bit address and re-loaded the f16 scale,
  and a row-at-a-time loop left two loads in flight per thread (measured
  209 GB/s on a 632 GB/s card). Four values per 32-bit payload load, one
  LDS.128 store, pointer walks instead of scaled indices, and a whole tile's
  loads issued before the first is consumed got the same traffic to 281 GB/s.
- **The tile size is a dtype decision.** q8 stages twice the bytes per row and
  wants a 4-row tile: 8 KB per block, 8 blocks in the 64 KB of LDS per CU, 48
  warps. q4 fits an 8-row tile (16 KB, 24 warps) and does better with half the
  barriers. Same binary, `KV_TILE_Q8` / `KV_TILE_Q4`.

### Depth against the ceiling

Ceiling is `632.5 GB/s / (17.19 GB + depth * bytes_per_token)`, the same
arithmetic as the roofline table above, restated at the measured bandwidth.
"Measured" rows substitute the measured attention time into the step; f16 has no
fast hipml kernel (the f32 `attn_decode_f32` reaches 6.8 GB/s), so its attention
column is calculated, not measured, and is labelled as such.

| depth | dtype | KV GB | ceiling tok/s (calc) | attn ms/step | step ms | tok/s | share |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | f16 | 0.537 | 35.68 | calc | 28.01 | 35.68 | calc |
| 8,192 | q8 | 0.285 | 36.16 | 1.23 | 28.39 | 35.23 | 4.3% |
| 8,192 | q4 | 0.151 | 36.39 | 1.43 | 28.59 | 34.98 | 5.0% |
| 32,768 | f16 | 2.148 | 32.73 | calc | 30.55 | 32.73 | calc |
| 32,768 | q8 | 1.141 | 34.54 | 4.21 | 31.37 | 31.88 | 13.4% |
| 32,768 | q4 | 0.604 | 35.51 | 5.00 | 32.16 | 31.10 | 15.5% |
| 100,000 | f16 | 6.554 | 26.68 | calc | 37.48 | 26.68 | calc |
| 100,000 | q8 | 3.482 | 30.62 | 12.40 | 39.56 | 25.28 | 31.3% |
| 100,000 | q4 | 1.843 | 33.21 | 14.62 | 41.78 | 23.94 | 35.0% |
| 131,072 | f16 | 8.389 | 24.63 | calc | 40.61 | 24.63 | calc |
| 131,072 | q8 | 4.564 | 29.12 | 16.12 | 43.28 | 23.11 | 37.2% |
| 131,072 | q4 | 2.416 | 32.26 | 18.98 | 46.14 | 21.67 | 41.1% |

**Quantization is worth 1.34 tok/s at 100K, not the 2.6 the ceiling suggests.**
The q8 cache halves KV bytes, but the measured step is 39.6 ms against 41.8 ms
for q4: attention is no longer reading-limited at q8 either (281 of 632 GB/s),
so the smaller cache buys DRAM margin that the compute pass cannot spend. q4
only starts to win once the compute pass is cheaper than the byte savings.

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

`%LLAMA_SERVER%`, `-m %HIPML_GGUF% -ngl 99 -c 102400 -np 1 --port 8092`,
prompt built from `%HIPML_LONGCTX_CORPUS%` to 97,932 tokens, `n_predict 128`,
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

## Whole-model residency (WP9)

`hipml load --gguf %HIPML_GGUF%` uploads every tensor once,
in GGUF order, one `hipMalloc` per tensor, and never moves it. Quantized payloads
stay in their on-disk block layout; F32 tensors upload as raw little-endian f32.

| dtype | tensors | bytes | resident | note |
| --- | ---: | ---: | ---: | --- |
| F32 | 449 | 104,953,856 | 104,953,856 | supported |
| Q8_0 | 64 | 1,392,640,000 | 1,392,640,000 | supported |
| Q4_K | 257 | 11,449,958,400 | 11,449,958,400 | supported |
| Q5_K | 2 | 122,552,320 | 122,552,320 | supported (`gemv_q5_k`, WP16) |
| Q6_K | 79 | 4,122,316,800 | 4,122,316,800 | supported |
| **total** | **851** | **17,192,421,376** | **17,192,421,376** | 16.01 GiB |

`hipMemGetInfo` free fell by 17,169,842,176 B (15.99 GiB); the remaining 22.6 MB
is allocator bookkeeping across 851 allocations. Read+upload took 5.28 s
(3.26 GB/s host-to-device). A 20-tensor readback (`hipMemcpy` D2H, fresh file
read) matched byte-for-byte and by FNV-1a checksum, 20/20.

Embedding rows for token ids 0, 1, 1234, 100000 and 248319 matched the CPU
`dequant_q4_k` reference exactly, 5120/5120 values, `max_abs = 0`.

`lm_head` (`output.weight`, 248320x5120 Q6_K) against a fixed 5120-value vector:
**1.785 ms** steady state, **584.1 GB/s**, 248,320 logits. The first launch after
the 17 GB upload is cold at 5.839 ms (178.6 GB/s); `--reps 5` removes that. Top-5
for the fixed probe vector: 12771 (0.098146), 113470 (0.096665), 32625
(0.095551), 173544 (0.095195), 41011 (0.092445).
