# PFT: what bounds the prefill GEMM tile, per weight format

Base `3401e1a` plus the GEMV dequant change from PR #9. Branch
`feat/prefill-tiles`. Box: R9700 (gfx1201, 64 CUs, 32 GB), Qwen3.8-27B
UD-Q4_K_M and stock Q4_K_M, chunk 2,048, f32 KV for prefill. All GPU work ran
under the `prefill-tiles` lock, port 8093.

Section 5.1 was measured later, on `89d0d8d`, branch `feat/prefill-pipeline`,
under the `prefill-pipe` lock on the same box, against a `89d0d8d` tree in a
second directory interleaved in the same hold. Nothing in the shipped kernel
changed; sections 0-4, 6 and 7 are the earlier hold's numbers and are restated
unchanged.

The measuring tool is `scripts/probes/probe_prefill_tiles.cpp`. It compiles
the shipped `prefill_wmma2.hip` (not a copy) and drives it with random bytes of
the right block layout, so the numbers are the tile's own cost: the decoder
runs, its arithmetic result does not matter. `-DPWW2_PROBE_*` switches remove
one half of the tile at a time; they are preprocessor-only and default off.

## 0. Question and answer

Question: what limits the prefill GEMM on UD-Q4_K_M, and does any tile change
raise it enough to beat llama.cpp's 1,354 tok/s at 8K?

Answer: the tile is pinned at **one block per CU**, so its **serialised
staging phase (worth 25-34% of every GEMM) cannot overlap the matrix work**.
The compute-only floor is 147-149 TFLOPS on gate/up and 167 TFLOPS on down; the
shipped tiles reach 97-114 TFLOPS. Nothing about the shape, the prefetch depth
or the IQ4 decoders can be changed to recover the gap, so this branch keeps no
kernel-behaviour change. The staging-free bound would be about 1,430 tok/s at
8K, and even removing all non-tile overhead only reaches about 1,300, so the
verdict is that tile tuning alone cannot beat 1,354.

## 1. The tile

`prefill_wmma2.hip` runs 256 tokens x 128 rows per block, 256 threads, 16
accumulators, fp16 activation and weight tiles in LDS, `KW` k-values per
iteration. Each `KW` step is

```
load_x / commit_x (activation -> fp16 LDS)
load_w / commit_w (weights -> decode -> fp16 LDS)
__syncthreads()
compute (PWW2_KW/16 WMMA rounds)
__syncthreads()
```

The activation is re-read once per row tile, so a ffn GEMM at 2,048 tokens
reads `m/128 * 2048*k*2` bytes of activation: **2.85 GB** for either ffn shape,
against 50 MB of Q4_K weights (178 MB if fp16). The activation stream is the
larger one by 16-57x, and it is served from the 64 MB Infinity Cache.

Measured occupancy is one block per CU: the five `_w2h` kernels compile to
181-211 VGPRs (no spills), and two blocks of 256 threads need at most 128
VGPRs/thread. There is no second resident block to run while the first one
sits at the staging barrier, so staging and math never overlap.

## 2. Shipped throughput per format

2,048 tokens, 3 reps averaged, mean of two interleaved rounds (ms / TFLOPS):

| format | gate/up 17408x5120 | down 5120x17408 |
|---|---:|---:|
| Q4_K | 3.204 / 113.9 | 3.358 / 108.7 |
| IQ4_XS | 3.424 / 106.6 | 3.543 / 103.0 |
| IQ4_NL | 3.533 / 103.3 | 3.650 / 100.0 |
| Q5_K | 3.672 / 99.4 | 3.767 / 96.9 |
| fp16 weights | 3.687 / 99.0 | 3.718 / 98.2 |

Two things follow. First, a 4-bit format is **faster than fp16 weights** in
the same tile: the decoder costs less than the doubled weight stream. Second,
the format spread is small (99-114 TFLOPS) while the *format-independent*
floor is much higher, so the decoder is not the limiter.

## 3. The ceiling, per shape

| configuration | gate/up | down |
|---|---:|---:|
| register-only WMMA issue (`probe_wmma_ceiling.cpp`) | 171-188 TF | 171-188 TF |
| this tile, all staging removed | 2.46-2.49 ms = 147-149 TF | 2.185 ms = 167 TF |
| this tile, shipped (IQ4_XS) | 3.424 ms = 106.6 TF | 3.543 ms = 103.0 TF |
| this tile, fp16 weights | 3.687 ms = 99.0 TF | 3.718 ms = 98.2 TF |

Gate/up runs 13-21% below the register-only ceiling even with staging removed;
down, whose k is 17,408 rather than 5,120, reaches 89-97%. That is a property of
the k-loop schedule, not of the format: every measured change to the schedule
lost.
The gap between the shipped and staging-free columns is the whole problem: it
is 0.8-1.4 ms per GEMM, not a few percent.

One cell is an outlier and is reported as measured: Q4_K gate/up with staging
removed read 2.804 ms (130 TF) in both rounds while the other four formats
read 2.46-2.49 ms. The staged Q4_K tile is the fastest of the five, so this is
an artefact of that build's allocation, not a property of the format.

## 4. Where the staging time goes

Single-switch runs at gate/up, ms, mean of two rounds. Each column removes one
thing from the shipped tile; the deltas do **not** add up, because the parts
overlap inside the same phase and removing one changes the register and LDS
pressure of the rest.

| format | shipped | no x copy | no weight copy | no decoder | no store |
|---|---:|---:|---:|---:|---:|
| Q4_K | 3.204 | 2.814 | 2.395 | 2.959 | 3.29 |
| IQ4_XS | 3.424 | 2.984 | 2.438 | 2.982 | 3.47 |
| Q5_K | 3.672 | 3.298 | 2.438 | 3.093 | 3.75 |
| IQ4_NL | 3.533 | 2.786 | 2.432 | 3.041 | 3.64 |

With the weights not staged, the same tile runs 2.43 ms for *every* format,
which is the format-blind part: activation copy + WMMA. The store is 0.03-0.04
ms (about 1%) and inside the noise; the guarded no-store build even perturbs
allocation on the fp16 path, so that row is an upper bound on the saving.

Attribution, as shares of the shipped time: weight staging 25-34%, the x copy
10-21%, the decoder ALU 8-16%. The activation copy is the
larger *stream*, but the weight staging is the larger *phase* because it also
converts and writes 128 rows per step against 256 tokens.

## 5. What was tried and reverted

Every experiment below was built and run interleaved with the shipped build in
the same lock hold, then reverted. Probe noise is +-3-5% between holds
(thermal and clock drift), so a variant had to win outside that band to be
kept; none did.

| change | result |
|---|---|
| tile 128x256, 512x64 | not faster at either ffn shape |
| tile 128x128 with WR32 so two blocks fit | not faster in the same hold |
| 512 threads / 8 accumulators | not faster; fewer rows per block doubles the activation stream |
| `KW=64` | slower (LDS footprint) |
| prefetch depth `PF=1` | slower; holds both tiles in registers, 61 KB LDS |
| prefetch depth `PF=2`, `PF=3` | no gain on the shipped shape |
| IQ4 16-value pair decode | IQ4_NL +-1%, IQ4_XS -9 to -14% |
| IQ4 codebook in shared memory | -8 to -9% |
| `vmems-to-lds` direct weight copy | unavailable on gfx1201 (`needs target feature vmem-to-lds-load-insts`) |
| token-major fast path (`PWW2_TOKFAST`) | invalid: grid mismatch, most blocks idle |

The accumulator budget is the reason the one promising lever cannot move.
Halving the activation re-read means doubling `RT` past 128, which needs 32
accumulators at this tile width; 32 accumulators do not fit at 256 threads
alongside the fp16 staging, and every shape that bought the rows back lost more
in compute than it saved in staging.

### 5.1 Hiding the staging phase: three mechanisms, all measured at a loss

Staging is 22-30% of every GEMM (section 4), and it cannot overlap the multiply
at one block per CU because all eight warps sit in the same phase and there are
only two warps per SIMD to hide a dependency chain. The three mechanisms that
could overlap it were built and measured. The answer is no, and the reason is
sharper than the earlier note's model: **the double buffer is a 21% tax before
any overlap is created**, so the best overlap mechanism would have to recover
more than that, and the best one recovers 5.6 points. The probe depths
`PWW2_PF=4` and `PWW2_PF=5` below are the kept, off-by-default residue, with
`PF=0` still the shipped depth.

**1. Direct-to-LDS global loads: not available on gfx1201.** The previous
entry said only that the `vmems-to-lds` target feature was missing. The
instruction itself settles it. `llvm-mc` for gfx1201 gives

```
global_load_lds_dword v[0:1], v2      error: instruction not supported on this GPU
global_load_lds_b32   v1, v[2:3]      error: invalid instruction
global_load_lds_b128  v[0:3], v4      error: invalid instruction
buffer_load_dword     v0, s[0:3], 0 offen lds   error: invalid operand
```

so the gfx9 `buffer_load ... lds` form does not assemble for this target and the
`global_load_lds_*` family exists in the assembler but is rejected for
`gfx1201`. There is no asynchronous global-to-shared path on this chip, and the
weight tile could not use it anyway: it is dequantised, not copied.

**2. Double-buffered staging through VGPRs: -21% before any overlap at all.**
The decisive measurement is `PF=1`, which is the plainest possible change from
the shipped loop: two buffers instead of one, and one barrier per k-tile
instead of two. It stages the next tile and then multiplies the current one. It
holds nothing across the multiply, so it costs 215 VGPRs against the shipped
191. It is 21% *slower*, and every mechanism below is measured against that
starting point. Means at `gate/up`, 2,048 tokens, 3 reps:

| build | VGPRs | LDS/block | Q4_K | IQ4_XS | fp16 weights |
|---|---:|---:|---:|---:|---:|
| `PF=0` shipped | 191 | 30,720 | 3.334 ms | 3.529 ms | 4.534 ms |
| `PF=1` two buffers | 215 | 61,440 | 4.036 (+21%) | 4.525 (+28%) | 4.033 (-11%) |
| `PF=2` fetch, commit after | 248 | 61,440 | 4.309 (+29%) | 4.508 (+28%) | 3.855 (-15%) |
| `PF=3` activation run only prefetched | 247 | 61,440 | 4.108 (+23%) | 4.331 (+23%) | 3.988 (-12%) |
| `PF=4` branch-free, sliced into the multiply | 231 | 61,440 | 4.161 (+25%) | 4.321 (+22%) | 3.987 (-12%) |
| `PF=5` split barrier | 213 | 61,440 | 3.808 (+14%) | 4.220 (+20%) | 3.811 (-16%) |

`PF=0` and `PF=4`/`PF=5` are three interleaved runs in one hold; `PF=1`-`PF=3`
are from the adjacent hold, whose own `PF=0` baseline read 3.315 / 3.528 /
4.495 (0.6% apart), so each percentage uses its hold's baseline. `PF=4` also
reaches 3.87 / 4.09 / 4.12 in the stage-late and fetch-early sub-variants; they
are within noise of the row above and are not kept.

The `PF=4` body is branch-free on purpose. With the `if (more)` guard the
scheduler splits the body into predicated blocks and sinks every prefetched
load down to the store that consumes it, which is why `PF=1..3` measured as the
serial loop; staging one extra tile into the buffer the loop leaves unused
removes the guard and the fetch then stays above the multiply (`clang -S`
verified). Removing the branch is worth 4 points of the 29 that `PF=2` loses.

Two facts bound the lever from here. First, the double buffer is a 21% tax
before any overlap is created, and the best overlap mechanism - the gfx12 split
barrier - buys back only 5.6 of those points. Second, the one format that
*gains* throughout is fp16 weights, which has no decoder: the quantised formats
carry a dependent `load -> ALU -> store` chain per staging chunk, and
interleaving it does not create overlap with two warps per SIMD, it moves the
chain's stalls into the compute phase. Holding the next tile also costs ~40
registers (191 -> 213-248), and the allocator pays for it out of the fragment
software pipeline. A `PF` where staging overlaps does exist for this tile - it
is the fp16-weight one, which is not a path the UD model's ffn uses.

**3. Split barrier: the best of them, and still -14 to -20%.** gfx12 splits
`s_barrier` into `s_barrier_signal` and `s_barrier_wait` (the plain barrier
lowers to that pair with id `-1`), so a warp that has finished its shared
stores starts multiplying while its neighbours are still storing, holding
nothing in registers: 213 VGPRs, between the shipped 191 and `PF=1`'s 215.
`PF=5` is exactly that around the `PF=1` order, and it recovers 5.6 points of
`PF=1`'s 21% loss: Q4_K 3.808 ms (+14%), IQ4_XS 4.220 (+20%), fp16 3.811
(-16%). It does not recover the rest because the eight warps are not skewed to
begin with - they do the same work in the same order, so the signal fires at
nearly the same instant in all of them, and the 21% tax on the double buffer is
paid regardless.

The producer/consumer split was not built: it needs the accumulator grid
covered by fewer warps, which is the `RT>128` problem one level down, and the
128x128 tile that makes it fit is the second row of this section's first table.


## 6. End to end

`hipml gpu-bench --gguf <UD> --prefill N`, chunk 2,048, f32 KV, three runs
each, same box and lock:

| prompt | `3401e1a`, earlier hold | final tree, this hold | interleaved: main / branch |
|---|---:|---:|---:|
| 512 | 1205.5 / 1206.2 / 1203.6 | 1190.8 / 1188.0 / 1188.7 | 1206.3 / 1210.8 |
| 8,192 | 1229.3 / 1222.6 / 1215.7 | 1210.0 / 1203.4 / 1196.9 | 1219.8 / 1218.2 |

The interleaved column is the one that compares code: `origin/main` and the
branch were built in separate trees and run alternately, three times each, in
one lock hold. The branch is +0.3% at 512 and -0.2% at 8,192, i.e. no
regression at either length. The earlier hold read 1.4-1.6% higher than this
one, and the interleaved run drifts about 1% downward across its six 8K cells:
that is clock state, not the patch.

Stock Q4_K_M at 8K on the same tree: 1222.5 tok/s. hipml is format-blind
between UD and stock, unlike llama.cpp (1,354 vs 1,170), which is why the UD
file is the hard comparison.

Stage split at 8K (UD), ms:

| stage | ms | share |
|---|---:|---:|
| ffn GEMMs (gate, up, down x 64) | 3310.1 | 47.6% |
| GDN chunk recurrence | 1478.5 | 21.2% |
| attention (16 layers) | 880.7 | 12.7% |
| GDN projection GEMMs | 820.4 | 11.8% |
| attention projection GEMMs | 224.9 | 3.2% |
| elementwise, embed, final | 209.2 | 3.0% |
| stage sum | 6923.8 | 99.5% |
| wall, including the split's syncs | 6960 | |

The shares are of the 6,960 ms wall. The ffn bucket includes two
`prefill_xhalf` passes and `silu_mul` (about 250 ms at 8K), so the pure ffn GEMM
rate is 91.6 TFLOPS in engine against 106.6 TFLOPS in the probe. The rest of
the gap to the probe (whole bucket 84.7 TF) is launch and scheduling, and it is
outside this tile: the tile probe cannot see it, and fixing it needs the
launch/accounting path, not a kernel change.

## 7. Verdict on 1,354 tok/s

| scenario | 8K time | tok/s |
|---|---:|---:|
| shipped, split run | 6.96 s | 1,178 |
| all non-tile overhead removed (*model*) | 6.31 s | 1,298 |
| staging-free tile, not reachable (*model*) | 5.72 s | 1,431 |
| llama.cpp reference | 6.05 s | 1,354 |

The first row is the split run because it is the only run that reports the
stage buckets; it adds one stream sync per stage and reads about 3% below the
plain run. Both model rows are arithmetic, not plans. The staging-free one
credits the ffn GEMMs with the measured compute-only rate (2.47/2.47/2.19 ms
per layer-chunk) and leaves every other stage untouched; the middle one credits
them with the probe's shipped rate (about 400 ms) and removes the 250 ms of
`xhalf`/`silu` that the ffn bucket also counts. The staging-free row clears
1,354; the middle row, which removes everything outside the tile, does not. It
was an open question whether the staging phase could be *hidden* rather than
removed; section 5.1 answers it for every mechanism gfx1201 offers, and the
answer is no: the best of them is still 14-20% slower than the shipped loop, so
the shipped depth stays `PF=0` and 1,354 is still 0.9 s of GEMM staging away.
The three-and-a-half seconds of non-GEMM time (GDN scan 21.5%, attention
12.5%) is not a tile problem at all, and the 18% gap between the probe GEMM rate
and the engine's is the launch/accounting path.

## 8. What is unverified

* The ceiling probes use random bytes, so they measure the decoder's cost, not
  its arithmetic; a format whose decoder is data-dependent (none here) could
  behave differently on real weights.
* One block per CU is inferred from VGPR counts and the 65,536-VGPR file, not
  from a hardware occupancy counter; WDDM does not expose one here.
* The end-to-end comparison is three runs per tree in one lock hold. Its
  interleaved delta in section 6 (+0.3% at 512, -0.2% at 8,192) is smaller than
  the drift across that hold (about 1%), so that PR's effect is measured as
  zero but is not bounded tighter than about 1%.
* The staging-free row multiplies the probe's rate by the counted GEMM
  launches (256 per ffn tensor at 8K). It assumes the 8K chunk decomposes into
  exactly 4 x 2,048 tiles, which is true for this fixture length.
* Attention and GDN numbers are copied from the shipped split, not re-measured
  per variant; no variant changed them.
* Section 5.1's 21% cost for the double buffer is measured, not explained. Two
  candidate causes remain: the LDS footprint doubling from 30,720 to 61,440
  bytes per block, and the extra buffer making the fragment reads and the
  staging stores contend for the same shared-memory banks. `PF=5` at 213 VGPRs
  still loses 14%, so plain register pressure does not account for it on its
  own.
* `PF=4` and `PF=5` are kept as probe depths and are compiled out of the
  shipped build. Their machine code has been checked to be irrelevant to it:
  the gfx1201 assembly of the whole kernel set at `PF=0` is instruction-for-
  instruction identical to `89d0d8d` (`clang -S` diff, only the per-TU
  `__hip_fatbin` GUID differs).

