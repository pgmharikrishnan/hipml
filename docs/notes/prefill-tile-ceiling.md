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

Section 9 was measured on `2631720`, branch `feat/pro-1491-ffn-fuse`, under the
`prefill-fuse` lock with the `2631720` tree built in a second directory and run
interleaved in the same hold. It changes the kernel again: the ffn's gate and up
projections become one launch (`kernels/prefill_fuse.inc`).

Sections 2.1, 3, 5.2, 5.3, 6 and 7 were re-measured on `913550f`, branch
`feat/pro-1491-prefill-instr`, under the `prefill-instr` lock, base `913550f`
built in a second directory and run interleaved in the same hold. This branch
does change the kernel: it shortens the staging phase at the instruction level
(section 5.2) and moves the ffn's SwiGLU into the `up` projection's epilogue
(section 5.3). Sections 2, 4 and 5 keep the earlier holds' numbers where they
still describe the same mechanism; every number that moved says which hold it
came from.

The measuring tool is `scripts/probes/probe_prefill_tiles.cpp`. It compiles
the shipped `prefill_wmma2.hip` (not a copy) and drives it with random bytes of
the right block layout, so the numbers are the tile's own cost: the decoder
runs, its arithmetic result does not matter. `-DPWW2_PROBE_*` switches remove
one half of the tile at a time; they are preprocessor-only and default off.

Section 10 was measured on `1473a9c` + this branch, under the `iu4-probe` lock.
It does **not** change the shipped kernel: it adds one default-off probe switch
(`PWW2_W8`) and two probes, and section 10.5 shows the shipped build's gfx1201
assembly is unchanged by it. Its subject is the two gfx1201 matrix instructions
the tile does not use.

## 0. Question and answer

Question: what limits the prefill GEMM on UD-Q4_K_M, and does any tile change
raise it enough to beat llama.cpp's 1,354 tok/s at 8K?

Answer: the tile is pinned at **one block per CU**, so its **serialised
staging phase (worth 25-34% of every GEMM) cannot overlap the matrix work**.
The compute-only floor is 147-149 TFLOPS on gate/up and 167 TFLOPS on down; the
shipped tiles reach 97-114 TFLOPS. Nothing about the shape, the prefetch depth
or the IQ4 decoders can be changed to recover the gap, and the staging-free
bound is 1,410-1,460 tok/s at 8K (section 7).

The phase cannot be *hidden*, but it can be made shorter, and that is what this
branch does: the staging body is cut to about half its instructions (section
5.2, staging 0.67-0.95 ms instead of 0.89-1.41 ms per GEMM) and the ffn's
SwiGLU moves into the `up` projection's epilogue (section 5.3). The two
together are **+9.9% end to end at 8K (1,186 -> 1,303 tok/s) and +11.6% at 512**,
which is the largest prefill win this note has recorded, and it still lands
**3.8% short of 1,354** (section 9 takes another 1.1 points of that, to 1,332,
by fusing the ffn's gate and up projections into one launch): what is left is
the compute-side gap above plus the launch/accounting inside the engine's ffn
bucket, not the staging phase.

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

### 2.1 This branch, interleaved with `913550f`

Same probe (`probe_prefill_tiles.exe 2048 3`), base `913550f` and the branch in
one lock hold, 3 reps each. `now` is `-DPWW2_PROBE_NO_W=1`, which removes the
weight staging and leaves the decoder: the gap to `now` is therefore the
staging phase's wall clock, and it is the quantity sections 5.2/5.3 cut.

| format | shape | `913550f` ms / TF | branch ms / TF | Δ | staging `913550f` | staging branch |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | gate/up | 3.323 / 109.9 | 3.039 / 120.2 | -8.6% | 0.885 ms | 0.669 ms |
| Q4_K | down | 3.437 / 106.2 | 3.097 / 117.9 | -9.9% | 0.958 ms | 0.665 ms |
| Q5_K | gate/up | 3.775 / 96.7 | 3.339 / 109.3 | -11.5% | 1.312 ms | 0.949 ms |
| Q5_K | down | 3.898 / 93.7 | 3.328 / 109.7 | -14.6% | 1.411 ms | 0.883 ms |
| IQ4_XS | gate/up | 3.524 / 103.6 | 3.194 / 114.3 | -9.4% | 1.062 ms | 0.810 ms |
| IQ4_XS | down | 3.622 / 100.8 | 3.260 / 112.0 | -10.0% | 1.141 ms | 0.832 ms |
| IQ4_NL | gate/up | 3.672 / 99.4 | 2.916 / 125.2 | -20.6% | 1.205 ms | 0.538 ms |
| IQ4_NL | down | 3.737 / 97.7 | 2.938 / 124.3 | -21.4% | 1.275 ms | 0.503 ms |
| fp16 weights | gate/up | 4.440 / 82.2 | 4.481 / 81.5 | +0.9% | 1.972 ms | 2.099 ms |
| fp16 weights | down | 3.905 / 93.5 | 3.912 / 93.3 | +0.2% | 1.434 ms | 1.467 ms |

Everything this branch touches moved 8.6-21.4%; fp16 weights, which has no
decoder and is not a path UD's ffn uses, did not move outside noise. This
hold's absolute numbers are 3-4% slower than section 2's earlier hold on the
same shipped kernel; the interleaved column is the one that compares code.

The staging-free column also moved (Q4_K gate/up 2.438 -> 2.370 ms) because
`now` still runs the decoder: the same instruction cut that shortens staging
shortens the decoder.

## 3. The ceiling, per shape

| configuration | gate/up | down |
|---|---:|---:|
| register-only WMMA issue (`probe_wmma_ceiling.cpp`) | 171-188 TF | 171-188 TF |
| this tile, all staging removed (`89d0d8d` hold) | 2.46-2.49 ms = 147-149 TF | 2.185 ms = 167 TF |
| this tile, shipped (IQ4_XS), `89d0d8d` hold | 3.424 ms = 106.6 TF | 3.543 ms = 103.0 TF |
| this tile, fp16 weights, `89d0d8d` hold | 3.687 ms = 99.0 TF | 3.718 ms = 98.2 TF |
| this tile, all staging removed, `913550f` hold | 2.44-2.47 ms = 148-150 TF | 2.46-2.49 ms = 147-148 TF |
| this tile, shipped (Q4_K), `913550f` hold | 3.323 ms = 109.9 TF | 3.437 ms = 106.2 TF |
| this tile, shipped (Q4_K), this branch | 3.039 ms = 120.2 TF | 3.097 ms = 117.9 TF |
| this tile, all staging removed, this branch | 2.37-2.39 ms = 153-154 TF | 2.43-2.45 ms = 149-150 TF |
| this tile, shipped (IQ4_XS), this branch | 3.194 ms = 114.3 TF | 3.260 ms = 112.0 TF |

The last five rows are the same hold as section 2.1, so they compare directly.
The branch closes the staging gap from 0.89-1.41 ms per GEMM to 0.50-0.95 ms.
What remains after that is the staging-free column itself, 149-154 TF against
the register-only 171-188 TF, and that residual is what the verdict in section
7 turns on.

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

## 5. What was tried

Every experiment in this section was built and run interleaved with the
shipped build in the same lock hold. Probe noise is +-3-5% between holds
(thermal and clock drift), so a variant had to win outside that band to be
kept. Section 5.1 is the mechanism that lost and was reverted; sections 5.2 and
5.3 are the two that won and ship.

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


### 5.2 The staging loop, cut at the instruction level (shipped)

Section 5.1 asked whether the phase can be hidden. It cannot. This is the other
direction: make it shorter. A k-tile stages 16 weight values per thread, and on
Q4_K the staging body compiled to about 400 gfx1201 instructions of which 72
were the decoder's own arithmetic - the rest was guards, per-value address
math, header loads and waits. Five changes in `pww2_loadraw` /
`pww2_dequant8`:

1. **Out-of-range rows are clamped, not predicated.** A token row past the
   prompt, or a weight row past `m`, is redirected onto the last valid row
   instead of being skipped. The store loop already masks those
   (`gt < tokens && r < m`), so the staged tile only has to be defined, not
   correct: every exec-mask save/restore and guard branch leaves the k loop,
   and no load can go out of bounds.
2. **The block header arrives with the run.** Each run's load pulls its block's
   header window in the fetch phase (16 B for Q4_K/Q5_K/Q3_K, 2-8 B for the
   rest) and the decode reads registers. The old decoder loaded the scale, the
   min and the two block scales one byte at a time *at decode time*, which put
   three more dependent global loads inside the phase.
3. **Byte-wise header arithmetic becomes one shift.** The Q4_K/Q5_K 6-bit
   scale/min split, Q3_K's sixteen 6-bit scales and Q6_K's block scale are
   register ops on the prefetched words.
4. **Nibble and high-bit extracts use `v_ubfe`**, one scalar shift per value
   instead of mask/compare/select; Q5_K's fifth bit becomes a 1-bit extract.
5. **Per-thread row pointers and k offsets are hoisted** out of the k loop.

The staging body's instruction count fell from 393 to 217 on Q4_K, 529 to 261 on
Q5_K and 600 to 345 on Q3_K (blocks `clang -S` attributes to the k loop).
Measured as wall clock through the `now` builds (section 2.1), the phase goes
0.885 -> 0.669 ms on Q4_K gate/up and 1.275 -> 0.503 ms on IQ4_NL down: -24% to
-61% of the phase depending on the format, and 8.6-21.4% of the whole GEMM.

The two high-bit extracts are where this went wrong first: the per-byte offset
was dropped, so values 1-3 and 5-7 of every run took the wrong fifth/high bits.
`hipml parity prefill --deep` reported it as five q5_k/q6_k failures with
absurd residuals while every other format passed, which is the argument for
running the parity suite first on any decoder change: a single-format spot
check of one value per run would have missed it.

### 5.3 The ffn SwiGLU in the `up` projection's epilogue (shipped)

The graph ran three launches between the ffn's `up` and `down` projections:
`silu_mul` (read gate and up, write act), the fp16 copy of act, then `down`.
The `up` projection already holds those rows when it stores them, so its
epilogue applies the SwiGLU itself: `y = silu(gate) * acc` and `yh` the fp16
copy - one extra read of gate and one fp16 write, instead of two full
read-passes and two writes. The arithmetic is `silu_mul_f32`'s expression
verbatim.

It is a separate entry per format (`prefill_*_w2h_silu`) because only the store
loop changes; the ffn call site keeps the three-launch sequence for every shape
that cannot carry it (split-K, a projection with no fp16 tile, the q5 copy
path).

Measured on one binary with the fusion forced off by a temporary build, 8K,
three reps, interleaved in the same hold: 1,269.5 tok/s off, 1,303.7 on,
**+2.7%**, i.e. 169 ms of the 8K wall. A temporary per-launch trace showed 64
fused launches per chunk against 64 layers, so the whole ffn is covered, not a
subset.

Correctness: `oracle-gate` on both the UD and the stock file against
`bench/fixtures/p2_math_169.ids` passes with ids identical to the CPU
reference, and the greedy run reproduces `[11751, 13, 198, 760, 6511, 314, 9564,
369, 19241, 13, 198, 760]`.

## 6. End to end

`hipml gpu-bench --gguf <UD> --prefill N`, chunk 2,048, f32 KV. `913550f` and
this branch were built in separate trees and run alternately, three reps each,
in one lock hold, in two separate processes (p1, p2) to catch a per-process
offset:

| prompt | process | `913550f` tok/s (3 runs) | this branch tok/s (3 runs) | Δ |
|---|---|---:|---:|---:|
| 512 | p1 | 1,176.6 / 1,160.0 / 1,150.4 | 1,310.3 / 1,294.7 / 1,282.5 | +11.5% |
| 512 | p2 | 1,141.3 / 1,133.1 / 1,133.3 | 1,272.4 / 1,268.0 / 1,266.9 | +11.7% |
| 8,192 | p1 | 1,200.1 / 1,190.7 / 1,187.1 | 1,313.6 / 1,308.0 / 1,302.7 | +9.7% |
| 8,192 | p2 | 1,181.9 / 1,180.2 / 1,177.7 | 1,297.8 / 1,296.7 / 1,293.8 | +9.8% |

The p2 process read 2-3% below p1 on *both* trees, and both drift down across
their three reps; the interleaved per-hold delta is the number that compares
code, and it is +11.6% at 512 and +9.8% at 8,192. Absolute levels differ from
the earlier holds in this note by up to 4% (this hold's `913550f` reads 1,186
tok/s at 8K where the `89d0d8d` hold read 1,210), so cross-hold absolute
numbers are not comparable.

Of the 8K gain, the staging cut is +7.0% and the fused epilogue +2.7%,
measured separately in one hold against the same tree (section 5.3).

Stock Q4_K_M at 8K read 1,222.5 tok/s on the `89d0d8d` hold and was not
re-measured here; hipml is format-blind between UD and stock, unlike llama.cpp
(1,354 vs 1,170), which is why the UD file is the hard comparison.

Stage split at 8K (UD), one run each, ms:

| stage | `913550f` | this branch | share (branch) |
|---|---:|---:|---:|
| ffn GEMMs (gate, up, down x 64) | 3359.7 | 2847.8 | 44.3% |
| GDN chunk recurrence | 1513.1 | 1507.0 | 23.4% |
| attention (16 layers) | 894.8 | 855.6 | 13.3% |
| GDN projection GEMMs | 831.2 | 756.1 | 11.8% |
| attention projection GEMMs | 227.9 | 205.8 | 3.2% |
| elementwise, embed, final | 211.8 | 218.3 | 3.4% |
| stage sum | 7038.5 | 6390.6 | 99.4% |
| wall, including the split's syncs | 7070 | 6430 | |

The ffn bucket falls 15.2% (512 ms): 169 ms of that is the fused epilogue
(section 5.3), the rest is the staging cut on the same three GEMMs. The
projection buckets fall 9-10% for the same reason - those projections are IQ4
and Q3_K, the formats that gained most in section 2.1. The recurrence and
attention buckets are not GEMMs this branch touches; they read 0.4-4.4% lower,
which is drift within a single-run comparison and is not attributed.

The bucket still runs at 11.1 ms per layer-chunk against 8.9-9.7 ms of probe
time for its three GEMMs at the formats it carries (Q5_K, 18 of the 64 `up`
projections, is the slowest of them), so 1.4-2.2 ms per layer-chunk is launch
and scheduling. That part is outside this tile: the tile probe cannot see it,
and removing it needs the launch/accounting path, not a kernel change.

## 7. Verdict on 1,354 tok/s

| scenario | 8K time | tok/s |
|---|---:|---:|
| `913550f`, this hold, plain run | 6.90 s | 1,186 |
| this branch, this hold, plain run | 6.29 s | 1,302 |
| ffn GEMMs at the staging-free rate (*model*) | 5.6-5.8 s | 1,410-1,460 |
| llama.cpp reference | 6.05 s | 1,354 |

The plain runs are section 6's means. The split run reads 1.2% below the plain
one on both trees (7.07 vs 6.90 s, 6.43 vs 6.29 s) because it adds one stream
sync per stage, so it is used for the buckets and not for this table.

The model row credits the 256 ffn GEMMs at 8K with the staging-free rate of
section 2.1's `now` builds - 2.37-2.45 ms instead of the 2.94-3.34 ms they ship
at - and leaves every other stage untouched: 0.5-0.7 s off the wall. It is
arithmetic, not a plan. It clears 1,354, and that is the honest measure of how
far away the target now is: one phase, not a redesign.

It is also not reachable from here. Section 5.1 shows the phase cannot be hidden
on this chip, and shrinking it further needs a tile that stages less, which is
the accumulator budget this note opened with: past 128 rows the fp16 staging
and 32 accumulators do not both fit, and every shape that bought the rows back
lost more in compute than it saved. What this tile could still do was spend
fewer instructions on the phase and stop paying a separate elementwise pass for
the ffn's SwiGLU; the branch does both, for 422 ms of the 8K wall.

The last 244 ms is not staging either. The staging-free column is 149-154 TF
against the register-only 171-188 TF, so a tile that staged nothing would still
sit 19% under the compute ceiling - the k-loop schedule gap section 3 records,
which every measured change to the schedule lost to. That gap, plus the 1.4-2.2
ms per layer-chunk by which the engine's ffn bucket exceeds the probe time of
its own GEMMs, is where the rest lives. Section 9 takes 83 ms of it (one launch
per layer less) and leaves the rest.

## 8. What is unverified

* The ceiling probes use random bytes, so they measure the decoder's cost, not
  its arithmetic; a format whose decoder is data-dependent (none here) could
  behave differently on real weights.
* One block per CU is inferred from VGPR counts and the 65,536-VGPR file, not
  from a hardware occupancy counter; WDDM does not expose one here.
* The end-to-end comparison is three runs per tree in one lock hold, in two
  processes. The interleaved delta is +11.6% at 512 and +9.8% at 8,192, against
  a 2-3% offset between the two processes and a downward drift of about 0.5-1%
  across each process's six runs, so the deltas are well outside the noise.
* The stage buckets in section 6 and the model row in section 7 are single runs
  each; the split run's per-stage syncs make it unsuitable for rate comparisons,
  and the bucket-to-bucket deltas (9-10% on the projection buckets) are larger
  than the drift a single run shows, but they are not averaged.
* The fused epilogue's +2.7% comes from one hold comparing two builds of the
  same tree (one with the fused path forced off). It is a single-shot switch,
  not a variant family.
* The staging-free row multiplies the probe's rate by the counted GEMM launches
  (256 per ffn tensor at 8K). It assumes the 8K chunk decomposes into exactly
  4 x 2,048 tiles, which is true for this fixture length.
* The `now` builds still run the decoder, so the "staging" they remove is the
  weight fetch and its stores, not the ALU. The phase times quoted in section
  2.1 are therefore the phase minus the decoder.
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
* Section 9's end-to-end deltas are 6 interleaved pairs at each length in one
  hold, all positive, spread +0.85% to +1.30% at 8K. The 2,423-token row is two
  clean reps per tree: another slice stole the GPU lock mid-hold (the lock's own
  45-minute staleness rule) and two branch reps read 1.93 s and 9.30 s while its
  process was running. The buckets in section 9.3 are one run each, as in
  section 6.
* Section 9's format-pair list is the pairs the UD mixes actually use: 23
  instantiations, the 9 same-format pairs plus 14 mismatched ones. A pair that
  is not in the list runs the two launches, so a new mix loses the fusion and
  nothing else. Whether *all* pairs would fit the register file is answered for
  the shapes, not per pair: the 23 shipped ones are 186-208 VGPRs, 0 spills.
* The fused launch's win is attributed to the launch count by elimination: the
  probe says its kernel time is within +-3% of the two launches at 2,048 tokens,
  and the engine's ffn bucket falls 83 ms for 64 launches fewer per chunk. The
  83 ms was not decomposed further (per-launch driver cost against grid ramp
  and drain).


## 9. Fusing the ffn's gate and up projections (shipped)

The ffn pair is two `[17408, 5120]` matrices over one activation tile, and §7
names two residues: the staging-free rate is 149-154 TF against the 171-188 TF
register-only ceiling, and the ffn bucket still carries 1.4-2.2 ms per
layer-chunk of launch and accounting. Fusing the pair attacks the second one
twice over: one launch instead of two, and the SwiGLU moves into that launch's
epilogue with both products in registers, so the gate tile is never stored or
re-read.

### 9.1 What the accumulator budget allows

A block's accumulators are `TT * RT` f32 values over 256 threads, and the shipped
tile is pinned by them: 256 x 128 = 32,768 floats = **128 VGPRs per thread** in a
256-VGPR file with one resident block. The pair needs one output element's gate
*and* up value in the same block, so a fused block holds both. Doubling the
accumulators at the shipped shape is **256 VGPRs before a single staging
register** - the entire file - so the full fusion does not fit at 128 rows per
matrix. That is the register accounting the card asked for, and it is why the
activation staging cannot be halved: the x stream is `m * tokens * k * 2 / RT`
bytes per matrix, so halving it needs `RT = 128` per matrix *and* the pair in one
block, i.e. the 256-VGPR shape.

Two shapes keep the budget at one shipped block's (`TT * 2 * RT == 32,768`):

| shape | accumulators | staged x | staged weights | LDS | x stream | weight sweeps | launches |
|---|---:|---:|---:|---:|---|---|---:|
| 256 tokens, 128 rows, 1 matrix (shipped) | 128 | 64 | 64 | 30,720 B | 1x | 8 | 2 |
| 128 tokens, 128 rows x 2 matrices | 128 | 8 | 32 | 30,720 B | 0.5x | 16 | 1 |
| 256 tokens, 64 rows x 2 (shipped fusion) | 128 | 16 | 16 | 30,720 B | 1x | 8 | 1 |

Registers are per thread, VGPRs. Both fused shapes compile to **186-208 VGPRs
(190 for Q4_K/Q4_K), 0 spill slots, 0 scratch** - the shipped `_w2h` tile is
181-211 - and both keep 30,720 B of LDS (one x tile plus two weight tiles), so
both still fit one block per CU. The 128/128 shape is also *shorter* in
registers; its cost is the weight stream, not an occupancy loss.

### 9.2 Probe, both shapes, 2,048 tokens

One fused launch against the two shipped ones over the same bytes, same hold
(`probe_prefill_tiles.cpp`, 3 reps). The pairs are the ones a UD dynamic quant
mixes: 42 of 65 layers pair the same format and the rest pair IQ4_XS with
Q4_K/Q5_K/Q3_K.

| gate+up pair | two launches ms | fused 256/64 ms | delta | fused 128/128 ms | delta |
|---|---:|---:|---:|---:|---:|
| Q4_K + Q4_K | 6.046 | 5.972 | -1.2% | 9.232 | +54.3% |
| Q5_K + Q5_K | 6.538 | 6.361 | -2.7% | 12.394 | +91.6% |
| IQ4_XS + IQ4_XS | 6.297 | 6.408 | +1.8% | 8.353 | +33.5% |
| IQ4_XS + Q4_K | 6.180 | 6.487 | +5.0% | 8.201 | +32.7% |
| Q4_K + Q5_K | 6.313 | 6.146 | -2.6% | 11.155 | +79.6% |
| IQ4_XS + Q5_K | 6.435 | 6.624 | +2.9% | 10.219 | +60.5% |
| Q4_K + Q6_K | 6.257 | 6.489 | +3.7% | 16.280 | +164.9% |

So the shape that halves the activation stream is 33-165% *slower*, and the shape
that only halves the launch count is within +-3% of the pair. The reason is the
other stream: at 128 tokens per block the weight matrix is swept 16 times per
chunk instead of 8, and a pair of 4-bit matrices (37-73 MB each) does not stay in
the 64 MB Infinity Cache, so the halved activation reads are paid for twice over
in weight reads. **The duplicated activation stage cannot be removed inside this
register file**, and the fused pass's gain has to come from the launch count.

One more thing this hold settled: the format tag has to be a template argument.
The first build took the pair as two runtime `int`s so one kernel served every
pair; the decoder's 9-way branch inside the staging loop cost **317 spill slots,
760 B of scratch and 256 VGPRs**, and the launch ran 101 ms against 5.7 ms. The
shipped kernel therefore instantiates the 23 pairs the supported mixes use and
falls back to the two launches for anything else.

### 9.3 End to end, interleaved with `2631720`

`hipml gpu-bench --gguf <UD> --prefill N`, chunk 2,048, f32 KV. Both trees built
in separate directories, three reps each in two processes, alternating within one
lock hold:

| prompt | process | `2631720` tok/s | this branch tok/s | delta |
|---|---|---:|---:|---:|
| 512 | p1 | 1300.5 / 1300.3 / 1299.7 | 1329.4 / 1330.4 / 1329.2 | +2.2% |
| 512 | p2 | 1301.5 / 1299.9 / 1299.5 | 1330.5 / 1328.5 / 1327.5 | +2.2% |
| 8,192 | p1 | 1331.3 / 1319.9 / 1311.6 | 1345.2 / 1331.1 / 1327.7 | +1.1% |
| 8,192 | p2 | 1312.4 / 1312.3 / 1312.2 | 1329.5 / 1328.5 / 1327.8 | +1.2% |

All six interleaved pairs are positive: +0.85% to +1.30% at 8K (mean +1.14%)
and +2.2% at 512. Both trees drift down within a process (the first run of each
is the fastest), which is why the per-pair delta, not the absolute level, is the
number to read.

Launch count per 2,048-token chunk (64 layers): the ffn's gate/up pair goes from
2 launches to 1, so the ffn is **192 -> 128 launches** per chunk and 768 -> 512
at 8K. A temporary per-launch trace at 8K counted **256 fused launches = 4 chunks
x 64 layers**, so every layer's pair is fused at the 2,048-token chunk; the
one-token tail chunk keeps the two-launch path (the split-K predicate owns that
shape, and the fused epilogue has no partial-sum form).

Stage split at 8K, one run each, ms:

| stage | `2631720` | this branch |
|---|---:|---:|
| ffn GEMMs | 2808.3 | 2725.3 |
| GDN chunk recurrence | 1471.0 | 1485.1 |
| attention | 838.3 | 845.0 |
| GDN projection GEMMs | 746.8 | 749.5 |
| attention projection GEMMs | 203.4 | 204.3 |
| elementwise, embed, final | 212.4 | 208.9 |
| wall | 6.31 s | 6.25 s |

The ffn bucket falls **83.0 ms (-3.0%)**, from 10.97 to 10.65 ms per
layer-chunk; the other buckets move by single-run drift. That 83 ms is 1.3% of
the wall and the whole of the gain, and it is the launch/accounting part of the
1.4-2.2 ms per layer-chunk §6 identified: the probe says the fused tile's own
kernel time is not better.

### 9.4 A 2,423-token prompt

`--prefill 2423`: 1.85-1.87 s on `2631720` against 1.82-1.83 s on the branch, so
the prefill that the serve TTFT is made of drops by the same 1-2%. Two reps of
the branch read 1.93 s and 9.30 s in the same hold because another slice stole
the GPU lock (the lock's 45-minute staleness rule) and its process was running;
those two are contamination, not the tree. This is the `gpu-bench` prompt, not
the serve prompt the card named, so it is not comparable to llama's 2.06 s.

### 9.5 Correctness

| gate | result |
| --- | --- |
| greedy 12 ids, UD | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]`, byte-identical |
| `parity prefill --deep` | 15 passed (12 + 3 new fused cases), 0 failed |
| `parity --deep` | 74 passed, 0 failed, 1 skipped |
| `oracle-gate` UD, 20 steps | PASS 20/20, prefill max abs **0.1086** |
| `oracle-gate` stock Q4_K_M, 20 steps | PASS 20/20, prefill max abs **0.1481** |
| `cargo test --workspace`, `cargo fmt --check` | clean |

The two `oracle-gate` maxima are the *same values* the previous slice measured
(0.1086 and 0.1481, §6 of that report), i.e. the fused pass reproduces the
two-launch logits to the last printed digit rather than to a new tolerance. The
new parity cases (`prefill_gu q4_k/q4_k`, `q4_k/q5_k`, `iq4_xs/iq4_xs`, 256x512,
8 tokens) compare the fused activation against the two host GEMMs plus
`silu_mul_f32`'s expression, and the fp16 copy against the f32 one; their
absolute budget is 5e-3 of the tile's largest value because the case multiplies
two reductions, and the observed errors are 4-5e-4 of it.

### 9.6 Verdict

The fusion is real but small: **+1.14% at 8K (1,317 -> 1,332) and +2.2% at 512**,
with the ffn bucket down 83 ms. It does not reach 1,360, and it leaves 1.6%
(1,332 against llama's 1,354) on the table. The half of the card's premise that
promised more - halving the duplicated activation staging - is not reachable
here: it needs the pair's accumulators in one block at 128 rows each, which is
256 VGPRs of accumulators alone, and the shape that trades rows for tokens to
get it loses 33-165% because it doubles the weight sweeps. What is left after
this slice is the same residue §7 named, minus one launch per layer: the
compute-side gap (149-154 TF against 171-188 TF), and roughly 1 ms per
layer-chunk of launch/accounting that one launch per pair did not remove.

## 10. The gfx1201 int4 and fp8 matrix instructions (probe, not shipped)

Sections 1-9 treat the staging phase as bytes moved into LDS and then
dequantised to fp16. Two gfx1201 matrix instructions the tile does not use take
*narrower* operands, which is the one thing that would cut those bytes rather
than hide them: `v_wmma_f32_16x16x16_fp8_fp8` halves them and
`v_wmma_i32_16x16x32_iu4` quarters them per FLOP. Both are native on this chip
(`llvm-mc -mcpu=gfx1201`, AMD LLVM 22.0.0git, HIP 7.2.53211) and both were built
into the shipped tile. **They are faster, the staging phase does not move, two
blocks per CU still do not fit, and neither can be used without changing the
tokens.** This section is the measurement of each of those.

### 10.1 What the ISA actually gives

| instruction | gfx1201 | work | operand bytes/lane | FLOPs |
|---|---|---:|---:|---:|
| `v_wmma_f32_16x16x16_f16` (shipped) | native | 16x16x16 | A 16, B 16 | 8,192 |
| `v_wmma_f32_16x16x16_fp8_fp8` | native | 16x16x16 | A 8, B 8 | 8,192 |
| `v_wmma_i32_16x16x32_iu4` | native | 16x16x32 | A 8, B 8 | 16,384 |

The four fp8 pairings (`fp8_fp8`, `fp8_bf8`, `bf8_fp8`, `bf8_bf8`) and both iu4
shapes (`16x16x32`, `16x16x16`) assemble; there is no `16x16x32` fp8 form, and
`v_wmma_f32_32x16x128_f4` is rejected as `instruction not supported on this
GPU` (gfx1250 only). There is no `rocWMMA` in this SDK and no builtin for the
fp8 form, so the tile emits its encoding directly through inline asm; the iu4
form does have one,
`__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(bool negA, int2 a, bool negB,
int2 b, int8 acc, bool clamp)`.

Two consequences decide the rest of the section, and both follow from the
operand widths rather than from any measurement:

* **There is no mixed fp8 x fp16 or i4 x f16 form.** The instruction takes both
  operands in the narrow type, so an fp8 weight tile forces an fp8 activation
  tile, and an int4 weight tile forces an int4 activation tile. The activation
  is the high-precision side of the GEMM and is not quantised today.
* **iu4 accumulates in int32 and its scale changes every 32 k values**, which
  is exactly one `KW` step. The int32 result therefore cannot be carried across
  the k loop and scaled once; each 16x16x32 result has to be converted and
  folded into an f32 accumulator per k-tile, and Q4_K's `-dmin*mn` term needs a
  separate `sum(x)` correction per block.

### 10.2 Issue rate

Register-only, 8 independent accumulators per warp, no LDS traffic
(`scripts/probes/probe_wmma_int4_fp8.cpp`, 1024 blocks x 4096 iterations, 6
reps; the first rep of each row is the clock ramp):

| instruction | TFLOPS |
|---|---:|
| fp16 16x16x16 | 171.3 - 182.3 |
| fp8 16x16x16 | 351.1 - 382.5 |
| iu4 16x16x32 | 627.3 - 740.7 |

So the matrix pipe retires the fp8 form at **2.0x** and the int4 form at
**3.7-4.2x** the fp16 form's FLOPs. The fp16 row reproduces the 171-188 TF
ceiling section 3 quotes, which is what makes the other two rows comparable.

One row of that probe is reported as invalid rather than as a result: a fourth
kernel that adds the per-block int-to-float rescale reads *faster* than plain
iu4 (764-803 TF) because resetting the int32 accumulator each iteration breaks
the dependency chain the other rows carry. It measures the chain, not the
epilogue, and it is not used for anything below.

### 10.3 The fp8 tile at the real shapes

`PWW2_W8=1` stages both LDS tiles as OCP fp8 (E4M3) and multiplies with
`v_wmma_f32_16x16x16_fp8_fp8`; `PWW2_W8=0` is the shipped fp16 tile. Same
source, same decoder, same k order, same 256x128 shape, so the only difference
is the staged type and the multiply. Interleaved in one lock hold, 3
invocations of 3 reps each, 2048 tokens, Q4_K:

| shape | fp16 ms / TFLOPS | fp8 ms / TFLOPS | delta |
|---|---:|---:|---:|
| gate/up 17408x5120 | 3.056 / 3.101 / 3.020 — 119.4 / 117.7 / 120.9 | 2.661 / 2.689 / 2.670 — 137.2 / 135.8 / 136.8 | **+14.4%** |
| down 5120x17408 | 3.107 / 3.083 / 3.094 — 117.5 / 118.4 / 118.0 | 2.843 / 2.839 / 2.842 — 128.4 / 128.6 / 128.5 | **+8.9%** |

The deltas are the interleaved comparison and are outside the +-3-5% between-hold
drift. This hold's fp16 tile reads 119.4 / 118.0 TF where section 2's earlier
hold read 113.9 / 108.7; cross-hold absolute numbers are not comparable, and the
card's "current tile" figure is the 113.9 / 108.7 pair.

Register and LDS accounting, from the compiled gfx1201 assembly of the same
probe (`amdhsa_group_segment_fixed_size`, `num_vgpr` on the nine `_w2h`
entries, which are the ones the engine launches):

| build | LDS/block | VGPRs, `_w2h` set |
|---|---:|---:|
| shipped fp16 | 30,720 B | 183 - 193 |
| fp8 | **15,360 B** | **164 - 200** |

The staged footprint halves exactly. The register count is mostly lower - eight
of the nine entries drop, by 2 to 27, median 191 -> 171 - but not uniformly:
`iq4_xs` rises 192 -> 200, because its decoder already held its register budget
and the fp8 conversion adds to it. Register pressure was never the binding
constraint here, so the rise costs nothing.

A second A/B after the switch was scoped to the `_w2h` entries reproduced the
table: fp8 gate/up 134.0 / 137.0 TF against fp16 119.9 / 120.5, down 127.5 /
128.7 against 117.5 / 117.6, same hold. The 3-invocation means above are the
sample used for the deltas.

### 10.4 Where the win came from, and why staging did not move

This is the part that contradicts the hypothesis this section opened with.
`PWW2_PROBE_NO_W=1` removes the weight fetch, the decoder and the weight store;
the difference from the shipped build is the weight-staging phase's wall clock.

| shape | build | shipped ms | no weight staging ms | staging phase |
|---|---|---:|---:|---:|
| gate/up | fp16 | 3.059 | 2.389 | 0.670 ms |
| gate/up | fp8 | 2.673 | 2.019 | **0.654 ms** |
| down | fp16 | 3.095 | 2.436 | 0.659 ms |
| down | fp8 | 2.841 | 2.075 | **0.766 ms** |

**Halving the staged bytes did not shorten the staging phase.** It is flat on
gate/up (-2.4%, inside noise) and 16% *worse* on down. The whole gain is on the
other side of the barrier: the staging-free column goes 2.389 -> 2.019 ms on
gate/up and 2.436 -> 2.075 on down, -16% and -15%, which is the fp8 multiply
retiring the same 16x16x16 tile at twice the issue rate over half the LDS
fragment bytes.

Two mechanisms explain the flat phase. The weight half of it is a dependent
`load -> decode -> convert -> store` chain, not a store-bandwidth problem, so
the bytes the store writes are not what the phase waits on; that is the same
finding section 5.1 reached from the other direction when a 21% tax appeared
before any overlap was created. The activation half gets *worse* under fp8: the
shipped tile copies raw fp16 bytes into LDS with no conversion at all, and the
fp8 tile has to convert each of them, which is why the down shape - whose k is
17,408 rather than 5,120, so its activation stream is 3.4x larger - is the one
that pays.

### 10.5 Two blocks per CU: still no

LDS is no longer the obstacle: 15,360 B per block means two blocks fit the
64 KB budget with room to spare. The accumulator budget is, and it is
arithmetic rather than a tuning result. A block's accumulators are `TT * RT`
f32 values over `PWW2_THREADS` threads, so the shipped 256x128 shape is
256*128/256 = **128 f32 per thread = 128 VGPRs**, and two resident blocks of
256 threads get 65,536/512 = **128 VGPRs per thread** in total. The accumulators
alone consume the entire two-block allowance before a single staging or address
register, whatever the staged type is. The fp8 build's 164-200 VGPRs is already
a lower register count than 183-193 on eight of nine entries and it is nowhere
near 128.

The shapes that do fit two blocks are the ones section 5 already measured and
lost with: 128x128 has 64 accumulators, and it was `not faster in the same hold`
because halving the rows per block doubles the weight sweeps. Nothing about the
fp8 instruction changes that trade, and it is moot for the reason in 10.6.

### 10.6 Accuracy: the instruction is unusable, not merely lossy

`scripts/probes/probe_prefill_fp8_acc.cpp` runs the shipped tile and the fp8
tile over identical bytes and compares both against an f64 host reference built
from the exact dequantised weights and the exact fp16 activation values the
kernel reads. `kind 3` is fp16 weights (no decoder, so the residual is purely
the staged representation); `kind 0` is Q4_K blocks with the model's scale
range. 3 reps per cell, one lock hold.

| weights | m x k | tokens | fp16 max abs / rel | fp8 max abs / rel |
|---|---|---:|---:|---:|
| fp16 | 512 x 5120 | 8 | 1.57e-5 / 1.5e-6 | 0.376 / **3.7e-2** |
| fp16 | 512 x 5120 | 64 | 2.86e-5 / 2.2e-6 | 0.464 / **3.5e-2** |
| fp16 | 256 x 17408 | 8 | 2.45e-4 / 1.2e-5 | 0.676 / **3.4e-2** |
| Q4_K | 512 x 5120 | 8 | 9.39e-2 / 2.0e-4 | 17.2 / **3.7e-2** |
| Q4_K | 512 x 5120 | 64 | 0.120 / 1.9e-4 | 19.7 / **3.1e-2** |
| Q4_K | 256 x 17408 | 8 | 0.171 / 1.9e-4 | 28.1 / **3.2e-2** |

The fp16 tile reproduces the host GEMM to 2e-4 relative on Q4_K and 1.5e-6 on
fp16 weights, which is what validates the reference. The fp8 tile is **3-4%
relative on every cell, 150-2000x the fp16 tile's residual**, and it is the same
3.5% with the decoder removed (`kind 3`), so it is the E4M3 *representation* and
not the quantised format's interaction with it. 3 mantissa bits is the precision
floor; a per-block scale moves *range*, and range is not the problem - the Q4_K
values land between roughly 0 and 15 with the bulk around 1, well inside E4M3's
2^-9..448 - so no scaling recovers the mantissa.

For scale: `oracle-gate`'s whole-model prefill budget on this fixture is
`max |gpu - cpu| = 0.1086` over 64 layers. One GEMM at 3.2-3.7% relative (max abs
17-28 on a matrix whose largest output is 471-888) is orders of magnitude past
that, so the greedy stream cannot survive it. The card's rule applies: a fast
path that changes tokens is a different product decision, and it was not taken.

The int4 route is worse on both counts, so it was not built into the tile:

* its activation operand must be int4 (10.1), and per-32-block symmetric 4-bit
  quantisation of both operands costs **7.4-10.8% relative** on the same data
  and against the same reference (`probe_prefill_fp8_acc.cpp` reports it beside
  every fp8 cell; 8.0-10.8% on the Q4_K cells) - two to three times the fp8
  error, and that is the best case, before the per-block int32 rescale and the
  `sum(x)` correction the real tile would also carry;
* the per-32-block int32 rescale plus Q4_K's `sum(x)` correction is extra
  structure the fp16 tile does not have, for an instruction whose issue rate is
  3.7-4.2x fp16's but whose *tile* would still be bound by the same staging
  phase that did not move in 10.4.

Its issue rate is reported above because the card asked for it; it is not a
candidate.

### 10.7 Verdict, and what it means for the 1,500 target

Three of the card's stop conditions are met:

| question | answer |
|---|---|
| does either instruction beat the current tile's TFLOPS? | **yes** — fp8 +14.4% / +8.9% at the real shapes, interleaved |
| do two blocks per CU fit at <=128 VGPRs? | **no** — the 128 accumulator VGPRs are the whole two-block allowance |
| does the accuracy cost change the output stream? | **yes** — 3.2-3.7% relative per GEMM against 2e-4 |

The model for what it would have been worth, using section 6/9's bucket shares
and this hold's probe rates: the ffn's three GEMMs go from 9.19 to 8.18 ms per
layer-chunk, i.e. 1.01 ms of the 10.65 ms bucket, 4.2% of the 8K wall; the
projection buckets at a similar fraction add about 1.3%; so **about 5.5%, or
1,335 -> ~1,408 tok/s**. That clears llama.cpp's 1,354 and misses the card's
1,500, so even accepting a token-changing path would not have bought the
target. It is arithmetic, not a run: the path was not taken to end to end.

What the two instructions did establish is narrower and still useful. The
staging phase is not a byte-count problem - halving the staged bytes left it
where it was - so the next attempt at it should be aimed at the decoder's
dependency chain and the activation conversion, not at the width of the tile.
And the matrix unit has 2-4x of issue rate the shipped tile cannot reach because
the k loop is not shaped to feed it.

The residue this branch keeps is `PWW2_W8` (default 0, probe-only) and the two
probes. Section 10.8 records that the shipped build is unchanged by them.

### 10.8 The shipped build is unchanged

The switch is default-off, and that was verified rather than assumed. Compiling
`probe_prefill_tiles.cpp` against `1473a9c`'s `prefill_wmma2.hip` and against
this branch's with `PWW2_W8=0` produces **280,986 lines of gfx1201 assembly in
both, differing in 30 lines, all of them the per-TU `__hip_cuid_*` symbol name**
(`clang -S`, `Compare-Object` over the two files). That is the same check
section 8 records for `PF=0`.

The gates agree, on `1473a9c` + this branch, one lock hold, UD file asserted at
16,564,259,744 B and sha256 `1BCE504D...7DC529`, binary asserted
`built with hip feature: true`:

| gate | result |
| --- | --- |
| greedy 12 ids, UD | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]`, byte-identical |
| `parity prefill --deep` | 15 passed, 0 failed |
| `parity --deep` | 74 passed, 0 failed, 1 skipped |
| `oracle-gate` UD, 20 steps | PASS 20/20, prefill max abs **0.1086** at 101748 |
| `oracle-gate` stock Q4_K_M, 20 steps | PASS 20/20, prefill max abs **0.1481** at 34114 |
| `cargo test --workspace` | clean |
| `cargo fmt --all -- --check` | clean |

Both `oracle-gate` maxima are the *same values* sections 6 and 9 report, which
is the expected result for a change that does not alter the accumulation order.
