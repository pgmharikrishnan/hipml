# Decode bandwidth: the pure-stream ceiling at the decode stream's own shapes

Base `89d0d8d` (origin/main). Box: R9700 (gfx1201, 64 CUs, 32 GB, 640 GB/s
spec), `Qwen3.8-27B-UD-Q4_K_M.gguf`, q8 KV, chunk 2048, port 8093. Every GPU run
held the `decode-roofline` lock, checked at the start and end; each run also
re-read `stream-shape` as a calibration so a contaminated pass is visible.

## 0. Question and answer

Question: the decode stream's GEMV term is 29.80 ms for 15.48 GB (519 GB/s)
against a measured pure-read ceiling of 598 GB/s. Is that 13% gap the GGUF block
layout (a 2-byte scale per block, rows that are not 16-byte aligned, so the
kernel must fall back to narrow or unaligned loads), and would repacking the
resident weights into an internal SoA layout close it?

Answer: **no.** A kernel that reads exactly the same bytes with the GEMV's own
grid geometry, and nothing else, reaches **606–611 GB/s with four-byte loads and
630–631 GB/s with sixteen-byte loads** on these tensors — above the 598 figure,
not below it. The layout is not what costs the gap: the decode's routed GEMV
entries run at **96.6% of the same-geometry stream** (26.98 vs 26.08 ms), and the
remaining 1.06 ms is in the kernel's per-value load→dequant→FMA chain — not in row
alignment, and not recoverable by adding in-flight work (§2). The alignment half
of the hypothesis is measurable and worth **4.2% on the stream** (w16 vs w4),
which caps what any repack could return at 1.9 ms of a 32.29 ms token; the gain
the brief assumed from it does not exist.

The 598 GB/s figure itself was not a shape ceiling: a flat read of 768 MB needs
16 K blocks to reach 592 GB/s and 32 K to reach 605 (68.7 at 1 K, 269 at 4 K,
497 at 8 K), so it was a grid-limited number. At the grids the decode actually
launches, reads run at 587–626 GB/s.

## 0a. What ships

Nothing changes a kernel or a route in this PR. The slice added the two probes
that produced this note (`stream-shape`, `gemv-entries`), widened
`gpu-sweep --gemv` to all eight resident formats, and made `gemv-check` compare
against the entry `run_token` actually launches (`stream_entry`) rather than the
manifest name. Decode, VRAM and the batch tile are unchanged; §4 has the before/
after gate evidence.

## 1. The pure-stream ceiling per shape (step 1)

`hipml stream-shape --gguf … --reps 2` reads every 2-D quantized tensor the
decode graph streams (409 tensors, 15.80 GB, embedding excluded), one launch per
tensor, one synchronize per pass, best of three passes. Five geometries on the
same bytes:

- `w4`: the GEMV's row tile (16 rows/block, 2 rows/warp, 512-value lane steps)
  with four-byte loads — the width an unaligned block row forces;
- `w16`: the same tile, sixteen-byte loads — the width a 16-byte aligned row
  stride allows;
- `w4r1`: one row per warp (8 rows/block);
- `flat`: the tensor as one span at the GEMV's own block count;
- `flatmax`: the same span as widely gridded as the reader allows.

| geometry | ms | GB/s | note |
| --- | ---: | ---: | --- |
| `w4` | 26.08 | **605.8** | the GEMV's access pattern, no arithmetic |
| `w16` | 25.04 | **631.1** | +4.2%: the whole alignment effect |
| `w4r1` | 26.49 | 596.6 | more grid is slower |
| `flat` | 24.95 | **633.3** | rows ignored, GEMV block count |
| `flatmax` | 27.46 | 575.4 | 8192–16384 blocks: worse |

Three clean runs read 605.8, 607.3 and 611.2 GB/s at `w4`, a 0.9% spread: the
shapes are not the limit. (The table above is the same run as §2's per-type
data, which is why it reads 605.8; run 2 measured 611.2 / 630.4.)

Per type, `w4` / `w16` in that run:

| type | GB | w4 GB/s | w16 GB/s | w16 gain |
| --- | ---: | ---: | ---: | ---: |
| IQ4_XS | 5.696 | 615.0 | 635.8 | +3.4% |
| Q5_K | 5.230 | 593.1 | 618.8 | +4.3% |
| Q4_K | 2.427 | 625.5 | 639.2 | +2.2% |
| Q6_K | 1.804 | 609.3 | 619.2 | +1.6% |
| IQ4_NL | 0.431 | 591.2 | 652.0 | +10.3% |
| Q3_K | 0.077 | 588.8 | 618.3 | +5.0% |
| IQ3_S | 0.077 | 587.4 | 614.7 | +4.6% |
| Q8_0 | 0.061 | 355.6 | (L2) | — |

The type spread at `w4` is 6.5% and does not track size: IQ4_NL (0.43 GB, 3
shapes) reads slower than Q4_K (2.43 GB, 73 shapes), and the 0.077 GB three-bit
formats read faster than Q5_K's 5.2 GB. Grid width does not separate them either
(64 to 15,520 blocks). `Q8_0` is 61 MB and L2-resident, which is why its number is
meaningless.

**598 GB/s is not the ceiling at these shapes.** The GEMV's own access pattern
reads 606–611 GB/s, 1.3–2.2% above the 598 figure, and the 16-byte width reads
630–631. The premise that 47–61 MB tensors cannot stream at the card's rate is
refuted: the row tile, the 512-value lane step and the 320–1088 block grids are
all fine. Even the worst case here (four-byte loads) leaves only a 5% margin to
the nominal 640 GB/s.

## 2. Attribution (step 2)

The decode path launches `stream_entry(proj.entry)`, so the term that matters is
the routed set, not the manifest names. Measured one launch per tensor, one
synchronize per pass (`gemv-entries`, best of three):

| type | GB | entry | ms | GB/s | `w4` stream | GEMV / stream | gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Q5_K | 5.230 | `gemv_q5_k` | 9.133 | 572.6 | 593.1 | 96.5% | 0.31 ms |
| IQ4_XS | 5.696 | `gemv_iq4_xs` | 9.457 | 602.3 | 615.0 | 97.9% | 0.20 ms |
| Q4_K | 2.427 | `gemv_q4_k_r1` | 4.215 | 575.9 | 625.5 | 92.1% | 0.33 ms |
| Q6_K | 1.804 | `gemv_q6_k_r1` | 2.975 | 606.5 | 609.3 | 99.5% | 0.01 ms |
| IQ4_NL | 0.431 | `gemv_iq4_nl` | 0.765 | 563.1 | 591.2 | 95.2% | 0.04 ms |
| Q3_K | 0.077 | `gemv_q3_k` | 0.220 | 347.8 | 588.8 | 59.1% | 0.09 ms |
| IQ3_S | 0.077 | `gemv_iq3_s` | 0.206 | 371.0 | 587.4 | 63.2% | 0.08 ms |
| Q8_0 | 0.061 | `gemv_q8_0` | 0.013 | L2 | 355.6 | — | −0.16 ms |
| **all** | **15.80** | | **26.98** | **585.6** | **605.8** | **96.6%** | **1.06 ms** |

The per-type gaps sum to 1.06 ms but the routed total is 0.90 ms below the `w4`
total (26.08 ms), because Q8_0's row is negative: 61 MB stay in L2, so its GEMV
reads at 0.013 ms while the stream probe's one-launch-per-tensor pass over the
same 61 MB costs 0.172 ms. Excluding Q8_0, the GEMV term is 96.1% of the stream
(26.97 vs 25.91 ms over 15.74 GB).

The same eight formats measured through the manifest names instead (the
`gpu-sweep --gemv` table, which is what the brief's 519 GB/s came from) span
27.34 ms / 578.0 GB/s: routing Q4_K and Q6_K to the one-row entries is worth
0.27 + 0.08 ms of that, and the routed total is the 26.98 ms above. Note the
run-to-run spread on these grids: the plain Q4_K entry read 4.484 ms in the
manifest pass and 4.368 ms in the routed pass minutes later, while the routed
entry read 4.215 ms in that same pass, so per-type rows are ±6% across a long
session. The aggregate is the number to trust; it moves 578–586 GB/s between
passes.

**So the 13% the brief asked about is not there.** Against the same-geometry
stream the routed GEMV is 96.6% efficient at the exact shapes it runs, and the
whole residual is 1.06 ms — 3.3% of a 32.29 ms token. What the brief measured as
519 GB/s was the manifest entry set, not the decode.

Where the residual sits:

- **Big formats, 0.89 ms of the 1.06.** Q4_K (92.1%) and Q5_K (96.5%) are 0.65
  of it. The 320-block dip is real but not uniformly fixable: at m=5120 Q4_K's
  `_r1` entry reads 597.1 GB/s against the plain two-row entry's 499.7, while at
  m=17408 the two-row entry is ahead (581.6 vs 574.6) — the routing is a trade
  the aggregate still wins. IQ4_XS (544.6), Q5_K (547.0) and Q6_K (538.8) are
  also ~10% under their m=17408 rates (614.7 / 606.2 / 619.6), but the one-row
  variant does not help there: Q5_K is already one row per warp, and IQ4_XS read
  535.0 with one (below). So no single row-tile choice covers the formats.
- **Three-bit formats, 0.17 ms of the 1.06.** Q3_K and IQ3_S run at 59–63% of the
  stream, and that is decode arithmetic (codebook gathers), not loads: the same
  bytes stream at 588 GB/s. They are 1% of the bytes, so even doubling them
  returns 0.2 ms.
- **Idle between launches, outside the spans.** 1.99 ms over 409 launches (4.9 µs
  each). This is host-side launch spacing, not bandwidth; it belongs to the
  graph/scheduler, not to this slice.
- **Non-GEMV kernels.** 32.29 ms token − 26.98 ms GEMV − 1.99 ms idle ≈ 3.3 ms
  of attention, SSM, norms, rope and the logits tail.

Levers tested against that 1.06 ms, all rejected on measurement:

| lever | measured | verdict |
| --- | --- | --- |
| one row/warp for IQ4_XS (new `gemv_iq4_xs_r1`, bit-identical) | 601.8 → **535.0** GB/s | −11%, delete |
| two-chunk software pipeline, Q4_K `_r1` | 551.2 vs 575.9 GB/s | −4.3%, delete |
| the same, 128-bit payload (`_r1w`) | 550.4 | delete |
| the same, 4096-value x tile (`_r1_t4`) | 550.0 | delete |
| the same, Q5_K | 582.6 vs 572.6 (+1.7%) | inside spread and exactness never proven, delete |
| the same, Q6_K `_r1` | 610.7 vs 606.5 (+0.7%) | inside spread, delete |
| split-K 2 / 4 / 8 on the 0.59 GB Q4_K group | 496.2 / 456.9 / 410.5 vs 540.0 | slower, and not bit-exact (rel 1.5e-2) |
| streaming (`sc1`) loads | 478.7 vs 540.0 | slower |
| 128-bit loads in the real kernel (`_u4`), per projection | +0.0 to +2.3% | below the 3% target, not routed |
| GEMV grid wider than the work (`flatmax`) | 575.4 vs 633.3 GB/s | oversubscription hurts |

Two things this table establishes. First, every way of putting more loads in
flight — pipelining, split-K, streaming hints, more blocks — makes it slower or
leaves it flat, so the residual is not uncovered memory latency. Second, the only
lever that moves the big formats is rows per warp, and every format is already at
or better than its best value (Q4_K and Q6_K route to `_r1`; Q5_K is one row per
warp; IQ4_XS measured 601.8 two-row vs 535.0 one-row here).

## 3. The layout sweep and what it can and cannot buy (step 3)

The repack hypothesis has two halves: a wider load per value (because a
16-byte-aligned row stride would let the kernel issue `uint4` instead of four
dwords) and a separate scale plane (so the decoder's scale bytes are not
interleaved with the payload). The first half is what the `w4`/`w16` pair
measures on identical bytes, and its ceiling is **+4.2% on the stream** (3.1% in
the other run), +1.6% to +10.3% per type.

Putting that same width into the real kernel is the second half, and it is
already implemented as `gemv_q4_k_u4` (one 128-bit load per 16 values, otherwise
`_r1`). Per projection, against `_r1`:

| projection | 1 row/warp | 1 row + 128-bit load | gain | GB |
| --- | ---: | ---: | ---: | ---: |
| ffn_down | 539.2 | 550.0 | +2.0% | 0.60 |
| attn_qkv | 539.4 | 540.5 | +0.2% | 0.59 |
| ffn_up | 564.6 | 562.9 | −0.3% | 0.50 |
| ffn_gate | 563.6 | 564.3 | +0.1% | 0.57 |
| ssm_out | 457.3 | 457.8 | +0.1% | 0.20 |
| attn_output | 412.1 | 421.8 | +2.4% | 0.13 |

So the width is worth 0.6% of the GEMV term in practice, not 4%: the kernel's
loop is not load-issue-bound in the way the pure stream is, which is the same
thing the pipelining and split-K results in §2 say from the other direction.

A real repack would still have to be built and would still have to change the
dequant indexing for every format. Of the 1.9 ms that a perfect `w16` stream
would return, the part a kernel can actually collect is this 0.2 ms; the rest is
the stream's own headroom over an arithmetic loop. The alignment it would buy is
already there for the formats whose row strides are 16-byte multiples (Q4_K
144-byte blocks, Q5_K 176, IQ4_XS 136 at the decode's k values); the unaligned
ones are Q6_K (210-byte blocks, 1.8 GB) and Q3_K/IQ3_S (110, 0.15 GB).
**Rejected:** 0.2 ms for a second resident layout and eight format-specific
packers, when the bytes say the loss is in the decode chain.

### Roofline verdict on 35 tok/s

From the measured token, q8 KV, 64 tokens, best of three processes:

| | ms/token | tok/s |
| --- | ---: | ---: |
| measured @0 | 32.29 | 30.97 |
| measured @8K | 33.22 | 30.09 |
| GEMV at its `w4` stream ceiling | 31.39 | 31.86 |
| GEMV at its `w16`/`flat` stream ceiling | 30.35 | 32.95 |
| 35 tok/s needs | 28.57 | 35.00 |
| llama.cpp, same file | 31.70 | 31.55 |

**35 tok/s is not reachable on this file by any weight-read change.** The entire
GEMV term, replaced by a pure read at the same shapes, returns 2.0 ms — 6% — and
leaves 30.3 ms. Getting to 28.57 ms also requires removing ~1.7 ms from the
5.3 ms that is not GEMV: the 2.0 ms launch idle and the 3.3 ms of attention, SSM,
norm, rope and logits work. That is scheduler and non-GEMV kernel work, outside
this slice.

Beating llama.cpp's 31.55 tok/s (31.70 ms) needs 0.59 ms off the measured token.
The `w4` row already clears it on paper, but that requires the GEMV to run at its
pure-stream ceiling on the 320–1088-block shapes, and every mechanism tried here
to close that last 0.9 ms made it slower. This PR does not claim the win.

## 4. Before and after

Nothing on the token path changed, so this is a no-change check. The `before`
column is the base tree `89d0d8d` in its own session; `after` is this tree.
Decode, q8 KV, 64 tokens, three separate processes each:

| | before | after |
| --- | ---: | ---: |
| @0 ms/token | 32.285 / 32.285 / 32.328 | 32.401 / 32.426 / 32.449 |
| @0 tok/s | 30.97 / 30.97 / 30.93 | 30.86 / 30.84 / 30.82 |
| @8K ms/token | 33.206 / 33.231 / 33.247 | 33.371 / 33.597 / 33.363 |
| @8K tok/s | 30.12 / 30.09 / 30.08 | 29.97 / 29.76 / 29.97 |
| resident | 15.42 GiB | 15.42 GiB |
| `batch-bench --slots 16` | 96.31 aggreg. / 31.22 single | 95.46 / 31.06 |

Every after timing is 0.4–0.9% higher, in the same direction, which looks like a
change and cannot be one: the diff adds two subcommands, four manifest rows, a
new launch method and a gate change; no decode path reads any of them. The
paired run settles it — the base binary and this binary alternating inside one
lock window, three rounds per depth:

| round | base @0 | new @0 | base @8K | new @8K |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32.496 | 32.448 | 33.334 | 33.257 |
| 2 | 32.637 | 32.574 | 33.461 | 33.442 |
| 3 | 32.444 | 32.464 | 33.374 | 33.377 |
| mean | **32.526** | **32.495** | **33.390** | **33.359** |

Mean difference: −0.09% at both depths, i.e. inside the 0.5% spread three reps of
one binary show. Both binaries read ~32.5 / ~33.4 in that window and ~32.29 /
~33.22 in the earlier session, so the session-to-session 0.4% is the box, not the
tree. Batch is the same: 95.47 vs 95.45 aggregate, 31.08 vs 31.11 single.

Gates on the after tree:

| gate | result |
| --- | --- |
| greedy 12 ids, UD | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]`, byte-identical |
| `gemv-check`, 4 tensors × rows 1/2/4 | 12/12, **0 differ**, max abs 0.0 |
| `oracle-gate` 169 ids / 20 steps / chunk 2048 / q8 | **PASS 20/20**, prefill max abs 0.1086, step 19 = 1420 |
| `parity` | 69 passed, 0 failed, 1 skipped |
| `cargo test --workspace` | 16 suites, 0 failed |
| `cargo fmt --all -- --check` | clean |

`gemv-check` now exercises `gemv_q4_k_r1` and `gemv_q6_k_r1` (the entries decode
launches) rather than the manifest names, so the 12/12 is the gate following the
real path.
