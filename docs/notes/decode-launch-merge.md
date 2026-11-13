# Decode launch merge: the per-launch fill is real on the big GEMVs

Box: R9700 (gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf`
(sha256 `1BCE504D…7DC529`, 16,564,259,744 B), q8 KV, chunk 2048, port 8093.
Every GPU run held the `gdn-attn` lock around the whole script and asserted the
model's sha256 and length before taking it; the ambient `HIPML_GGUF` on this box
points at the *stock* file, so every path is passed literally. Every timed tree
asserted `built with hip feature: true` before it ran.

## 0. Question and answer

`origin/main` is `61f2a44`. Decode is 31.36 ms/token at depth 0 (31.9 tok/s)
and 31.92 at 8,192, interleaved against the base in §7.
[`decode-ablation.md`](decode-ablation.md) left three items on the table: the
GDN recurrence at 0.94 ms for 48 launches, the decode attention core at 0.83 ms
for 32 launches, and a **2.08 ms gap** between the in-situ weight stream
(15.485 GB at 542.9 GB/s) and the same bytes at the dedicated pass's
585.6 GB/s. It also ruled out host submission (a whole-GEMV graph replay read
28.548 ms against 28.524 eager) and occupancy (`--ctx 208` and 8,448 are
0.008 ms apart), but it never asked whether a *GPU-side* launch costs time when
the launch is a 47 MB GEMV.

Answer, all four items:

1. **The attention core was kernel work, not a launch floor.** The decode kernel
   sized its key-range splits at one per 64 positions, so the first 200 positions
   of a run ran **one to three blocks on 64 CUs** and each walked the whole
   prefix. Sizing them at one per 8 positions gives 1–25 blocks there and leaves
   8K and above on the counts they already measured (128 and 256).
2. **The GDN recurrence is cold state traffic, not launches.** Its state is
   3.146 MB a layer read *and* written, 302 MB a token, ~0.6 ms at 500 GB/s. A
   second identical launch at the same site costs 5 µs, not 19.5.
3. **The 2.08 ms is per-launch fill/drain on the big GEMVs.** Measured directly
   (kernel spans against wall: 6.6 µs a launch on Q4_K, 7.6 on IQ4_XS, 3.3 on
   Q5_K) and confirmed by removal: merging launches that share an input took
   `gemv_only` from 28.524 to 27.746 ms.
4. **What shipped:** the attention split fix and three merges — the four GDN
   input projections, the attention q/k/v triple, and the ffn gate/up pair —
   each one `gemv_jobs` launch instead of four, three and two.

| | base `61f2a44` | this tree | delta |
| --- | ---: | ---: | ---: |
| depth 0 | 31.356 ms (31.89 tok/s) | **30.019** (33.31) | **−1.34 ms** |
| depth 8,192 | 31.925 ms (31.32) | **31.208** (32.04) | **−0.72 ms** |

The target was ≤30.5 ms at depth 0; this lands 0.48 ms under it, with both
200-token streams byte-identical to the base at both depths.

## 1. Method

Three instruments, all untraced:

- `hipml decode-ablate` (§6): drops a stage's launches and times the rest,
  200 tokens a cell, three reps, cells interleaved in one process. `gemv_only`
  keeps only the weight stream, `gemv+X` adds one stage back, `-X` removes one
  from the full token.
- The `name*2` cells added by this slice: the *same* launch issued twice at the
  same site. The second one runs with its data already in L2 and its code object
  already warm, so `X*2 − X` is what one more launch costs at that site, which
  separates "the kernel does work" from "the launch costs time".
- `hipml gemv-entries` spans-vs-wall: each kernel is timed with its own event
  pair and the whole set against the wall clock; the difference over the launch
  count is the inter-launch gap.

Interleaved A/B for the headline number (§7): the base tree and this tree
alternating inside one lock hold, three reps × two processes × two depths, 200
tokens a run, each run a fresh process. Spread is reported; if it exceeds the
delta the two are called indistinguishable.

## 2. The attention core was one block on one CU

`DeepKv::splits()` sized the key-range split count at one per 64 positions. That
is a floor on how many positions a split walks, not a target, and below 1K it
leaves the warp kernel one to three tasks: 32 launches a token, each on one to
three of 64 CUs, each walking the whole prefix serially.

The fix adds a fine policy under the coarse one:

```rust
let fine = (seq / 8).clamp(1, 128);
let coarse = (seq / 64).clamp(1, 256);
fine.max(coarse)
```

Below 8K the fine policy wins (25 splits at 200 positions, 128 from 1K), and
from 8K up the coarse policy is unchanged — 128 at 8K, 256 at 16K and beyond —
which is where it was measured. Above 8K the two policies are identical, so no
depth regresses.

| cell | before | after |
| --- | ---: | ---: |
| `full`, depth 0 | 31.487 | **30.827** |
| `gemv+attn_reduce` over `gemv_only` | +0.795 | **+0.20** |
| `-attn_reduce` marginal | 0.73 | **0.14** |
| `full`, depth 8,192 | 32.039 | 32.025 |

At 8,192 the split count is unchanged (128 before and after), and the token
moves −0.014, inside that session's 0.15 ms spread.
The core now overlaps with the tail of the projection before it: in the final
table its `-attn_reduce` marginal is inside the spread, while the same launch
issued twice in an otherwise empty token still costs 13.6 µs.

## 3. The GDN recurrence is cold state, not launches

`hipml ssm` prices one f32 layer: the `recurrent` group (conv, the two L2 norms,
the delta-net recurrence, the gated norm) is 12.6–14.4 µs for five launches, of
which the recurrence itself is 6.2–7.5. In the token the same five launches cost
19.5 µs (`gemv+ssm_rec` 28.725 against `gemv_only` 27.790, over 48 layers). The
difference is not the launch: it is that in the token the state is cold.

- state a layer: 3.146 MB (`n_v × head_v × head_v` f32), read and written every
  token — 302 MB a token, 0.6 ms at 500 GB/s;
- `gemv+ssm_rec*2` (the isolated stream plus the recurrence twice): 28.965 ms
  against `gemv+ssm_rec` 28.725, so **the second copy costs 0.240 ms for 48
  launches = 5.0 µs each**, against 19.5 for the first. The first copy pays the
  state's HBM round trip; the second reads what the first just wrote.

So the recurrence's 0.72 ms marginal (`-ssm_rec`, §6) is mostly bytes that must move, and a
launch-shaped fix has nothing to take. The one structural knob measured,
`GDN_ROWS` 4 → 8 (two rows a warp instead of one), was slower: 30.858 against
30.827 in the full token and 7.5 against 7.0 µs in isolation. Reverted.
`HIPML_GDN_SPLIT` 8 remains the best of 8/16/32, within noise.

## 4. The 2.08 ms is per-launch fill on the big GEMVs

Two independent measurements agree.

**Spans against wall.** `hipml gemv-entries` times each kernel with its own
event pair and the set against the wall clock:

| tensor | wall | sum of spans | launches | gap a launch |
| --- | ---: | ---: | ---: | ---: |
| `gemv_q4_k_b` b=1 | 4.669 ms | 4.187 | 73 | **6.60 µs** |
| `gemv_iq4_xs` | 10.472 | 9.397 | 142 | **7.57** |
| `gemv_q5_k` | 9.600 | 9.137 | 142 | **3.26** |

A gap of 3–8 µs on a launch that streams 30–47 MB is the memory pipeline
draining at the end of one grid and filling at the start of the next. It is not
the host: [`decode-dispatch.md`](decode-dispatch.md) §0 already showed that
capturing the whole step into one `hipGraphLaunch` changes nothing, and this
slice's merges are also neutral to it (a graph replays the same launches).

**Removal.** Merging the projections that share an input removes launches
without changing a byte of the stream. Each row is the tree after that merge,
`decode-ablate` at depth 0, three reps of 200 tokens; `full` carries the
attention split fix from §2 in every row after the first.

| tree | launches removed | `gemv_only` | `full` at depth 0 |
| --- | ---: | ---: | ---: |
| base `61f2a44` | — | 28.524 | 31.487 |
| + attention split fix | — | 28.490 | 30.827 |
| + GDN qkv/gate/alpha/beta: 4 → 1 | 144 | 28.224 | 30.532 |
| + attention q/k/v: 3 → 1 | 32 | 28.174 | 30.500 |
| + ffn gate/up: 2 → 1 (and the IQ4_XS body) | 64 | **27.746** | **30.071** |

240 launches removed, 0.78 ms off the stream — the same order as the span gaps,
and it is the *stream* that moves, which is what a fill/drain cost looks like.
A stock-file probe measured the same thing on the ffn pair before any of this
shipped: 128 launches 11.374 ms, 64 merged launches 10.731, one launch over both
10.284.

The per-merge steps are not equally well measured. The GDN merge moved
`gemv_only` 28.490 → 28.224 and `full` 30.827 → 30.532; the ffn pair moved
28.174 → 27.746 and 30.500 → 30.071. Each step is 0.27–0.43 ms, but they come
from separate sessions whose own rep spread is 0.15–0.31 ms, so a single step is
one to two spreads. The attention q/k/v step is −0.032 against a 0.31 ms spread
in its own run and is **not** separable at all; it is kept because it is the
same mechanism on the same job kernel (and with the IQ4_XS body it now covers
all 16 attention layers), but the evidence is the aggregate — 240 launches,
`gemv_only` 28.524 → 27.746, measured in one run with a 0.15 ms spread.

The rate this buys is 15.485 GB / 27.746 ms = **558 GB/s** in situ, up from
542.9, against the dedicated pass's 585.6. The remaining ~1.3 ms is ~256
launches that cannot be merged because the layer's data dependencies sit
between them (the output projection of every block, `ffn_down`, `lm_head`).

## 5. What the merges are

`gemv_jobs` takes a device list of `HipmlGemvJob` and runs each entry's own tile
function under `blockIdx.y`, so a merged row is dequantised and reduced by
exactly the code the standalone entry runs and writes the same bits. This slice
adds an **IQ4_XS body** (`iq4x_tile<2>` with its own 16-row tile) so the UD
file's ffn pair merges too; Q3_K, IQ4_NL and the f32 parity weights still fall
back to their own launches.

- **GDN, 4 → 1.** `attn_qkv` (Q5_K, 10,240 rows), `attn_gate` (Q5_K, 6,144),
  `ssm_alpha` and `ssm_beta` (Q8_0, 48 each). The list is built once at load
  (`plan_proj_jobs`) because every pointer in it is a buffer the layer owns.
- **Attention, 3 → 1.** `attn_q`, `attn_k`, `attn_v`, bound by
  `bind_qkv_input` once the engine's block input exists (it is allocated after
  the layers), so nothing is allocated inside graph capture.
- **FFN, 2 → 1.** `ffn_gate` + `ffn_up`, the same pattern, planned in
  `GpuEngine::new`.

`launches_per_token` (which the probe and `gpu-profile` both read, so it cannot
drift from the code) drops from **1,299 to 1,059**: four GEMV launches become
one in each of the 48 GDN layers, three become one in each of the 16 attention
layers, and two become one in all 64 FFN blocks.

## 6. The table after

`hipml decode-ablate --gguf … --depth 0 --tokens 200 --reps 3 --kv q8`, cells
interleaved across reps in one process, on the shipped tree. Mean of the three
reps; the within-cell spread is 0.01–0.15 ms.

| cell | ms/token | vs `gemv_only` | `full − cell` |
| --- | ---: | ---: | ---: |
| `full` | **30.017** | +2.227 | — |
| `gemv_only` | **27.790** | — | −2.227 |
| `gemv+norm` | 28.194 | **+0.404** | |
| `gemv+ssm_rec` | 28.725 | **+0.935** | |
| `gemv+attn_reduce` | 28.003 | **+0.213** | |
| `gemv+ssm_conv` | 27.989 | +0.199 | |
| `gemv+ssm_l2` | 27.919 | +0.129 | |
| `gemv+ssm_gn` | 27.916 | +0.126 | |
| `gemv+ssm_gates` | 27.913 | +0.123 | |
| `gemv+attn_front` | 27.866 | +0.076 | |
| `gemv+q_gate` | 27.855 | +0.065 | |
| `gemv+kv_append` | 27.844 | +0.054 | |
| `gemv+attn` (all four) | 28.130 | +0.340 | |
| `gemv+ssm` (all five) | 29.039 | +1.249 | |
| `gemv+attn+ssm` | 29.340 | +1.550 | |
| `-norm` | 29.609 | | **+0.408** |
| `-ssm_rec` | 29.302 | | **+0.715** |
| `-ew` | 29.819 | | +0.198 |
| `-attn_reduce` | 29.907 | | +0.110 |
| `-ssm_conv` | 29.956 | | +0.061 |
| `-q_gate` | 30.066 | | +0.049 |
| `-kv_append` | 30.062 | | +0.045 |
| `-ssm_l2` | 29.984 | | +0.033 |
| `-ssm_gates` | 30.031 | | +0.014 |
| `-attn_front` | 30.028 | | +0.011 |
| `-argmax` | 30.008 | | +0.009 |
| `-embed` | 30.024 | | −0.007 |
| `-ssm_gn` | 30.104 | | −0.087 |
| `-lmhead` | 28.298 | | +1.719 |
| `-attn` (whole block, weights too) | 27.787 | | +2.230 |
| `-ssm` (whole block, weights too) | 22.680 | | +7.337 |

The three stages that matter are now `norm` 0.40, `ssm_rec` 0.94 and the
attention core 0.21 added to the isolated stream — 1.55 of the 2.23 non-GEMV
total — and the thirteen marginal removals sum to 1.56 against that whole, 70%.
That is the same sub-additivity [`decode-ablation.md`](decode-ablation.md) §4
measured (it was 51% there), and it moved towards additive for the same reason
`-attn_reduce` fell from 0.70 to 0.11: the attention core no longer disappears
into the projection that precedes it. `-attn_reduce` at +0.11 is what §2 bought: the core still costs
13.6 µs when it runs alone (`attn_reduce2_only` 0.436 ms for 32 launches), but
in the token it now hides inside the projection before it.

At 8,192 the same tree reads `full` 31.344, `gemv_only` 27.828, `-attn_reduce`
30.009 (+1.335), `-ssm_rec` 30.461 (+0.883), `-attn` 27.912 (+3.432) and `-ssm`
23.839 (+7.505). The attention core's marginal grows from 0.11 to 1.34 because
it reads 8,192 keys instead of one to three; the weight stream does not move
(27.79 against 27.83).

## 7. The interleaved A/B

Two trees, two binaries, alternating inside one lock hold: three reps × two
processes × two depths, 200 tokens a run, each run a fresh process. The base
tree is a checkout of `61f2a44`; both binaries asserted
`built with hip feature: true` before the first timed run.

| depth 0 | ms/token | | depth 8,192 | ms/token |
| --- | ---: | --- | --- | ---: |
| base | 31.178 / 31.324 / 31.366 / 31.396 / 31.429 / 31.442 | | base | 31.792 / 31.903 / 31.927 / 31.952 / 31.984 / 31.991 |
| this tree | 29.894 / 30.018 / 30.042 / 30.015 / 30.058 / 30.087 | | this tree | 31.111 / 31.190 / 31.215 / 31.238 / 31.238 / 31.258 |
| **mean** | **31.356** (31.89 tok/s) | | **mean** | **31.925** (31.32) |
| **mean** | **30.019** (33.31) | | **mean** | **31.208** (32.04) |
| delta | **−1.337** | | delta | **−0.717** |

All twelve paired deltas have the same sign: 1.284 / 1.306 / 1.324 / 1.381 /
1.371 / 1.355 ms at depth 0 and 0.681 / 0.713 / 0.712 / 0.714 / 0.746 / 0.733 at
8,192. The smallest paired delta (1.284) is five times the within-cell spread
(0.26 ms at depth 0, 0.20 at 8,192), so the two cells are distinguishable, not a
tie.

`decode-ablate`'s own `full` cell agrees in sign and size: 31.487 on the base
against 30.017 here, −1.47. The two instruments differ by 0.13 ms on the delta
and 0.02 on the level of this tree, the same disagreement
[`decode-ablation.md`](decode-ablation.md) §5 recorded (0.17) between a harness
that restarts every cell from a zeroed scratch and one that streams.

## 8. What was tried and reverted

- **`GDN_ROWS` 4 → 8** (two rows a warp): 30.858 against 30.827 in the full
  token, 7.5 against 7.0 µs isolated. Reverted.
- **A second identical launch at each site** (`*2` cells) to price a launch
  where it runs: 5.0 µs for the GDN recurrence (0.240 ms / 48) and 4.4 for the
  attention core (0.140 / 32). Kept as a probe, not as a change.
- **`HIPML_GDN_SPLIT` 8/16/32**: 8 is the best of the three; no change.

## 9. Gates

`scripts/probes/gates.ps1`, one lock hold, model sha256 asserted before the
lock, on the shipped tree.

| gate | result |
| --- | --- |
| greedy 12 ids, UD, chunk 2048 | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]`, byte-identical |
| 200-token stream, depth 0, vs `61f2a44` | **byte-identical**, 200/200, last id 108899 |
| 200-token stream, depth 8,192, vs `61f2a44` | **byte-identical**, 200/200, last id 197 |
| `gemv-check`, 4 tensors × rows 1/2/4 | 12 checks, **0 differ**, max abs 0.0 |
| `column-bench --check` q8 cols 2, q8 cols 4, f32 cols 4 | 0 differ on 7,946,240 logits, hidden and residual |
| `spec-bench` drafts 0–3, q8 | **128/128 exact** on every row; 33.32 / 51.88 / 60.50 / 63.22 tok/s |
| `batch-bench --slots 16 --tokens 16 --check` | all 16 slots reproduce the B=1 path over the full stream |
| `oracle-gate` UD-Q4_K_M, 20 steps, chunk 2048, q8 | **PASS 20/20**, prefill max abs **0.1086** at 101748 |
| `oracle-gate` stock Q4_K_M, 20 steps, chunk 2048, q8 | **PASS 20/20**, prefill max abs **0.1481** at 34114 |
| `parity --deep` | 74 passed, 0 failed, 1 skipped |
| `cargo test --workspace` | 16 suites, 0 failed |
| `cargo fmt --all -- --check` | clean |

The ids are byte-identical because every merged row is written by the same tile
function the standalone entry calls, with the same per-row accumulation order —
`q4k_tile<1,false,false>`, `q6k_tile<1>`, `q5k_rows`, `q80_row` and now
`iq4x_tile<2>`, which is what `gemv_q4_k_r1`, `gemv_q6_k_r1`, `gemv_q5_k`,
`gemv_q8_0` and `gemv_iq4_xs` run on `blockIdx.x`. The one thing that changes is
which block writes a row, so a mis-sized tile would drop or double rows rather
than change them; that is what the 200-token streams and the two oracle gates
are for.

The stock file matters here for a second reason: its ffn gate/up are Q4_K and
Q6_K, so the stock engine now takes the merged path too, and its oracle gate is
the check that the merge is not UD-specific.

The gate set ran on the tree of the merge commit; the only later change is the
launch count `launches_per_token` prints (1,299 → 1,059), which is not on the
decode path. The shipped binary re-took both 200-token streams after it:
byte-identical to the base at both depths, same last ids, and `gpu-profile`
reads `launches/token: 1059`, a 2.179 ms launch floor.
