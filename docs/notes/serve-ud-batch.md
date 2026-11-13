# SUD: why 16 slots collapsed on UD-Q4_K_M, and the tile entries that fixed it

Base `6ec4491` (origin/main), branch `feat/pro-1491-serve-ud`. Box: R9700
(gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf` and the stock
`Qwen3.8-27B-Q4_K_M.gguf`, q8 KV, port 8093. Every GPU run held the
`serve-ud` lock.

Sections 0–6 are that slice. Sections 7–9 are the follow-up slice
(`06e4433`, branch `feat/1491-serve-parity`, same box, model, KV type and
port, lock label `serve-parity`): the one-slot step, the Q5_K tile's rate, and
the per-slot MTP plan. Section 10 is the lone-slot follow-up
(`0123ecb`, branch `agent/ownboard-1491-oneslot`, lock label `serve-oneslot`).

## 0. Question and answer

Question: the server served one stream at ~29 tok/s on UD-Q4_K_M but only
~98 tok/s aggregate at 16 slots, while the stock Q4_K_M file reached ~216 in
the same cell. Was the 16-slot path silently taking the exact `_b` GEMV because
the skinny WMMA tile had no entries for the UD formats, and what did fixing it
cost?

Answer: **yes, and it was worse than the guess.** The tile covered Q4_K, Q6_K
and Q8_0 only. On UD the missing set was not just IQ4_XS/IQ4_NL/Q3_K/IQ3_S: **Q5_K
was also missing, and Q5_K is 32% of the file's resident bytes**, so every
large FFN projection ran `gemv_q5_k_b`. The route tracer named `gemv_*_b` for
all eight formats at b=16, and turning the tile flag on bought only +12.7% on UD
(98.13 -> 110.63 tok/s in the diagnostic session) against +152% on the stock
file, because the flag was on and the tile was still unreachable. The five
missing entries are **69.5% of the file's resident bytes**; the tile had been
reaching only Q4_K, Q6_K and Q8_0, 30.4%. Adding them moved cell c from
**97.42 to 201.59 tok/s (2.07x)** and cell b from 86.69 to 117.32 (1.35x), and
the 1-slot route change below took cell a's plain cell from 26.13 to 28.37
(1.09x).

## 1. The diagnosis, with the evidence

The tracer is `HIPML_BATCH_TRACE=1` (`graph_launch.rs::trace_route`): it prints
one line per `(entry, m, k, batch)` the first time a batched projection takes
it, so a long server run prints a table rather than a firehose. **Both routes
print a line**, and `route=` names which one ran. A server run therefore shows
two different shapes, and they are the two things this note is about.

**The batch engine at four or more rows takes the tile.** `serve` sets
`HIPML_BATCH_SKINNY=1` for the whole process, and the skinny condition reads
that variable directly (`graph_launch.rs:201-204`), so cell c at 16 slots is
`gemm_skinny_*` / `route=skinny`:

```
[batch] gemm_skinny_q5_k    m=10240  k=5120 b=16 route=skinny
[batch] gemm_skinny_iq4_xs  m=17408  k=5120 b=16 route=skinny
[batch] gemm_skinny_iq4_nl  m=5120   k=17408 b=16 route=skinny
[batch] gemm_skinny_q4_k    m=5120   k=5120 b=16 route=skinny
[batch] gemm_skinny_q6_k    m=248320 k=5120 b=16 route=skinny
[batch] gemm_skinny_q3_k    m=17408  k=5120 b=16 route=skinny
[batch] gemm_skinny_iq3_s   m=17408  k=5120 b=16 route=skinny
[batch] gemm_skinny_q8_0    m=48     k=5120 b=16 route=skinny
```

The batch shrinks as slots retire, and every width from four up is still the
tile: cell c prints `b=16`, `15`, `14`; cell b prints `b=8`, `7`, `6`. At `b=2`
and `b=1` the route is exact instead: the tile needs four rows, and those two
widths are the MTP verify at one draft and the single-stream kernel.

**The MTP verify never takes the tile, at any draft count.** The verify is a
`forward_columns` call, which holds `exact_scope()` because `skinny_on()` reads
`HIPML_SKINNY` and not `HIPML_BATCH_SKINNY` (`columns.rs:220`); `gemv_batched_at`
refuses the tile while that scope is held. Drafts 1, 2 and 3 are 2, 3 and 4
columns:

```
[batch] gemv_q5_k_b   m=10240  k=5120 b=2 route=exact
[batch] gemv_q6_k_b   m=17408  k=5120 b=3 route=exact
[batch] gemv_q6_k_b   m=248320 k=5120 b=4 route=exact
```

Those three lines come from three servers, one per draft count; a single server
prints the width it was started with. Cell c's `b=2` lines are the same thing
seen from the other side: the first request is verified alone before the other
fifteen join it.

**The trap.** `b=4` is not by itself evidence of the tile. Both routes print at
every width, so a `b=4` line naming `gemv_*_b` is the exact `_b` kernel and a
`b=4` line naming `gemm_skinny_*` is the tile; `route=` says the same thing in
one word. Reading the bare `b=` value, or reading the presence of a line, is
what turns the exact verify into a false alarm.

For the record, the diagnosis below was made at base `6ec4491`, before the five
UD entries existed. There the flag was already on and the tile was still
unreachable, so the same cell printed the exact `_b` entries for all eight
formats — that is the defect, not what serve prints now:

```
[batch] gemv_q5_k_b    m=17408  k=5120 b=16 route=exact
[batch] gemv_iq4_xs_b  m=5120   k=5120 b=16 route=exact
[batch] gemv_iq4_nl_b  m=5120   k=5120 b=16 route=exact
[batch] gemv_q4_k_b    m=5120   k=5120 b=16 route=exact
[batch] gemv_q6_k_b    m=248320 k=5120 b=16 route=exact
[batch] gemv_q3_k_b    m=5120   k=5120 b=16 route=exact
[batch] gemv_iq3_s_b   m=5120   k=5120 b=16 route=exact
[batch] gemv_q8_0_b    ...      ...    b=16 route=exact
```

The decision that produced it was a static match:

```rust
fn skinny_for_kind(kind: i32) -> Option<&'static str> {
    Some(match kind {
        GEMV_B_Q4K => "gemm_skinny_q4_k",
        GEMV_B_Q6K => "gemm_skinny_q6_k",
        GEMV_B_Q80 => "gemm_skinny_q8_0",
        _ => return None,          // every UD format landed here
    })
}
```

The A/B before any fix, same box, same session, cell c, `--tokens 64`:

| tree | tile entry for the UD formats | cell c aggregate |
|---|---|---:|
| origin/main | none (the `_b` GEMV) | 97.42 |
| origin/main + `HIPML_BATCH_SKINNY=1` | none: still the `_b` GEMV | 110.63 (+12.7%, diagnostic session) |
| origin/main, stock file | Q4_K/Q6_K/Q8_0 | 274.23 (batch-bench), 216.88 (serve) |

A +12.7% "tile" number is the tell: the flag was on, the tile was not reached for
any UD-shaped projection except the Q4_K/Q6_K/Q8_0 ones.

## 2. What changed

1. `gemm_skinny.hip` gained five entry points, `gemm_skinny_{q5_k,iq4_xs,iq4_nl,
   q3_k,iq3_s}`, over the existing tile body. Kinds 4-6 (Q5_K, IQ4_XS, IQ4_NL)
   stage their bytes with dedicated vector loads, because the generic prefill
   reader measured 2.5-6x slower in this loop: Q5_K reads a 16-byte header plus
   the `qh` run, IQ4_XS reads `d|scales_h` as one dword and its 8-value run as
   one `uint2`, IQ4_NL reads 18-byte blocks as four u16 pairs. Kinds 7-8 (Q3_K,
   IQ3_S) reuse `pf_dequant8`: together they are 1.4% of the file's bytes and
   their 110-byte blocks make the vector split awkward.
2. The IQ4 codebook is staged into shared memory per block (`shkv[16]`): a
   `__constant__` table compiles to a global load on this target, and the index
   is per-lane.
3. The dispatch is now **one table**. `BATCH_RESIDENT` in
   `crates/hipml-kernels/src/lib.rs` lists `(_b kind, gemv entry, matrix kind)`
   for the eight resident formats, and `batch_kind_for`, `batch_kind_entry`, the
   new `batch_kind_gemv` and `graph_launch::skinny_for_kind` all derive from it.
   `skinny_entry` is the single place a format opts into the tile. There is no
   second kind table to drift.
4. `batch == 1` now routes to the single-column stream entry
   (`crate::stream_entry(name)`) instead of `_b` with one row. It is a routing
   decision inside `gemv_batched_at`, not a second scheduler: the batch engine,
   the slot bookkeeping and the exactness contract are the same. Measured
   28.87 -> 31.57 tok/s, 1.01x the single-stream engine.
5. `skinny_launch::skinny_resident_rows` states the alignment contract per
   format instead of hard-coding "Q4_K is 16, Q6_K/Q8_0 are 2": Q4_K needs 16,
   Q5_K and IQ4_XS need 8 (their staging loads are 8-byte runs), the rest need
   2. `k % 256 == 0` holds for every resident block format.

## 3. Results

`batch-bench`, depth 0, q8 KV, tokens 16, one process per cell, origin/main vs
this branch in the same session (branch cells repeated; both runs shown where
they differ). The exact path is the bit-identical `_b` GEMV; the skinny path is
the approximate WMMA tile and is **always labelled**.

| cell | path | origin/main | branch | vs before |
| --- | --- | ---: | ---: | ---: |
| UD B=16 | exact | 96.47 | 96.87 / 95.84 | 1.00x (unchanged, by design) |
| UD B=16 | skinny | 110.04 | **255.58 / 253.18** | **2.31x** |
| UD B=8 | exact | 93.53 | 93.29 | 1.00x |
| UD B=8 | skinny | 97.16 | **138.63 / 137.26** | **1.42x** |
| UD B=1 | -- | 28.83 | **31.51 / 31.59** | **1.09x** |
| stock B=16 | skinny | 274.23 | 276.18 | 1.01x |
| stock B=16 | exact | 108.97 | 109.50 | 1.00x |

Server cells (`bench/serve/serve_bench.py`, `--tokens 64`, no drafter, same box
and session, three waves of five/eight/sixteen conversations; the base and
branch cells for each row ran back to back):

| cell | slots | ctx/slot | origin/main | branch | vs before |
| --- | ---: | ---: | ---: | ---: | ---: |
| a | 1 | 32,768 | 26.13 | 28.37 | 1.09x |
| b | 8 | 16,384 | 86.69 | **117.32** | **1.35x** |
| c | 16 | 8,192 | 97.42 | **201.59** | **2.07x** |

Peak VRAM: cell a 20,082.8 MiB (20,584.0 with the drafter, measured earlier),
cell b 24,485.7, cell c 26,539.3. Zero request errors in every cell, and
`prefix_hit_tokens_p50` 2,396 of 2,422 in both b and c.

Cell c is 92.9% of the stock file's 216.88 on the same session, and 2.07x its
own baseline. The remaining gap is the tile's rate on the UD formats, not the
dispatch: `skinny-bench --ud` measures Q5_K 356 GB/s staged / 275 GB/s through
the arithmetic phase against ~400 for Q4_K, IQ4_XS 387/311, IQ4_NL 312. That is
the staging design's cost (the prefill note reaches the same conclusion: the
staging phase cannot overlap the matrix work), and closing it is a tile rewrite,
not an entry. The card's absolute 212 tok/s line is therefore **not met**
(201.59, 95% of it); the 2x-today condition is.

## 4. What deliberately did not change

- **No second scheduler.** The 1-slot win is a kernel choice inside the batch
  GEMV, so `serve` has one scheduling policy and one exactness story.
- **No speculative decoding.** See section 5.
- **No Q5_K micro-optimisation.** A packed-fifth-bit variant of the Q5_K finish
  measured 356 GB/s against 350 for the shipped form — inside run-to-run noise —
  and was reverted; the diff keeps the simpler kernel.
- **No new storage format.** Every tensor stays in its GGUF blocks; the tile
  stages from the resident weights the same way the prefill tile does.

## 5. Per-slot MTP: analysis only, not implemented

The scheduler still picks `Mode::Spec` only when a request is alone. The
question is what per-slot drafting would add once the batch fills a 16-row tile.

Facts that decide it:

- A verify column is a batched projection row like any other, so a spec step
  costs `ceil(M/16)` weight passes for `M = B * (1 + R)` rows.
- At B=16 a plain step is already one full 16-row pass. Any `R > 0` adds a
  whole second pass for at most `1 + R` tokens per slot. With the tile now at
  ~60 ms per pass, B=16 MTP cannot pay: the measured `mtp-batch-bench` B=16 R=1
  cell was 175.5 tok/s against a 263.6 plain baseline (0.67x), and the plain
  baseline on this branch is faster still.
- At B=8 the arithmetic works while `M <= 16`, i.e. `R = 1`: measured 166.5 vs
  148.0 plain (1.13x), and the same shape on this branch keeps that ratio.
  B=4 R=3 and B=1 R=3 measured 2.07x and 2.20x.
- The verify pass is exact by construction today (`HIPML_BATCH_SKINNY` is
  ignored for MTP verify), and per-slot acceptance/rollback needs a GDN state
  snapshot per column. That is a scheduler change with its own correctness
  story, not a dispatch entry.

Recommendation: leave it out of this change. If it is taken up, do it with the
GDN chain batched across slots first (the note in `mtp-batch.md` estimates that
removes ~25 ms at B=8 and ~45 ms at B=16), and cap it at `B <= 8`,
`R = floor(16/B) - 1`. At B=16 the honest answer is "no": the tile is the
bottleneck, and it is already one pass.

## 6. Gates

| gate | result |
| --- | --- |
| `parity skinny` | 13 passed, 0 failed (5 new UD cases) |
| `parity --deep` | 71 passed, 0 failed, 1 skipped (was 66) |
| `batch-bench --check` UD B=8 exact | all 8 slots 16/16 vs B=1 |
| `batch-bench --check` UD B=16 exact | all 16 slots 16/16 vs B=1 |
| `batch-bench --check` UD B=16 skinny | all 16 slots 16/16 |
| `generate --ids 760,6511,314,9338,369 --tokens 20 --device` UD | exactly the published 20 ids |
| `oracle-gate` UD | PASS 20/20, max 0.1086 at 101748 |
| `oracle-gate` stock | PASS 20/20, max 0.1481 at 34114 |
| `cargo test --workspace` | 0 failed |
| `cargo fmt --check` | clean |

The tile is still **approximate**. `--check` on the skinny path reports 16/16
ids on this file, and that is the honest wording: the token ids agree, the
logits do not (the `parity skinny` cases show fp16-level error against the f32
reference), so every skinny throughput number above is labelled as such and the
exact path is the one `--check` validates. `HIPML_BATCH_LEGACY=1` forces the
exact path and reproduces the published ids.

## 7. The one-slot step, and Q5_K's rate

### 7.1 The step is the forward pass

Cell a ran 28.37 tok/s (35.3 ms) after section 2 and 29.70 (33.7 ms) now. The
question was whether the remaining difference against the CLI was the server's
scheduling and HTTP work. Two hooks in `serve/sched/run.rs`, both off unless
their variable is set, answer it: `HIPML_SERVE_STEP_LOG=1` prints, per batched
step, the `batch.forward` wall time and the wall gap since the previous step
began — the gap minus forward is everything the server adds between steps.
`HIPML_SERVE_SPLIT=1` turns on the engine's stage trace and prints nine syncing
buckets every 16 steps. Neither touches a step when unset.

| one slot, ctx 32,768, cached prompt | ms |
| --- | ---: |
| wall gap between consecutive steps | 33.9 |
| `batch.forward` inside it | 33.7 |
| scheduler, HTTP, detokenise | ~0.2 |

So the scheduler is not the story; the forward pass is the step.

### 7.2 The stage trace names one bucket

Same prompt (2,423 tokens), one slot, two cache contexts:

| stage bucket | ctx 32,768 | ctx 2,495 | delta |
| --- | ---: | ---: | ---: |
| `attn_kv` | 3.89 ms | 1.25 ms | **+2.64** |
| the eight other buckets | — | — | equal within 0.1 ms |

Buckets are offsets between syncs, so only same-bucket comparisons across runs
are meaningful; this one is unambiguous. The cache is the only thing that
changed.

### 7.3 The defect: split count from the position, run size from the cache

`attn_decode_kv.hip` splits the cache into `splits` runs of
`ceil(stride / splits)` rows, fixed in cache positions, and a run whose start is
past the live sequence does nothing. The batched engine chose `splits` from the
live position, `(pos + 1) / 64`, while the run size was keyed on the cache row
count. At a 2.4K sequence in a 32K cache that is **37 runs of 886 rows: three
live runs, 12 blocks on 64 CUs**, and every dead run still pays its share of the
split-reduce traffic. The single-column path has the same shape
(`deep_kv::splits(seq)` with the same `per = ceil(stride / splits)`).

Only cell a passes `--ctx 32768`, which is why the defect surfaced there. b (16K)
and c (8K) have the same geometry error at smaller amplitude: 37 runs of 443 and
222 rows leave 6 and 11 live runs.

### 7.4 The fix: derive the count from the cache, keep the run size

```rust
pub(super) fn attn_splits(stride: usize, slots: usize) -> usize {
    let per = 64 * slots.div_ceil(4).max(1);
    stride.div_ceil(per).clamp(1, SPLITS_CAP)
}
```

The count now comes from the cache's stride, never from the live position. The
run stays 64 rows per four slots, so a wide batch grows the run instead of
narrowing it: the block count stays near 150 at every B and the reduce traffic
is where it was. `SPLITS_CAP = 256` is a measurement, not a guess — 512 runs of
64 rows at one slot was 0.5 ms *slower*, because the reduce reads every run's
partial. Scratch is allocated from the count in use, not from the cap — 32 runs
at 16 slots in an 8K cache instead of a fixed 256 — and that scratch is per
full-attention layer, which is where cell c's peak VRAM drop comes from
(26,548 → 25,201 MiB).

Every caller follows the cache: `forward.rs` launches one attention call per
slot group with `dk.splits`, `verify.rs` dropped its per-column split groups
(every column of a slot shares one count, so the whole group is one launch), and
`mtp.rs` uses `self.kv.splits`. The kernel is untouched.

One slot, same session: `attn_kv` 3.89 → 2.26 ms, `batch.forward` 35.18 →
33.39 ms, cell a 28.33 → 29.70 tok/s.

### 7.5 Cell results

`bench/serve/serve_bench.py --engine hipml --tokens 64`, base and branch built
from `06e4433` in the same session, one process per cell, branch cells at the
card's rep counts (a 10, b 6, c 3). The exact path is the `_b` GEMV; the skinny
path is the approximate WMMA tile and is labelled as such everywhere it is
quoted.

| cell | slots | ctx/slot | before | after | vs before | peak VRAM before → after |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| a | 1 | 32,768 | 28.33 | **29.70** | 1.05x | 20,091 → 20,091 MiB |
| b | 8 | 16,384 | 117.48 | **130.36** | 1.11x | 24,494 → 24,110 MiB |
| c | 16 | 8,192 | 202.92 | **220.91** | 1.09x | 26,548 → 25,201 MiB |

Zero request errors in every cell. Cell a is the attention fix alone: at one
slot `batch == 1` still routes to the single-column stream entry, so the tile's
Q5_K rewrite is not on that path. b and c carry both changes; Q5_K is 32% of the
file's resident bytes and its tile rate rose 1.5x, which bounds the tile's share
of those two cells.

### 7.6 Why cell a stops at 29.7

The card's 31.0 tok/s needs 32.26 ms per step. At this shape the attention is
2.26 ms of a 33.39 ms forward pass; taking it to its small-context floor of
1.25 ms would give 32.4 ms, or 30.9 tok/s. That is arithmetic, not a
measurement, and it is still short of the target. What is left is the staged
attention kernel's 64-row run on a 2.4K sequence — the kernel file this change
was scoped out of. The CLI has the same defect and the same headroom: the
identical 2,423-token prompt measured 34.02 ms per token at ctx 2,495 and
42.82 ms at ctx 32,768 (29.39 vs 23.35 tok/s for identical work).

### 7.7 The Q5_K tile: all loads in flight, one header load

Section 3 measured Q5_K as the worst of the resident formats at the gate shape,
and Q5_K is 32% of a UD file's bytes. The tile's block body for Q5_K
did three things in series: read two `uint2`s of header, then the `qh` run, then
the `qs` run, with the dequant waiting on each. The rewrite:

- reads the 16-byte header (d, dmin, the 12 scale bytes) as one `uint4` instead
  of two `uint2`s,
- hoists the `qs` load into `pws_prep` beside `qh`, so every staging load is in
  flight before the arithmetic starts (`pws_raw` grows from six words to eight),
- dequantises in the GEMV's packed form: the nibble and its fifth bit are
  extracted together per 8-value run instead of per element.

`skinny-bench --ud`, base and branch interleaved three times in one session,
`--reps 10`, GB/s at R=1 and R=16:

| shape | R | before (3 rounds) | after (3 rounds) | ratio |
| --- | ---: | --- | --- | ---: |
| q5_k 17408x5120 ffn_gate | 1 | 339 / 333 / 332 | **474 / 478 / 478** | 1.43x |
| q5_k 17408x5120 ffn_gate | 16 | 339 / 341 / 344 | **485 / 484 / 487** | 1.43x |
| q5_k 5120x17408 ffn_down | 1 | 270 / 266 / 266 | **410 / 405 / 413** | 1.53x |
| q5_k 5120x17408 ffn_down | 16 | 265 / 263 / 266 | **416 / 421 / 418** | 1.58x |

Interleaved with those rounds, the other three UD shapes did not regress: IQ4_XS
up 367 → 363, IQ4_XS down 288 → 285, IQ4_NL down 312 → 308, all inside the
per-round spread. `pws_raw` grew from four dwords to eight and that costs nothing
measurable at these shapes. What remains is the staging budget itself: the tile
now runs at 73% (gate) and 65% (down) of the GEMV's rate on the same bytes, up
from 51% and 42%, so the next step is the tile body, not another entry.

Note that `skinny-bench` runs these kernels at 0.13-0.15 ms, so its absolute
GB/s moves by up to 15% with the rep count and the card's clock state; only the
interleaved pairs above are comparable. An earlier reading of this rewrite
quoted 520 and 425 from unpaired runs, 4-9% optimistic.

## 8. Per-slot MTP: the plan, with today's pass times

Section 5's recommendation still holds: per-slot drafting is not in this change.
The tile changed the arithmetic, so the schedule is now written against measured
pass times instead of estimates.

Facts:

- A verify column is a batched row like any other, so a spec step costs
  `ceil(M / 16)` weight passes for `M = B * (1 + R)` rows. A plain step is one.
- One pass at 16 slots is now **72.4 ms** (cell c, 220.91 tok/s) and at 8 slots
  **61.4 ms** (cell b, 130.36); at depth 0 with a 24-token cache the same 16-row
  tile pass is 56.9 ms (`batch-bench`), so the rest is the 8K cache's attention
  plus sampling and the serve loop. Drafting pays only while `M <= 16`, i.e.
  `B * (1 + R) <= 16`; past that the step doubles for at most `1 + R` tokens per
  slot.
- So the schedule is `R = floor(16 / B) - 1`: B=1 R=3 (measured 2.20x), B=4 R=3
  (2.07x), B=8 R=1 (1.13x on the pre-tile tree, better now because the 16-row
  pass costs less than twice the 8-row one), and **off at B=16** — measured
  175.5 tok/s against 263.6 plain, 0.67x, and today's 16-slot plain step is
  220.91.
- The verify pass is exact by construction (`HIPML_BATCH_SKINNY` is ignored for
  MTP verify), and that must stay: the draft is approximate, the accept is not.

Plan, in order:

1. **Batched GDN chain across slots.** Today the recurrent state advances per
   slot per column; `mtp-batch.md` estimates the batched chain removes ~25 ms at
   B=8 and ~45 ms at B=16. Without it B=8 R=1 spends most of its win on the
   chain.
2. **Per-slot state snapshot and rollback.** A GDN snapshot per
   `[slot][column][layer]`, restored to each slot's own accepted prefix. This is
   a scheduler mode with its own correctness argument, not a dispatch entry.
3. **Admission:** choose `R` from the live slot count when a batch is formed, not
   per request; a slot joining an occupied batch gets `R = 0` for that step.
4. **Gate:** `mtp-batch-bench --slots 8 --drafts 1` must reproduce the plain
   batched run's streams token for token, in both GEMV modes, plus
   `--slots 9 --active 8 --admit-at 100` for the mid-run admission case. Only
   then does `serve` switch a batched request to `Mode::Spec`.

Do not do 3 without 1 and 2: at B=16 the honest answer is still no, and at B=8
the gain is a fraction of a pass, not a multiplier.

## 9. Gates for the follow-up slice

| gate | result |
| --- | --- |
| `parity skinny` | 13 passed, 0 failed |
| `batch-bench --check` UD B=8 exact | all 8 slots 16/16 against B=1 |
| `batch-bench --check` UD B=16 exact | all 16 slots 16/16 against B=1 |
| `batch-bench --check` UD B=16 skinny | all 16 slots 16/16 (approximate path; not a gate) |
| `oracle-gate` UD | PASS 20/20, max 0.1086 at 101748 |
| `oracle-gate` stock | PASS 20/20, max 0.1481 at 34114 |
| `serve` sixteen conversations at ctx 32,768 | replies byte-identical to the base build, sha1 per reply (`bench/serve/probe_ids.py`) |
| `generate --ids 760,6511,314,9338,369 --tokens 20 --device` UD | exactly the 12 published ids |
| `cargo test --workspace` | 0 failed |
| `cargo fmt --all -- --check` | clean |

The attention change is exact by construction: the partition is fixed in cache
positions and a column attends the same rows either way, so no id moved and the
skinny tile is still the only approximate path. `HIPML_BATCH_LEGACY=1` forces
the exact GEMV and still reproduces the published ids. On the stock file the
same skinny check reports `TOKENS DIFFER` at B=16, which is the behaviour
[`../reproduce.md`](../reproduce.md) already documents for that file: neither
skinny result is a gate, and the exact path is the one `--check` validates.

## 10. The lone slot: the cache-shaped attention run, and the GDN chain

Sections 7–9 closed most of the one-slot gap but left cell a at 29.70 tok/s
against the CLI's 31.1 on the same prompt. This slice (`0123ecb`, branch
`agent/ownboard-1491-oneslot`, same box, model, q8 KV and port, lock label
`serve-oneslot`) attributes the last millisecond and removes it. Cell a is
**30.98 tok/s (32.28 ms)** in a ten-wave interleaved A/B, 99.6% of the CLI's
31.08 on the same session. The 31.5 line is **not met**: the CLI itself measures
31.08 today, so the remaining 1.3% is the weight stream both engines share, not
a server path.

### 10.1 What one slot actually pays

The attribution is untraced end-to-end, not the stage trace: `HIPML_SERVE_SPLIT`
adds a sync per bucket and inflated the same step from 33.7 to 44.3 ms, so every
bucket in a traced run is a different run. The comparable numbers are wall
times of whole steps.

| one slot, 2,423-token prompt, 64 tokens | tok/s | ms/token |
| --- | ---: | ---: |
| server cell a, ctx 32,768, base | 29.68 | 33.68 |
| server, ctx 2,495, base | 30.61 | 32.67 |
| CLI `generate --device --kv q8 --ctx 32,768` | 31.08 | 32.18 |
| CLI, ctx 2,495 | 31.11 | 32.14 |

The 32K cache costs the base server 1.01 ms and the CLI nothing: the CLI's
`deep_kv::splits(seq)` is derived from the live sequence, the batched kernel's
`per = ceil(stride / splits)` from the cache. The gap was one kernel choice, not
scheduling (7.1 already showed the server adds ~0.2 ms between steps).

One measurement note: the CLI control must pass `--kv q8`. The default is f32,
and f32 KV at 32K reads 42.2 ms/token against 33.2 at ctx 2,495 — a real cache
effect, but not the server's configuration.

### 10.2 The staged run is sized by the cache

`attn_decode_batch` launched the staged `_b` kernel with `splits` from the live
position and a run of `ceil(stride / splits)` cache rows. At 2,423 tokens in a
32K cache that is 37 runs of 886 rows: three live runs, 12 blocks on 64 CUs, and
the reduce reads 34 dead partials. The single-stream engine's whole-group warp
kernel (`attn_decode_kv_w2`, the same one `DeepKv` picks for q8 at head_dim 256)
partitions over the live sequence and takes its split count from it.

At `slots == 1`, q8, head_dim 256 and a GQA group ≤ 8, the batch launcher now
calls that kernel through `launch_kv::launch_cols_ptrs` (selector 4,
`attn_decode_kv_w2_b`) with one split per 32 live positions, clamped to 256 and
to the allocated scratch:

```rust
let live = (live_seq / 32).clamp(1, WARP_SPLITS).min(splits);
```

The caller states the live length: `forward.rs` passes
`positions.max() + 1` for a plain step, and `mtp.rs` passes 0, which keeps the
staged path for the verify group — its columns share one cache and one split
count, which is the property the staged kernel provides. `HIPML_ATTN_STAGED=1`
opts out.

### 10.3 The GDN recurrence had the same shape

`gdn_ar_b` takes `grid.y` as the slot, so a lone slot ran one block per value
head with no striping and none of the four-row unroll the single-stream step got
in section 7 (`ssm_ops::recurrence`, 1.178 → 0.193 ms over 48 layers). At
`slots == 1` the layer now launches `gdn_ar_f32` with the slot-0 state, conv
window and gate pointers and eight stripes — the same fma chain spelled out
explicitly, so the slot's result is unchanged. `HIPML_GDN_STAGED=1` opts out.

### 10.4 Attribution, and what did not ship

Two-wave cell a sweeps in one session, staged base 29.71:

| variant | tok/s |
| --- | ---: |
| staged attention, staged GDN | 29.71 |
| warp attention, `live/64` | 30.85 |
| warp attention, 96 splits | 31.11 |
| warp attention, 112 / 128 / 256 splits | 31.04 / 31.03 / 30.29 |
| `w4_b` (kernel 5) instead of `w2_b` | 30.83 |
| 96 splits + staged GDN | 30.57 |
| 96 splits + GDN stripes 4 / 16 / 24 / 32 | 31.05–31.14 |

So attention is worth ~1.1 tok/s, the split policy ~0.3, and the GDN route
~0.5. The shipped policy is one split per 32 live positions (75 at this
length), inside the measured 64–128 plateau. `w4_b` and splits 112/128/256 were
worse and are not in the diff; the GDN stripe count is the single-stream
policy's 8, since 4–32 are indistinguishable here.

The fused attention front end (`attn_front_f32`) was considered for `rows == 1`
and rejected before implementation: it reads `pos[0..4]`, while the batch's
`pos_buf` holds two ints per slot. Matching the layouts would be a batch-buffer
change, not a dispatch entry.

### 10.5 Results

`bench/serve/serve_bench.py --engine hipml --tokens 64`, base `0123ecb` and the
branch interleaved per cell in one lock hold, branch at the card's rep counts
(a 10, b 6, c 3), 64 tokens per request, q8 KV, no drafter. Cell a is the exact
path (`batch == 1` routes the GEMV to the stream entry); b and c are the
approximate WMMA tile, as labelled.

| cell | slots | ctx/slot | path | before | after | vs before | peak VRAM |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| a | 1 | 32,768 | exact | 29.681 | **30.978** | 1.044x | 20,091 MiB both |
| b | 8 | 16,384 | skinny | 130.562 | 130.658 | 1.001x | 24,110 MiB both |
| c | 16 | 8,192 | skinny | 220.569 | 220.675 | 1.000x | 25,201 MiB both |

Zero request errors in every cell (50, 48 and 48 requests). Cell a's p50/min
across its 50 requests is 30.978/30.934 (0.14%); b and c move by 0.08% and
0.05%, below the run-to-run spread of the same binary, so those two cells are
**indistinguishable**, as expected — neither change is on their path.

MTP is unchanged. Cell a with `--mtp` at the default drafts 1 reads 47.19 on
both builds (two interleaved passes of 15 requests each: base 47.10/47.19,
branch 47.13/47.19), and at drafts 3, which reproduces the note's 51.84 line,
base 52.13 against branch 52.06, peak 20,892 MiB, 0 errors. The verify group
still takes the staged attention and the staged GDN chain.

### 10.6 Where the residual sits

The token is 32.28 ms, and the same prompt's CLI token is 32.18: the server and
the CLI now pay the same attention and GDN. The weight stream is ~24.5 ms of
it, and the per-launch kernel-handle lookups are ~0.5 ms in both engines. The
card's 31.5 tok/s needs 31.75 ms, which is 0.43 ms below today's CLI; closing it
means changing the shared weight stream, not the one-slot path, and is out of
this change's scope.

### 10.7 Gates

| gate | result |
| --- | --- |
| `parity --deep` | 74 passed, 0 failed, 1 skipped |
| `batch-bench --check` UD B=8 / B=16 exact | all slots 16/16 ids against B=1 |
| `spec-bench` drafts 0–3 | 128/128 token-exact, drafts 0 baseline 31.32 tok/s |
| `oracle-gate` UD / stock | PASS 20/20 each (0.1086 / 0.1481 max) |
| `generate --ids 760,6511,314,9338,369 --tokens 20 --device` UD | exactly the published 12 ids |
| `serve` sixteen conversations at ctx 32,768 | all 16 replies byte-identical to base `0123ecb`, sha1 per reply (`bench/serve/probe_ids.py`) |
| serve cell a with MTP, drafts 3 | 52.06 tok/s against base 52.13 |
| `cargo test --workspace` | 0 failed |
| `cargo fmt --all -- --check` | clean |

The lone-slot routes are exact by construction: the warp kernel is the
single-stream kernel on the same rows, and the GDN kernel is the same
contraction. The 16-conversation sha1 identity is the end-to-end check that no
id moved.
