# SUD: why 16 slots collapsed on UD-Q4_K_M, and the tile entries that fixed it

Base `6ec4491` (origin/main), branch `feat/pro-1491-serve-ud`. Box: R9700
(gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf` and the stock
`Qwen3.8-27B-Q4_K_M.gguf`, q8 KV, port 8093. Every GPU run held the
`serve-ud` lock.

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
it, so a long server run prints a table rather than a firehose.

```
[batch] gemv_q5_k_b    m=17408 k=5120 b=16
[batch] gemv_iq4_xs_b  m=5120  k=5120 b=16
[batch] gemv_iq4_nl_b  m=5120  k=5120 b=16
[batch] gemv_q4_k_b    m=5120  k=5120 b=16
[batch] gemv_q6_k_b    m=248320 k=5120 b=16
[batch] gemv_q3_k_b    m=5120  k=5120 b=16
[batch] gemv_iq3_s_b   m=5120  k=5120 b=16
[batch] gemv_q8_0_b    ...      b=16
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
