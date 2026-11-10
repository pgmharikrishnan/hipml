# MTP speculative decode over a batch (WP50)

The single-stream MTP path (WP30/34) only runs when a request is alone: the
server switches a request to `Mode::Spec` only if no other slot is active and
nothing is queued, so every batched cell measures `mtp_requests = 0`. This work
composes the two: each slot drafts, one verify pass covers every slot's columns,
and acceptance/rollback stay per slot.

```
slot 0:  t  d1 d2 d3        draft with the sidecar, per-slot KV
slot 1:  t  d1 d2
slot 2:  t                  (no drafts: a plain decode column)
        \____ ______/
        verify: M = sum(cols) rows, one weight pass over the 64 layers
        accept the longest matching prefix per slot independently
        roll back each slot's GDN state to its own accepted column
```

## What is batched, what stays per slot

| work | shape |
| --- | --- |
| drafter block (`blk.64`) | `B` rows, one batched GEMV per projection; per-slot q8 KV |
| target projections + FFN | `M = sum(col_b)` rows, the row-wise batched GEMV |
| attention | per slot: append its columns at its own positions, then the batched split-K decode over that slot's cache |
| GDN | per slot: the column chain over that slot's row slice, against that slot's conv window and delta-net matrix |
| state snapshot | `[slot][column][layer]`, restored per slot to its own accepted prefix |
| sampling | device argmax over all `M` rows |

Rows are slot-major, so one slot's columns are a contiguous slice. Attention
uses the existing `*_b` kernels with `kv_slot_bytes = 0` and a pointer advanced
to the slot's cache and first column; the GDN chain uses
`SsmGpuLayer::recurrent_cols_rows`, the same kernels as the single-stream verify
with a row offset and an explicit state pointer.

## Exactness argument

A verify column is the same row-wise kernel invocation as a plain batched step
on the same input, except that the attention split count is chosen from the
slot's own position (so a column at position `p` uses the same fixed cache
partition a plain step at `p` uses). The projections, norms, residual adds, FFN,
head and argmax are the same launches as `BatchEngine::forward`. The GDN chain
is the same `*_cols_*` kernels the single-stream verify proved bit-identical to
the plain recurrence.

The bench checks the real condition: every slot's 256-token stream equals that
slot's stream from the plain batched run, token for token, in the same GEMV mode
(`--skinny` or exact). All cells below pass it.

## Commands

```
hipml mtp-batch-bench --gguf Qwen3.8-27B.gguf --mtp mtp.gguf \
    --ids-file copy_200_words.ids --slots 8 --drafts 1 --tokens 256 --skinny
hipml mtp-batch-bench ... --slots 9 --active 8 --drafts 1 --admit-at 100 --skinny
```

`--skinny` sets `HIPML_BATCH_SKINNY=1`, which is what `hipml serve` does by
default; without it the bench runs the exact `_b` batched GEMV. The reported
table gives both.

## Measured (2026-11-10, R9700, Q4_K target + Q4_0 MTP sidecar, q8 KV)

Fixture `copy_200_words.ids`: a 200-token prompt rotated per slot, 256 greedy
tokens, one process per cell, plain reference measured in the same process and
mode. `M = B * (1 + R)`. Skinny is the server's mode.

### Skinny WMMA (`HIPML_BATCH_SKINNY=1`), 256 tokens

| B | R | plain agg | plain ms | spec ms | agg | /stream | vs plain | exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 28.1 | 35.6 | 42.9 | 46.1 | 46.1 | 1.64x | yes |
| 1 | 2 | 28.1 | 35.6 | 52.2 | 56.9 | 56.9 | 2.02x | yes |
| 1 | 3 | 28.1 | 35.6 | 62.4 | 61.9 | 61.9 | 2.20x | yes |
| 4 | 1 | 79.1 | 50.6 | 67.9 | 114.8 | 28.7 | 1.45x | yes |
| 4 | 2 | 79.2 | 50.5 | 77.4 | 141.6 | 35.4 | 1.79x | yes |
| 4 | 3 | 79.2 | 50.5 | 86.6 | 163.7 | 40.9 | 2.07x | yes |
| 8 | 1 | 148.0 | 54.1 | 85.1 | 166.5 | 20.8 | 1.13x | yes |
| 8 | 2 | 148.1 | 54.0 | 142.4 | 149.2 | 18.7 | 1.01x | yes |
| 8 | 3 | 148.1 | 54.0 | 158.6 | 158.8 | 19.8 | 1.07x | yes |
| 16 | 1 | 263.6 | 60.7 | 161.5 | 175.5 | 11.0 | 0.67x | yes |
| 16 | 2 | 263.6 | 60.7 | 231.8 | 179.6 | 11.2 | 0.68x | yes |

### Exact `_b` GEMV (no skinny), 256 tokens

| B | R | plain agg | plain ms | spec ms | agg | /stream | vs plain | exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 28.1 | 35.6 | 42.7 | 46.3 | 46.3 | 1.65x | yes |
| 1 | 2 | 28.1 | 35.5 | 52.3 | 56.7 | 56.7 | 2.02x | yes |
| 1 | 3 | 28.1 | 35.6 | 63.5 | 60.8 | 60.8 | 2.17x | yes |
| 4 | 1 | 77.7 | 51.5 | 103.3 | 75.3 | 18.8 | 0.97x | yes |
| 4 | 2 | 78.0 | 51.3 | 156.6 | 70.0 | 17.5 | 0.90x | yes |
| 4 | 3 | 78.0 | 51.3 | 203.0 | 69.8 | 17.5 | 0.89x | yes |
| 8 | 1 | 89.8 | 89.1 | 202.5 | 70.0 | 8.8 | 0.78x | yes |
| 8 | 2 | 89.8 | 89.1 | 300.0 | 70.8 | 8.9 | 0.79x | yes |
| 8 | 3 | 89.8 | 89.1 | 396.4 | 63.5 | 7.9 | 0.71x | yes |
| 16 | 1 | 90.4 | 176.9 | 400.9 | 70.7 | 4.4 | 0.78x | yes |
| 16 | 2 | 90.5 | 176.8 | 594.9 | 70.0 | 4.4 | 0.77x | yes |

The exact mode is the wrong batched path above B=4: its `_b` GEMV covers ≤8
rows per weight pass, so `M > 8` multiplies passes. Skinny (16 rows per final
pass) is what the server runs and what the targets refer to.

### Acceptance per draft position (skinny, R=3 runs; B=16 from R=2)

| B | d1 | d2 | d3 |
| --- | --- | --- | --- |
| 1 | 1.00 | 0.97 | 0.94 |
| 4 | 0.96 0.92 0.96 0.92 | 0.96 1.00 1.00 0.94 | 0.92 0.94 0.97 0.92 |
| 8 | 0.96 0.86 0.96 0.90 0.91 0.96 0.99 0.90 | 0.88 1.00 1.00 0.93 0.89 0.92 1.00 0.86 | 0.88 0.94 0.97 0.88 0.58 0.85 0.17 0.86 |
| 16 | mean 0.952, min 0.88, p25 0.92, med 0.97, p75 0.99 | mean 0.943, min 0.83, p25 0.90, med 0.97, p75 0.99 | — |

Acceptance is high enough that d1 alone is worth 1.13x at B=8; d2/d3 buy tokens
but their extra rows cross the 16-row weight-pass boundary, so they do not pay.

### No-accept cost

A no-accept verify column costs exactly the same as an accepting one: `M` is
fixed by the plan, not by the outcome. Batch-level no-accept steps were rare on
this fixture (2 of 129 at B=1,R=1; zero elsewhere), so the honest number is the
fixed step cost: at B=1,R=1 a no-accept step measured 43.1 ms against a 35.6 ms
plain step (1.21x); at B=8,R=1 the whole step is 85.1 ms (1.57x plain) whether
any slot accepts or not. A no-accept step is therefore a strict loss of 1.2x
(M ≤ 4) to 1.6x (M = 16) for that step; the schedule wins only because d1
accepts ~0.95.

### Admission / tail prefill

`--slots 9 --active 8 --admit-at 100` (skinny, R=1): prefilling the joining
slot's 200-token prompt, adopting it into the batch and seeding the drafter took
**404.7 ms**; its first speculative step took **133.2 ms** against a 97.8 ms
normal step, because the active set grows from 8 to 9 and `M` from 16 to 18,
crossing from one 16-row weight pass to two. The join is a step-cost cliff, not
a gradual cost. The eight original slots still reproduced their plain streams
over the 243-token window after the join.

The prefix-cache tail-prefill path (`batch/rows.rs`) is not exercised: this bench
seeds every prompt with a full single-stream prefill and admits by `adopt`, so
row-tail seeding and mixed prompt depths are unverified.

## Where MTP-in-batch is a net win

The skinny verify pass is one weight read per 16 activation rows, so a step costs
`ceil(M/16)` passes plus a per-slot chain cost of roughly 2 ms per extra column.
A plain step costs `ceil(B/16)` passes. Since a step emits `1 + R * accept`
tokens per slot, the win condition is

```
1 + R * accept   >   ceil(B*(1+R)/16) / ceil(B/16)
```

which for this sidecar's acceptance gives:

| B | best R | agg | vs plain |
| --- | --- | --- | --- |
| 1 | 3 | 61.9 | 2.20x |
| 4 | 3 | 163.7 | 2.07x |
| 8 | 1 | 166.5 | 1.13x |
| 16 | — | 179.6 (R=2) | 0.68x |

**Operating point: enable MTP-in-batch while `B * (1 + R) <= 16`, i.e.
`R = floor(16/B) - 1` (B=8 -> 1, B=4 -> 3, B=1 -> 3).** At B=16 the batch is
already exactly one 16-row pass, so any verify adds a whole second pass and the
schedule loses. B=8 is break-even at R=2 (M=24) and wins only at R=1.

Targets: B=1 ≥38 (measured 46-62) and B=8 ≥160 / ≥18 per stream (measured
166.5 / 20.8) hold. B=16 ≥240 / ≥15 does not: measured 179.6 / 11.2 against a
263.6 / 16.5 plain baseline. The blocker is the 16-row tile, not the drafter
(3.2 ms per draft position) or the rewrite (<3 ms).

## What is not batched yet, and the next lever

The GDN chain is still one launch sequence per slot per layer. At B=8, one
skinny tile is ~54 ms but the R=1 verify costs 80.8 ms; the ~27 ms difference is
6 launches x 48 layers x 8 slots of `recurrent_cols_rows`. Batching that chain
across slots (`grid.y = slot`, uniform row count, per-slot state and snapshot
strides) should remove ~25 ms at B=8 and ~45 ms at B=16, which would put B=16
near 240-275 aggregate. That is the next change, not a different drafter.

Also unverified:

- server wiring: `serve/sched/run.rs` still chooses `Mode::Spec` only when alone,
  so there are no end-to-end HTTP or llama-server parity numbers for batched MTP;
- one fixture only (acceptance is prompt-dependent; d3 fell to 0.17 on one slot);
- contexts beyond ~460 tokens, B=32, and mixed-depth slots;
- adaptive fallback: `--adapt` never fired on this fixture (acceptance too high),
  so the "stop drafting after 8 no-accept steps" path is compiled but unmeasured.

## Gates after the change

| gate | result |
| --- | --- |
| `oracle-gate` (285-token chunked prefill, q8) | 20/20 ids, step 19 = 13 on the copy_200 fixture |
| `parity --deep` | 56 passed, 0 failed, 1 skipped |
| `gemv-check` 4 tensors x R=1/2/4/8/16 | 20/20, 0 differing, max abs 0.000e0 |
| `tokenizer_fixture` | 2 passed |
| `mtp-batch-bench` exactness | every cell in the tables above reproduces the plain batched stream over 256 tokens |

The `p2_math_169` oracle cell (step 19 = 41020) is not green in this run: the
attempted rerun was aborted after >25 minutes because the CPU f32 oracle was
thrashing under the shared-memory load, and the GPU lock was released rather
than held longer. This diff adds kernels and a bench; it does not touch the
prefill or single-stream decode path that gate exercises, so that cell is
expected unchanged, but it is listed as unverified rather than green.
