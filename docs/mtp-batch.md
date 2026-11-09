# MTP speculative decode over a batch (WP50)

The single-stream MTP path (WP30/34) only runs when a request is alone: the
server switches a request to `Mode::Spec` only if no other slot is active and
nothing is queued, so every batched cell measures `mtp_requests = 0`. This work
composes the two: each slot drafts, one verify pass covers every slot's columns,
and acceptance/rollback stay per slot.

```
slot 0:  t  d1 d2 d3        draft with the 1.37 GB sidecar, per-slot KV
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
column's own position (so a column at position `p` uses the same fixed cache
partition a plain step at `p` uses). The projections, norms, residual adds, FFN,
head and argmax are the same launches as `BatchEngine::forward`. The GDN chain
is the same `*_cols_*` kernels the single-stream verify proved bit-identical to
the plain recurrence.

The bench checks the real condition: every slot's 256-token stream equals that
slot's stream from the plain batched run, token for token.

## Commands

```
hipml mtp-batch-bench --gguf Qwen3.8-27B.gguf --mtp mtp.gguf \
    --ids-file copy_200_words.ids --slots 8 --drafts 2 --tokens 256
hipml mtp-batch-bench ... --slots 16 --drafts 2 --skinny   # server's WMMA path
hipml mtp-batch-bench ... --slots 8 --drafts 2 --admit-at 40
```

`--skinny` sets `HIPML_BATCH_SKINNY=1`, which is what `hipml serve` does by
default; without it the bench runs the exact `_b` batched GEMV. The reported
table gives both.
