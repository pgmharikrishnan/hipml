# One session, one binary, every axis: four wins, four losses, one tie

*Every earlier article in this series measured one axis on the tree that was
current at the time. This one measured all of them in a single window, from a
single binary, with the gates run first and the spread recorded for every cell —
the losses included.*

The interesting number is not 33.15 tokens per second. It is that 33.15, 62.33,
25,201 MiB and 6.42 s all came out of the same 56 minutes: one binary built from
one tree, one model file measured, one card, every axis measured between 11:56
and 12:52 UTC. Earlier articles each took one axis — prefill, batching,
speculation, the roofline — and each had to say which tree it came from. This is
the first table where the same build produced the prefill number, the decode
number, the server number, the speculation number and the memory number, so the
columns can be read against each other without that footnote.

It also means the losses sit in the same table as the wins, measured the same
way with the same reps. The engine wins four axes, loses four, and ties one.

## The method that makes the numbers worth reading

Three things make a table like this trustworthy, and none of them is a kernel.

**The gates ran first, on the binary that was then timed.** Greedy ids reproduce
12 of 12. `oracle-gate` is 20 of 20 at 20 steps on both models, with the maximum
absolute error against the CPU reference at 0.1086 on the UD file (index
101748) and 0.1481 on the stock file (index 34114) — the same values to four
decimals and the same index as the previous session, on a different tree.
`parity --deep` passes 74 cases; `parity prefill --deep` passes 15;
`batch-bench --slots 16 --check` reproduces the single-row path on all 16 slots,
16 ids each; `spec-bench` is 128 of 128 token-exact at q8 and again at f32. A
cell that had failed a gate would not have been timed, which is the only reason
a timing table means anything.

**The hashes were asserted before and after.** The binary and all three model
files were hashed at the start of the window and re-hashed after the last cell;
every one is byte-identical. No model file was written.

**The spread is reported for every cell, not only the good ones.** CLI rows are
three reps in each of two separate processes, so six values. Serve rows are the
harness's wave aggregates: ten waves at one slot, six at eight, three at
sixteen. The llama.cpp cells were run in the same window as the hipml cells, not
in a block afterwards. Where the gap is smaller than the spread, this article
says so instead of calling it a win, and the cells where that happens are named
below.

## The table

`Δ` is (hipml − llama) on this session's medians; negative means hipml is below.
Spread is max−min over the cell, as a percentage of the median.

| axis | hipml | llama.cpp | Δ | spread hipml / llama |
| --- | ---: | ---: | ---: | --- |
| CLI prefill @512 | **1,346.1 tok/s** | 1,215.2 | **+10.8%** | 2.3% / 13.5% |
| CLI prefill @8,192 | 1,326.1 | **1,352.5** | −1.9% | 0.5% / 0.9% |
| CLI decode @0 | **33.15 tok/s / 30.17 ms** | 32.24 / 31.01 | **+2.8%** | 0.13% / 1.0% |
| CLI decode @8,192 | **31.98 / 31.27 ms** | 31.41 / 31.84 | **+1.8%** | 0.35% / 0.42% |
| serve 1 slot | 31.64 tok/s, TTFT p50 2,500 ms | **32.12, 1,921 ms** | −1.5% | 0.3% / 0.5% |
| serve 8 slots | **132.37, TTFT 489 ms** | 109.42, 563 ms | **+21.0%** | 0.7% / 12.0% |
| serve 16 slots | 222.13, TTFT 742 ms | **234.61, 1,123 ms** | −5.3% | 0.7% / 11.2% |
| MTP serve, drafts 3 *(pending)* | 52.32 | **62.33** | **−16.1%** | 1.0% / 1.7% |
| `spec-bench` d0→d3 (q8) | 33.30 → 51.75 → 60.58 → **63.57** | no harness | — | — |
| peak VRAM, CLI @8,192 | 18,395 MiB | **15,809 MiB** | +16.4% | 0 |
| peak VRAM, serve 16 slots | 25,201 MiB | **22,608 MiB** | +11.5% | 0 |
| streamed bytes/token | 15,484,418,048 B | not printed | — | — |

## The verdict, axis by axis

**Won: single-stream decode, at both depths.** 33.15 against 32.24 tok/s at
depth 0 (+2.8%, 30.17 against 31.01 ms per token) and 31.98 against 31.41 at
8,192 (+1.8%). The spreads are 0.13% and 1.0% at depth 0, 0.35% and 0.42% at
8,192, so these are separable. They are also small: 0.84 ms and 0.57 ms per
token.

**Won: eight-slot serve, by a wide margin.** 132.37 aggregate against 109.42,
+21.0%, with the 95th-percentile time to first token at 491 ms against 1,252.
The delta is 22.9 tok/s against llama's 13.2 spread, so the cell is separable
even though llama's batching is noisy. A re-run of the same cell put llama's
median at 106.78, which widens the lead rather than narrowing it.

**Won: prefill at 512 tokens — with a caveat that has to travel with the
number.** 1,346.1 against 1,215.2, +10.8%. But llama's spread on that cell is
13.5%, driven entirely by the first request of each process (1069.3 and 1108.5).
Drop each process's cold first request and llama is 1209.6–1233.3, a 1.9%
spread, against hipml's 1,349.6 at 0.7% — a +10.2% lead. The full-set delta is
130.9 tok/s against a 164.0 spread, so the strict reading of the full set is
*indistinguishable*; the warm reading is a real 10% lead. Both readings are in
the table and neither should be quoted without the other. The warm-up repeats in
a third process on both engines, in the same direction, so it is warm-up inside
the process rather than thermal decay.

**Lost: prefill at 8,192 tokens, narrowly.** 1,326.1 against 1,352.5, −1.9%.
The delta is 26.4 tok/s against spreads of 7.2 and 11.9, so this one is
separable too. hipml is flat from 512 to 8,192; llama gains about 11% over the
same range.

**Lost: one-slot serve.** 31.64 against 32.12, −1.5% — 0.48 tok/s. The latency
is worse by more: median time to first token 2,500 ms against 1,921, +30%. That
gap is at least partly the 256-token chunk the serve harness pins hipml to, not
decode; the mechanics are below.

**Lost: MTP at matched depth.** 52.32 against 62.33, −16.1%, with acceptance
nearly equal. The mechanics are below.

**Lost: memory, on every cell.** Peak VRAM is 17,031 against 15,546 MiB at 512
tokens (+9.6%), 18,395 against 15,809 at 8,192 (+16.4%), and 20,091 / 24,110 /
25,201 against 16,728 / 21,417 / 22,608 across the three serve cells (+20.1% /
+12.6% / +11.5%). Peak VRAM is identical across reps on both engines, so this is
the one axis with no measurement noise at all. It is the q8 KV layout plus the
recurrent state plus, in serve, a second KV/SSM engine, a prefix snapshot and
the drafter.

## Three results that are not claims

**Sixteen slots is a tie, not a loss.** 222.13 against 234.61 is −5.3%, but
llama's three waves were 234.6, 211.4 and 237.8 — a 26.4 tok/s spread, 11.2% —
against a 12.5 tok/s delta. The medians are the middle of a noisy distribution
and the difference is smaller than the noise. On latency the cell is not close:
median time to first token 742 against 1,123 ms, p95 744 against 2,550, and
hipml's own spread is 0.7%. Throughput is indistinguishable; tail latency is not.

**Load time is level, and it is not the same quantity.** hipml loads the engine
in 6.17 and 6.42 s across the two CLI processes. llama.cpp reports 6.89, 7.00
and 7.05 s from process start to model loaded, and 8.25 s with the drafter.
hipml's number is tensor upload plus graph build; llama's includes process
start. No claim.

**Cell a is not the same metric as cells b and c.** The serve harness computes
cell a as the median of per-stream decode rates, so it equals single-stream
decode, and cells b and c as `sum(decode_tokens) / (last − first)` across the
wave, which is aggregate goodput. Comparing 31.64 with 132.37 and calling it a
fourfold speed-up would be comparing a rate with a sum. The harness does this on
both engines, so the comparison within a cell is fair; across cells it is not.

## The mechanics behind the two interesting losses

**The MTP gap is per-step cost, not draft quality.** llama.cpp was started with
its own MTP draft model and no depth flag, but this build's
`--spec-draft-n-max` default is 3, and its log reports 2.95–3.65 accepted tokens
per step over 50 requests — which cannot come from fewer than three drafts. So
llama-at-default is llama-at-3, and the d3 row is depth-matched. Acceptance is
nearly equal: 0.699 for hipml's serve path at three drafts against llama's 0.712
median. The drafts are not worse; the step that verifies them is more expensive.

There is a second, unresolved point about that row. The batch trace reports a
width of 4 at three drafts — exactly the threshold where the projection path
switches from the exact batched GEMV to the approximate tile — and widths of 2
and 3, both exact, at one and two drafts. The batching note records the
opposite: the verify pass is exact by construction, because
`HIPML_BATCH_SKINNY` is ignored for MTP verify. Those two statements cannot both
be right. Whether the trace's label or the dispatch is wrong is being settled
now, and that contradiction — not the throughput gap — is why the row is marked
pending rather than quoted as final.

**hipml's serve MTP loses about 11 tok/s against its own kernel path.** The CLI
`spec-bench` at three drafts is 63.57 tok/s at q8; the serve cell at the same
depth is 52.32. Different harness, different chunk, different prompt, and a
64-token stream against a 128-token one, so the 11 tok/s is not a measurement of
serve overhead. It is a warning not to read either number across, and an open
item rather than a result.

**Part of the one-slot TTFT gap is the chunk.** The serve harness pins hipml to a
256-token prefill chunk and llama-server has no chunk knob. hipml's CLI prefill
at chunk 2048 is 1,346 tok/s, and its one-slot TTFT of 2,500 ms for a
2,423-token prompt implies about 969 tok/s. Time to first token is also not a
prefill number — it includes HTTP, admission, queueing and the first token — so
the gap is visible in TTFT, not proof of cause.

## Decode is finished as a bandwidth problem

The engine's own counter says one decode token streams 15,484,418,048 bytes —
14.42 GiB of weights. Against the **30.17 ms** the table measures, that is
**513.2 GB/s, 80.2% of the card's 640 GB/s spec**. The ablation's instrumented
plain step is **30.02 ms**, and the same bytes against it are **515.7 GB/s,
80.6%**. The 0.15 ms between the two is the gap between an instrumented step and
the wall clock behind 33.15 tok/s; either denominator puts decode at about four
fifths of the spec, and the conclusion does not depend on which is used.

That number is the whole argument. A dense decode cannot stream fewer bytes than
the model, so the ceiling is the model divided by the bandwidth, and both
engines sit at roughly four fifths of it, 2.8% apart. The +2.8% is the last few
percent of a problem that is essentially solved, not a new regime. The two
levers that remain are the ones the arithmetic points at: fewer bytes per
weight, and less per-step overhead in the roughly one fifth that is not the
stream. Neither is a rewrite of the matrix-vector multiply.

It is also why the prefill win does not transfer. A prefill tile holds a weight
tile in shared memory and multiplies 256 prompt tokens against it; a decode GEMV
reads the same bytes once and throws them away.

## What this does not say

It does not say the engine is faster. It wins four axes, loses four and ties
one, and the four it wins are three narrow ones and one wide one: 2.8%, 1.8% and
10.8% on the medians (10.2% on the warm prefill set), and 21% at eight slots.
Against that it loses 1.9% on long-context prefill, 1.5% on one-slot throughput
with a 30% worse median TTFT, 16.1% on MTP at matched depth, and 9.6% to 20.1%
on peak VRAM in every cell. Any summary that leads with "faster than llama.cpp"
without those four is not this table.

The earlier articles quote an earlier tree's matrix, where decode was 6.1–7.4%
behind. This table has decode ahead at both depths. The one-slot serve numbers
are not directly comparable: the release run's cell a ran the speculative path
and came out 30.4% ahead, while this table's cell a is plain decode at −1.5%.
Both engines were re-measured here; the earlier tree was not re-run, so the
difference between the two is a comparison against a report rather than a
measurement. It is consistent with the decode-path work that landed in between.
Readers comparing the two should treat the older decode column as superseded by
this table; the older prefill columns stand.

Three of the comparisons are not like-for-like and are stated rather than
corrected:

- The serve prefill chunk is 256 tokens against the CLI's 2,048, and llama-server
  has no chunk knob.
- Time to first token is not a prefill number; only the CLI rows are prefill
  like-for-like.
- Cells b and c run hipml's approximate WMMA tile at batch widths 8 and 16, while
  llama.cpp runs its production kernel. Cell a and the MTP plain step are the
  only exact-path hipml serve rows.

And one cell is still moving: the MTP rows are pending re-measurement while the
serve-path exactness of the three-draft verify step is settled. Until then,
52.32 against 62.33 is what this session measured, not a final number.

## Where the numbers come from

- The table, the run identity, the spreads and every caveat:
  [`../benchmarks.md`](../benchmarks.md).
- The model file, its tensor histogram and the load checks:
  [`../native-qwen.md`](../native-qwen.md).
- The decode work that moved this axis since the release run:
  [`../notes/decode-launch-merge.md`](../notes/decode-launch-merge.md),
  [`../notes/decode-ablation.md`](../notes/decode-ablation.md) and
  [`../notes/decode-epilogue.md`](../notes/decode-epilogue.md).
- The serve VRAM decomposition: [`../notes/serve-vram.md`](../notes/serve-vram.md).
- The earlier stock-model matrix this one is not comparable with on decode:
  [`../release-run.md`](../release-run.md).
