# One session, one binary, every axis: four wins, four losses, one tie

*Every earlier article in this series measured one axis on the tree that was
current at the time. This one measured all of them in a single window, from a
single binary, with the gates run first and the spread recorded for every cell —
the losses included.*

The interesting number is not 33.22 tokens per second. It is that 33.22, 62.36,
25,380 MiB and 6.20 s all came out of the same 57 minutes: one binary built from
one tree, one model file measured, one card, every axis measured between 18:38
and 19:35 UTC. Earlier articles each took one axis — prefill, batching,
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
16 ids each; `spec-bench` is 128 of 128 token-exact on all eight rows, at q8 and
again at f32, and it now installs the server's dispatch and exits non-zero on
any drafts row that is not exact; `probe_exact --ctx 32768 --drafts 1,2,3`
reproduces plain decode id for id, 48 replies compared one for one; and
`column-bench --check` finds 0 differing logits in 7,946,240. A cell that had
failed a gate would not have been timed, which is the only reason a timing table
means anything.

**The hashes were asserted before and after.** The binary, the tree, the head
and all three model files were hashed at the start of the window and re-hashed
after the last cell; every one is byte-identical. No model file was written.

**The spread is reported for every cell, not only the good ones.** CLI rows are
three reps in each of two separate processes, so six values; the two
prefill@512 cells took a third process, nine values, after the drift check.
Serve rows are the harness's wave aggregates: ten waves at one slot, six at
eight, three at sixteen. The llama.cpp cells were run in the same window as the
hipml cells, not in a block afterwards. Where the gap is smaller than the
spread, this article says so instead of calling it a win, and the cells where
that happens are named below.

## The table

`Δ` is (hipml − llama) on this session's medians; negative means hipml is below.
Spread is max−min over the cell, as a percentage of the median.

| axis | hipml | llama.cpp | Δ | spread hipml / llama |
| --- | ---: | ---: | ---: | --- |
| CLI prefill @512 | **1,353.8 tok/s** | 1,210.5 | **+11.8%** | 3.3% / 16.1% |
| CLI prefill @512, cold first request dropped | **1,357.6** | 1,219.9 | **+11.3%** | 1.6% / 1.9% |
| CLI prefill @8,192 | 1,328.0 | **1,353.3** | −1.9% | 0.5% / 0.9% |
| CLI decode @0 | **33.22 tok/s / 30.10 ms** | 32.26 / 30.99 | **+3.0%** | 0.35% / 1.1% |
| CLI decode @8,192 | **31.99 / 31.26 ms** | 31.48 / 31.77 | **+1.6%** | 0.24% / 0.84% |
| serve 1 slot | 31.64 tok/s, TTFT p50 2,130 ms | **32.17, 1,918 ms** | −1.7% | 0.6% / 0.2% |
| serve 8 slots | **131.96, TTFT p50 488 ms** | 114.51 (re-run 105.89), 548 ms | **+15.2%** (+24.6%) | 0.6% / 7.9%–11.0% |
| serve 16 slots | 222.42, TTFT 738 ms | **215.70** (re-run 229.45), 1,055 ms | +3.1% / −3.1% | 0.2% / 18.1%–6.0% |
| MTP serve, drafts 3 | 52.36 | **62.36** | **−16.0%** | 0.8% / 0.7% |
| `spec-bench` d0→d3 (q8) | 33.28 → 51.77 → 60.52 → **63.66** | no harness | — | — |
| peak VRAM, CLI @8,192 | 18,395 MiB | **15,809 MiB** | +16.4% | 0 |
| peak VRAM, serve 16 slots | 25,380 MiB | **22,638 MiB** | +12.1% | 0 |
| streamed bytes/token | 15,484,418,048 B | not printed | — | — |

## The verdict, axis by axis

**Won: single-stream decode, at both depths.** 33.22 against 32.26 tok/s at
depth 0 (+3.0%, 30.10 against 30.99 ms per token) and 31.99 against 31.48 at
8,192 (+1.6%, 31.26 against 31.77 ms). The spreads are 0.35% and 1.1% at depth
0, 0.24% and 0.84% at 8,192, so these are separable. They are also small: 0.89 ms
and 0.51 ms per token.

**Won: eight-slot serve, by a wide margin.** 131.96 aggregate against 114.51,
+15.2%, with the 95th-percentile time to first token at 490 ms against 1,392.
The delta is 17.5 tok/s against llama's 9.05 spread, so the cell is separable
even though llama's batching is noisy. A re-run of the same cell put llama's
median at 105.89, which widens the lead to +24.6% rather than narrowing it; the
table quotes the first run, which is the conservative one.

**Won: prefill at 512 tokens — with a caveat that has to travel with the
number.** 1,353.8 against 1,210.5, +11.8%. But llama's spread on that cell is
16.1%, 194.7 tok/s, driven entirely by the first request of each process
(1035.0, 1085.6, 1088.4). The full-set delta is 143.3 tok/s against that 194.7
spread, so the strict reading of the full set is *indistinguishable*. Drop each
process's cold first request and llama is 1,219.9 at a 1.9% spread against
hipml's 1,357.6 at 1.6% — a 137.7 tok/s delta, +11.3% — and that lead holds.
Both readings are in the table and neither should be quoted without the other.
The warm-up repeats in a third process on both engines, in the same direction
and the same magnitude, so it is warm-up inside the process rather than thermal
decay.

**Lost: prefill at 8,192 tokens, narrowly.** 1,328.0 against 1,353.3, −1.9%.
The delta is 25.3 tok/s against spreads of 6.4 and 11.6, so this one is
separable too. hipml is flat from 512 to 8,192; llama gains about 11.8% over the
same range.

**Lost: one-slot serve.** 31.64 against 32.17, −1.7% — 0.53 tok/s. The latency
is worse by more: median time to first token 2,130 ms against 1,918, +11.0%.
That is much closer than it used to be, and what closed it is the serve prefill
chunk, which now defaults to 512. The mechanics are below, and it is still a
loss.

**Lost: MTP at matched depth.** 52.36 against 62.36, −16.0%, with acceptance
nearly equal and the verify exact on both sides of our own path. The mechanics
are below.

**Lost: memory, on every cell.** Peak VRAM is 17,031 against 15,546 MiB at 512
tokens (+9.6%), 18,395 against 15,809 at 8,192 (+16.4%), 20,271 / 24,288 /
25,380 against 16,728 / 21,417 / 22,638 across the three serve cells (+21.2% /
+13.4% / +12.1%), and 21,072 against 18,624 MiB with the drafter at three
drafts (+13.1%). Peak VRAM is identical across reps on both engines, so this is
the one axis with no measurement noise at all. It is the q8 KV layout plus the
recurrent state plus, in serve, a second KV/SSM engine, a prefix snapshot and
the drafter.

## Three results that are not claims

**Sixteen slots is a tie, not a loss.** 222.42 against 215.70 is 6.7 tok/s
ahead, +3.1%, but llama's three waves were 199.20, 215.70 and 238.34 — a 39.15
tok/s spread, 18.1%. A re-run of the same cell gave 224.04, 229.45 and 237.81,
median 229.45 and spread 13.77, which puts hipml 7.0 tok/s behind at −3.1%. The
sign flips between runs, the medians are the middle of a noisy distribution,
and the difference is smaller than the noise both times. On latency the cell is
not close: median time to first token 738 against 1,055 ms, p95 742 against
2,621, and hipml's own spread is 0.2%. Throughput is indistinguishable; tail
latency is not.

**Load time is level, and it is not the same quantity.** hipml loads the engine
in 6.18 to 6.95 s across five values, median 6.20. llama.cpp reports 6.85, 7.01
and 7.07 s from process start to model loaded, and 8.09 s with the drafter.
hipml's number is tensor upload plus graph build; llama's includes process
start. No claim.

**Cell a is not the same metric as cells b and c.** The serve harness computes
cell a as the median of per-stream decode rates, so it equals single-stream
decode, and cells b and c as `sum(decode_tokens) / (last − first)` across the
wave, which is aggregate goodput. Comparing 31.64 with 131.96 and calling it a
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

That row is also exact on both sides of our own path, and this time it is read
from the trace rather than inferred from the width. All three draft counts trace
`route=exact` — b=1+2 at one draft, b=1+3 at two, b=1+4 at three. The verify
holds `exact_scope`, so even the four-column step runs the exact `gemv_*_b`
entries; the approximate tile appears only at b=6, 7, 8, 14, 15 and 16, which is
serve cells b and c, not the verify. 52.36 against 62.36 is therefore an
exact-path comparison on both sides, and the −16.0% is what a scalar GEMV that
issues four FMAs per weight value at b=4 costs against llama's own verify. What
it does not say is whether llama's MTP is exact against its own plain decode:
that was not diffed, and hipml's 128/128 `spec-bench` and 48/48 `probe_exact`
do not transfer to it.

**hipml's serve MTP loses about 11 tok/s against its own kernel path.** The CLI
`spec-bench` at three drafts is 63.66 tok/s at q8; the serve cell at the same
depth is 52.36. Different harness, different chunk, different prompt, and a
64-token stream against a 128-token one, so the 11 tok/s is not a measurement of
serve overhead. It is a warning not to read either number across, and an open
item rather than a result.

**The one-slot TTFT gap is now +11.0%, and the chunk is why.** The serve default
used to be a 256-token prefill chunk and llama-server has no chunk knob, so
hipml was choosing a smaller tile than the reference. At 256 the one-slot TTFT
was 2,500 ms for a 2,423-token prompt; at 512 it is 2,130 ms, +11.0% against
llama's 1,918, where the gap was +30.1%. 512 is llama-server's own `-ub 512`, so
the physical tile is now matched. The sweep puts the cost of that change at
370 ms going from 256 to 512 — five tiles — and the cost does not stay constant
per tile: the next halving, to 1,024, saves only 66 ms over two tiles, and the
last, to 2,048, another 32 ms over one. The first step carries most of it and
the curve then flattens near 32 ms a tile, so no single per-tile figure
describes it. A linear fit over the four sweep points returns ≈59 ms per tile,
and it is worth keeping only as a fit that fails: 74, 33 and 32 ms per tile is
not a line. What is left against llama is 212 ms, 11.0%, and that is a residual,
not a per-tile law. Time to first token is still not a prefill number — it
includes HTTP, admission, queueing and the first token — so the remaining 11.0%
is not all prefill either, and it is still a loss.

## Decode is finished as a bandwidth problem

The engine's own counter says one decode token streams 15,484,418,048 bytes —
14.42 GiB of weights. Against the **30.10 ms** the table measures, that is
**514.4 GB/s, 80.4% of the card's 640 GB/s spec**. The ablation's instrumented
plain step is **30.02 ms**, and the same bytes against it are **515.7 GB/s,
80.6%**. The 0.08 ms between the two is the gap between an instrumented step and
the wall clock behind 33.22 tok/s; either denominator puts decode at about four
fifths of the spec, and the conclusion does not depend on which is used.

That number is the whole argument. A dense decode cannot stream fewer bytes than
the model, so the ceiling is the model divided by the bandwidth, and both
engines sit at roughly four fifths of it, 3.0% apart. The +3.0% is the last few
percent of a problem that is essentially solved, not a new regime. The two
levers that remain are the ones the arithmetic points at: fewer bytes per
weight, and less per-step overhead in the roughly one fifth that is not the
stream. Neither is a rewrite of the matrix-vector multiply.

It is also why the prefill win does not transfer. A prefill tile holds a weight
tile in shared memory and multiplies a chunk of prompt tokens against it; a
decode GEMV reads the same bytes once and throws them away.

## What this does not say

It does not say the engine is faster. It wins four axes, loses four and ties
one, and the four it wins are three narrow ones and one wide one: 3.0%, 1.6% and
11.8% on the medians (11.3% on the warm prefill set), and 15.2% at eight slots.
Against that it loses 1.9% on long-context prefill, 1.7% on one-slot throughput
with an 11.0% worse median TTFT, 16.0% on MTP at matched depth, and 9.6% to
21.2% on peak VRAM in every cell. Any summary that leads with "faster than
llama.cpp" without those four is not this table.

The earlier articles quote an earlier tree's matrix, where decode was 6.1–7.4%
behind. This table has decode ahead at both depths. The one-slot serve numbers
are not directly comparable: the release run's cell a ran the speculative path
and came out 30.4% ahead, while this table's cell a is plain decode at −1.7%.
Both engines were re-measured here; the earlier tree was not re-run, so the
difference between the two is a comparison against a report rather than a
measurement. It is consistent with the decode-path work that landed in between.
Readers comparing the two should treat the older decode column as superseded by
this table; the older prefill columns stand.

Three comparisons need a note rather than a correction:

- The serve prefill chunk is 512 tokens against the CLI's 2,048; 512 is
  llama-server's own ubatch, so this one is like-for-like.
- Time to first token is not a prefill number; only the CLI rows are prefill
  like-for-like.
- Cells b and c run hipml's approximate WMMA tile at batch widths 8 and 16, while
  llama.cpp runs its production kernel. Cell a and all three MTP rows are the
  exact-path hipml serve rows.

And the MTP row is settled on the point that once made it provisional: all three
draft counts trace `route=exact`, so 52.36 against 62.36 compares the exact path
on both sides and its −16.0% is per-step cost, not a kernel substitution. What
it does not say is whether llama's own MTP matches its own plain decode, and why
our four-column step costs more; those are the open items.

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
- The serve prefill chunk decision: [`../notes/serve-prefill-chunk.md`](../notes/serve-prefill-chunk.md).
- The earlier stock-model matrix this one is not comparable with on decode:
  [`../release-run.md`](../release-run.md).
