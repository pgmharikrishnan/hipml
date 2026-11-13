# Benchmarks

Every number in the table below, and the identity that produced it, comes from
**one session on one card**: a single binary built from tree `af132a4`, every
axis measured between 2026-11-13T18:38:38Z and 19:35:33Z, against llama.cpp
build 11820 (commit `d7725699`), whose binary is byte-identical to the one every
earlier session used. Nothing is carried over from an earlier run. The model is
`Qwen3.8-27B-UD-Q4_K_M.gguf` with q8-class KV on both engines.

The stock-model release run in [`release-run.md`](release-run.md) is the earlier
record of the release tree. The decode path this table measures changed
materially in eight landed slices since that run, so its decode and serve
columns are not comparable with these; its prefill columns are.

This table replaces the `409f4ad` table that used to be here. Six merges landed
after `409f4ad` and several changed behaviour: `spec-bench` now installs serve's
dispatch and exits non-zero on a non-exact drafts row, the serve prefill chunk
default moved 256 → 512 and the harness stopped pinning it, the MTP verify now
holds `exact_scope`, and the trace line prints a route word. So the release can
be tagged on one clean single-session table rather than a table with re-measured
rows grafted onto it.

## Identity of the run

| field | value |
| --- | --- |
| date (UTC) | 2026-11-13T18:38:38Z → 19:35:33Z |
| hipml tree | `af132a458e614dc0aabf2de42012b6389ed47b8a` (`af132a4`), detached worktree off a separate clone, `git status` empty, tree `a8db4671` |
| hipml binary | `hipml.exe`, 2,971,136 B, sha256 `de09134b58dac0a10fc839b948d19c02b0f584959f978d6476e46400746ab969` |
| build | `cargo build --release -p hipml-cli --features hip`, `HIPML_ARCH=gfx1201`, 39.61 s, `BUILD_EXIT=0` |
| build assert | `hipml probe` → `built with hip feature: true` |
| model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, 16,564,259,744 B, sha256 `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529` |
| MTP drafter | `mtp-Qwen3.8-27B-Q4_0.gguf`, 1,369,590,656 B, sha256 `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
| stock Q4_K_M (gates only) | `Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,336 B, sha256 `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` |
| GPU | AMD Radeon AI PRO R9700, gfx1201, 31.9 GiB (an iGPU, gfx1036, is also enumerated) |
| driver / ROCm / OS | 32.0.32015.7029 / 7.2.1 / Windows 11 Pro build 26200 |
| llama.cpp | build 11820, commit `d7725699`, sha256 `6f6bf5e975dd3d5efb314f61688a8ea3b2f6516b0503c28b3783887f47d00ab3` — byte-identical to the `409f4ad` session |
| harness | `bench/job.py` and `bench/serve/serve_bench.py` as committed in `af132a4` |

All three model hashes, the binary hash, the tree hash and the head were
re-read after the last cell and are byte-identical to before it. No model file
was written.

`HIPML_GGUF` was set in the environment to the **stock** file, so every batch
file printed the ambient pair and then cleared both variables on their own lines
before any child process ran, and every command named its file literally with
`--gguf`/`--model`/`--mtp`. No cell could read the ambient stock path.

## Method

- **Gates first, so timing was allowed.** Greedy ids 12/12; `oracle-gate` 20/20
  on both models, UD `0.1086 @101748` and stock `0.1481 @34114`; `parity --deep`
  74 passed, 0 failed, 1 skipped (`sample_top_p_f32`); `parity prefill --deep`
  15 passed; `batch-bench --slots 16 --check` 16/16 on every slot;
  `spec-bench` 128/128 exact on all eight rows, at q8 and again at f32;
  `probe_exact --ctx 32768 --drafts 1,2,3` 48 replies id-for-id; and
  `column-bench --check` 0 differ on 7,946,240 logits. All were run on the
  binary that was then timed, before any cell. Both oracle gates reproduce the
  reference max-error to four decimal places and the same index on a third
  tree, so the prefill numerics have not moved since `409f4ad`. `spec-bench`
  prints `dispatch: batched engine on (serve's HIPML_BATCH_SKINNY=1; the verify
  holds exact_scope), exactness required`, so the gate covers the server's
  dispatch, not just the CLI's.
- **Identical inputs.** Both engines received byte-identical token ids, recorded
  as `ids_sha256` in every result file.
- **Reps.** CLI rows are 3 reps in each of 2 separate processes, so 6 values;
  the two prefill@512 cells took a third process (9 values) after the drift
  check. Serve rows are the harness's wave aggregates: 10 waves in cell a, 6 in
  b, 3 in c.
- **Interleaved.** The llama.cpp cells were run in the same window as the hipml
  cells, not in a block afterwards.
- **Spread for every cell.** `spread` is max−min over the cell, as a percentage
  of the median. A gap smaller than the spread is called indistinguishable, not
  a win.
- **KV.** q8-class on both engines: hipml `q8` plus the SSM recurrent state,
  llama.cpp `-ctk q8_0 -ctv q8_0`. Same precision class, different layout.
- **Decode window.** Tokens 2..33 of a 33-token greedy run on both engines.
  hipml times 32 internal `forward_token_id` calls; llama.cpp reports its
  server-side rate over the same span.

## Headline table

`Δ` is (hipml − llama) / llama on this session's medians; negative means hipml
is below. Spread is max−min over the cell, as a percentage of the median. Where
a `Δ` is smaller than llama's spread, the cell is not a result.

| axis | hipml | llama.cpp | Δ | spread hipml / llama |
| --- | ---: | ---: | ---: | --- |
| CLI prefill @512 tok/s | **1,353.8** | 1,210.5 | **+11.8%** | 3.3% / 16.1% |
| CLI prefill @512, cold first request dropped | **1,357.6** | 1,219.9 | **+11.3%** | 1.6% / 1.9% |
| CLI prefill @8,192 tok/s | 1,328.0 | **1,353.3** | −1.9% | 0.5% / 0.9% |
| CLI decode @0 tok/s | **33.22** (30.10 ms) | 32.26 (30.99 ms) | **+3.0%** | 0.35% / 1.1% |
| CLI decode @8,192 tok/s | **31.99** (31.26 ms) | 31.48 (31.77 ms) | **+1.6%** | 0.24% / 0.84% |
| serve a, 1 slot, aggregate tok/s | 31.64 | **32.17** | −1.7% | 0.6% / 0.2% |
| serve a, TTFT p50 / p95 ms | 2,130 / 2,138 | **1,918 / 1,933** | +11.0% / +10.6% | — |
| serve a, errors | 0 / 50 | 0 / 50 | — | — |
| serve b, 8 slots, aggregate tok/s | **131.96** | 114.51 (re-run 105.89) | **+15.2%** (+24.6%) | 0.6% / 7.9%–11.0% |
| serve b, TTFT p50 / p95 ms | **488 / 490** | 548 / 1,392 | −11.0% / −64.8% | — |
| serve b, errors | 0 / 48 | 0 / 48 | — | — |
| serve c, 16 slots, aggregate tok/s | 222.42 | **215.70** (re-run 229.45) | +3.1% / −3.1% | 0.2% / 18.1%–6.0% |
| serve c, TTFT p50 / p95 ms | **738 / 742** | 1,055 / 2,621 | −30.0% / −71.7% | — |
| serve c, errors | 0 / 48 | 0 / 48 | — | — |
| MTP serve a, 1 draft, tok/s | 47.81 | no cell | n/a | 0.9% / — |
| MTP serve a, 2 drafts, tok/s | 51.30 | no cell | n/a | 0.4% / — |
| MTP serve a, 3 drafts, tok/s | 52.36 | **62.36** | **−16.0%** | 0.8% / 0.7% |
| MTP serve a, d3 TTFT p50 ms | 2,504 | **2,057** | +21.8% | — |
| MTP acceptance, serve d1 / d2 / d3 | 0.882 / 0.799 / 0.699 | 0.712 median (0.661–0.918) | — | — |
| MTP tokens/step, serve d1 / d2 / d3 | 1.882 / 2.597 / 3.097 | 2.95–3.65 (mean 3.17) | — | — |
| MTP `spec-bench` d0→d1→d2→d3 tok/s, q8 | 33.28 → 51.77 → 60.52 → **63.66** | no such harness | — | — |
| MTP `spec-bench` d0→d1→d2→d3 tok/s, f32 | 32.59 → 50.78 → 59.36 → **62.16** | no such harness | — | — |
| peak VRAM, CLI @512 MiB | 17,031 | **15,546** | +9.6% | 0 |
| peak VRAM, CLI @8,192 MiB | 18,395 | **15,809** | +16.4% | 0 |
| peak VRAM, serve a / b / c MiB | 20,271 / 24,288 / 25,380 | **16,728 / 21,417 / 22,638** | +21.2% / +13.4% / +12.1% | 0 |
| peak VRAM, MTP serve a d3 MiB | 21,072 | **18,624** | +13.1% | 0 |
| resident weight bytes | 15.42 GiB, 866 tensors | 15.42 GiB, 27.32 B params | equal | — |
| streamed bytes/token | 15,484,418,048 B (14.42 GiB) | not printed by llama.cpp | n/a | — |
| load time | 6.18–6.95 s engine load (5 values, median 6.20) | 6.85 / 7.01 / 7.07 s start→loaded (a / b / c), 8.09 s with the drafter | n/a | — |

MTP speed-up over each engine's **own** plain cell a: hipml **1.51× / 1.62× /
1.65×** at drafts 1 / 2 / 3; llama.cpp **1.94×** in its one MTP cell. hipml's
serve-path speed-up is again smaller than its own kernel-path speed-up
(`spec-bench` 1.56× / 1.82× / 1.91×) — see the caveats.

**Cell a is a per-stream median; cells b and c are aggregate goodput.** They are
different metrics computed by the same harness (`serve_bench.py:summarize`) and
must not be compared with each other — only a-to-a and b-to-b.

The main table and the spread table both use the per-wave median; the harness's
own `summary.aggregate_tok_s_p50` agrees with it to ≤0.05% in every cell
(largest gap: MTP d1, 47.83 against 47.81) and is kept beside it in the results
file.

### What the spread does to the reading

- **Prefill @512 — both readings, together.** The +11.8% is the median of the
  full set, but llama's spread there is 16.1% (194.7 tok/s), driven entirely by
  the first request of each process (1035.0, 1085.6, 1088.4). The full-set
  delta is 143.3 tok/s against that 194.7 spread, so the strict rule says
  **indistinguishable**. Drop each process's cold first request and llama is
  1,219.9 with a 1.9% spread against hipml's 1,357.6 at 1.6%, a delta of
  137.7 tok/s = **+11.3%**, and that lead holds. Both readings are in the
  table; do not quote one without the other. The warm-up repeats in a third
  process on both engines, in the same direction and the same magnitude, so it
  is warm-up inside the process rather than thermal decay.
- **Prefill @8,192.** The −1.9% is 25.3 tok/s against spreads of 6.4 and 11.6,
  so it is separable. hipml is flat 512→8,192 (−1.9%); llama gains 11.8% over
  the same range.
- **Serve b.** llama's spread (9.05 tok/s) is smaller than the delta (17.5), so
  the cell is separable even though it is noisy. A re-run of the same cell put
  its median at 105.89, which widens the lead to +24.6% rather than narrowing
  it. The table quotes the first run, which is the conservative one.
- **Serve c — indistinguishable on throughput.** The delta is 6.7 tok/s against
  llama's 39.15 spread, and the re-run gives −7.0 against 13.77, so
  **throughput is a tie**, not a win or a loss. The latency rows are a separate
  claim and are safe: hipml's own spread in this cell is 0.52 tok/s (0.2%).
- **Everything else.** Decode spreads are 0.24%–1.1%, all hipml serve cells are
  at or below 0.9%, and peak VRAM is identical across reps (spread 0).

## Roofline

Decode streams every weight through the memory bus for every token, so the
arithmetic is short. hipml's own counter says one token streams
**15,484,418,048 B** (14.42 GiB). Against the **30.10 ms** the decode table
measures, that is **514.4 GB/s — 80.4% of the card's 640 GB/s spec**. The
ablation's instrumented plain step is **30.02 ms**, which gives **515.7 GB/s,
80.6%**; the 0.08 ms between them is the gap between an instrumented step and
the wall clock. Either denominator puts decode at about four fifths of the spec,
and the conclusion does not depend on which is used. llama.cpp prints no
per-token stream counter, so that row has no llama value rather than a
comparable one.

That is why the +3.0% at depth 0 is the last few percent and not a new regime:
both engines sit at roughly four fifths of the same DRAM wall. The remaining
levers are bytes per weight and per-step overhead. It is also why the prefill
win does not transfer — a prefill tile keeps a weight tile in shared memory and
multiplies 256 prompt tokens against it, while a decode GEMV reads the same
bytes once and discards them. The measurement history, including the 633.6 GB/s
fp32 GEMV result that established the memory path, is in
[`notes/roofline-log.md`](notes/roofline-log.md).

## Which path each serve cell took

The trace line on this tree carries the route word, so the route is read, not
inferred from the width: `[batch] <entry> m=.. k=.. b=.. route=<exact|skinny>`.
Per-cell counts, 33 projection entries per width:

| cell | banner chunk | routes seen (count) | widths seen | path |
| --- | ---: | --- | --- | --- |
| serve a plain | 512 | `exact` 33 | b=1 | **exact** single-row GEMV |
| serve b plain | 512 | `exact` 66, `skinny` 99 | b=1, 2 / 6, 7, 8 | **approximate** WMMA tile at b≥6 |
| serve c plain | 512 | `exact` 66, `skinny` 99 | b=1, 2 / 14, 15, 16 | **approximate** WMMA tile at b≥14 |
| MTP d1 (serve a) | 512 | `exact` 66 | b=1, b=2 | **exact** batched GEMV |
| MTP d2 (serve a) | 512 | `exact` 66 | b=1, b=3 | **exact** batched GEMV |
| MTP d3 (serve a) | 512 | `exact` 66 | b=1, **b=4** | **exact** — the 4-column verify holds `exact_scope` |

Cells b and c take `route=skinny` at their wide steps, and every MTP verify
takes `route=exact` — including d3 at b=4, which was on the approximate tile in
the `409f4ad` session. That is the `exact_scope` fix and it is a real behaviour
change, not drift.

The server banner states a **capability**, not a per-call fact: the WMMA tile is
available at B≥4 unless `exact_scope` holds. It used to say `batch exact: no
(WMMA tile at B>=4)` in every cell, including the d3 cell where the trace shows
b=4 ran `route=exact`; that line is now worded as the capability it is and
points at `HIPML_BATCH_TRACE=1` for the per-call route. Read the route word, not
the banner.

llama.cpp has no equivalent switch; all its numbers are its production path.

## Caveats

- **The serve prefill chunk is not the CLI chunk.** These serve rows were
  measured at the server's default `--chunk 512`; the CLI prefills at 2,048.
  llama-server has no chunk knob but runs `-b 2048 -ub 512` by its own defaults,
  so the 512-token *physical* tile is now matched. hipml's TTFT fell 2,500 →
  2,130 ms (cell a) and 2,879 → 2,504 ms (MTP d3) against the `409f4ad` session
  as a result, and the cell-a TTFT gap to llama narrowed from +30.1% to +11.0%.
  llama may still schedule more logical tokens per ubatch than hipml's fixed
  tile. Do not read the serve TTFT gap as a prefill-quality gap, and do not
  re-pin the harness to a chunk: it measures the default. See
  [`notes/serve-prefill-chunk.md`](notes/serve-prefill-chunk.md).
- **TTFT is not a prefill number.** It includes HTTP, admission, queueing and
  the first token. Only the CLI rows are prefill like-for-like.
- **Cells b and c run the approximate tile.** hipml's batched path uses the WMMA
  tile at b=8 and b=16, approximate by design; cell a and all three MTP rows are
  the exact path, including the d3 verify at b=4. llama.cpp runs its own kernel.
- **llama's b and c medians are the middle of a noisy distribution.** Its
  per-wave aggregates swing 7.9–11.0% (b) and 6.0–18.1% (c) across two runs
  each, and the two medians are 7.5% (b) and 6.0% (c) apart. hipml's same cells
  swing 0.6% and 0.2%. The wave sequences oscillate rather than decay, so this
  is queueing noise in llama's batching, not a thermal trend. Cross-session
  comparisons of these cells are not measurements of change: the same cell and
  the same binary produced 234.61 in the `409f4ad` session and 215.70 / 229.45
  here.
- **Cell a and cells b/c are different metrics.** `serve_bench.py` computes cell
  a as the median of per-stream decode rates (so it equals single-stream decode)
  and b/c as `sum(decode_tokens) / (last − first)` across the wave (aggregate
  goodput). They must not be compared to each other.
- **MTP depth is matched at 3, but only at 3.** llama.cpp's
  `--spec-draft-n-max` default is 3 and its own log reports 2.95–3.65 accepted
  tokens per step, so llama-at-default is llama-at-3. hipml's d1 and d2 rows have
  no llama counterpart.
- **The d3 MTP row is exact on both sides of hipml's own path.** All three draft
  counts trace `route=exact` at b=1+2, b=1+3 and b=1+4: `serve` sets only
  `HIPML_BATCH_SKINNY` while `skinny_on()` reads `HIPML_SKINNY`, so
  `exact_scope` holds the verify columns and the verify runs the exact
  `gemv_*_b` entries at every draft count
  ([`notes/serve-ud-batch.md`](notes/serve-ud-batch.md)). Acceptance is nearly
  equal (0.699 against 0.712), so the −16.0% is per-step cost, not draft
  quality: at b=4 the scalar GEMV issues four FMAs per weight value. A control
  run with `HIPML_SKINNY=1` forced on does diverge — 1 of 48 replies at drafts
  3, first differing id at position 5 — so the protection is load-bearing.
  `spec-bench` exits 1 on a drafts row that is not exact.
- **llama.cpp's MTP exactness was not verified.** Its speculative ids were not
  diffed against its own plain greedy; hipml's `spec-bench` 128/128 and
  `probe_exact` 48/48 do not transfer to it.
- **`spec-bench` has no llama.cpp counterpart.** Those rows are hipml-only;
  llama.cpp's only MTP evidence here is the one serve cell.
- **hipml's serve path costs it about 11 tok/s against its own kernel path at
  d3.** 52.36 in serve against 63.66 in `spec-bench`. Different harness,
  different chunk, different prompt and a 64-token stream, so this is not a
  measurement of the serve overhead — it is a warning not to read either number
  across.
- **Peak VRAM is whole-process WDDM dedicated usage.** It includes prefill
  scratch (chunk 2048 on the CLI, chunk 512 in serve), the prefix cache and, on
  the MTP rows, the drafter. It is not a weights-versus-KV decomposition.
- **Load time is not the same quantity.** hipml's is engine load (tensor upload
  plus graph build); llama.cpp's is process start to model loaded, read from its
  own log. hipml serve prints no load timestamp.
- **`prefix_hit_tokens_p50 = 0` for llama is a reporting gap, not a miss.**
  `serve_bench.py` reads `timings.prefix_hit_tokens`, which llama-server does not
  emit. Its log shows LCP-similarity slot selection on all 48 requests in cell b,
  40 of them at `f_sim_best = 1.000` and the rest at 0.989–0.992, so it is
  reusing the shared prefix.
- **The serve workload is not the CLI workload.** Cells b and c are concurrent
  chats sharing a ~2.4K-token system prompt through the prefix cache; cell a is
  sequential single-stream requests at 32K context. Neither is a bare prefill
  benchmark.
- **No cell paged.** No allocation failure, OOM, `to CPU` or `cpu buffer` line
  appears in any llama log, and peak VRAM stayed at or below 25,380 MiB of
  31.9 GiB. The only `error`-matching lines in any hipml `server.log` are
  connection aborts at keep-alive teardown; the harness recorded 0 failed
  requests in every cell.
- **One GPU, one OS, one build.** gfx1201, Windows 11, llama.cpp build 11820
  (`d7725699`). No second card, no Linux, no CUDA.

## Reproduce it

```powershell
$env:HIPML_GGUF   = "<UD-Q4_K_M.gguf>"
$env:LLAMA_SERVER = "<llama-server.exe>"      # build 11820 to match
bash bench/run.sh --depth 0,8192 --reps 3
python bench/serve/serve_bench.py --engine hipml --cell a --reps 10
python bench/serve/serve_bench.py --engine llama --cell a --reps 10
```

`bench/run.sh` runs locally when `HIPML_BENCH_HOST` is unset and over ssh when it
is set. The generated results file records the GPU, the driver, the binary
SHA-256 and the llama.cpp version next to every number; please attach it if you
report a result. The environment table is in [`usage.md`](usage.md).
