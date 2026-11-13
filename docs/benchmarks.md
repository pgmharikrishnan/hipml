# Benchmarks

Every number in the table below, and the identity that produced it, comes from
**one session on one card**: a single binary built from tree `409f4ad`, every
axis measured between 2026-11-13T11:56Z and 12:52Z, against llama.cpp build
11820 (commit `d7725699`). Nothing is carried over from an earlier run. The
model is `Qwen3.8-27B-UD-Q4_K_M.gguf` with q8-class KV on both engines.

The stock-model release run in [`release-run.md`](release-run.md) is the earlier
record of the release tree. The decode path this table measures changed
materially in eight landed slices since that run, so its decode and serve
columns are not comparable with these; its prefill columns are.

## Identity of the run

| field | value |
| --- | --- |
| date (UTC) | 2026-11-13T11:56Z → 12:52Z |
| hipml tree | `409f4ad`, `git status` empty |
| hipml binary | `hipml.exe`, 2,968,064 B, sha256 `3520aaa15790e2f1061743b16e824e711849c4930e2117c64ee1fcca29444e35` |
| build | `cargo build --release -p hipml-cli --features hip`, `HIPML_ARCH=gfx1201`, `built with hip feature: true` |
| model | `Qwen3.8-27B-UD-Q4_K_M.gguf`, 16,564,259,744 B, sha256 `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529` |
| MTP drafter | `mtp-Qwen3.8-27B-Q4_0.gguf`, 1,369,590,656 B, sha256 `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e` |
| stock Q4_K_M (gates only) | `Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,336 B, sha256 `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` |
| GPU | AMD Radeon AI PRO R9700, gfx1201, 31.9 GiB |
| driver / ROCm / OS | 32.0.32015.7029 / 7.2.1 / Windows 11 Pro build 26200 |
| llama.cpp | build 11820, commit `d7725699`, sha256 `6f6bf5e975dd3d5efb314f61688a8ea3b2f6516b0503c28b3783887f47d00ab3` |
| harness | `bench/job.py` and `bench/serve/serve_bench.py` as committed in `409f4ad` |

All three model hashes and the binary hash were re-read after the last cell and
were byte-identical to before it.

## Method

- **Gates first, so timing was allowed.** Greedy ids 12/12, `oracle-gate` 20/20
  on both models, `parity --deep` 74 cases, `parity prefill --deep` 15 cases,
  `batch-bench --slots 16 --check` 16/16 on every slot, and `spec-bench`
  128/128 exact at q8 and again at f32. All were run on the binary that was
  then timed, before any cell.
- **Identical inputs.** Both engines received byte-identical token ids, recorded
  as `ids_sha256` in every result file.
- **Reps.** CLI rows are 3 reps in each of 2 separate processes, so 6 values.
  Serve rows are the harness's wave aggregates: 10 waves in cell a, 6 in b, 3
  in c.
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
is below. Spread is max−min over the cell. Where a `Δ` is smaller than llama's
spread, the cell is not a result.

| axis | hipml | llama.cpp | Δ | spread hipml / llama |
| --- | ---: | ---: | ---: | --- |
| CLI prefill @512 tok/s | **1,346.1** | 1,215.2 | **+10.8%** | 2.3% / 13.5% |
| CLI prefill @512, cold first request dropped | **1,349.6** | 1,224.5 | **+10.2%** | 0.7% / 1.9% |
| CLI prefill @8,192 tok/s | 1,326.1 | **1,352.5** | −1.9% | 0.5% / 0.9% |
| CLI decode @0 tok/s | **33.15** (30.17 ms) | 32.24 (31.01 ms) | **+2.8%** | 0.13% / 1.0% |
| CLI decode @8,192 tok/s | **31.98** (31.27 ms) | 31.41 (31.84 ms) | **+1.8%** | 0.35% / 0.42% |
| serve a, 1 slot, aggregate tok/s | 31.64 | **32.12** | −1.5% | 0.3% / 0.5% |
| serve a, TTFT p50 / p95 ms | 2,500 / 2,512 | **1,921 / 1,936** | +30.1% / +29.8% | — |
| serve b, 8 slots, aggregate tok/s | **132.37** | 109.42 | **+21.0%** | 0.7% / 12.0% |
| serve b, TTFT p50 / p95 ms | **489 / 491** | 563 / 1,252 | −13.2% / −60.8% | — |
| serve c, 16 slots, aggregate tok/s | 222.13 | **234.61** | −5.3% | 0.7% / 11.2% |
| serve c, TTFT p50 / p95 ms | **742 / 744** | 1,123 / 2,550 | −34.0% / −70.8% | — |
| MTP serve a, 3 drafts, tok/s *(pending re-measurement)* | 52.32 | **62.33** | **−16.1%** | 1.0% / 1.7% |
| MTP `spec-bench` d0→d1→d2→d3 tok/s, q8 | 33.30 → 51.75 → 60.58 → **63.57** | no such harness | — | — |
| peak VRAM, CLI @512 MiB | 17,031 | **15,546** | +9.6% | 0 |
| peak VRAM, CLI @8,192 MiB | 18,395 | **15,809** | +16.4% | 0 |
| peak VRAM, serve a / b / c MiB | 20,091 / 24,110 / 25,201 | **16,728 / 21,417 / 22,608** | +20.1% / +12.6% / +11.5% | 0 |
| peak VRAM, MTP serve a d3 MiB *(pending)* | 20,892 | **18,624** | +12.2% | 0 |
| resident weight bytes | 15.42 GiB, 866 tensors | 15.42 GiB, 27.32 B params | equal | — |
| streamed bytes/token | 15,484,418,048 B (14.42 GiB) | not printed | — | — |
| load time | 6.17 / 6.42 s engine load | 6.89 / 7.00 / 7.05 s start→loaded (8.25 s with drafter) | level | — |

The two MTP rows are **pending re-measurement**: the serve-path exactness of the
d3 verify step is still open (see the caveats), so these figures are what this
session measured, not a final result.

### What the spread does to the reading

- **Prefill @512.** The +10.8% is the median of the full set, but llama's spread
  there is 13.5%, driven entirely by the first request of each process (1069.3,
  1108.5). Dropping each process's cold first request, llama is 1209.6–1233.3
  (spread 23.7, 1.9%) and hipml leads by **10.2%** against a 0.7% spread. Both
  readings are in the table; do not quote one without the other. The same
  warm-up repeats in a third process on both engines, in the same direction, so
  it is warm-up inside the process rather than thermal decay.
- **Serve b.** llama's spread (13.2 tok/s) is smaller than the delta (22.9), so
  the cell is separable even though it is noisy. A re-run of the same cell put
  its median at 106.78, 2.4% below the 109.42 in the table; that widens the lead
  to +24%, not narrows it.
- **Serve c.** The delta is 12.5 tok/s against llama's 26.4 spread, so
  **throughput is indistinguishable**. The latency claim is safe: hipml's own
  spread is 0.7%.
- **Everything else.** Decode spreads are 0.13%–1.0%, and peak VRAM is identical
  across reps (spread 0).

## Roofline

Decode streams every weight through the memory bus for every token, so the
arithmetic is short. hipml's own counter says one token streams
**15,484,418,048 B** (14.42 GiB). Against the **30.17 ms** the decode table
measures, that is **513.2 GB/s — 80.2% of the card's 640 GB/s spec**. The
ablation's instrumented plain step is **30.02 ms**, which gives **515.7 GB/s,
80.6%**; the 0.15 ms between them is the gap between an instrumented step and
the wall clock. Either denominator puts decode at about four fifths of the spec,
and the conclusion does not depend on which is used. llama.cpp prints no
per-token stream counter, so that row has no llama value rather than a
comparable one.

That is why the +2.8% at depth 0 is the last few percent and not a new regime:
both engines sit at roughly four fifths of the same DRAM wall. The remaining
levers are bytes per weight and per-step overhead. It is also why the prefill
win does not transfer — a prefill tile keeps a weight tile in shared memory and
multiplies 256 prompt tokens against it, while a decode GEMV reads the same
bytes once and discards them. The measurement history, including the 633.6 GB/s
fp32 GEMV result that established the memory path, is in
[`notes/roofline-log.md`](notes/roofline-log.md).

## Which path each serve cell took

The server labels its own batch path. Widths observed per `server.log`:

| cell | hipml batch widths | path |
| --- | --- | --- |
| serve a plain | `b=1` (33) | **exact** single-row GEMV |
| serve b plain | `b=8` (33) plus `b=7`, `b=6`, `b=2`, `b=1` tails | **approximate** WMMA tile at b=8 |
| serve c plain | `b=16` (33) plus `b=15`, `b=14`, `b=2`, `b=1` tails | **approximate** WMMA tile at b=16 |
| MTP d1 / d2 (serve a) | `b=1` + `b=2` / `b=3` | **exact** batched GEMV |
| MTP d3 (serve a) | `b=1` + `b=4` | **disputed** — see the caveat below |

llama.cpp has no equivalent switch; every one of its numbers is its production
path.

## Caveats

- **The serve prefill chunk is not the CLI chunk.** `serve_bench.py` pins hipml
  to `--chunk 256`; llama-server has no chunk knob. hipml's CLI prefill at chunk
  2048 is 1,346 tok/s, and its one-slot TTFT of 2,500 ms for a 2,423-token
  prompt implies about 969 tok/s. Do not read the serve TTFT gap as a
  prefill-quality gap.
- **TTFT is not a prefill number.** It includes HTTP, admission, queueing and
  the first token. Only the CLI rows are prefill like-for-like.
- **Cells b and c run the approximate tile.** hipml's batched path uses the WMMA
  tile at b=8 and b=16, approximate by design; cell a and the MTP plain step are
  the only exact-path hipml serve rows. llama.cpp's MTP is exact by construction.
- **llama's b and c medians are the middle of a noisy distribution.** Its
  per-wave aggregates swing 11–12% (b: 101.7–114.9, and 102.8–115.1 on a re-run;
  c: 211.4–237.8), while hipml's same cells swing 0.7%. The wave sequences
  oscillate rather than decay, so this is queueing noise in llama's batching,
  not a thermal trend.
- **Cell a and cells b/c are different metrics.** `serve_bench.py` computes cell
  a as the median of per-stream decode rates (so it equals single-stream decode)
  and b/c as `sum(decode_tokens) / (last − first)` across the wave (aggregate
  goodput). They must not be compared to each other.
- **MTP depth is matched at 3, but only at 3.** llama.cpp's
  `--spec-draft-n-max` default is 3 and its own log reports 2.95–3.65 accepted
  tokens per step, so llama-at-default is llama-at-3. hipml's d1 and d2 rows have
  no llama counterpart. Acceptance is nearly equal (0.699 against 0.712), which
  makes the gap per-step cost rather than draft quality.
- **The d3 MTP verify path is disputed, and that is why the row is pending.**
  The batch trace reports width 4 at three drafts — the approximate-tile
  threshold — and widths 2 and 3, both exact, at one and two drafts.
  [`notes/serve-ud-batch.md`](notes/serve-ud-batch.md) §8 records the verify pass
  as exact by construction (`HIPML_BATCH_SKINNY` is ignored for MTP verify).
  Both cannot be right; whether the trace label or the dispatch is wrong is
  unresolved.
- **llama.cpp's MTP exactness was not verified.** Its speculative ids were not
  diffed against its own plain greedy; hipml's `spec-bench` exactness does not
  transfer to it.
- **`spec-bench` has no llama.cpp counterpart.** Those rows are hipml-only;
  llama.cpp's only MTP evidence here is the one serve cell.
- **hipml's serve path costs it about 11 tok/s against its own kernel path at
  d3** (52.32 serve against 63.57 `spec-bench`). Different harness, different
  chunk, different prompt and a 64-token stream, so this is not a measurement of
  the serve overhead — it is a warning not to read either number across.
- **Peak VRAM is whole-process WDDM dedicated usage.** It includes prefill
  scratch (chunk 2048 on the CLI, chunk 256 in serve), the prefix cache and, on
  the MTP rows, the drafter. It is not a weights-versus-KV decomposition.
- **Load time is not the same quantity.** hipml's is engine load (tensor upload
  plus graph build); llama.cpp's is process start to model loaded, read from its
  own log. hipml serve prints no load timestamp.
- **`prefix_hit_tokens_p50 = 0` for llama is a reporting gap, not a miss.**
  `serve_bench.py` reads `timings.prefix_hit_tokens`, which llama-server does not
  emit. Its log shows LCP-similarity slot selection on all 48 requests in cell b,
  40 of them at `f_sim_best = 1.000`, so it is reusing the shared prefix.
- **The serve workload is not the CLI workload.** Cells b and c are concurrent
  chats sharing a ~2.4K-token system prompt through the prefix cache; cell a is
  sequential single-stream requests at 32K context. Neither is a bare prefill
  benchmark.
- **One GPU, one OS, one build.** gfx1201, Windows 11, llama.cpp build 11820
  (`d7725699`). No second card, no Linux, no CUDA.
- **No cell paged.** Peak VRAM stayed at or below 25,201 MiB of 31.9 GiB, and no
  log carries an allocation failure.

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
