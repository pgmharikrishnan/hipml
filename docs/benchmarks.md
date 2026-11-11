# Benchmarks

Every number here is measured, committed and re-derivable. The raw report is
[`bench/release/`](../bench/release/) and the prose summary is
[`release-run.md`](release-run.md). The method is in
[`bench/README.md`](../bench/README.md).

**Status: superseded.** The current numbers are the **release run** on the stock
`Qwen3.8-27B-Q4_K_M.gguf`, in
[`release-run.md`](release-run.md) and [`../bench/release/`](../bench/release/).
Quote only that file.

The table and per-feature rows below are the **historical** pre-release
matrix: hipml binary `5713f143…`, commit `cef7657`, on a Radeon AI PRO R9700
(gfx1201), Windows 11, ROCm 7.2.1, against llama.cpp build 11820 (commit
`d7725699`). They predate the projection-path correctness fix and name a
model that is not the release model, so they are kept only as engineering
history.

## Method

- **Identical inputs.** Each prompt is tokenized once by `hipml tokenize` and the
  token ids are committed in [`bench/fixtures/prompts.json`](../bench/fixtures/prompts.json).
  Both engines receive those ids; neither retokenizes at run time.
- **Depth.** Depth 0 is the prompt as-is. Deeper cells are
  `BOS + real document prefix + prompt`, truncated to the exact token count.
- **Cold reps.** Each hipml rep is a fresh engine (zeroed KV and recurrent
  state) with the weights still resident. Each llama rep is a request with
  `cache_prompt=false`; one extra warm request with `cache_prompt=true` is
  recorded separately as a cache-hit check, not as a prefill number.
- **KV.** q8-class on both engines: hipml `q8`, llama.cpp `-ctk q8_0 -ctv q8_0`.
  The formats are the same 8-bit class, not byte-identical, so this is a
  same-precision-class comparison.
- **Decode window.** Tokens 2..33 (32 intervals) of a 33-token greedy run.
- **Reps.** 3 cold reps at depths 0 and 8192 for all five prompts; 1 rep at
  32768 and 100000 for `code_edit` (marked `1` in the reps column).

## Headline matrix, `code_edit`

| depth | tokens | engine | TTFT ms | prefill tok/s | decode tok/s | peak GiB |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 672 | hipml | 561 (561–572) | **1,198** (1,175–1,199) | 29.2 (29.2–29.3) | 17.62 |
| 0 | 672 | llama.cpp | 664 (663–811) | 1,014 (845–1,016) | **30.2** (30.0–30.2) | 16.22 |
| 8,192 | 8,192 | hipml | 6,541 (6,527–6,586) | **1,252** (1,244–1,255) | 28.8 (28.7–28.8) | 18.84 |
| 8,192 | 8,192 | llama.cpp | 6,930 (6,912–7,000) | 1,184 (1,173–1,188) | **29.4** (29.4–29.5) | 16.47 |
| 32,768 | 32,768 | hipml | 28,647 | **1,144** | 27.0 | 19.75 |
| 32,768 | 32,768 | llama.cpp | 30,736 | 1,067 | **27.4** | 17.38 |
| 100,000 | 100,000 | hipml | 107,777 | **928** | **23.4** | 21.93 |
| 100,000 | 100,000 | llama.cpp | 120,903 | 827 | 21.8 | 19.88 |

Gaps, hipml minus llama.cpp: prefill +18.1% / +5.7% / +7.2% / +12.1%, decode
−3.1% / −2.2% / −1.2% / +7.4%, TTFT ratio 1.18× / 1.06× / 1.07× / 1.12×.
This historical matrix was retired with `bench/RESULTS.md`; the release run's
full 5-prompt × 4-depth table is in [`bench/release/`](../bench/release/).

## Roofline

The card's spec bandwidth is 640 GB/s. Single-stream decode at depth 0 runs at
29.2 tok/s, which is about 83% of the 35.3 tok/s that the 17.19 GB of weights
allow at that ceiling; llama.cpp is at the same place (30.2 tok/s). That is the
reason plain decode is not the interesting axis: both engines are pinned to the
same DRAM roofline. The prefill win comes from doing more work per weight pass,
not from moving weights faster. The measurement history, including the 633.6
GB/s fp32 GEMV result that established the memory path, is in
[`notes/roofline-log.md`](notes/roofline-log.md).

## Per-feature tables (historical)

The rows in this section are the pre-release measurements. The release
cells are in [`release-run.md`](release-run.md).

| feature | result | source |
|---|---|---|
| MTP speculative decode, chat prompts, EOS stop, f32 KV | **1.46–1.80×** plain decode, median **1.58×**, token-exact vs greedy | [`notes/wp43-chat-parity.md`](notes/wp43-chat-parity.md) |
| server, 8 concurrent chats | **129.5** tok/s aggregate, TTFT p50 **491** ms, vs llama-server `-np 8` 95.3 tok/s / 637 ms | [`notes/wp45-51-serve.md`](notes/wp45-51-serve.md) |
| server, 16 concurrent chats | 214.4 tok/s vs 215.8 tok/s (a tie) | [`notes/wp45-51-serve.md`](notes/wp45-51-serve.md) |
| batch decode B=8, WMMA tile | **152.33** tok/s vs llama.cpp 102.03 (**+49.3%**) | [`notes/wp41-final-tree.md`](notes/wp41-final-tree.md) |
| batch decode B=8, bit-exact path | 92.25 tok/s (**−9.6%** vs llama.cpp) | [`notes/wp41-final-tree.md`](notes/wp41-final-tree.md) |

The MTP table is f32 KV. MTP at q8 KV is implemented but not verified, and
llama.cpp is run **without** its own draft-model speculation, so the comparison
is against plain greedy decode. The server rows are a warm shared-system-prompt
workload (prefix-cache hits), not cold prefill.

## Caveats

- **One-rep deep cells.** 32K and 100K are one cold rep each. Treat the range as
  absent, not as zero variance.
- **Same box, sequential runs.** The harness cannot control GPU clocks or
  thermal state, driver scheduling, or processes outside the GPU lock.
- **Peak VRAM** is the process's peak dedicated GPU memory from the WDDM
  counter, one number for the whole cell.
- **hipml uses more VRAM** than llama.cpp in every cell: the q8 KV layout plus
  the recurrent state.
- **Single prompt class at depth.** 32K and 100K use `code_edit` only.

## Reproduce it

```powershell
$env:HIPML_GGUF   = "D:\models\Qwen3.8-27B-Q4_K_M.gguf"
$env:LLAMA_SERVER = "D:\llama.cpp\build\bin\llama-server.exe"   # build 11820 to match
bash bench/run.sh --depth 0,8192 --reps 3
bash bench/run.sh --depth 32768,100000
python bench/validate_results.py
```

`bench/run.sh` runs locally when `HIPML_BENCH_HOST` is unset and over ssh when it
is set. The generated results file records the GPU, the driver, the binary
SHA-256 and the llama.cpp version next to every number; please attach it if you
report a result. The environment table is in [`usage.md`](usage.md).
