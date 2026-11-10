# hipml

**A from-scratch LLM inference engine for AMD Radeon on Windows — Rust host,
HIP kernels, zero external crates.**

One model, one card: hipml runs Qwen3.8-27B Q4_K_M on a Radeon AI PRO R9700 and
prefills faster than llama.cpp in 11 of 12 measured cells (up to +17.2%), reaches
the first token sooner in every cell (1.02–1.22×), and tells you exactly where it
still loses.

## What it is

hipml is a Rust host plus a set of HIP kernels that load a GGUF file, tokenize,
render the model's chat template, and run the full forward pass on an AMD GPU.
There is no PyTorch, no ONNX, no MLIR, no graph compiler, no Python in the hot
path, and no external crate in the dependency graph. The model is a `qwen35`
hybrid: 16 full-attention layers, a gated-delta-net recurrence for the rest, and
a multi-token-prediction head. It serves Qwen3.8-27B Q4_K_M on one R9700, and the
numbers below compare it with llama.cpp on identical token ids.

## Numbers

| context depth | reps | prefill tok/s hipml | llama.cpp | Δ | TTFT ms hipml | llama.cpp | Δ (llama/hipml) | decode tok/s hipml | llama.cpp | Δ | peak GiB hipml / llama |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (672 tokens) | 3 | 1,172 | 1,000 | +17.2% | 574 | 698 | 1.22× | 29.7 | 31.8 | −6.6% | 16.94 / 15.50 |
| 8,192 | 3 | 1,225 | 1,160 | +5.6% | 6,689 | 7,080 | 1.06× | 29.0 | 30.9 | −6.1% | 18.17 / 15.75 |
| 32,768 | 1 | 1,133 | 1,048 | +8.2% | 28,923 | 31,300 | 1.08× | 27.4 | 28.7 | −4.6% | 19.08 / 16.66 |
| 100,000 | 1 | 920 | 813 | +13.1% | 108,698 | 122,997 | 1.13× | 23.7 | 22.7 | +4.3% | 21.26 / 19.16 |

`code_edit` prompt, `Qwen3.8-27B-Q4_K_M.gguf` (16,810,714,336 bytes), AMD Radeon
AI PRO R9700 (gfx1201, 32 GiB), driver 32.0.32015.7029 (2026-10-18), Windows 11
Pro build 26200, ROCm 7.2.1, llama.cpp 0.7.0-dev build 11820 (commit `d7725699`,
Clang 24.0.0). q8-class KV on both engines (hipml `q8`, llama.cpp `q8_0`), no
draft model on either side, cold reps. 0 and 8,192 are the median of 3; 32,768
and 100,000 are 1 rep each. Positive prefill/decode Δ means hipml is faster.

The rest of the matrix, and the other measured paths, are in
[`docs/release-run.md`](docs/release-run.md): hipml wins prefill in 11 of 12
prompt×depth cells (the loss is `copy_200_words` at depth 0, 831 vs 848 tok/s).
MTP speculative decode reaches 1.53×–2.00× plain greedy over five chat
conversations, token-exact in every cell at f32 KV. The server aggregates
41.19 vs 31.58 tok/s at 1 slot, 125.25 vs 94.75 tok/s at 8 slots, and 212.14 vs
214.41 tok/s at 16 slots. Batch decode at B=8 runs at 107.46 tok/s on the exact
path and 149.43 tok/s on the opt-in WMMA tile (llama.cpp `-np 8`: 97.54 tok/s);
at B=16 the figures are 107.33 and 272.97 vs 227.89 tok/s. A prefix-cache hit
restores 2,048 tokens in 2.0 ms versus 1,619.9 ms cold.

## Honest limits

- **Plain decode is at or just behind llama.cpp at shallow depth.** 29.7 vs 31.8
  tok/s at 672 tokens and 29.0 vs 30.9 tok/s at 8,192; both engines are pinned to
  the DRAM roofline. At 100,000 tokens hipml is ahead, 23.7 vs 22.7 tok/s.
- **16-slot aggregate is a tie.** 212.14 vs 214.41 tok/s (−1.1%), while TTFT p50
  is 779 ms vs 1,237 ms.
- **The fast batch tile is not bit-identical.** At B=8 it gives 149.43 tok/s
  against 107.46 tok/s for the exact path. Set `HIPML_BATCH_LEGACY=1` when you
  need the exact one.
- **VRAM is higher.** Peak is 16.94 GiB vs 15.50 GiB at depth 0 and 18.17 GiB vs
  15.75 GiB at 8,192 tokens; the 16-slot server peaks at 26.28 GiB.
- **The server is greedy-only and not hardened.** `temperature` must be 0 and
  `n` must be 1, or the request is refused. It binds `127.0.0.1` by
  default, has no authentication, and has been stress-tested for minutes, not
  hours. Do not expose it to the public internet.
- **One card, one OS, one driver.** Every number here is one cold run on
  gfx1201, Windows 11 build 26200, driver 32.0.32015.7029. No other GPU, OS or
  driver has been measured.
- **The deep cells are single-rep.** 32,768 and 100,000 tokens are 1 cold rep
  each; treat a small difference there as noise.
- **The default prefix-cache budget is 1 GiB.** An 8,192-token snapshot is
  1,174.6 MiB and is refused at capture; the refusal is counted, the ids stay
  exact, and an exact repeat of that prompt costs a full prefill (6,766.2 ms
  against 6,691.7 ms cold). Pass `--bytes-mib 4096` for the 4.6 ms hit (1,460×).
- **Some paths are unmeasured.** MTP at q8 KV, batch decode at depth > 0, the
  full `mtp-batch-bench` matrix, and servers above 16 slots have no release
  numbers.

## Requirements

- An AMD Radeon AI PRO R9700 (RDNA4, `gfx1201`, 32 GiB). Other RDNA4 cards have
  not been measured; the WMMA kernels target `gfx1201`.
- Windows 11 and the driver named above. ROCm 7.2.1 for Windows (the
  `rocm_sdk_core` + `rocm_sdk_devel` wheels).
- A Rust stable toolchain with the MSVC target, and Python for the bench
  scripts.
- At least 16.94 GiB of VRAM for the Q4_K_M weights and KV at depth 0; the
  measured single-stream peak is 21.26 GiB at 100,000 tokens and the 16-slot
  server peaks at 26.28 GiB. Leave headroom for the desktop compositor.

## Build

The ROCm Windows wheels give you `hipcc` without the driver-touching installer:

```powershell
pip install --no-deps --target C:\rocm-sdk `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl

$env:HIP_PATH   = "C:\rocm-sdk\_rocm_sdk_core"   # the directory with bin\ and lib\
$env:HIPML_ARCH = "gfx1201"
$env:PATH       = "$env:HIP_PATH\bin;$env:PATH"    # amdhip64.dll at run time

git clone https://github.com/pgmharikrishnan/hipml.git
cd hipml
cargo build --release -p hipml-cli --features hip
.\target\release\hipml.exe probe
.\target\release\hipml.exe parity
```

Without `HIP_PATH` the build still succeeds: both `build.rs` files skip kernel
compilation with a visible warning and you get the CPU reference engine, which
is exact but slow. `HIPML_ARCH` defaults to `gfx1201` when unset.

## Get the model

hipml ships no weights, no tokenizer files and no converted GGUFs. Download the
file the numbers were measured on from its publisher:

| file | bytes | SHA-256 | license |
| --- | ---: | --- | --- |
| `Qwen3.8-27B-Q4_K_M.gguf` | 16,810,714,336 | `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` | Apache-2.0 per its model card |

<https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q4_K_M.gguf>

```powershell
$env:HIPML_GGUF = "D:\models\Qwen3.8-27B-Q4_K_M.gguf"
Get-FileHash $env:HIPML_GGUF -Algorithm SHA256   # must match the table
```

The file is a standard GGUF v3: 866 tensors, 16,799,719,424 bytes of tensor data
(15.65 GiB), all supported. MTP cells additionally need the model's MTP-head
GGUF (`mtp-Qwen3.8-27B-Q4_0.gguf`); v0.1.0 publishes no download for it.

## Run

```powershell
# one prompt, GPU, greedy
.\target\release\hipml.exe generate --gguf $env:HIPML_GGUF --device `
  --text "Explain the difference between a mutex and a semaphore." `
  --tokens 128 --stop-eos

# one chat turn rendered with the model's own template
@"
--- user
What does a mutex protect that a semaphore does not?
"@ | Set-Content conversation.txt
.\target\release\hipml.exe chat --gguf $env:HIPML_GGUF --messages conversation.txt

# the server: OpenAI-style /v1/chat/completions, llama-style /completion, SSE
.\target\release\hipml.exe serve --gguf $env:HIPML_GGUF `
  --host 127.0.0.1 --port 8093 --slots 8 --ctx 16384

curl.exe http://127.0.0.1:8093/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":64,\"temperature\":0}"
```

Keep the server running across crashes with the restart supervisor. It reads
`HIPML_EXE`, `HIPML_GGUF`, `HIPML_MTP_GGUF` and `HIP_PATH` from the environment,
backs off on repeated young exits, and writes timestamped logs under `logs\`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\hipml-serve.ps1 `
  -Slots 8 -Ctx 16384
```

## How it works

```text
 HTTP clients ──► hipml serve ─┐          hipml generate / chat / *-bench
                 admit, queue  │                    │
                               ▼                    ▼
 ┌──────────────────────── hipml-cli ─────────────────────────────┐
 │ scheduler · prefill tiles · MTP or batched step · prefix cache │
 └──────────────┬─────────────────────────────────────────────────┘
                ▼
 ┌─────────────────────── hipml-model ────────────────────────────┐
 │ GpuEngine (one stream) · BatchEngine (rows) · MTP drafter      │
 └──────────────┬─────────────────────────────────────────────────┘
                ▼
 ┌──────── hipml-kernels ────────┐   ┌────── hipml-core (no GPU) ──────┐
 │ HIP: GEMV, WMMA prefill,      │   │ GGUF mmap · BPE tokenizer ·     │
 │ split-K attention, GDN, KV    │   │ chat template · quant formats · │
 └──────────────┬────────────────┘   │ CPU f32 reference forward       │
                ▼                    └─────────────────────────────────┘
        hipml-hip (raw amdhip64 FFI) ──► amdhip64.dll ──► R9700 / gfx1201
```

- **Host.** `hipml-cli` parses arguments, schedules work and owns the HTTP
  server. `hipml-model` owns the layer loop and the resident weights.
- **Tokenizer.** A BPE tokenizer and the model's chat template, including tool
  schemas, tool-call escaping and the thinking controls. It matched llama.cpp on
  6,912 strings and 18 of 18 template cases.
- **Loader.** The GGUF is mapped read-only and its tensors are validated and
  uploaded once; the decode loop does no allocation.
- **Quant kernels.** Q4_K, Q5_K, Q6_K and Q8_0 weights stay quantized in VRAM and
  are dequantized inside the GEMV that consumes them.
- **WMMA prefill.** Prompt tokens are cut into 2,048-token tiles. Each tile runs
  the whole model in one graph, with WMMA tiles over the quantized weights.
- **Attention and SSM.** Full-attention layers run split-K attention over the
  tile plus the cached prefix; the gated-delta-net layers keep a fixed-size
  recurrent state per sequence.
- **Decode.** One token per step, one graph launch per token, q8 or q4 KV.
- **MTP.** The model's own multi-token-prediction head drafts 1–3 tokens and the
  target verifies them in one weight pass; the output is token-exact against
  plain greedy.
- **Batching.** Two or more concurrent requests take the row-batched engine; a
  lone request may take MTP.
- **Cache.** The prefix cache stores KV rows plus SSM snapshots, so a shared
  system prompt is prefilled once.

The full crate map, memory budget and correctness model are in
[`docs/architecture.md`](docs/architecture.md); the command reference is
[`docs/usage.md`](docs/usage.md) and the server reference is
[`docs/server.md`](docs/server.md).

## Reproduce

The runbook is [`docs/reproduce.md`](docs/reproduce.md); the measured values and
the exact commands are in [`docs/release-run.md`](docs/release-run.md), and the
harness is [`bench/README.md`](bench/README.md). The bench feeds both engines the
same token ids and records the GPU, driver, binary SHA-256 and llama.cpp version
next to every number.

These numbers are one GPU, one OS and one driver. On the same card and the same
file the harness should reproduce these cells; the three-rep cells show their
min–max spread in [`docs/release-run.md`](docs/release-run.md), and the 32K and
100K cells are single cold reps, so a small difference there is noise. Other
hardware is welcome as new data, never as a replacement for the R9700 rows.

## Articles

1. [Beating llama.cpp at prefill: 11 cells of 12, and a 6% decode bill](docs/articles/01-beating-llama-cpp-prefill.md) — what the 2,048-token tile buys, and what the decode path pays for it.
2. [The bug that survived every parity gate: 40 of 48 heads running on zeros](docs/articles/02-the-bug-that-survived-the-parity-gates.md) — a launch bug that passed kernel parity, and the oracle gate that caught it.
3. [Speculative decoding that is exact: 1.53× to 2.00× from the model's own draft head](docs/articles/03-mtp-speculative-decoding-exact.md) — the MTP drafter, the verify pass, and why the output never changes.
4. [Roofline first: why neither engine can decode faster on a 640 GB/s card](docs/articles/04-roofline-first.md) — the bandwidth budget behind the ~30 tok/s decode cell.
5. [Serving eight sessions on one 32 GB card](docs/articles/05-serving-8-agents-on-one-card.md) — continuous batching, its 16-slot tie, and the prefix-cache defect the release run found and fixed.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). The engine, the HIP
kernels, the server and the benchmark harness are Copyright 2026 Hari Krishnan
(pgmharikrishnan).

Parts of the model math, the tokenizer and the quant formats are ported from
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, Copyright (c) the ggml
authors). Those files say so in their headers, and `docs/reference/` keeps the
unmodified sources they were audited against, with their license in
[`docs/reference/LICENSE`](docs/reference/LICENSE); those copies are
documentation only and are not part of the build. The Unicode category table is
generated from llama.cpp's Unicode data (Unicode License v3); `NOTICE` lists
every derived file.

**Model weights are not included.** hipml is an engine: it ships no weights, no
tokenizer files and no converted GGUFs. The engine is tested on Qwen3.8-27B, on
the quantized file `Qwen3.8-27B-Q4_K_M.gguf`. Qwen3.8-27B is distributed by its
publisher under its own license; check that license before downloading. Your use
of the weights is governed by that license, not by hipml's Apache-2.0 license,
and the benchmark numbers here are measurements of the engine on that file, not
claims about the model.

ROCm is AMD's and is installed separately: no AMD ROCm or HIP SDK component is
redistributed in this repository. llama.cpp, Qwen, ROCm, Radeon, Windows and
Rust are their owners' trademarks, named here only to identify them.

## Citation and contact

If you use these numbers, cite the repository and the run they come from:
`hipml v0.1.0`, <https://github.com/pgmharikrishnan/hipml>, and
[`docs/release-run.md`](docs/release-run.md).

Questions and bug reports: open an issue at
<https://github.com/pgmharikrishnan/hipml/issues>. Security reports go through
GitHub Security Advisories, as described in [SECURITY.md](SECURITY.md).
