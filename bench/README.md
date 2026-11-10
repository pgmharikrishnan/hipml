# bench: hipml vs llama.cpp, same prompts

Every previous hipml number was measured on a synthetic prompt or with a
pre-filled cache, while the llama.cpp numbers were `llama-bench pp512/tg128`.
This bench removes both differences:

- five prompt classes, tokenised once by `hipml tokenize`;
- one JSON fixture (`fixtures/prompts.json`) holds the exact token ids;
- both engines receive those ids, so neither retokenises at run time;
- depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use
  `code_edit` at 1 rep and are marked;
- both engines use q8-class KV at every depth (hipml `q8`, llama
  `-ctk q8_0 -ctv q8_0`), so the head-to-head is same-precision;
- hipml uses `generate --device`; llama.cpp uses `llama-server` on **8092**;
- each cell is its own row, with cold TTFT, prefill tok/s, decode tok/s over
  tokens 2..33, peak VRAM, KV type, spill status, and median/min/max over reps.

## Rerun

```bash
bench/run.sh                       # all prompts, all depths, both engines
bench/run.sh --depth 8192 --engine llama
bench/run.sh --engine hipml --prompt code_edit --reps 5
bench/run.sh --force               # rerun cells that already have results
```

The script runs one (engine, prompt, depth) job per optional GPU-lock
acquisition, which keeps every hold short and makes the 100K cells resumable.
With no host set it builds and runs in this checkout; with
`HIPML_BENCH_HOST` (or `--host`) it syncs `HEAD` there over ssh, builds on that
host and pulls the raw results back.

Everything machine-specific is an environment variable, and nothing here falls
back to one machine's layout:

| variable | what it is | default if unset |
|---|---|---|
| `HIPML_GGUF` | the target GGUF | required: `run.sh` exits and says so |
| `HIPML_MTP_GGUF` | the MTP drafter GGUF | unset: MTP cells are skipped |
| `LLAMA_SERVER` | the llama.cpp server binary | required only for `--engine llama` |
| `HIPML_EXE` | the hipml binary | `<repo>/target/release/hipml[.exe]` |
| `HIP_PATH` | ROCm SDK root | the build skips the kernels |
| `HIPML_ARCH` | offload arch | `gfx1201` |
| `HIPML_GPU_LOCK` | optional lock script | unset = no locking |
| `HIPML_LOCK_LABEL` | label passed to the lock | `bench` |
| `HIPML_BENCH_HOST` | ssh host | unset = local mode |
| `HIPML_BENCH_REMOTE_DIR` | checkout on that host | required in remote mode |
| `HIPML_BENCH_OUT` | raw output root | `<repo>/bench/out/<tool>` |

The comparison baseline is llama.cpp **build 11820, commit `d7725699`**, HIP
backend for gfx120x. `bench/remote/with-env.cmd` reads `HIP_PATH` and
`HIPML_BENCH_REMOTE_DIR` and fails when either is unset.

## What is measured

| field | hipml | llama.cpp |
| --- | --- | --- |
| TTFT | time around `prefill_chunked` (includes first-token argmax) | request start to first streamed token |
| prefill tok/s | prompt tokens / prefill time | `timings.prompt_per_second` |
| decode tok/s | 32 `forward_token_id` calls / their sum | `timings.predicted_per_second` (generation loop) |
| peak VRAM | WDDM `GPU Process Memory \ Dedicated Usage` sampler | same |
| KV | `q8` | `q8_0` |
| reps | 3 at 0/8192, 1 at 32768/100000; fresh `GpuEngine` each rep | same rep counts, `cache_prompt=false` |

llama.cpp also gets one extra `cache_prompt=true` request after the cold reps.
That is a cache-hit TTFT, reported separately; it is not a prefill number.

Depth 0 is the prompt as tokenised. At depth D > 0 the sequence is

```
document_tokens[:D - len(prompt)] + prompt
```

with no BOS inserted by the server (the model's `add_bos_token` is false), so
the total is exactly D tokens.

## Files

```
bench/
  prompts/             five prompt texts + the real document
  fixtures/prompts.json  committed token ids and per-cell hashes
  _paths.py            the one place an environment path is resolved
  prepare_fixture.py   regenerate the fixture (hipml tokenize)
  job.py               one cell, inside the GPU lock
  aggregate.py         raw jobs -> results.json + RESULTS.md
  run.sh               the one command
  ps/peak-vram.ps1     per-process peak dedicated VRAM
  out/                 raw job output (gitignored)
  results.json         committed machine-readable result
  RESULTS.md           committed table
```

## Caveats the harness cannot fix

- GPU clocks and thermals drift; runs are sequential, but not
  temperature-controlled beyond the lock.
- hipml and llama.cpp both run q8-class KV (`q8` vs `q8_0`). The formats are
  the same 8-bit class but not byte-identical, so this is a same-precision
  comparison. The f32/f16 sensitivity runs are kept under `bench/sensitivity/`.
- At 100K, q8 KV is ~3.2 GiB (hipml) and ~3.5 GiB (llama.cpp). `RESULTS.md`
  reports peak VRAM and a spill flag; `spilled=no` means the server logged no
  allocation failure or CPU offload and peak stayed below 24 GiB.
- llama-server adds HTTP handling to TTFT (small on localhost, not zero).
- hipml's TTFT includes the first-token argmax; llama's prompt timing does not.
- A 100K cell marked `1` rep was too slow to repeat inside the lock window.
  It is still a real cold prefill, never a cache-prefilled one.
