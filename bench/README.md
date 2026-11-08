# bench: hipml vs llama.cpp, same prompts

Every previous hipml number was measured on a synthetic prompt or with a
pre-filled cache, while the llama.cpp numbers were `llama-bench pp512/tg128`.
This bench removes both differences:

- five prompt classes, tokenised once by `hipml tokenize`;
- one JSON fixture (`fixtures/prompts.json`) holds the exact token ids;
- both engines receive those ids, so neither retokenises at run time;
- depths 0 / 8192 / 32768 / 100000 are built by prefixing real document text
  and truncating to the exact token count;
- hipml uses `generate --device`; llama.cpp uses `llama-server` on **8092**;
- each cell is its own row, with cold TTFT, prefill tok/s, decode tok/s over
  tokens 2..33, peak VRAM, and median/min/max over reps.

## Rerun

```bash
bench/run.sh                       # all prompts, all depths, both engines
bench/run.sh --depth 8192 --engine llama
bench/run.sh --engine hipml --prompt code_edit --reps 5
bench/run.sh --force               # rerun cells that already have results
```

The script syncs `HEAD` to `rtx`, builds, and runs one
(engine, prompt, depth) job per GPU-lock acquisition. That keeps every lock
hold short and makes the 100K cells resumable.

Prerequisites on the host:

```
C:\models\Qwen3.8-27B-Q4_K_M.gguf
C:\llama.cpp\rocm-gfx120x\llama-server.exe   (build 11820, commit d7725699)
C:\models\gpu-lock.ps1
```

Environment is set by `remote/with-env.cmd`:

```
HIP_PATH=C:\rocm-sdk\_rocm_sdk_core
HIPML_ARCH=gfx1201
PATH=C:\rocm-sdk\_rocm_sdk_core\bin;%PATH%
```

## What is measured

| field | hipml | llama.cpp |
| --- | --- | --- |
| TTFT | time around `prefill_chunked` (includes first-token argmax) | request start to first streamed token |
| prefill tok/s | prompt tokens / prefill time | `timings.prompt_per_second` |
| decode tok/s | 32 `forward_token_id` calls / their sum | `timings.predicted_per_second` (generation loop) |
| peak VRAM | WDDM `GPU Process Memory \ Dedicated Usage` sampler | same |
| reps | 5 cold `GpuEngine`s in one process (zeroed KV/state) | 5 `cache_prompt=false` requests |

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
- hipml's chunked-prefill path requires f32 KV, so hipml runs f32. llama.cpp
  runs its default f16 KV. The dtype is recorded in every result. This is an
  as-shipped comparison, not a same-KV-dtype one: at 100K the f32 KV path costs
  llama.cpp most of its decode rate (the harness measured 6.36 tok/s with f32
  versus the ~22 tok/s the project's own f16 measurement reports). Run
  `--llama-kv f32` for the same-dtype sensitivity.
- llama-server adds HTTP handling to TTFT (small on localhost, not zero).
- hipml's TTFT includes the first-token argmax; llama's prompt timing does not.
- A 100K cell marked `1` rep was too slow to repeat inside the lock window.
  It is still a real cold prefill, never a cache-prefilled one.
