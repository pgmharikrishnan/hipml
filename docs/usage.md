# Usage

`hipml` is one binary with subcommands. `hipml help` prints the list; this page
is the reference for the flags you are likely to type and the environment
variables every script reads.

## Build

```powershell
pip install --no-deps --target C:\rocm-sdk `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl `
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl
$env:HIP_PATH   = "C:\rocm-sdk\_rocm_sdk_core"   # any directory works; this is an example
$env:HIPML_ARCH = "gfx1201"
$env:PATH       = "$env:HIP_PATH\bin;$env:PATH"

cargo build --release -p hipml-cli --features hip
.\target\release\hipml.exe probe
```

Without `HIP_PATH` the build still succeeds and produces the CPU reference
engine (no kernels). `hipml probe` says which one you have.

## Environment

Every path is an environment variable or a flag. Unset means "ask", never a
silent fallback to somebody else's disk.

| variable | what it is | default if unset |
|---|---|---|
| `HIPML_GGUF` | the target model GGUF | **none: commands fail with "set HIPML_GGUF"** |
| `HIPML_MTP_GGUF` | the MTP drafter GGUF | none: MTP cells are skipped with a visible message |
| `HIPML_EXE` | the hipml binary used by scripts | `<repo>/target/release/hipml[.exe]` |
| `LLAMA_SERVER` | the llama.cpp server binary | none: llama.cpp cells are skipped with a message |
| `HIP_PATH` | ROCm SDK root | build skips the kernels; runtime scripts fail fast |
| `HIPML_ARCH` | offload architecture | `gfx1201` |
| `HIPML_SKINNY` | `1` opts the batched decode into the WMMA tile (approximate, not bit-identical) | unset = the exact `_b` path |
| `HIPML_BATCH_LEGACY` | `1` forces the server's batched path to the exact `_b` kernels | unset = the tile at four or more slots |
| `HIPML_BENCH_OUT` | raw bench output root | `<repo>/bench/out/<tool>` |
| `HIPML_GPU_LOCK` | optional lock script taking `acquire <label>`, `release <label>`, `status` | unset = no locking |
| `HIPML_BENCH_HOST` | ssh host for the remote bench | unset = run locally |
| `HIPML_BENCH_REMOTE_DIR` | checkout on that host | required in remote mode |
| `HIPML_LONGCTX_CORPUS` | long-context text for `llm_depth.py` | none; the committed corpus is `bench/prompts/longdoc.txt` |
| `HIPML_PYTHON` | interpreter `bench/run.sh` uses | `python3` |
| `HIPML_BASE_URL` | llama-server URL for the tokenizer scripts | `http://127.0.0.1:8092` |

Ports are flags with defaults, not secrets: 8092 for llama-server, 8093 for
hipml. `bench/_paths.py` is the one place these are resolved, so a script
imports it instead of defining its own constants.

## Commands

### Generate

```powershell
# GPU, chunked prefill, greedy
hipml generate --gguf $env:HIPML_GGUF --device `
  --text "Explain the difference between a mutex and a semaphore." `
  --tokens 128 --stop-eos

# the same ids through the CPU reference engine
hipml generate --gguf $env:HIPML_GGUF --cpu --text "..." --tokens 16
```

Flags: `--text` / `--ids` / `--ids-file`, `--tokens`, `--ctx`, `--chunk`
(default 2048; `0` keeps the token loop), `--kv f32|q8|q4`, `--topk`,
`--logprobs`, `--stop-eos`, `--stop-ids`, `--compare-cpu`, `--json`.

### Chat

Renders a conversation with the model's own template, then generates.

```powershell
hipml chat --gguf $env:HIPML_GGUF --messages conversation.txt --json
```

`conversation.txt` uses the `--- role` format:

```text
--- system
You are terse.
--- user
What is a semaphore?
```

Flags: `--messages` (or `--json-file`), `--tools`, `--enable-thinking`,
`--no-thinking`, `--reasoning-effort`, `--no-preserve-thinking`,
`--no-generation-prompt`, `--json`, `--out`, `--ids-out`.

### Tokenize

```powershell
hipml tokenize --gguf $env:HIPML_GGUF --text "hello" --json
hipml tokenize-batch --gguf $env:HIPML_GGUF --corpus corpus.txt --out ids.json
```

### Serve

```powershell
hipml serve --gguf $env:HIPML_GGUF --mtp $env:HIPML_MTP_GGUF `
  --port 8093 --slots 8 --ctx 16384
```

Endpoints, headers, metrics and failure scopes: [`server.md`](server.md).

### Benchmarks

| command | measures |
|---|---|
| `hipml parity [filter] [--deep]` | every kernel against its CPU twin |
| `hipml oracle-gate --gguf … --ids-file … --steps N` | production chunked prefill vs the CPU oracle, step by step |
| `hipml gpu-bench --gguf …` | prefill/decode at a depth |
| `hipml gpu-sweep --gguf …` | launch-boundary cost |
| `hipml spec-bench --gguf … --mtp … --ids-file …` | plain greedy vs 1..N drafts, acceptance and exactness |
| `hipml batch-bench --gguf … --slots 8 --check` | batched decode rows, exact path vs tile path |
| `hipml cache-bench`, `bench-kv`, `rows-check`, `attn-bench`, `attn-parity`, `ssm`, `gdn-parity`, `prefill-bench`, `prefill-model`, `skinny-bench`, `column-bench`, `gemv-check`, `gemv-bench`, `bench`, `decode-bench`, `load`, `logits`, `matbench` | the individual pieces; `hipml help` lists them all |

The like-for-like comparison against llama.cpp is the Python harness:

```powershell
$env:LLAMA_SERVER = "D:\llama.cpp\build\bin\llama-server.exe"   # build 11820 to match
bash bench/run.sh --depth 0,8192 --reps 3
python bench/validate_results.py
```

`bench/run.sh` builds and runs in this checkout when `HIPML_BENCH_HOST` is
unset; with it set it syncs `HEAD` to that host over ssh and pulls the raw
results back. The server, MTP, batch and tokenizer harnesses are under
`bench/serve/`, `bench/chat/` and `bench/tokenizer/`; each takes `--help`.

## GPU sharing

If the GPU is shared, point `HIPML_GPU_LOCK` at a script that takes
`acquire <label>` and `release <label>`; every harness calls it around a run
and releases it on failure. Unset means the GPU is yours for the run.
