# Server

`hipml serve` is an HTTP server with continuous batching, MTP speculation for a
lone request, and a hybrid KV + SSM prefix cache. It is built for one client
class: many concurrent chat sessions on one GPU, greedy decoding.

```powershell
hipml serve --gguf $env:HIPML_GGUF --mtp $env:HIPML_MTP_GGUF `
  --host 127.0.0.1 --port 8093 --slots 8 --ctx 16384
```

**It is not a hardened internet service.** It binds `127.0.0.1` by default, has
no authentication, and has been stress-tested for minutes, not hours. Report
security issues privately via GitHub Security Advisories.

## Flags

| flag | default | meaning |
|---|---|---|
| `--gguf PATH` | required | the model |
| `--mtp PATH` | none | the MTP drafter; without it every request takes the batched engine |
| `--host ADDR` | `127.0.0.1` | bind address |
| `--port N` | `8093` | listen port |
| `--ctx N` | `32768` | context per slot |
| `--slots N` | `16` | concurrent sequences |
| `--chunk N` | `256` | prefill tile size |
| `--drafts N` | `1` | MTP drafts per speculative step |
| `--prefix-entries N` | `16` | prefix-cache entries |
| `--prefix-mb N` | `2048` | prefix-cache budget |
| `--max-queue N` | `2 × slots` | requests allowed to wait for a slot |
| `--max-conns N` | `64` | open connections |

Environment: `HIPML_BATCH_LEGACY=1` forces the bit-exact batched path instead of
the WMMA tile (see *Exactness* below). `HIPML_FAULT_SLOT`, `HIPML_FAULT_STEP`,
`HIPML_FAULT_BATCH` and `HIPML_FAULT_BATCH_ALL` inject faults for the probe
suite; they are test hooks, not deployment knobs.

## Endpoints

| method | path | notes |
|---|---|---|
| `GET` | `/health`, `/v1/health` | `200` when the model is loaded and the scheduler is up |
| `GET` | `/metrics` | Prometheus text |
| `GET` | `/props` | model name, `n_ctx`, `n_slots`, `drafts` |
| `GET` | `/` | `{"status":"ok","hipml":true}` |
| `POST` | `/tokenize` | `{"content": "..."}` → ids |
| `POST` | `/v1/chat/completions`, `/chat/completions` | OpenAI-style; SSE with `"stream": true` |
| `POST` | `/completion` | llama.cpp-style; SSE |

Requests are greedy only: `temperature` must be 0 and `n`/`n_predict` must be 1,
or the server answers `400` and says it is greedy-only. `top_p`, `seed` and the
sampling fields are accepted and ignored so existing clients do not break.

## Backpressure

Admission is against slots plus `--max-queue`. A request that cannot get in is
answered **`503` with `Retry-After` before it is tokenized**, so a flood costs
the server nothing but a header write. Past `--max-conns` the connection is
closed rather than queued. `/metrics` counts the refusals in
`hipml_requests_rejected_busy_total` and the accepted work in
`hipml_requests_total`.

## Failure scopes

Every fallible step has a scope, so one bad request does not drop every stream
([`notes/wp45-51-serve.md`](notes/wp45-51-serve.md) has the injected-fault
evidence):

| scope | example sites | what happens |
|---|---|---|
| one job | admit, prefill tile, spec seed/step, adopt, activate, a corrupt id | that job gets an error event, its slot is freed, its prefix entries are dropped, the engine keeps serving |
| one batch | the shared weight pass | every job in the batch fails, the batch engine is reset, the next admission starts clean |
| the device | a reset that fails, or three resets with no successful step | every stream is drained with an error, the server waits up to 2 s for connections to close, then exits 1 |

The last scope is why a supervisor exists: `scripts/hipml-serve.ps1` restarts a
crashed server with a 2 s → 60 s backoff that resets after a stable run. It
restarts a crashed server, not a wedged one.

A prompt longer than the slot context is refused at admission with a `500` that
names the overflow, not a `400`, and it is on the known-gaps list.

```powershell
$env:HIPML_GGUF     = "D:\models\Qwen3.8-27B-Q4_K_M.gguf"
$env:HIPML_MTP_GGUF = "D:\models\mtp-Qwen3.8-27B-Q4_0.gguf"
powershell -ExecutionPolicy Bypass -File scripts\hipml-serve.ps1 -Slots 8 -Ctx 16384
```

It reads `HIPML_EXE`, `HIPML_GGUF`, `HIPML_MTP_GGUF`, `HIP_PATH` and
`HIPML_ARCH`, and logs to `<repo>/logs` unless `-LogDir` says otherwise.

## Exactness

Batched decode at B ≥ 4 uses a WMMA skinny GEMV that is 1.4x (B=8) to 2.5x
(B=16) faster than the exact kernels at fleet slot counts. It covers every
resident block type, including the Unsloth UD formats: the tile's five
UD-only entries (Q5_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S) are what keep a UD file
off the exact `_b` GEMV at 16 slots. It is not bit-identical to single-stream
decode at 4+ rows. `HIPML_BATCH_TRACE=1` names the kernel each projection
actually took, once per shape. The server says which path served a reply:

| header | meaning |
|---|---|
| `X-Hipml-Batch-Exact: 1` | the exact batched kernels produced this reply |
| `X-Hipml-Batch-Exact: 0` | the WMMA tile did, with `X-Hipml-Batch-Tile-Min: 4` |

A lone request, and every MTP verify, always take the exact path.
`HIPML_BATCH_LEGACY=1` forces exact for everything; it costs 28% at 8 slots
(107.46 against 149.43 tok/s) and still beats llama.cpp's 97.54 tok/s there,
but at 16 slots the exact path is 107.33 against llama.cpp's 227.89.

## Metrics

All names are prefixed `hipml_`. The ones an operator wants first:

| metric | type | meaning |
|---|---|---|
| `hipml_active_slots`, `hipml_queue_depth`, `hipml_inflight` | gauge | current load |
| `hipml_queue_max`, `hipml_inflight_max`, `hipml_connections_max` | gauge | configured bounds |
| `hipml_requests_total`, `hipml_requests_done_total`, `hipml_requests_failed_total`, `hipml_requests_cancelled_total` | counter | request outcomes |
| `hipml_requests_rejected_busy_total`, `hipml_connections_rejected_total` | counter | backpressure |
| `hipml_prompt_tokens_total`, `hipml_generated_tokens_total` | counter | work done |
| `hipml_prefill_chunks_total`, `hipml_prefill_seconds_total`, `hipml_prefill_interleaved_total` | counter | prefill path |
| `hipml_decode_steps_total`, `hipml_batch_steps_total` | counter | decode path |
| `hipml_batch_exact_path` | gauge | 1 when the last batch used exact kernels |
| `hipml_batch_resets_total`, `hipml_job_errors_total`, `hipml_slot_faults_total` | counter | failure scopes |
| `hipml_prefix_lookups_total`, `hipml_prefix_hits_total`, `hipml_prefix_misses_total`, `hipml_prefix_hit_rate`, `hipml_prefix_hit_tokens_total` | counter/gauge | prefix cache |
| `hipml_prefix_captures_total`, `hipml_prefix_refused_total`, `hipml_prefix_invalidated_total`, `hipml_prefix_capture_seconds_total`, `hipml_prefix_restore_seconds_total` | counter | prefix cache cost; `refused` counts a snapshot larger than the cache bound |
| `hipml_spec_steps_total`, `hipml_spec_drafted_total`, `hipml_spec_accepted_total`, `hipml_mtp_requests_total` | counter | MTP |
| `hipml_interleave_stall_seconds_total`, `hipml_interleave_stall_seconds_max` | counter/gauge | prefill/decode interleaving |

## What the numbers look like

8 concurrent chats on a warm shared system prompt: 125.25 tok/s aggregate and
518 ms TTFT p50, against llama-server `-np 8` at 94.75 tok/s and 497 ms p50
(p95 526 ms against 1,377). 16 concurrent chats is a tie (212.14 vs 214.41
tok/s), with a lower p50 (779 vs 1,237 ms). The method and the full table are in
[`release-run.md`](release-run.md); the raw cells are in
[`../bench/release/server/`](../bench/release/server/).
