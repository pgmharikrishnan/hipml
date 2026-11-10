# Architecture

hipml is a from-scratch LLM inference engine for one model on one card: a Rust
host, HIP kernels, and no external crates. This page is the map of the current
engine. The development logs behind it are in [`docs/notes/`](notes/), starting
with [`m0-m2-architecture.md`](notes/m0-m2-architecture.md) for the original
design decisions.

## Crate map

| crate | what it owns | needs a GPU |
|---|---|---|
| `hipml-core` | GGUF reader, tokenizer, chat template, quant block formats, KV layout, the CPU f32 reference forward pass and the parity case table | no |
| `hipml-hip` | raw `amdhip64` FFI: device, streams, buffers, module load, launch | yes |
| `hipml-kernels` | 56 `.hip` kernels plus the launcher ABI and the build script that runs `hipcc --genco` | yes |
| `hipml-model` | `GpuEngine` (single stream, one graph launch per token) and `BatchEngine` (rows), the layer loop, MTP drafter, resident weights | yes |
| `hipml-cli` | the commands in [`usage.md`](usage.md) and the HTTP server in [`server.md`](server.md) | for GPU commands |

Without `HIP_PATH`, `cargo build` still succeeds: the two `build.rs` files skip
kernel compilation with a warning and you get the CPU reference engine, which
is exact but slow. That is the supported way to read and test the engine on a
machine without a Radeon.

## Data flow

```text
 HTTP clients ──► hipml serve ─┐            hipml generate / chat / *-bench
                 (admit, queue,│                      │
                  backpressure)▼                      ▼
 ┌──────────────────────── hipml-cli ─────────────────────────────┐
 │ scheduler: admit → prefill tiles (2048) → batched step | MTP   │
 │ prefix cache (KV rows + SSM snapshots) · failure scopes · /metrics
 └──────────────┬─────────────────────────────────────────────────┘
                ▼
 ┌──────────────────────── hipml-model ───────────────────────────┐
 │ GpuEngine (single stream, graph launch) · BatchEngine (rows)   │
 │ MTP drafter + one-pass verify · resident Q4_K/Q5_K/Q6_K/Q8_0   │
 └──────────────┬─────────────────────────────────────────────────┘
                ▼
 ┌──────── hipml-kernels ────────┐   ┌────────── hipml-core (no GPU) ─────────┐
 │ 56 .hip → hipcc --genco       │   │ GGUF mmap · tokenizer · chat template  │
 │ GEMV, WMMA prefill, split-K   │   │ quant formats · CPU f32 oracle forward │
 │ attention, GDN recurrence, KV │   │ parity harness (every kernel has a ref)│
 └──────────────┬────────────────┘   └────────────────────────────────────────┘
                ▼
        hipml-hip (raw amdhip64 FFI) ──► amdhip64.dll (ROCm) ──► R9700 / gfx1201
```

**Prefill.** Prompt tokens are cut into 2048-token tiles. Each tile runs the
whole model in one graph: embedding, then per layer the quantized GEMV and WMMA
tile GEMMs, full attention over the tile plus the cached prefix (split-K over
the context), and the gated-delta-net recurrence for the SSM layers. The tile
size is a flag (`--chunk`), because it is the main prefill knob.

**Decode.** One token per step. Weights stay quantized in VRAM and are
dequantized inside the GEMV that consumes them; the KV cache is quantized
(`q8` or `q4`) and the recurrent state is a fixed-size per-slot buffer. A
single stream runs one graph launch per token; the batched engine runs a
row-block over the active slots.

**MTP.** The model ships a multi-token-prediction head. It drafts 1–3 tokens
from the same weight stream and the target verifies all of them in one pass,
so a correct draft converts a weight pass into several tokens. The output is
token-exact against plain greedy; see
[`wp43-chat-parity.md`](notes/wp43-chat-parity.md).

**Batch and serve.** Two or more concurrent requests take the batched engine;
one lone request may take MTP. The server admits against slots plus a queue
bound, answers the overflow with 503 and `Retry-After` before tokenizing, and
keeps a prefix cache of KV rows plus SSM snapshots so a shared system prompt is
prefilled once.

## Memory budget

Measured on the release model (17,203,416,064 bytes), q8 KV, from
[`bench/RESULTS.md`](../bench/RESULTS.md):

| item | size |
|---|---|
| weights, resident and quantized | 16.0 GiB (17.19 GB) |
| KV, q8, 100K tokens | ≈3.2 GiB |
| recurrent (SSM) state | 156.9 MB per sequence |
| peak VRAM, single stream, depth 0 / 8K / 32K / 100K | 17.62 / 18.84 / 19.75 / 21.93 GiB |
| peak VRAM, 8-slot server | see [`wp45-51-serve.md`](notes/wp45-51-serve.md) |

Two arenas, no general allocator: one persistent device allocation per weight
tensor (uploaded once, never moved) and a per-graph activation arena sized at
load. There is no `hipMalloc` in the decode loop.

## Correctness model

The reference is the specification. Every kernel has a plain-Rust twin with the
same name, signature and layout, and `hipml parity` compares them from one case
table. `hipml parity --deep` adds the 100K-context cases.

Kernel-vs-kernel parity cannot catch a launch-configuration bug in the
production path, which is why there is a second gate: `hipml oracle-gate` runs
the production chunked prefill and the CPU f32 forward pass on the same ids, for
the same number of greedy steps, and compares the id streams and the logits.
The bug that motivated it is written up in
[`wp42-forward-parity.md`](notes/wp42-forward-parity.md).

## What it does not do

One model architecture, one quantization mix, one GPU target (`gfx1201`), one
OS. No training, no autograd, no multi-GPU, no sampling (greedy only in
v0.1.0), no general checkpoint import. See the honest-limits block in the
README for the measured consequences.
