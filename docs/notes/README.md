# Engineering notes

These are the development logs, kept as provenance. Every claim carries its
measurement, its controls and what was left unverified. They read as internal
logs on purpose: work-package IDs and machine-specific paths have been removed
or sanitized, but the evidence is unchanged.

> *Engineering log from development; internal work-package IDs and paths have
> been sanitized or removed where they were machine-specific.*

Commit hashes quoted inside a note come from the pre-publication history: the
author identity was normalised at publication, so they cannot be resolved in
this repository. The measurements and controls are what the notes are for, and
the current numbers are in [`../release-run.md`](../release-run.md).

| file | work package | date | what it decided | headline number |
|---|---|---|---|---|
| [`m0-m2-architecture.md`](m0-m2-architecture.md) | M0–M2 | 2026-11-06 | HIP over Vulkan; no external crates; the reference is the specification | fp32 GEMV 633.6 GB/s, 99% of the 640 GB/s spec |
| [`roofline-log.md`](roofline-log.md) | M1–M3, many | 2026-11-09 | where the prefill and decode time goes; every roofline measurement | 1,198 tok/s prefill vs llama.cpp 1,014 at depth 0 |
| [`forward-op-map.md`](forward-op-map.md) | M3 | 2026-11-08 | which llama.cpp function each hipml op replaces | the whole forward pass mapped op by op |
| [`gpu-engine.md`](gpu-engine.md) | M3 | 2026-11-09 | the graph engine: resident weights, one launch per token | 29.2 tok/s decode at 83% of the roofline |
| [`ssm-gpu.md`](ssm-gpu.md) | M3 | 2026-11-08 | gated-delta-net recurrence on the device | 156.9 MB of recurrent state for all 48 SSM layers |
| [`fullattn-gpu.md`](fullattn-gpu.md) | M3 | 2026-11-09 | full attention without Flash Attention, split-K over context | per-layer time at 0 / 8K / 100K |
| [`mtp.md`](mtp.md) | M3 | 2026-11-09 | the model's own MTP head as the drafter | draft/verify design and acceptance |
| [`wp26-batch.md`](wp26-batch.md) | WP26 | 2026-11-09 | multi-sequence batching: rows share one weight pass | +53% aggregate at B=8 in that tree |
| [`wp35-merged-tree.md`](wp35-merged-tree.md) | WP35 | 2026-11-09 | the merged-tree measurement after the batch work | 59 tests, 0 failed |
| [`wp41-final-tree.md`](wp41-final-tree.md) | WP41 | 2026-11-09 | the final-tree batch numbers, tile vs exact path | 152.33 vs 102.03 tok/s (+49.3%) at B=8 |
| [`wp42-forward-parity.md`](wp42-forward-parity.md) | WP42 | 2026-11-09 | the launch bug that survived kernel parity; `oracle-gate` | 40 of 48 heads on zeros; 9.74 nats |
| [`wp43-chat-parity.md`](wp43-chat-parity.md) | WP43 | 2026-11-09 | MTP acceptance on chat-formatted prompts with EOS stop | 1.46–1.80×, median 1.58×, token-exact |
| [`wp45-51-serve.md`](wp45-51-serve.md) | WP45–WP51 | 2026-11-10 | the server: batching, prefix cache, backpressure, failure scopes | 129.5 vs 95.3 tok/s at 8 slots |
| [`review-levers.md`](review-levers.md) | review | 2026-11-09 | what is left to win, ranked by expected gain | WMMA skinny GEMM and MTP identified |
| [`review4-server-audit.md`](review4-server-audit.md) | review 4 | 2026-11-10 | adversarial server audit: exactness, malformed input, stress | the malformed-input cases are 400s, server stays up |
| [`wp48-tokenizer-fidelity.md`](wp48-tokenizer-fidelity.md) | WP48 | 2026-11-09 | the tokenizer and template fixtures | 266 strings and 16 conversations match llama |
