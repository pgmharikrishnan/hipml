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
| [`wp52.md`](wp52.md) | WP52 | 2026-11-10 | the split-K ffn scratch was sized from the chunk, not the tile; `splitk-gate` | 8,193-token prefill completes in 6.72 s |
| [`mtp-batch.md`](mtp-batch.md) | WP50 | 2026-11-10 | MTP drafting per slot inside one verify pass; the schedule is `R = floor(16/B) - 1` | B=8 R=1 166.5 tok/s aggregate, 1.13× plain, token-exact |
| [`integrate.md`](integrate.md) | WP50–WP53 | 2026-11-11 | the split-K fix and MTP-over-batch merged onto the server tree | 20/20 oracle at step 19 = 41020; 8,193-token prefill 6.82 s |
| [`prefill-tile-ceiling.md`](prefill-tile-ceiling.md) | PFT | 2026-11-12 | what bounds the prefill GEMM tile, per weight format; no shape or decoder change retained | IQ4_XS 106.6 TF against a 148 TF compute-only floor; staging is 25-34% of each GEMM |
| [`serve-ud-batch.md`](serve-ud-batch.md) | SUD | 2026-11-12 | the 16-slot collapse on UD-Q4_K_M: the tile had no entry for five of its eight block types (Q5_K was 32% of the bytes); the follow-up moved the one-slot attention split count from the position to the cache stride and re-ordered the Q5_K tile's loads | cell c 97.42 → 201.59 → 220.65 tok/s; one slot 28.37 → 29.71 |
| [`decode-roofline.md`](decode-roofline.md) | DBR | 2026-11-12 | what separates the decode GEMV from a pure read at its own shapes; no repack retained | routed GEMV 96.6% of the same-geometry stream (26.98 vs 26.08 ms); 35 tok/s out of reach |
| [`decode-dispatch.md`](decode-dispatch.md) | DBD | 2026-11-13 | the 5.3 ms that is not the weight stream is kernel execution, not launch cost; graphs and a launch fusion both measured and the fusion reverted | one graph step vs 1,299 launches: +0.37 / −0.01 ms; 128 fewer launches: +0.19 ms |
| [`decode-epilogue.md`](decode-epilogue.md) | DEK | 2026-11-13 | the non-GEMV kernels by name; the GDN recurrence was on half the CUs; two more launch merges measured neutral | −0.80 ms at depth 0 (30.50 → 31.26 tok/s), −0.82 ms at 8,192; 32 tok/s still out of reach |
| [`decode-residual-gemv.md`](decode-residual-gemv.md) | DRG | 2026-11-13 | the 128 residual adds folded into the epilogue of the projection that produces their input: all 128 fold, the token does not move, and the residual pot the previous note budgeted was mostly the stage trace | 1,299 → 1,171 launches, −0.02 / −0.03 ms, reverted |
| [`decode-ablation.md`](decode-ablation.md) | DAB | 2026-11-13 | the decode budget by removing stages instead of tracing them; the in-situ weight stream is 543 GB/s not 585.6, and the non-GEMV work is 3.5 ms not 0.58 | `gemv_only` 28.52 ms, norm 1.48 + GDN 0.94 + attention 0.83; a bit-identical `rmsnorm_f32` −0.66 ms at both depths |
| [`serve-vram.md`](serve-vram.md) | SVR | 2026-11-13 | the Q5_K fp16 prefill copy was off by default and unreachable on the UD file, so deleting it changes no peak; the 1.6–2.8 GiB over llama-server is a second KV/SSM engine, a 641.5 MiB prefix snapshot and a 501 MiB drafter | peaks 20,592.5 / 24,284.9 / 25,286.9 vs llama 17,789.0 / 22,489.6 / 23,668.2; unchanged by the deletion |
