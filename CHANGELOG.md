# Changelog

All notable changes to hipml are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); 0.x makes no API or
CLI stability promise.

## [0.1.0] - 2026-11-11

First public release. One model, one card: Qwen3.8-27B Q4_K_M on a Radeon AI
PRO R9700 (`gfx1201`, RDNA4) under Windows 11 and ROCm 7.2.1.

### Added

- A from-scratch Rust host and HIP kernel inference engine with no external
  crates: GGUF v3 loader, BPE tokenizer, chat template, quant block formats,
  CPU f32 reference forward pass and a parity harness.
- A GPU engine for the `qwen35` hybrid model: quantized GEMV and dequant-on-read
  for Q4_K/Q5_K/Q6_K/Q8_0, 2,048-token chunked prefill with WMMA tiles, split-K
  attention over the cached prefix, q8/q4 KV, and the gated-delta-net recurrence
  held in a fixed-size per-slot state.
- MTP speculative decode: 1–3 drafts from the model's own prediction head,
  verified in one weight pass, token-exact against plain greedy.
- Row-batched decode for concurrent sequences, with an exact path and an opt-in
  WMMA tile (`HIPML_SKINNY=1`, approximate; `HIPML_BATCH_LEGACY=1` forces the
  exact path).
- An HTTP server with continuous batching, MTP for a lone request, a hybrid
  KV + SSM prefix cache, a bounded queue with `503` and `Retry-After`,
  per-job failure scopes, `/metrics`, and a restart supervisor
  (`scripts/hipml-serve.ps1`).
- A like-for-like benchmark harness that feeds both engines identical token ids,
  plus `docs/release-run.md` (the numbers this release quotes) and
  `docs/reproduce.md` (the GPU runbook).
- Apache-2.0 licensing with `NOTICE`, the llama.cpp provenance and the
  `docs/reference/` audit sources.

### Changed

- The README, the benchmark page and the release notes now quote the release run
  on the stock `Qwen3.8-27B-Q4_K_M.gguf` instead of the pre-fix development
  numbers. Docs that still carry the older matrix are labelled historical.
- Stale documentation links point at `docs/notes/`, `docs/architecture.md`,
  `docs/usage.md`, `docs/server.md` and `docs/reproduce.md`; every relative link
  in the README resolves to a file in the tree.

### Fixed

- Prompts longer than one prefill tile could return wrong tokens: the chunked
  prefill was computing most SSM gate rows from zeros. The projection path is
  corrected, and `oracle-gate` now compares the production chunked path with the
  CPU f32 engine at depth.
- Server replies could diverge from plain greedy at token 20 when MTP was enabled
  with three drafts, because the verify pass used the approximate WMMA tile. MTP
  verification now uses the exact path and is token-exact in the server.
- A prompt whose length was one token more than a multiple of 2,048 could abort
  the process in split-K scratch sizing. The scratch is sized for the tail, and
  `splitk-gate` covers the boundary.
- The prefix cache could store a snapshot larger than the whole cache as a
  0-byte entry and then serve a shorter hit as if it were the full prefix. It
  now refuses an entry that cannot fit (counted in `refused`, never truncated),
  verifies the hit length on lookup, evicts LRU entries before refusing, and
  publishes the KV and SSM halves together or not at all. `cache-gate` replays
  the old consumer, requires it to diverge, and checks the fallback is exact.

### Known issues

- Plain single-stream decode is at or just behind llama.cpp at shallow depth:
  29.7 vs 31.8 tok/s at 672 tokens and 29.0 vs 30.9 tok/s at 8,192. Both engines
  sit on the DRAM roofline; hipml is ahead at 100,000 tokens, 23.7 vs 22.7 tok/s.
- At 16 slots the aggregate is a tie: 212.14 vs 214.41 tok/s (−1.1%), while TTFT
  p50 is 779 ms vs 1,237 ms.
- The fast batched tile is not bit-identical to single-stream decode. At B=8 it
  gives 149.43 tok/s against 107.46 tok/s for the exact path; use
  `HIPML_BATCH_LEGACY=1` when exactness matters.
- Peak VRAM is higher than llama.cpp: 16.94 GiB vs 15.50 GiB at depth 0, 18.17
  GiB vs 15.75 GiB at 8,192 tokens, and 26.28 GiB for the 16-slot server.
- The server is greedy-only (`temperature` must be 0, `n` must be 1), binds
  `127.0.0.1` by default, has no authentication, and is not hardened for the
  public internet.
- One card, one OS, one driver: all numbers are one cold run on gfx1201, Windows
  11 build 26200, driver 32.0.32015.7029. No other GPU, OS or driver has been
  measured.
- The 32,768- and 100,000-token cells are one cold rep each.
- MTP at q8 KV, batch decode at depth > 0, the full `mtp-batch-bench` matrix and
  servers above 16 slots have no release measurements.

[0.1.0]: https://github.com/pgmharikrishnan/hipml/releases/tag/v0.1.0
