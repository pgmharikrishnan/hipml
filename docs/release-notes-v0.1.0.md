# hipml v0.1.0

hipml is a from-scratch LLM inference engine for AMD Radeon on Windows: a Rust
host and HIP kernels, no external crates, no PyTorch and no MLIR. It runs
Qwen3.8-27B Q4_K_M on one Radeon AI PRO R9700 and exists to answer one question
— can a small, readable engine match a mature one on the same hardware? The
[README](../README.md) is the tour; [`docs/release-run.md`](release-run.md) is
the measurement it quotes, and [`docs/reproduce.md`](reproduce.md) is the runbook.

On identical token ids, hipml prefills faster than llama.cpp in 11 of 12
prompt×depth cells (up to +17.2%) and reaches the first token sooner in every
cell (1.02–1.22×). Plain decode is 6.1–7.4% behind at shallow depth and 4.3%
ahead at 100,000 tokens; both engines sit on the DRAM roofline at about 30 tok/s.
MTP speculative decode reaches 1.53×–2.00× plain greedy and is token-exact, the
server aggregates 125.25 vs 94.75 tok/s at 8 slots, and at 16 slots it is a tie
(212.14 vs 214.41 tok/s) with a lower TTFT p50 (779 vs 1,237 ms). Every number
here is one cold run on one card — gfx1201, Windows 11 build 26200, driver
32.0.32015.7029, ROCm 7.2.1, llama.cpp build 11820 (`d7725699`) — so treat it as
a data point, not a promise about other hardware.

What is next: MTP at q8 KV and batch decode at depth are implemented but
unmeasured; the prefix cache's over-budget admission is a known unpatched bug;
and no GPU other than gfx1201 and no OS other than Windows 11 has been tried.
Reports from other RDNA4 cards are welcome as new rows. The engine, kernels,
server and bench harness are Apache-2.0 (see [NOTICE](../NOTICE)); no model
weights are distributed.
