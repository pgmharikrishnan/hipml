# Articles

Long-form write-ups of what was measured, why it came out that way, and what the
numbers do not cover. Each one carries its own provenance section; the current
headline table is in [`../benchmarks.md`](../benchmarks.md).

| # | article | subject |
| ---: | --- | --- |
| 01 | [Beating llama.cpp at prefill: 11 cells of 12, and a 6% decode bill](01-beating-llama-cpp-prefill.md) | the release matrix, prompt by prompt, and why the prefill win does not transfer to decode |
| 02 | [The bug that survived every parity gate: 40 of 48 heads running on zeros](02-the-bug-that-survived-the-parity-gates.md) | one launch argument, six correctness checks blind to it, and the gate that now catches it |
| 03 | [Speculative decoding that is exact: 1.53× to 2.00× from the model's own draft head](03-mtp-speculative-decoding-exact.md) | draft/verify with the model's own MTP block, token-exact against plain greedy |
| 04 | [Roofline first: why neither engine can decode faster on a 640 GB/s card](04-roofline-first.md) | the decode ceiling, the levers that were measured and dropped, and the prefill machine |
| 05 | [Serving eight sessions on one 32 GB card: 125 tok/s, a 16-slot tie, and a cache that could not keep its promise](05-serving-8-agents-on-one-card.md) | continuous batching, the prefix cache, and the hit that was not the hit asked for |
| 06 | [What is inside a dynamic quant: nine tensor types in one Q4_K_M file](06-whats-inside-unsloth-dynamic-quants.md) | what the `UD-` label means, the IQ formats, and the residency trap |
| 07 | [One session, one binary, every axis: four wins, four losses, one tie](07-one-session-every-axis.md) | one window, one build, every axis at once, with the losses in the same table |
