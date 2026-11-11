# Beating llama.cpp at prefill: 11 cells of 12, and a 6% decode bill

*On one R9700 with a 16.8 GB Qwen3.8-27B Q4_K_M file, prefill is 4.6% to 17.2% faster in eleven of twelve cells; decode is 6% slower, for a reason no kernel change addresses.*

At 672 prompt tokens the Qwen3.8-27B Q4_K_M reaches first token in 574 ms at 1,172 tok/s. llama.cpp, on the same file with the same token ids, takes 698 ms at 1,000 tok/s. At 100,000 tokens the same pair is 108,698 ms against 122,997 ms. Across five prompt classes and four depths, prefill is faster in eleven of twelve cells, by 4.6% to 17.2%.

The twelfth is a loss. On a 285-token copy prompt at depth 0 the engine prefills at 831 tok/s against llama.cpp's 848, a 2.1% deficit. It still reaches first token 9 ms sooner (343 against 352 ms), because at that length fixed launch cost is a larger share than the GEMM.

## The matrix

One file: `Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,336 bytes, SHA-256 `e00082f7…e520`. One card: Radeon AI PRO R9700, gfx1201, 32 GiB, ROCm 7.2.1, Windows 11. llama.cpp is 0.7.0-dev build 11820, commit `d7725699`. Both engines used q8-class KV, hipml `q8` and llama `q8_0`. The fixture prompts were tokenised once and the same ids fed to both.

llama.cpp was started with `-ngl 99 -c <ctx> --parallel 1 -ctk q8_0 -ctv q8_0 --no-webui --no-cache-idle-slots`, and with no `-fa`, `-b` or `-ub` flag. On this build the defaults were `-fa auto` (which resolves to on), `-b 2048` and `-ub 512`. It ran no draft model and no speculative decoding.

| depth | prefill | decode | TTFT (llama / hipml) |
| ---: | --- | --- | --- |
| 0 | −2.1% to +17.2% | −6.3% to −7.4% | 1.02–1.22× |
| 8,192 | +5.2% to +5.6% | −6.1% to −6.5% | 1.05–1.06× |
| 32,768 | +8.2% | −4.6% | 1.08× |
| 100,000 | +13.1% | **+4.3%** | 1.13× |

Ranges are across the five prompt classes where more than one was run; 32,768 and 100,000 are `code_edit` only. Depth 0 and 8,192 are three cold reps; 32,768 and 100,000 are one.

## What bought the prefill win

Prefill is a GEMM problem: 2,048 prompt tokens multiplied against each weight matrix, so each weight byte loaded from DRAM is used many times. Four decisions account for most of the difference.

**A WMMA tile over quantized weights.** A block owns 256 prompt tokens by 128 output rows, with 256 threads and 16 accumulators per warp. The weight tile is dequantised into shared memory once per tile, then consumed by `wmma_f32_16x16x16_f16`, so the same bytes are read once and reused.

**A 2,048-token chunk.** At 512 tokens the gate/up projection launches 80 blocks on a 64-CU card, so a quarter of the GEMM time runs at 25% occupancy. Four times the chunk fills the machine.

**Split-K over the key axis.** At depth, chunk attention stops being a GEMM and becomes a traversal of the KV cache. Splitting the live key range across blocks, with a combine pass at the end, shortens the critical path. On an earlier Qwen3.8-27B quantisation at 100,000 tokens, attention was 64.4 s of a 137.0 s prefill; the split took it to 52.3 s.

**Whole-chunk attention.** Giving each block the whole 256-wide head instead of half of it computes the QK score tile once rather than twice. The same earlier measurement took the attention stage from 52.3 s to 38.4 s; an interior-mask skip was worth 5.4% on its own. The combine pass costs about 1.2 s across 100,000 tokens against the 12.1 s the split saves.

The stage and occupancy numbers in this section come from an earlier Qwen3.8-27B quantisation on the same card, not from the stock release file; the release matrix measures whole-prompt TTFT, not the stages. The mechanism is the same. The size of the win is not.

## Why decode cannot follow

Decode streams 16.81 GB of weights for one token. At the card's 640 GB/s that is 26.3 ms, or 38 tok/s, before any kernel overhead; with the 1.4 ms ramp the project's roofline calculation charges, about 36 tok/s. Measured, hipml decodes at 29.7 tok/s and llama.cpp at 31.8 on the 672-token prompt. Both are at the wall.

That is the whole argument, and it is why the prefill work does not transfer. A prefill tile keeps a weight tile in shared memory and multiplies 256 tokens against it. A decode GEMV multiplies one vector against the same tile and then discards it. Tiling, split-K and chunk width change the number of times a weight is reused within a token's work; they cannot make the weight stream smaller than the model. [Article 04](04-roofline-first.md) derives the ceiling and lists the levers that were measured and dropped.

Decode is 6.1% to 7.4% slower at 8,192 tokens and shallower, 4.6% slower at 32,768, and only ahead at 100,000 (+4.3%, 23.7 against 22.7 tok/s). Peak VRAM is 1.4 GiB higher on the 672-token prompt (16.94 against 15.50 GiB), 2.4 GiB higher at 8,192 (18.17 against 15.75) and 2.1 GiB higher at 100,000 (21.26 against 19.16), from the q8 cache layout and the recurrent state.

## The batch path and its exactness bill

Batching is the other way past the decode wall: one weight pass serves N streams. On the exact path, eight slots give 107.46 aggregate tok/s (13.43 per stream, +10.2% over llama.cpp's `-np 8`) and sixteen give 107.33, which is 52.9% behind llama.cpp's 227.89. The fast path is an opt-in WMMA tile: 149.43 and 272.97 tok/s, +53.2% and +19.8% over llama.cpp. It is not bit-identical to the single-sequence path; `batch-bench --check` reports `TOKENS DIFFER`. The exact path matches B=1 on every slot, 16 ids of 16.

So the honest batch claim is narrow. Exact batching wins at eight slots and loses badly at sixteen. The sixteen-slot number that beats llama.cpp requires a tile that changes tokens, and it is quoted as approximate for that reason.

## MTP is the honest answer to the decode gap

The one lever the project found that raises single-stream decode is speculation. A sidecar holds the model's next-token-prediction block; it drafts up to three tokens and the target verifies them in one weight pass. On the five chat prompts in the release run that is 1.53× to 2.00× (44.79 to 58.60 tok/s against a plain 25.14 to 29.43), and every cell is token-exact against plain greedy: 384/384, 384/384, 65/65, 174/174, 100/100. [Article 03](03-mtp-speculative-decoding-exact.md) covers the mechanism and the server bug it exposed.

The caveat matters: those cells are f32 KV. MTP at q8 KV on the stock model is implemented but not measured, so the decode column above is plain decode, not the speculative path.

## What was not measured

- One GPU (gfx1201), one OS (Windows 11), one llama.cpp build. No CUDA, no second card, no Linux.
- One model file, one quantisation, one SHA-256. Nothing here says how a different Q4_K_M or a Q6_K would compare.
- Three cold reps at 0 and 8,192 tokens; one rep at 32,768 and 100,000. The depth cells have no error bar.
- llama.cpp had no draft model and no speculative decoder. Comparing its plain 31.8 tok/s with the speculative path's 44.8 to 58.6 would be a different experiment, so both sides of the matrix are plain.
- Batch depth above zero, server behaviour at 32 slots, and any GPU other than this one were not measured here. The server has its own measurements in [article 05](05-serving-8-agents-on-one-card.md).
- Prefill tok/s at depth 0 includes fixed launch and sampling overhead. That is why the 285-token copy cell loses throughput and wins TTFT; the throughput number is not a pure GEMM rate.
- Correctness is a separate claim. The launch bug that survived every parity gate, and the end-to-end gate that now catches it, are [article 02](02-the-bug-that-survived-the-parity-gates.md).

## Where the numbers come from

- The matrix, gap table, model hash, engine settings and the unmeasured list: [`../release-run.md`](../release-run.md).
- Raw per-cell results and peaks: [`../../bench/release/matrix/results.json`](../../bench/release/matrix/results.json).
- Tile shape, chunk occupancy, split-K and whole-width attention: [`../notes/wp41-final-tree.md`](../notes/wp41-final-tree.md), [`../notes/fullattn-gpu.md`](../notes/fullattn-gpu.md) and [`../notes/roofline-log.md`](../notes/roofline-log.md).
- The decode ceiling and the levers that failed: [`../notes/review-levers.md`](../notes/review-levers.md) and [`04-roofline-first.md`](04-roofline-first.md).
- MTP speedups and exactness: [`../release-run.md`](../release-run.md) and [`../../bench/release/mtp/mtp_summary.txt`](../../bench/release/mtp/mtp_summary.txt).
- Batch exact and skinny numbers, and the check result: [`../../bench/release/batch/exact_throughput.txt`](../../bench/release/batch/exact_throughput.txt) and [`../../bench/release/batch/exact_check.txt`](../../bench/release/batch/exact_check.txt).
