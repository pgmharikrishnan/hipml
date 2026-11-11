# Roofline first: why neither engine can decode faster on a 640 GB/s card

*The useful question is not what is slow but what cannot be fast. Decode on this card has a ceiling of 36.1 tokens per second, and the two engines sit at 82% and 88% of it.*

Decode streams every weight through the memory bus for every token. The stock model is 16.81 GB of Q4_K_M — 15.65 GiB across 866 tensors — so the arithmetic is short:

```
16.81 GB / 633.6 GB/s = 26.5 ms  -> 37.7 tok/s  (stream only)
16.81 GB / 640 GB/s   = 26.3 ms; + 1.4 ms = 27.7 ms -> 36.1 tok/s (ceiling below)
29.7 tok/s / 36.1     = 82%      (hipml, depth 0)
31.8 tok/s / 36.1     = 88%      (llama.cpp, depth 0)
```

The first line uses the bandwidth a hand-written fp32 GEMV actually reached: **633.6 GB/s** against a 640 GB/s spec. The second charges the spec rate and adds **1.4 ms** of unavoidable kernel ramp and drain, which is the number the project's ceiling quotes. The two engines are 6.6% apart and sit at 82% and 88% of that wall. That is the entire result: there is no decode win of consequence from a better matrix-vector kernel.

## What the ceiling ruled out

The decode ledger from the earlier Qwen3.8-27B quantisation, 33.5 ms per token, says the same thing from the other side: 27.2 ms of weight stream (81%), 5.6 ms of per-kernel memory ramp (17%), and 0.7 ms of exposed host time (2%). The weight stream is at 99% of measured bandwidth. Only the ramp is left, and it was attacked from every angle the driver exposes:

| lever | measured | verdict |
|---|---|---|
| HIP graph capture of the whole step | 33.39 ms vs 33.37 eager (0.998×) | no change |
| one job-list kernel over independent projections | 561–562 GB/s vs 568 | flat |
| split-K to widen a narrow grid | 546 / 551 / 531 GB/s vs 559 | worse |
| streaming (`sc1`) weight loads | 491 GB/s vs 559 (−12%) | worse; L2 was helping |
| two weight streams on two HIP streams | 500 → 370 GB/s (0.74×) | worse |
| f16 KV mirror instead of q8 staging | 993 vs 998 tok/s at 8K; 871 vs 867 at 32K | slower at 8K, level at 32K |
| q4 KV instead of q8 at 100K | 14.62 vs 12.40 ms of attention | halves bytes, slower |

All rows are from that earlier quantisation; the verdicts carry over.

Graph capture is the sharpest example. The step is 1,299 launches, 2.35 ms of API time, and the host cannot run ahead before the sampled token returns. Capturing the graph removes the API time and changes the wall clock by 0.06 ms at most. The 1.6 ms the event trace reports as idle between launches is GPU-side drain at kernel boundaries; capture does not remove it.

Quantized KV is the one place where a real byte reduction did not arrive as wall time. At 100K, moving from a q8 cache to q4 halves the cache again, 3.48 GB to 1.84 GB, and the ceiling says that is worth 2.6 tok/s. Measured on the earlier quantisation, q4 is **1.34 tok/s slower** (23.94 vs 25.28), because attention is issue-bound at 281 GB/s: the smaller cache buys DRAM margin the compute pass cannot spend.

None of that says the kernels are finished. It says the remaining margin is small, shared with llama.cpp, and not where a from-scratch engine should spend its first month.

## Prefill is a different machine

One token of this 26.9-billion-parameter model costs 53.8 GFLOP in prefill. At the vendor's 95.7 TFLOPS fp16 dense figure that is 1,779 tok/s. Measured on the stock release file, the shipped engine reaches **1,172 tok/s** at 672 prompt tokens against llama.cpp's 1,000, **1,225 vs 1,160** at 8K, and **920 vs 813** at 100K. That is 66–69% of the vendor budget at the shallow depths.

The vendor number is also not the wall it looks like. A register-only probe of the matrix instruction the GEMM actually uses — `wmma_f32_16x16x16_f16`, eight independent accumulators, no shared memory — measured **171–188 TFLOPS** on the earlier quantisation, roughly twice the quoted dense peak. The VALU path underneath it measures 41.4 TFLOPS for `v_dot2_f32_f16`. So there are three different ceilings on this card depending on which instruction you are willing to pay for, and a GEMM scored against the wrong one looks twice as far from the wall as it is.

The prefill work followed the real one. A block now owns 256 prompt tokens by 128 output rows, 256 threads, 16 accumulators per warp, with the weight tile dequantised into shared once per tile. On the same earlier quantisation, at the engine's 2048-token chunk, that measures **95.0 TFLOPS for Q6_K and 108.9 for Q4_K**, against 66–76 before. The chunk size matters for the same reason: at 512 tokens the gate/up shape launches 80 blocks on a 64-CU card, so a quarter of the GEMM time runs at 25% occupancy.

At depth the bound moves again, from arithmetic to KV traffic. On that earlier quantisation, with attention unsplit, the 100K cell spent **64.4 s of a 136.98 s prefill** in attention: the 3.48 GB q8 cache re-read 2,345 times at an effective 137 GB/s. Two changes cut that to **38.37 s**. Splitting the live key range across blocks took it to 52.29 s; giving each block the whole 256-wide head, so the QK score tile is computed once instead of twice for the two d-halves, took it to 38.37 s, the interior-mask skip worth 5.4% on its own. The combine pass costs about 1.2 s against the 12.1 s the split saves. That run's whole-prompt TTFT went from 136.98 s to 107.78 s. On the stock release file the same 100,000-token cell reaches first token in **108.70 s** against llama.cpp's 123.00 s, 11.6% faster.

## What the roofline does not say

It does not say the engine is correct. A launch bug can leave 40 of 48 projection rows unwritten while every bandwidth number stays exactly where the model predicts; that is [a separate article](02-the-bug-that-survived-the-parity-gates.md), and a separate gate.

It also does not say stop. The 36.1 tok/s ceiling assumes one token per weight pass, which is a property of dense decode, not of the hardware. Speculative decoding with the model's own MTP head produces 1.53–2.00× on chat prompts, median 1.83×, token-exact against greedy — the ceiling was the argument for building it, not against.

And the losses are still there, in the same table as the wins. Plain decode trails llama.cpp by 6.6% at depth 0, 6.1–6.5% at 8K and 4.6% at 32K; it only wins at 100K (+4.3%). Peak VRAM is 1.4 to 2.4 GiB higher in every cell (21.26 vs 19.16 GiB at 100K), because of the q8 layout and the recurrent state. The bit-exact batch path wins 10.2% at B=8 and loses 52.9% at B=16; the +53.2% and +19.8% headlines need an opt-in kernel. A 100K prompt still takes 108.7 s to first token. On a 640 GB/s card, that is what the roofline buys: a fast prefill, a decode that is honest about being at the wall, and a short list of ideas that were measured and dropped instead of carried for three weeks.

## Where the numbers come from

- 16.81 GB / 15.65 GiB, 866 tensors, the head-to-head matrix, the decode and prefill rates, the MTP range and the batch cells: [`../release-run.md`](../release-run.md).
- Bandwidth, WMMA/dot2/tile TFLOPS and the 64.4 s → 38.37 s attention: earlier Qwen3.8-27B quantisation, [`../notes/roofline-log.md`](../notes/roofline-log.md) and [`../notes/fullattn-gpu.md`](../notes/fullattn-gpu.md).
- The 1.4 ms tail, the 33.5 ms decode ledger, and the split-K/two-stream/graph refutations: earlier quantisation, [`../notes/review-levers.md`](../notes/review-levers.md).
- Graph capture, job-list, split-K and sc1 measurements: earlier quantisation, [`../notes/gpu-engine.md`](../notes/gpu-engine.md).
- q8 attention 12.40 ms and q4 14.62 ms at 100K, 1.34 tok/s from quantization: earlier quantisation, [`../notes/roofline-log.md`](../notes/roofline-log.md).
- f16 mirror 993 vs 998 tok/s (8K) and 871 vs 867 (32K): earlier quantisation, [`../notes/fullattn-gpu.md`](../notes/fullattn-gpu.md).
- 1,172 / 1,225 / 920 tok/s against llama's 1,000 / 1,160 / 813: [`../release-run.md`](../release-run.md); the tile and chunk figures: [`../notes/wp41-final-tree.md`](../notes/wp41-final-tree.md).
- 136.98 s → 107.78 s, 52.29 s and 38.37 s attention, 12.2% win at 100K: earlier quantisation, [`../notes/wp41-final-tree.md`](../notes/wp41-final-tree.md), "Merge value".
- MTP 1.53–2.00×, median 1.83×, and the five exact streams: [`../release-run.md`](../release-run.md), "MTP decode, chat prompts, EOS-stopped".
