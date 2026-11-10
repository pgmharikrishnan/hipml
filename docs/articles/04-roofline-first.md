# Roofline first: why neither engine can decode faster on a 640 GB/s card

*The useful question is not what is slow but what cannot be fast. Decode on this card has a ceiling of 35.3 tokens per second, and the two engines sit at 83% and 86% of it.*

Decode streams every weight through the memory bus for every token. This model is 17.19 GB of Q4_K_M — 16.01 GiB across 851 tensors — so the arithmetic is short:

```
17.19 GB / 632 GB/s          = 27.2 ms  -> 36.8 tok/s   (stream only)
17.19 GB / 640 GB/s + 1.4 ms = 28.3 ms  -> 35.3 tok/s   (ceiling used below)
29.2 tok/s / 35.3            = 82.7%    (hipml, depth 0)
30.2 tok/s / 35.3            = 85.6%    (llama.cpp, depth 0)
```

The first line uses the bandwidth a hand-written fp32 GEMV actually reached on this card: **633.6 GB/s** against a 640 GB/s spec. The second charges the spec rate and adds **1.4 ms** of unavoidable kernel ramp and drain, which is the number the project's ceiling quotes. The two engines are 3.1% apart and sit at 83% and 86% of that wall. That is the entire result: there is no decode win of consequence available from a better matrix-vector kernel.

## What the ceiling ruled out

The decode ledger at depth 0, 33.5 ms per token, says the same thing from the other side: 27.2 ms of weight stream (81%), 5.6 ms of per-kernel memory ramp (17%), and 0.7 ms of exposed host time (2%). The weight stream is at 99% of measured bandwidth. Only the ramp is left, and it was attacked from every angle the driver exposes:

| lever | measured | verdict |
|---|---|---|
| HIP graph capture of the whole step | 33.39 ms vs 33.37 eager (0.998×) | no change |
| one job-list kernel over independent projections | 561–562 GB/s vs 568 | flat |
| split-K to widen a narrow grid | 546 / 551 / 531 GB/s vs 559 | worse |
| streaming (`sc1`) weight loads | 491 GB/s vs 559 (−12%) | worse; L2 was helping |
| two weight streams on two HIP streams | 500 → 370 GB/s (0.74×) | worse |
| f16 KV mirror instead of q8 staging | 993 vs 998 tok/s at 8K; 871 vs 867 at 32K | slower at 8K, level at 32K |
| q4 KV instead of q8 at 100K | 14.62 vs 12.40 ms of attention | halves bytes, slower |

Graph capture is the sharpest example. The step is 1,299 launches, 2.35 ms of API time, and the host cannot run ahead because it needs the sampled token back before issuing the next step. Capturing the graph removes the API time and changes the wall clock by 0.06 ms at most. The 1.6 ms the event trace reports as idle between launches is GPU-side drain at kernel boundaries; capture does not remove it. A 128-bit payload load variant exists for the same reason and remains an A/B switch.

Quantized KV is the one place where a real byte reduction did not arrive as wall time. At 100K, moving from a q8 cache to q4 halves the cache again, 3.48 GB to 1.84 GB, and the ceiling says that is worth 2.6 tok/s. Measured, q4 is **1.34 tok/s slower** (23.94 vs 25.28), because the attention kernel is issue-bound at 281 GB/s: the smaller cache buys DRAM margin the compute pass cannot spend.

None of that says the kernels are finished. It says the remaining margin is small, shared with llama.cpp, and not where a from-scratch engine should spend its first month.

## Prefill is a different machine

One token of this 26.9-billion-parameter model costs 53.8 GFLOP in prefill. At the vendor's 95.7 TFLOPS fp16 dense figure that is 1,779 tok/s. Measured, the shipped engine reaches **1,198 tok/s** at 672 prompt tokens against llama.cpp's 1,014, **1,252 vs 1,184** at 8K, and **928 vs 827** at 100K. That is 67–70% of the vendor budget at the shallow depths.

The vendor number is also not the wall it looks like. A register-only probe of the matrix instruction the GEMM actually uses — `wmma_f32_16x16x16_f16`, eight independent accumulators, no shared memory — measures **171–188 TFLOPS**, roughly twice the quoted dense peak. The VALU path underneath it measures 41.4 TFLOPS for `v_dot2_f32_f16`. So there are three different ceilings on this card depending on which instruction you are willing to pay for, and a GEMM scored against the wrong one looks twice as far from the wall as it is.

The prefill work followed the real one. A block now owns 256 prompt tokens by 128 output rows, 256 threads, 16 accumulators per warp, with the weight tile dequantised into shared once per tile. At the engine's 2048-token chunk that measures **95.0 TFLOPS for Q6_K and 108.9 for Q4_K**, against 66–76 before. The chunk size matters for the same reason: at 512 tokens the gate/up shape launches 80 blocks on a 64-CU card, so a quarter of the GEMM time runs at 25% occupancy.

At depth the bound moves again, from arithmetic to KV traffic. With attention unsplit, the 100K cell spent **64.4 s of a 136.98 s prefill** in attention: the 3.48 GB q8 cache re-read 2,345 times at an effective 137 GB/s. Two changes cut that to **38.37 s**. Splitting the live key range across blocks took it to 52.29 s; giving each block the whole 256-wide head, so the QK score tile is computed once instead of twice for the two d-halves, took it to 38.37 s, with the interior-mask skip worth 5.4% on its own. The combine pass costs about 1.2 s across a 100K prompt against the 12.1 s the split saves. Whole-prompt TTFT went from 136.98 s to **107.78 s**, 12.2% faster than llama.cpp's 120.90 s.

## What the roofline does not say

It does not say the engine is correct. A launch bug can leave 40 of 48 projection rows unwritten while every bandwidth number stays exactly where the model predicts; that is [a separate article](02-the-bug-that-survived-the-parity-gates.md), and a separate gate.

It also does not say stop. The 35.3 tok/s ceiling assumes one token per weight pass, which is a property of dense decode, not of the hardware. Speculative decoding with the model's own MTP head produces 1.46–1.80× on chat prompts, median 1.58×, token-exact against greedy — the ceiling was the argument for building it, not against.

And the losses are still there, in the same table as the wins. Plain decode trails llama.cpp by 3.1% at depth 0, 2.2% at 8K and 1.2% at 32K; it only wins at 100K (+7.4%). Peak VRAM is 1.1 to 2.4 GiB higher in every cell (21.93 vs 19.88 GiB at 100K), because of the q8 layout and the recurrent state. The bit-exact batch path loses 9.6% at B=8, and the +49.3% headline needs an opt-in kernel. A 100K prompt still takes 107.8 s to first token. On a 640 GB/s card, that is what the roofline buys: a fast prefill, a decode that is honest about being at the wall, and a short list of ideas that were measured and dropped instead of carried for three weeks.

## Where the numbers come from

- 633.6 GB/s measured, 640 GB/s spec, 17.19 GB / 16.01 GiB of weights, 171–188 TFLOPS WMMA, 41.4 TFLOPS dot2, 95.0/108.9 TFLOPS tiles, 64.4 s → 38.37 s attention: [`../notes/roofline-log.md`](../notes/roofline-log.md) and [`../notes/fullattn-gpu.md`](../notes/fullattn-gpu.md).
- 35.3 tok/s ceiling, the 1.4 ms tail, the 33.5 ms decode ledger, and the split-K/two-stream/graph refutations: [`../notes/review-levers.md`](../notes/review-levers.md).
- Graph capture 33.39 vs 33.37 ms, job-list 561–568 GB/s, split-K 546/551/531, sc1 491 vs 559: [`../notes/gpu-engine.md`](../notes/gpu-engine.md).
- q8 attention 12.40 ms and q4 14.62 ms at 100K, 1.34 tok/s from quantization: [`../notes/roofline-log.md`](../notes/roofline-log.md), "Depth against the ceiling".
- f16 mirror 993 vs 998 tok/s (8K) and 871 vs 867 (32K): [`../notes/fullattn-gpu.md`](../notes/fullattn-gpu.md), "Reading the quantized cache instead of the mirror".
- 1,198 / 1,252 / 928 tok/s, llama 1,014 / 1,184 / 827, decode gaps, VRAM, batch −9.6%: [`../benchmarks.md`](../benchmarks.md) and [`../notes/wp41-final-tree.md`](../notes/wp41-final-tree.md).
- 136.98 s → 107.78 s, 52.29 s and 38.37 s attention, 12.2% win at 100K: [`../notes/wp41-final-tree.md`](../notes/wp41-final-tree.md), "Merge value".
- MTP 1.46–1.80×, median 1.58×: [`../notes/wp43-chat-parity.md`](../notes/wp43-chat-parity.md).
- The 128-bit load A/B entry: [`../../crates/hipml-kernels/kernels/gemv_q4_k.hip`](../../crates/hipml-kernels/kernels/gemv_q4_k.hip).
