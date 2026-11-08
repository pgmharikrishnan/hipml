# Multi-sequence batching (WP26)

Every earlier number in this project is single-stream latency. The product is
an agent fleet, so the number that pays for the machine is aggregate
throughput: many concurrent requests sharing one weight stream.

Decode is bandwidth-bound: one step reads 17.19 GB of weights. With B
sequences decoded together, one read serves B tokens, so the roofline for a
batched step is still ~27 ms at 632 GB/s and aggregate throughput scales until
something else saturates.

## What is per sequence

`BatchEngine` (`crates/hipml-model/src/gpu/batch.rs`) gives every slot its own
state; nothing in the forward path assumes slot 0 is the only sequence.

| state | per slot |
| --- | --- |
| full attention (16 layers) | one quantized `DeepKv` cache per layer, q8 or q4 |
| SSM (48 layers) | one conv window + delta-net matrix per layer (156.9 MB total) |
| position | `positions[slot]`; slots may sit at different depths |
| sampling | one logit row + one greedy id per slot |

`SlotPool` hands out the slot indices (`alloc`/`free`), so a server can keep a
slot per request without the engine caring which index it got.

## What is batched

The decode GEMV reads each weight byte once and now produces B outputs:

- `gemv_q4_k_b`, `gemv_q6_k_b`, `gemv_q5_k_b`, `gemv_q8_0_b`, one per resident
  block format. Each row keeps its own accumulator and walks `k` in exactly the
  order the single-sequence kernel does, so row b is bit-identical to a B=1
  launch on the same input.
- `lm_head` uses the same Q6_K kernel over B rows.
- attention/SSM projections use the batched kernels; attention itself, the SSM
  recurrence and the KV append stay per slot (different position/state), split
  across the two streams.
- `argmax_part_rows_f32`/`argmax_final_rows_f32` sample all B rows on device.

Two reuse axes matter. A decoded weight value feeds all B rows, and one shared
x read feeds two weight rows per warp. Without the second axis the kernel
becomes shared-memory-bandwidth bound around B=4; the measurement that shows
it is in `docs/benchmarks.md`.

## Measuring

```
hipml batch-bench --gguf MODEL --slots 4 --tokens 16 --check
hipml batch-bench --gguf MODEL --slots 4 --depths 0,1024,8192,32768
hipml batch-bench --gguf MODEL --slots 8 --probe     # per-tensor GB/s
hipml batch-bench --gguf MODEL --slots 4 --split     # stage ms/step
```

`--check` runs the same token stream through the single-stream engine and
requires the greedy ids to match, slot by slot.

The llama.cpp comparison is `bench/batch_llama.py --np B`, which starts
`llama-server` on 8092 with `-np B` and q8_0 K/V and issues B identical greedy
requests concurrently.
