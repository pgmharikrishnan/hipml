# Multi-sequence batching (WP26)

> *Engineering log from development; internal work-package IDs and paths have been sanitized or removed where they were machine-specific. Current numbers: [docs/benchmarks.md](../benchmarks.md).*

Every earlier number in this project is single-stream latency. The target
workload is many concurrent agent sessions, so the number that pays for the
machine is aggregate throughput: many concurrent requests sharing one weight
stream.

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
- The batched projections default to the exact f32-from-LDS `_b` kernels, so a
  row is bit-identical to a B=1 launch on the same input (gemv-check). Setting
  `HIPML_SKINNY=1` (or `HIPML_BATCH_SKINNY=1`) routes them through the WP33
  skinny WMMA GEMM instead: one 16-row tile per launch, so B=4..16 cost one
  weight pass and B=17..32 cost two. Q5_K has no skinny entry and always keeps
  `_b` in groups of 8. The measured table below used the opt-in skinny path;
  the default exact path is reported separately.
- Attention, the SSM recurrence and the KV append run **one launch per stage for
  all slots**: the slot index is `blockIdx.y`, the per-slot KV caches and SSM
  state are contiguous, and the per-slot position/seq pair is read from a
  `[slots][2]` device array written once per step. `gemv_f32_b` does the same for
  the alpha/beta projections, `qg_split_b`/`head_rmsnorm_b`/`mrope_b` for the
  per-row q-gate/norm/rope, and one `q_gate_f32` covers every row's output gate.
- `argmax_part_rows_f32`/`argmax_final_rows_f32` sample all B rows on device.

The batched `_b` kernels decode each weight block once and feed all B x-rows.
The skinny tile pads R to one 16-wide WMMA tile, so its cost is the weight
stream, not R.

## Measured (Qwen3.8-27B Q4_K_M, R9700, q8 KV, depth 0)

`batch-bench --tokens 16`, one run; the single-stream reference is measured
inside each run (30.0-30.2 tok/s, 33.1-33.3 ms/step).

| B | aggregate tok/s | per stream | ms/step | weight GB/s | B x single |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 54.6 | 27.3 | 36.6 | 459 | 0.90x |
| 4 | 82.1 | 20.5 | 48.8 | 345 | 0.68x |
| 8 | **155.0** | 19.4 | 51.6 | 326 | 0.64x |
| 16 | **275.4** | 17.2 | 58.1 | 290 | 0.57x |
| 32 | 284.5 | 8.9 | 112.5 | 150 | 0.30x |

The knee is between 16 and 32: one 16-row skinny tile serves B<=16, so B=8 and
B=16 add only the per-slot recurrence work; B=32 needs two weight passes and
drops to the 150 GB/s row of the table. Aggregate throughput is still rising at
32, but per-step weight efficiency halves.

Token parity (`--check`, distinct token stream per slot, 16 greedy steps): B=1
8/8; B=2 8/8 both slots; B=4, B=8 and B=16 16/16 on every slot. At B=32 the
skinny path reproduces 511 of 512 slot-streams: slot 30 flips one id at step 12
(a near-tie; the upstream WMMA/fp16 logits differ from the GEMV path by ~1e-2).
The exact `_b` path at B=32 (`HIPML_BATCH_LEGACY=1`) matches 32/32 at 94 tok/s.

Mixed depth `0,1024,2048,4096,8192,8192,8192,8192` at B=8: **138.4 tok/s**
(57.8 ms/step, 291 GB/s weight). Uniform 0,1024,8192,32768 at the pre-WP37
engine was 58.7 tok/s.

### Against llama.cpp at equal B

Same model file, q8_0 K/V, `-np B`, B concurrent greedy requests:

| B | hipml batch | llama.cpp | hipml / llama |
| ---: | ---: | ---: | ---: |
| 2 | 54.6 | 56.3 | 0.97 |
| 4 | 82.1 | 87.3 | 0.94 |
| 8 | 155.0 | 101.3 | 1.53 |
| 16 | 275.4 | 231.2 | 1.19 |
| 32 | 284.5 | 302.7 | 0.94 |

llama.cpp's client-overlap figures are lower (np8 89.0, np16 179.5, np32
229.8); the table uses its server clock as before.

### VRAM per slot and how many slots fit

At 8K context, q8 KV: **421.6 MiB per slot** (272.0 MiB KV + 149.6 MiB SSM
state), on top of the 16.01 GiB weight residency. After the B=4 run the
allocator reports **37 total slots** at 8K; B=8 leaves 12.96 GiB and still says
39. The mixed 8K run reports 34. VRAM, not compute, is the slot ceiling: B=32
at depth 0 already streams 4.8 GiB of SSM state, and 32 slots at 8K would be
16.01 + 13.2 = 29.2 GiB of the 31.86 GiB card, i.e. ~37 slots is the measured
fit. Compute limits at ~B=87 if state were free.

### Where the B=4 step goes (before and after WP37)

Stage timing (`--split`; the split itself costs a sync per stage, so read the
shares). Before = WP33 skinny GEMM with per-slot attention/SSM; after = this
work's one-launch batching.

| stage | before ms | after ms | after share |
| --- | ---: | ---: | ---: |
| ffn (3 batched GEMMs x 64 layers) | 30.42 | 30.85 | 51.2% |
| ssm proj | 11.39 | 10.41 | 17.3% |
| ssm out | 5.48 | 5.68 | 9.4% |
| attn proj (incl. split/norm/rope) | 7.72 | 3.65 | 6.1% |
| final (lm_head + argmax) | 1.94 | 1.96 | 3.2% |
| attn out | 1.87 | 1.84 | 3.1% |
| **ssm recurrence (per slot)** | **9.40** | **4.85** | 8.0% |
| **attn kv + core (per slot)** | **4.13** | **0.79** | 1.3% |
| embed | 0.14 | 0.05 | 0.1% |
| total (traced) | 72.58 | 60.30 | |

The per-slot half fell from 13.5 ms to 5.6 ms at B=4. At B=8 it is 7.9 ms vs
24.4 ms before (90.16 -> 63.96 ms traced); at B=16 the same stages are 12.0 ms
of a 69.6 ms traced step. The remaining growth is the real delta-net arithmetic
(one block per value head per slot), not launch gaps.

### Per-slot SSM state traffic

Each slot reads and writes 313.8 MB of recurrent state per step (151 MB of
conv window + delta-net matrix read, 149.6 MB written). The whole 48-layer
state for B=8 is 1.2 GB, far beyond the card's L2, so it must stream. Measured
`ssm rec` stage: 4.85 ms at B=4, 7.07 ms at B=8, 11.23 ms at B=16, i.e.
259/355/447 GB/s of state traffic. It fits HBM, but it is 8-16% of the step and
grows with B; keeping a slot group's state resident would need a state tile
under L2 (a few slots per layer), not the whole 1.2 GB.

## What is not verified

- B=32 token parity on the skinny path: 511/512 slot-streams (one near-tie
  flip); the legacy path is 32/32.
- Mixed-depth token parity: `--check` is defined at depth 0 only.
- B=32 at 8K (29.2 GiB arithmetic, not exercised).
- The per-tensor `--probe` numbers: the probe crashed on a one-row x buffer and
  the fix built after the last measurement window.
- The 37-43 slot estimates were not exercised by actually allocating that many.


## Measuring

```
hipml batch-bench --gguf MODEL --slots 4 --tokens 16 --check
hipml batch-bench --gguf MODEL --slots 4 --depths 0,1024,8192,32768
hipml batch-bench --gguf MODEL --slots 8 --probe     # per-tensor GB/s
hipml batch-bench --gguf MODEL --slots 4 --split     # stage ms/step
hipml batch-bench --gguf MODEL --slots 32 --active 16 # capacity vs active slots
```

`--check` runs the same token stream through the single-stream engine and
requires the greedy ids to match, slot by slot.

The llama.cpp comparison is `bench/batch_llama.py --np B`, which starts
`llama-server` on 8092 with `-np B` and q8_0 K/V and issues B identical greedy
requests concurrently.
