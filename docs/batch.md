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
  recurrence and the KV append stay per slot (different position/state). An
  opt-in two-stream split (`HIPML_BATCH_SPLIT=1`) alternates slots, but it was
  slower than the sequential step at every measured B, so it is off by default.
- `argmax_part_rows_f32`/`argmax_final_rows_f32` sample all B rows on device.

The batched kernels decode each weight block once and feed all B x-rows, and one
shared x read feeds two weight rows per warp. The k-tile is 1024 values for B<=4
and 512 above that, so the shared x staging stays under 32 KB.

## Measured (Qwen3.8-27B Q4_K_M, R9700, q8 KV, depth 0)

`batch-bench --tokens 16`, best of one run; the single-stream reference is
measured inside each run.

| B | aggregate tok/s | per stream | ms/step | weight GB/s | B x single |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 44.5 | 22.3 | 44.9 | 375 | 0.74x |
| 4 | **64.4** | 16.1 | 62.1 | 271 | 0.53x |
| 8 | 68.6 | 8.6 | 116.6 | 144 | 0.28x |

Single-stream reference: 30.0-30.2 tok/s (33.1-33.3 ms/step). The batched path
at B=1 is slower than the production single-stream kernels (20.6 tok/s,
48.7 ms/step with stage tracing), which is the first clue where the gap is: the
`_b` GEMV costs ~1.47x the single-stream GEMV on the same weights even when
there is only one row.

Token parity (`--check`, distinct token stream per slot, 16 greedy steps): all
4 slots reproduce the B=1 path, 16/16 ids each, on the default (unsplit) step.

Mixed depth `0,1024,8192,32768`: 58.7 tok/s aggregate (68.1 ms/step,
247 GB/s).

### Against llama.cpp at equal B

Same model file, q8_0 K/V, `-np B`, B concurrent greedy requests:

| B | hipml batch | llama.cpp | hipml / llama |
| ---: | ---: | ---: | ---: |
| 2 | 45.7 | 56.3 | 0.81 |
| 4 | 64.8 | 87.3 | 0.74 |
| 8 | **102.0** | 101.3 | **1.01** |

The B=8 number is WP33: `Launcher::gemv_batched` routes batched projections to
`gemm_skinny_*` at B >= 4, which holds ~400 GB/s where the `_b` kernels fall to
154 GB/s (B=2 stays on `_b`, measured better there: 45.3 vs 37.2). B=4 is a wash
because both paths sit at 273 GB/s; the remaining 1.3x there is the
non-projection half of the step. See `docs/review.md` §9 for the cost curve,
the exactness statement and the routing switches (`HIPML_BATCH_SKINNY`,
`HIPML_BATCH_LEGACY`, `HIPML_BATCH_SKINNY_TRACE`).

llama.cpp's np8 client-overlap figure was 88.96 tok/s; its server clock says
101.26.

### VRAM per slot and how many slots fit

At 8K context, q8 KV: **421.6 MiB per slot** (272.0 MiB KV + 149.6 MiB SSM
state), on top of the 16.01 GiB weight residency. With 13.59 GiB free after a
4-slot run, `batch-bench` estimates **37 slots**; a B=8 run left 12.52 GiB and
estimated 38. Both are allocator arithmetic, not a stress test.

### Where the B=4 step goes

Stage timing (`--split`, B=4; the split overhead itself makes the total 72.9 ms,
so read the shares, not the absolute values):

| stage | ms/step | share |
| --- | ---: | ---: |
| ffn (3 batched GEMVs x 64 layers) | 28.92 | 39.7% |
| ssm proj | 10.67 | 14.6% |
| ssm recurrence (per slot) | 9.60 | 13.2% |
| attn proj | 7.32 | 10.0% |
| ssm out | 6.65 | 9.1% |
| attn kv + core (per slot) | 4.20 | 5.8% |
| attn out | 3.25 | 4.5% |
| final (lm_head + argmax) | 2.07 | 2.8% |

The FFN batched GEMV is the single biggest kernel, and the batched GEMV family
as a whole runs at 271 GB/s at B=4 against 469 GB/s for the single-stream
kernel (15.68 GiB in 33.1 ms): a ~1.5x efficiency loss. The per-slot SSM
recurrence and attention together are ~14 ms of the 62.1 ms step.

## What is not verified

- The per-tensor `--probe` numbers: the probe crashed on a one-row x buffer and
  the fix built after the last measurement window.
- Mixed-depth token parity: `--check` is defined at depth 0 only.
- Split-mode parity at B=8 (B=4 passes).
- The 37-38 slot estimate was not exercised by actually allocating that many.


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
