# Decode residuals: the 1.93 ms was the trace, not the kernels

Box: R9700 (gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf`, q8 KV,
chunk 2048, port 8093. Every GPU run held one lock; the two binaries under
comparison alternated inside a single lock window for every A/B below.
Resident 15.42 GiB, 14.42 GiB streamed per token, 0.272 GiB of KV at 8,192.

## 0. Question and answer

[`decode-epilogue.md`](decode-epilogue.md) closed with one pot left: "the only
remaining pot is the 1.93 ms of residual adds" — 128 `residual_add_f32`
launches a token at 15.0 µs a stage span, with all three ways of merging them
into a *consumer* already measured (residual + norm: 0.19 ms slower, 96 fewer
GDN launches: 5 µs, wider norm grid: not bit-identical).

One way was untested, because it moves in the other direction: don't merge the
residual into the op that *reads* it, merge it into the op that *produces* the
other addend. Every one of the 128 is `x1 = x + block_out` or `x = x1 +
ffn_down`, and both producers are projection GEMVs, so the add can be the
GEMV's epilogue — `y[row] = a + res[row]` — and the launch disappears.

Answer: **built, exact, and neutral. −0.016 ms at depth 0, −0.026 ms at 8,192;
reverted.** 32.019 ms/token at depth 0 against the base tree's 32.035 — 0.05%,
below the run-to-run spread of either binary — and 1.7 ms short of the 30.3 ms
the card wanted. The 1.93 ms the previous note budgeted for these 128 launches
is mostly the stage table: 128 launches a token are worth ~0.02 ms of the
token, and the epilogue work the fusion adds to the producers costs about as
much as the launches it removes.

The second result in here is methodological and it is the one to keep. Two of
eighteen stage rows were deleted and the traced token fell 0.94 ms while the
untraced token in the same process fell 0.05 ms. A stage span is not a kernel
measurement, and the residual pot in the previous note was read off spans.

## 1. Attribution: all 128 fold, none left over

Counted by asking the layer plan which projection writes each GEMV's output:

| add | shape | producer | layers |
| --- | --- | --- | ---: |
| `x1 = x + block_out` | after the full-attention block | `attn_output` GEMV | 16 |
| `x1 = x + block_out` | after the GDN block | `ssm_out` GEMV | 48 |
| `x = x1 + ffn_down` | after the FFN | `ffn_down` GEMV | 64 |

128 residual adds, 128 producers, 0 left over. Six of the model's block types
appear as producers, and Q4_K and Q6_K are launched in two geometries (one row
per warp and the default tile), so the epilogue needed **eight** entries, not
six.

## 2. What was built

| where | change |
| --- | --- |
| `gemv_{q4_k,q4_k_r1,q5_k,q6_k,q6_k_r1,iq4_xs,iq4_nl,iq3_s}.hip` | `_res` entry with one extra `const float* res`; `y[row] = acc + res[row]` |
| the shared tile helpers (`q4k_tile`, `q5k_rows`, `gemv_jobs`) | a `RES` template bool, so the plain entry compiles to the code it had before (`false, nullptr`) |
| `quant_launch.rs`, `lib.rs` | `run_quant_resident_res`, `gemv_res_entry` (name → fused entry) |
| `gpu_weights.rs`, `attn_gpu.rs`, `ssm_ops.rs`, `gpu.rs` | `Proj::gemv_res`, `apply_output_res`, `output_gemv_res`; `run_token` asks for the fused form and issues the separate `residual_add_f32` when the format has no entry |

The accumulation order of every GEMV is untouched — the epilogue runs after the
reduction, which is what keeps the exactness gates green. Nothing in the
attention core, the GDN step, the prefill tiles, the serve scheduler, the
batched-GEMV dispatch or `graph_launch.rs` was touched; those paths keep their
`residual_buf` and their separate adds.

The residual value is fetched into a register *before* the k loop rather than
after it, because a dependent load at the end of the last wave would sit on the
store's critical path. Both builds measured the same token (32.0 ms, inside the
spread), so it was never a latency problem.

`launches_per_token` 1,299 → **1,171**; the traced token 1,140 → **1,012**.
Both counts are structural, from the layer plan, so they cannot drift from the
code.

## 3. e2e: a wash

200 tokens a run, three reps per binary per depth, two processes, both binaries
alternating inside one lock hold. ms/token, all twelve runs:

| depth | base | fused | delta |
| --- | --- | --- | ---: |
| 0 | 31.871 / 32.027 / 32.053 / 32.059 / 32.100 / 32.100 | 31.900 / 32.011 / 32.013 / 32.057 / 32.065 / 32.070 | **−0.016** |
| 0, mean | **32.035** (31.22 tok/s) | **32.019** (31.23 tok/s) | −0.05% |
| 8,192 | 32.542 / 32.596 / 32.602 / 32.616 / 32.642 / 32.645 | 32.527 / 32.560 / 32.579 / 32.584 / 32.618 / 32.619 | **−0.026** |
| 8,192, mean | **32.607** (30.67 tok/s) | **32.581** (30.69 tok/s) | −0.08% |

Eleven of the twelve paired deltas are negative; the twelfth is the first run
of the campaign (+0.029 ms, clock ramp). The within-cell spread is 0.09–0.23 ms
— four to nine times the delta — so the honest reading is a systematic sign
with no magnitude: **the fusion is free, and it is not worth anything.**

Target check: ≤30.3 ms at depth 0. Measured 32.02. The 128 launches the card
priced at ~1.9 ms are 0.02 ms of the token.

## 4. Where the 1.93 ms went

`gpu-profile --trace`, four tokens, base and fused, alternating in the same
lock hold, depth 0, ms/token:

| stage | base | fused | delta |
| --- | ---: | ---: | ---: |
| `ffn_down` | 6.796 | 7.254 | **+0.458** |
| `ssm_out` | 2.304 | 2.726 | **+0.422** |
| `attn_wo` | 0.704 | 0.781 | **+0.077** |
| `residual_out` | 0.971 | — | −0.971 |
| `residual_in` | 0.929 | — | −0.929 |
| everything else | 21.163 | 21.168 | +0.005 |
| **sum** | **32.867** | **31.929** | **−0.938** |
| launches/token | 1,140 | 1,012 | −128 |

At 8,192 the same table reads 33.904 → 32.928 (−0.976), with the producers
taking +0.456 / +0.462 / +0.130 and the two residual rows (0.978 + 0.973)
gone.

So the traced token says the trade wins 0.94 ms, and the untraced step in the
same process says 0.05 ms. The same process measures both: `graph token` (the
untraced step, warm) moved 31.597 → 31.551 at depth 0 and 32.950 → 32.704 at
8,192. The trace overhead over the base token is **0.95–1.27 ms** and over the
fused token **0.22–0.38 ms**, with the only difference between the two runs
being two of eighteen stage rows.

A stage mark is one recorded HIP event that opens a span, and a span holds
sixty-four launches here. Whatever the mechanism — the event's own cost, or the
bubble a boundary opens in front of a 3 µs kernel — the table charges a
residual span ~15 µs a span (8.5 µs from the per-kernel probe, itself inflated
by its own event pair; `decode-epilogue.md` §1) and the token only notices
0.02 ms of it. **Read the stage table for ranking, as
[`decode-dispatch.md`](decode-dispatch.md) §1 already said about the per-kernel
column; do not price a launch from it.**

The per-kernel mode (`HIPML_KERNEL_SPAN=1`) was re-run on both builds and is
not quoted anywhere above. Its launch column is right for the elementwise rows
and wrong for the rows this slice cares about. Both runs fold spans from about
**6.5 tokens** — the untraced warm-up as well as the four traced ones — and
divide by four: `gdn_ar_f32` is 312 = 6.5 × 48, `rmsnorm_f32` 838 = 6.5 × 129,
`q_gate_f32` and `attn_front_f32` 104 = 6.5 × 16, `silu_mul_f32` 416 = 6.5 × 64.
Against that, `residual_add_f32` reads 1,848 in the base (6.5 × 128 is 832) and
**1,016 in the fused tree, whose decode issues none**: in that same run the
stage table counts 1,012 launches in total, and the fused tree's stage table has
no residual stage at all. The eight `_res` rows do appear with the weights their
shapes predict, so every fused entry is exercised, but a table that cannot count
the one kernel it is there to find is not evidence. The stage table's launch
column matches the layer plan exactly, which is why that is the one quoted.

## 5. Why the fusion does not pay

The epilogue's own traffic is nothing: a 20 KB read and a 20 KB write against
the tens of megabytes the projection streams — 14.42 GiB a token over 64
layers is 241 MB of weights a layer — and one more host argument to a launch
the driver already batches. Yet the producer spans grew by 0.96 ms over 128
launches, 7.5 µs a launch, which is the same order as a whole launch in the
span table. Both numbers are span-table numbers, so treat the 7.5 as an upper
bound and the e2e as the truth: the work moved, the time did not.

That is consistent with everything else measured on this card. The token is
submitted as one batched launch list in which the host runs a token ahead
(`decode-dispatch.md` §2a, §2b), so a kernel that costs 2.6 µs to *issue*
standalone costs the GPU far less to *consume* back-to-back: deleting a 60 KB
elementwise launch buys ~0.0004 ms, which is the 0.02 ms of §3 over 128
launches. There is no launch-side money left in the residual adds, and folding
their work into the neighbouring GEMV does not create any.

This is the fourth launch-count lever in the series to come back neutral or
worse (whole-step HIP graph +0.37 ms, residual into the following norm +0.19,
the two GDN merges 5 µs for 96 launches, this one −0.02 for 128), and the first
one that removes the launch by *moving its work* rather than by joining two
elementwise ops. The conclusion of the series now covers both shapes: on this
card a merge pays only when it removes memory traffic or exposes parallelism —
`attn_front_f32` (−0.15 ms, three global round-trips removed) and the GDN
row-stripe (−0.49 ms, 32 of 64 CUs idle) are the two that did.

## 6. Gates, on the fused build

Everything below was run on the fused tree before it was reverted, so the
negative result is about time and not about correctness.

| gate | result |
| --- | --- |
| `parity` / `parity --deep` | 72 / 74 passed, 0 failed, 1 skipped (`sample_top_p_f32`, skipped in both) |
| `gemv-check`, 6 tensors × rows 1/2/4 | 18 checks, 0 differ, max abs 0.0 |
| `column-bench --check` q8 cols 2, q8 cols 4, f32 cols 4 | 0 differ on logits (7,946,240 values), hidden and residual |
| `batch-bench --slots 16 --tokens 16 --check` | 102.38 tok/s aggregate, 16/16 ids on all 16 slots |
| `spec-bench` drafts 0–3, q8 and f32, 128 tokens | 128/128 exact on every row; 31.2 / 50.2 / 59.0 / 62.3 tok/s (q8), 31.3 / 50.1 / 58.5 / 61.4 (f32) |
| `serve` cell a, 1 slot at 32K, `--drafts 1`, 5 requests × 3 waves | 47.08 / 47.16 tok/s against base 47.17 / 47.21, same 448/508 accepted, 0 errors |
| `oracle-gate` UD-Q4_K_M, 20 steps, chunk 2048, q8 | PASS 20/20, prefill max abs 0.1086 at 101748 |
| `oracle-gate` stock Q4_K_M, 20 steps, chunk 2048, q8 | PASS 20/20, prefill max abs 0.1481 at 34114 |
| greedy 20 ids, UD | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760, 6511, 314, 14898, 369, 21047, 13, 198, 760]`, 31.31 ms/token |
| greedy 200 ids, depth 0 and 8,192 | byte-identical to `45f6bb5`: last id 108899 / 197, `fc /b` no differences |
| `cargo test --workspace` | 0 failed |
| `cargo fmt --all -- --check` | clean |

## 7. What was reverted

All of it: the eight `_res` entries, the `RES` template parameter on the tile
helpers, the host plumbing (`gemv_res_entry`, `run_quant_resident_res`,
`Proj::gemv_res`, `apply_output_res`, `output_gemv_res`), the `folded` branch in
`run_token`, and the launch-count comments that went with them. `main` keeps
the separate `residual_add_f32` launches and 1,299 launches a token.

It was correct and it was 0.05–0.08% faster; 440 lines across fifteen files, a
second copy of six formats' epilogue arithmetic that has to stay bit-identical
forever, and eight entry points for the batched path to keep in step, is not
what 0.02 ms buys. The bar in the card was 2.0 ms.

Two pots named in [`decode-epilogue.md`](decode-epilogue.md) §7 are still open
and were not touched here: `q_gate_f32` folded into the attention reduce
(~0.15 ms, bit-identical) and the 8,192 attention core at 258 GB/s (−0.55 ms at
8,192, still short of 31.25 ms on its own).
