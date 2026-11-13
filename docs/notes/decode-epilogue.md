# Decode epilogue: the non-GEMV kernels, kernel by kernel

Box: R9700 (gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf`, q8 KV,
chunk 2048, port 8093. Every GPU run held the `decode-epilogue` lock, and the
binaries under comparison alternated inside one lock window for every A/B.

## 0. Question

[`decode-dispatch.md`](decode-dispatch.md) closed with a budget it could not
move: the token is 32.8 ms at depth 0 and 33.5 ms at 8,192, the weight stream
is 24.5 ms of it at 632 GB/s, and **5.3 ms is GPU-side kernel execution in the
~800 launches that are not GEMVs**. Launch-count levers were measured there and
did not pay (a whole-step HIP graph: neutral; 128 fewer launches: 0.6% slower).

This slice took the kernels themselves — the attention front end and core, the
GDN step, and norm/rope/residual/silu — and asked which of them the 5.3 ms is
in, and how much a kernel can give back.

Answer: **0.80 ms, to 31.99 ms / 31.26 tok/s at depth 0 and 32.58 ms /
30.69 tok/s at 8,192.** 32.0 tok/s is not reachable from this side either: it
needs 31.25 ms, which is the 24.5 ms weight floor plus 6.75 ms of everything
else, and the non-GEMV kernels that remain are 128 residual adds at 8.5 µs, 129
RMS norms at 6.5 µs and ~2.7 ms of gap between GEMV launches — none of them a
launch-count problem, which is why two more launch merges measured neutral and
one measured slower.

## 1. Instrumentation: per-stage, then per-kernel

`gpu-profile --trace` (shipped here, with `--depth` and `--kv`) opens a stage
with one recorded HIP event and counts the launches inside it, so a token
reports `ms/token` / launch count / `GB/s` per stage. One event per stage is
cheap — traced minus untraced wall over ~1,140 launches is 0.76 ms — and the
stage table is what found the two levers in §3.

A stage is a *span*, so it cannot separate a kernel from the gap around it.
`HIPML_KERNEL_SPAN=1` adds the other axis: an event pair around every launch,
folded into one row per kernel name. It costs more than the stage table (the
box reports a cached launch at 25–27 µs with it on against 1.7 µs with it off,
i.e. two event records are ~12 µs of host time each), so the absolute figures
are inflated and only the ranking is usable. That ranking is what pointed at
the two kernels this slice could not fix and the one it could:

| kernel | launches/token | µs/launch | what it is |
| --- | ---: | ---: | --- |
| `residual_add_f32` | 128 | **8.5** | `x = x + block_out`, 20 blocks of 256 |
| `rmsnorm_f32` | 129 | **6.5** | one block of 256 over the 5,120-float row |
| `q_gate_f32` | 16 | 9.6 | `attn_out *= silu(gate)` |
| `gdn_ar_f32` | 48 | **41** | the recurrence, 32 blocks |
| `ssm_gate_norm_f32` | 48 | 2.9 | gated RMS norm |
| `attn_front_f32` | 16 | 2.4 | the fused front end (§3) |
| `ssm_conv_silu_f32` | 48 | 2.3 | the fused conv + SiLU (§3) |
| `silu_mul_f32` | 64 | 1.3 | the FFN gate, 68 blocks |
| `ssm_gates_f32` | 48 | 0.7 | 32-element softplus/sigmoid |

Two things follow from the column, and both were tested rather than assumed.
`silu_mul_f32` moves three times the bytes of `residual_add_f32` in a sixth of
the time, so the cost of the two X-stream kernels is *not* their size: it is
that 20 blocks (or one, for the norm) cannot cover memory latency on a 64-CU
card. And `gdn_ar_f32` is 41 µs on a 32-block grid — half the CUs idle, which
is the one place the ranking turned into a fix with real headroom.

## 2. The GDN recurrence was on 32 of 64 CUs

The kernel is one block of 128 threads per value head, and a head is 128 value
rows: eight passes of `GDN_ROWS × nwave = 16` rows. 32 heads is 32 blocks, so
half the CUs were idle and the rest were latency-bound on state loads. The grid
now has a y axis that stripes a head's value rows across `gridDim.y` blocks —
rows never interact, so which block runs a row changes nothing about what it
computes. `HIPML_GDN_SPLIT` selects the width; eight is one full row pass per
block, and it is also where the curve flattens:

| split | `ssm_step` stage | graph token, depth 0 |
| ---: | ---: | ---: |
| 1 (shipped before) | 1.038 ms | 31.945 ms |
| 2 | 0.573 ms | 31.783 ms |
| 4 | 0.280 ms | 31.558 ms |
| 8 | **0.193 ms** | **31.497 ms** |

The stage falls 5.4× but the token only 0.45 ms: the box re-absorbs most of it
as gap elsewhere, which is the same behaviour every launch-count lever in this
series has shown. The e2e A/B in §4 is the number to quote.

## 3. Three fusions, one of which paid

| what | launches/layer | stage, before → after | e2e |
| --- | ---: | --- | ---: |
| front end: `qg_split` + 2 head RMS norms + 2 M-RoPE → `attn_front_f32` | 5 → 1 | `attn_qkv` 1.631 → 1.475 ms | −0.15 ms |
| GDN conv + SiLU → `ssm_conv_silu_f32`; the two L2 norms → `ssm_l2norm2_f32` | 7 → 5 | `ssm_step` 1.073 → 1.068 ms (same binary, `HIPML_SSM_SPLIT`) | ~0 |
| GDN recurrence: four state rows in flight per wave, then the row stripes of §2 | — | `ssm_step` 1.178 → 1.073 → 0.193 ms | −0.10 then −0.49 ms |

The front end is five launches over one 5,120-float row; one block per head
also drops three intermediate global round-trips, and it is bit-identical
because each of the five steps is a per-element or per-head op with no
cross-head reduction.

The two GDN merges are the control: 96 fewer launches a token for 5 µs, i.e.
nothing. That is the third independent measurement of the same thing (a
whole-step graph, the residual-into-norm merge in `decode-dispatch.md`, and
these two), so the note's negative result is now well supported: on this card
a merge pays only when it removes memory traffic or exposes parallelism.

## 4. e2e

200 tokens a run, three reps per binary per depth, interleaved in one lock
hold. `o` is this tree with `HIPML_GDN_SPLIT=1`, so the split's own share is
`o − n` and the tree's is `b − n`.

| depth | base `da21476` | `o` (split 1) | `n` (this tree) | base → tree |
| --- | --- | --- | --- | --- |
| 0, ms/token | 32.640 / 32.837 / 32.896 (32.791) | 32.394 / 32.484 / 32.543 (32.474) | 31.889 / 32.010 / 32.061 (31.987) | **−0.80 ms** |
| 0, tok/s | 30.50 | 30.80 | **31.26** | +2.5% |
| 8,192, ms/token | 33.349 / 33.409 / 33.451 (33.403) | 32.994 / 33.078 / 33.115 (33.062) | 32.544 / 32.585 / 32.624 (32.584) | **−0.82 ms** |
| 8,192, tok/s | 29.94 | 30.25 | **30.69** | +2.5% |

No rep reverses; the three within-binary spreads are 0.2–0.3% and the
between-binary gaps are 25–120× the spread.

Rebased onto `main`, which by then carried the FFN gate/up fusion (#19), the
same binary reads **31.92 ms / 31.33 tok/s at depth 0 and 32.53 ms /
30.74 tok/s at 8,192**; the table above is this slice's own contribution
against the parent it was cut from, not against current `main`.

## 5. The budget, and why 32 tok/s is still out of reach

Depth 0, ms/token, from the stage table on this tree (`sum` 32.660 in the
traced run, against 31.99 untraced — the trace itself is the 0.7 ms):

| component | ms | note |
| --- | ---: | --- |
| weight stream, 15.48 GB at 632 GB/s | 24.50 | floor, measured in `decode-roofline.md` |
| GEMV above that floor, in-graph | ~5.6 | 497 GEMV launches reading at 438–735 GB/s |
| `residual_add_f32`, 128 launches | 1.93 | 15.0 µs a stage span, of which the kernel probe says 8.5 and the rest is the gap to the next launch |
| RMS norms, 129 launches | 0.14 | 6.5 µs a launch |
| GDN step, 48 layers × 5 launches | 0.13 | after §2 |
| attention core and KV append | 0.31 | the fused front end is inside `attn_qkv` |
| FFN SiLU, embed, output norm | 0.12 | |
| **token, traced** | **32.77** | the same work untraced is 31.99 |

32.0 tok/s is 31.25 ms: the weight floor plus 6.75 ms. The non-GEMV kernels
measured here are 2.6 ms of that, and the GEMV launch gaps are ~2.8 ms more, so
the only remaining pot is the 1.93 ms of residual adds. That is 128 kernels of
60 KB each that cannot cover memory latency at 20 blocks, and the three ways to
touch them are all measured and all closed: merging one into the following norm
is 0.19 ms slower (`decode-dispatch.md`), merging the two GDN pairs is 5 µs for
96 fewer launches (§3), and the norm side of the residual/norm pair cannot take
a wider grid without changing its reduction order, i.e. without leaving
bit-identical.

## 6. Gates

| gate | result |
| --- | --- |
| `parity --deep` | 74 passed, 0 failed, 1 skipped |
| `column-check` q8 col 2 / col 4, f32 col 4 | 0 differ, max abs 0.0 (logits, hidden, residual) |
| `spec-bench` drafts 0–3, q8 and f32, 128 tokens | 128/128 exact on every row; 31.2 / 50.2 / 59.1 / 62.3 tok/s (q8), 31.2 / 50.0 / 58.4 / 61.3 (f32) |
| `oracle-gate` UD-Q4_K_M, 20 steps, chunk 2048, q8 | PASS 20/20, prefill max abs 0.1086 at 101748 |
| `oracle-gate` stock Q4_K_M, 20 steps, chunk 2048, q8 | PASS 20/20, prefill max abs 0.1481 at 34114 |
| greedy 20 ids, UD | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760, …]`, identical to `da21476` |
| greedy 200 ids, depth 0 and 8,192 | last id 108899 / 197, byte-identical to `da21476` (`fc /b`, no differences) |
| `batch-bench --slots 16 --tokens 16 --check` | 103.25 tok/s aggregate, 16/16 ids on all 16 slots |
| `cargo test --workspace` | 87 passed, 0 failed |
| `cargo fmt --all -- --check`, `cargo check --workspace` | clean |

## 7. What is not in here

- **A merged residual + RMS norm for decode.** Already measured at +0.19 ms
  and left reverted in `decode-dispatch.md`; this slice's two GDN merges are
  the same result, so it was not retried.
- **A faster `rmsnorm_f32`.** Its sum-of-squares tree is pinned to a 256-slot
  block to match the reference's rounding; any wider grid changes the order and
  the exactness gates.
- **`q_gate_f32` folded into the attention reduce kernel** (9.6 µs × 16, i.e.
  ~0.15 ms). Bit-identical and worth doing, but it threads a param through the
  f32 reduce, the q8 reduce and the `rows`/`cols` call sites for 0.5% of the
  token, so it belongs in a slice that can measure the attention core too.
- **The 8,192 attention core.** `attn_decode_kv_w2` reads the q8 cache at
  258 GB/s (65 µs a layer against a ~26 µs floor); halving it would be −0.55 ms
  at 8,192 and still leave the depth short of 31.25 ms.
