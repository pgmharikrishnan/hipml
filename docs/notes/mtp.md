# MTP speculative decode on the GPU engine

> *Engineering log from development; internal work-package IDs and paths have been sanitized or removed where they were machine-specific. Current numbers: [docs/benchmarks.md](../benchmarks.md).*

Speculative decode with the model's own multi-token-prediction (MTP) head: a
small drafter proposes tokens, the target verifies them in **one weight pass**,
and greedy decode emits the accepted prefix plus a correction. The point is
bandwidth, not compute: decode streams the whole weight set per token, so
`R = 1 + drafts` tokens through one pass is the only way past
`achieved bandwidth / weight bytes`.

```
draft:  d1, d2, .. dD      drafter, D <= 3, recursive
verify: [t, d1, .. dD]     all 64 layers, one weight pass, R columns
accept: ids[c] == d_{c+1}  longest matching prefix a
emit:   d1..da, ids[a]     a accepted drafts + the correction
```

## The drafter

The sidecar GGUF (`mtp-Qwen3.8-27B-Q4_0.gguf`) holds one extra block, `blk.64`,
plus the MTP glue (`nextn.eh_proj`, `shared_head_norm`). `Model::load_sidecar`
loads those tensors into the **target's** resident tensor table, so the drafter
shares the target's launcher, stream and memory ordering; its own `token_embd`,
`output` and `output_norm` are skipped and the target's Q4_K embedding and Q6_K
head are reused. Drafter weights are **253.6 MiB** (Q4_K/Q6_K blocks), against
16.0 GiB resident for the target.

`MtpGpu` runs one block per drafted token:

* `eh_proj` (Q4_K 5120x10240) fuses the target hidden state with the previous
  token's embedding,
* one attention layer over a **f32, position-major KV cache** of its own
  (3.4 MiB at ctx 429, 64.0 MiB at 8K),
* `shared_head_norm` + the target's head for the next-token logits.

The drafter's prompt KV is built with a KV-only pass (`append_kv_only`: K/V
projections, k norm+rope, append), not a full block pass, so it costs one
projection per prompt token instead of a whole layer.

## The verify pass

`forward_columns` runs R tokens through all 64 layers **layer-major**, so the
weights are streamed once for the whole step:

* every projection uses the batched GEMV (`gemv_*_b`), which keeps the
  single-sequence k-accumulation order per row, so column `c` is bit-identical
  to a one-column decode (proved by `gemv-check`, see below);
* the GDN chain (gates, conv, both L2 norms, recurrence and gated RMS norm),
  the attention front end (q/gate split, head RMS norm, M-RoPE) and the KV
  append are each **one launch over all R columns**; the kernels fold the row
  index into the block index and keep the single-token expression order;
* causal attention itself is one split-K launch over all R columns, with the
  same fixed position partition (`per = ceil(n_ctx/splits)`) the one-column
  launch uses, so a column with a shorter sequence folds into the same sweep
  and contributes exactly zero past its position;
* each GDN layer writes the recurrent state after every column but the last
  into a `[columns][layers][state_len]` snapshot, one column stride per kernel;
  a rejected draft restores column `a` and moves `pos`, and the KV rows past
  `pos` are simply overwritten by the next append;
* after a step the drafter's KV is rewritten for the accepted positions from the
  target's post-norm hidden state, and the next draft is seeded from the last
  accepted column.

Bit-identity is against the plain decode kernels, which are separate
translation units: every mul+add in the conv, L2 norm, gated RMS norm, head RMS
norm and GDN reductions uses explicit `__fmaf_rn` on both sides, and the
L2-norm kernel indexes rows by the conv buffer's `conv_ch` stride. Before that
the compiler contracted the two files differently, a verify column differed
from a plain decode by a few ulps, and 13/128 greedy tokens flipped.

`gemv-check` proves the exactness premise on the GPU: for Q4_K, Q5_K, Q6_K, Q8_0
and the Q6_K head, **0 differing values** between the batched launch and the
row-tile decode entry at R=1, 2 and 4 (max |d| 0.000e0). `column-check`
compares the whole pass: at R=1, 2 and 4, the residual stream, the post-norm
hidden row and all `n_vocab` logits of every column are **bit-identical** to R
plain decodes (0 differing values).

## Measured

Radeon AI PRO R9700, `--kv f32`, 128 generated tokens per row, one process per
prompt (plain decode first, then 1..3 drafts on fresh engines). Every row was
run after the bit-identity fixes, so the `exact` column is again the honest
128-token token-for-token comparison.

### Decode tok/s, by drafts

| prompt | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| prose_explanation (373) | 29.23 | 41.82 | **42.58** | 36.96 |
| json_tool_call (300) | 29.16 | 42.93 | 49.56 | **50.06** |
| copy_200_words (285) | 29.16 | 45.86 | 55.40 | **61.12** |
| code_edit (672) | 29.11 | 44.23 | 49.53 | **51.43** |

Tokens per step and acceptance (`P(draft k | 1..k-1 correct)`):

| prompt | drafts | tok/step | accept | exact |
|---|---|---|---|---|
| prose | 1 | 1.814 | 0.814 | 128/128 |
| prose | 2 | 2.268 | 0.804, 0.600 | 128/128 |
| prose | 3 | 2.396 | 0.736, 0.590, 0.565 | 128/128 |
| json | 1 | 1.868 | 0.868 | 128/128 |
| json | 3 | 3.256 | 0.846, 0.909, 0.867 | 128/128 |
| copy | 1 | 1.984 | 1.000 | 128/128 |
| copy | 3 | 3.969 | 1.000, 1.000, 1.000 | 128/128 |
| code_edit | 1 | 1.924 | 0.924 | 128/128 |
| code_edit | 3 | 3.342 | 0.895, 0.912, 0.806 | 128/128 |

**Exactness**: all speculative rows are token-identical to plain greedy decode
for all 128 generated tokens, on all four prompts, at depth 0. `column-check`
shows the mechanism is exact, not lucky: at R=1, 2 and 4 every column's residual
stream, post-norm hidden row and `n_vocab` logits are bit-identical (0 differing
values) to that many plain decodes.

**No-accept downside**: a step that accepts nothing still costs the full step,
so the floor is `ms/step / plain ms/step`:

| drafts | ms/step | no-accept | breakdown |
|---|---|---|---|
| 1 | 43.4 | 1.27x | draft 2.7 + verify 40.6 + rollback 0.2 |
| 2 | 53.3 | 1.56x | draft 5.2 + verify 47.6 + rollback 0.5 |
| 3 | 64.9 | 1.89x | draft 7.7 + verify 56.5 + rollback 0.6 |

This is down from 1.46x/1.84x/2.28x in WP30 but still misses the 1.12x target:
the verify pass, not the drafter, is the cost. One extra column was 1.11x the
one-column pass; two extra were 1.29x; three extra 1.54x (measured before the
row-scaling fix below).

### 8K context

`longdoc_qa` (7694 prompt tokens), `--ctx 8192 --kv q8`, 64 tokens:

| drafts | ms/step | tok/step | tok/s | accept | exact |
|---|---|---|---|---|---|
| 0 | 34.87 | 1.000 | 28.68 | - | - |
| 1 | 48.22 | 1.537 | **31.87** | 0.561 | 64/64 |
| 2 | 59.93 | 1.800 | 30.04 | 0.543, 0.526 | 64/64 |
| 3 | 73.49 | 1.969 | 26.79 | 0.594, 0.526, 0.400 | 64/64 |

At 8K the drafter's KV is 64 MiB and its prompt pass costs 1.16 s. Acceptance
falls to ~0.55 (the drafter is weaker on long-document continuation), so d1 is
1.11x plain, d2 1.05x, and d3 is **below plain** (0.93x). Reported honestly:
speculation is not worth it at 8K with this drafter.

## Where the verify's time goes

`column-bench` times `forward_columns` on its own (prose prompt, 373 tokens).
Raw column throughput:

| columns | ms/pass | ms/column | tok/s | vs 1 col |
|---|---|---|---|---|
| 1 | 36.59 | 36.59 | 27.3 | 1.00x |
| 2 | 40.59 | 20.29 | 49.3 | 1.11x |
| 3 | 47.38 | 15.79 | 63.3 | 1.29x |
| 4 | 56.47 | 14.12 | 70.8 | 1.54x |

The weight pass is ~32 ms; the second column is the cheapest (1.11x), the third
and fourth add per-column work almost linearly. `--trace` at 2 columns:

| stage | ms/pass | note |
|---|---|---|
| up / gate / post_norm | 7.67 / 7.04 / 6.38 | batched GEMV, 477-514 GB/s |
| ssm_in / ssm_rec | 6.59 / 3.88 | GDN projections, 402 / 343 GB/s |
| ssm_proj | 3.34 | gates + conv + norms + recurrence, one launch |
| attn_in | 1.86 | q/k/v/qg GEMV + batched split/norm/rope |
| out_norm | 1.70 | head, 614 GB/s |
| attn_wo | 1.42 | gate + output GEMV |
| attn_mix | 1.18 | f32 split-K causal attention, all columns |
| lm_head | 0.13 | |
| down / ssm_step / block / embed / attn_append | <0.05 each | |
| sum | 41.3 | |

The batched f32 attention is now 1.18 ms/pass (was 8.5 ms when it ran per
column with 12-block grids); the split grid is `ceil(n_ctx/32)` clamped to 64.
The KV append is 0.007 ms. The remaining per-column cost is the GDN chain
(`ssm_rec` 3.88 + part of `ssm_proj`), which is sequential in token order and
cannot be weight-batched.

### Row scaling of the batched GEMV (2026-11-12)

The `_b` kernels already dequantised each weight block once and applied it to
all `B` rows, so the k loop was never per row. The growth with `B` came from
the x side: every k-tile staged all `B` x-rows through shared memory and then
read them back as scalar shared loads, so each block wrote `B * TILE` floats
per tile and the read-back cost scaled with `B` while the weight bytes did not.
Three smaller costs sat on top: IQ4_XS/IQ4_NL resolved the codebook through
`__constant__`, which is a global load per value on this target; Q5_K's
batched dequant used a per-value nibble plus fifth-bit select where the
single-row kernel uses one packed dword extraction; and Q8_0 read x as 32
scalar loads.

The fix reads each row's x straight from global (four 128-bit loads per lane;
`x` is at most `B * k` floats, so it stays in L2), drops the staging loop and
its two `__syncthreads()`, keeps the IQ4 codebook in 16 floats of shared, gives
Q5_K the single-row packed extraction, and vectorises Q8_0's x reads. Nothing
in the accumulation order changed: `gemv-check` reports 0 differing values at 2
and 4 columns for every kind it touches, and `column-check` reports 0
differing logits/hidden/residual at q8 and f32.

One real tensor per format, `gemv-bench --rows 16 --reps 20`, ms of the batched
pass with the ratio against the same kernel at `B=1`:

| tensor (kind) | B=1 | B=2 | B=4 | B=8 | B=16 |
| --- | --- | --- | --- | --- | --- |
| blk.22.ffn_up (Q5_K) before | 0.115 | 0.161 (1.40x) | 0.185 (1.61x) | 0.253 (2.20x) | 0.474 (4.12x) |
| blk.22.ffn_up (Q5_K) after | 0.114 | 0.118 (1.04x) | 0.153 (1.34x) | 0.209 (1.83x) | 0.384 (3.37x) |
| blk.0.ffn_gate (IQ4_XS) before | 0.091 | 0.165 (1.81x) | 0.158 (1.74x) | 0.305 (3.35x) | 0.559 (6.14x) |
| blk.0.ffn_gate (IQ4_XS) after | 0.096 | 0.104 (1.08x) | 0.133 (1.39x) | 0.298 (3.10x) | 0.556 (5.79x) |
| output (Q6_K) before | 1.696 | 1.856 (1.09x) | 2.378 (1.40x) | 3.557 (2.10x) | 7.021 (4.14x) |
| output (Q6_K) after | 1.695 | 1.885 (1.11x) | 2.179 (1.29x) | 2.943 (1.74x) | 5.838 (3.44x) |

Two rows now cost one weight pass plus 4-11%, four rows 1.3-1.4x. Above four
rows the per-row x traffic is no longer free (the same kernel covers `B` up to
8 and 16 with a smaller k-tile), but the verify never asks for more than 4.

End to end on the UD `Q4_K_M` model (`p2_math_169`, 169-token context), the
same two binaries, one session:

| measure | before | after |
| --- | --- | --- |
| `column-bench` q8, 1/2/3/4 cols (ms/pass) | 36.08 / 53.87 / 55.55 / 61.75 | 35.95 / 41.25 / 49.59 / 56.35 |
| `--vs 1 col` q8 | 1.00x / 1.49x / 1.54x / 1.71x | 1.00x / 1.15x / 1.38x / 1.57x |
| `column-bench` f32, 1/2/3/4 cols | 33.31 / 49.89 / 51.28 / 56.56 | 33.37 / 37.68 / 45.19 / 51.03 |
| `--vs 1 col` f32 | 1.00x / 1.50x / 1.54x / 1.70x | 1.00x / 1.13x / 1.35x / 1.53x |
| `spec-bench` q8, drafts 0/1/2/3 (tok/s) | 30.44 / 33.90 / 45.71 / 48.34 | 30.38 / 43.98 / 50.29 / 52.35 |
| `spec-bench` f32, drafts 0/1/2/3 | 30.45 / 36.01 / 48.90 / 51.99 | 30.37 / 47.24 / 55.08 / 57.24 |

All eight speculative rows stay 128/128 exact against plain greedy. On this
measurement a two-row verify is 1.15x (q8) / 1.13x (f32) a one-row pass, under
the 1.2x target; four rows are 1.57x / 1.53x and still miss it. What is left at
four rows is not the GEMV: at q8 KV the per-column attention arm launches
`attn_decode_kv_w2` once per column (a dozen-odd blocks at short context) where
f32 runs one `attn_decode_cols_f32` for all columns, and the GDN chain stays
sequential in token order. The batched q8 attention kernel
(`attn_decode_kv_q8_b`) is not the same kernel the plain path runs, so
swapping it in would break the bit-identity this table relies on.

Measured and reverted: hoisting all `B` rows' x above the dequant (regressed
`B=4`), packed IQ4_XS nibble extraction (0.109 -> 0.179 ms at `B=2`), and
raising `ROWS_PER_WARP` from 2 to 4 (regressed `B=4`). IQ4_XS keeps the
per-`B` preload only up to `B=4`; above that it reads one row at a time, because
keeping all rows live spilled and made the wide batch slower than the staged
version it replaced.

### Four-column verify: per-column attention and the GDN chain (2026-11-12)

The row-scaling fix left the four-column verify at 1.57x (q8) / 1.53x (f32) a
one-column pass. A trace at four columns put two stages on the wrong side of
the line: `attn_mix` 7.37 ms at q8 (f32 ran one `attn_decode_cols_f32` for all
columns, but q8 launched `attn_decode_kv_w2` once per column) and the GDN
recurrence 6.03 ms (one block per value head, each walking all of the head's
state rows). Two changes:

* `attn_decode_kv_w2_b`: the same whole-group warp body with one `grid.y` step
  per column, carrying the column's query row, partial scratch slice and
  `(position, seq)` pair. One launch per group of columns that share a split
  count. The staged `attn_decode_kv_q8_b` would not do: it is a different
  kernel, so a column would stop being bit-identical to the plain step.
* `gdn_cols_f32`: `gridDim.y` splits the head's state rows over several blocks
  (independent chains, so any split is bit-identical), and the `(token, head)`
  gate pair is computed once per block into shared instead of once per
  (state row, token) — the state rows are the inner loop, so `expf` was being
  recomputed per row.

Measured separately on `column-bench --cols 4` (q8 unless stated):

| stage / measure | before | attention fix only | both fixes |
| --- | ---: | ---: | ---: |
| `attn_mix` | 7.37 ms | 1.16 ms | 1.19 ms |
| GDN chain (`ssm_proj` span) | 6.03 ms | 6.02 ms | 1.46 ms |
| four-column pass | 55.37 ms | 50.23 ms | 46.07 ms |
| f32 four-column pass | 50.73 ms | 50.87 ms | 47.11 ms |

`column-check` is 0-differ for q8 at 2 and 4 columns and f32 at 4 (logits,
hidden and residual), and `spec-bench` stays 128/128 exact on every row.

| measure | before | after |
| --- | --- | --- |
| `column-bench` q8, 1/2/3/4 cols (ms/pass) | 34.94 / 40.37 / 48.55 / 55.37 | 34.51 / 35.19 / 41.68 / 46.07 |
| `--vs 1 col` q8 | 1.00x / 1.16x / 1.39x / 1.58x | 1.00x / 1.02x / 1.21x / 1.33x |
| `column-bench` f32, 1/2/3/4 cols | 33.20 / 37.43 / 44.90 / 50.73 | 32.14 / 35.45 / 42.40 / 47.11 |
| `--vs 1 col` f32 | 1.00x / 1.13x / 1.35x / 1.53x | 1.00x / 1.10x / 1.32x / 1.47x |
| `spec-bench` q8, drafts 0/1/2/3 (tok/s) | 30.38 / 43.98 / 50.29 / 52.35 | 30.47 / 50.23 / 59.22 / 62.32 |
| `spec-bench` f32, drafts 0/1/2/3 | 30.37 / 47.24 / 55.08 / 57.24 | 30.43 / 50.06 / 58.55 / 61.55 |

Serve cell a (one request at a time, 5 conversations, greedy, 64 tokens,
2,423-token prompt) moves from 42.22 / 44.09 / 44.53 tok/s at drafts 1/2/3 to
47.27 / 50.71 / 51.91, TTFT 3.08 -> 3.11 s. The two fixes are what the verify
pass gained; the cell is still below llama.cpp's 62.45 because its prompts
accept less (0.70 here) and because the remaining 4-row cost is the batched
GEMV, not the attention or the scan.

**The 1.2x target is not reached, and the residue is the GEMV.** At four
columns the weight stages run at 337-448 GB/s where a one-column pass runs at
450-660 GB/s; in the q8 trace the seven weight stages total 42.2 ms of the
46.2 ms pass, against 30.7 ms for the same stages traced at one column (f32
run, same model bytes), while the two stages this slice fixed are down to
2.7 ms. Removing *all* remaining non-weight work would leave 43.5 ms, still
1.25x. Four rows therefore need the same 4-11% treatment two rows got in
`gemv_*.hip`, which is a separate change.

The `_b` kernels the serve process runs at B=4 are the exact ones even though
`serve` sets `HIPML_BATCH_SKINNY=1`, and that is deliberate: `skinny_on()` reads
`HIPML_SKINNY`, `forward_columns` takes an `exact_scope()` unless that is set,
and `gemv_batched_at` refuses the WMMA tile while the scope is held. Reproduced
with `HIPML_BATCH_SKINNY=1 HIPML_BATCH_SKINNY_TRACE=1` on `spec-bench --drafts
3`: no `[skinny]` line is printed. The non-bit-identical tile at B>=4 is the
exact defect the review-4 audit fixed; the exact route must stay.

## What is not verified / not done

* The prose ≥48 / JSON ≥53 tok/s targets are **not met** (42.6 / 50.1, before
  the row-scaling fix; the four-prompt table was not re-run afterwards). copy
  reaches 61.1 (target 65) and code 51.4. The gap is the verify pass, not
  acceptance.
* The 8K path is measured with `--kv q8` only; `--kv f32` at 8K was not run.
  At 8K, 3 drafts is slower than plain.
* Acceptance is measured on four fixtures, 128 tokens each; no longer-horizon
  drift study.
* Rollback restores the `[columns][layers]` snapshot and relies on the KV
  overwrite; the state snapshot itself is not compared against a replayed
  prefix. Exactness of the emitted ids over 128 tokens is the evidence.
* The WP33 WMMA skinny quantized GEMM (`feat/skinny-wmma`) is copied in
  behind `HIPML_SKINNY=1`; the exact batched GEMV stays the default. Measured
  both ways on the four fixtures (128 tokens):

  | prompt | exact best | skinny best | logit max |d| | skinny exact |
  |---|---|---|---|---|
  | prose | 42.58 (d2) | 38.71 (d2) | 9.4e-3 | 128/128 |
  | json | 50.06 (d3) | 51.21 (d3) | 9.4e-3 | 128/128 |
  | copy | 61.12 (d3) | 62.51 (d3) | 9.4e-3 | 128/128 |
  | code | 51.43 (d3) | 52.46 (d3) | 9.4e-3 | 128/128 |

  The skinny path is *not* a token divergence (128/128 on every prompt), but it
  is not faster enough: its dequantise-into-shared staging runs at ~40-68 GB/s
  of weight against the batched GEMV's 400-500 GB/s, so at r=2/3 it is slower
  and only at r=4 does it reach parity (prose verify 55.3 vs 56.5 ms). It does
  not reach the 65-80 tok/s class. Keep the exact path default.
* No graph capture on the verify path; launch overhead is inside every number.
