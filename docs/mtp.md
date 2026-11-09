# MTP speculative decode on the GPU engine

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
| code_edit (672) | 29.11 | 44.23 | **49.53** | 51.43 |

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
the verify pass, not the drafter, is the cost. One extra column is 1.11x the
one-column pass; two extra are 1.29x; three extra 1.54x.

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

## What is not verified / not done

* The prose ≥48 / JSON ≥53 tok/s targets are **not met** (42.6 / 50.1). copy
  reaches 61.1 (target 65) and code 49.5. The gap is the verify pass, not
  acceptance.
* The 8K path is measured with `--kv q8` only; `--kv f32` at 8K was not run.
  At 8K, 3 drafts is slower than plain.
* Acceptance is measured on four fixtures, 128 tokens each; no longer-horizon
  drift study.
* Rollback restores the `[columns][layers]` snapshot and relies on the KV
  overwrite; the state snapshot itself is not compared against a replayed
  prefix. Exactness of the emitted ids over 128 tokens is the evidence.
* The WP33 WMMA skinny quantized GEMM (`feat/skinny-wmma`) was **not adopted**
  in this change: it is not bit-identical (max_abs ~1.6e-2) and the priority was
  the exact path. Its measured cost curve (r=4 ≈ 0.93-1.05x of one column)
  makes it the obvious next lever for the 65-80 tok/s class, behind a
  `--skinny` switch with the 128-token divergence check. If it costs a token,
  keep the exact batched GEMV as default.
* No graph capture on the verify path; launch overhead is inside every number.
