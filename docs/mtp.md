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
* work that is genuinely sequential in time stays per column and runs the same
  decode kernels: the GDN recurrence, the conv window, the KV append and causal
  attention;
* each GDN layer copies its recurrent state to `snap[c]` right after column
  `c`; a rejected draft restores `snap[a]` and moves `pos`, and the KV rows past
  `pos` are simply overwritten by the next append;
* after a step the drafter's KV is rewritten for the accepted positions from the
  target's post-norm hidden state, matching the CPU reference's per-position
  seeding, and the next draft is seeded from the last accepted column.

`gemv-check` proves the exactness premise on the GPU: for Q4_K, Q5_K, Q6_K, Q8_0
and the Q6_K head, **0 differing values** between the batched launch and R
single launches at R=2 and R=4 (max |d| 0.000e0).

## Measured

Radeon AI PRO R9700, `--kv f32`, 128 generated tokens per row, one process per
prompt (plain decode first, then 1..3 drafts on fresh engines).

### Decode tok/s, by drafts

| prompt | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| prose_explanation (373) | 28.10 | 34.82 | 34.66 | 29.56 |
| json_tool_call (300) | 27.76 | 35.33 | 39.65 | **39.74** |
| copy_200_words (285) | 27.84 | 37.79 | 44.58 | **49.00** |
| code_edit (672) | 27.72 | 36.39 | 39.73 | 40.64 |

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

**Exactness**: all 12 speculative rows are token-identical to plain greedy
decode for all 128 generated tokens, on all four prompts, at depth 0. No
fallback to "argmax equal" was needed.

**No-accept downside**: a step that accepts nothing still costs the full step,
so the floor is `ms/step / plain ms/step`:

| drafts | ms/step | no-accept | breakdown |
|---|---|---|---|
| 1 | 52.1 | 1.46x | draft 2.7 + verify 49.3 + rollback 0.1 |
| 2 | 65.4 | 1.84x | draft 5.3 + verify 59.7 + rollback 0.4 |
| 3 | 81.1 | 2.28x | draft 8.0 + verify 72.5 + rollback 0.6 |

This misses the <1.10x target: the verify pass, not the drafter, is the cost.

### 8K context

`longdoc_qa` (7694 prompt tokens), `--ctx 8192 --kv q8`, 64 tokens:

| drafts | ms/step | tok/step | tok/s | accept | exact |
|---|---|---|---|---|---|
| 0 | 34.72 | 1.000 | 28.80 | - | - |
| 1 | 51.02 | 1.537 | 30.12 | 0.561 | 64/64 |
| 2 | 64.53 | 1.800 | 27.89 | 0.543, 0.526 | 64/64 |

At 8K the drafter's KV is 64 MiB and its prompt pass costs 1.2 s. Acceptance
falls to 0.54 (the drafter is weaker on long-document continuation), so d1 is
only 1.05x plain and d2 is below plain.

## Where the verify's time goes

`column-bench` times `forward_columns` on its own and `--trace` folds the
per-stage GPU time (prose prompt, 373 tokens, 2 columns):

| stage | ms/pass | note |
|---|---|---|
| gate / up / post_norm | 7.1 / 7.3 / 6.4 | batched GEMV, 500-508 GB/s |
| ssm_in / ssm_rec | 6.3 / 3.5 | GDN projections, 421 / 379 GB/s |
| out_norm | 1.7 | head, 613 GB/s |
| attn_in | 4.4 | q/k/v/qg GEMV + per-column split/rope |
| **attn_mix** | **8.5** | f32 causal attention, per column |
| attn_append | 1.9 | KV append, per column |
| ssm_proj | 2.7 | GDN gates/conv/norm/recurrence, per column |

The projections are the easy part: the batched GEMV costs **1.06-1.15x** one
single GEMV at B=2 (`gemv-bench`), i.e. the second column is nearly free on
weights. The pass costs 1.21x a one-column pass, and the extra is per-column
sequential work:

* f32 causal attention is ~**0.5 us per cached position per column** (3.7 ms
  per pass at 374 positions for 2 columns, ~10% of a plain token too). Its grid
  is `n_kv x splits` with `splits = n_ctx / 128`, so at this context only 12
  blocks run; the same kernel is ~40x faster per position on the quantized-KV
  path, whose split count is derived from `seq`.
* each `set_step` is a 32-byte async upload that costs **37 us** on this driver;
  the per-column loops issue 6 per layer, i.e. ~3.5 ms per 2-column pass.
* the GDN per-token chain (gates, conv, two l2norms, recurrence, gated norm)
  runs once per column: 1.3 ms for the second column.

So the ceiling here is not the batched GEMV but the per-column kernels. The
2-column pass streams its weights in 32.3 ms; if the per-column work went to
~1 ms the pass would be ~33 ms and 1.85 tokens/step would be ~56 tok/s.

The drafter's own cost is 2.7 ms per step for 253.6 MiB (94 GB/s): its block is
small and launch-bound (one block plus the Q6_K head), not bandwidth-bound.

## What is not verified

* The 48 tok/s (prose) / 53 tok/s (JSON) targets are **not met**: 34.8 / 35.3
  at one draft, best case 49.0 (verbatim copy, 3 drafts).
* The 8K path is measured with `--kv q8` only; `--kv f32` at 8K was not run.
* Acceptance is measured on four fixtures, 128 tokens each; no longer-horizon
  drift study, and no study of the drafter on prompts unlike the fixtures.
* Rollback is exercised by every rejected draft, but the state snapshots are
  not compared against a replayed prefix; exactness of the emitted ids over
  128 tokens is the evidence.
* No graph capture: the per-column loops issue ~1300 launches per column, and
  that launch cost is inside every number above.
