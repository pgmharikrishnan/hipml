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

**The trace prints both routes, so read the route word.** A later serve run with
`HIPML_BATCH_TRACE=1` and `--drafts 3` prints `b=4` lines and was read as "the
verify is on the approximate tile, exactly at the threshold". It was not: the
lines name `gemv_q5_k_b`, `gemv_q6_k_b`, `gemv_iq4_xs_b` and so on — the exact
`_b` entries — and the verify prints no `[skinny]` line and no `gemm_skinny_*`
entry at any draft count. The batch engine in the same process does take the
tile at four or more slots (`gemm_skinny_*`, `route=skinny`); both shapes are
real and both are in
[`serve-ud-batch.md`](serve-ud-batch.md) §1. The route is now printed on the
line (`route=exact` / `route=skinny`) so that reading a bare `b=4` cannot
repeat the mistake:

```
[batch] gemv_q6_k_b m=17408  k=5120 b=4 route=exact
[batch] gemv_q6_k_b m=248320 k=5120 b=4 route=exact
```

**The gate that would have caught a real regression.** `spec-bench` passed
128/128 through the review-4 defect because it never set the server's flag. It
now installs `serve`'s dispatch (`HIPML_BATCH_SKINNY=1` unless
`HIPML_BATCH_LEGACY=1`) and exits 1 if any drafts row is not token-exact, so the
CLI gate and the server run the same decision. `bench/serve/probe_exact.py` is
the end-to-end check: `serve` plain against `serve` at drafts 1, 2 and 3, the
same 16 conversations at ctx 32,768, ids compared one for one.

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

## The four-row batched GEMV: measured, and why the row-scaling levers fail

> R9700 (`gfx1201`), `Qwen3.8-27B-UD-Q4_K_M.gguf`, GPU lock held, best of two
> passes, one launch per tensor. Base tree `eb08bc9`.

`gemv-entries --batch N` streams one whole resident format through
`Launcher::gemv_batched` at `N` token rows — the call `forward_columns` makes —
and checks every row bit-for-bit against the one-row entry. The weight bytes are
identical at every `N`, so GB/s against the one-row number is exactly the
row-scaling cost of the batched GEMV. `b=1` is the dispatch's own choice: it
routes to the single-sequence entry (`gemv_q4_k_r1`, `gemv_q6_k_r1`), which is
what a one-column pass actually runs.

| format | GB | b=1 | b=2 | b=3 | b=4 | b=8 | b=16 | `w4` read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 2.427 | 578.7 | 515.7 | 448.0 | 403.3 | 240.9 | 121.9 | 625.5 |
| Q5_K | 5.230 | 569.8 | 567.6 | 477.4 | 431.2 | 295.4 | 153.4 | 593.1 |
| Q6_K | 1.804 | 600.7 | 595.3 | 586.3 | 525.5 | 382.0 | 185.4 | 609.3 |
| IQ4_XS | 5.696 | 603.9 | 548.0 | 421.9 | 388.3 | 158.5 | 85.0 | 615.0 |
| IQ4_NL | 0.431 | 558.9 | 458.1 | 428.0 | 370.2 | 174.2 | 87.6 | 591.2 |
| Q3_K | 0.077 | 310.5 | 252.0 | 234.5 | 223.7 | 155.6 | 81.9 | 588.8 |
| IQ3_S | 0.077 | 326.4 | 278.4 | 251.0 | 229.2 | 144.7 | 72.2 | 587.4 |
| **all** | **15.74** | **581.6** | 545.4 | 455.0 | **413.1** | 218.8 | 114.3 | **608.1** |

The `w4` column is the same bytes read by the GEMV's own access pattern with no
arithmetic, from `stream-shape` in the same session (608.1 GB/s over 15.80 GB);
the per-type numbers are [`decode-roofline.md`](decode-roofline.md)'s. `b=16` is
two 8-row groups — two weight passes — so its rate is not comparable; `b<=8` is
one pass.

**At four rows the kernel is at 63–86% of the read ceiling, so it is not
bandwidth-bound** and the residue is in the batched kernel, not the weight
bytes. The cost is not occupancy either: the register footprint, read out of the
compiled ISA (`.amdhsa_next_free_vgpr`, one kernel per `(format, B, rows/warp)`
so the numbers are per instantiation rather than the union the shipped entry's
`switch` produces), is the same order at 1 and 4 rows.

| B | q4_k r2 | q5_k r2 | q6_k r2 | iq4_xs r2 | q4_k r1 | q4_k r4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 67 | 63 | 68 | 92 | 38 | 112 |
| 2 | 83 | 81 | 78 | 88 | 55 | 119 |
| 3 | 80 | 71 | 88 | 108 | 74 | 118 |
| 4 | 76 | 73 | 81 | 100 | 93 | 146 |
| 8 | 84 | 81 | 99 | 86 | 75 | 138 |

Shipped entries (the union over the `switch (batch)`): `gemv_q4_k_b` 83,
`gemv_q5_k_b` 82, `gemv_q6_k_b` 133, `gemv_iq4_xs_b` 127; the one-row entries
are 47/48/92. No kernel has a private segment, so nothing spills. At 4 rows
`gemv_iq4_xs_b` needs 100 VGPRs against the one-row kernel's 92 — 20 waves/CU
against 22 — while running 1.56x slower, and `gemv_q6_k_b` runs at 133 VGPRs
(15 waves) and only 1.14x slower than its one-row entry. Occupancy does not
order the numbers; the per-value instruction count does: the batched tile issues
`B` FMAs per weight value where the one-row kernel issues one, plus `B/RPW` x
reads, and the formats whose dequant is cheapest per value (Q6_K) lose least.

What was tried against that, and measured at `b=2/3/4` on the same probe:

| lever | b=4 result | verdict |
| --- | --- | --- |
| rows/warp 4 (half the blocks, half the x reads per weight value) | Q4_K 403→350, Q5_K 431→415, IQ4_XS 388→361; b=2 also −22% on Q4_K | delete: the 146-VGPR tile costs more than the reuse returns |
| rows/warp 1 (double the blocks, half the accumulators) | Q4_K 403→336, Q5_K 431→406, IQ4_XS 388→280 | delete: more x reads per weight value, same arithmetic |
| nibble select → four dword shifts in `q4k_load` / `iq4x_load_b` | Q4_K 403→389, IQ4_XS 388→352, b=2 IQ4_XS 548→394 | delete: the compiler's `cndmask` form is cheaper than a predicated 4-shift block |
| weight reads with `sc1` (evict-first) | not adopted | at b=1 the wide load splits into four dwords and loses 11% ([`decode-roofline.md`](decode-roofline.md)); the load path is per block and does not depend on `B`, so it was not carried to four rows |

Rows/warp 2 — what ships — is the optimum of the three tilings at every `B`.
The remaining lever is not a tuning constant: at `b=4` the kernel must issue four
FMAs per weight value where the one-row kernel issues one, and that arithmetic
is 4x the memory term's headroom only if the dequant chain is short. Q6_K, whose
dequant is the shortest, is at 86% of the ceiling; the four-bit formats are at
63–64%.

### On the real path

`column-bench --cols 4 --tokens 32` on the same model, two passes per tree, mean
of the two; `base` is `eb08bc9` and the branch adds only the probe, so the two
columns should agree and do:

| KV | tree | 1 col | 2 cols | 3 cols | 4 cols | 4 col / 1 col |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| q8 | base | 35.00 | 35.49 | 42.17 | 46.62 | 1.33 |
| q8 | branch | 34.79 | 35.57 | 42.40 | 46.81 | 1.35 |
| f32 | base | 32.19 | 35.56 | 42.45 | 47.37 | 1.47 |
| f32 | branch | 32.19 | 35.59 | 42.55 | 47.39 | 1.47 |

Milliseconds per pass. The four-row pass costs 1.33–1.35x the one-row pass at
q8 and 1.47x at f32, and the batched GEMV is the bulk of that excess rather than
a side effect: the measured GEMV set reads in 38.1 ms at `b=4` against 27.1 ms
at `b=1`, an excess of 11.0 ms, where the four-column pass adds 11.6 ms over the
one-column pass. If the batched GEMV cost the same per weight value at four rows
as at one, the pass would be 35.0 ms and — the four-column verify is 46.3 of a
55.1 ms drafts-3 step — that step about 43 ms, 79 tok/s against 62. No measured
lever came within reach of it, so no kernel change ships and the 1.20x target is
not met.

`spec-bench --drafts 3 --tokens 128` in the same window, 128/128 exact on both
KV modes and both trees: q8 62.01 → 61.75 tok/s, f32 61.31 → 61.15. The branch
reads 0.3–0.4% lower, inside the 0.8% spread between two passes of the one
binary, and the kernels are byte-identical. Serve cell a — one request at a
time, 32K context, 5 conversations, 15 requests — lands at 51.84 tok/s p50 with
0 errors and a 3.12 s TTFT, against 51.91 for the tree it is based on.

## The one-slot serve cell and the CLI, like for like

Cell a reads 52.5 tok/s at drafts 3 where `spec-bench` reads 63.7 on the same
box, model and KV. The two rows differ in harness, prompt, chunk and stream
length, so the first question is whether the 11 tok/s is the serve path at all.

It is not. Measured in one lock hold on `6795fce`, UD-Q4_K_M (16,564,259,744 B),
`mtp-Qwen3.8-27B-Q4_0.gguf`, q8 KV, greedy, drafts 3. The CLI rows are
`spec-bench --ctx 32768 --chunk 256 --kv q8 --tokens N` on the same prompt ids
the server tokenizes (`hipml chat --ids-out` on the cell's own messages), one
process per prompt.

| harness | prompt | ctx | stream | ms/step | tok/step | tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLI | p2_math_169 (169) | 313 | 128 | 53.85 | 3.432 | **63.74** |
| CLI | p2_math_169 | 249 | 64 | 54.13 | 3.000 | 55.42 |
| CLI | cell a u0 (2,425) | 32,768 | 64 | 56.40 | 2.864 | 50.77 |
| CLI | cell a u0 | 32,768 | 128 | 57.10 | 2.761 | 48.35 |
| CLI, 5 cell-a prompts | cell a | 32,768 | 64 | 56.89 | — | 52.76 (median) |
| serve cell a | cell a | 32,768 | 64 | 56.55 | — | **52.48** |
| serve cell a | cell a | 32,768 | 128 | — | — | 48.16 |

`tok/step` is not quoted for the server rows: `spec-bench` counts the tokens it
streams, while the server's counter includes the tokens a step produced past
`max_tokens`, which the client never sees. Its `ms/step` and the client's tok/s
are both exact. The 64-token `p2_math` row and the 128-token cell-a row were
taken on `6795fce` before the measurement hook below existed; the rest on the
final build, and the two agree — the 5-prompt median is 52.76 against 52.65.

Both directions move the wrong way for the "serve is slow" reading. The CLI at
a 64-token stream drops to 55.42 on `p2_math` (from 63.74 at 128) and to 50.77
on the cell-a prompt; serve at a 128-token stream drops to 48.16. **ms/step is
what the engine charges and it does not move with stream length**: 53.85 against
54.13 on `p2_math`, 56.40 against 57.10 on the cell-a prompt, 56.55 in the
server. Only `tok/step` moves, and only because a shorter reply samples a
different stretch of acceptance.

### The 15.7 against 19.1 ms per token

Per token, at drafts 3: `p2_math` 53.85 / 3.432 = **15.69 ms**; cell a's own
prompt 56.40 / 2.864 = **19.69 ms** in the CLI and 56.55 / 3.03 = 18.7 in the
server. Splitting the CLI's 4.00 ms on the same harness (so the difference can
only be workload):

| item | ms/token | evidence |
| --- | ---: | --- |
| 32K cache instead of 313 | 0.74 | 56.40 − 53.85 = 2.55 ms/step |
| acceptance on this prompt | 3.11 | 53.85 × (1/2.864 − 1/3.432) |
| interaction | 0.15 | remainder |
| **serve path** | **0.00** | 56.55 against 56.89 ms/step, −0.1% |

The cell-a prompt accepts 0.818 / 0.667 / 0.917 against `p2_math`'s
0.946 / 0.829 / 0.931; at a fixed step cost that alone is 3.1 ms per token.
The 32K cache costs 2.6 ms per step of a four-column verify and 0.5 ms of a
plain one, which is the split-geometry fix in
[`serve-ud-batch.md`](serve-ud-batch.md) §7 still holding. Nothing is left for
the server: **52.48 against 52.76 for the same prompt, context, stream length
and draft count is 0.5%**, and `spec-bench`'s 63.7 was never a number the serve
cell could be read against.

### Inside the serve step

`HIPML_SERVE_SPEC_LOG=1` (new, off unless set, the spec counterpart of
`HIPML_SERVE_STEP_LOG`) prints the step-to-step wall period and the step's own
draft/verify/rollback buckets. 104 steps over the five requests, 315 streamed
tokens, one slot, ctx 32,768:

| bucket | ms/step | share |
| --- | ---: | ---: |
| draft (3 MTP blocks) | 10.34 | 18.3% |
| verify (4 columns) | 45.60 | 80.6% |
| rollback + drafter KV rewrite | 0.61 | 1.1% |
| scheduler, token channel, HTTP/SSE, detokenise | 0.00–0.10 | ≤0.2% |
| **step-to-step wall** | **56.55** | |

At 3.03 streamed tokens per step that is 18.7 ms per token, against the client's
19.05 — the difference is the first token, which prefill produces. The hook is
inert when unset (`spec_log` is false, so the step takes the same branch it did
before); with it on the cell reads 53.07 tok/s against 52.48 off, 1.1% and the
same size as the cell's own rep-to-rep variation (52.84 / 52.48 / 52.35).

The gap between consecutive step starts is the step's own buckets to within
0.1 ms on every request, so the scheduler and the streaming write cost nothing
measurable. That is the same verdict §7.1 reached for the plain batched step
(~0.2 ms) and it holds for the speculative one.

### The batched verify against four single-stream launches

The remaining 45.6 ms of verify is the weight pass plus ~7.6 ms of attention,
GDN chain, norms, head and argmax. `gemv-entries --batch N --reps 20`, one
launch per tensor over every resident tensor of the format, GPU spans:

| format | GB | b=1 | b=2 | b=3 | b=4 | 4 × b=1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 2.427 | 4.121 | 4.548 | 5.215 | 5.871 | 16.48 |
| Q5_K | 5.230 | 9.081 | 9.028 | 11.000 | 12.118 | 36.32 |
| Q6_K | 1.804 | 2.920 | 2.962 | 3.087 | 3.380 | 11.68 |
| IQ4_XS | 5.696 | 9.310 | 10.351 | 13.479 | 14.649 | 37.24 |
| IQ4_NL | 0.431 | 0.757 | 0.909 | 0.971 | 1.124 | 3.03 |
| **sum** | **15.59** | **26.19** | | | **37.14** | **104.75** |

Milliseconds. One b=4 launch costs **37.1 ms against 104.8 ms for four
single-stream launches** — 2.8x cheaper, 9.3 against 26.2 ms per token. The
b=4 route is already the fast one; sending the verify through the single-stream
entries would add ~68 ms to a 56.6 ms step. Every row also reports `0/N differ
from gemv_*_r1` at b=2, 3 and 4, so the batched entries the verify holds are
bit-identical to the plain step's.

So the hypothesis that a four-column verify pays a per-step penalty "that has
nothing to do with drafting" is not supported: the step is 81% verify, the
verify is 81% weight pass, and the batched weight pass is the cheapest route
available at four rows. What is left against llama.cpp's 62.33 on the same cell
is the row scaling of that pass — 37.1 ms at b=4 where the same 15.6 GB reads in
26.2 ms at b=1, the 10.9 ms excess this note's four-row section already probed
with five levers and no winner. It is kernel arithmetic, not a dispatch choice,
and it is out of this slice.

### Measured, on the shipped tree

One lock hold, 3 reps per cell, 64 tokens per request, no drafter for b and c;
"before" is the line this note's table above records.

| cell | slots | ctx/slot | before | this run | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| a, plain | 1 | 32,768 | 31.64 | **31.68** | 20,091 MiB |
| a, drafts 3 | 1 | 32,768 | 52.32 | **52.48** | 20,892 MiB |
| b | 8 | 16,384 | 132.37 | **132.43** | 24,110 MiB |
| c | 16 | 8,192 | 222.13 | **223.03** | 25,201 MiB |

Zero request errors; rep spread 0.19 (b), 0.66 (c) and 0.29 (a plain) tok/s.
The diff adds one measurement hook and no engine change, which is why the cells
sit on their previous numbers.
