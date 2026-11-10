# Review: what is left between hipml and llama.cpp on one R9700

> *Engineering log from development; internal work-package IDs and paths have been sanitized or removed where they were machine-specific. Current numbers: [docs/benchmarks.md](../benchmarks.md).*

Scope: Qwen3.8-27B Q4_K_M, single stream, Radeon AI PRO R9700 (gfx1201, 64 CUs,
32 GB, 640 GB/s spec, 633.6 GB/s measured GEMV, 171-188 TFLOPS measured WMMA).
Every number below is measured on the box unless it is marked *model*
(arithmetic from measured parts) or *unmeasured*.

## 1. Where the time goes

Head-to-head, both engines token-exact:

| Depth | Prefill tok/s hipml / llama | TTFT hipml / llama | Decode tok/s hipml / llama |
|---|---|---|---|
| 0 | 940 / 1025 | 715 ms / 675 ms | 29.8 / 30.6 |
| 8K | 995 / 1185 | 8.23 s / 6.93 s | 28.7 / 29.8 |
| 32K | 864 / 1069 | 37.9 s / 30.7 s | 26.2 / 27.7 |
| 100K | 637 / 828 | 156.9 s / 120.9 s | 21.2 / 22.1 |

### Decode ledger (33.5 ms / token at depth 0)

| Item | ms | Share | Floor |
|---|---:|---:|---|
| Weight stream, 17.19 GB at 632 GB/s | 27.2 | 81% | 26.9 at 640 GB/s |
| Per-kernel ramp (fill and drain of each launch) | 5.6 | 17% | about 1 |
| Host submit, mostly overlapped with the GPU | about 2, 0.7 exposed | 2% | 0 |
| Depth: q8 KV read + attention (measured +4.7 ms at 32K, +13.7 ms at 100K) | depth | | 1.1 / 3.48 GB at 632 GB/s = 1.8 / 5.5 ms |

The weight stream is already at 99% of measured bandwidth. Only the ramp is
left, and HIP graphs, a persistent job-list kernel and multi-stream prefetch
were all tried and refuted. **Non-speculative decode ceiling: 17.19 GB / 640
GB/s + 1.4 ms of unavoidable tail = 28.3 ms → 35.3 tok/s**; a realistic one
with half the ramp removed is 31.5 tok/s, 3% over llama. Decode cannot beat
llama by a margin that matters without producing more than one token per
weight pass.

### Prefill ledger (`gpu-bench --prefill N --prefill-split`, before this review)

| Stage | 512 tokens | 8K tokens | Share at 8K | What bounds it |
|---|---:|---:|---:|---|
| FFN GEMMs (gate, up, down × 64) | 250.4 ms | 4098 ms | 49% | WMMA at 62-64 TFLOPS, 35-43% of 175 |
| GDN projection GEMMs | 56.7 | 926 | 11% | same kernel |
| Attention projection GEMMs | 20.7 | 339 | 4% | same kernel |
| GDN chunk recurrence (48 layers) | 120.2 | 1970 | 23% | **48 blocks on 64 CUs, a dependent global load per token** |
| Attention (16 layers, `attn_chunk_wmma`) | 28.1 | 832 | 10% | 16.9 TFLOPS at 8K; K/V re-read per query tile |
| Elementwise, embed, final | 17.6 | 271 | 3% | bandwidth |
| Wall (split adds syncs) | 0.51 s | 8.45 s | | 8.22 s without `--prefill-split` |

At 100K (*model* from the 8K split, scaled per token for GEMM and GDN and
quadratically for attention): GEMM 63.9 s, GDN 23.3 s, attention 65.6 s,
elementwise 2.4 s, against 156.9 s measured. The attention term is memory:
the 3.48 GB q8 KV is re-read 2,345 times at an effective 137 GB/s. Its FLOPs
(1.97 PFLOP causal over 16 layers) would take 11 s at 175 TFLOPS.

### Ceilings per axis

| Axis | Ceiling | Why |
|---|---|---|
| Decode, 1 token per weight pass | 35.3 tok/s hard, 31.5 realistic | 17.19 GB / 640 GB/s + 1.4 ms tail; ramp halved |
| Decode at 100K | 25.3 tok/s (measured 21.2, llama 22.1) | + 3.48 GB q8 KV at 632 GB/s; today's q8 decode attention reads it at about 250 GB/s |
| Decode with an MTP drafter, 1 draft | 48-54 tok/s at the measured a1 = 0.80-1.00 | (1 + a1) tokens per 37.1 ms step, §3 |
| Prefill at depth 0 | 1,985 tok/s with GEMM at 175 TFLOPS; 2,720 with GDN 4x on top | 512-token stages above; GEMM is 53.8 GFLOP/token |
| Prefill TTFT at 8K | 2.6 s (all at ceiling), 6.0 s realistic | ceiling: GEMM and attention at 175, GDN 4x; realistic: GEMM at 100 TFLOPS, rest as today |
| Prefill TTFT at 100K | 41 s (all at ceiling), about 80 s realistic | GEMM 120 TFLOPS, attention 80 TFLOPS compute-bound, GDN as today |
| Aggregate decode, B sequences | B / (27.2 + 5.6 + 0.5 B + KV_B) ms | weights shared, 157 MB of SSM state per sequence not (§4) |

## 2. What the 27 work packages missed

1. **The GDN recurrence is a quarter of prefill and was never retuned.**
   `gdn_chunk_f32` runs one 1024-thread block per value head: 48 blocks on 64
   CUs, so 16 CUs idle for 24% of every prefill chunk, and every token's q, k,
   v, g and beta is a dependent global load inside the sequential token loop.
   The rows of a head's 128×128 state are independent: row j needs only v[j]
   and the head's shared q, k, decay and beta. Splitting a head across four
   blocks of 32 rows (192 blocks) and staging operands one 16-token tile ahead
   in shared memory changes nothing in the arithmetic. This is the lever
   prototyped here (§5).
2. **The quantized batch GEMV is the wall for both speculation and batching.**
   `gemm_batch_q` (feat/spec-decode) and the feat/batch GEMV both multiply in
   f32 from LDS, so cost grows with columns: Q4_K [17408, 5120] costs 0.98x at
   2 columns, 1.27x at 3, 1.90x at 4 and 2.95x at 8. A WMMA skinny GEMM (pad
   R ≤ 16 columns to one 16-wide tile; each lane's A fragment is 8 consecutive
   k values, which is exactly one Q4_K sub-block row run) needs about 2 bytes
   of LDS per weight regardless of R and should hold 1.0-1.1x to 16 columns.
3. **An MTP drafter exists on the box and nobody loaded it.**
   the MTP drafter GGUF (`mtp-Qwen3.8-27B-Q4_0.gguf`) is a qwen35-family nextn block
   (blk.64: eh_proj 10240→5120, one full-attention block, one FFN, Q6_K/Q4_K)
   plus a Q3_K head. It is trained for a sibling base, not for this model's SFT,
   so
   its acceptance had to be measured (§3).
4. **Chunk size was only ever measured at 128/256/512.** At 512 tokens the
   gate/up GEMM launches 80 output tiles on 64 CUs (1.25 waves), so a quarter
   of the GEMM time runs at 25% occupancy. `--chunk 1024` gives 2.5 waves.
   Measured here: 8K TTFT 7.82 s at 512, 7.45 s at 1024, 7.40 s at 2048
   (with the §5 kernel), same last id; now the default (17fce0e).

## 3. Hypothesis 1: self-speculative decoding

### Early exit (first K layers + lm_head as drafter): not worth building

A K-layer drafter reads K/64 of the weights plus the 1.04 GB lm_head
(6% of a pass) for every draft, and on this hybrid the drafted tokens still
have to advance the GDN state of all 64 layers in the verify pass, so nothing
is skipped there. Break-even for one draft with a 1.02x verify is
`a > K/64 + 0.08`: a > 0.33 at K = 16, and 1.3x needs a > 0.73 from the
hidden state a quarter of the way through a model that was never trained to
exit early. The MTP block below costs 0.27 GB instead of 4.3 GB and was
trained for exactly this job, so early exit is dominated. *Model, unmeasured*;
no prototype.

### MTP drafter: measured acceptance

`hipml mtp-accept` (CPU oracle, `crates/hipml-cli/src/mtp_cmd.rs`) runs
greedy decode and, at every position, drafts 3 tokens recursively with the
sidecar block, under both hidden-state conventions (pre/post output_norm).

48 generated tokens per prompt (45 scored windows), greedy on the CPU
oracle, bench fixture prompts:

| Prompt | Hidden fed | P(draft 1) | P(2 \| 1) | P(3 \| 1,2) | Tokens/step, 1 / 2 / 3 drafts |
|---|---|---:|---:|---:|---|
| prose_explanation (373) | pre-norm | 0.778 | 0.771 | 0.630 | 1.78 / 2.38 / 2.76 |
| prose_explanation | **post-norm** | **0.800** | 0.722 | 0.577 | **1.80** / 2.38 / 2.71 |
| json_tool_call (300) | pre-norm | 0.956 | 0.884 | 0.789 | 1.96 / 2.80 / 3.47 |
| json_tool_call | **post-norm** | **1.000** | 0.956 | 0.860 | **2.00** / 2.96 / 3.78 |
| copy_200_words (285) | pre-norm | 0.978 | 0.977 | 0.930 | 1.98 / 2.93 / 3.82 |
| copy_200_words | **post-norm** | **1.000** | 1.000 | 1.000 | **2.00** / 3.00 / 4.00 |

Feeding the hidden state after `output_norm` (and `shared_head_norm` when
recursing) is the right convention. The sidecar was trained for the base
model, not for this model's SFT, and still agrees with its greedy choice 80% of
the time on free prose and essentially always on structured output. Caveats:
45 windows per prompt, three prompts, CPU f32 reference rather than the GPU
quantized path (both read the same Q4_K weights), and copy_200_words is an
upper bound by construction.

Step cost with one draft (*model*): verify 33.5 ms (2 columns at 0.98x Q4_K,
1.01x Q6_K, measured on feat/spec-decode) + 1.0 ms of per-column GDN,
attention and elementwise work + 2.6 ms draft (0.27 GB block + 1.04 GB head at
632 GB/s, plus ramp) = 37.1 ms for 1 + a1 tokens:

| a1 | tok/s | vs llama 30.6 |
|---:|---:|---:|
| 0.80 (prose) | 48.5 | 1.59x |
| 1.00 (json, copy) | 53.9 | 1.76x |
| 0.107 | 29.8 | break-even |

Two drafts on today's 3-column kernel (1.27x) give 49 tok/s on prose and 61 on
JSON; three drafts only pay with a flat-cost verify (lever 5).

A 32K-row truncated head would cut the draft from 7.7% of a pass to 2.4%.

### Exactness

Greedy speculation is token-exact only if each verify column produces the same
logits as a 1-column decode would. feat/spec-decode got only 5-10 of 128
tokens identical because its batch GEMV reduces in a different order. The
feat/batch GEMVs keep the single-sequence `k` order per row and are
bit-identical to a B = 1 launch, so a 2-column verify built from them, with
the GDN recurrence and attention run per column by the decode kernels, is
exact by construction. Rollback on a rejected draft restores the GDN state
written after column 1 (157 MB, about 0.25 ms) and truncates the KV by one.
A WMMA verify (lever 5) loses that property unless decode moves to the same
kernel.

## 4. Hypothesis 2: multi-sequence batching

Per decode step with B sequences (*model*): weights once (27.2 ms), ramp once
(5.6 ms), and per sequence the GDN state (48 layers × 48 heads × 128 × 128 ×
4 B = 151 MB read + written = 0.30 GB, about 0.48 ms) plus its KV.

| B | f32-LDS GEMV (today) | WMMA skinny (*model*) |
|---|---|---|
| 1 | 29.8 tok/s | 29.8 |
| 2 | 2 / (0.98 × 27.2 + 5.6 + 1.0) = 59 | 59 |
| 4 | 4 / (1.90 × 27.2 + 5.6 + 1.9) = 68 | 4 / (27.7 + 7.5) = 114 |
| 8 | 8 / (2.95 × 27.2 + 5.6 + 3.8) = 89 | 8 / (28.3 + 9.4) = 212 |
| 16 | memory: 16 × 157 MB state + KV, fits at 8K | 16 / (29.5 + 13.3) = 374 |

**The "WMMA skinny" column is a projection from §4's arithmetic; it was built
and measured in §9 and the B = 4/8 entries are 65 and 102 tok/s, not 114/212.**

Where it breaks: at B ≥ 4 on today's kernel (cost per column grows), at about
B = 30 on VRAM at 8K context (state 157 MB + KV 0.54 GB per sequence), and
at about B = 87 on compute. llama `-np B` was **not measured** (running
llama-server is forbidden in this review); llama's MMVQ also reads quantized
weights once per column group, so it shows the same knee near 4-8.

## 5. Prototype: row-split GDN recurrence

Chosen because it is the largest lever that (a) nobody had touched, (b) can
be bit-identical, so the token check is a proof and not a sample, and (c)
fits in a 20-minute GPU hold. The bigger levers (MTP, GEMM pipeline,
deep-prefill tile) are multi-week or already in flight on another branch.

`crates/hipml-kernels/kernels/gdn_rows_f32.hip` (c5ed8c7): grid
`(n_v, 128/32)` = 192 blocks of 256 threads, 8 waves × 4 state rows; each
lane keeps 4 elements of each of its 4 rows in registers. A 16-token tile of
q, k (shared by the head), the block's 32 v values, g and beta is loaded into
registers one tile ahead and stored to the other half of a double-buffered
shared array, one barrier per tile. The per-row arithmetic is
`gdn_chunk_f32`'s, in the same order. Selected in `steps_chunk` when
head_k = head_v = 128; `HIPML_GDN_HEAD=1` restores the old kernel for A/B.

Measured on the R9700 (lock "review", about 10 minutes held):

| Check | Before | After | |
|---|---:|---:|---|
| `hipml gdn-parity`, output and state bits at 1, 77, 512 tokens | | 0 differ | PASS |
| One layer, 512 tokens | 1702 us | 1212 us | 1.40x |
| `seq ssm` stage, 512-token prompt | 120.2 ms | 94.2 ms | -22% |
| `seq ssm` stage, 8K prompt | 1970 ms | 1536 ms | -22% |
| Prefill, 512 tokens | 1009 tok/s | 1059 tok/s | +4.9% |
| TTFT 8K, chunk 512 | 8.22 s | 7.82 s | -4.9% |
| TTFT 8K, + chunk 2048 (17fce0e) | 8.22 s | **7.40 s** | -10.0%, 1107 tok/s |
| TTFT 32K, + chunk 2048 | 37.92 s | **34.35 s** | -9.4%, 954 tok/s |
| TTFT 100K, + chunk 2048 | 156.9 s | **144.7 s** | -7.8%, 691 tok/s |
| Last id, 8K / 32K / 100K | 292 / 868 / 100 | 292 / 868 / 100 | identical |
| Acceptance prompt, `generate --device`, 20 tokens | | | identical, below |
| Decode | 29.85 tok/s | 29.85 tok/s | untouched path |

Gap to llama after the prototype: 8K 7.40 vs 6.93 s (was 8.22), 32K 34.4 vs
30.7 s, 100K 144.7 vs 120.9 s. It closes 64% of the 8K gap and none of the
structural deep-prefill gap, which is attention.

Why only 1.40x: a row now costs two 32-lane shuffle reductions per token
(8 per wave per token), and those, not loads, are the new floor at 2.37 us per
token per layer. The next step is the chunkwise (WY/UT-transform) form of the
delta rule, which turns 64-token blocks into matrix products on WMMA; that
cannot be bit-identical to the sequential recurrence.


## 6. Hypotheses 3 and 4

**3. 64/128-query prefill tile, 2x-not-3x K traffic (feat/deep-prefill).**
This is the deep-prefill lever and the largest single prefill item at depth:
attention is about 64 s of the 144.7 s at 100K and is memory-bound at an
effective 137 GB/s. Going from 32- to 128-query tiles cuts KV re-reads 4x and
makes it compute-bound: about 25 s at 80 TFLOPS (*model*), saving about 40 s
and putting 100K at about 105 s against llama's 120.9 s. At 8K it is worth
only about 0.4 s (attention is 0.8 s). No numbers on that branch yet.

**4. The rest.**

| Idea | Verdict | Evidence |
|---|---|---|
| Chunk 1024/2048 to raise GEMM waves at 512 | **Done**, -0.42 s at 8K | measured above |
| Prefill GEMM pipeline (62-68 → 100+ TFLOPS) | **Done for Q4_K** (109 TFLOPS) and 95 for Q6_K: 8K 7.40 → **6.61 s** | 4.4 s of GEMM at 8K; the 256×128/16-accumulator tile is the register-file ceiling, and staging (38%, load latency) does not overlap without spilling (see benchmarks WP32) |
| Decode attention at depth (q8 split kernel to bandwidth) | +14% decode at 100K (21.2 → about 25), +6% at 32K | measured +13.7 ms at 100K against 5.5 ms of KV bytes |
| Fusing norm / gate / SiLU into GEMV prologue and epilogue | up to +8% decode (5.6 ms ramp → about 3) | ramp measured; fusion is not in the refuted list |
| fp16 instead of q8 KV for prefill | worse: doubles the bytes of a memory-bound loop | 65.5 KB vs 34.8 KB per token |
| Weight repacking for decode | none | GEMV already at 633.6 of 640 GB/s |
| Infinity Cache residency (64 MB) | only via the loop order in (3): one KV head of one layer is 4.5 MB of q8 at 8K and 54 MB at 100K, so walking all query tiles of one head before the next keeps it resident | 544 B per head per token |
| Prefill/decode overlap | zero for single-stream TTFT; only matters with several requests | one request has no decode during its prefill |


## 7. Ranked lever table

Single-stream gains are against the state after this review (8K TTFT 7.40 s,
decode 29.8 tok/s). "Measured" means on the box in this review; "model" means
arithmetic from measured parts.

| # | Lever | Expected gain | Effort | Risk | Evidence |
|---|---|---|---|---|---|
| 1 | **MTP speculative decode, 1 draft**, verified with feat/batch's 2-column GEMVs (bit-identical per column to the 1-column decode) | decode 29.8 → 48-54 tok/s (1.6-1.8x; llama 30.6) | high: GPU MTP block, 2-token verify through 64 layers, GDN state checkpoint after token 1, MTP KV | medium: acceptance on prompts unlike these; exact by construction only if every per-column op matches decode | acceptance measured (§3); 2-column Q4_K/Q6_K cost 0.98/1.01x measured on feat/spec-decode |
| 2 | Deep-prefill 64/128-query tile, KV-outer loop (feat/deep-prefill) | 100K TTFT 144.7 → about 105 s (llama 120.9); 32K -3 s | medium, in flight | low | attention about 64 s at 100K, memory-bound at 137 GB/s (model) |
| 3 | Prefill GEMM pipeline, 65 → 100+ TFLOPS | 8K TTFT 7.40 → about 6.0 s (llama 6.93); 100K -20 s | high | medium: new reduction order, needs the parity suite | 5.0 s of GEMM at 8K, 35-43% of the measured WMMA ceiling |
| 4 | q8 decode attention to bandwidth at depth | 100K decode 21.2 → about 25 tok/s (llama 22.1); 32K 26.2 → about 28 | low-medium | low | +13.7 ms per token at 100K measured against 5.5 ms of KV bytes |
| 5 | WMMA skinny GEMM, R ≤ 16 columns | **done**: cost curve R=4/8 within 1.05x of R=1 on all five shapes; batch B=8 73.2 → **102.0 tok/s** (llama 101.3), B=4 unchanged | done (`gemm_skinny_*`) | none: same ids as the B=1 path over 16 tokens | measured §9 |
| 6 | Chunkwise (WY) GDN recurrence on WMMA | 8K -1.1 s, 100K -13 s | high | medium: cannot be bit-identical | `seq ssm` 1.54 s at 8K after §5, shuffle-bound |
| 7 | Norm / gate / SiLU fused into GEMV prologue and epilogue | decode +5-8% | medium | low | 5.6 ms of ramp measured; not in the refuted list |
| 8 | Row-split GDN recurrence | **done**: 8K -0.40 s, 1.40x per layer | done (c5ed8c7) | none: bit-identical | measured §5 |
| 9 | Prefill chunk 2048 | **done**: 8K -0.42 s, 32K and 100K included above | done (17fce0e) | none: same ids | measured §5 |
| - | Early-exit self-speculation | below break-even (needs a > 0.33 at K = 16) | | | model, §3; dominated by MTP |
| - | fp16 KV, decode weight repacking, prefill/decode overlap | none | | | §6 |


## 8. Verdict

hipml loses to llama.cpp by 3% in decode and 6-30% in TTFT, and the 27 work
packages left two first-order things on the table. In decode the weight
stream is at 99% of bandwidth, so kernel work can buy about 3% over llama
(15% at the hard ceiling); the only real win is more than one token per pass, and the box
already holds an MTP drafter that agrees with the model's greedy choice 80-100% of
the time, which with the existing bit-identical 2-column GEMV models to 48-54
tok/s, 1.6-1.8x llama. That is the project's highest-value lever. Early exit
is dominated by it, and batching is a throughput lever, not a latency one,
that hits the same f32-from-LDS kernel wall at B ≈ 4 until a WMMA skinny GEMM
exists. In prefill the GDN recurrence was a quarter of every chunk on 48 of 64
CUs. The row-split kernel (bit-identical) plus chunk 2048 (same ids) is
measured on the box: 8K TTFT 8.22 → 7.40 s, 32K 37.9 → 34.4 s, 100K 156.9 → 144.7 s,
closing 64% of the 8K gap. What remains is GEMM efficiency at shallow depth
(65 of 175 TFLOPS) and attention KV traffic at depth, which is the
feat/deep-prefill tile. With MTP decode, the deep-prefill tile and a 100-TFLOPS
GEMM, hipml beats llama on every row of the table; without MTP it ties
decode at best.

## 9. WP33: the WMMA skinny GEMM, measured

`gemm_skinny_{q4_k,q6_k,q8_0}` computes `y[r,:] = W @ x[r,:]` for `r <= 16` by
padding R to one 16-wide WMMA tile, so the weight stream is the cost and the
column count is nearly free. Parity: 8/8 cases against the CPU oracle
(`hipml parity skinny`).

### Cost curve (R9700, reps 20, best of 3, one resident weight)

| shape | MB | skinny r=1 | r=2 | r=4 | r=8 | r=16 | r=4 / r=1 | r=8 / r=1 |
|---|---|---|---|---|---|---|---|---|
| q4_k 17408x5120 ffn_gate | 50.1 | 397 | 411 | 401 | 389 | 383 | 0.99 | 1.02 |
| q6_k 17408x5120 ffn_up | 73.1 | 500 | 488 | 478 | 475 | 470 | 1.05 | 1.05 |
| q6_k 5120x17408 ffn_down | 73.1 | 388 | 408 | 419 | 408 | 401 | 0.93 | 0.95 |
| q4_k 5120x5120 o_proj | 14.7 | 329 | 330 | 323 | 320 | 315 | 1.02 | 1.03 |
| q8_0 8192x5120 ssm | 44.6 | 715 | 727 | 728 | 724 | 740 | 0.98 | 0.99 |

GB/s of weight, from the block format. Worst case **R=4 1.05x, R=8 1.05x**
against R=1 — the gate was 1.1x / 1.25x. The single-column GEMV on the same
bytes: 652 (ffn_gate), 596, 565, 605, 1000 GB/s, so the tile costs 53-73% of the
GEMV's bandwidth and buys a flat column curve in exchange.

### Batch integration (`hipml batch-bench --tokens 16`, q8 KV, depth 0)

All batched projections route through the tile at B >= 4 (`SKINNY_MIN_BATCH`);
B <= 3 keeps the f32-from-LDS `_b` kernels, because one 16-wide tile is mostly
padding there and that is a measured regression (B=2: 45.3 with `_b`, 37.2 with
the tile).

| B | `_b` kernels (WP26) | WMMA tile | llama -np | vs llama |
|---|---|---|---|---|
| 2 | 45.25 | 45.74 (`_b`) | 56.33 | 0.81x |
| 4 | 64.58 | 64.79 | 87.34 | 0.74x |
| 8 | 73.24 | **102.04** | 101.26 | **1.01x** |

B=8 is +39% over WP26 and level with llama, because the `_b` kernels fall to
154 GB/s at B=8 while the tile stays at ~400. B=4 is a wash: both paths sit at
273 GB/s, so 1.3x is still on the table there from the non-projection half of
the step (SSM recurrence, attention, argmax), not from this kernel.

Exactness: greedy streams identical to the B=1 path over 16 tokens for every
slot at B=4 and B=8 (`--check`). No logit divergence was observed; the kernel
level difference against the CPU oracle at production k is max_abs 1.59e-2
(q4_k 5120x5120) to 2.57e-2 (q6_k), which is the fp16 operand budget, plus the
k-split reduction order (4 accumulators folded at the end).

### What it cost to get there, and what is left

Three defects, each worth an order of magnitude:

1. `pf_scale_min` branches on the sub-block index, which is per-lane here, so
   every 8-value group became two exec-masked blocks with a `s_wait_loadcnt`
   between them: 20 GB/s. Replaced by a branch-free dword unpack.
2. The staging loads were serialised (one 8-byte load in flight per warp).
   `pws_prep`/`pws_finish` split the load from the arithmetic so a small array
   of raw words can be hoisted.
3. The bench summed `reps` launches and reported the total as one launch, so
   the whole cost curve read 10x low. Fixed in `run_skinny_resident_reps`.

Remaining split at the shipped geometry (probes, ffn_gate): compute 47%,
staging 46%, barriers 8%. The next lever is the compute half — RT=64 with a
128-value k tile would put all four warps on n-tiles instead of two, which is
the one geometry not measured (every 16-wide tile at RT <= 48 was 10-40% worse
for shared-footprint reasons). Note `KernelKind::RowTile::rows` must equal
`PWS_RT`: a mismatch silently halves the grid and looks like a 2x speedup.

### Call signature for WP30

```rust
// resident weight, x is [r, k] f32, y is [r, n] f32, r <= 16, k a multiple of 256
launcher.run_skinny_resident("gemm_skinny_q4_k", &w, &x, &y, n, k, r)?;
// or, for the graph: hipml_kernels::launch::skinny_resident(
//     &module, &stream, "gemm_skinny_q4_k", &w, &x, &y, n, k, r)
```
`hipml_kernels::skinny_entry("gemv_q4_k")` maps a resident GEMV entry to the
tile entry. Grid is `ceil(n / 32)` at 128 threads. For MTP verify pass r = 1 +
drafts: at r = 4 the verify costs 0.99-1.05x of one token.
