# Decode ablation: the 4.9 ms is three kernels the trace cannot see, not a slower weight stream

Box: R9700 (gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf`
(sha256 `1BCE504D…7DC529`, 16,564,259,744 B), q8 KV, chunk 2048, port 8093.
Every GPU run held the `decode-ablate` lock, acquired and released by
`scripts/probes/ablate-run.ps1` around each run; the ambient `HIPML_GGUF` on this
box points at the *stock* file, so every path is passed literally and the model's
sha256 is asserted before the lock is taken.

## 0. Question and answer

`origin/main` was `13e5b8d` when this was measured, and that is the base of the
A/B; the branch is rebased onto `871d4fb`, which changes only the unreachable
Q5_K fp16 prefill copy and adds the VRAM note, and §6's gates were re-run there.
Decode is 31.33 tok/s = 31.91 ms/token at depth 0 and 30.67 tok/s = 32.61 ms at
8,192. The routed GEMV term measured in isolation is
26.98 ms at 585.6 GB/s ([`decode-roofline.md`](decode-roofline.md) §2), and the
stage trace attributes 0.31 ms to attention, 0.13 to the GDN step and 0.14 to the
norms. That leaves **4.9 ms unaccounted for**, and the tracing tools cannot settle
it: `gpu-profile`'s per-kernel column is inflated by its own event pair, and
[`decode-residual-gemv.md`](decode-residual-gemv.md) §4 showed a stage-row
ablation that moved the traced token −0.94 ms and the untraced step in the same
process −0.05 ms.

So this slice does not trace. It **removes**: `hipml decode-ablate` drops one
stage's launches from the decode step and times the rest, untraced, over 200
tokens, with the cells interleaved across reps in one process. A dropped stage
issues no launch and records no event, so it cannot contribute to the number it is
used to price.

Answer, at depth 0, mean of three reps of 200 tokens:

| term | ms/token |
| --- | ---: |
| the token | **32.007** |
| the weight stream, every projection and nothing else (`gemv_only`) | **28.524** (542.9 GB/s) |
| the RMS norms (129 launches) | **1.48** |
| the GDN recurrence (48 launches) | **0.94** |
| the decode attention core (32 launches) | **0.83** |
| everything else, 370 of the 579 non-GEMV launches | 0.23 |
| **sum** | **32.01** |

Three things follow, and the first is the one the card did not expect.

1. **The in-situ weight stream is 28.52 ms, not 26.98.** A decode token streams
   15.485 GB of weights; the dedicated pass reads 15.802 GB, 0.35 GB of which is
   the MTP block a decode token never touches. Per byte of its own stream the
   token runs at **543 GB/s** against the pass's **586**, and at the pass's rate
   the token's bytes would take 26.44 ms — so the in-situ stream is **2.08 ms**
   above its own byte count at that rate, and 1.54 ms above the 26.98 the card
   assumed. It is not host launch spacing: every projection GEMV submitted as one
   captured `hipGraphLaunch` per token read 28.548 ms, within 0.1% of the eager
   path (§3).
2. **The non-GEMV work is 3.48 ms, not 0.58.** The stage trace sees the three
   kernels that matter — the norm, the GDN recurrence and the attention core — as
   0.58 ms together. Removing them one at a time says 1.05 + 0.64 + 0.70 = 2.39 ms
   of marginal cost, and removing all of it says 3.48 ms.
3. **The KV cache's footprint is not a factor.** The same work at depth 0 with the
   caches allocated for 208 positions and for 8,448 reads 32.084 and 32.092
   ms/token, a 0.008 ms difference (§4). At 8,192 the attention core costs 0.55 ms
   more than at depth 0, and that is its own work over 8,192 keys, not occupancy
   lost to the allocation.

What shipped: a bit-identical `rmsnorm_f32` that puts a thread's twenty loads in
flight before summing any of them. It halves the kernel (6.07 → 3.07 µs a launch)
and is worth **−0.66 ms at both depths**: 32.08 → 31.43 ms at depth 0 (31.18 →
31.82 tok/s) and 32.64 → 31.98 at 8,192 (30.64 → 31.27), interleaved, 3 reps × 2
processes (§5).

The target of ≤30.2 ms is not reached; 31.43 ms leaves 1.23 ms. Of the 3.0 ms that
is still not the weight stream, 0.43 is the norm, 0.94 the GDN recurrence, 0.83
the attention core and ~0.7 the rest; the two big ones are kernels the previous
two slices already took to their measured optimum, and the 0.95–1.69 ms that no
single-stage removal recovers is overlap, not a term that can be deleted.

## 1. The method

`crates/hipml-model/src/gpu/ablate.rs` defines seventeen droppable sets of
launches; `run_token` consults the set and skips the calls. Default is empty, so
the shipped decode is the same code path.

| cell | launches dropped |
| --- | --- |
| `embed` | the embedding row dequant |
| `attn_front` | q\|gate split, q/k head norm, M-RoPE |
| `kv_append` | the quantized KV row append |
| `attn_reduce` | the decode attention core (reads the KV cache) |
| `q_gate` | `attn_out *= silu(gate)` |
| `attn_gemv` | the four attention projections |
| `ssm_gates` | `softplus(alpha + dt) * a`, `sigmoid(beta)` |
| `ssm_conv` | the causal conv and its SiLU |
| `ssm_l2` | the two L2 norms |
| `ssm_rec` | the delta-net recurrence |
| `ssm_gn` | the gated RMS norm |
| `ssm_gemv` | the five GDN projections |
| `ffn_gemv` | `ffn_gate`, `ffn_up`, `ffn_down` |
| `norm` | the 129 RMS norms on the residual stream |
| `ew` | the 128 residual adds and `silu(gate)*up` |
| `lmhead` | the `output.weight` GEMV |
| `argmax` | the device argmax (the 4-byte read-back and its sync stay) |

Three kinds of cell carry the arithmetic:

- **`gemv_only`** drops every non-GEMV stage, so what is left is the weight
  stream and nothing else — the in-situ GEMV term, measured by the same clock as
  the full token.
- **`gemv+X`** adds exactly one stage back to `gemv_only`. The GEMV bytes are
  identical in every one of these, so the delta is that stage's cost with the
  weight stream running around it and nothing else competing.
- **`-X`** drops one stage from the full token, so the delta is its marginal cost
  with everything else present.

Every cell zeroes the decode scratch first (`reset_scratch`) and starts from a
cache filled with format-valid blocks. Both matter for the measurement, not just
for tidiness: a dropped producer leaves its consumer reading the same buffer, and
uninitialized memory can hold subnormals, which this card executes an order of
magnitude slower than zeros. The first version of the sweep did neither and took
38 minutes instead of 10.

## 2. Depth 0: the table

`hipml decode-ablate --gguf … --depth 0 --tokens 200 --reps 3 --kv q8`, cells
interleaved across reps in one process. Mean of the three reps; the within-cell
spread is 0.01–0.23 ms (0.01–0.19 at 8,192).

| cell | ms/token | vs `gemv_only` | `full − cell` |
| --- | ---: | ---: | ---: |
| `full` | **32.007** | +3.482 | — |
| `gemv_only` | **28.524** | — | −3.482 |
| `gemv+norm` | 30.001 | **+1.477** | |
| `gemv+attn_reduce` | 29.350 | **+0.826** | |
| `gemv+ssm_rec` | 29.459 | **+0.935** | |
| `gemv+ssm_conv` | 28.724 | +0.200 | |
| `gemv+ssm_l2` | 28.661 | +0.137 | |
| `gemv+ssm_gn` | 28.657 | +0.133 | |
| `gemv+ssm_gates` | 28.650 | +0.126 | |
| `gemv+attn_front` | 28.621 | +0.097 | |
| `gemv+q_gate` | 28.589 | +0.065 | |
| `gemv+kv_append` | 28.577 | +0.053 | |
| `gemv+attn` (all four) | 29.485 | +0.961 | |
| `gemv+ssm` (all five) | 29.777 | +1.253 | |
| `gemv+attn+ssm` | 30.709 | +2.185 | |
| `-norm` | 30.955 | | **+1.051** |
| `-attn_reduce` | 31.310 | | **+0.696** |
| `-ssm_rec` | 31.369 | | **+0.638** |
| `-ssm_l2` | 31.986 | | +0.021 |
| `-ssm_conv` | 31.987 | | +0.020 |
| `-attn_front` | 32.029 | | −0.022 |
| `-argmax` | 32.038 | | −0.031 |
| `-embed` | 32.039 | | −0.032 |
| `-ssm_gates` | 32.049 | | −0.042 |
| `-kv_append` | 32.080 | | −0.073 |
| `-q_gate` | 32.083 | | −0.076 |
| `-ssm_gn` | 32.160 | | −0.153 |
| `-ew` | 32.208 | | −0.201 |
| `-lmhead` | 30.322 | | +1.685 |
| `-attn` (whole block, weights too) | 29.075 | | +2.932 |
| `-ssm` (whole block, weights too) | 24.357 | | +7.650 |

`-lmhead` is in the table for scale, not as a stage: the `output.weight` GEMV is
1.04 GB of the weight stream, and removing it saves 1.685 ms, i.e. the stream runs
at ~617 GB/s on that one tensor. It belongs to `gemv_only`, which is why it is not
in the marginal sum below.

### The three kernels that matter

- **The norm, 1.48 ms for 129 launches — 11.4 µs each.** One block of 256 threads
  over a 5,120-float row: 20 KB in, 20 KB out, on one of 64 CUs, and the next
  launch on the stream cannot start until it finishes. It is the largest single
  item in the token after the weight stream. The standalone probe prices the same
  launch at 6.07 µs and removing it from the full token recovers 8.1 µs a launch,
  so this kernel is the cleanest statement of §4: adding it back to the isolated
  stream costs more than the kernel, removing it from the full token costs less.
- **The GDN recurrence, 0.94 ms for 48 launches — 19.6 µs each.** This is the
  `gdn_ar_f32` row-stripe the previous slice took from 32 to 64 CUs; it is
  already the fixed version.
- **The attention core, 0.83 ms for 32 launches — 26 µs each at depth 0.** Two
  launches a layer (the warp kernel and its reduce), reading one to three KV rows
  at this depth.

Everything else — the embedding, the front end, the KV append, `q_gate`, the four
GDN elementwise steps, 128 residual adds, `silu`, the argmax — is **0.23 ms for
370 launches**, and four of those cells measure *negative* (removing the launch
makes the token slower, by up to 0.20 ms for `-ew`). A launch in this stream is
not worth 5 µs; the 370 of them are worth 0.23 ms, and 129 norms are worth six
times as much.

## 3. The in-situ weight stream is 543 GB/s, not 585.6

`gemv_only` issues exactly the projection GEMVs `run_token` issues, in the same
order, at the same shapes, with one synchronize per token — the same measurement
the standalone `gemv-entries` probe makes, but inside the engine.

| measurement | GB streamed | ms | GB/s |
| --- | ---: | ---: | ---: |
| `gemv-entries`, one launch per tensor, best of three (`decode-roofline.md` §2) | 15.802 | 26.98 | 585.6 |
| `gemv_only`, eager, 200 tokens, mean of 3 reps | **15.485** | **28.524** | **542.9** |
| `gemv_only`, one captured `hipGraphLaunch` per token | 15.485 | **28.548** | **542.4** |
| `full` token | 15.485 | 32.007 | 483.8 (whole token) |

The byte counts differ, and that matters to the arithmetic. `gemv-entries` sums
every 2-D quantized tensor in the file above 1 MB, which includes `blk.64` — the
MTP block, whose 0.351 GB of projections a decode token does not stream
(`nextn_predict_layers` 1, so 64 decoder layers). `weight_bytes()` counts the 64
decoder layers plus `output.weight`, 14.421 GiB = 15.485 GB, and that is what
`gemv_only` actually reads. At the dedicated pass's 585.6 GB/s those bytes would
take 26.44 ms; the token takes 28.524, so **2.08 ms is the rate gap** and the
other 0.54 ms of the 1.54 is the pass reading bytes the token does not.

The graph cell is the control: if the rate gap were host-side launch spacing,
capturing the whole GEMV set into one submission would remove it. It does not
move the number at all (28.548 vs 28.524), and it makes the *full* token slightly
worse (32.486 vs 32.007), which is what [`decode-dispatch.md`](decode-dispatch.md)
§0 found for the whole-step graph. So 543 GB/s is what the decode's weight stream
runs at, and the 585.6 figure is a dedicated-pass best-of-three over a slightly
larger set of tensors.

**1.54 ms of the 4.9 is this.** It is not recoverable by anything in this slice: it
is the same launches on the same bytes, and every mechanism tried to raise them
([`decode-roofline.md`](decode-roofline.md) §2's pipeline, split-K, streaming-hint
and wider-load table) measured slower or flat.

## 4. Interleaved against isolated: the stream is not slowed, the stage is exposed

The brief's hypothesis was that the stages evict weight lines between GEMV
launches, so the stream should read slower when the stages are interleaved. The
`gemv+X` cells measure exactly that: identical GEMV bytes, one stage added back.

| stage added to `gemv_only` | depth 0 GB/s | 8,192 GB/s | ms added at 8,192 |
| --- | ---: | ---: | ---: |
| — (isolated) | **542.9** | **542.2** | — |
| `norm` | 516.1 | **516.1** | +1.440 |
| `attn_reduce` | 527.6 | **517.8** | +1.344 |
| `ssm_rec` | 525.6 | **525.5** | +0.907 |
| `ssm_conv` | 539.1 | 539.2 | +0.158 |
| `ssm_gn` | 540.3 | 540.0 | +0.111 |
| `ssm_l2` | 540.3 | 540.1 | +0.109 |
| `ssm_gates` | 540.5 | 540.4 | +0.094 |
| `attn_front` | 541.0 | 540.4 | +0.092 |
| `kv_append` | 541.9 | 541.0 | +0.060 |
| `q_gate` | 541.6 | 541.5 | +0.035 |
| all four attention | 525.2 | 515.7 | +1.465 |
| all five GDN | 520.0 | 519.8 | +1.229 |
| both blocks | 504.2 | 495.2 | +2.706 |

The rate does drop, and by a lot for three stages — but **the drop is the stage's
own duration, not a slowdown of the GEMV.** The columns are the same time divided
by the same bytes: a stage that costs 1.48 ms out of 28.5+1.48 makes the
bytes-per-second-of-wall-time fall, without the GEMV itself reading one byte
slower. Nothing here separates the two, which is why the arithmetic below is done
in milliseconds.

The additive test is the one that answers the hypothesis, because pollution is
super-additive by construction:

| depth | `full − gemv_only` (A) | Σ of the 13 marginal removals (B) | A − B |
| --- | ---: | ---: | ---: |
| 0 | 3.482 | 1.792 | **1.690** |
| 8,192 | 4.056 | 3.106 | **0.950** |

Read the other way: adding the three real stages to the isolated stream costs
1.477 + 0.935 + 0.826 = 3.238 ms at depth 0 and 1.440 + 0.907 + 1.344 = 3.691 at
8,192, against a whole non-GEMV set of 3.482 and 4.056. So the stages are
*additive with each other* (93% and 91%), and **sub-additive with respect to
removal**: B / A is 51% at depth 0 and 77% at 8,192, i.e. taking a stage out of
the full token recovers only half to three-quarters of what it cost to put in.

That asymmetry is the signature of overlap, not of eviction. In `gemv_only` a
small kernel sits between two 240 MB weight passes and is fully exposed; in the
full token the same kernel overlaps with the tail of the pass before it. It also
explains why every single-stage trace budget in this series has been wrong in both
directions: a stage's span is neither its cost to add nor its cost to remove.

**So the answer to "what is the missing 4.93 ms" is: 1.54 ms by which the
dedicated pass's 26.98 ms understates what the token's own 15.485 GB costs in
situ (2.08 ms of rate gap less 0.54 ms of bytes the token does not read), 3.24 ms
of three kernels the trace under-reports, and 0.15 ms of everything else.** There
is no pollution term to attack: the weight stream does not slow down when the
stages are interleaved, it just stops being the only thing on the card. The
residual A − B is the overlap a stage enjoys when the other stages are still there
— removing one alone cannot take it back, which is why every single-stage budget
in this series has under- or over-shot.

### The KV footprint, separately

Same work, same depth 0, only the cache allocation changes:

| `--ctx` | KV allocated | ms/token |
| --- | ---: | ---: |
| 208 | 0.007 GiB | 32.084 |
| 8,448 | 0.274 GiB | 32.092 |

0.008 ms apart — the allocation's footprint costs nothing. At depth 8,192 the
attention core's cost rises from 0.83 to 1.37 ms (+0.55) because it reads 8,192
keys instead of one to three, and that is the whole depth effect on the non-GEMV
side.

## 5. What was fixed

`rmsnorm_f32` — the only norm on the decode path, 129 launches a token, and the
largest non-GEMV item at 1.48 ms. The old kernel walked the row one strided load
at a time:

```c
for (int i = threadIdx.x; i < n; i += BLOCK) acc += xr[i] * xr[i];
```

With one block on one CU there is nothing else to hide the latency behind, so the
loop is a chain of twenty dependent loads. The new kernel keeps a thread's twenty
values in registers, issues all twenty loads, and *then* sums them in the same
strided order, so the result is bit-identical: same add order, same shared-memory
tree, same `v * scale * w`. The scale pass is elementwise, so it is widened to
`float4` (same expression per element) with a scalar fallback for a width that is
not a multiple of four, and the first pass keeps a strided tail for rows wider than
`MAXK * BLOCK` = 5,120.

`probe_norm.cpp` measures the shipped source on its own at 129 launches, chained
(each launch reads the row the previous one wrote, which is the decode's shape)
and independent:

| kernel | independent | chained | GB/s |
| --- | ---: | ---: | ---: |
| `13e5b8d` | 6.068 µs | 6.095 µs | 6.7 |
| this tree | **3.302 µs** | **3.072 µs** | **13.3** |

40 KB a launch at 6.7 GB/s is one block of 256 threads with nothing to hide the
latency behind; 13.3 GB/s is close to what one CU can do on a 20 KB row, so this
is the floor for a bit-identical single-block norm. Widening the block would
change the reduction order, and a two-block split would need a second launch.

In the token, at depth 0, three reps of 200 tokens, one process:

| cell | `13e5b8d` | this tree | delta |
| --- | ---: | ---: | ---: |
| `gemv+norm` over `gemv_only` | +1.477 | **+0.425** | −1.052 |
| `-norm` marginal | 1.051 | **0.562** | −0.489 |
| `full` | 32.007 | **31.516** | −0.491 |

The `decode-ablate` `full` cell moves −0.491 while the end-to-end harness below
moves −0.655. The loops are the same shape — two warm tokens, 200 greedy steps
through `forward_token_id` — but this command restarts each cell from
`state_at(depth)` with a zeroed scratch 30 times inside one ten-minute run, and
`gpu-bench` streams continuously with the two binaries alternating. That is
enough to move the level by 0.07 ms and the delta by 0.17 ms; both agree on the
sign and the size to within 0.2 ms, and the interleaved A/B below is the number to
quote.

End to end, 200 tokens, three reps, two processes, the two binaries alternating
inside one lock hold:

| depth | base `13e5b8d` | this tree | delta |
| --- | ---: | ---: | ---: |
| 0 ms/token | 32.039 / 32.063 / 32.079 / 32.101 / 32.102 / 32.100 | 31.386 / 31.414 / 31.437 / 31.435 / 31.437 / 31.446 | **−0.655** |
| 0 mean | **32.081** (31.17 tok/s) | **31.426** (31.82 tok/s) | −2.04% |
| 8,192 ms/token | 32.604 / 32.625 / 32.647 / 32.655 / 32.642 / 32.643 | 31.951 / 31.961 / 31.988 / 31.980 / 31.989 / 31.983 | **−0.661** |
| 8,192 mean | **32.636** (30.64 tok/s) | **31.975** (31.27 tok/s) | −2.03% |

All twelve paired deltas are negative and the same sign. The within-cell spread is
0.037–0.063 ms, ten times smaller than the delta, so the two cells are
distinguishable, not a tie. The norm's share of the e2e win (−0.66) is larger than
its marginal removal (−0.49) and smaller than its isolated cost (−1.05), which is
the same sub-additivity §4 measures for every stage.

Re-run on the rebased tree (§6), same script and the same base binary: depth 0
32.056 → 31.410 ms (31.20 → 31.84 tok/s, **−0.646**) and 8,192 32.616 → 31.959
(**−0.658**), with `full` 31.515, `gemv+norm` over `gemv_only` +0.425 and `-norm`
0.565 in the `decode-ablate` cells. The two sessions agree to 0.03 ms on the level
and 0.01 ms on the delta, and the norm probe reads 6.049/6.070 → 3.316/3.084 µs a
launch.

### What was tried and reverted

Nothing else was built. Two mechanisms were measured and rejected on the evidence
rather than implemented:

- **KV-cache layout / prefetch / reordering the layer so the weight stream is
  contiguous.** The premise was that the stages evict weight lines. §4 shows the
  stages are additive with each other to 93% and the stream's rate is unchanged
  when they are interleaved, so there is no eviction term to recover. The
  footprint test in §4 closes the occupancy variant of the same hypothesis.
- **Fusing the norm into its producer or consumer.** Already measured in
  [`decode-epilogue.md`](decode-epilogue.md) §5 (residual + norm: +0.19 ms) and
  [`decode-residual-gemv.md`](decode-residual-gemv.md) §3 (residual into the
  producing GEMV: −0.02 ms for 128 launches). This slice's table says why: a
  launch here is worth ~0.6 µs, so a fusion that removes one and adds work to a
  240 MB pass cannot pay.

The one thing this slice did revert was its own first measurement harness: the
cells did not zero the scratch and did not fill the cache, and the resulting
subnormal traffic made a sweep take 38 minutes instead of 10. That is a
correctness bug in the probe, and it is recorded here because a 4× wall-time
anomaly is exactly the kind of thing that would otherwise be read as a real
effect.

## 6. Gates, on the shipped tree

`scripts/probes/gates.ps1`, one lock hold, model sha256 asserted before the lock.

| gate | result |
| --- | --- |
| greedy 12 ids, UD, chunk 2048 | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]`, byte-identical |
| 200-token stream, depth 0, vs `13e5b8d` | **byte-identical**, 200/200, last id 108899 |
| 200-token stream, depth 8,192, vs `13e5b8d` | **byte-identical**, 200/200, last id 197 |
| `gemv-check`, 4 tensors × rows 1/2/4 | 12 checks, **0 differ**, max abs 0.0 |
| `column-bench --check` q8 cols 2, q8 cols 4, f32 cols 4 | 0 differ on 7,946,240 logits, hidden and residual |
| `spec-bench` drafts 0–3, q8 | **128/128 exact** on every row; 31.87 / 50.98 / 59.84 / 62.65 tok/s |
| `batch-bench --slots 16 --tokens 16 --check` | all 16 slots reproduce the B=1 path over the full stream |
| `oracle-gate` UD-Q4_K_M, 20 steps, chunk 2048, q8 | **PASS 20/20**, prefill max abs **0.1086** at 101748 |
| `oracle-gate` stock Q4_K_M, 20 steps, chunk 2048, q8 | **PASS 20/20**, prefill max abs **0.1481** at 34114 |
| `parity --deep` | 74 passed, 0 failed, 1 skipped |
| `cargo test --workspace` | 16 suites, 0 failed |
| `cargo fmt --all -- --check` | clean |

The ids are byte-identical because the norm change is bit-identical: the new
kernel sums `x[t]`, `x[t + BLOCK]`, … in the same order, uses the same
shared-memory tree, and multiplies `x * scale * w` in the same order. `probe_norm`
measures the same buffer both ways; the gate that matters is the 200-token stream,
and it does not move a single id at either depth.

The branch was rebased onto `871d4fb` (the Q5_K fp16 prefill copy, unreachable on
this file, and the VRAM note) after these gates were taken, so the whole set was
re-run on the rebased tree: same greedy ids, both 200-token streams still
byte-identical to `13e5b8d` with the same last ids, and the `decode-ablate` cells,
the A/B and the norm probe reproduce §2 and §5 inside the spread. The rebase
touches no decode path — `gpu_weights.rs` loses only the copy's dequantiser and
`launch.rs` only the copy's gate.
