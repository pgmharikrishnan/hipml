# Decode dispatch: the 5.3 ms that is not the weight stream is not launch cost

Box: R9700 (gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf`, q8 KV,
chunk 2048, port 8093. Every GPU run held the `decode-dispatch` lock; the
same binary pair alternated inside one lock window for every A/B below.
Resident 15.42 GiB, 14.42 GiB streamed per token, 0.272 GiB of KV at 8,192.

## 0. Question and answer

The decode token is 32.8 ms at depth 0 (30.5 tok/s) and 33.5 ms at 8,192
(29.9 tok/s). The GEMV term is 26.98 ms at 585.6 GB/s, 96.6% of a pure read of
the same bytes in the same row tile; the weight path is done. That leaves
**5.3 ms** to explain: ~1,300 kernel launches, and ~3.3 ms of time in the
kernels that are not GEMVs.

The question this slice took: is that 5.3 ms **submission-bound** (the host
cannot feed 1,300 launches fast enough, so a HIP graph of the whole step
would remove it), or is it **GPU-side** (each launch costs fill/drain on the
grid, and only fewer or bigger kernels can remove it)?

Answer: **GPU-side, and it is inside the kernels, not between them.** Capturing
the entire step — 1,299 launches, the KV append and the device argmax — into
one `hipGraphLaunch` does not make the token faster at either depth (33.20 vs
32.84 ms at 0, 33.46 vs 33.47 at 8,192). Deleting 128 of the 1,299 launches by
fusing the two residual+rmsnorm pairs per layer does not make it faster either
(32.82 → 33.01 ms, 0.6% slower, reverted). Both levers removed launches and
both left the token where it was, so the remaining 5.3 ms is kernel execution,
not dispatch.

**34.0 tok/s is not reachable from the dispatch side.** It needs 29.41 ms, i.e.
26.08 ms of weight stream (the `w4` pure-read floor) plus 3.33 ms of everything
else. The measured non-GEMV kernels are 3.3 ms and the inter-kernel component
is under 0.2 ms of that; there is no 3.4 ms to take here.

## 1. The launch count

Counted from the layer plan, not instrumented (`launches_per_token`, which both
the probe and `gpu-profile` read, so it cannot drift from the code):

```
1 embed
+ 64 × [1 attn_norm + (9 attention + 4 GEMV | 7 GDN + 5 GEMV) + 7 FFN]
+ 1 output rmsnorm + 1 lm_head
= 1,299 launches per token
```

The 64 layers are 16 full-attention and 48 gated-delta-net. By category:

| category | launches | per layer | what |
| --- | ---: | --- | --- |
| GEMV projections | **497** | 3 FFN + 4 attn / 5 GDN | gate, up, down; q, k, v, wo; qkv, gate, alpha, beta, out; `lm_head` |
| attention front end + core | 144 | 9 | qg_split, 2 head norms, 2 M-RoPE, kv_append, split, reduce, q_gate |
| GDN step | 336 | 7 | gates, conv, silu, 2 l2norm, gdn, gate_norm |
| norms, residuals, silu | 320 | 5 | attn_norm, post rmsnorm, silu, 2 residual |
| embed + output norm | 2 | — | embedding row, `output_norm` |
| **total** | **1,299** | | |

The 409 in the brief is the GEMV-only count of an earlier layer plan; the
routed GEMV count here is 497, and the elementwise launches are 802 more.

## 2. CPU or GPU: two experiments

### 2a. One submission instead of 1,299

`gpu-bench --graph` captures the whole plain decode step — every launch,
the KV append, the device argmax — and replays it with one `hipGraphLaunch`
per token, writing only the 32-byte step buffer (position, sequence length,
token id) from the host. Same process, same state, 200 tokens each, three
rounds alternating base, new, graph:

| depth | base (eager) | shipped tree (eager) | graph replay |
| --- | ---: | ---: | ---: |
| 0, ms/token | 32.722 / 32.891 / 32.892 | 32.787 / 32.992 / 32.896 | 33.178 / 33.207 / 33.218 |
| 0, mean | **32.835** | **32.892** | **33.201** |
| 0, tok/s | 30.52 | 30.48 | 30.11 |
| 8,192, ms/token | 33.454 / 33.483 / 33.489 | 33.459 / 33.456 / 33.494 | 33.451 / 33.463 / 33.476 |
| 8,192, mean | **33.475** | **33.470** | **33.463** |
| 8,192, tok/s | 29.87 | 29.88 | 29.89 |

Replacing all 1,299 host submissions with one is **+0.37 ms at depth 0 and
−0.01 ms at 8,192**. The host was never the limiter: `gpu-profile` measures a
cached launch at **2.603 µs**, i.e. 3.38 ms of host work per token against a
32.8 ms token, so the host runs a token ahead of the GPU and the gaps are not
CPU time. The graph probe in the same process says the same: eager 31.824 ms
with 29.871 ms of host issue, graph 32.090 with 30.112.

### 2b. Adding host delay per launch

`HIPML_LAUNCH_SPIN_US=N` burned N µs of host time before each launch on the
base tree (temporary probe, not retained), 16 tokens:

| added host time | token | delta |
| --- | ---: | ---: |
| 0 µs | 32.125 ms | — |
| 5 µs (6.5 ms/token of extra issue) | 32.701 ms | +0.58 |
| 50 µs (65 ms/token of extra issue) | 72.651 ms | +40.53 |

At 5 µs the host still needs only ~9.9 ms per token and the delay is
**absorbed**: the token does not grow with it. At 50 µs the host needs ~68 ms,
past the GPU's 32.8, and the token becomes host-bound. The crossover is where
host issue per launch (~2.6 µs + N) exceeds the GPU's per-kernel consumption
(~25 µs average), and the shipped path sits 10× on the GPU side of it. A graph
therefore had nothing to win, which is what 2a measured.

### 2c. The capture position bug this exposed

`build_token_graph` captured at position 0, and a launch *shape* depends on the
position: quantized decode attention divides the key range into one split per
64 cached positions, and a graph bakes its grid. Replaying at 8,192 therefore
ran the mono kernel over the whole cache: **115.364 ms/token (8.67 tok/s)**.
The capture now takes the position it will replay from (`build_token_graph(st.pos)`),
and the same replay is 33.46 ms/token, with the greedy ids identical to eager
over 200 tokens at both depths (§4).

## 3. Fusing the small kernels: tried, measured, reverted

Per layer the decoder issued `residual_add` then `rmsnorm` twice: once after the
block and once after the FFN, and a third time for `output_norm`. A fused
`residual_rmsnorm_f32` already ships (used by prefill; bit-identical by
construction: same `a+b`, same ascending sum of squares, same `v*scale*w`
order). The decode driver was changed to use it everywhere — layer 0 keeps one
standalone norm, every later block's input norm rides on the previous layer's
residual, and the last layer folds `output_norm`:

| | launches/token | @0 ms | @8K ms |
| --- | ---: | ---: | ---: |
| eager, separate residual + norm | 1,299 | 32.916 (3 reps) | 33.524 |
| eager, fused | **1,171** | **33.126** | **33.705** |

The first campaign ran base first every round, so it was repeated with the
order alternating inside the lock window: 32.820 → 33.012 ms at 0 and
33.465 → 33.656 at 8,192. 128 fewer launches is **+0.19 ms, consistently
slower**. Reverted; the fused kernel is kept where it already was (prefill).

This is the strongest single piece of evidence in the note: a 10% cut in
launch count did not move the token, so the per-launch component is small
enough to be lost in the kernel-to-kernel difference of the merge itself.

## 4. What the 5.3 ms is, and the verdict on 34 tok/s

The budget, with everything dispatch could touch now measured:

| component | ms | status |
| --- | ---: | --- |
| weight stream, `w4` at the GEMV's own geometry | 26.08 | floor |
| GEMV above that stream (kernel chain, not layout) | 0.90 | refuted for repack |
| launched-kernel component (1,299 → 1, or → 1,171) | **< 0.2** | measured here |
| GDN recurrence alone, 48 layers | 1.17 | kernel work |
| everything else non-GEMV (attention front end + core, norms, residuals, silu, embed, lm_head tail) | ~2.1 | kernel work |
| **measured token** | **32.84** | |

34.0 tok/s is 29.41 ms. The weight floor alone is 26.08, so all remaining
work must fit in 3.33 ms while the non-GEMV kernels already measure 3.3 ms
(with the launches they need). The only lever that could take it is doing
that elementwise work *inside* the neighbouring GEMV or attention launch (one
kernel, one ramp), which is a kernel change in the files another slice owns;
the residual/norm merge shows that even a 2→1 merge of this exact kind does
not automatically pay.

**Verdict: no. The 5.3 ms is GPU-side kernel execution, and the measured
cost of a launch is below 0.2 ms of it.** Decode stays at 32.8 / 33.5 ms
(30.5 / 29.9 tok/s) on this tree.

## 5. What ships

1. `build_token_graph(pos)`: capture at the position the graph will replay
   from, with the split grid that position needs. Without it the depth graph
   was 3.5× slower than eager (115.36 ms); with it, 33.46 ms and id-exact.
2. `gpu-bench --ids-out FILE`: a greedy 200-token run through either path,
   feeding each sampled id back, written one id per line. This is the tool
   the graph-vs-eager comparison needs; the timing loop is unchanged.
3. This note.

Nothing on the eager decode path changes. `--graph` stays opt-in: it is
correct now, and equally fast, but not faster.

## 6. Gates and invariants

| gate | result |
| --- | --- |
| greedy 12 ids, UD | `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]`, byte-identical |
| 200 greedy ids, graph vs eager | **identical** at depth 0 and at 8,192 |
| 200 greedy ids, base vs this tree | **identical** at depth 0 and at 8,192 |
| `oracle-gate` UD, 169 ids, 20 steps, q8 | **PASS 20/20**, prefill max abs **0.1086** at 101748 |
| `oracle-gate` stock, 169 ids, 20 steps, q8 | **PASS 20/20**, prefill max abs **0.1481** at 34114 |
| `gemv-check`, 6 tensors × 2 columns | **0 differ**, max abs 0.0 |
| `parity` | 69 passed, 0 failed, 1 skipped |
| `batch-bench --slots 16 --tokens 16 --check` | 102.52 / 101.55 / 101.08 / 101.20 aggregate over two alternating pairs; 16/16 ids per slot on all 16 slots |
| `cargo test --workspace` | 0 failed |
| `cargo fmt --all -- --check` | clean |

The batch pair is included because it is the neighbour the change could reach:
the batch engine's own step is untouched, and the 1.4% spread between pairs
reverses when the order reverses (new 101.08 before base 101.20), i.e. it is
the box, not the tree.
