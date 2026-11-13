# SVR: where serve's extra VRAM goes on UD-Q4_K_M, and the Q5_K copy that was never there

Base `0123ecb` (origin/main), branch `feat/pro-1491-vram`. Box: R9700
(gfx1201, 64 CUs, 32 GB), `Qwen3.8-27B-UD-Q4_K_M.gguf`
(16,564,259,744 B, sha256 `1bce504dfecd84153666928a41d8a66706f0158d3a317bc5196b72819d7dc529`),
q8 KV, MTP drafter `mtp-Qwen3.8-27B-Q4_0.gguf`, port 8093, every GPU run under
the `vram` lock. Cells are the ones in `bench/serve/serve_bench.py`: a = 1 slot
at ctx 32,768, b = 8 at 16,384, c = 16 at 8,192, all with
`--prefix-entries 16 --prefix-mb 2048 --chunk 256 --drafts 1`. llama.cpp peaks
are the `llama_{a,b,c}.json` files in `bench/serve/results/`, same box, same
cells, same q8 KV, `-ngl 99 --no-webui --no-cache-idle-slots`, total context
131,072 for b and c.

## 0. Question and answer

Question: `hipml serve` holds ~2.5–3.3 GiB more VRAM than `llama-server` per
serve cell on the UD file. The card suspected the optional Q5_K fp16 prefill
copy. Is it the copy, and if not, what is it?

Answer: **not the copy.** The copy is off by default, and on this file it could
never have been on: the Q5_K tensors are 4.87 GiB, so their fp16 copies would be
14.17 GiB, and `HIPML_Q5K_COPY_GB` is unset. The prefill actually launches the
dequant-in-shared tile kernel `prefill_q5_k`. Deleting the copy path changes no
measured peak by a byte (section 5) because the path was already dead.

The excess is structural, and it is the same three things in every cell: a
**second KV cache and SSM state** for the single-sequence engine that serves
prefill and speculation, a **prefix snapshot** (two are live at peak), and the
**MTP drafter** (its own weights plus a full-context f32 KV). llama-server has
one cache, no drafter and no snapshot of that kind. Per-kernel scratch — the
256-token chunk, split-K partials, the fp16 alpha/beta copies — is *smaller*
than llama's own compute buffers and is not the gap. Section 3 closes the
difference line by line, cell by cell.

The one part that is removable without an architecture change is cell a's
`BatchEngine`: it is built for every cell, and the measured single-slot workload
never enters it. Making it lazy saves ~1,245 MiB in cell a and needs the
scheduler, which this slice is told not to touch. It is the follow-up.

## 1. What the Q5_K copy was

`Proj::of` could materialise an fp16 copy of every Q5_K matrix, gated on
`HIPML_Q5K_COPY_GB` naming a budget in GiB. The copy was 2.9× the block bytes
(176 bytes per 256 values become 512), and the prefill then used
`prefill_halfw_h` / `prefill_halfw_w2h`, which stream half the weight passes of
the block form. `q5_copy_budget_bytes()` returned `None` when the variable was
unset, so **no copy was made by default**; the default prefill route was already
the tile kernel, `prefill_q5_k` / `prefill_q5_k_w2h`.

On the UD file the copy could not have been made even if someone asked:

```
Q5_K tensors          142, 5,230,100,480 B (4.87 GiB)
fp16 copies           15,214,837,760 B (14.17 GiB)
resident weights      15,788.9 MiB
```

A budget below 14.17 GiB was refused by the `q5_copy_bytes(model) <= max` test;
a budget at or above it would have tried to put 14.17 GiB beside 15.42 GiB of
weights on a 32 GiB card. The only reachable states were "no copy" and "OOM",
which is why the knob was a footgun rather than a fallback.

## 2. The deletion

`crates/hipml-model/src/gpu_weights.rs`: −136/+14 lines. Removed
`q5_copy_budget_bytes`, `q5_copy_bytes`, the local `dequant_q5_k` wrapper, the
`Proj::q5_half` field and every branch that consulted it
(`gemm`/`gemm_half`/`gemm_half_silu`/`gemm_half_gu`/`bytes`/`has_half`).
`HIPML_Q5K_COPY_GB=0` still selects the old batched-GEMV route
(`q5_tile_prefill_enabled()`), so `docs/native-qwen.md`'s 3.0–3.7× comparison
between the tile and the batched GEMV stays reproducible. The doc comment in
`crates/hipml-kernels/src/launch.rs` no longer claims the resident loader reads
the variable, and `docs/articles/06-whats-inside-unsloth-dynamic-quants.md`
now says the repository used to keep that copy and no longer does.

Nothing else changed: no kernel, no arithmetic, no schedule.

## 3. The difference against llama, line by line

Measured peaks, same cells, same box:

| cell | slots × ctx | llama.cpp | hipml | delta |
|---|---|---|---|---|
| a | 1 × 32,768 | 17,789.0 | 20,592.5 | **+2,803.5** (2.74 GiB) |
| b | 8 × 16,384 | 22,489.6 | 24,284.9 | **+1,795.3** (1.75 GiB) |
| c | 16 × 8,192 | 23,668.2 | 25,286.9 | **+1,618.7** (1.58 GiB) |

Two measured ablations on cell a price the two largest hipml-only items:

| change | peak | delta |
|---|---|---|
| full cell a | 20,592.5 | — |
| `--prefix-mb 1` | 19,951.0 | **−641.5** |
| no `--mtp` | 20,091.3 | **−501.2** |

The rest is allocation-site arithmetic. For cell a it sums to 20,009.1 MiB
against a measured 20,592.5, so 583.4 MiB (2.8%) is HIP allocation rounding,
the driver and runtime, not a buffer:

| cell a, MiB | |
|---|---|
| resident weights, 866 tensors | 15,788.9 |
| KV: single-sequence engine, 1 × 32,768 | 1,088.0 |
| KV: batch engine, 1 × 32,768 | 1,088.0 |
| SSM state: single-sequence engine | 149.6 |
| SSM state: batch, 1 slot | 149.6 |
| SSM alpha/beta fp16 copies (both engines) | 90.0 |
| attention split-K scratch (2 engines, 256 splits) | 192.0 |
| deep-KV split-K scratch | 96.0 |
| batch KV split-K scratch | 6.0 |
| prefill chunk scratch (256 tokens) | 200.2 |
| prefix snapshots, 2 live at peak | 641.5 |
| MTP drafter weights | 253.7 |
| MTP f32 KV, 1 × 32,768 | 256.0 |
| batch per-slot activations + logits | 1.4 |
| batch SSM scratch | 8.1 |
| **accounted** | **20,009.1** |
| allocator rounding + runtime | 583.4 |
| **measured peak** | **20,592.5** |

The same table for b and c differs only in the per-slot rows (KV × slots, SSM
state × slots, slot scratch, and the drafter KV at the cell's ctx). Those two
close to 23,733.0 / 24,658.1 against 24,284.9 / 25,286.9.

Laid against llama, the delta decomposes to within 6 MiB of rounding:

| delta vs llama, MiB | a | b | c |
|---|---|---|---|
| second KV cache (single-sequence engine) | +1,088.0 | +544.0 | +272.0 |
| second SSM state (same engine) | +149.6 | +149.6 | +149.6 |
| SSM alpha/beta fp16 copies | +90.0 | +90.0 | +90.0 |
| prefix snapshots (2 live) | +641.5 | +641.5 | +641.5 |
| MTP drafter (weights + f32 KV) | +509.7 | +381.7 | +317.7 |
| batch per-slot activations + SSM scratch | +9.5 | +76.1 | +152.2 |
| hipml scratch less llama compute buffer | −274.3 | −639.5 | −633.1 |
| allocator rounding + runtime | +583.4 | +551.8 | +628.8 |
| **sum** | **+2,797.4** | **+1,795.2** | **+1,618.7** |
| measured | +2,803.5 | +1,795.3 | +1,618.7 |

"hipml scratch less llama compute buffer" is derived: hipml's chunk, split-K and
alpha/beta scratch is 488.2 / 512.2 / 500.2 MiB, while llama's peak leaves
762.5 / 1,151.7 / 1,133.3 MiB above weights plus one q8 KV cache. That llama
remainder is its graph compute buffers and any recurrent state; this note does
not separate them.

So the answer to "which part" is: **the duplicate engine, the prefix snapshot
and the drafter.** In cell a the duplicate engine's 1,088.0 MiB of KV is never
read — `sched/run.rs` builds `GpuEngine` and `BatchEngine` unconditionally, and a
one-slot cell is served from the former. Deleting that allocation when `slots`
is 1 (or making the batch engine lazy) would put cell a at ~19,348 MiB, within
1.6 GiB of llama with the drafter, and within ~1.0 GiB without it. That is a
scheduler change, outside this slice; cells b and c use the batch engine, so
they would not move.

## 4. Prefix cache: what it costs and what it buys

The prefix cache stores one snapshot per cut key: the 48 SSM states
(48 × 817,152 f32 = 149.6 MiB), the attention KV rows `[0, pos)`, and the
logits. At the bench's 2,396-token system prompt that is 230.1 MiB of data, and
the snapshot costs **320.5 MiB** on the device once the 48 state buffers are
rounded to allocation granularity. A re-capture builds the replacement before
`insert` drops the old payload, so **two are live at the peak**.

Measured: `--prefix-mb 1` on cell a moves the post-warm sample from 20,224.1 to
19,903.6 (**−320.5**, one snapshot) and the peak from 20,592.5 to 19,951.0
(**−641.5**, two), with the static value unchanged — nothing is reserved up
front; `--prefix-mb` is a hard bound in `PrefixCache::new`, not an allocation.

It buys a full hit on the shared system prompt in the multi-slot cells:

| cell | hit rate | hit tokens | TTFT p50 | aggregate tok/s |
|---|---|---|---|---|
| a (reuse off) | 0.00 | 0 | 2,885 ms | 47.56 |
| b | 1.00 | 2,396 | 486 ms | 131.66 |
| c | 1.00 | 2,396 | 742 ms | 221.12 |

Cell a sends `cache_prompt: false` — the bench only asks for reuse on the
multi-slot path — so it reports 0 by construction. b and c send the same
2,396-token system prompt to every slot and hit on every request after the
first.

Disabling reuse on cell b prices the snapshot in the cell that uses it:

| cell b, after binary | prefix on | `--prefix-mb 1` |
|---|---|---|
| peak | 24,284.9 | 23,934.8 (−350.1) |
| TTFT p50 | 486 ms | 14,163 ms |
| aggregate tok/s | 131.7 | 21.2 |
| errors | 0 | 0 |

The peak moves less than cell a's 641.5 MiB because the 250 ms sampler does not
catch every re-capture double at 8 slots; the steady-state snapshot is the same
320–350 MiB. The time is unambiguous: without a hit each wave re-prefills
8 × 2,396 tokens, which is 14.2 s of TTFT against 0.49 s. That is what the
snapshot buys, and it is worth its third of a GiB. Against llama's 637 ms at b
and 1,380 ms at c, the hit path is the same order.

## 5. Before and after

The after binary is the same tree with the copy path deleted. Everything that
touches the GPU is byte-identical to the base, which is the point: the removed
code never ran.

| cell | base peak | after peak | base tok/s | after tok/s | load s (base/after) |
|---|---|---|---|---|---|
| a | 20,592.5 | 20,592.5 | 47.56 | 47.58 | 13.18 / 13.12 |
| b | 24,284.9 | 24,284.9 | 131.66 | 131.73 | 13.15 / 13.15 |
| c | 25,286.9 | 25,286.9 | 221.12 | 221.31 | 13.13 / 13.13 |
| a, `--prefix-mb 1` | 19,951.0 | 19,951.0 | 47.12 | 47.11 | 13.16 / 13.13 |
| a, no `--mtp` | 20,091.3 | 20,091.3 | 29.77 | 29.78 | 10.11 / 10.18 |

Static, post-warm and post-run samples are also identical to the byte, for
example cell a: 19,706.3046875 / 20,224.1328125 / 20,592.5234375 on both
binaries. The card's reference peaks (20,091 / 24,110 / 25,201) are not these
numbers: cell a matches a run **without** the drafter to 0.1 MiB, and b and c
sit 175 and 86 MiB below this table, so the reference table was measured with
different flags. The before/after comparison above is same-flags on both sides.
Prefill is unchanged as well, because the tile kernel is the same kernel:

| CLI prefill | base tok/s | after tok/s |
|---|---|---|
| 512 tokens | 1,335.3 | 1,340.2 |
| 8,192 tokens | 1,333.7 | 1,333.5 |

(Decode in those runs: 31.22 / 30.64 base, 31.22 / 30.65 after.)

The MTP drafter is worth +60% on cell a's decode (29.77 → 47.56 tok/s) for its
501.2 MiB. That is the trade already in the tree, not one this slice made.

## 6. What was tried and what was left

- **Deleted** the Q5_K fp16 copy path. No peak, throughput or load change; it
  was unreachable. Reverted nothing because nothing else was changed.
- **Measured** prefix cost and MTP cost by ablation, so the note's structural
  claim rests on deltas, not on arithmetic alone.
- **Not done:** lazy `BatchEngine`. Cell a would recover ~1,245 MiB (KV 1,088.0
  + SSM 149.6 + slot/SSM scratch 9.5, plus its share of alpha/beta), and the
  measured single-slot workload never uses it. It requires `serve/sched/*` and
  `gpu/batch.rs`, which this slice was told to leave alone. Recorded here as the
  next lever.
- **Not done:** a single shared KV between the two engines. It is the larger
  win at b/c (544/272 MiB of duplicate KV plus 149.6 MiB of SSM) but it is a
  scheduler and engine-ownership change, not a residency fix.

## 7. Invariants

All gates on the after binary, one hold, 876 s:

- `oracle-gate` UD: PASS 20/20, `max |gpu - cpu| = 0.1086 at 101748`.
- `oracle-gate` stock Q4_K_M: PASS 20/20, `0.1481 at 34114`.
- `parity --deep`: 74 passed, 0 failed, 1 skipped.
- `batch-bench --check`: 8 slots and 16 slots, 16/16 exact.
- `spec-bench`: q8 and f32 KV, drafts 0–3, 128/128 exact.
- 16 serve conversations at ctx 32,768: sha1 of every response identical to
  `0123ecb`, 0 errors.
- Greedy ids on UD: `[11751, 13, 198, 760, 6511, 314, 9564, 369, 19241, 13, 198, 760]`.
- `cargo test --workspace`: 92 tests, 0 failed.
- `cargo fmt --all -- --check`: clean.

## 8. Reproduction

```
C:\hipml-vram\build.cmd            # HIP_PATH=C:\rocm-sdk\_rocm_sdk_core, HIPML_ARCH=gfx1201
powershell C:\models\gpu-lock.ps1 acquire "vram"
python temp\vram_probe.py --cell a --reps 3 ^
  --model C:\models\Qwen3.8-27B-UD-Q4_K_M.gguf ^
  --mtp   C:\models\mtp-Qwen3.8-27B-Q4_0.gguf ^
  --hipml C:\hipml-vram\hipml-after.exe --out C:\hipml-vram\temp\after-a
powershell C:\models\gpu-lock.ps1 release "vram"
```

The probe writes the ambient `HIPML_GGUF` / `HIPML_MTP_GGUF` /
`HIPML_Q5K_COPY_GB` into `server.log` before it starts the server, and always
passes `--gguf` and `--mtp` literally: the ambient values on this box are the
stock Q4_K_M file and would silently measure the wrong model. The model sha is
asserted by `hipml load --gguf`, which prints it.
