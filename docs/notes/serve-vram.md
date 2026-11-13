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
**MTP path** — a full-context f32 KV, one recurrent-state snapshot per draft and
the verify path's scratch, but *not* the block's weights, which are inline in
the model file and already in the resident-weights row (§3). llama-server has
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

### The sidecar adds nothing; what the MTP path costs

`Model::load_sidecar` uploads only the tensors whose names are not already
resident, and returns only what it uploaded
(`crates/hipml-model/src/lib.rs:380-430`):

```rust
for info in &infos {
    if self.index.contains_key(&info.name) {
        continue;
    }
```

`Model::load` has no block filter — it uploads every tensor in the file
(`lib.rs:265-311`) — and this file carries the MTP block inline.
`Qwen3.8-27B-UD-Q4_K_M.gguf` is an 866-tensor file, the count the stock family
has with `block_count` 65 = 64 decoder + 1 MTP and `nextn_predict_layers` 1
([`../native-qwen.md`](../native-qwen.md), "Architecture config"); the private
fine-tune, which has no inline block, has 851. The difference is 15 tensors,
exactly the 15 `blk.64.*` names `MtpGpu::new` reads
(`crates/hipml-model/src/gpu/mtp.rs:94-131`), and `mtp_blk(cfg)` returns
`cfg.n_layer` = 64 (`mtp.rs:29-31`), so the sidecar's block and the inline block
collide by name.

Measured, not inferred: with the sidecar loaded and MTP running, the stock
file's recorded serve banner reads

```
hipml serve on http://127.0.0.1:8093 | 866 tensors, 15.65 GiB resident + 0.00 GiB drafter
```

(`bench/release/server/hipml_a/server.log:1`; that cell's `result.json` records
1,681 spec steps), while the fine-tune reads `851 tensors, 16.01 GiB resident +
0.25 GiB drafter` (`bench/serve/results/wp51/hipml_b/server.log:1`).
`load_sidecar` adds **0 bytes** on a file that already has `blk.64`, and
0.25 GiB — the block itself — only on the fine-tune. The sidecar file is
1,369,590,656 B because it also carries the target's `token_embd`, `output` and
`output_norm`, which `load_sidecar` skips by name on both files. The 253.7 MiB
row this note used to carry was that fine-tune number: it is not a cost on this
file, and llama.cpp holds the same weights (15.42 GiB resident on both sides,
[`../benchmarks.md`](../benchmarks.md)), so it is not a delta against it either.

What the MTP path does cost is its runtime state, and the `no --mtp` ablation
measures the whole of it. Two pieces are computable from the code: the drafter's
f32 KV, `2 × n_ctx × (4 kv heads × 256) × 4 B` = 256.0 MiB at 32,768
(`crates/hipml-kernels/src/attn_gpu.rs:320-322`), and one verify snapshot
column, a full 48-layer recurrent state = 149.6 MiB
(`crates/hipml-model/src/gpu/columns.rs:145-155`). Together they leave 95.6 MiB
of the 501.2: the speculative path's own scratch, which is the engine's
four-column tile, allocated on the first speculative step with `cap =
max(rows, 4)` (`columns.rs:478-500`), and the prompt-hidden capture, `prompt ×
5,120 × 4 B` = 47.3 MiB at this bench's 2,423-token prompt
(`crates/hipml-model/src/gpu/spec.rs:263`). This note does not itemise the
remainder further.

At three drafts the same three terms are 256.0 + 3 × 149.6 + 95.6 = 800.4
against a measured 801 (21,072 − 20,271), and the d1 → d3 difference — 299.8
measured against 2 × 149.6 = 299.2 computed — is the two extra snapshot columns.
**501.2 and 801 are the same path at drafts 1 and 3, not a contradiction; 509.7
was `253.7 + 256.0`, the block's resident bytes counted a second time.**

The rest is allocation-site arithmetic. For cell a it sums to 20,000.6 MiB
against a measured 20,592.5, so 591.9 MiB (2.9%) is HIP allocation rounding,
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
| MTP drafter f32 KV, 1 × 32,768 | 256.0 |
| MTP verify snapshot, 1 column | 149.6 |
| MTP verify scratch + capture (residual to the 501.2) | 95.6 |
| batch per-slot activations + logits | 1.4 |
| batch SSM scratch | 8.1 |
| **accounted** | **20,000.6** |
| allocator rounding + runtime | 591.9 |
| **measured peak** | **20,592.5** |

The blk.64 weights are not a row here: they are inside the 15,788.9 MiB of
resident weights, because the file carries them and `Model::load` uploads every
tensor in it.

The same table for b and c differs in the per-slot rows (KV × slots, SSM state
× slots, slot scratch) and in the drafter KV at the cell's ctx. b and c run
`--drafts 1`, so the drafter's f32 KV is resident there too, 128.0 / 64.0 MiB,
but no speculative step was priced in those cells: the snapshot columns and the
four-column tile are allocated on the first speculative step
(`columns.rs:478-500`, `spec.rs:263,278`) and whether one ran inside the b/c
peak is not measured. Counting the KV alone, those two close to 23,479.3 /
24,404.4 against 24,284.9 / 25,286.9, leaving 805.6 / 882.5 MiB (3.3% / 3.5%)
unattributed.

Laid against llama, the delta decomposes as follows. Every row is measured or
computed except the last, which is the remainder this note does not attribute —
2.9–3.5% of each cell's peak:

| delta vs llama, MiB | a | b | c |
|---|---|---|---|
| second KV cache (single-sequence engine) | +1,088.0 | +544.0 | +272.0 |
| second SSM state (same engine) | +149.6 | +149.6 | +149.6 |
| SSM alpha/beta fp16 copies | +90.0 | +90.0 | +90.0 |
| prefix snapshots (2 live) | +641.5 | +641.5 | +641.5 |
| MTP drafter f32 KV (computed) | +256.0 | +128.0 | +64.0 |
| MTP verify path (measured at a; not priced at b/c) | +245.2 | — | — |
| batch per-slot activations + SSM scratch | +9.5 | +76.1 | +152.2 |
| hipml scratch less llama compute buffer | −274.3 | −639.5 | −633.1 |
| unattributed (allocator, driver, runtime, b/c verify) | +598.0 | +805.6 | +882.5 |
| **sum** | **+2,803.5** | **+1,795.3** | **+1,618.7** |
| measured | +2,803.5 | +1,795.3 | +1,618.7 |

Cell a's remainder is 591.9 in the allocation table above and 598.0 here
because the two tables derive the scratch row independently; the 6.1 MiB was
already in the first version of this note.

"hipml scratch less llama compute buffer" is derived: hipml's chunk, split-K and
alpha/beta scratch is 488.2 / 512.2 / 500.2 MiB, while llama's peak leaves
762.5 / 1,151.7 / 1,133.3 MiB above weights plus one q8 KV cache. That llama
remainder is its graph compute buffers and any recurrent state; this note does
not separate them.

So the answer to "which part" is: **the duplicate engine, the prefix snapshot
and the MTP path's KV and verify scratch.** In cell a the duplicate engine's
1,088.0 MiB of KV is never read — `sched/run.rs` builds `GpuEngine` and
`BatchEngine` unconditionally, and a one-slot cell is served from the former.
Deleting that allocation when `slots` is 1 (or making the batch engine lazy)
would put cell a at ~19,348 MiB, within 1.6 GiB of llama with the drafter, and
within ~1.0 GiB without it. That is a scheduler change, outside this slice;
cells b and c use the batch engine, so they would not move.

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

The MTP path is worth +60% on cell a's decode (29.77 → 47.56 tok/s) for its
501.2 MiB at one draft. That is the trade already in the tree, not one this
slice made.

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

```powershell
$env:HIP_PATH   = "C:\rocm-sdk\_rocm_sdk_core"   # the directory with bin\ and lib\
$env:HIPML_ARCH = "gfx1201"
$env:PATH       = "$env:HIP_PATH\bin;$env:PATH"

$MODELS   = "<models>"        # the directory holding the two gguf files
$BUILD    = "<build.cmd>"     # builds hipml with the variables above
$HIPML    = "<hipml.exe>"     # the binary under test
$OUT      = "<out-dir>"       # one subdirectory per probe
$GPU_LOCK = "<gpu-lock.ps1>"  # acquire <label> / release <label>

& $BUILD
& powershell -NoProfile -File $GPU_LOCK acquire "vram"
python temp\vram_probe.py --cell a --reps 3 `
  --model "$MODELS\Qwen3.8-27B-UD-Q4_K_M.gguf" `
  --mtp   "$MODELS\mtp-Qwen3.8-27B-Q4_0.gguf" `
  --hipml $HIPML --out "$OUT\after-a"
& powershell -NoProfile -File $GPU_LOCK release "vram"
```

The probe writes the ambient `HIPML_GGUF` / `HIPML_MTP_GGUF` /
`HIPML_Q5K_COPY_GB` into `server.log` before it starts the server, and always
passes `--gguf` and `--mtp` literally: the ambient values on this box are the
stock Q4_K_M file and would silently measure the wrong model. The model sha is
asserted by `hipml load --gguf`, which prints it.
