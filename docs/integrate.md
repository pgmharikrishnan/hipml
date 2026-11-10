# Release integration (WP52 + WP50 onto WP51)

Base: `6a1207f` (`feat/release`, the WP51 server work). Branch `feat/integrate`.
Box: R9700 (gfx1201, 32 GB), the gate model (the fine-tune the gate corpus was
captured on — not the stock Q4_K_M the release numbers use) +
`mtp-Qwen3.8-27B-Q4_0.gguf`, q8 KV. Ports 8093 only; every GPU cell ran under
the `wp53` lock.

## Merge order and cost

| # | branch | commits | textual conflicts | integration work |
| --- | --- | --- | --- | --- |
| 1 | `feat/splitk-fix` (WP52) | 6 | none | none |
| 2 | `feat/mtp-batch` (WP50) | 6 | none (ort auto-merge) | one build fix: `ssm_cols_rows` was an unconditional module, so the default (no-`hip`) workspace build failed on `Buffer::as_ptr`; gated it like `ssm_cols` |

WP50 was based on `feat/final2`, an ancestor of the base, but its changes are
additive in the model/kernel crates; the overlap in `batch.rs`, `gpu.rs` and
`lib.rs` was import/module plumbing. The conflict the task anticipated did not
materialise as a text conflict; it materialised as the feature gate above.

**The served contract is unchanged.** `git diff 6a1207f HEAD --
crates/hipml-cli/src/serve` is empty: `Mode::Spec` is still selected only when
`alone && cfg.mtp`, and no scheduler path calls the batched drafter. MTP-in-batch
is reachable only through `hipml mtp-batch-bench`, i.e. opt-in and explicitly
scheduled. The measured schedule is `R = floor(16/B) - 1` (B=1→3, B=4→3,
B=8→1) and MTP off at B=16, from the `B·(1+R) ≤ 16` 16-row verify tile.

## Gates on the merged tree

| gate | result |
| --- | --- |
| `cargo test --workspace` (local and gpu-host) | all suites pass; 1 hip unit test (`split_scratch_cap_covers_every_selectable_tile`) passes on gpu-host |
| `cargo check --workspace --features hip` | clean |
| `oracle-gate --ids-file p2_math_169.ids --steps 20 --chunk 2048 --kv q8` | **20/20**, step 19 = **41020** |
| 20-id acceptance | PASS, 20/20; the string is model-specific and is in [`release-run.md`](release-run.md) |
| `parity --deep` | 56 passed, 0 failed, 1 skipped (`sample_top_p_f32`, host-only) |
| `parity prefill --deep` | 6 passed, 0 failed, 0 skipped |
| `splitk-gate` synthetic | PASS 44/44, worst \|dlogit\| 0.0112 ≤ 0.05 |
| `splitk-gate --ids-file p2_math_169.ids` | PASS 44/44, ids identical, worst \|dlogit\| 9.3098 ≤ 16 |
| 8,193-token prefill | completes in **6.82 s**, last id 293 (the crash length) |
| `gemv-check` 4 tensors × R=1/2/4 | 12/12, 0 differing, max \|d\| 0.000e0 |
| `spec-bench` drafts 0..3 | every row 128/128 exact; drafts 3 = 50.6 tok/s, 1.73x |
| `batch-bench --check` B=8 | all 8 slots 16/16 ids match B=1; 91.5 tok/s aggregate |
| `rows-check` 16 tails | 48 decode ids and the whole 39,223,296-word SSM state bit-identical |
| `tokenizer_fixture` | 2/2 pass |
| prefill 8K q8 (gpu-bench) | **6.66 s** (≤ 6.7), last id 292 |
| prefill 100K q8 (gpu-bench) | **107.98 s** (≤ 108.5), last id 100 |
| serve cell b, 8 slots | **130.3 tok/s** aggregate, TTFT p50 **489 ms**, 0 errors, 26.3 GiB (`bench/integrate/serve_cell_b.json`) |
| single-stream MTP | **38.9 tok/s** p50 (39.2/38.9/38.9), TTFT 247 ms (`bench/integrate/exact.jsonl`) |
| `mtp-batch-bench` B=8 R=1 skinny | 165.6 agg / 20.7 per stream, 1.12x plain, exact over 256 tokens |
| `mtp-batch-bench` B=1 R=3 skinny | 62.0 agg / 62.0 per stream, 2.21x plain, exact over 256 tokens |

The batch exactness cells in `exact.jsonl` also reproduce WP51's routing:
tile 4 slots 67.3 vs exact 66.1, tile 8 slots 106.9 vs exact 75.2.

## Unverified / notes

* Full `mtp-batch-bench` matrix (B=4, B=16, admission, exact `_b` mode) was not
  re-run on the merged tree; the two operating-point cells above were.
* The 8,193 gpu-bench cell reports last id **293** on this tree (twice, and
  6.74/6.78 s); `docs/wp52.md` records 852 for the same command. This tree's
  `gpu/prefill.rs`, `prefill_launch.rs` and `gpu_bench_cmd.rs` are byte-identical
  to `feat/splitk-fix`, and WP51 did not touch them, so 852 cannot be produced
  by the shipped split path. It is most likely a `HIPML_SPLITK=0` or pre-fix
  observation; the gate result (ids identical, worst \|dlogit\| 9.309817, exactly
  WP52's number) is unaffected. 8,192 still gives 292.
* WP50's doc lists the p2 oracle cell as unverified because that run was
  aborted. On this tree it is green (20/20, step 19 = 41020).
* Peak VRAM for the batched verify tile was not sampled beyond the serve cell's
  process-wide 26.3 GiB.
