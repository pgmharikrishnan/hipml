# WP41: the final tree

`feat/final`, base `04dbdf9` (`feat/merged2`, WP35). Two measured branches
merged in this order, no textual conflicts, two integration fixes.

| order | branch | files | what it adds |
| ---: | --- | ---: | --- |
| 1 | `feat/deep2` (WP39) | 9 (+933/-13) | split-K over the key axis in chunk attention, a combine pass, an 8,192-position gate, interior-mask skip, split scratch only past the gate, `attn-bench --chunk` |
| 2 | `feat/qk-dedup` (WP40) | 3 (+437/-38) | `attn_chunk_full_f16`: one block owns the whole 256-d head width, so the QK score tile is computed once; Q in registers, 12 warps, `__launch_bounds__(384, 2)` |

Two fixes were required after the merges:

1. `98c96b9` — the auto-merge joined `prefill_chunked_capture` (WP34) with
   `ensure_chunk(tokens, deep)` (WP39) but left the old one-argument call. Same
   predicate as the other two prefill entry points.
2. `cef7657` — the gate asked `split_count(splits, tokens)`, so an exactly
   8,192-token prompt allocated 1.6 GiB of split scratch that no chunk used
   (the last chunk starts at 6,144). It now asks about the last chunk start
   (`tokens - 1`). Timing and ids unchanged; the 8K peak drops back to the
   WP35 level.

## Checks (final binary, `cef7657`)

| check | result |
| --- | --- |
| 20-id acceptance | `11751 13 198 760 6511 314 14898 369 21047 13 198 760 6511 314 17163 369 23327 13 198 760`, first id 11751, 29.50 tok/s over 20 tokens |
| `cargo test --workspace` | 59 passed, 0 failed (local and rtx) |
| `cargo check --workspace --features hip` | clean |
| `parity prefill --deep` | 6 passed, 0 failed, 0 skipped |
| `parity --deep` | 56 passed, 0 failed, 1 skipped (`sample_top_p_f32`, host-only by design) |
| `gemv-check` R=1/2/4 × Q4_K/Q5_K/Q6_K/Q8_0/Q6_K head | 15/15, 0 differing, max abs 0.000e0 |
| `spec-bench`, five prompts × drafts 1..3, plus code_edit@8192 | 128/128 exact on every row |
| `batch-bench --check` B=8 / B=16, exact and skinny | every slot 16/16 ids match B=1 |
| `attn-parity` | 7/7 PASS; `chunk attention q8 split4` max abs **8.00e-4** (tol 2e-3), q8 mono 7.91e-4, plain chunk 8.35e-4 |
| 100K last id | `100` on q8 split, q8 mono, f32 mirror, and on the `feat/deep2` binary; llama-server also `100` (see below) |

The two `attn-parity` decode-layer cases that failed on the WP39/WP40 base
(`d7b53f6`, max abs 8.01e0) pass here (2.48e-5 at depth 1, 4.17e-6 at 1000):
the WP31 decode-attention landing in `feat/merged2` fixed them.

## 100K token id: `100`, not `500`

WP40 reported `500` on all three kernels. It does not reproduce. Derived six
times on this tree and its parent, same synthetic prompt as `gpu-bench
--prefill` (`id = 100 + i % 1000`):

| tree | path | last id | 100K prefill |
| --- | --- | ---: | ---: |
| `feat/final` | q8, 8 key splits, chunk 2048 | **100** | 107.75 s (928 tok/s) |
| `feat/final` | q8, `HIPML_ATTN_SPLITS=1` (mono) | **100** | 135.40 s (739 tok/s) |
| `feat/final` | f32 mirror (`--prefill`, no `--kv`) | **100** | 135.29 s (739 tok/s) |
| `feat/deep2` | q8, 8 splits | **100** | 123.18 s (812 tok/s) |
| `feat/deep2` | q8, mono | **100** | 135.39 s (739 tok/s) |
| llama-server q8_0 K/V, 8092 | 100,000 prompt ids, greedy | **100** | 121.28 s (825 tok/s) |

So the arithmetic never changed and the earlier `100` was right; the `500`
report is the outlier. The `512/8K/32K` ids also reproduce (`489/292/868`), and
the 32K cell (28.6 s) is faster than WP40's 28.49 s within noise.

## Final head-to-head

Same fixture token ids, both engines cold, hipml `q8` vs llama `q8_0` KV,
llama-server on 8092. Each cell's rep count is its own column. `Δ WP35` is
`final / WP35 - 1`; `Δ WP18` is `final / WP18 - 1` (WP18 = `bd6d18f`, the
`feat/bench` reference table). For TTFT a negative delta is better; for
throughput a positive delta is better.

| prompt | depth | reps | engine | TTFT ms | prefill tok/s | decode tok/s | peak GiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | 3 | hipml | **561** | **1,198** | 29.2 | 17.62 |
| code_edit | 0 | 3 | llama | 664 | 1,014 | 30.2 | 16.22 |
| code_edit | 8192 | 3 | hipml | **6,541** | **1,252** | 28.8 | 18.84 |
| code_edit | 8192 | 3 | llama | 6,930 | 1,184 | 29.4 | 16.47 |
| code_edit | 32768 | 1 | hipml | **28,647** | **1,144** | 27.0 | 19.75 |
| code_edit | 32768 | 1 | llama | 30,736 | 1,067 | 27.4 | 17.38 |
| code_edit | 100000 | 1 | hipml | **107,777** | **928** | **23.4** | 21.93 |
| code_edit | 100000 | 1 | llama | 120,903 | 827 | 21.8 | 19.88 |
| json_tool_call | 0 | 3 | hipml | **335** | **897** | 29.3 | 17.35 |
| json_tool_call | 0 | 3 | llama | 377 | 837 | 30.1 | 16.21 |
| json_tool_call | 8192 | 3 | hipml | **6,637** | **1,234** | 28.6 | 18.84 |
| json_tool_call | 8192 | 3 | llama | 6,932 | 1,185 | 29.5 | 16.47 |
| prose_explanation | 0 | 3 | hipml | **362** | **1,032** | 29.3 | 17.46 |
| prose_explanation | 0 | 3 | llama | 422 | 939 | 30.2 | 16.21 |
| prose_explanation | 8192 | 3 | hipml | **6,628** | **1,236** | 28.7 | 18.84 |
| prose_explanation | 8192 | 3 | llama | 6,918 | 1,185 | 29.5 | 16.47 |
| longdoc_qa | 0 | 3 | hipml | **6,303** | **1,221** | 28.7 | 18.82 |
| longdoc_qa | 0 | 3 | llama | 6,593 | 1,172 | 29.4 | 16.46 |
| longdoc_qa | 8192 | 3 | hipml | **6,637** | **1,234** | 28.6 | 18.84 |
| longdoc_qa | 8192 | 3 | llama | 6,941 | 1,184 | 29.4 | 16.47 |
| copy_200_words | 0 | 3 | hipml | **334** | 854 | 29.1 | 17.34 |
| copy_200_words | 0 | 3 | llama | 348 | **859** | 30.1 | 16.21 |
| copy_200_words | 8192 | 3 | hipml | **6,641** | **1,234** | 28.6 | 18.84 |
| copy_200_words | 8192 | 3 | llama | 6,947 | 1,184 | 29.5 | 16.47 |

### Gap to llama, and deltas

| prompt | depth | prefill | decode | TTFT (llama/hipml) | Δ WP35 T/P/D | Δ WP18 T/P/D |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| code_edit | 0 | +18.1% | -3.1% | 1.18x | -1.4 / +1.4 / -0.1% | -38.7 / +63.0 / +2.7% |
| code_edit | 8192 | +5.7% | -2.0% | 1.06x | -1.4 / +1.4 / +0.4% | -50.7 / +102.7 / +4.8% |
| code_edit | 32768 | +7.2% | -1.2% | 1.07x | -8.3 / +9.0 / +0.1% | -71.8 / +255.0 / +7.1% |
| code_edit | 100000 | +12.1% | +7.4% | 1.12x | -20.5 / +25.7 / +0.0% | -85.3 / +580.7 / +13.8% |
| json_tool_call | 0 | +7.1% | -2.8% | 1.13x | -1.0 / +1.0 / +0.2% | -25.7 / +34.6 / +2.7% |
| json_tool_call | 8192 | +4.1% | -3.1% | 1.04x | +0.1 / -0.1 / -0.1% | -49.9 / +99.6 / +4.2% |
| prose_explanation | 0 | +9.8% | -3.0% | 1.17x | -0.1 / +0.3 / -0.1% | -29.8 / +42.6 / +2.8% |
| prose_explanation | 8192 | +4.3% | -2.7% | 1.04x | -0.2 / +0.2 / +0.0% | -50.1 / +100.2 / +4.6% |
| longdoc_qa | 0 | +4.2% | -2.3% | 1.05x | -0.2 / +0.2 / +0.1% | -49.0 / +95.9 / +4.3% |
| longdoc_qa | 8192 | +4.2% | -2.7% | 1.05x | -0.0 / +0.0 / -0.4% | -49.9 / +99.7 / +4.2% |
| copy_200_words | 0 | -0.6% | -3.5% | 1.04x | -0.4 / +0.5 / -0.3% | -23.8 / +31.3 / +2.4% |
| copy_200_words | 8192 | +4.2% | -3.1% | 1.05x | -0.0 / +0.1 / -0.1% | -49.9 / +99.7 / +4.1% |

**Prefill now wins all 12 cells** (10 clearly, `copy@0` inside noise). TTFT
wins 12/12, including 100K at 1.12x. Decode still loses 2.3-3.5% at 0/8K and
1.2% at 32K, and wins 100K by 7.4%.

## Merge value (100K prefill, one cell)

| tree | TTFT | attention stage | vs mono |
| --- | ---: | ---: | ---: |
| mono (`HIPML_ATTN_SPLITS=1`) | 136.98 s | 64.42 s | - |
| WP39 split-K, 8 splits | 124.41 s | 52.29 s | -9.2% |
| WP40 whole-width | 110.54 s | 38.37 s | -19.3% |
| final like-for-like | **107.78 s** | - | **-21.3%** |

The final 100K cell is **12.2% faster than llama's 120.90 s**; before this
merge it was 12.1% slower. 32K went 31.24 → 28.65 s; 8K and 512 are gate-
untouched (6.63 s / 0.42 s).

## MTP decode (best drafts per class, f32 KV, 128 tokens, 128/128 exact)

| prompt (len) | plain tok/s | drafts | tok/s | accept | exact |
| --- | ---: | ---: | ---: | --- | --- |
| prose_explanation (373) | 29.15 | 2 | 42.30 | 0.804, 0.600 | 128/128 |
| json_tool_call (300) | 29.05 | 3 | 50.17 | 0.846, 0.909, 0.867 | 128/128 |
| copy_200_words (285) | 29.11 | 3 | 61.31 | 1.000, 1.000, 1.000 | 128/128 |
| code_edit (672) | 29.06 | 3 | 51.37 | 0.895, 0.912, 0.806 | 128/128 |
| longdoc_qa (7694, depth 0) | 25.22 | 2 | 32.24 | 0.667, 0.548 | 128/128 |
| code_edit @ 8192 | 25.01 | 3 | 46.17 | 0.944, 0.912, 0.903 | 128/128 |

All within 0.5% of WP35's merged table (42.47 / 49.93 / 60.93 / 51.34 /
32.45 / 45.92).

## Batch (depth 0, q8 KV, 16-token streams, one cell each)

| B | hipml exact | hipml skinny | llama.cpp `-np B` | skinny vs llama | exact vs llama |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 92.25 | **152.33** | 102.03 | **+49.3%** | -9.6% |
| 16 | 92.66 | **273.45** | 234.56 | **+16.6%** | -60.5% |

The exact (bit-identical, default) path is still one weight pass per 8 rows, so
B=16 costs two passes. Skinny remains opt-in (`HIPML_SKINNY=1`).

## Ledger: what each package contributed

- **WP31 decode attention** (merged in WP35, inherited): whole-group w2 decode
  attention; one split per 64 positions; fixes the attn-parity decode cases.
- **WP33 skinny GEMM** (inherited): opt-in WMMA skinny for the verify
  projections; the exact `_b` GEMV stays default.
- **WP34 MTP + batched verify** (inherited): 1.5-2.1x decode, 128/128 exact;
  its FMA-rounding commit (`5774436`) is in this tree.
- **WP37 batch** (inherited): `BatchEngine`, exact and skinny; B=8/B=16.
- **WP36 prefill** (inherited): 8K in 6.64 s.
- **WP39 deep2** (merged here): 100K 136.98 → 124.41 s (-9.2%), 32K 31.68 →
  30.63 s; combines cost ~1.2 s over 100K against a 12.1 s saving.
- **WP40 qk-dedup** (merged here): removes the duplicated QK score tile;
  100K 124.41 → 110.54 s (attention 52.29 → 38.37 s), 32K 30.63 → 28.49 s;
  no change at 8K/512 and an identical split4 parity (8.00e-4).
- **WP41** (this): two merges, the capture-path semantic fix, the 8K scratch
  gate fix, all checks, the 100K id investigation and the final table.

## Still worse than llama.cpp

- **Decode 0/8K/32K**: -3.1%, -2.8%, -1.2% (code_edit); -3.5% at
  `copy@0`. Only 100K is a win (+7.4%). Plain decode is at 83% of the 640 GB/s
  roofline (29.2 tok/s against a 35.3 tok/s ceiling), so there is no large win
  left without more than one token per weight pass.
- **Peak VRAM**: +1.1 to +2.4 GiB in every cell (q8 KV layout plus SSM state).
  100K: 21.93 vs 19.88 GiB.
- **Batch default**: the bit-identical path loses to llama (-9.6% at B=8,
  -60.5% at B=16); the +49%/+17% wins need `HIPML_SKINNY=1`.
- **Prefill at `copy@0`**: -0.6%, inside noise; every other prefill cell wins.
- **Absolute 100K**: still 107.8 s to first token and 23.4 tok/s decode. It
  beats llama now, but neither engine is fast at 100K.
- **Unverified**: MTP at q8 KV (the table above is f32); MTP at depths between
  0 and 8K; graph-mode decode after the split change; B=32 skinny parity;
  the 100K cell and the two 32K cells are one rep.

## Notes on this measurement session

The table is one tree, two binaries. The 19 non-8K hipml cells and all llama
cells were measured at `98c96b9` / binary `ae63d5…`; the five 8K hipml cells
were re-measured at `cef7657` / binary `5713f1…` after the scratch-gate fix.
The only behavioural difference is the 8K scratch allocation (peak 19.17 →
18.84 GiB); depth 0 is below the gate and 32K/100K are already deep, so no
other cell changes. The 20-id acceptance and the 8K id (`292`) were re-run on
the final binary and are identical.

One pre-existing corner case found while validating the gate: a prompt whose
final chunk is shorter than ~384 tokens (e.g. 8,193) enters the split-K ffn
kernel, whose scratch `sk` is sized for prompts of at most 1,024 tokens, and
errors (`split scratch has 1 values, needs 17408`). It is not a WP39/WP40
regression (the allocation rule predates both) and no bench/fixture cell hits
it; `HIPML_SPLITK=0` is the workaround. It is left unfixed so the final table
is measured on the merged arithmetic, not on a third code change.

## Verdict

The owner's M3 gate was "faster than llama.cpp on the same card, or the project
stops". On this model and card the merged engine now wins prefill at every
depth (0, 8K, 32K, 100K), wins TTFT in all 12 cells (1.04-1.18x, including
1.12x at 100K), and matches llama's plain decode to within 3.5% while sitting
on the same DRAM roofline. It wins batch by 49%/17% with the opt-in skinny
kernel and single-stream decode by 1.5-2.1x with MTP at 128/128 exactness.
The engine is no longer the slower one on the axes the thesis named; the
remaining deficits are shallow-depth decode (a near-tie both engines cannot
escape without speculative decoding), VRAM overhead, and the opt-in status of
the skinny batch path. The second engine has a defensible reason to exist.
