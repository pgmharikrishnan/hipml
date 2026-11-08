# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33. Depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in the reps column. Both engines use q8-class KV at every depth.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `a8017373eda5beb6adfc9b999abf77c6a20ef5f2` |
| hipml binary sha256 | `031621036a232d7da372fe411f80ee8cd20e270d52edf22303b1a50e885bcb92` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | q8_0 |
| hipml KV type | q8 |
| generated | 2026-11-08T18:18:28.131792+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| code_edit | 0 | 672 | hipml | q8 | 3 | 715 (714-721) | 940 (932-942) | 29.8 (29.8-29.8) | 17.49 | - |
| code_edit | 0 | 672 | llama | q8_0 | 3 | 675 (651-855) | 1,025 (790-1,034) | 30.6 (30.5-30.6) | 16.22 | no |
| code_edit | 8192 | 8192 | hipml | q8 | 3 | 8,233 (8,196-8,248) | 995 (993-1000) | 28.7 (28.6-28.9) | 18.75 | - |
| code_edit | 8192 | 8192 | llama | q8_0 | 3 | 6,928 (6,907-6,999) | 1,185 (1,173-1,189) | 29.8 (29.8-29.8) | 16.47 | no |
| code_edit | 32768 | 32768 | hipml | q8 | 1 | 37,937 | 864 | 26.2 | 22.34 | - |
| code_edit | 32768 | 32768 | llama | q8_0 | 1 | 30,670 | 1,069 | 27.7 | 17.38 | no |
| code_edit | 100000 | 100000 | hipml | q8 | 1 | 860,789 | 116 | 21.2 | 20.50 | - |
| code_edit | 100000 | 100000 | llama | q8_0 | 1 | 120,879 | 828 | 22.1 | 19.88 | no |
| json_tool_call | 0 | 300 | hipml | q8 | 3 | 381 (380-390) | 787 (770-789) | 29.8 (29.8-29.9) | 17.36 | - |
| json_tool_call | 0 | 300 | llama | q8_0 | 3 | 365 (348-491) | 851 (631-865) | 30.6 (30.6-30.7) | 16.21 | no |
| json_tool_call | 8192 | 8192 | hipml | q8 | 3 | 8,259 (8,225-8,265) | 992 (991-996) | 28.7 (28.6-28.8) | 18.75 | - |
| json_tool_call | 8192 | 8192 | llama | q8_0 | 3 | 6,921 (6,887-6,995) | 1,188 (1,174-1,192) | 29.8 (29.7-29.9) | 16.47 | no |
| prose_explanation | 0 | 373 | hipml | q8 | 3 | 414 (414-422) | 901 (883-902) | 29.8 (29.8-29.8) | 17.36 | - |
| prose_explanation | 0 | 373 | llama | q8_0 | 3 | 405 (392-534) | 955 (720-965) | 30.6 (30.5-30.6) | 16.21 | no |
| prose_explanation | 8192 | 8192 | hipml | q8 | 3 | 8,251 (8,239-8,264) | 993 (991-994) | 28.7 (28.6-28.7) | 18.75 | - |
| prose_explanation | 8192 | 8192 | llama | q8_0 | 3 | 6,918 (6,906-6,995) | 1,187 (1,174-1,190) | 29.9 (29.8-29.9) | 16.47 | no |
| longdoc_qa | 0 | 7694 | hipml | q8 | 3 | 7,877 (7,822-7,899) | 977 (974-984) | 28.8 (28.8-28.9) | 18.67 | - |
| longdoc_qa | 0 | 7694 | llama | q8_0 | 3 | 6,542 (6,536-6,611) | 1,177 (1,167-1,182) | 29.8 (29.8-29.9) | 16.46 | no |
| longdoc_qa | 8192 | 8192 | hipml | q8 | 3 | 8,259 (8,240-8,270) | 992 (991-994) | 28.7 (28.6-28.7) | 18.75 | - |
| longdoc_qa | 8192 | 8192 | llama | q8_0 | 3 | 6,930 (6,897-6,991) | 1,186 (1,174-1,190) | 29.8 (29.7-29.9) | 16.47 | no |
| copy_200_words | 0 | 285 | hipml | q8 | 3 | 380 (377-387) | 751 (736-756) | 29.7 (29.7-29.7) | 17.36 | - |
| copy_200_words | 0 | 285 | llama | q8_0 | 3 | 340 (336-470) | 872 (628-885) | 30.5 (30.4-30.6) | 16.21 | no |
| copy_200_words | 8192 | 8192 | hipml | q8 | 3 | 8,251 (8,231-8,262) | 993 (991-995) | 28.8 (28.6-28.8) | 18.75 | - |
| copy_200_words | 8192 | 8192 | llama | q8_0 | 3 | 6,928 (6,905-7,002) | 1,186 (1,173-1,189) | 29.8 (29.8-29.9) | 16.47 | no |

## Merged engine vs the WP18 like-for-like table

WP18 (`bd6d18f`, the `feat/bench` tree) is the reference: same model, prompts, 33 generated tokens and q8-class KV. The merged tree is `a8017373eda5`. Delta is `merged / WP18 - 1`; for TTFT a negative delta is better, for throughput a positive delta is better.

| prompt | depth | WP18 TTFT | merged TTFT | Δ TTFT | WP18 prefill | merged prefill | Δ prefill | WP18 decode | merged decode | Δ decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 915 | 715 | -21.9% | 734.7 | 940.1 | +28.0% | 28.45 | 29.83 | +4.9% |
| json_tool_call | 0 | 450 | 381 | -15.4% | 666.0 | 786.9 | +18.2% | 28.49 | 29.81 | +4.6% |
| prose_explanation | 0 | 515 | 414 | -19.7% | 723.7 | 901.3 | +24.5% | 28.47 | 29.78 | +4.6% |
| longdoc_qa | 0 | 12,349 | 7,877 | -36.2% | 623.1 | 976.8 | +56.8% | 27.55 | 28.84 | +4.7% |
| copy_200_words | 0 | 438 | 380 | -13.4% | 650.6 | 750.9 | +15.4% | 28.41 | 29.72 | +4.6% |
| code_edit | 8192 | 13,259 | 8,233 | -37.9% | 617.8 | 995.1 | +61.1% | 27.48 | 28.68 | +4.4% |
| json_tool_call | 8192 | 13,250 | 8,259 | -37.7% | 618.2 | 991.9 | +60.4% | 27.46 | 28.71 | +4.6% |
| prose_explanation | 8192 | 13,272 | 8,251 | -37.8% | 617.3 | 992.8 | +60.8% | 27.45 | 28.68 | +4.5% |
| longdoc_qa | 8192 | 13,260 | 8,259 | -37.7% | 617.8 | 991.8 | +60.5% | 27.45 | 28.70 | +4.5% |
| copy_200_words | 8192 | 13,260 | 8,251 | -37.8% | 617.8 | 992.9 | +60.7% | 27.48 | 28.79 | +4.8% |
| code_edit | 32768 | 101,704 | 37,937 | -62.7% | 322.2 | 863.8 | +168.1% | 25.25 | 26.18 | +3.7% |
| code_edit | 100000 | 733,577 | 860,789 | +17.3% | 136.3 | 116.2 | -14.8% | 20.61 | 21.17 | +2.8% |

Quantized engines now prefill straight out of the q8 cache at every depth (WP25, below); the f32 mirror is gone. The table above predates that work at 100K; "WP25: 100K prefill" has the new cells.

## Gap (hipml minus llama.cpp)

Positive means hipml is faster; negative means llama.cpp is faster. Each row is its own prompt class; no number is averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | -8.3% | -2.5% | 0.94x |
| code_edit | 8192 | -16.0% | -3.8% | 0.84x |
| code_edit | 32768 | -19.2% | -5.6% | 0.81x |
| code_edit | 100000 | -86.0% | -4.1% | 0.14x |
| json_tool_call | 0 | -7.5% | -2.6% | 0.96x |
| json_tool_call | 8192 | -16.5% | -3.5% | 0.84x |
| prose_explanation | 0 | -5.6% | -2.6% | 0.98x |
| prose_explanation | 8192 | -16.3% | -4.0% | 0.84x |
| longdoc_qa | 0 | -17.0% | -3.4% | 0.83x |
| longdoc_qa | 8192 | -16.3% | -3.8% | 0.84x |
| copy_200_words | 0 | -13.8% | -2.6% | 0.90x |
| copy_200_words | 8192 | -16.3% | -3.5% | 0.84x |

## Headline gaps

- **prefill**: hipml did not win any cell; llama.cpp fastest at `code_edit@100000` (-86.0%, reps 1/1).
- **decode**: hipml did not win any cell; llama.cpp fastest at `code_edit@32768` (-5.6%, reps 1/1).
- **ttft**: hipml did not win any cell; llama.cpp fastest at `code_edit@100000` (-86.0%, reps 1/1).

## llama.cpp warm pass (cache_prompt=true)

Same ids immediately after the cold reps. A near-1 `prompt_n` and a large `cache_n` confirm the prefix was reused; this is a cache-hit TTFT, not a prefill number.

| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 117 | 43.0 | 668 | 4 |
| code_edit | 8192 | 121 | 40.4 | 8188 | 4 |
| code_edit | 32768 | 135 | 37.1 | 32764 | 4 |
| code_edit | 100000 | 184 | 27.7 | 99996 | 4 |
| json_tool_call | 0 | 116 | 41.4 | 296 | 4 |
| json_tool_call | 8192 | 118 | 40.9 | 8188 | 4 |
| prose_explanation | 0 | 118 | 42.6 | 369 | 4 |
| prose_explanation | 8192 | 111 | 41.6 | 8188 | 4 |
| longdoc_qa | 0 | 119 | 40.6 | 7690 | 4 |
| longdoc_qa | 8192 | 121 | 40.9 | 8188 | 4 |
| copy_200_words | 0 | 117 | 42.2 | 281 | 4 |
| copy_200_words | 8192 | 98.0 | 42.1 | 8188 | 4 |

## hipml prefill stage breakdown

One extra synchronised run per cell; the per-stage values include a stream sync each, so `total_wall` is larger than the table TTFT. Reported as a diagnostic of where prefill time goes.

| prompt | depth | total_wall | embed | elementwise | gemm_attn | gemm_ssm | gemm_ffn | final | seq_ssm | seq_attn | stage_sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 747 | 1 | 22 | 33 | 91 | 387 | 4 | 166 | 41 | 745 |
| json_tool_call | 0 | 395 | 1 | 10 | 18 | 51 | 217 | 2 | 77 | 18 | 395 |
| prose_explanation | 0 | 427 | 1 | 11 | 19 | 52 | 228 | 2 | 92 | 22 | 427 |
| longdoc_qa | 0 | 8,237 | 12 | 231 | 335 | 914 | 4,017 | 29 | 1,898 | 800 | 8,235 |
| copy_200_words | 0 | 391 | 1 | 10 | 18 | 51 | 217 | 2 | 75 | 18 | 391 |


## Prefill bisect (512 and 8K, `gpu-bench --prefill`, chunk 512)

| commit | what changed | 512 tok/s | 8K tok/s | 512 last id |
| --- | --- | ---: | ---: | ---: |
| `5945a08` | attn-mma base | 1,273.8 | 1,214.0 | 122271 (half-GEMM) |
| `9532d84` | + graphs | 994.4 | 959.8 | 122271 (half-GEMM) |
| `a967e18` | + wmma2 full-GEMM fix | 854.3 | 825.3 | 489 |
| `219045a` | + cache | 851.8 | 821.1 | 489 |
| `954af72` | + bench, f32 path | 844.8 | 817.7 | 489 |
| `954af72` | + bench, q8 path | 429.6 | 362.1 | 0 (NaN) |
| `4c8bc72` | + chunk sizing, f32 | 837.3 | 818.1 | 489 |
| `4c8bc72` | + chunk sizing, q8 | 405.8 | 354.6 | 489 |
| `4b02c75` | Q5_K half-copy prefill restored | 1,050.3 | 1,008.9 | 489 |
| `145e872` | q8 prefill through the f32 mirror | 1,052.6 | 1,009.9 | 489 |

The two real regressions were **graphs** (it replaced attn-mma's fp16 Q5_K prefill GEMM with one decode GEMV per row: -22%) and **feat/bench** (it forced the q8 prefill through per-row decode attention: another -57%). Both are fixed. The remaining gap to the 1,260/1,200 targets is the **wmma2 correctness fix**: WP17 computed half of every GEMM. The only commit that reaches the targets is `5945a08`, and it emits 122271/103687 for the 512/8K last ids instead of 489/292. With correct tokens the ceiling on this kernel is **~1,050 at 512 and ~1,010 at 8K**.

## What each merge contributed

| branch | effect in the merged engine | measured |
| --- | --- | --- |
| `feat/attn-mma` | matrix-unit chunk attention | 8K attention stage 3.58 s -> 0.84 s; used by both f32 and q8 prefill now |
| `feat/graphs` | KV stride fix, graph, streamed GEMV/Q5_K | decode +4.8% at 0/8K, +3.7% at 32K, +2.8% at 100K vs WP18; its Q5_K prefill regression is fixed in `4b02c75` |
| `feat/wmma2` | full-GEMM accumulator fix | correctness: 512/8K last ids 489/292. It is the ~17% prefill gap to the old inflated numbers |
| `feat/stream` | 1-row-per-warp GEMV, Q5_K GEMV | contained in graphs; part of the decode gain |
| `feat/cache` | SSM+KV prefix snapshots | not in a cold bench; exact-repeat TTFT 1801 -> 7.6 ms, shared prefix 1.88-3.66x |
| `feat/bench` | q8/q4 chunked prefill + bench CLI | its per-row q8 prefill was the 57% regression; the mirror covers <=40K and the chunk kernel reads the q8 cache above that |

## Validation

- 20-id acceptance identical on CPU, GPU f32, GPU q8 and GPU q8 `--graph`:
  `11751 13 198 760 6511 314 14898 369 21047 13 198 760 6511 314 17163 369 23327 13 198 760`.
- Prefill last ids are the correct ones: 512 -> 489, 8K -> 292, 32K -> 868.
- `hipml parity`: 46 passed, 0 failed, 1 skipped. `hipml parity --deep`: 48 passed, 0 failed, 1 skipped.
- `cargo test --workspace`: all green. `cargo check --features hip`: clean.
- Every hipml cell generated the same 33-token sequence across reps; prompt ids are shared byte-for-byte with llama.cpp.

## Completeness

All planned cells are present: 5 prompts x depths 0 and 8192 at 3 reps, and `code_edit` x depths 32768 and 100000 at 1 rep. No ERROR rows.

## Reps and caveats

- **One-rep cells:** `code_edit@32768`, `code_edit@100000`. These are marked `1` in the reps column; treat the range as absent, not as zero variance.
- The earlier incomplete table, every `ERROR: missing` cell, and all reruns are documented in `bench/RUNLOG.md`.
- `hipml commit` is the Rust commit used; the `(dirty worktree)` flag on that run refers to bench-script edits made after the Rust commit, not to a different binary. The binary SHA-256 is recorded above.
- Same model file, same box, sequential engine runs. Both engines use quantized KV at every depth: hipml `q8` and llama.cpp `-ctk q8_0 -ctv q8_0`. The formats are the same 8-bit class but not byte-identical, so this is a same-precision-class comparison, not a same-format one.
- At 100K the q8 KV is about 3.2 GiB for hipml and 3.5 GiB for llama.cpp. The 100K rows report peak VRAM and whether llama.cpp spilled; the run is only accepted if the server logged no allocation failure or CPU offload.
- The earlier f16/f32 sensitivity runs are preserved under `bench/sensitivity/` and are not part of the head-to-head.
- 100K cells are real cold prefill. If a cell shows `1` rep, the run was too slow to repeat inside the lock window; it is not cache-prefilled.
- The harness cannot control: GPU clocks/thermal state, driver scheduling, other processes on the box outside the lock, llama.cpp's own thread and graph choices, or the difference between hipml's argmax-in-prefill TTFT and llama-server's prompt-only timing.


## WP25: 100K prefill (q8 KV tiles read in the chunk kernel)

The 100K cliff was the f32 KV mirror: 12.2 GiB beside 17 GiB of weights does not
fit, so every 100K prompt fell back to one decode-attention call per token
(116 tok/s). `attn_chunk_wmma_f16` now stages each KV tile through `aw_ld4`,
which turns a 4-value group of a q8_0 (34 B) or q4_0 (18 B) block into the same
f16 shared tile the mirror filled. With that in place the mirror stopped paying
for itself: 8K prefills at 998 tok/s without it (993 with it) and 32K is level
(867 vs 871), so quantized engines allocate no mirror at any depth. Prefill and
decode now read the same q8 cache.

Measured on the same card, `generate --device --kv q8`, 33-token decode window,
code_edit, 3 cold reps at 0/8K and one at 32K/100K:

| depth | prefill (before -> after) | TTFT (before -> after) | decode (before -> after) | peak VRAM (after) |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 940 -> 939 | 715 -> 716 ms | 29.8 -> 29.8 | 17,777 MiB |
| 8K | 995 -> 998 | 8.23 -> 8.21 s | 28.7 -> 28.7 | 18,086 MiB |
| 32K | 864 -> 867 | 37.9 -> 37.8 s | 26.2 -> 26.1 | 18,807 MiB |
| 100K | **116 -> 637** | **860.8 -> 156.9 s** | 21.2 -> 21.2 | 21,384 MiB* |

\* with the extra `--stage-report` engine alive; the plain run peaks at
20,995 MiB. The mirror removal also takes 1.1 GiB off 8K (19,202 -> 18,086) and
4.0 GiB off 32K (22,871 -> 18,807).

### Effective KV bandwidth of the q8 chunk attention

Traffic per (KV head, 32-query tile, d half) is one full K row and half a V row,
and both d halves read K: `3 * row_bytes` = 816 B per staged key position.
Summed over query blocks the kernel re-reads the prefix 193x (8K), 769x (32K)
and 2,345x (100K) the 0.29/1.14/3.48 GB unique cache.

| depth | staged KV | `seq_attn` stage | stage GB/s | kernel GB/s* |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 55.0 GB | 839 ms | 66 | 161 |
| 32K | 877 GB | 8,192 ms | 107 | 141 |
| 100K | 8,163 GB | 65.6 s | 124 | 137 |

\* subtracting the per-token front/o-projection cost measured at depth 0
(61 us/token). 137-161 GB/s is 21-25% of the card's 640 GB/s, so the deep chunk
attention is not HBM-bound: the prefix reads hit L2, and the kernel runs at
~50 TFLOPS (29% of the measured 171 TFLOPS matrix-unit ceiling).
