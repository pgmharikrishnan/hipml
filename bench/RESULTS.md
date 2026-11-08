# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33. Depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in the reps column. Both engines use q8-class KV at every depth.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `4c8bc720d7aeed46d0878abc007f7100bab28828` (dirty worktree) |
| hipml binary sha256 | `94cedd9e8f5c2240f91ae5fce991a393a7c4831cef31efb1a26d2ef21d4e60c7` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | q8_0 |
| hipml KV type | q8 |
| generated | 2026-11-08T13:43:02.332864+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| code_edit | 0 | 672 | hipml | q8 | 3 | 1,711 (1,698-1,721) | 393 (390-396) | 29.8 (29.8-29.9) | 17.03 | - |
| code_edit | 0 | 672 | llama | q8_0 | 3 | 681 (671-800) | 1,022 (857-1,028) | 30.6 (30.5-30.6) | 16.22 | no |
| code_edit | 8192 | 8192 | hipml | q8 | 3 | 23,081 (22,968-23,111) | 355 (354-357) | 28.8 (28.8-28.8) | 17.33 | - |
| code_edit | 8192 | 8192 | llama | q8_0 | 3 | 6,928 (6,900-6,996) | 1,185 (1,174-1,189) | 29.8 (29.7-29.8) | 16.47 | no |
| code_edit | 32768 | 32768 | hipml | q8 | 1 | 143,039 | 229 | 26.2 | 17.99 | - |
| code_edit | 32768 | 32768 | llama | q8_0 | 1 | 30,705 | 1,068 | 27.7 | 17.38 | no |
| code_edit | 100000 | 100000 | hipml | q8 | 1 | 877,847 | 114 | 21.2 | 20.17 | - |
| code_edit | 100000 | 100000 | llama | q8_0 | 1 | 120,953 | 827 | 22.0 | 19.88 | no |
| json_tool_call | 0 | 300 | hipml | q8 | 3 | 816 (814-837) | 368 (358-369) | 29.9 (29.8-29.9) | 16.97 | - |
| json_tool_call | 0 | 300 | llama | q8_0 | 3 | 375 (351-506) | 842 (612-858) | 30.5 (30.5-30.5) | 16.21 | no |
| json_tool_call | 8192 | 8192 | hipml | q8 | 3 | 23,010 (22,922-23,122) | 356 (354-357) | 28.8 (28.8-28.8) | 17.33 | - |
| json_tool_call | 8192 | 8192 | llama | q8_0 | 3 | 6,906 (6,905-7,005) | 1,187 (1,172-1,191) | 29.8 (29.8-29.8) | 16.47 | no |
| prose_explanation | 0 | 373 | hipml | q8 | 3 | 966 (963-979) | 386 (381-388) | 29.9 (29.7-29.9) | 16.97 | - |
| prose_explanation | 0 | 373 | llama | q8_0 | 3 | 416 (400-539) | 948 (714-963) | 30.5 (30.5-30.6) | 16.21 | no |
| prose_explanation | 8192 | 8192 | hipml | q8 | 3 | 23,041 (23,010-23,168) | 356 (354-356) | 28.8 (28.6-28.8) | 17.33 | - |
| prose_explanation | 8192 | 8192 | llama | q8_0 | 3 | 6,924 (6,885-6,995) | 1,186 (1,174-1,190) | 29.8 (29.7-29.9) | 16.47 | no |
| longdoc_qa | 0 | 7694 | hipml | q8 | 3 | 21,617 (21,438-21,670) | 356 (355-359) | 28.9 (28.8-28.9) | 17.31 | - |
| longdoc_qa | 0 | 7694 | llama | q8_0 | 3 | 6,595 (6,567-6,644) | 1,171 (1,161-1,176) | 29.8 (29.8-29.9) | 16.46 | no |
| longdoc_qa | 8192 | 8192 | hipml | q8 | 3 | 22,998 (22,962-23,206) | 356 (353-357) | 28.8 (28.8-28.8) | 17.33 | - |
| longdoc_qa | 8192 | 8192 | llama | q8_0 | 3 | 6,918 (6,899-6,997) | 1,186 (1,174-1,191) | 29.7 (29.7-29.8) | 16.47 | no |
| copy_200_words | 0 | 285 | hipml | q8 | 3 | 791 (788-792) | 360 (360-362) | 29.8 (29.7-29.8) | 16.97 | - |
| copy_200_words | 0 | 285 | llama | q8_0 | 3 | 350 (334-458) | 869 (624-883) | 30.5 (30.4-30.5) | 16.21 | no |
| copy_200_words | 8192 | 8192 | hipml | q8 | 3 | 23,074 (23,022-23,394) | 355 (350-356) | 28.7 (28.7-28.8) | 17.33 | - |
| copy_200_words | 8192 | 8192 | llama | q8_0 | 3 | 6,922 (6,901-6,992) | 1,187 (1,174-1,191) | 29.8 (29.7-29.8) | 16.47 | no |

## Merged engine vs the WP18 like-for-like table

WP18 (`bd6d18f`, the `feat/bench` tree) is the reference table: same model, same prompts, same 33 generated tokens, same q8-class KV. The merged tree is `4c8bc720d7ae`. Delta is `merged / WP18 - 1`; for TTFT a negative delta is better, for throughput a positive delta is better.

| prompt | depth | WP18 TTFT ms | merged TTFT ms | Δ TTFT | WP18 prefill | merged prefill | Δ prefill | WP18 decode | merged decode | Δ decode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 915 | 1,711 | +87.0% | 734.7 | 392.8 | -46.5% | 28.45 | 29.84 | +4.9% |
| json_tool_call | 0 | 450 | 816 | +81.1% | 666.0 | 367.8 | -44.8% | 28.49 | 29.87 | +4.8% |
| prose_explanation | 0 | 515 | 966 | +87.4% | 723.7 | 386.1 | -46.6% | 28.47 | 29.86 | +4.9% |
| longdoc_qa | 0 | 12,349 | 21,617 | +75.1% | 623.1 | 355.9 | -42.9% | 27.55 | 28.87 | +4.8% |
| copy_200_words | 0 | 438 | 791 | +80.6% | 650.6 | 360.2 | -44.6% | 28.41 | 29.78 | +4.8% |
| code_edit | 8192 | 13,259 | 23,081 | +74.1% | 617.8 | 354.9 | -42.6% | 27.48 | 28.80 | +4.8% |
| json_tool_call | 8192 | 13,250 | 23,010 | +73.7% | 618.2 | 356.0 | -42.4% | 27.46 | 28.80 | +4.9% |
| prose_explanation | 8192 | 13,272 | 23,041 | +73.6% | 617.3 | 355.5 | -42.4% | 27.45 | 28.80 | +4.9% |
| longdoc_qa | 8192 | 13,260 | 22,998 | +73.4% | 617.8 | 356.2 | -42.3% | 27.45 | 28.80 | +4.9% |
| copy_200_words | 8192 | 13,260 | 23,074 | +74.0% | 617.8 | 355.0 | -42.5% | 27.48 | 28.71 | +4.5% |
| code_edit | 32768 | 101,704 | 143,039 | +40.6% | 322.2 | 229.1 | -28.9% | 25.25 | 26.20 | +3.7% |
| code_edit | 100000 | 733,577 | 877,847 | +19.7% | 136.3 | 113.9 | -16.4% | 20.61 | 21.18 | +2.8% |

Every merged cell is the code path the merge ships; the WP18 reference used the WP17 prefill GEMM (half of each GEMM's MACs were dead-code-eliminated) and the pre-fix KV stride. The prefill delta is therefore mostly the cost of doing the full GEMM, and the decode delta is the gain from the stride fix and the streamed GEMV/Q5_K path. No number is averaged across prompt classes.

## Gap (hipml minus llama.cpp)

Positive means hipml is faster; negative means llama.cpp is faster. Each row is its own prompt class; no number is averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | -61.6% | -2.4% | 0.40x |
| code_edit | 8192 | -70.1% | -3.2% | 0.30x |
| code_edit | 32768 | -78.6% | -5.5% | 0.21x |
| code_edit | 100000 | -86.2% | -3.9% | 0.14x |
| json_tool_call | 0 | -56.3% | -1.9% | 0.46x |
| json_tool_call | 8192 | -70.0% | -3.5% | 0.30x |
| prose_explanation | 0 | -59.2% | -2.3% | 0.43x |
| prose_explanation | 8192 | -70.0% | -3.5% | 0.30x |
| longdoc_qa | 0 | -69.6% | -3.2% | 0.31x |
| longdoc_qa | 8192 | -70.0% | -3.2% | 0.30x |
| copy_200_words | 0 | -58.5% | -2.3% | 0.44x |
| copy_200_words | 8192 | -70.1% | -3.6% | 0.30x |

## Headline gaps

- **prefill**: hipml did not win any cell; llama.cpp fastest at `code_edit@100000` (-86.2%, reps 1/1).
- **decode**: hipml did not win any cell; llama.cpp fastest at `code_edit@32768` (-5.5%, reps 1/1).
- **ttft**: hipml did not win any cell; llama.cpp fastest at `code_edit@100000` (-86.2%, reps 1/1).

## llama.cpp warm pass (cache_prompt=true)

Same ids immediately after the cold reps. A near-1 `prompt_n` and a large `cache_n` confirm the prefix was reused; this is a cache-hit TTFT, not a prefill number.

| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 97.7 | 41.3 | 668 | 4 |
| code_edit | 8192 | 123 | 41.3 | 8188 | 4 |
| code_edit | 32768 | 126 | 37.8 | 32764 | 4 |
| code_edit | 100000 | 189 | 28.2 | 99996 | 4 |
| json_tool_call | 0 | 113 | 42.6 | 296 | 4 |
| json_tool_call | 8192 | 114 | 41.2 | 8188 | 4 |
| prose_explanation | 0 | 109 | 42.4 | 369 | 4 |
| prose_explanation | 8192 | 111 | 41.5 | 8188 | 4 |
| longdoc_qa | 0 | 117 | 40.3 | 7690 | 4 |
| longdoc_qa | 8192 | 111 | 41.3 | 8188 | 4 |
| copy_200_words | 0 | 115 | 42.8 | 281 | 4 |
| copy_200_words | 8192 | 127 | 40.4 | 8188 | 4 |

## hipml prefill stage breakdown

One extra synchronised run per cell; the per-stage values include a stream sync each, so `total_wall` is larger than the table TTFT. Reported as a diagnostic of where prefill time goes.

| prompt | depth | total_wall | embed | elementwise | gemm_attn | gemm_ssm | gemm_ffn | final | seq_ssm | seq_attn | stage_sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 1,732 | 1 | 22 | 33 | 91 | 528 | 4 | 166 | 885 | 1,730 |
| json_tool_call | 0 | 826 | 1 | 10 | 18 | 51 | 276 | 2 | 77 | 391 | 826 |
| prose_explanation | 0 | 969 | 1 | 11 | 19 | 52 | 304 | 2 | 92 | 488 | 969 |
| longdoc_qa | 0 | 21,836 | 11 | 196 | 325 | 894 | 5,492 | 28 | 1,801 | 13,087 | 21,835 |
| copy_200_words | 0 | 809 | 1 | 11 | 19 | 50 | 273 | 2 | 74 | 381 | 809 |


## What each merge contributed

| branch | what it changed in the merged engine | measured effect |
| --- | --- | --- |
| `feat/attn-mma` (`5945a08`) | matrix-unit chunk attention for the f32 prefill path; split-K decode evidence | not on the q8 like-for-like prefill path; on the f32 path `gpu-bench --prefill` is **857 tok/s at 512** and **828 tok/s at 8K** in the merged tree |
| `feat/graphs` (`ddac7e3`) | quantized-KV decode uses `dk.seq` as the cache stride, not the position count; reusable token graph; streamed GEMV jobs | every deep decode cell now reads the correct KV rows; decode **+4.8%** at 0/8K, **+3.7%** at 32K, **+2.8%** at 100K vs WP18 |
| `feat/wmma2` (`cf41cbe`) | prefill accumulator-index fix (WP17 computed half of every GEMM) and kernel-config corrections | this is why prefill falls from WP18's 618 to **355 tok/s** at 8K: WP18's number was half a GEMM. The full-GEMM f32 path is 828 tok/s |
| `feat/stream` (`fe0b644`) | 1-row-per-warp GEMV, Q5_K GEMV, streamed weight reads | already contained in `feat/graphs`; Q5_K no longer uses an f32 copy; part of the decode gain above |
| `feat/cache` (`66bba78`) | SSM-state + KV prefix snapshots and an LRU prefix cache | not exercised by a cold like-for-like run; its own exact-repeat TTFT is 1801 -> 7.6 ms and shared-prefix TTFT 1.88-3.66x |
| `feat/bench` (`2e0f642`) | q8/q4 chunked prefill and the `--json/--reps/--stage-report` bench CLI | required to run this comparison at all; it also exposed a short-final-tile NaN in the decode attention kernel, fixed in `0bd5403` |

## Validation

- 20-id acceptance, CPU oracle and GPU eager: `11751 13 198 760 6511 314 14898 369 21047 13 198 760 6511 314 17163 369 23327 13 198 760`. GPU q8 and `--graph` produce the same string.
- `hipml parity`: 46 passed, 0 failed, 1 skipped. `hipml parity --deep`: 48 passed, 0 failed, 1 skipped (includes `attn_decode_kv_q8_hd256_s100000`).
- `cargo test --workspace`: all test binaries pass on the host. `cargo check --features hip`: clean.
- Every hipml cell's three reps generated the same 33-token id sequence; the bench's prompt ids are shared byte-for-byte with llama.cpp.

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

