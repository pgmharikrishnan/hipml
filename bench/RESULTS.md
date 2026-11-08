# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33. Depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in the reps column. Both engines use q8-class KV at every depth.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `bd6d18fa77b6441c8c65c0b1f3f98ac076c81100` (dirty worktree) |
| hipml binary sha256 | `9eb31029af894915bf7ca34437a8e35910565f2ec9fcb5d11c212c0fa225d76d` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | q8_0 |
| hipml KV type | q8 |
| generated | 2026-11-08T12:00:09.360933+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| code_edit | 0 | 672 | hipml | q8 | 3 | 915 (914-924) | 735 (727-735) | 28.5 (28.4-28.5) | 17.98 | - |
| code_edit | 0 | 672 | llama | q8_0 | 3 | 677 (675-779) | 1,030 (864-1,030) | 30.6 (30.5-30.7) | 16.22 | no |
| code_edit | 8192 | 8192 | hipml | q8 | 3 | 13,259 (13,241-13,261) | 618 (618-619) | 27.5 (27.4-27.5) | 18.29 | - |
| code_edit | 8192 | 8192 | llama | q8_0 | 3 | 6,909 (6,902-6,995) | 1,187 (1,174-1,191) | 29.9 (29.8-29.9) | 16.47 | no |
| code_edit | 32768 | 32768 | hipml | q8 | 1 | 101,704 | 322 | 25.3 | 18.94 | - |
| code_edit | 32768 | 32768 | llama | q8_0 | 1 | 30,656 | 1,070 | 27.8 | 17.38 | no |
| code_edit | 100000 | 100000 | hipml | q8 | 1 | 733,577 | 136 | 20.6 | 21.13 | - |
| code_edit | 100000 | 100000 | llama | q8_0 | 1 | 120,906 | 827 | 22.1 | 19.88 | no |
| json_tool_call | 0 | 300 | hipml | q8 | 3 | 450 (449-456) | 666 (658-668) | 28.5 (28.5-28.5) | 17.92 | - |
| json_tool_call | 0 | 300 | llama | q8_0 | 3 | 353 (349-508) | 851 (610-861) | 30.6 (30.5-30.6) | 16.21 | no |
| json_tool_call | 8192 | 8192 | hipml | q8 | 3 | 13,250 (13,220-13,270) | 618 (617-620) | 27.5 (27.4-27.5) | 18.29 | - |
| json_tool_call | 8192 | 8192 | llama | q8_0 | 3 | 6,924 (6,896-6,992) | 1,187 (1,174-1,191) | 29.8 (29.7-29.9) | 16.47 | no |
| prose_explanation | 0 | 373 | hipml | q8 | 3 | 515 (513-521) | 724 (716-728) | 28.5 (28.5-28.5) | 17.92 | - |
| prose_explanation | 0 | 373 | llama | q8_0 | 3 | 408 (400-534) | 958 (720-969) | 30.5 (30.5-30.6) | 16.21 | no |
| prose_explanation | 8192 | 8192 | hipml | q8 | 3 | 13,272 (13,245-13,272) | 617 (617-619) | 27.4 (27.4-27.5) | 18.29 | - |
| prose_explanation | 8192 | 8192 | llama | q8_0 | 3 | 6,930 (6,905-6,992) | 1,186 (1,174-1,191) | 29.9 (29.8-29.9) | 16.47 | no |
| longdoc_qa | 0 | 7694 | hipml | q8 | 3 | 12,349 (12,271-12,362) | 623 (622-627) | 27.6 (27.5-27.6) | 18.27 | - |
| longdoc_qa | 0 | 7694 | llama | q8_0 | 3 | 6,544 (6,510-6,587) | 1,178 (1,171-1,184) | 29.8 (29.8-29.9) | 16.46 | no |
| longdoc_qa | 8192 | 8192 | hipml | q8 | 3 | 13,260 (13,237-13,260) | 618 (618-619) | 27.5 (27.4-27.5) | 18.29 | - |
| longdoc_qa | 8192 | 8192 | llama | q8_0 | 3 | 6,907 (6,896-6,998) | 1,186 (1,174-1,191) | 29.9 (29.8-29.9) | 16.47 | no |
| copy_200_words | 0 | 285 | hipml | q8 | 3 | 438 (437-447) | 651 (638-653) | 28.4 (28.4-28.4) | 17.92 | - |
| copy_200_words | 0 | 285 | llama | q8_0 | 3 | 345 (330-469) | 866 (629-887) | 30.5 (30.5-30.6) | 16.21 | no |
| copy_200_words | 8192 | 8192 | hipml | q8 | 3 | 13,260 (13,238-13,285) | 618 (617-619) | 27.5 (27.5-27.5) | 18.29 | - |
| copy_200_words | 8192 | 8192 | llama | q8_0 | 3 | 6,913 (6,900-6,988) | 1,187 (1,175-1,191) | 29.9 (29.8-29.9) | 16.47 | no |

## Gap (hipml minus llama.cpp)

Positive means hipml is faster; negative means llama.cpp is faster. Each row is its own prompt class; no number is averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | -28.6% | -6.9% | 0.74x |
| code_edit | 8192 | -48.0% | -8.0% | 0.52x |
| code_edit | 32768 | -69.9% | -9.2% | 0.30x |
| code_edit | 100000 | -83.5% | -6.7% | 0.16x |
| json_tool_call | 0 | -21.8% | -6.8% | 0.78x |
| json_tool_call | 8192 | -47.9% | -7.8% | 0.52x |
| prose_explanation | 0 | -24.5% | -6.8% | 0.79x |
| prose_explanation | 8192 | -48.0% | -8.1% | 0.52x |
| longdoc_qa | 0 | -47.1% | -7.7% | 0.53x |
| longdoc_qa | 8192 | -47.9% | -8.2% | 0.52x |
| copy_200_words | 0 | -24.9% | -6.9% | 0.79x |
| copy_200_words | 8192 | -48.0% | -8.0% | 0.52x |

## Headline gaps

- **prefill**: hipml did not win any cell; llama.cpp fastest at `code_edit@100000` (-83.5%, reps 1/1).
- **decode**: hipml did not win any cell; llama.cpp fastest at `code_edit@32768` (-9.2%, reps 1/1).
- **ttft**: hipml did not win any cell; llama.cpp fastest at `code_edit@100000` (-83.5%, reps 1/1).

## llama.cpp warm pass (cache_prompt=true)

Same ids immediately after the cold reps. A near-1 `prompt_n` and a large `cache_n` confirm the prefix was reused; this is a cache-hit TTFT, not a prefill number.

| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 110 | 42.0 | 668 | 4 |
| code_edit | 8192 | 98.3 | 41.7 | 8188 | 4 |
| code_edit | 32768 | 137 | 36.7 | 32764 | 4 |
| code_edit | 100000 | 194 | 27.4 | 99996 | 4 |
| json_tool_call | 0 | 109 | 42.5 | 296 | 4 |
| json_tool_call | 8192 | 115 | 40.5 | 8188 | 4 |
| prose_explanation | 0 | 115 | 41.8 | 369 | 4 |
| prose_explanation | 8192 | 116 | 40.8 | 8188 | 4 |
| longdoc_qa | 0 | 114 | 40.9 | 7690 | 4 |
| longdoc_qa | 8192 | 112 | 41.3 | 8188 | 4 |
| copy_200_words | 0 | 111 | 43.3 | 281 | 4 |
| copy_200_words | 8192 | 126 | 40.2 | 8188 | 4 |

## hipml prefill stage breakdown

One extra synchronised run per cell; the per-stage values include a stream sync each, so `total_wall` is larger than the table TTFT. Reported as a diagnostic of where prefill time goes.

| prompt | depth | total_wall | embed | elementwise | gemm_attn | gemm_ssm | gemm_ffn | final | seq_ssm | seq_attn | stage_sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 925 | 1 | 22 | 26 | 71 | 301 | 4 | 150 | 348 | 924 |
| json_tool_call | 0 | 454 | 1 | 11 | 15 | 38 | 159 | 2 | 69 | 160 | 454 |
| prose_explanation | 0 | 515 | 1 | 12 | 14 | 39 | 168 | 2 | 83 | 196 | 515 |
| longdoc_qa | 0 | 12,358 | 11 | 211 | 248 | 680 | 2,984 | 28 | 1,620 | 6,575 | 12,357 |
| copy_200_words | 0 | 444 | 1 | 11 | 15 | 38 | 158 | 2 | 67 | 153 | 444 |

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

