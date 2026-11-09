# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33. Depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in the reps column. Both engines use q8-class KV at every depth.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `cef765724eb4f7add7d52fa14c85185878f5560b` (dirty worktree) |
| hipml binary sha256 | `5713f143872ea3c3128376f2fa5ec09aba24cf18babdfa5e48fc4db6331a725f` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | q8_0 |
| hipml KV type | q8 |
| generated | 2026-11-09T07:12:10.572341+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| code_edit | 0 | 672 | hipml | q8 | 3 | 561 (561-572) | 1,198 (1,175-1,199) | 29.2 (29.2-29.3) | 17.62 | - |
| code_edit | 0 | 672 | llama | q8_0 | 3 | 664 (663-811) | 1,014 (845-1,016) | 30.2 (30.0-30.2) | 16.22 | no |
| code_edit | 8192 | 8192 | hipml | q8 | 3 | 6,541 (6,527-6,586) | 1,252 (1,244-1,255) | 28.8 (28.7-28.8) | 18.84 | - |
| code_edit | 8192 | 8192 | llama | q8_0 | 3 | 6,930 (6,912-7,000) | 1,184 (1,173-1,188) | 29.4 (29.4-29.5) | 16.47 | no |
| code_edit | 32768 | 32768 | hipml | q8 | 1 | 28,647 | 1,144 | 27.0 | 19.75 | - |
| code_edit | 32768 | 32768 | llama | q8_0 | 1 | 30,736 | 1,067 | 27.4 | 17.38 | no |
| code_edit | 100000 | 100000 | hipml | q8 | 1 | 107,777 | 928 | 23.4 | 21.93 | - |
| code_edit | 100000 | 100000 | llama | q8_0 | 1 | 120,903 | 827 | 21.8 | 19.88 | no |
| json_tool_call | 0 | 300 | hipml | q8 | 3 | 335 (331-340) | 897 (883-906) | 29.3 (29.2-29.3) | 17.35 | - |
| json_tool_call | 0 | 300 | llama | q8_0 | 3 | 377 (359-504) | 837 (614-849) | 30.1 (30.1-30.2) | 16.21 | no |
| json_tool_call | 8192 | 8192 | hipml | q8 | 3 | 6,637 (6,630-6,640) | 1,234 (1,234-1,236) | 28.6 (28.6-28.6) | 18.84 | - |
| json_tool_call | 8192 | 8192 | llama | q8_0 | 3 | 6,932 (6,890-7,007) | 1,185 (1,172-1,189) | 29.5 (29.3-29.5) | 16.47 | no |
| prose_explanation | 0 | 373 | hipml | q8 | 3 | 362 (356-366) | 1,032 (1,021-1,049) | 29.3 (29.3-29.3) | 17.46 | - |
| prose_explanation | 0 | 373 | llama | q8_0 | 3 | 422 (410-539) | 939 (713-948) | 30.2 (30.0-30.2) | 16.21 | no |
| prose_explanation | 8192 | 8192 | hipml | q8 | 3 | 6,628 (6,627-6,648) | 1,236 (1,232-1,236) | 28.7 (28.6-28.7) | 18.84 | - |
| prose_explanation | 8192 | 8192 | llama | q8_0 | 3 | 6,918 (6,913-7,010) | 1,185 (1,173-1,189) | 29.5 (29.5-29.5) | 16.47 | no |
| longdoc_qa | 0 | 7694 | hipml | q8 | 3 | 6,303 (6,274-6,317) | 1,221 (1,218-1,226) | 28.7 (28.6-28.7) | 18.82 | - |
| longdoc_qa | 0 | 7694 | llama | q8_0 | 3 | 6,593 (6,565-6,646) | 1,172 (1,161-1,176) | 29.4 (29.4-29.5) | 16.46 | no |
| longdoc_qa | 8192 | 8192 | hipml | q8 | 3 | 6,637 (6,636-6,652) | 1,234 (1,232-1,234) | 28.6 (28.6-28.6) | 18.84 | - |
| longdoc_qa | 8192 | 8192 | llama | q8_0 | 3 | 6,941 (6,921-7,007) | 1,184 (1,172-1,188) | 29.4 (29.4-29.5) | 16.47 | no |
| copy_200_words | 0 | 285 | hipml | q8 | 3 | 334 (329-336) | 854 (849-867) | 29.1 (29.1-29.1) | 17.34 | - |
| copy_200_words | 0 | 285 | llama | q8_0 | 3 | 348 (333-476) | 859 (620-874) | 30.1 (30.0-30.2) | 16.21 | no |
| copy_200_words | 8192 | 8192 | hipml | q8 | 3 | 6,641 (6,634-6,641) | 1,234 (1,234-1,235) | 28.6 (28.6-28.7) | 18.84 | - |
| copy_200_words | 8192 | 8192 | llama | q8_0 | 3 | 6,947 (6,906-7,012) | 1,184 (1,172-1,188) | 29.5 (29.3-29.5) | 16.47 | no |

## Gap (hipml minus llama.cpp)

Positive means hipml is faster; negative means llama.cpp is faster. Each row is its own prompt class; no number is averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | +18.1% | -3.1% | 1.18x |
| code_edit | 8192 | +5.7% | -2.2% | 1.06x |
| code_edit | 32768 | +7.2% | -1.2% | 1.07x |
| code_edit | 100000 | +12.1% | +7.4% | 1.12x |
| json_tool_call | 0 | +7.1% | -2.8% | 1.13x |
| json_tool_call | 8192 | +4.2% | -3.0% | 1.04x |
| prose_explanation | 0 | +9.8% | -3.0% | 1.17x |
| prose_explanation | 8192 | +4.3% | -2.7% | 1.04x |
| longdoc_qa | 0 | +4.2% | -2.3% | 1.05x |
| longdoc_qa | 8192 | +4.2% | -2.8% | 1.05x |
| copy_200_words | 0 | -0.6% | -3.5% | 1.04x |
| copy_200_words | 8192 | +4.2% | -2.9% | 1.05x |

## Headline gaps

- **prefill**: hipml fastest at `code_edit@0` (+18.1%, reps 3/3); llama.cpp fastest at `copy_200_words@0` (-0.6%, reps 3/3).
- **decode**: hipml fastest at `code_edit@100000` (+7.4%, reps 1/1); llama.cpp fastest at `copy_200_words@0` (-3.5%, reps 3/3).
- **ttft**: hipml fastest at `code_edit@0` (+18.4%, reps 3/3); llama.cpp did not win any cell for this metric.

## llama.cpp warm pass (cache_prompt=true)

Same ids immediately after the cold reps. A near-1 `prompt_n` and a large `cache_n` confirm the prefix was reused; this is a cache-hit TTFT, not a prefill number.

| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 112 | 40.7 | 668 | 4 |
| code_edit | 8192 | 103 | 40.1 | 8188 | 4 |
| code_edit | 32768 | 118 | 36.3 | 32764 | 4 |
| code_edit | 100000 | 197 | 26.9 | 99996 | 4 |
| json_tool_call | 0 | 104 | 38.9 | 296 | 4 |
| json_tool_call | 8192 | 124 | 38.8 | 8188 | 4 |
| prose_explanation | 0 | 118 | 38.7 | 369 | 4 |
| prose_explanation | 8192 | 129 | 38.9 | 8188 | 4 |
| longdoc_qa | 0 | 130 | 38.8 | 7690 | 4 |
| longdoc_qa | 8192 | 124 | 38.7 | 8188 | 4 |
| copy_200_words | 0 | 121 | 39.9 | 281 | 4 |
| copy_200_words | 8192 | 117 | 38.3 | 8188 | 4 |

## hipml prefill stage breakdown

One extra synchronised run per cell; the per-stage values include a stream sync each, so `total_wall` is larger than the table TTFT. Reported as a diagnostic of where prefill time goes.

| prompt | depth | total_wall | embed | elementwise | gemm_attn | gemm_ssm | gemm_ffn | final | seq_ssm | seq_attn | stage_sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 584 | 1 | 20 | 22 | 76 | 294 | 2 | 125 | 43 | 583 |
| json_tool_call | 0 | 352 | 1 | 14 | 16 | 48 | 187 | 2 | 64 | 20 | 352 |
| prose_explanation | 0 | 377 | 1 | 15 | 17 | 50 | 194 | 2 | 75 | 23 | 376 |
| longdoc_qa | 0 | 6,437 | 10 | 189 | 209 | 723 | 3,086 | 7 | 1,425 | 785 | 6,434 |
| copy_200_words | 0 | 349 | 1 | 14 | 16 | 49 | 187 | 2 | 62 | 19 | 349 |

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

