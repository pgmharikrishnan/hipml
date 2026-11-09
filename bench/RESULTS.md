# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33. Depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in the reps column. Both engines use q8-class KV at every depth.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `6c4bc14c4d5ab666f7fe4586dc4eb80b22e39045` (dirty worktree) |
| hipml binary sha256 | `4f744daf1604c83699121fb4df9cb8fe269b60da627d1bf8d82049225f241708` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | q8_0 |
| hipml KV type | q8 |
| generated | 2026-11-09T05:15:33.843940+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| code_edit | 0 | 672 | hipml | q8 | 3 | 569 (567-577) | 1,181 (1,164-1,185) | 29.2 (29.2-29.2) | 17.60 | - |
| code_edit | 0 | 672 | llama | q8_0 | 3 | 688 (686-817) | 1,015 (839-1,019) | 30.2 (29.9-30.2) | 16.22 | no |
| code_edit | 8192 | 8192 | hipml | q8 | 3 | 6,636 (6,618-6,656) | 1,235 (1,231-1,238) | 28.7 (28.6-28.7) | 18.73 | - |
| code_edit | 8192 | 8192 | llama | q8_0 | 3 | 6,915 (6,888-6,979) | 1,187 (1,177-1,192) | 29.4 (29.4-29.5) | 16.47 | no |
| code_edit | 32768 | 32768 | hipml | q8 | 1 | 31,227 | 1,049 | 27.0 | 19.31 | - |
| code_edit | 32768 | 32768 | llama | q8_0 | 1 | 30,718 | 1,067 | 27.4 | 17.38 | no |
| code_edit | 100000 | 100000 | hipml | q8 | 1 | 135,529 | 738 | 23.4 | 21.55 | - |
| code_edit | 100000 | 100000 | llama | q8_0 | 1 | 120,837 | 828 | 21.8 | 19.88 | no |
| json_tool_call | 0 | 300 | hipml | q8 | 3 | 338 (333-339) | 888 (884-901) | 29.2 (29.2-29.2) | 17.34 | - |
| json_tool_call | 0 | 300 | llama | q8_0 | 3 | 368 (358-507) | 841 (611-852) | 30.2 (30.1-30.2) | 16.21 | no |
| json_tool_call | 8192 | 8192 | hipml | q8 | 3 | 6,633 (6,632-6,661) | 1,235 (1,230-1,235) | 28.6 (28.6-28.7) | 18.73 | - |
| json_tool_call | 8192 | 8192 | llama | q8_0 | 3 | 6,943 (6,913-7,006) | 1,184 (1,172-1,188) | 29.4 (29.3-29.5) | 16.47 | no |
| prose_explanation | 0 | 373 | hipml | q8 | 3 | 362 (357-363) | 1,029 (1,026-1,045) | 29.3 (29.2-29.3) | 17.43 | - |
| prose_explanation | 0 | 373 | llama | q8_0 | 3 | 417 (395-541) | 944 (724-948) | 30.2 (30.1-30.3) | 16.21 | no |
| prose_explanation | 8192 | 8192 | hipml | q8 | 3 | 6,638 (6,634-6,648) | 1,234 (1,232-1,235) | 28.7 (28.6-28.7) | 18.73 | - |
| prose_explanation | 8192 | 8192 | llama | q8_0 | 3 | 6,933 (6,890-7,004) | 1,185 (1,173-1,190) | 29.5 (29.4-29.5) | 16.47 | no |
| longdoc_qa | 0 | 7694 | hipml | q8 | 3 | 6,315 (6,293-6,329) | 1,218 (1,216-1,223) | 28.7 (28.7-28.8) | 18.71 | - |
| longdoc_qa | 0 | 7694 | llama | q8_0 | 3 | 6,533 (6,482-6,578) | 1,181 (1,173-1,188) | 29.5 (29.4-29.5) | 16.46 | no |
| longdoc_qa | 8192 | 8192 | hipml | q8 | 3 | 6,639 (6,635-6,650) | 1,234 (1,232-1,235) | 28.7 (28.6-28.7) | 18.80 | - |
| longdoc_qa | 8192 | 8192 | llama | q8_0 | 3 | 6,939 (6,914-7,017) | 1,184 (1,172-1,188) | 29.4 (29.4-29.5) | 16.47 | no |
| copy_200_words | 0 | 285 | hipml | q8 | 3 | 335 (330-340) | 850 (839-863) | 29.2 (29.2-29.2) | 17.32 | - |
| copy_200_words | 0 | 285 | llama | q8_0 | 3 | 336 (332-474) | 860 (622-878) | 30.2 (30.1-30.2) | 16.21 | no |
| copy_200_words | 8192 | 8192 | hipml | q8 | 3 | 6,644 (6,629-6,646) | 1,233 (1,233-1,236) | 28.6 (28.6-28.7) | 18.73 | - |
| copy_200_words | 8192 | 8192 | llama | q8_0 | 3 | 6,921 (6,906-7,008) | 1,184 (1,172-1,188) | 29.4 (29.3-29.5) | 16.47 | no |

## Gap (hipml minus llama.cpp)

Positive means hipml is faster; negative means llama.cpp is faster. Each row is its own prompt class; no number is averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | +16.4% | -3.1% | 1.21x |
| code_edit | 8192 | +4.0% | -2.5% | 1.04x |
| code_edit | 32768 | -1.7% | -1.5% | 0.98x |
| code_edit | 100000 | -10.9% | +7.3% | 0.89x |
| json_tool_call | 0 | +5.5% | -3.4% | 1.09x |
| json_tool_call | 8192 | +4.3% | -2.7% | 1.05x |
| prose_explanation | 0 | +9.0% | -3.0% | 1.15x |
| prose_explanation | 8192 | +4.1% | -2.6% | 1.04x |
| longdoc_qa | 0 | +3.2% | -2.6% | 1.03x |
| longdoc_qa | 8192 | +4.2% | -2.5% | 1.05x |
| copy_200_words | 0 | -1.3% | -3.3% | 1.00x |
| copy_200_words | 8192 | +4.1% | -2.6% | 1.04x |

## Headline gaps

- **prefill**: hipml fastest at `code_edit@0` (+16.4%, reps 3/3); llama.cpp fastest at `code_edit@100000` (-10.9%, reps 1/1).
- **decode**: hipml fastest at `code_edit@100000` (+7.3%, reps 1/1); llama.cpp fastest at `json_tool_call@0` (-3.4%, reps 3/3).
- **ttft**: hipml fastest at `code_edit@0` (+20.9%, reps 3/3); llama.cpp fastest at `code_edit@100000` (-10.8%, reps 1/1).

## llama.cpp warm pass (cache_prompt=true)

Same ids immediately after the cold reps. A near-1 `prompt_n` and a large `cache_n` confirm the prefix was reused; this is a cache-hit TTFT, not a prefill number.

| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 118 | 41.5 | 668 | 4 |
| code_edit | 8192 | 117 | 39.3 | 8188 | 4 |
| code_edit | 32768 | 120 | 36.0 | 32764 | 4 |
| code_edit | 100000 | 191 | 27.6 | 99996 | 4 |
| json_tool_call | 0 | 115 | 39.2 | 296 | 4 |
| json_tool_call | 8192 | 124 | 39.3 | 8188 | 4 |
| prose_explanation | 0 | 119 | 40.3 | 369 | 4 |
| prose_explanation | 8192 | 124 | 39.3 | 8188 | 4 |
| longdoc_qa | 0 | 123 | 37.8 | 7690 | 4 |
| longdoc_qa | 8192 | 120 | 39.3 | 8188 | 4 |
| copy_200_words | 0 | 114 | 39.2 | 281 | 4 |
| copy_200_words | 8192 | 109 | 37.6 | 8188 | 4 |

## hipml prefill stage breakdown

One extra synchronised run per cell; the per-stage values include a stream sync each, so `total_wall` is larger than the table TTFT. Reported as a diagnostic of where prefill time goes.

| prompt | depth | total_wall | embed | elementwise | gemm_attn | gemm_ssm | gemm_ffn | final | seq_ssm | seq_attn | stage_sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 586 | 1 | 19 | 23 | 76 | 297 | 2 | 125 | 43 | 586 |
| json_tool_call | 0 | 347 | 1 | 12 | 16 | 48 | 187 | 2 | 63 | 20 | 347 |
| prose_explanation | 0 | 375 | 1 | 13 | 17 | 50 | 195 | 2 | 75 | 23 | 374 |
| longdoc_qa | 0 | 6,443 | 10 | 186 | 209 | 723 | 3,091 | 7 | 1,430 | 786 | 6,443 |
| copy_200_words | 0 | 347 | 1 | 13 | 16 | 48 | 187 | 2 | 62 | 19 | 347 |

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

