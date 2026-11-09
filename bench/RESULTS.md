# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33. Depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in the reps column. Both engines use q8-class KV at every depth.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `fb69b4d0713c8b9be94ba38fa28463c040972b22` |
| hipml binary sha256 | `70a3ef1be5496df2a03fad25868c3de05131444597a4c3b8aea165a4e716fa1a` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | q8_0 |
| hipml KV type | q8 |
| generated | 2026-11-09T04:16:54.501616+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| code_edit | 0 | 672 | hipml | q8 | 3 | 570 (566-579) | 1,178 (1,161-1,188) | 26.9 (26.9-27.1) | 17.60 | - |
| code_edit | 0 | 672 | llama | q8_0 | 3 | 679 (675-837) | 1,022 (808-1,025) | 30.4 (30.3-30.5) | 16.22 | no |
| code_edit | 8192 | 8192 | hipml | q8 | 3 | 6,624 (6,614-6,630) | 1,237 (1,236-1,239) | 27.5 (27.4-27.5) | 18.73 | - |
| code_edit | 8192 | 8192 | llama | q8_0 | 3 | 6,939 (6,891-6,996) | 1,185 (1,174-1,189) | 29.7 (29.7-29.8) | 16.47 | no |
| code_edit | 32768 | 32768 | hipml | q8 | 1 | 31,172 | 1,051 | 26.5 | 19.31 | - |
| code_edit | 32768 | 32768 | llama | q8_0 | 1 | 30,689 | 1,068 | 27.6 | 17.38 | no |
| code_edit | 100000 | 100000 | hipml | q8 | 1 | 135,536 | 738 | 23.5 | 21.49 | - |
| code_edit | 100000 | 100000 | llama | q8_0 | 1 | 120,849 | 828 | 22.0 | 19.88 | no |
| json_tool_call | 0 | 300 | hipml | q8 | 3 | 337 (332-347) | 890 (866-903) | 27.3 (27.3-27.3) | 17.34 | - |
| json_tool_call | 0 | 300 | llama | q8_0 | 3 | 363 (357-499) | 843 (621-857) | 30.5 (30.4-30.5) | 16.21 | no |
| json_tool_call | 8192 | 8192 | hipml | q8 | 3 | 6,652 (6,647-6,653) | 1,231 (1,231-1,233) | 27.4 (27.4-27.5) | 18.80 | - |
| json_tool_call | 8192 | 8192 | llama | q8_0 | 3 | 6,924 (6,900-7,014) | 1,188 (1,171-1,191) | 29.7 (29.6-29.8) | 16.47 | no |
| prose_explanation | 0 | 373 | hipml | q8 | 3 | 363 (358-365) | 1,029 (1,022-1,043) | 26.8 (26.7-26.8) | 17.44 | - |
| prose_explanation | 0 | 373 | llama | q8_0 | 3 | 414 (399-538) | 948 (714-964) | 30.4 (30.4-30.5) | 16.21 | no |
| prose_explanation | 8192 | 8192 | hipml | q8 | 3 | 6,637 (6,632-6,664) | 1,234 (1,229-1,235) | 27.4 (27.4-27.5) | 18.73 | - |
| prose_explanation | 8192 | 8192 | llama | q8_0 | 3 | 6,932 (6,899-7,006) | 1,186 (1,173-1,190) | 29.8 (29.7-29.8) | 16.47 | no |
| longdoc_qa | 0 | 7694 | hipml | q8 | 3 | 6,326 (6,312-6,342) | 1,216 (1,213-1,219) | 27.4 (27.4-27.5) | 18.71 | - |
| longdoc_qa | 0 | 7694 | llama | q8_0 | 3 | 6,582 (6,561-6,648) | 1,173 (1,160-1,177) | 29.7 (29.7-29.8) | 16.46 | no |
| longdoc_qa | 8192 | 8192 | hipml | q8 | 3 | 6,639 (6,638-6,650) | 1,234 (1,232-1,234) | 27.4 (27.4-27.5) | 18.73 | - |
| longdoc_qa | 8192 | 8192 | llama | q8_0 | 3 | 6,908 (6,885-6,986) | 1,186 (1,176-1,190) | 29.7 (29.7-29.7) | 16.47 | no |
| copy_200_words | 0 | 285 | hipml | q8 | 3 | 333 (331-338) | 856 (844-862) | 27.3 (27.3-27.3) | 17.32 | - |
| copy_200_words | 0 | 285 | llama | q8_0 | 3 | 341 (337-481) | 871 (613-886) | 30.4 (30.4-30.5) | 16.21 | no |
| copy_200_words | 8192 | 8192 | hipml | q8 | 3 | 6,640 (6,632-6,641) | 1,234 (1,233-1,235) | 27.4 (27.4-27.4) | 18.80 | - |
| copy_200_words | 8192 | 8192 | llama | q8_0 | 3 | 6,911 (6,901-6,982) | 1,186 (1,174-1,190) | 29.8 (29.6-29.8) | 16.47 | no |

## Gap (hipml minus llama.cpp)

Positive means hipml is faster; negative means llama.cpp is faster. Each row is its own prompt class; no number is averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | +15.3% | -11.5% | 1.19x |
| code_edit | 8192 | +4.4% | -7.5% | 1.05x |
| code_edit | 32768 | -1.6% | -4.1% | 0.98x |
| code_edit | 100000 | -10.9% | +6.7% | 0.89x |
| json_tool_call | 0 | +5.5% | -10.5% | 1.08x |
| json_tool_call | 8192 | +3.7% | -7.6% | 1.04x |
| prose_explanation | 0 | +8.5% | -12.0% | 1.14x |
| prose_explanation | 8192 | +4.1% | -7.8% | 1.04x |
| longdoc_qa | 0 | +3.7% | -7.8% | 1.04x |
| longdoc_qa | 8192 | +4.0% | -7.7% | 1.04x |
| copy_200_words | 0 | -1.7% | -10.0% | 1.02x |
| copy_200_words | 8192 | +4.0% | -7.8% | 1.04x |

## Headline gaps

- **prefill**: hipml fastest at `code_edit@0` (+15.3%, reps 3/3); llama.cpp fastest at `code_edit@100000` (-10.9%, reps 1/1).
- **decode**: hipml fastest at `code_edit@100000` (+6.7%, reps 1/1); llama.cpp fastest at `prose_explanation@0` (-12.0%, reps 3/3).
- **ttft**: hipml fastest at `code_edit@0` (+19.0%, reps 3/3); llama.cpp fastest at `code_edit@100000` (-10.8%, reps 1/1).

## llama.cpp warm pass (cache_prompt=true)

Same ids immediately after the cold reps. A near-1 `prompt_n` and a large `cache_n` confirm the prefix was reused; this is a cache-hit TTFT, not a prefill number.

| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 119 | 41.4 | 668 | 4 |
| code_edit | 8192 | 122 | 41.9 | 8188 | 4 |
| code_edit | 32768 | 125 | 37.6 | 32764 | 4 |
| code_edit | 100000 | 178 | 29.2 | 99996 | 4 |
| json_tool_call | 0 | 117 | 42.6 | 296 | 4 |
| json_tool_call | 8192 | 120 | 42.1 | 8188 | 4 |
| prose_explanation | 0 | 99.5 | 40.6 | 369 | 4 |
| prose_explanation | 8192 | 126 | 40.4 | 8188 | 4 |
| longdoc_qa | 0 | 119 | 41.7 | 7690 | 4 |
| longdoc_qa | 8192 | 111 | 41.6 | 8188 | 4 |
| copy_200_words | 0 | 95.5 | 42.3 | 281 | 4 |
| copy_200_words | 8192 | 116 | 41.7 | 8188 | 4 |

## hipml prefill stage breakdown

One extra synchronised run per cell; the per-stage values include a stream sync each, so `total_wall` is larger than the table TTFT. Reported as a diagnostic of where prefill time goes.

| prompt | depth | total_wall | embed | elementwise | gemm_attn | gemm_ssm | gemm_ffn | final | seq_ssm | seq_attn | stage_sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 586 | 1 | 19 | 23 | 76 | 297 | 2 | 126 | 43 | 586 |
| json_tool_call | 0 | 349 | 1 | 12 | 16 | 48 | 187 | 2 | 63 | 20 | 349 |
| prose_explanation | 0 | 376 | 1 | 13 | 17 | 50 | 195 | 2 | 74 | 23 | 374 |
| longdoc_qa | 0 | 6,435 | 10 | 181 | 209 | 723 | 3,087 | 7 | 1,429 | 788 | 6,434 |
| copy_200_words | 0 | 345 | 1 | 12 | 16 | 48 | 187 | 2 | 61 | 19 | 345 |

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

