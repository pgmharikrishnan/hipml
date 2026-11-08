# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33. Depths 0 and 8192 cover all five prompts at 3 reps; 32768 and 100000 use `code_edit` at 1 rep, marked in the reps column. Both engines use q8-class KV at every depth.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `c2ad7d77c21627e3daa9ad679cb344cbaec191c8` (dirty worktree) |
| hipml binary sha256 | `bc724a21a0d611f2704f204dd6c2392c268fa1c05870711a9a0fa78d5969274a` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | q8_0 |
| hipml KV type | q8 |
| generated | 2026-11-08T11:02:41.958358+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | KV | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB | spilled |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| code_edit | 0 | 672 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| code_edit | 0 | 672 | llama | - | 0 | ERROR: missing | - | - | - | - |
| code_edit | 8192 | 8192 | hipml | q8 | 1 | 13,125 | 624 | 27.5 | 18.14 | - |
| code_edit | 8192 | 8192 | llama | - | 0 | ERROR: missing | - | - | - | - |
| code_edit | 32768 | 32768 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| code_edit | 32768 | 32768 | llama | - | 0 | ERROR: missing | - | - | - | - |
| code_edit | 100000 | 100000 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| code_edit | 100000 | 100000 | llama | - | 0 | ERROR: missing | - | - | - | - |
| json_tool_call | 0 | 300 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| json_tool_call | 0 | 300 | llama | - | 0 | ERROR: missing | - | - | - | - |
| json_tool_call | 8192 | 8192 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| json_tool_call | 8192 | 8192 | llama | - | 0 | ERROR: missing | - | - | - | - |
| prose_explanation | 0 | 373 | hipml | q8 | 1 | 518 | 719 | 28.5 | 17.87 | - |
| prose_explanation | 0 | 373 | llama | - | 0 | ERROR: missing | - | - | - | - |
| prose_explanation | 8192 | 8192 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| prose_explanation | 8192 | 8192 | llama | - | 0 | ERROR: missing | - | - | - | - |
| longdoc_qa | 0 | 7694 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| longdoc_qa | 0 | 7694 | llama | - | 0 | ERROR: missing | - | - | - | - |
| longdoc_qa | 8192 | 8192 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| longdoc_qa | 8192 | 8192 | llama | - | 0 | ERROR: missing | - | - | - | - |
| copy_200_words | 0 | 285 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| copy_200_words | 0 | 285 | llama | - | 0 | ERROR: missing | - | - | - | - |
| copy_200_words | 8192 | 8192 | hipml | - | 0 | ERROR: missing | - | - | - | - |
| copy_200_words | 8192 | 8192 | llama | - | 0 | ERROR: missing | - | - | - | - |

## Gap (hipml minus llama.cpp)

Positive means hipml is faster. Each row is its own prompt class; no number is averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | - | - | - |
| code_edit | 8192 | - | - | - |
| code_edit | 32768 | - | - | - |
| code_edit | 100000 | - | - | - |
| json_tool_call | 0 | - | - | - |
| json_tool_call | 8192 | - | - | - |
| prose_explanation | 0 | - | - | - |
| prose_explanation | 8192 | - | - | - |
| longdoc_qa | 0 | - | - | - |
| longdoc_qa | 8192 | - | - | - |
| copy_200_words | 0 | - | - | - |
| copy_200_words | 8192 | - | - | - |

## Headline gaps


## Completeness

| prompt | depth | missing engine | reason |
| --- | ---: | --- | --- |
| code_edit | 0 | hipml | missing |
| code_edit | 0 | llama | missing |
| code_edit | 8192 | llama | missing |
| code_edit | 32768 | hipml | missing |
| code_edit | 32768 | llama | missing |
| code_edit | 100000 | hipml | missing |
| code_edit | 100000 | llama | missing |
| json_tool_call | 0 | hipml | missing |
| json_tool_call | 0 | llama | missing |
| json_tool_call | 8192 | hipml | missing |
| json_tool_call | 8192 | llama | missing |
| prose_explanation | 0 | llama | missing |
| prose_explanation | 8192 | hipml | missing |
| prose_explanation | 8192 | llama | missing |
| longdoc_qa | 0 | hipml | missing |
| longdoc_qa | 0 | llama | missing |
| longdoc_qa | 8192 | hipml | missing |
| longdoc_qa | 8192 | llama | missing |
| copy_200_words | 0 | hipml | missing |
| copy_200_words | 0 | llama | missing |
| copy_200_words | 8192 | hipml | missing |
| copy_200_words | 8192 | llama | missing |

## Reps and caveats

- **One-rep cells:** `code_edit@8192`, `prose_explanation@0`. These are marked `1` in the reps column; treat the range as absent, not as zero variance.
- Same model file, same box, sequential engine runs. Both engines use quantized KV at every depth: hipml `q8` and llama.cpp `-ctk q8_0 -ctv q8_0`. The formats are the same 8-bit class but not byte-identical, so this is a same-precision-class comparison, not a same-format one.
- At 100K the q8 KV is about 3.2 GiB for hipml and 3.5 GiB for llama.cpp. The 100K rows report peak VRAM and whether llama.cpp spilled; the run is only accepted if the server logged no allocation failure or CPU offload.
- The earlier f16/f32 sensitivity runs are preserved under `bench/sensitivity/` and are not part of the head-to-head.
- 100K cells are real cold prefill. If a cell shows `1` rep, the run was too slow to repeat inside the lock window; it is not cache-prefilled.
- The harness cannot control: GPU clocks/thermal state, driver scheduling, other processes on the box outside the lock, llama.cpp's own thread and graph choices, or the difference between hipml's argmax-in-prefill TTFT and llama-server's prompt-only timing.

