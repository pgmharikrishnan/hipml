# Like-for-like bench: hipml vs llama.cpp

Five prompt classes, each tokenised once by `hipml tokenize`. Both engines receive the identical token ids from `bench/fixtures/prompts.json`; neither retokenises at run time. Depth 0 is the prompt as-is. Deeper cells are `BOS + real document prefix + prompt`, truncated to the exact token count. Every numeric cell is `median (min-max)` over cold reps; decode is measured over tokens 2..33.

| field | value |
| --- | --- |
| model | `C:\models\Qwen3.8-27B-Q4_K_M.gguf` (17,203,416,064 bytes) |
| GPU | AMD Radeon AI PRO R9700 32.0.32015.7029  |
| host | rtx |
| hipml commit | `267f615f6fe10573a13ba9649fecd3e0a6736152` (dirty worktree) |
| hipml binary sha256 | `95e3a5fa7e6166c4e056188a02f72df24c2958df340ed083d9203949f9b29c74` |
| llama.cpp | version: 0.7.0-dev (build 11820, commit d7725699) |
| llama.cpp build | `11820` commit `d7725699` |
| llama KV type | f32 |
| generated | 2026-11-08T10:28:20.293859+00:00 |

## Results

`TTFT` is lower-is-better. `prefill` and `decode` are higher-is-better. `Peak` is the process's peak dedicated GPU memory (WDDM counter); it is one number for the whole cell, not per rep.

| prompt | depth | tokens | engine | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 672 | hipml | 0 | ERROR: missing | - | - | - |
| code_edit | 0 | 672 | llama | 0 | ERROR: missing | - | - | - |
| code_edit | 8192 | 8192 | hipml | 0 | ERROR: missing | - | - | - |
| code_edit | 8192 | 8192 | llama | 0 | ERROR: missing | - | - | - |
| code_edit | 32768 | 32768 | hipml | 0 | ERROR: missing | - | - | - |
| code_edit | 32768 | 32768 | llama | 0 | ERROR: missing | - | - | - |
| code_edit | 100000 | 100000 | hipml | 0 | ERROR: missing | - | - | - |
| code_edit | 100000 | 100000 | llama | 0 | ERROR: missing | - | - | - |
| json_tool_call | 0 | 300 | hipml | 0 | ERROR: missing | - | - | - |
| json_tool_call | 0 | 300 | llama | 0 | ERROR: missing | - | - | - |
| json_tool_call | 8192 | 8192 | hipml | 0 | ERROR: missing | - | - | - |
| json_tool_call | 8192 | 8192 | llama | 0 | ERROR: missing | - | - | - |
| json_tool_call | 32768 | 32768 | hipml | 0 | ERROR: missing | - | - | - |
| json_tool_call | 32768 | 32768 | llama | 0 | ERROR: missing | - | - | - |
| json_tool_call | 100000 | 100000 | hipml | 0 | ERROR: missing | - | - | - |
| json_tool_call | 100000 | 100000 | llama | 0 | ERROR: missing | - | - | - |
| prose_explanation | 0 | 373 | hipml | 1 | 340 | 1,098 | 26.5 | 17.87 |
| prose_explanation | 0 | 373 | llama | 1 | 568 | 661 | 30.1 | 16.25 |
| prose_explanation | 8192 | 8192 | hipml | 0 | ERROR: missing | - | - | - |
| prose_explanation | 8192 | 8192 | llama | 0 | ERROR: missing | - | - | - |
| prose_explanation | 32768 | 32768 | hipml | 0 | ERROR: missing | - | - | - |
| prose_explanation | 32768 | 32768 | llama | 0 | ERROR: missing | - | - | - |
| prose_explanation | 100000 | 100000 | hipml | 1 | 577,591 | 173 | 12.6 | 30.11 |
| prose_explanation | 100000 | 100000 | llama | 1 | 130,244 | 768 | 6.4 | 28.86 |
| longdoc_qa | 0 | 7694 | hipml | 0 | ERROR: missing | - | - | - |
| longdoc_qa | 0 | 7694 | llama | 0 | ERROR: missing | - | - | - |
| longdoc_qa | 8192 | 8192 | hipml | 0 | ERROR: missing | - | - | - |
| longdoc_qa | 8192 | 8192 | llama | 0 | ERROR: missing | - | - | - |
| longdoc_qa | 32768 | 32768 | hipml | 0 | ERROR: missing | - | - | - |
| longdoc_qa | 32768 | 32768 | llama | 0 | ERROR: missing | - | - | - |
| longdoc_qa | 100000 | 100000 | hipml | 0 | ERROR: missing | - | - | - |
| longdoc_qa | 100000 | 100000 | llama | 0 | ERROR: missing | - | - | - |
| copy_200_words | 0 | 285 | hipml | 0 | ERROR: missing | - | - | - |
| copy_200_words | 0 | 285 | llama | 0 | ERROR: missing | - | - | - |
| copy_200_words | 8192 | 8192 | hipml | 0 | ERROR: missing | - | - | - |
| copy_200_words | 8192 | 8192 | llama | 0 | ERROR: missing | - | - | - |
| copy_200_words | 32768 | 32768 | hipml | 0 | ERROR: missing | - | - | - |
| copy_200_words | 32768 | 32768 | llama | 0 | ERROR: missing | - | - | - |
| copy_200_words | 100000 | 100000 | hipml | 0 | ERROR: missing | - | - | - |
| copy_200_words | 100000 | 100000 | llama | 0 | ERROR: missing | - | - | - |

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
| json_tool_call | 32768 | - | - | - |
| json_tool_call | 100000 | - | - | - |
| prose_explanation | 0 | +66.0% | -12.2% | 1.67x |
| prose_explanation | 8192 | - | - | - |
| prose_explanation | 32768 | - | - | - |
| prose_explanation | 100000 | -77.5% | +98.8% | 0.23x |
| longdoc_qa | 0 | - | - | - |
| longdoc_qa | 8192 | - | - | - |
| longdoc_qa | 32768 | - | - | - |
| longdoc_qa | 100000 | - | - | - |
| copy_200_words | 0 | - | - | - |
| copy_200_words | 8192 | - | - | - |
| copy_200_words | 32768 | - | - | - |
| copy_200_words | 100000 | - | - | - |

## Headline gaps

- **prefill**: hipml fastest at `prose_explanation@0` (+66.0%, reps 1/1); llama.cpp fastest at `prose_explanation@100000` (-77.5%, reps 1/1).
- **decode**: hipml fastest at `prose_explanation@100000` (+98.8%, reps 1/1); llama.cpp fastest at `prose_explanation@0` (-12.2%, reps 1/1).
- **ttft**: hipml fastest at `prose_explanation@0` (+67.2%, reps 1/1); llama.cpp fastest at `prose_explanation@100000` (-77.5%, reps 1/1).

## llama.cpp warm pass (cache_prompt=true)

Same ids immediately after the cold reps. A near-1 `prompt_n` and a large `cache_n` confirm the prefix was reused; this is a cache-hit TTFT, not a prefill number.

| prompt | depth | TTFT ms | warm prefill tok/s | cache_n | prompt_n |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose_explanation | 0 | 107 | 41.5 | 369 | 4 |
| prose_explanation | 100000 | 271 | 17.0 | 99996 | 4 |

## Reps and caveats

- **One-rep cells:** `prose_explanation@0`, `prose_explanation@100000`. These are marked `1` in the reps column; treat the range as absent, not as zero variance.
- Same model file, same box, sequential engine runs. The two engines do not share a KV dtype: hipml's chunked path requires f32, llama.cpp is run with `-ctk/-ctv` set by the harness. That is recorded per run.
- 100K cells are real cold prefill. If a cell shows `1` rep, the run was too slow to repeat inside the lock window; it is not cache-prefilled.
- The harness cannot control: GPU clocks/thermal state, driver scheduling, other processes on the box outside the lock, llama.cpp's own thread and graph choices, or the difference between hipml's argmax-in-prefill TTFT and llama-server's prompt-only timing.

