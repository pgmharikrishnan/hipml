# Greedy parity against llama-server

Oracle: `llama-server` on the RTX host, `C:\models\Qwen3.8-27B-Q4_K_M.gguf`,
`-c 4096 -ngl 99`, request `temperature 0, top_k 1, ignore_eos true,
n_predict 24`, token ids read from `completion_probabilities`. The prompt token
ids come from `/tokenize` and are identical to `hipml tokenize` for all three
prompts.

The installed server reports `build 11820 (commit d7725699)`, not the `b12126`
tag the reference sources were taken from. `qwen35.cpp` is byte-identical
between the two; `delta-net-base.cpp` differs only in the disabled
`n_rs_seq > 0` rollback path, so the graph the port follows is the same.

Run: `hipml generate --gguf ... --ids <prompt> --tokens 20` (the Windows host
keeps the weights resident, 1.8 s/token; the Mac build maps them, 51 s/token on
a loaded machine).

## Results

| prompt | text | prompt ids | first divergence in 20 | verdict |
|---|---|---|---|---|
| (a) English | `The capital of France is` | 760, 6511, 314, 9338, 369 | **none** | exact |
| (b) code | `def fibonacci(n):\n    if n < 2:\n        return n\n    return` | 727, 73111, 1393, 1590, 198, 262, 413, 307, 361, 220, 17, 25, 198, 285, 460, 307, 198, 262, 460 | **none** | exact |
| (c) repeated | `the the the the the the the the` | 1719, 279, 279, 279, 279, 279, 279, 279 | **none** | exact |

### (a) The capital of France is

```
llama: 11751 13 198 760 6511 314 14898 369 21047 13 198 760 6511 314 17163 369 23327 13 198 760
hipml: 11751 13 198 760 6511 314 14898 369 21047 13 198 760 6511 314 17163 369 23327 13 198 760
text:  " Paris.\nThe capital of Italy is Rome.\nThe capital of Spain is Madrid.\nThe"
```

### (b) Fibonacci

```
llama: 73111 1393 12 16 8 478 73111 1393 12 17 8 271 727 73111 697 6515 1393 11 20781 65787
hipml: 73111 1393 12 16 8 478 73111 1393 12 17 8 271 727 73111 697 6515 1393 11 20781 65787
text:  " fibonacci(n-1) + fibonacci(n-2)\n\ndef fibonacci_memo(n, memo={}"
```

### (c) Repeated token

```
llama: 279 × 20
hipml: 279 × 20
```

## Numerics: log-probabilities, not just argmax

Same prompt (a), the log-probability of each chosen token as reported by
`llama-server` and by `hipml generate --logprobs`:

| step | token | llama | hipml | Δ |
|---:|---:|---:|---:|---:|
| 0 | 11751 | -0.315485 | -0.312977 | +0.0025 |
| 1 | 13 | -0.375134 | -0.371650 | +0.0035 |
| 2 | 198 | -1.698934 | -1.721977 | -0.0230 |
| 3 | 760 | -0.895761 | -0.906570 | -0.0108 |
| 4 | 6511 | -0.371336 | -0.366320 | +0.0050 |
| 5 | 314 | -0.014403 | -0.014293 | +0.0001 |
| 6 | 14898 | -1.403715 | -1.397785 | +0.0059 |
| 7 | 369 | -0.004061 | -0.004057 | +0.0000 |
| 8 | 21047 | -0.012195 | -0.012141 | +0.0001 |
| 9 | 13 | -0.012015 | -0.012166 | -0.0002 |
| 10 | 198 | -0.077616 | -0.076771 | +0.0008 |
| 11 | 760 | -0.070425 | -0.070628 | -0.0002 |
| 12 | 6511 | -0.010748 | -0.011536 | -0.0008 |
| 13 | 314 | -0.000748 | -0.000820 | -0.0001 |
| 14 | 17163 | -0.521246 | -0.527008 | -0.0058 |
| 15 | 369 | -0.001381 | -0.001447 | -0.0001 |
| 16 | 23327 | -0.004219 | -0.004031 | +0.0002 |
| 17 | 13 | -0.005560 | -0.005746 | -0.0002 |
| 18 | 198 | -0.120522 | -0.126074 | -0.0056 |
| 19 | 760 | -0.047398 | -0.050294 | -0.0029 |

Max |Δ| = 0.023 nats, and the logits gap to the runner-up at step 0 was
`-3.42` vs `-0.31`, so every pick is far from a tie.

Sources of the residual difference, in likely order of size:

1. The prompt is prefilled token by token here; llama.cpp flashes it through
   the chunked delta-net formulation, a different f32 summation order through
   the 48 recurrent layers. Step 2 (right after prefill) is the largest gap.
2. Q4_K/Q6_K block dot products: llama.cpp's kernels accumulate four 8-value
   lanes with fp16 scales; this reference dequantises a 256-value tile and
   reduces in a single f32 chain.
3. RMS-norm reduction order (f32 here, `ggml_float`/f64 in ggml's CPU path).
4. RoPE angles built in f64 here, f32 in llama.cpp.

None of these change a ranking in 60 tokens across the three prompts.

## Depth parity: the CPU f32 oracle gate

The table above prefills token by token. Every benchmark number comes from the
chunked path (`--chunk 2048`), and until WP42 nothing compared that path with an
independent oracle at depth: `attn-parity`, `gemv-check` and `spec-bench` all
compare a kernel with a kernel, or the GPU with itself.

`hipml oracle-gate` fills the gap. It runs hipml's own CPU f32 engine (no GPU,
the same arithmetic as `--no-device`) and the production GPU chunk path on the
same ids, for the same number of greedy steps, and compares the id streams and
the prefill logits.

```
hipml oracle-gate --gguf C:\models\Qwen3.8-27B-Q4_K_M.gguf \
  --ids-file bench/fixtures/p2_math_169.ids --steps 20 --chunk 2048 --kv q8
```

It exits non-zero when the streams disagree and the oracle's margin at the
first disagreement is at least `--tol-nats` (default 0.05 nats); a smaller
margin is a near-tie that fp16 prefill operands and q8 KV may reorder.

### What it found

169 prompt tokens, q8 KV, max |logit difference| at the last prefill row
against the CPU f32 oracle:

| hypothesis | measurement | verdict |
|---|---|---|
| the chunked prefill is the faithful path | chunk vs oracle **9.74** (rms 1.77) | false |
| the oracle is wrong | GPU token loop vs oracle **0.0001** | false |
| KV size / context length | 0.192 with f32 KV, 1 layer | false |
| `HIPML_PREFILL_F32`, split-K, token steps | no change | false |
| the SSM alpha/beta projection | `HIPML_CHUNK_GEMV=2` (one f32 row per launch) 0.192 → 0.00027; alpha/beta up to 12.19 off with 40 of 48 rows **zero**; qkv identical | **true** |

`gemm_rows_f32` passed a block count to `LaunchConfig::linear`, which divides
by the block size again, so a 48-row projection launched 8 blocks and left the
rest of the output untouched. `cd2c6d6` added `gemm_rows_grid` (with a host
unit test that every output row is covered) and a tiled f32 `prefill_f32`
kernel for the chunk path; the token loop was never affected.

### Drift after the fix

| path | worst logit error vs the CPU oracle (169 tokens) |
|---|---|
| token loop, f32 weights | 0.0001 |
| chunk, f32 weights + f32 KV | 0.0165 |
| chunk, quantized weights + q8 KV | 0.113 |
| chunk, quantized weights + q8 KV + fp16 alpha/beta | 0.128 |
| shipped: the same, k-sliced eight ways | 0.161 |

The last row is the shipped path. The two alpha/beta tensors are raw f32 (not
quantized), so the chunk keeps an fp16 copy of each made at load and runs them
through the same fp16-activation WMMA kernel as qkv/z; the exact f32 tiled GEMM
stays behind `HIPML_PREFILL_F32`.

That trade costs 0.04 of worst-case logit (0.16 against the f32 projection's
0.12, both dominated by the q8 KV cache) and it does not move a token: the gate
returns **20/20 ids identical** (step 19 = 41020, the oracle's pick).

### Cost of the projection

The f32 tile cost 3.4 s of a 100K prefill, and the first fp16 version still cost
0.82 s: 64 output rows is half a 128-row tile and a 2048-token chunk is eight
token tiles, so the projection launched **8 blocks on a 64-block card** and ran
at ~7 TFLOPS. `prefill_halfw_w2h_sn` stacks eight k slices along `blockIdx.z`
(`prefill_slices` picks the count for one wave) and `reduce_slices_f32` sums the
partials in slice order, so the result stays deterministic. Measured on the same
binary with the projection skipped as the floor, 100K prefill:

| build | 100K prefill | 8K prefill |
|---|---|---|
| alpha/beta skipped (floor, same binary) | 107.57-107.73 s | 6.52-6.64 s |
| f32 tiled projection | 110.90 s | 6.84 s |
| fp16, single pass (8 blocks) | 108.50 s | 6.61 s |
| fp16, eight k slices (shipped) | 107.94-108.08 s | 6.59-6.63 s |

The shipped projection is inside the floor's own run-to-run band. End to end,
`bench/run.sh --engine hipml --prompt code_edit` (medians; 0 and 8K are three
cold reps, 32K and 100K one):

| depth | TTFT | decode | WP41 (TTFT / decode) |
|---:|---:|---:|---|
| 0 (672 tokens) | 0.560 s | 29.3 tok/s | 0.561 s / 29.2 tok/s |
| 8192 | 6.598 s | 28.6 tok/s | 6.541 s / 28.8 tok/s |
| 32768 | 28.73 s | 27.0 tok/s | 28.647 s / 27.0 tok/s |
| 100000 | 108.00 s | 23.4 tok/s | 107.777 s / 23.4 tok/s |

The 0.2-0.9% spread against the WP41 table is the same ±0.5 s / ±30 ms band
that two consecutive runs of one binary show here (107.56 → 108.08 s at 100K,
6.52 → 6.64 s at 8K), and it is the reason the projection is benchmarked
against a skip-alpha build of itself rather than against last week's table.
