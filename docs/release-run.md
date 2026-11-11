# Release run: stock Qwen3.8-27B-Q4_K_M on one R9700

This is the run the README, the articles and the release notes quote. It was
made **once, cold**, on the stock release model, with the same fixture prompts
and the same token ids fed to both engines. Raw output is under
[`bench/release/`](../bench/release/); the method is
[`bench/README.md`](../bench/README.md).

## Machine and model

| field | value |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700 (gfx1201), 32 GiB |
| GPU driver | 32.0.32015.7029 (2026-10-18) |
| OS | Windows 11 Pro (build 26200) |
| ROCm | 7.2.1 (`C:\rocm-sdk\_rocm_sdk_core`, HIP packaging 53211-158bd99533) |
| llama.cpp | 0.7.0-dev, build **11820**, commit **`d7725699`**, Clang 24.0.0 |
| model | `Qwen3.8-27B-Q4_K_M.gguf` |
| model bytes | **16,810,714,336** |
| model SHA-256 | `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520` |
| provenance | `lmstudio-community/Qwen3.8-27B-GGUF`, Apache-2.0 per its card |
| hipml commit | `6aeb5d962acb0b5973fa5c8130cb0913b3db503b` (release tree) |
| hipml binary SHA-256 | `ab3074b02f6c281f815a06c223b09b08549c44a5208baf1f0d07d05020c4b220` |

The file is a standard GGUF v3 (magic `47 47 55 46`, version 3). `hipml load`
reports **866 tensors, 16,799,719,424 B (15.65 GiB)**, all supported.

### Tensor type histogram

| type | tensors | bytes | GiB | engine |
| --- | ---: | ---: | ---: | --- |
| Q4_K | 439 | 12,262,440,960 | 11.42 | supported |
| Q6_K | 67 | 4,526,592,000 | 4.22 | supported |
| F32 | 360 | 10,686,464 | 0.01 | supported |
| total | 866 | 16,799,719,424 | 15.65 | |

No IQ type, no Q3_K, no Q5_K. This file avoids exactly the types that blocked
the Unsloth dynamic variants (see [`native-qwen.md`](native-qwen.md)).

### Config, read from the GGUF

```
config (qwen35): 65 blocks (64 decoder + 1 nextn), hidden 5120, ffn 17408,
  heads 24 / kv 4, head_dim 256, vocab 248320, eps 1e-6, rope base 1e7 rot 64
  sections [11, 11, 10, 0], ctx 262144
  ssm: conv 4, state 128, groups 16, dt_rank 48, inner 6144 (head_v 128)
  full-attention layers (16): [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
```

Identical to what P9 recorded in [`native-qwen.md`](native-qwen.md). Residency
checks on the stock file: 4/4 round-trip tensors byte-identical; embedding rows
0/1/1234/100000/248319 exact (5,120/5,120, max_abs 0); lm_head 248,320 logits in
5.150 ms (202.5 GB/s). Full output in `bench/release/gates/gateStock.out`.

## Exactly what was run

```bash
# fixture + head-to-head matrix, both engines
HIPML_GGUF=<stock-Q4_K_M.gguf> HIPML_MTP_GGUF=<mtp-Q4_0.gguf> \
LLAMA_SERVER=<llama-server> HIPML_GPU_LOCK=<gpu-lock.ps1> \
HIPML_LOCK_LABEL=wp55 bench/run.sh --host <gpu-host> --remote-dir <checkout> \
  --refresh-fixture --validate --llama-kv q8_0 --hipml-kv q8

# MTP, chat divergence, server, batch, cache, tokenizer
python bench/chat/compare.py --tokens 256 --llama-fa on
python bench/chat/mtp_chat.py --tokens 384 --drafts 3
python bench/serve/serve_bench.py --engine {hipml,llama} --cell {a,b,c} --reps {10,6,3}
hipml batch-bench --gguf MODEL --slots {8,16} --tokens 16 --kv q8 [--check]
python bench/batch_llama.py --np {8,16} --predict 32
hipml cache-bench --gguf MODEL --len 8192 --shared 4096 --chunk 2048 --bytes-mib 4096
python bench/tokenizer/diff.py --corpus corpus.txt --hipml-ids ids.txt
python bench/tokenizer/template_diff.py --gguf MODEL
```

**llama.cpp settings in the head-to-head.** `-ngl 99 -c <ctx> --parallel 1 -ctk
q8_0 -ctv q8_0 --no-webui --no-cache-idle-slots`, no draft model and no
speculative decoding. It was started with no `-fa`, `-b` or `-ub` flag, so the
build defaults applied: **`-fa auto`, `-b 2048`, `-ub 512`**. On this build
`-fa auto` resolves to **on**: a quantized V cache is rejected with
`quantized V cache requires flash_attn to be enabled` when it is off (observed
while preparing the tokenizer cell). hipml used its own MTP only where a cell
says so; the plain matrix has MTP off on both engines.

## Head-to-head matrix

5 prompts × depth 0 and 8,192 at **3 cold reps**; `code_edit` at 32,768 and
100,000 at **1 rep**. `median (min-max)`; decode is tokens 2..33; q8-class KV on
both engines (hipml `q8`, llama `q8_0`).

| prompt | depth | tokens | engine | reps | TTFT ms | prefill tok/s | decode tok/s | peak GiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| code_edit | 0 | 672 | hipml | 3 | 574 (572-582) | 1,172 (1,154-1,175) | 29.7 (29.6-29.7) | 16.94 |
| code_edit | 0 | 672 | llama | 3 | 698 (672-731) | 1,000 (943-1,003) | 31.8 (31.6-31.8) | 15.50 |
| code_edit | 8,192 | 8,192 | hipml | 3 | 6,689 (6,684-6,698) | 1,225 (1,223-1,226) | 29.0 (29.0-29.1) | 18.17 |
| code_edit | 8,192 | 8,192 | llama | 3 | 7,080 (7,054-7,090) | 1,160 (1,156-1,162) | 30.9 (30.9-31.0) | 15.75 |
| code_edit | 32,768 | 32,768 | hipml | 1 | 28,923 | 1,133 | 27.4 | 19.08 |
| code_edit | 32,768 | 32,768 | llama | 1 | 31,300 | 1,048 | 28.7 | 16.66 |
| code_edit | 100,000 | 100,000 | hipml | 1 | 108,698 | 920 | 23.7 | 21.26 |
| code_edit | 100,000 | 100,000 | llama | 1 | 122,997 | 813 | 22.7 | 19.16 |
| json_tool_call | 0 | 300 | hipml | 3 | 345 (343-348) | 869 (861-875) | 29.6 (29.5-29.6) | 16.67 |
| json_tool_call | 0 | 300 | llama | 3 | 385 (366-385) | 822 (781-843) | 31.7 (31.7-31.7) | 15.49 |
| json_tool_call | 8,192 | 8,192 | hipml | 3 | 6,700 (6,694-6,714) | 1,223 (1,220-1,224) | 29.0 (29.0-29.1) | 18.17 |
| json_tool_call | 8,192 | 8,192 | llama | 3 | 7,063 (7,063-7,093) | 1,162 (1,157-1,163) | 31.0 (30.9-31.0) | 15.75 |
| prose_explanation | 0 | 373 | hipml | 3 | 372 (370-376) | 1,002 (993-1,009) | 29.6 (29.5-29.6) | 16.77 |
| prose_explanation | 0 | 373 | llama | 3 | 420 (409-420) | 932 (892-944) | 31.7 (31.6-31.8) | 15.48 |
| prose_explanation | 8,192 | 8,192 | hipml | 3 | 6,702 (6,691-6,704) | 1,222 (1,222-1,224) | 29.0 (28.9-29.1) | 18.17 |
| prose_explanation | 8,192 | 8,192 | llama | 3 | 7,077 (7,063-7,094) | 1,162 (1,158-1,163) | 31.0 (30.9-31.0) | 15.75 |
| longdoc_qa | 0 | 7,694 | hipml | 3 | 6,381 (6,373-6,395) | 1,206 (1,203-1,207) | 29.0 (28.9-29.1) | 18.16 |
| longdoc_qa | 0 | 7,694 | llama | 3 | 6,675 (6,662-6,716) | 1,153 (1,147-1,156) | 31.0 (30.9-31.0) | 15.73 |
| longdoc_qa | 8,192 | 8,192 | hipml | 3 | 6,707 (6,701-6,716) | 1,221 (1,220-1,223) | 29.0 (28.9-29.0) | 18.17 |
| longdoc_qa | 8,192 | 8,192 | llama | 3 | 7,071 (7,069-7,098) | 1,161 (1,157-1,161) | 31.0 (30.9-31.1) | 15.75 |
| copy_200_words | 0 | 285 | hipml | 3 | 343 (340-347) | 831 (821-839) | 29.4 (29.4-29.5) | 16.66 |
| copy_200_words | 0 | 285 | llama | 3 | 352 (343-374) | 848 (796-865) | 31.8 (31.6-31.8) | 15.49 |
| copy_200_words | 8,192 | 8,192 | hipml | 3 | 6,701 (6,691-6,702) | 1,223 (1,222-1,224) | 29.0 (29.0-29.0) | 18.17 |
| copy_200_words | 8,192 | 8,192 | llama | 3 | 7,069 (7,067-7,100) | 1,161 (1,157-1,162) | 31.0 (30.8-31.0) | 15.75 |

### Gap (hipml minus llama.cpp)

Positive means hipml is faster. Every row is its own prompt class; nothing is
averaged across classes.

| prompt | depth | prefill | decode | TTFT (llama/hipml) |
| --- | ---: | ---: | ---: | ---: |
| code_edit | 0 | +17.2% | −6.6% | 1.22× |
| code_edit | 8,192 | +5.6% | −6.1% | 1.06× |
| code_edit | 32,768 | +8.2% | −4.6% | 1.08× |
| code_edit | 100,000 | +13.1% | +4.3% | 1.13× |
| json_tool_call | 0 | +5.8% | −6.8% | 1.11× |
| json_tool_call | 8,192 | +5.3% | −6.4% | 1.05× |
| prose_explanation | 0 | +7.5% | −6.9% | 1.13× |
| prose_explanation | 8,192 | +5.2% | −6.5% | 1.06× |
| longdoc_qa | 0 | +4.6% | −6.3% | 1.05× |
| longdoc_qa | 8,192 | +5.2% | −6.5% | 1.05× |
| copy_200_words | 0 | **−2.1%** | −7.4% | 1.02× |
| copy_200_words | 8,192 | +5.3% | −6.3% | 1.05× |

hipml wins prefill in 11 of 12 cells; the one loss is `copy_200_words` at depth
0 (831 vs 848 tok/s, a 285-token prompt). Decode is llama.cpp's only consistent
win — both engines are pinned to the DRAM roofline — except at 100K, where
hipml is +4.3%.

## MTP decode, chat prompts, EOS-stopped

`bench/chat/mtp_chat.py --tokens 384 --drafts 3`, f32 KV, greedy, stop at EOS.
"Best drafts" is the drafts count with the highest measured token rate; the
speedup is against plain greedy on the same ids.

| conversation | prompt tok | plain tok/s | best drafts | MTP tok/s | speedup | accept | exact | EOS |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 01_single_turn_code | 91 | 29.34 | 2 | 49.49 | **1.69×** | 0.874, 0.758 | 384/384 | no (limit) |
| 02_multi_turn | 134 | 29.29 | 2 | 44.79 | **1.53×** | 0.754, 0.714 | 384/384 | no (limit) |
| 03_system_tool | 364 | 29.35 | 3 | 58.60 | **2.00×** | 0.944, 0.941, 0.875 | 65/65 | yes |
| 04_longdoc_qa | 8,192 | 25.14 | 3 | 46.01 | **1.83×** | 0.885, 0.870, 0.900 | 174/174 | yes |
| 05_copy | 89 | 29.43 | 3 | 56.33 | **1.91×** | 0.897, 0.923, 0.875 | 100/100 | yes |

The draft stream is token-exact against plain greedy over the whole stream in
every cell (`exact` column). Raw: `bench/release/mtp/`.

CLI `spec-bench` on the stock model (128 tokens) agrees: drafts 0/1/2/3 at
29.46 / 46.47 / 55.54 / 56.03 tok/s, 128/128 exact on every row
(`bench/release/gates2/spec-bench.out`).

## Server, 1 / 8 / 16 slots

Same conversations, greedy, 64 new tokens each, warm shared ~2.4K-token system
prompt. 50 requests in cell a (10 waves × 5 conversations), 48 in b and c.

| cell | slots | ctx/slot | engine | requests | errors | TTFT p50 ms | TTFT p95 ms | aggregate tok/s | per-stream tok/s | peak GiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| a | 1 | 32,768 | hipml | 50 | **0** | 2,999 | 3,031 | **41.19** | 41.19 | 20.38 |
| a | 1 | 32,768 | llama | 50 | **0** | 2,240 | 2,326 | 31.58 | 31.58 | 16.65 |
| b | 8 | 16,384 | hipml | 48 | **0** | 518 | 526 | **125.25** | 16.92 | 24.36 |
| b | 8 | 16,384 | llama | 48 | **0** | 497 | 1,377 | 94.75 | 12.12 | 21.24 |
| c | 16 | 8,192 | hipml | 48 | **0** | 779 | 784 | 212.14 | 15.01 | 26.28 |
| c | 16 | 8,192 | llama | 48 | **0** | 1,237 | 2,682 | 214.41 | 14.17 | 22.40 |

Aggregate: +30.4% at 1 slot (MTP path), +32.2% at 8, −1.1% at 16 (a tie, llama
marginally ahead). Cell a's TTFT is hipml's cold path for five unrelated
conversations (no shared prefix, so no cache hit); llama prefills that 2,423-token
prompt 25% sooner. llama-server reports its prefix-cache hits as `cache_n`,
which `serve_bench.py` does not read, so its `prefix_hit_tokens` column reads 0;
its 100 ms TTFT minimum in cells b/c shows the shared prefix was cached.

## Batch decode, B=8 and B=16

`tokens 16`, depth 0, q8 KV, one weight pass per step. The **exact** path is the
bit-identical one (`_b` kernels) and is what `--check` validates. The **skinny**
path is the opt-in WMMA tile (`HIPML_SKINNY=1`) and is *not* bit-identical.

| B | engine | path | aggregate tok/s | per stream | vs llama |
| ---: | --- | --- | ---: | ---: | ---: |
| 8 | hipml | exact | 107.46 | 13.43 | +10.2% |
| 8 | hipml | skinny | **149.43** | 18.68 | **+53.2%** |
| 8 | llama | `-np 8` | 97.54 | 12.19 | — |
| 16 | hipml | exact | 107.33 | 6.71 | **−52.9%** |
| 16 | hipml | skinny | **272.97** | 17.06 | **+19.8%** |
| 16 | llama | `-np 16` | 227.89 | 14.24 | — |

Single-stream reference on the same model: 30.02–30.16 tok/s.

**Exactness.** `batch-bench --check` B=8 and B=16: every slot reproduces the
B=1 path 16/16 ids over the full stream. The skinny path diverges (`check:
TOKENS DIFFER`) and must be quoted as approximate, not exact.

## Prefix cache

`cache-bench`, q8 KV, chunk 2,048.

| case | cache | tokens matched | TTFT ms | vs cold | prefill% | ids |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cold P (2,048) | miss | 0 | 1,614.2 | 1.00× | 0% | ok |
| warm exact P | hit | 2,048 | **1.8** | **895.2×** | 100% | equal |
| cold Q (2,048) | miss | 0 | 1,616.1 | 1.00× | 0% | ok |
| warm shared Q (50% prefix) | hit | 1,024 | **836.0** | **1.93×** | 50% | equal |
| cold P (8,192, default 1 GiB) | miss | 0 | 6,722.7 | 1.00× | 0% | ok |
| warm exact P (default cap) | refused | 0 | 6,794.5 | 0.99× | 0% | equal |
| warm shared Q (default cap) | hit | 4,096 | **3,499.7** | **1.95×** | 50% | equal |
| cold P (8,192, cache 4 GiB) | miss | 0 | 6,724.4 | 1.00× | 0% | ok |
| warm exact P (4 GiB) | hit | 8,192 | **4.6** | **1,460.2×** | 100% | equal |
| warm shared Q (4 GiB) | hit | 4,096 | **3,498.8** | **1.94×** | 50% | equal |

Hit latency (lookup + restore + argmax, no compute): 1.89 ms at 2,048 tokens,
4.59 ms at 8,192.

**The default-cap defect is fixed.** At 1 GiB the 8,192-token snapshot needs
1,174.6 MiB, so it is **refused at capture time** (counted in `refused`, never
truncated) and the exact repeat falls back to a full prefill: the ids are
token-identical, and the row is reported as `refused`, not `hit`. With
`--bytes-mib 4096` the same cell is a 4.6 ms exact hit. Admission evicts LRU
entries to make room first and refuses only when one entry cannot fit at all;
KV and SSM halves are published together or not at all, and a snapshot whose
recorded position disagrees with its entry is a miss. `cache-gate` replays the
old consumer (a partial hit used as exact) and requires it to diverge — it
diverges at id 0 — while the fixed fallback matches the uncached ids. Peak
process VRAM at the default cap fell from 21,270.9 to 20,340.9 MiB (the refused
KV+SSM buffers are no longer allocated); at 4 GiB it fell from 22,553.5 to
21,462.9 MiB (reserve evicts before allocating the next snapshot).

## Tokenizer and chat template

- `bench/tokenizer/diff.py` on the full WP48 corpus: **6,912 strings, 0
  mismatches**, 0 llama errors.
- `bench/tokenizer/template_diff.py`: **18/18 cases pass** — tool schemas,
  tool-call argument escaping, `preserve_thinking`, `enable_thinking`,
  reasoning effort, and image/video rejection.
- `tokenizer_fixture` integration test: **2/2 pass** (266 multilingual strings
  and 16 stored conversations).

## Divergences from llama.cpp on the chat prompts

`bench/chat/compare.py --tokens 256 --llama-fa on`, llama's own
`/apply-template` + `/tokenize` on one side and `hipml chat` on the other. The
**rendered template ids match for all five conversations** (91/134/364/8,192/89
tokens). The greedy output streams:

| conversation | result | first divergence | llama / hipml id | logit margin |
| --- | --- | ---: | --- | ---: |
| 01_single_turn_code | diverge | 2 | 310 / 4087 | **0.2064 nats** |
| 02_multi_turn | diverge | 130 | 30 / 6970 | **0.0059 nats** |
| 03_system_tool | **identical**, 65 tokens (EOS) | — | — | — |
| 04_longdoc_qa | **identical**, 174 tokens (EOS) | — | — | — |
| 05_copy | diverge | 63 | 5,469 / 1,330 | **0.0260 nats** |

At every divergence the two candidates are nearly tied in llama's own top-k:
`30:−1.249` vs `6970:−1.255` (0.006 nats) in `02_multi_turn`,
`5469:−0.847` vs `1330:−0.873` in `05_copy`, and `310:−0.795` vs
`4087:−1.002` in `01_single_turn_code`. A different q8 KV layout and a
different accumulation order are enough to flip a near-tie; there is no
unexplained large-margin divergence.

## Gates on the merged tree

Merged `feat/integrate` (WP53) then `feat/native-qwen` (P9) onto `2661314`.
Both merges were clean; the anticipated conflicts in `prefill.rs`, the config
loader and the docs index did not materialise as text conflicts. The merged
tree keeps WP52's split-K scratch sizing and WP53's `splitk-gate`.

| gate | result |
| --- | --- |
| `cargo test --workspace` | **84 passed, 0 failed** (local and on the GPU host) |
| `cargo check --workspace --features hip` | clean |
| `cache-gate --len 8192 --shared 4096 --chunk 2048` | **PASS**: capture refused (1,174.6 > 1,024 MiB), fallback ids equal the uncached run, legacy partial-hit path diverges at id 0 |
| `oracle-gate --ids-file p2_math_169.ids --steps 20 --chunk 2048 --kv q8` | **20/20**, step 19 = **41020** |
| 20-id acceptance (stock model) | `11751 13 198 760 6511 314 9564 369 19241 13 198 760 6511 314 14898 369 21047 13 198 760`; chunked = token loop; CPU text identical |
| `parity --deep` | **56 passed, 0 failed, 1 skipped** (`sample_top_p_f32`, host-only) |
| `parity prefill --deep` | **6 passed, 0 failed, 0 skipped** |
| `gemv-check` 4 tensors × R=1/2/4 | **12/12**, 0 differing, max \|d\| 0.000e0 |
| `spec-bench` drafts 0..3 | 128/128 exact on every row |
| `batch-bench --check` B=8 / B=16 | every slot 16/16 ids match B=1 |
| `rows-check` 16 tails | 112 decode ids match the single-sequence path; SSM state differs in last bits (max \|d\| 2.45e-2) |
| `tokenizer_fixture` | 2/2 pass |
| chunk boundary, 8,193 tokens | prefill completes in **6.67 s**, last id **293** |
| `splitk-gate` synthetic | PASS 44/44, worst \|dlogit\| **0.011199** ≤ 0.05 |
| `splitk-gate --ids-file p2_math_169.ids` | PASS 44/44, ids identical, worst \|dlogit\| **9.309817** ≤ 16 |

`rows-check` on this tree reports the wide-GEMM tile's last-bit movement in the
39.2M-word SSM state; the decoded ids are identical, which is what the gate
asserts. Raw gate logs: `bench/release/gates/`.

The gate rows above (oracle, parity, gemv, spec, batch check, rows, splitk,
chunk-boundary prefill) were run before the switch to the stock model. The
20-id acceptance and every release cell in this document use
`Qwen3.8-27B-Q4_K_M.gguf`.

## What is still unmeasured

- MTP at **q8 KV** on the stock model: the release MTP cells use f32 KV, like
  the earlier chat tables. q8 KV MTP is implemented but not measured here.
- `mtp-batch-bench` full matrix (B=4, B=16, admission, exact `_b` mode) on the
  stock model; only the CLI `spec-bench` and single-stream server paths were
  exercised.
- Batch depth > 0 (8K/32K) for the stock model; the batch cells are depth 0.
- The skinny batch path's error against the exact path beyond the token-id
  check (no logit margin measured).
- Server cells at 32 slots or on a different context split.
- Any GPU other than gfx1201, and any OS other than Windows 11.

## Still citing pre-fix numbers

The pre-fix benchmark tables were measured at `cef7657`, on an earlier model,
and predate the WP42 correctness fix. `bench/results.json` and `bench/RESULTS.md`
have been removed, and `docs/benchmarks.md` now opens with a "superseded"
banner that points here.

The engineering logs under [`notes/`](notes/) record that generation of numbers
on purpose; each carries the "engineering log" banner and none is release copy.
`docs/integrate.md`, `docs/mtp-batch.md` and `docs/wp52.md` are gate logs (which
model each gate ran on is stated there and above). Everything in this file is
superseded by this file for release copy; the older notes remain useful as
engineering history and are labelled as such, not deleted.
