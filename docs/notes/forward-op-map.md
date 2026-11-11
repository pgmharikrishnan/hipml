# The forward pass, as a CPU reference

> *Engineering log from development; internal work-package IDs and paths have been sanitized or removed where they were machine-specific. Current numbers: [docs/benchmarks.md](../benchmarks.md).*

`hipml generate` runs Qwen3.8-27B Q4_K_M end to end on the CPU in f32 and prints
greedy token ids. It exists to be the oracle the GPU graph is checked against,
so it ports llama.cpp's code paths literally rather than deriving them.

```
hipml generate --gguf /path/Qwen3.8-27B-Q4_K_M.gguf --text "The capital of France is" --tokens 20
hipml generate --gguf ... --ids 760,6511,314,9338,369 --tokens 20   # skip the tokenizer
hipml tokenize --gguf ... --text "..."                              # diff against /tokenize
```

Weights stay in their GGUF quantized form (17 GB); each GEMV dequantises one
row at a time and accumulates in f32. The file is memory-mapped read-only
(`mmap.rs`), so the weights are file-backed pages the kernel can evict instead
of anonymous memory that gets swapped. A full model pass reads all 17 GB, so
the first token costs seconds, and work is parallelised over output rows with
`HIPML_THREADS` (default: available cores).

## What the model is

`qwen35`: 64 blocks of 5120-wide, FFN 17408, vocab 248320, 24 q heads and 4 KV
heads of 256 (16 full-attention layers, every 4th) and 48 gated-delta-net
layers (everything else). The GDN side is `d_state 128`, 16 key heads, 48 value
heads, conv width 10240, inner size 6144.

## File map

| File | Contents |
|---|---|
| `config.rs` | hyperparameters from the GGUF metadata table |
| `mat.rs` | quantized matrix + parallel reference GEMV |
| `attention.rs` | full-attention block, KV cache, causal softmax |
| `delta.rs` | conv1d, gated delta net, gated RMS norm |
| `model.rs` | weight loading, per-sequence state, the layer loop |
| `gen.rs` | greedy loop and argmax |
| `tokenizer.rs` | GPT-2 byte BPE with the `qwen35` pretokenizer |
| `gguf_value.rs` | GGUF metadata values (tokens, merges, hparams) |

## Op map: hipml → llama.cpp

Line numbers are from the files saved under `docs/reference/` (tag `b12126`).

| hipml | llama.cpp / ggml |
|---|---|
| `ops::rmsnorm_f32` | `ggml_compute_forward_rms_norm_f32`, ggml-cpu-ops.cpp:4082; called via `build_norm`, llama-graph.cpp:1791 |
| `ops::rope_multi_f32` | `ggml_mrope_cache_init` ggml-cpu-ops.cpp:6204 + `ggml_compute_forward_rope_flt` NEOX/MROPE branch ggml-cpu-ops.cpp:6423 |
| `ops::silu_mul_f32` | `build_ffn` LLM_FFN_SILU + LLM_FFN_PAR → `ggml_swiglu_split`, llama-graph.cpp:1956 |
| `ops::sigmoid_f32` | `ggml_sigmoid`, used for the attention gate (qwen35.cpp:327) and GDN beta |
| `ops::softplus_f32` | `ggml_compute_softplus_f32`, ggml-impl.h:107 |
| `attention::attn_causal_f32` | `build_attn` → `ggml_soft_max_ext(..., kq_scale)`, llama-graph.cpp:3103 |
| `attention::attn_layer_f32` | `llama_model_qwen35::graph::build_layer_attn`, qwen35.cpp:266 |
| `delta::ssm_conv1d_f32` | `ggml_compute_forward_ssm_conv_f32`, ggml-cpu-ops.cpp:9943 |
| `delta::gdn_ar_f32` | `build_delta_net_autoregressive`, delta-net-base.cpp:286; fused-kernel semantics in gated_delta_net_f32.c |
| `delta::l2norm_heads` | `build_gdn_l2_norm`, models.h:14 |
| `delta::ssm_layer_f32` | `build_layer_attn_linear`, qwen35.cpp:347 |
| `model::Model::forward_token` | `llama_model_qwen35::graph::graph`, qwen35.cpp:137 |
| `gen::argmax_f32` | greedy sampler |

## State layouts (the part that is easy to get wrong)

- **KV cache** (`attention::KvState`): `[n_ctx][n_kv * head_dim]`, position
  major, appended per token. Query head `h` reads KV head `h / 6`.
- **Conv state** (`delta::SsmState::conv`): `[(d_conv-1)][channels]`, channel
  innermost; per token the window shifts left and the current `attn_qkv`
  projection is appended (`build_conv_state`).
- **Delta-net state** (`delta::SsmState::s`): `[n_v][head_v][head_v]` with the
  key/state axis innermost, i.e. `state[h][j][i]` for `S[i][j]`. This is the
  layout the fused kernel documents (`s_out[j*S_v + i] = S[i][j]`,
  gated_delta_net_f32.c:11) and the one `ggml_reshape_4d(state, S, S, H)` gives.
  Update: `S[:,j] = S[:,j]*exp(g) + k*(v[j] - S[:,j]·k*exp(g))*beta`; output
  `o[j] = S[:,j]·q / sqrt(S)`.

## Numerics choices

- f32 throughout for weights and activations; RMS-norm sums in f64 where the
  CPU code does (`ggml_float`), matching `ggml_compute_forward_rms_norm_f32`.
- RoPE angles are built in f64 (`pow` + `sin_cos`), consistent with the existing
  `rope_neox_f32` kernel reference; llama.cpp accumulates the angle in f32.
- Attention softmax subtracts the row max, like `ggml_soft_max_ext`.
- GDN uses the sequential per-token recurrence for both prefill and decode.
  llama.cpp flashes a prompt through the *chunked* delta-net formulation
  (delta-net-base.cpp:20), which is the same algebra in a different summation
  order; the difference is f32 rounding, not semantics.

## Q5_K

Two tensors are Q5_K: `blk.51.ffn_down.weight` and `blk.55.ffn_down.weight`.
They are required by the forward pass, so `quant::dequant_q5_k` was added
(port of `dequantize_row_q5_K`, ggml-quants.c:1741). The reference also reads
the IQ4_XS, IQ4_NL, Q3_K and IQ3_S formats an Unsloth UD file mixes in, each
against a hand-computed block in `hipml-core/tests/quant_reference.rs`.

An Unsloth UD dynamic quant uses Q5_K far more widely (115 tensors in
`Qwen3.8-27B-UD-Q6_K.gguf`), so the format also has a dequant-in-shared prefill
GEMM: `prefill_q5_k.hip` dequantises a 64-row weight tile once for a 256-token
block (`prefill_q5_k`), with the WMMA occupancy shape in `prefill_wmma2.hip`
(`prefill_q5_k_w2`, `prefill_q5_k_w2h`, `prefill_q5_k_w2h_sk`). The tile
expression is Q4_K's `y = d*scale*q - dmin*min` with `q = qs_nibble + 16*bit`
from `qh`, where the bit for value `j` is `1 << 2g` for `j % 64 < 32` and
`2 << 2g` for the high half of the 64-value group `g`.

## Tokenizer

`tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre = "qwen35"`, no BOS.
The pretokenizer is a direct port of `unicode_regex_split_custom_qwen35`
(unicode.cpp:779) and runs on the raw text before the GPT-2 byte encoder, as
llama.cpp does. BPE merges use the GGUF rank table. ASCII is exact; `\p{M}` and
non-ASCII whitespace use Rust's approximations instead of llama.cpp's generated
Unicode tables.

## Comparing against llama.cpp

Reference: `llama-server` on the model at `-c 4096`, `temperature 0`,
`ignore_eos true`, 24 predicted tokens, token ids from `completion_probabilities`.
The comparison table and first-divergence indices live in
`docs/notes/wp42-forward-parity.md`.
