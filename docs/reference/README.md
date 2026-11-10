# llama.cpp sources this module ports

These eleven files are **unmodified copies** from llama.cpp, tag `b12126`
(https://github.com/ggml-org/llama.cpp/tree/b12126), kept so the port can be
audited line by line. They are © the ggml authors, MIT License (see `LICENSE`
in this directory). They are not part of the hipml build, are not compiled or
linked, and are **not** covered by hipml's Apache-2.0 license; `hipml-core` has
no build dependency on them. Line numbers quoted in `docs/notes/` refer to
these files as upstream published them, so the bodies are left byte-identical
and carry no header of ours. See `NOTICE` at the repository root for the
attribution and for what was derived from each one.

| file | used for |
|---|---|
| `qwen35.cpp` | the architecture: layer loop, attention block, GDN block, FFN |
| `delta-net-base.cpp` | conv state, autoregressive gated delta net, recurrent attention |
| `gated_delta_net_f32.c` | fused-kernel semantics and the state layout |
| `models.h` | `build_gdn_l2_norm` |
| `llama-graph.cpp` | `build_norm`, `build_ffn`, `build_attn`, rope parameters |
| `llama-vocab.cpp` | BPE tokenizer |
| `unicode.cpp`, `unicode.h` | the `qwen35` pretokenizer and byte encoder |
| `ggml-cpu-ops.cpp` | rms_norm, rope/mrope, ssm_conv, softmax; also carries the YaRN notice ("MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng") |
| `ggml-quants.c` | Q5_K dequant (the two Q5_K tensors the model stores) |
| `ggml-impl.h` | `ggml_compute_softplus_f32` |

The Unicode category table in `crates/hipml-core/src/unicode_tables.rs` is
generated from llama.cpp commit `d7725699` (`src/unicode-data.cpp`), not from
these files; see `NOTICE` §3 for the Unicode data license.
