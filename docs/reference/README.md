# llama.cpp sources this module ports

Tag `b12126` (https://github.com/ggml-org/llama.cpp/tree/b12126), fetched so
the port can be audited line by line. These are reference copies only: nothing
here is compiled or linked, and `hipml-core` has no build dependency on them.

| file | used for |
|---|---|
| `qwen35.cpp` | the architecture: layer loop, attention block, GDN block, FFN |
| `delta-net-base.cpp` | conv state, autoregressive gated delta net, recurrent attention |
| `gated_delta_net_f32.c` | fused-kernel semantics and the state layout |
| `models.h` | `build_gdn_l2_norm` |
| `llama-graph.cpp` | `build_norm`, `build_ffn`, `build_attn`, rope parameters |
| `llama-vocab.cpp` | BPE tokenizer |
| `unicode.cpp`, `unicode.h` | the `qwen35` pretokenizer and byte encoder |
| `ggml-cpu-ops.cpp` | rms_norm, rope/mrope, ssm_conv, softmax |
| `ggml-quants.c` | Q5_K dequant (the two Q5_K tensors Qwen3.8-27B stores) |
| `ggml-impl.h` | `ggml_compute_softplus_f32` |
