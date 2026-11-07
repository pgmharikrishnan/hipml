# Gated delta net on the GPU (WP11)

The 48 recurrent blocks of Qwen3.8-27B (qwen35) now run on the card. The CPU
reference in `crates/hipml-core/src/delta.rs` is the specification; the kernels
are ports of the same algebra, with the layout taken from
`docs/reference/gated_delta_net_f32.c` (`state[h][j][i]`, key axis innermost).

## What is on the device

| kernel | file | job |
| --- | --- | --- |
| `ssm_conv1d_f32` | `ssm_conv1d_f32.hip` | causal depthwise 4-tap conv over 10240 channels, window carried in `state` |
| `ssm_silu_inplace_f32` | `ssm_silu_inplace_f32.hip` | activation applied to the conv output |
| `l2norm_f32` | `ssm_l2norm_f32.hip` | per-head `x/sqrt(sum(x²)+eps)` for q and k |
| `gdn_ar_f32` | `gdn_ar_f32.hip` | one-token gated delta rule, one block per value head |
| `ssm_gates_f32` | `ssm_gates_f32.hip` | `g = softplus(alpha+dt_bias)*a`, `beta = sigmoid(beta_raw)` |
| `ssm_gate_norm_f32` | `ssm_gate_norm_f32.hip` | gated RMS norm (`ssm_norm`) × `silu(z)` |
| `ssm_layer_f32` | `ssm.rs` / `ssm_ops.rs` | the whole layer over resident buffers |

`SsmGpuLayer` owns the weight and state buffers and exposes `step()`. The four
input projections and the output projection use the existing `gemv_f32`.

The recurrence is one block of 128 threads per value head; a pinned 32-lane
group owns one value row, holding `head_dim/32 = 4` state elements per lane in
registers. Both dots are XOR-butterfly reductions, so every lane has the same
`dot` for the state update (a down-shuffle reduction leaves it in lane 0 only —
that was the one real bug found during bring-up).

## Parity vs the CPU reference

`hipml ssm` compares the same deterministic inputs. Tolerances and worst
observed error (absolute / relative):

| case | tolerance abs / rel | worst abs | worst rel |
| --- | --- | ---: | ---: |
| conv1d alone (56 ch, d_conv 4) | 1e-6 / 1e-5 | 5.96e-8 | 4.42e-6 |
| recurrence alone (3 v-heads, 2 k-heads, sd 8) | 1e-4 / 1e-3 | 1.79e-7 | 2.96e-6 |
| l2norm alone (5 heads × 16) | 1e-6 / 1e-5 | 5.96e-8 | 1.17e-7 |
| gated RMS norm alone (3 × 8) | 1e-4 / 1e-3 | 1.49e-8 | 6.63e-8 |
| full layer, small | 1e-4 / 1e-3 | 4.47e-8 | 8.84e-6 |
| **full layer, real shape** | 2e-3 / 1e-2 | **7.27e-6** | 5.19e-2 |

The real-layer relative figure is a near-zero output element (5120-term GEMVs
against a sequential f32 reference); the absolute error is the meaningful one
and is 7.3e-6 over the full 5120-wide layer output plus both state buffers.
The real-shape case is deliberately outside `cases()` because it allocates
464 MB of f32 weights; `hipml ssm` builds it explicitly.

## Per-layer timing (R9700, real shapes)

Weights are f32 here (the loader branch keeps them quantized), so the GEMV rows
are a bandwidth measurement of 463.5 MB of f32, not the model's 17.19 GB of
packed weights. The recurrent row does not depend on weight format.

| stage | µs / layer | 48 layers |
| --- | ---: | ---: |
| GEMV, 4 input projections | 554.8 | 26.63 ms |
| GEMV, output projection | 206.5 | 9.91 ms |
| **recurrent: gates + conv + L2 + GDN + gated norm** | **28.0** | **1.35 ms** |
| full layer (f32 weights) | 798.5 | 38.33 ms |

- Weight-read reference (frozen number): 17.19 GB / 632 GB/s = **27.19 ms/token**.
- Recurrent add: **1.35 ms/token**, 4.9% of the weight-read time (run-to-run
  28–32 µs/layer; one 54 µs outlier under contention).
- Full 48-layer projection with f32 weights: 38.33 ms/token; the +11.14 ms over
  the weight-read figure is the f32-vs-packed weight size, not SSM compute.

## VRAM state footprint

Per sequence, per layer (carried across tokens, not per token):

- conv window: `(d_conv-1) × 10240 = 30,720` floats = 122.9 KB
- delta-net state: `48 × 128 × 128 = 786,432` floats = 3.15 MB

**3.27 MB per layer → 156.9 MB for all 48 SSM layers, per sequence.** The
17.19 GB of weights is separate and shared across sequences.

## Commands

On the GPU box, from `C:\hipml-ssm`:

```
target\release\hipml.exe ssm
target\release\hipml.exe parity ssm      # small cases, standard harness
```

## Not verified here

- The layer consumes **f32** resident weights. Wiring it to the quantized loader
  (`feat/loader`, `DeviceTensor` + `gemv_q4_k/q6_k/q8_0`) is the next step; the
  arithmetic of the recurrent kernels does not change.
- One token at a time. Prefill still uses the CPU reference or the chunked
  formulation; the kernels implement the autoregressive path llama.cpp uses for
  decode.
- Knobs beyond qwen35's (per-element `g`, `d_state != head_v`) are not handled.
