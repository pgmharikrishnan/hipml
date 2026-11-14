# The MoE FFN: the conventions, settled against llama.cpp's ids

> *Engineering log from development; internal work-package IDs and paths have
> been sanitized or removed where they were machine-specific.*

`Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` is the first mixture-of-experts file the
engine has opened: 753 tensors, 22,853,663,008 B, `qwen35moe`, 40 decoder blocks
plus one inline MTP block, 256 experts with 8 used. Every attention and
gated-delta-net tensor in it is shaped exactly like the 27B's; the FFN is the
only new thing.

This note covers two slices. The first settled the numerical conventions on the
CPU, before any kernel existed, because four independent conventions (softmax
domain, top-k order, renormalisation, where the router weight is applied) can
each leave the ids subtly wrong, and a kernel built on a wrong one is the wrong
kernel. The second built the device path — a row gather for every format, an
expert GEMV that resolves the gather on the device, device-side top-k, and a
prefill loop — and it is the first time this file produces tokens on the GPU.

**Result: the CPU reference reproduces llama.cpp's 64 greedy ids on a fixed
prompt, 64/64, first try; the device path reproduces the same 64 ids, 64/64, at
125.76 tok/s decode and 143.99 tok/s prefill.** The file loads: 753 tensors, 0
unsupported, 21.41 GiB resident (21.69 GiB with the engine's scratch).

## 1. Two things stopped the file opening

### 1.1 BF16 (GGML type 30)

`TensorType` had no arm for 30, so the two BF16 tensors became
`Unsupported(30)`, `size_only(30)` returned `(0, 0)`, and `Gguf::open` died in
the tensor table before any config was read:

```
tensor blk.40.ffn_gate_inp.weight: cannot size type type_30 over 524288 values
```

Exactly two tensors are BF16, both MTP router rows:

| tensor | shape | values | bytes |
| --- | --- | ---: | ---: |
| `blk.40.ffn_gate_inp.weight` | `[2048, 256]` | 524,288 | **1,048,576** |
| `blk.40.ffn_gate_inp_shexp.weight` | `[2048]` | 2,048 | **4,096** |

BF16 is not a block format: one value, two bytes, and the value is the **top 16
bits of an f32**. Widening is a shift (`bf16_to_f32(bits) = f32::from_bits(bits
<< 16)`), which is `ggml_bf16_to_fp32` in upstream `ggml-impl.h`; the low 16
mantissa bits are zero, which is the only rounding the format has. So
`block_values = 1`, `block_bytes = 2`, and `nbytes(n) = 2n` — 2 bytes per value,
with no super-block to divide.

Added: `TensorType::BF16` (id 30, name `BF16`), `bf16_to_f32`, `dequant_bf16`,
`Kind::BF16` in the CPU matrix, and `supported()` in the loader. The resident
loader needed nothing else: `DeviceTensor` already carries arbitrary `dims` and
uploads the payload byte for byte, so a 3-D expert tensor and a 2-byte scalar
type both just work. There is **no BF16 GEMV kernel** and none is needed here:
both tensors are MTP-only, and MTP is out of scope.

### 1.2 The missing `feed_forward_length`

`config.rs` called `u("feed_forward_length")?`, and this architecture does not
have that key — it has the expert widths instead. That was the *second* failure,
after BF16.

## 2. Config: the MoE keys

Five fields: `n_expert`, `n_expert_used`, `n_ff_exp`, `n_ff_shexp`, `has_shexp`.

The four MoE metadata keys are read **as a set**:

```
expert_count                       256
expert_used_count                    8
expert_feed_forward_length         512
expert_shared_feed_forward_length  512
```

A file that declares **none** of them is dense. A file that declares **some but
not all** is refused, and the error names the missing key:

```
GGUF metadata qwen35moe.expert_shared_feed_forward_length is missing:
the file declares 3 of the 4 MoE keys
```

That is deliberate: defaulting a width to zero would produce a file that loads
and computes nonsense. `has_shexp` is not a key — it is the presence of
`blk.0.ffn_up_shexp.weight`, which is how `qwen35moe.cpp` decides it too
(`ffn_up_shexp != nullptr`; `expert_shared_count` is absent from these files).

`feed_forward_length` is now optional:

* present → `n_ff` is the dense width, exactly as before;
* absent and `n_expert > 0` → `n_ff = n_ff_exp * n_expert_used` = **4,096**,
  the widest per-token FFN intermediate the decode path stages;
* absent and no experts → an error naming the key.

Validation, only when `n_expert > 0`: `1 <= expert_used_count <= expert_count`,
`expert_feed_forward_length` a positive multiple of `QK_K`, `n_embd` a multiple
of `QK_K`, and a shared expert implies `expert_shared_feed_forward_length > 0`.

`hipml load` prints the resolved values, which is the probe:

```
config (qwen35moe): 41 blocks (40 decoder + 1 nextn), hidden 2048, ffn 4096, ...
  moe: 256 experts, 8 used, expert ffn 512, shared expert 512, ffn 4096
  full-attention layers (10): [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
```

## 3. The authority, and why it had to be vendored

The vendored `llama-graph.cpp` has the generic `build_moe_ffn`
(`docs/reference/llama-graph.cpp:2227`), but the vendored `qwen35.cpp` is the
**dense** architecture and asserts `Qwen3.5 does not use MoE FFN`. There was no
`qwen35moe.cpp` and no `LLAMA_ARCH_QWEN35MOE`, so the shared-expert wiring had no
authority in the tree at all.

`docs/reference/qwen35moe.cpp` is now vendored: **unmodified**, from llama.cpp
commit **`d7725699`** (build 11820 — the revision every reference number in this
repository is measured against), path **`src/models/qwen35moe.cpp`**. It is 735
lines; the FFN is `build_layer_ffn` at line 488. `docs/reference/README.md` and
`NOTICE` record the revision, since it is newer than the `b12126` copies.

## 4. The conventions

Each one is stated where it is implemented, in `crates/hipml-core/src/moe.rs`.
Line numbers are the vendored copies.

1. **Router logits** are `ffn_gate_inp · x` — `[256, 2048] · [2048]`, no bias
   (`exp_probs_b` is null at this call site).
2. **Softmax is over all 256 experts**, not over the selected 8
   (`llama-graph.cpp:2279`).
3. **Top-k = 8, value descending, ties by expert id ascending.** llama.cpp's
   CPU argsort is `std::sort` with a strict `>` comparator, so equal values are
   left in an unspecified order; id-ascending is this engine's deterministic
   refinement, and it is the rule the GPU kernel must use too. A 256-way f32
   tie is unreachable with real weights, but the rule is written down because a
   disagreement here is silent. `top_k` in `moe.rs` is the only implementation;
   the test pins `[1, 3, 4]` for a three-way tie.
4. **The 8 weights are the unbiased softmax probabilities, renormalised by their
   sum, clamped at `6.103515625e-5`** (the smallest normal f16,
   `llama-graph.cpp:2381`), and **no `expert_weights_scale`**: the key is absent
   from this architecture's files, so llama.cpp's `0.0` default leaves the
   scaling off (`:2390`).
5. **Per expert: `up`, then `gate`, then `silu(gate) * up`, then `down`, then
   multiply by the router weight** — `:2434`, `:2447`, `:2490`, `:2560`,
   `:2586`. The weight is applied **after** the down projection; only llama4
   applies it before the FFN.
6. **The combine is a left fold in top-k rank order**:
   `((out[0] + out[1]) + out[2]) + …` (`:2608`), not a sum over expert id. The
   rank order is descending router probability, which is the order `weights`
   was gathered in.
7. **The shared expert** is a plain dense SwiGLU (`LLM_FFN_SILU, LLM_FFN_PAR`,
   `qwen35moe.cpp:511-517`) scaled by
   `sigmoid(dot(x, ffn_gate_inp_shexp))` — **one dot product, one scalar for the
   whole vector** (`:523-533`) — and added after the MoE sum (`:535`). It is not
   a 257th expert.

Arithmetic follows llama.cpp's CPU kernels rather than being a second f64
implementation: f32 projections and f32 elementwise work, with f64 only where
llama.cpp accumulates into `ggml_float` (a `double`) — the softmax sum
(`ggml_vec_soft_max_f32`) and the router-weight sum (`ggml_sum_rows`). Matching
the accumulation is what makes the id gate meaningful; an all-f64 reference
would be a different engine that agrees on nothing bit for bit.

The CPU expert slice needs no new tensor type: a GGML `[ne0, ne1, n_expert]`
tensor is `ne1 * n_expert` rows of `ne0` values, with the experts stacked along
the row axis, so expert `e` is the row range `e * rows`. `Mat::gemv_range` is
that slice. On the device side, `gpu_weights::Experts` is the same idea as a
base pointer plus `stride`; `Proj::of` still rejects anything that is not 2-D,
because a dense tensor with a third dimension is a shape error.

## 5. The gate: 64 ids against llama.cpp

The reference ids came from `llama-server` on the same file, build 11820
(`d7725699`), greedy:

```powershell
llama-server -m <file> -ngl 99 -c 4096 `
  --host 127.0.0.1 --port 8092 -np 1
# POST /completion
# {"prompt":"Write a short paragraph about the water cycle.","n_predict":64,
#  "temperature":0.0,"top_k":1,"seed":0,"cache_prompt":false,"return_tokens":true}
```

The prompt tokenizes to the same 9 ids through llama's `/tokenize` and
`hipml tokenize`: `7734 264 2716 13901 883 279 2919 10458 13`. The 64 reference
ids and the exact command are committed in
`bench/fixtures/moe_gate_qwen36.json`; `hipml moe-gate --gguf <file> --fixture
<json>` runs the CPU reference over the same prompt and requires the same 64
ids. On a divergence it prints the logit of each id and the top-2 margin, so a
convention error (large margin) is distinguishable from arithmetic precision
(near-tie).

```
moe-gate: qwen35moe in 15.7s (40 decoder layers, 256 experts, 8 used, expert ffn 512,
          shared expert 512, ffn 4096)
prompt:    9 ids, 64 reference ids
cpu:       64 ids in 44.4s (0.69 s/token)
PASS: 64 / 64 ids identical to llama.cpp
```

The 64 ids (llama.cpp and the reference, identical):

```
271 248068 198 90700 8340 25 271 16 13 220 2972 1847 7355 279 5952 64700 198
262 348 256 31819 25 9664 10458 13 198 262 348 256 14606 25 10382 13901 13
271 17 13 220 2972 27382 1386 5141 73182 64700 198 262 348 256 3437 369 279
2919 10458 30 1049 579 279 18677 6977 314 2919 383 11 3294
```

Why this is the strongest cheap gate: an id stream is not tolerant. A logit
comparison with a tolerance hides a routing difference, because one wrong expert
out of eight moves the logits by less than the tolerance. A token flip does not
hide. 64 ids cover 40 layers × 64 steps of routing decisions, and a single wrong
convention anywhere in §4 changes the stream.

## 6. The file opens

`hipml load` on the MoE file, with the HIP build asserted (`built with hip
feature: true`):

| quantity | value |
| --- | ---: |
| tensors | 753 |
| tensor data | 22,842,671,616 B (21.27 GiB) |
| read + upload | 6.97 s (3.28 GB/s) |
| resident delta | 22,991,470,592 B (21.41 GiB) |
| BF16 tensors | 2, 1,052,672 B, supported |
| unsupported | **none** |
| round-trip | 20/20 byte-identical |

The five dtypes are all ones the engine already had; the two that carry the
experts are Q5_K and Q4_K, which is why the 8-of-256 gather streams 613 MB per
token and not the whole bank:

| dtype | tensors | bytes | GiB |
| --- | ---: | ---: | ---: |
| F32 | 368 | 104,624,640 | 0.10 |
| BF16 | 2 | 1,052,672 | 0.00 |
| Q8_0 | 260 | 2,614,820,864 | 2.44 |
| Q6_K | 3 | 660,602,880 | 0.62 |
| Q5_K | 40 | 7,381,975,040 | 6.88 |
| Q4_K | 80 | 12,079,595,520 | 11.25 |

The embedding probe now runs on this file. `token_embd.weight` is Q8_0, and the
row gather decodes every format that has a decoder, so the loader's five probes
are exact on the device against the host dequant — 2048/2048 values each,
`max_abs 0.000e0`:

```
embedding rows (token_embd.weight [248320, 2048] Q8_0; device dequant vs CPU dequant):
  token        exact/values      max_abs      max_rel
  0               2048/2048      0.000e0      0.000e0
  1               2048/2048      0.000e0      0.000e0
  1234            2048/2048      0.000e0      0.000e0
  100000          2048/2048      0.000e0      0.000e0
  248319          2048/2048      0.000e0      0.000e0
```

`lm_head` on the resident Q8_0 head runs at 138.5 GB/s (3.900 ms for
540,344,320 B).

## 7. The device path

Four pieces were needed, and only the first is about this file specifically.

### 7.1 One row gather, every format

`token_embd.weight` is a gather, not a matvec: the token id selects one row and
nothing else is read. The old kernels (`embed_q4_k_row`, `embed_q6_k_row`)
existed because the first two files happened to use those formats, and
`GpuEngine::new` rejected everything else — which is how a Q8_0 embedding table,
the commonest choice for a 248k vocabulary, stopped a 22.85 GB file from
opening at all. A row gather has no shape constraint that could justify a
per-type allowlist.

`kernels/dequant_rows.hip` is one decoder with a `kind` selector:

| kind | format | unit bytes per 256 values |
| ---: | --- | ---: |
| 0 | F32 | 1024 |
| 1 | BF16 | 512 |
| 2 | Q8_0 | 272 |
| 3 | Q3_K | 110 |
| 4 | Q4_K | 144 |
| 5 | Q5_K | 176 |
| 6 | Q6_K | 210 |
| 7 | IQ4_NL | 144 |
| 8 | IQ4_XS | 136 |
| 9 | IQ3_S | 110 |

One 256-thread block per 256-value unit, one value per thread; for the K-quants
that unit is exactly one super-block. Two entries share the decoder:
`dequant_row_any` takes the row address on the host (the loader's probe), and
`embed_row_any` reads the row index from the captured step buffer, so the
graph's embedding needs no host argument. `row_kind` in
`launch/device/graph_launch.rs` maps a `TensorType` to a kind by name and is
called in `GpuEngine::new` and `BatchEngine::new`, so a type with no decoder
fails at load, naming the type, instead of at the first token.
`ROW_BLOCK_VALUES` (256) is the host mirror of `DR_VALUES`.

The proof is `hipml row-dequant`: for each format it builds deterministic block
bytes with the same generators the host reference uses, dequantises one
2048-value row on the device, and compares bit for bit. Every format is exact,
which is the expected result — both sides walk the block in the same order and
do the same f32 arithmetic:

```
  type       exact/values      max_abs       status
  Q4_K          2048/2048      0.000e0 exact
  Q5_K          2048/2048      0.000e0 exact
  Q6_K          2048/2048      0.000e0 exact
  Q3_K          2048/2048      0.000e0 exact
  IQ3_S         2048/2048      0.000e0 exact
  IQ4_XS        2048/2048      0.000e0 exact
  Q8_0          2048/2048      0.000e0 exact
  IQ4_NL        2048/2048      0.000e0 exact
  F32           2048/2048      0.000e0 exact
  BF16          2048/2048      0.000e0 exact
PASS: every format decodes the same bits on device and host
```

It earned its keep immediately. The first IQ3_S arm read the sign of the second
four-value group at the host's local index (`jj - 4`) instead of the thread's
index (`jj`), so four of every eight values took the sign of a different value.
`row-dequant` reported 1024/2048 differing, `HIPML_ROW_DEBUG=1` printed the
first eight, and the split was exactly `jj < 4` against `jj >= 4`. The 4-bit
grid itself is now `kernels/iq3s_grid.inc`, included by both `gemv_iq3_s.hip`
and `dequant_rows.hip`, so the GEMV and the row decoder cannot index different
tables.

### 7.2 A block's FFN is an enum

`Block` held three `Proj`s; it now holds `ffn: Ffn<'a>`, either
`Dense { gate, up, down, jobs }` or `Moe(Box<MoeBlock>)`. An enum rather than
three options because the shapes genuinely differ — a MoE layer has no
`ffn_gate.weight` at all — and every caller has to name which one it means.
`MoeBlock` (`gpu/moe.rs`) owns the router, the three `Experts` bank views, the
shared expert, and one step's scratch. `Experts` is the same idea the CPU slice
used (`Mat::gemv_range`): a base pointer plus one expert's byte `stride`.
`Proj::vector` is the one place a 1-D tensor is a projection, for
`ffn_gate_inp_shexp` whose "matrix" is `[1, n_embd]`.

The batched engine and `column-bench` refuse a MoE file up front with a message
that names the reason: neither has a routed FFN body. That is deliberate — a
missing-tensor error from `Proj::of` would be the same failure with a worse
explanation.

### 7.3 Top-k and the gather stay on the device

The requirement that decides the whole design: **no per-layer host readback**.
`run_token` is captured once into a hipGraph, and a readback of the selected
ids would end capture. So the router decision is device data end to end:

1. `router.gemv` — the F32 `[256, 2048]` router, one GEMV to 256 logits.
2. `moe_topk_f32` — one block: softmax over all 256, eight selection passes
   (value descending, id ascending), renormalise by the sum clamped at
   `6.103515625e-5`. Writes `ids[k]` (u32) and `weights[k]` (f32). Both stay
   resident across steps. `MOE_TOP_K_MAX` is 1024 experts; a wider bank is
   refused rather than read past the shared array.
3. `moe_gemv` — one launch per bank. `grid.y` is the top-k rank and the kernel
   resolves `w + ids[j] * stride` itself; `x_stride` is 0 for gate and up (all
   ranks read the same activation) and `n_ff_exp` for down (each rank reads its
   own). It is `gemv_jobs` with the job's weight pointer resolved on the device
   and calls the same tile bodies (`q4k_tile`, `q6k_tile`, `q5k_rows`,
   `q80_row`, `iq4x_tile`), so an expert row is bit-identical to the same row
   through `gemv_jobs`. Those five formats are the ones an expert bank uses;
   `MoeBlock::new` fails at load, naming the bank, if one has no body.
4. `silu_mul_buf`, then `moe_combine_f32` — `((w0*d0 + w1*d1) + …)`, a left
   fold in rank order, which is the convention §4.6 pinned.
5. The shared expert: three dense GEMVs, `silu_mul`, `sigmoid_f32` over the
   single `dot(x, ffn_gate_inp_shexp)` value, and `moe_shared_add_f32`. One
   scalar, added outside the routed sum.

`MoeBlock::route_host` exists for the routing gate and says so: it is a
synchronising readback and nothing on the decode path calls it.

### 7.4 Prefill is the decode path, once per token

A routed chunk has no grouped GEMM, so `forward_chunk` hands a MoE model to
`forward_chunk_token_loop`: one `run_token` per prompt token, in order, with
`head = false` for all but the last. `head = false` still runs `output_norm`
(so a capture has every row) and skips only the 540 MB `lm_head` projection,
which is worth about 1 ms a token. `capture` still works — each token's row is
copied into the capture buffer after its step, which is what `ppl` reads.
`ensure_chunk` is a no-op for a MoE model: the chunk scratch would be hundreds
of MiB that this path never touches.

That makes a MoE layer 17 launches a token — residual, rmsnorm, and 15 in the
FFN, of which 14 are the routed path — and the whole token 1,103. §8 reports
what that costs.

## 8. What the device path measures

`moe-gate --device` runs the resident engine over the same 9 prompt ids and the
same fixture, and the id stream is the gate:

```
resident:  21.27 GiB in 753 tensors on AMD Radeon AI PRO R9700 (upload 11.2s)
moe-gate: qwen35moe (40 decoder layers, 256 experts, 8 used, expert ffn 512, shared expert 512, ffn 4096)
graph:     40 layers, 2.56 GiB streamed/token, 0.8 MiB KV at ctx 81 (build 13857 ms)
prefill:   9 ids in 0.06s (139.7 tok/s, one decode step per token)
decode:    63 steps in 0.49s (127.70 tok/s)
vram:      21.69 GiB resident of 31.86 GiB
PASS: 64 / 64 ids identical to llama.cpp
```

The 169-token prompt `bench/fixtures/p2_math_169.ids` measures the loop without
the gate's per-step bookkeeping, two reps:

| rep | prefill | decode |
| --- | ---: | ---: |
| 0 | 143.15 tok/s (169 tokens in 1180.6 ms) | 126.21 tok/s (15 steps, 118.9 ms) |
| 1 | **143.99 tok/s** (169 tokens in 1173.7 ms) | **125.76 tok/s** (15 steps, 119.3 ms, **7.95 ms/token**) |

Prefill is faster than decode because the loop runs `lm_head` once for the
whole prompt instead of once per token (§7.4).

### 8.1 Against the roofline

The recon projected **176 tok/s** at 4,096 context from 2.921 GB/token at
514.6 GB/s (the engine's measured 80.4% of 640 GB/s), and **≈24.5 GB** peak
VRAM. Measured: 125.76 tok/s, 21.69 GiB (23.29 GB).

That is 71% of the prediction, and the profile says where it went rather than
leaving it as "the gather is slow":

| quantity | value |
| --- | ---: |
| streamed per token | 2,750,765,568 B (2.56 GiB) |
| token, captured graph | 8.102 ms (warm 8.553 ms) |
| effective bandwidth | 346 GB/s |
| weight stream inside the token | 4.352 ms at **632 GB/s** |
| overhead (launch + elementwise) | 3.749 ms |
| launches/token | 1,103 |
| launch cost (cached) | 2.605 µs, so a 2.873 ms floor |

The GEMV stream inside the token runs at 632 GB/s — the sequential rate the
engine already reaches, and above the 514.6 GB/s the projection assumed. The
shortfall is the other 3.749 ms: 1,103 launches at 2.605 µs are 2.873 ms of it,
35% of the token. The MoE FFN is 560 of those launches, and the trace puts it
first:

| stage | ms/token | share | launches | GB/s |
| --- | ---: | ---: | ---: | ---: |
| `ffn_moe` | 2.846 | 32.7% | 560 | 292.2 |
| `ssm_proj` | 2.099 | 24.1% | 120 | 389.7 |
| `lm_head` | 0.789 | 9.1% | 2 | 684.6 |
| `ssm_out` | 0.627 | 7.2% | 30 | 426.6 |
| `attn_qkv` | 0.606 | 7.0% | 20 | 478.2 |
| `post_norm` | 0.459 | 5.3% | 40 | |
| `ssm_step` | 0.295 | 3.4% | 150 | |
| `residual_out` | 0.265 | 3.0% | 40 | |
| `residual_in` | 0.238 | 2.7% | 40 | |
| `attn_wo` | 0.219 | 2.5% | 20 | |
| `attn_kv` | 0.134 | 1.5% | 10 | |
| `attn_core` | 0.088 | 1.0% | 20 | |
| `attn_norm` | 0.033 | 0.4% | 40 | |
| `embed` | 0.019 | 0.2% | 1 | |
| `out_norm` | −0.005 | −0.1% | 1 | |
| sum | 8.712 | 100% | 1,094 | |

So `ffn_moe`'s apparent 292 GB/s is not a gather-bandwidth problem: the stage is
14 launches a layer, each a few microseconds of work, and the launch cost is
most of its 2.846 ms. Fusing the router and the shared-expert scalars into
`gemv_jobs`, and the 8 ranks of a bank into one launch, is the tuning the recon
lists and this slice does **not** do — the number is reported, not chased.
(The launch floor is measured with empty launches, so in a real token part of it
hides behind execution; the split above is the profile's attribution, not
something this slice proved kernel by kernel.)

VRAM is also below the projection: 21.69 GiB with the engine's scratch at the
gate's 81-token context. The KV cache grows with context, so that is a
measurement, not a ceiling.

### 8.2 Perplexity

`hipml ppl` on an 849,455-token calibration corpus (`--ctx 512 --chunks 4`, f32
KV, 1,020 scored positions) against `llama-perplexity` on the same tokens
(`-c 512 -b 512 -ub 512 --ppl-stride 0 --chunks 4 -ngl 99`):

| window | hipml | llama.cpp | delta |
| ---: | ---: | ---: | ---: |
| 1 | 2.2926 | 2.2821 | +0.46% |
| 2 | 2.0091 | 2.0038 | +0.26% |
| 3 | 2.1946 | 2.1856 | +0.41% |
| 4 | **2.2275** | **2.2228** | **+0.21%** |

The dense 27B on the same build is unchanged at 2.1824 against llama's 2.1791,
so the device path did not move the dense number.

### 8.3 The routing diagnostic

`moe-gate --device` also diffs every layer's router decision against the CPU
reference at the last prompt token. It is a diagnostic, not a gate — the id
stream is the gate — and it reads:

```
routing:   40 layers; 30 identical, 8 rank-order only, 2 selected a different expert set
routing:   worst |dweight| 5.580e-3 over shared expert ids
```

The two set differences are layer 3 and layer 5, each swapping the 8th-ranked
expert (31→231, 236→139), and the eight "rank-order only" layers pick the same
eight experts in a different order. That is what f32 router arithmetic looks
like when the GPU GEMV and the CPU GEMV sum in a different order: the weights
differ by at most 5.6e-3 and the ids do not. A convention error would move the
whole set, not the last rank.

## 9. Gates

On the binary this branch ships, `--features hip`, model paths passed with
`--gguf`, `HIPML_MTP_GGUF` removed from the process environment (not blanked)
and `--mtp` passed literally where a drafter is wanted.

| gate | expected | observed |
| --- | --- | --- |
| `moe-gate` CPU (Qwen3.6-35B-A3B) | 64/64 ids | **PASS 64/64**, 0.69 s/token |
| `moe-gate --device` | 64/64 ids | **PASS 64/64**, 127.70 tok/s, 21.69 GiB |
| `row-dequant` | 10 formats exact | **10/10 exact**, 2048/2048 values each |
| `load` (Qwen3.6-35B-A3B) | 0 unsupported, 256/8/512/512 | **0 unsupported**, 21.41 GiB, 20/20 round-trip, Q8_0 rows exact |
| `generate --device`, 169-token prompt | tokens on the GPU | **143.99 tok/s prefill, 125.76 tok/s decode** |
| `ppl` (Qwen3.6-35B-A3B) | close to llama | **2.2275** vs 2.2228 (+0.21%) |

Dense path, same binary, so the new `Ffn` enum and the row gather did not move
anything that was already gated:

| gate | expected | observed |
| --- | --- | --- |
| `oracle-gate` UD, 20 steps, chunk 2048, q8 | PASS 20/20, 0.1086 at 101748 | PASS 20/20, **0.1086 at 101748** |
| `oracle-gate` stock, 20 steps | PASS 20/20, 0.1481 at 34114 | PASS 20/20, **0.1481 at 34114** |
| greedy 12 ids, stock | `11751 13 198 760 6511 314 9564 369 19241 13 198 760` | identical |
| `parity --deep` | 74 passed | **74 passed**, 0 failed, 1 skipped (`sample_top_p_f32`) |
| `parity prefill --deep` | 15 passed | **15 passed**, 0 failed, 0 skipped |
| `batch-bench --slots 16 --check` | 16/16 on every slot | 16 slots, each **16/16** |
| `spec-bench --drafts 3 --kv q8` | 128/128 exact every row | **128/128** on all four rows |
| `spec-bench --drafts 3 --kv f32` | 128/128 exact every row | **128/128** on all four rows |
| `column-bench --check` | 0 differ | **0 differ** on 7,946,240 logits, hidden and residual |
| `probe_exact.py --ctx 32768` | PASS | PASS, 48 replies id-for-id across drafts 1,2,3 |
| `ppl` (Arete-27B) | 2.1824 vs llama 2.1791 | **2.1824**, unchanged |
| `cargo test --workspace` | clean | clean (local and on the GPU host) |
| `cargo fmt --all -- --check` | clean | clean |
| `scripts/check-public-tree.sh` | clean | clean |
| `scripts/check-doc-links.py` | all links resolve | 271 links in 52 files |

The two oracle maxima are the same numbers every slice since the prefill-tile
note reports, which is the expected result for a change that does not touch the
dense forward pass.

## 10. What is not verified, and what is out of scope

* **MTP.** `blk.40` is a MoE block and `mtp.rs:129-131` requires a dense
  `ffn_gate`, so the drafter cannot load this file. The two BF16 tensors are
  MTP-only; the type exists so the file opens, not so the block runs.
* **Grouped GEMM / bucketed prefill.** A MoE chunk runs the decode path once per
  prompt token (§7.4). That is correct and it is 143.99 tok/s, but it re-streams
  the whole model per prompt token: a 2,000-token prompt takes about 14 s, where
  the bucket-sort + grouped-GEMM path the recon describes would read each expert
  once per tile.
* **No tuning and no fusion.** The router, top-k, combine and shared-expert
  scalar are separate launches; the 8 ranks of a bank are one launch but the
  three banks are three. §8.1 names what that costs. Nothing here was tuned to
  reach the 176 tok/s projection.
* **Batch/serve and `column-bench`.** Both refuse a MoE file up front, naming
  the reason: neither has a routed FFN body. The single-sequence decode path is
  the one that does.
* **One prompt, one seed, 64 ids.** The gate says the conventions are right; it
  does not bound how often a near-tie would flip a token between the f32 CPU
  reference, this GPU path and llama's GPU kernels. The margin print in
  `moe-gate` and the routing diff in §8.3 are the tools for that if it ever
  happens.
* **The load-bearing assumption** is that llama.cpp's MoE path on the ROCm
  backend and its CPU path agree on these ids. Only the GPU path was run; the
  box did not have the RAM for a CPU llama-server beside the loaded GPU model.
  If a later slice sees a divergence, that is the first control to repeat.
