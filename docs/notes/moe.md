# The MoE FFN: the conventions, settled against llama.cpp's ids

> *Engineering log from development; internal work-package IDs and paths have
> been sanitized or removed where they were machine-specific.*

`Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` is the first mixture-of-experts file the
engine has opened: 753 tensors, 22,853,663,008 B, `qwen35moe`, 40 decoder blocks
plus one inline MTP block, 256 experts with 8 used. Every attention and
gated-delta-net tensor in it is shaped exactly like the 27B's; the FFN is the
only new thing.

This note covers three slices. The first settled the numerical conventions on the
CPU, before any kernel existed, because four independent conventions (softmax
domain, top-k order, renormalisation, where the router weight is applied) can
each leave the ids subtly wrong, and a kernel built on a wrong one is the wrong
kernel. The second built the device path — a row gather for every format, an
expert GEMV that resolves the gather on the device, device-side top-k, and a
prefill loop — and it is the first time this file produces tokens on the GPU.
The third measured the file against llama.cpp on the same box (§8.4) and built a
bandwidth instrument to find why the small expert reads run at a third of the
large-read rate (§8.5).

**Result: the CPU reference reproduces llama.cpp's 64 greedy ids on a fixed
prompt, 64/64, first try; the device path reproduces the same 64 ids, 64/64, at
125.76 tok/s decode and 143.99 tok/s prefill.** Against llama.cpp on the same
file, hipml decodes **33.5% faster at depth 0 and 24.0% faster at depth 8,192**
and prefills **95.6–96.7% slower**; the 8-slot serve cell is `n/a` because the
batched engine has no routed FFN. The bandwidth instrument finds the memory
system can read 0.59 MB at 495 GB/s where the shipped tile gets 182, and
installing the wider tile is **neutral end to end**, so 150 tok/s stays out of
reach (§8.5). The file loads: 753 tensors, 0 unsupported, 21.41 GiB resident
(21.69 GiB with the engine's scratch).

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
VRAM. Measured: 125.5 tok/s (7.97 ms/token), 21.69 GiB (23.29 GB). That is 71%
of the prediction, and this subsection is about the other 29%.

An earlier version of this note answered that question with a **stage trace**
and got it wrong. It read the trace's 1,103 launches at the `gpu-profile`
"cached launch" figure of 2.605 µs and concluded a 2.873 ms launch floor, 35% of
the token. Three measurements since say that number is not a GPU cost at all.

#### The launch is not the node

`probe_launch.cpp` launches N kernels that do nothing, N = 64 … 4096, eager and
captured, and times them end to end:

| N | empty, eager µs/node | empty, graph µs/node | 4 KiB read, graph µs/node |
| ---: | ---: | ---: | ---: |
| 64 | 0.474 | 0.418 | 0.492 |
| 256 | 0.573 | 0.507 | 0.454 |
| 512 | 0.594 | 0.437 | 0.445 |
| 1024 | 0.486 | 0.436 | 0.496 |
| 2048 | 0.588 | 0.436 | 0.473 |
| 4096 | 0.486 | 0.489 | 0.482 |

A node costs **0.43 µs** on the GPU timeline, flat from 64 to 4,096, and a node
that reads a page costs the same as one that reads nothing. Eager submission is
0.58 µs. So the token's 1,103 launches are worth **0.47 ms, 6% of the token**,
not 2.873 ms and 35%. The 2.605 µs `gpu-profile` reports is the host's cost to
issue a launch, and the host is a token ahead of the GPU, so it is not on the
critical path at all.

The same probe found the number that matters more. A single `hipGraphLaunch` of
a **one-node** graph takes **1.04 ms** on this runtime. The token's captured
graph therefore costs more than the 0.64 ms of host issue it replaces, and the
engine's own two paths agree: the same 200 tokens through the same engine, one
lock hold, three reps each:

| path | ms/token | tok/s |
| --- | ---: | ---: |
| eager | **7.838** | 128.77 |
| one captured `hipGraphLaunch` | 8.853 | 112.96 |

**Capturing the token makes it 13% slower.** That is why the shipped decode path
is eager, and it retires "put the token in one graph" as a lever on this card for
this file. (The dense path reaches the same verdict from the other side:
[`decode-dispatch.md`](decode-dispatch.md) §2a measured graph replay +0.37 ms
against 1,299 eager launches, and [`decode-ablation.md`](decode-ablation.md) §3
measured the whole GEMV set as one launch at 28.548 ms against 28.524 eager.)

#### The ablation, not the trace

`hipml decode-ablate` now carries the routed FFN as six ablation groups
(`moe_router`, `moe_topk`, `moe_bank`, `moe_act`, `moe_combine`, `moe_shared`)
instead of one, and `--cells` accepts any ablation spec so a stage can be priced
inside a confound. A dropped stage issues no launch and records no event, so it
cannot contribute to the number it is used to price. Depth 0, 200 tokens, three
reps, cells interleaved in one process:

| cell | ms/token | tok/s | what it prices |
| --- | ---: | ---: | --- |
| `full` | **7.838** | 128.77 | the token |
| `gemv_only` | **5.777** | 173.09 | every projection and nothing else — 476 GB/s |
| `-moe` | 4.891 | 204.48 | the whole routed FFN, 560 launches |
| `-ssm` | 5.141 | 194.53 | the whole GDN block, weights included |
| `-attn` | 6.998 | 142.90 | the whole attention block, weights included |
| `-lmhead` | 6.893 | 145.07 | `output.weight`, 540 MB in 2 launches, 571 GB/s |
| `-ssm_rec` | 7.433 | 134.54 | the delta-net recurrence, 30 launches |
| `-attn_reduce` | 7.611 | 131.39 | the decode attention core, 20 launches |
| `-norm` | 7.613 | 131.35 | the 81 RMS norms, 2.8 µs a launch |
| `-ew` | 7.789 | 128.39 | 80 residual adds and `silu`, **0.6 µs a launch** |

`full` is 5.777 ms of weight stream plus 2.061 ms of everything else, and the
2.061 is kernels. The `-ew` row is the sharpest statement of that: **80 elementwise
launches cost 0.049 ms between them**, a sixteenth of what the trace attributed
to `residual_in` + `residual_out` alone. A launch in this stream is worth
0.6 µs where the kernel is 8 KB wide, and 12.9 µs where it reads 4.7 MB, and
0.43 µs of either is the node.

#### The routed FFN, group by group

The six groups sum to 2.947 ms. Priced against `-moe_bank` rather than `full`,
because dropping the top-k alone leaves every routed rank reading expert 0 and
the L2 then serves seven of the eight — the naive `-moe_topk` cell reads 0.74 ms
and is 60% confound:

| group | launches | ms | µs/launch |
| --- | ---: | ---: | ---: |
| the three bank GEMVs | 120 | **1.545** | 12.9 |
| the shared expert (3 GEMVs + silu + dot + sigmoid + add) | 280 | **0.734** | 2.6 |
| the top-k, one block, eight selection passes | 40 | **0.326** | 8.2 |
| the router GEMV, 2.1 MB of F32 | 40 | **0.204** | 5.1 |
| the rank fold | 40 | 0.150 | 3.8 |
| `silu_mul` on the ranks | 40 | 0.088 | 2.2 |

The banks are the largest single item in the token after the stream itself, and
they are **not** dispatch-bound. `gemv+moe_bank*2` issues the down-bank launch a
second time at its own site: +0.570 ms for 40 more 4.7 MB reads, i.e. 14.3 µs a
launch and **342 GB/s**. One more launch there costs 14.3 µs and 0.43 µs of that
is the node, 3%.

That is the whole answer to "is the token launch-bound": **no.** The banks read
566 MB in 1.545 ms = 366 GB/s, and the reason is the size of the read, not the
count of the launches. Rate rises with the size of the read on this card:

| launch | bytes | GB/s |
| --- | ---: | ---: |
| `output.weight` | 540 MB | 571 |
| the router GEMV | 2.1 MB | 466 |
| one routed bank | 4.7 MB | 366 |
| one shared-expert GEMV | 0.59 MB | 125 |

But size is not the only variable, and an earlier version of this note made it
the whole story. §8.5 sweeps the grid at a fixed byte count and finds the rate
moving **2.6×** — 182 to 472 GB/s at 0.59 MB — on geometry alone. The honest
statement is that bytes per launch sets the ceiling and the number of bytes in
flight sets how close a small read gets to it; the shipped tiles sit low on that
curve, and widening them does not move the token (§8.5).

The routing structure fixes the size. Gate, up and down are three sequential
launches of eight scattered expert slices each — 4.7 MB, then 4.7 MB, then
4.7 MB — and no merge changes what the DRAM sees, which is what the fusion
table below measured.

#### The fusions: four built, four measured, none kept

The card's hypothesis was that cutting graph node count closes the gap. Four
node-count reductions were built and each was measured **end to end, untraced,
alternating inside one lock hold** against a base binary of the parent commit.

| fusion | launches | arithmetic | measured |
| --- | ---: | --- | --- |
| shared expert gate+up as one `gemv_jobs` list | −40 | bit-exact: same tile body, each writes its own buffer | +1.1 tok/s in one bisect, **neutral** in the interleaved A/B |
| gate+up banks in one two-bank launch | −40 | bit-exact: same tile body, rank j still resolves its own expert | **neutral** |
| the GDN's four projections in one job list (needed an F32 job body for `ssm_alpha`/`ssm_beta`) | −90 | bit-exact | **−4.3 tok/s**: the two 32-row F32 matrices share a 512-tile grid, so 960 blocks a layer retire without work |
| the top-k's eight 256-way barrier trees as warp reductions | 0 | bit-exact: max is exact and associative, the tie-break is an integer compare | **neutral** — the kernel was never barrier-bound |

None is kept. A fusion that removes 40 to 90 launches and leaves the token where
it was is not a win, and the shipped decode path is byte-identical to the parent
commit. The instruments are kept instead: `probe_launch.cpp` and the six-group
ablation, because they are what refuted the hypothesis and what any later slice
should re-run before spending a day on launch count.

The one measurement that does move is the graph: **not capturing the token is
worth 1.02 ms, 13%.** The engine already does not capture it on this path.

#### Where 150 tok/s would have to come from

150 tok/s is 6.67 ms, 1.17 ms below the measured token. The entire dispatch bill
is 0.47 ms and the graph is already off, so no arrangement of launches reaches
it. What is left, in order of size:

* **476 GB/s on the stream, not 632.** 2.75 GB at 632 is 4.35 ms; the token's own
  projections read 476 GB/s with nothing else on the card, because the reads are
  0.59–4.7 MB. Closing this is a kernel-shape question — more bytes in flight per
  launch at a fixed read size — and the four launch merges above are evidence
  that it is not a merge-count question. §8.5 measures the shape directly and
  finds the headroom is real in isolation (0.59 MB: 182 → 472 GB/s on geometry)
  and **not taken by the token**: installing the wider tile was neutral, so the
  remaining work is on the tiles that were not re-cut (the Q5_K/Q6_K down banks)
  and on the arithmetic, not on the load width of the Q4_K path.
* **The GDN block, 2.697 ms** for 30 layers, of which 0.406 is the recurrence and
  the rest is five projections and four elementwise steps a layer.
* **The shared expert, 0.734 ms** for 71 MB: seven launches a layer reading
  0.59 MB each at 125 GB/s.
* **The top-k, 0.326 ms** for 40 single-block launches. It is eight dependent
  selection passes over 256 experts on one CU; only a different algorithm moves
  it, and the reduction tree is not what it spends its time on.

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

### 8.4 Against llama.cpp, on this file

The dense matrix measured the 27B against llama.cpp; this is the same protocol on
the MoE file. One GPU-lock hold, one model resident at a time, the two engines
interleaved per depth (hipml, llama, llama, hipml) so a drift in the box lands on
both, three reps per process and six per cell, q8 KV on both, the same 512-token
fixture. Peak VRAM comes from the same external WDDM sampler for both engines.
hipml is `786b8c88…`, 3,111,424 B, `built with hip feature: true`; llama.cpp is
`0.7.0-dev (build 11820, commit d7725699)` with `-ngl 99`.

| cell | hipml tok/s | llama.cpp tok/s | delta | ranges separate |
| --- | ---: | ---: | ---: | --- |
| prefill 512 @ depth 0 | 147.10 (146.70–148.39) | 3375.05 (1528.65–3665.29) | **−95.6%** | yes |
| decode @ depth 0 | **128.24** (127.81–129.02) | 96.04 (94.20–96.50) | **+33.5%** | yes |
| prefill 512 @ depth 8192 | 137.66 (136.17–138.60) | 4203.73 (3934.29–4232.96) | **−96.7%** | yes |
| decode @ depth 8192 | **115.05** (113.99–115.56) | 92.79 (91.96–93.31) | **+24.0%** | yes |

Every value is a median of six reps and the parentheses are min–max. llama.cpp's
prefill rep 0 is a cold outlier at both depths (1,528 and 1,886 tok/s); its reps
1–2 are 3,300–3,660 at depth 0 and about 4,200 at depth 8,192, so the ranges
above include the cold rep and dropping it would make the delta slightly worse
for hipml.

**Decode: hipml wins, at both depths, and the ranges do not overlap.** 33.5% at
depth 0 and 24.0% at depth 8,192. Both engines lose speed as the KV cache grows
(hipml −10.3%, llama −3.4%), so the win narrows with depth but does not close.

**Prefill: hipml loses by an order of magnitude.** 147 tok/s against 3,375 and
138 against 4,204. The cause is §7.4: a MoE chunk runs the decode path once per
prompt token, so prefill re-streams 2.56 GiB per prompt token. At 147 tok/s that
is a ~405 GB/s effective stream, the same wall as decode — llama.cpp's grouped
GEMM reads each expert once per tile instead of once per token. This is the one
cell where the engine is not competitive, and it is a missing kernel, not a
missing convention.

Peak VRAM is 3.9% higher than llama.cpp at both depths:

| depth | hipml | llama.cpp |
| ---: | ---: | ---: |
| 0 | 22,372 MiB | 21,526 MiB |
| 8,192 | 22,459 MiB | 21,614 MiB |

Serve cell b is 8 slots at 16K context, a 2,384-token prompt and 64 tokens a
request, three reps:

| engine | aggregate tok/s p50 | per-stream tok/s p50 | TTFT p50 | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp | 379.0 | 47.5 | 324 ms | 23,548 MiB |
| hipml | **n/a** | — | — | 22,542 MiB |

hipml's cell is `n/a` for a reason in the engine, not a missing measurement:
`BatchEngine::new` refuses a MoE file — *"batch decode has no MoE FFN yet (256
experts, 8 used): the batched engine runs a dense FFN per block; the
single-sequence decode path does have the expert gather"* — so all 24 requests
fail and there is nothing to compare. llama.cpp's cell fits (23,548 MiB of
31.86 GiB) and is the number an 8-slot MoE server would have to beat.

### 8.5 The bandwidth wall, measured

`hipml bw-probe` is new in this slice. It asks §8.1's question with nothing else
in the kernel: at the byte count of each read the token issues, what does the
memory system deliver? It launches one read kernel `--launches` times back to
back with a single event pair around the burst, sweeps read size × bytes per load
(4/8/16) × independent loads per thread (1/2/4/8) × workgroups (32…4,096), and
subtracts the same grid's empty-kernel span. Small reads rotate over distinct
windows of one 2 GB allocation, so a burst never re-reads bytes still in the
64 MB Infinity Cache; a rate measured out of cache is not the rate decode sees.
The kernel has no dequant, no staging and no reduction — one XOR chain per
accumulator — so what it reports is memory and dispatch only.

Calibration and the two floor numbers the sweep has to respect:

* 540 MB reads at **639.1 GB/s** against the card's 640 GB/s.
* Host issue is **0.476 µs/launch** (queue only, no sync); the empty-kernel GPU
  span at 32 blocks is **0.539 µs**, which matches the 0.43 µs node §8.1 reports.
* A burst carries a **~80 µs fixed cost**, so `--launches` had to be raised to
  400+ before a 0.59 MB burst was above the noise. The table below is 128
  launches with the floor subtracted and 3 reps a cell, 624 cells in all.

Best cell per read size, out of 84 cells a size (`raw` is what one launch of that
shape delivers):

| read | bytes | best GB/s | geometry |
| --- | ---: | ---: | --- |
| shared-expert GEMV | 0.59 MB | 494.7 | 16 B, 1 in flight, 512 blocks |
| one expert down (q5_k) | 0.72 MB | 510.0 | 16 B, 8 in flight, 512 blocks |
| one expert down (q6_k) | 0.86 MB | 527.8 | 16 B, 1 in flight, 512 blocks |
| the router | 2.1 MB | 443.4 | 4 B, 4 in flight, 64 blocks |
| one routed bank | 4.7 MB | 526.4 | 16 B, 1 in flight, 1,024 blocks |
| one layer's banks | 18.9 MB | 597.8 | 16 B, 4 in flight, 1,024 blocks |
| `lm_head` | 540 MB | 639.1 | 16 B, 8 in flight, 4,096 blocks |

Two things fall out. First, **0.59 MB can be read at 495 GB/s**, four times the
125 GB/s §8.1 inferred for the shipped shared-expert GEMV: at that size the
memory system is not the limit. Second, **the sweep is not monotonic in any one
lever** — the best cell at 2.1 MB is 4-byte loads and 64 blocks, and at 0.59 MB
the 4,096-block grid is the worst cell in the whole sweep (105.9 GB/s) because
36,864 vectors cannot fill a million threads and the tail retires without work.
The levers only mean something together, which is what "bytes in flight" is.

The GEMV's own shape is the decisive table. It is 512 rows of a Q4_K gate/up tile
(0.59 MB), issued one warp per row and up, the same lane-strided body
`q4k_tile` runs; `raw` GB/s:

| rows/warp | warps | blocks | 4 B | 8 B | 16 B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 512 | 287.3 | 363.1 | **471.7** |
| 1 | 2 | 256 | 283.1 | 357.2 | 396.7 |
| 1 | 4 | 128 | 311.8 | 353.1 | 396.2 |
| 1 | 8 | 64 | 280.4 | 349.2 | 391.2 |
| 2 | 8 | 32 | 182.2 | 231.4 | 314.6 |
| 4 | 8 | 16 | 85.3 | 109.1 | 187.2 |

The shipped tile is the 32-block row — 8 warps, 2 rows a warp — and at 4-byte
loads it is **182.2 GB/s**: the shipped configuration, and below every
one-row-per-warp cell in the table. One workgroup per row at 16 bytes is
**471.7 GB/s, 2.6× faster, at the same byte count.** The Q5_K down tile has the
same shape of headroom: 2,048 rows, 223 GB/s at its shipped geometry against
454.6 at one row per block and 16 bytes.

So the tile was re-cut: `gemv_q4_k`'s stream entry now takes the 16-byte load and
`q4k_tile<1, true, false>` — one row per warp. It is bit-exact by construction
(same tile body, same order, wider loads). Measured against the base binary,
three reps a cell, cells interleaved, two rounds, one lock hold:

| cell | base ms/token | 16-byte tile ms/token |
| --- | ---: | ---: |
| `full` | 7.729 / 7.792 | 7.785 / 7.797 |
| `gemv_only` | 5.688 / 5.745 | 5.744 / 5.754 |
| `-moe_shared` | 6.963 / 7.026 | 7.017 / 7.023 |

and untraced `generate`, four reps each: base **125.10–126.86 tok/s**, new
**125.48–126.95 tok/s**. Every range overlaps. The change is **neutral and is
not kept** — the shipped decode path is byte-identical to the parent commit.

That is the result, and it is worth stating plainly: the instrument found 2.6× of
headroom in the read and the token did not take any of it. The reason is that the
probe prices a read and the kernel does not only read. The shared expert's group
is 2.6 µs a launch of which 0.54 µs is the node; the rest is Q4_K unpack, the dot
and the elementwise chain around it, none of which a wider load shortens. Where
the kernel is arithmetic- or launch-bound, no geometry reaches the read rate, and
that is the wall on this card at these read sizes.

For 150 tok/s: the isolated GEMV set is 5.688 ms at 483.6 GB/s over 2.562 GiB. At
the 632 GB/s the large reads reach it would be 4.35 ms, which would put the token
at 6.4 ms, **156 tok/s** — but that requires every read to run at the large-read
rate, and the one change that was supposed to move the small ones moved the token
by nothing. **150 tok/s is not reachable from this card's memory system by
re-shaping the existing GEMVs; it needs the down-bank tiles re-cut (Q5_K/Q6_K,
not touched here) and less arithmetic per byte, and neither is a load-width
change.** Measured wall: 126–128 tok/s.

## 9. Gates

On the binary this branch ships, `--features hip`, model paths passed with
`--gguf`, `HIPML_MTP_GGUF` removed from the process environment (not blanked)
and `--mtp` passed literally where a drafter is wanted.

| gate | expected | observed |
| --- | --- | --- |
| `moe-gate` CPU (Qwen3.6-35B-A3B) | 64/64 ids | **PASS 64/64**, 0.69 s/token |
| `moe-gate --device` | 64/64 ids | **PASS 64/64**, 127.39 tok/s, 21.69 GiB |
| `row-dequant` | 10 formats exact | **10/10 exact**, 2048/2048 values each |
| `load` (Qwen3.6-35B-A3B) | 0 unsupported, 256/8/512/512 | **0 unsupported**, 21.41 GiB, 20/20 round-trip, Q8_0 rows exact |
| `generate --device`, 169-token prompt | tokens on the GPU | **143.0 tok/s prefill, 125.5 tok/s decode** |
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
| dense decode, depth 0, 200 tokens | ~33.2 tok/s | **33.47 tok/s** (29.879 ms/token) |
| `cargo test --workspace` | clean | **98 passed, 0 failed** (local and on the GPU host) |
| `cargo fmt --all -- --check` | clean | clean |
| `scripts/check-public-tree.sh` | clean | clean |
| `scripts/check-doc-links.py` | all links resolve | 283 links in 53 files |

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
  once per tile. §8.4 prices the cost of not having it: **147 tok/s against
  llama.cpp's 3,375**, the one cell in the head-to-head where the engine is not
  competitive.
* **No fusion is retained, and no tile change either.** Four node-count
  reductions were built and measured (§8.1) and all four were reverted: they
  removed 40 to 90 launches and did not move the token. The instruments that
  refuted the launch hypothesis are kept. The 16-byte, one-row-per-warp tile
  §8.5 measured 2.6× faster in isolation was installed, A/B'd interleaved, found
  **neutral** and reverted for the same reason. The decode number is therefore
  the same 125.5 tok/s, and §8.1 and §8.5 name what a 150 tok/s token would have
  to come from instead.
* **The MoE ablation groups are a diagnostic.** `moe_router`, `moe_topk`,
  `moe_bank`, `moe_act`, `moe_combine` and `moe_shared` are default-off and
  exist so `decode-ablate` can price the routed FFN by removal. They add a
  `drop` argument to `MoeBlock::forward`; with nothing dropped it issues the
  same launches in the same order, and the token is byte-identical to the
  parent commit.
* **Batch/serve and `column-bench`.** Both refuse a MoE file up front, naming
  the reason: neither has a routed FFN body. The single-sequence decode path is
  the one that does. This is why §8.4's 8-slot serve cell is `n/a` rather than a
  measurement, and it is the first thing an 8-slot MoE server needs.
* **The bandwidth instrument prices reads, not kernels.** `bw-probe` has no
  dequant and no dot, so its 0.59 MB number is a ceiling the token never pays.
  §8.5's neutral A/B is the evidence that the two are not the same question. The
  Q5_K/Q6_K down banks and the F32 router were not re-tiled; only `gemv_q4_k`
  was.
* **One prompt, one seed, 64 ids.** The gate says the conventions are right; it
  does not bound how often a near-tie would flip a token between the f32 CPU
  reference, this GPU path and llama's GPU kernels. The margin print in
  `moe-gate` and the routing diff in §8.3 are the tools for that if it ever
  happens.
* **The load-bearing assumption** is that llama.cpp's MoE path on the ROCm
  backend and its CPU path agree on these ids. Only the GPU path was run; the
  box did not have the RAM for a CPU llama-server beside the loaded GPU model.
  If a later slice sees a divergence, that is the first control to repeat.
