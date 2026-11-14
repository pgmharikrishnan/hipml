# The MoE FFN: the conventions, settled against llama.cpp's ids

> *Engineering log from development; internal work-package IDs and paths have
> been sanitized or removed where they were machine-specific.*

`Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` is the first mixture-of-experts file the
engine has opened: 753 tensors, 22,853,663,008 B, `qwen35moe`, 40 decoder blocks
plus one inline MTP block, 256 experts with 8 used. Every attention and
gated-delta-net tensor in it is shaped exactly like the 27B's; the FFN is the
only new thing.

This note covers five slices. The first settled the numerical conventions on the
CPU, before any kernel existed, because four independent conventions (softmax
domain, top-k order, renormalisation, where the router weight is applied) can
each leave the ids subtly wrong, and a kernel built on a wrong one is the wrong
kernel. The second built the device path — a row gather for every format, an
expert GEMV that resolves the gather on the device, device-side top-k, and a
prefill loop — and it is the first time this file produces tokens on the GPU.
The third measured the file against llama.cpp on the same box (§8.4) and built a
bandwidth instrument to find why the small expert reads run at a third of the
large-read rate (§8.5). The fourth replaced the prefill loop with a bucketed
grouped GEMM, one launch per expert bank per chunk (§7.4, §8.6). The fifth
settled which of llama.cpp's two published 512-token numbers that kernel should
be compared against, and swept the tile height to show the padding is not what
limits it (§8.6).

**Result: the CPU reference reproduces llama.cpp's 64 greedy ids on a fixed
prompt, 64/64, first try; the device path reproduces the same 64 ids, 64/64, at
125.76 tok/s decode and 143.99 tok/s prefill, and 2,849 tok/s prefill on the
bucketed path.** Against llama.cpp on the same file, hipml decodes **33.5%
faster at depth 0 and 24.0% faster at depth 8,192** on the ROCm build and
prefills at **79%** of its warm 512-token rate and **88%** of its 8,184-token
rate (§8.6); the 8-slot serve cell is `n/a` because the batched engine has no
routed FFN. The 4,529.55 tok/s that `llama-bench` reports for the same cell is
the top of a 29% spread in the harness, not a baseline, and the 3,375.05 this
note used before charges llama.cpp for its cold first request (§8.6.1). The
bandwidth instrument finds the memory system can read 0.59 MB at 495 GB/s where
the shipped tile gets 182, and installing the wider tile is **neutral end to
end**, so 150 tok/s stays out of reach (§8.5). The file loads: 753 tensors, 0
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

### 7.4 Prefill: the bucketed grouped GEMM, and the token loop below it

A token reads 8 of 256 experts. A chunk of `T` tokens reads
`256 * (1 - (1 - 1/256)^(8T))` distinct ones — 212 at `T = 512`, 252 at
`T = 2,048` — so the token loop streams the 2.56 GiB bank once per prompt token
while a grouped GEMM reads each expert once per chunk. `prefill_moe.hip` is that
GEMM, and `MoeBlock::forward_chunk` is the path around it.

The shape is llama.cpp's `ggml_mul_mat_id`: `n_expert` blocks, each one
compacting the slots that selected its expert and running one dense GEMM. The
compaction is a counting sort (`moe_bucket_f32`) rather than a pass over the
activation tile: `off[e]..off[e+1]` is expert `e`'s run in a slot-major list,
`xtok` maps a slot to its activation row and `yslot` to its output row, and both
are gather indices, so the activation tile is read where it lies.

One tile is **64 slots x 128 rows**, 256 threads, 32-value k tiles — the shipped
`prefill_wmma2.hip` fragment layout and `pf_dequant8` decoders with the row
indices moved. 64 and not the dense tile's 256 because an expert's run is short:
at `T = 512` the median run is **5 slots** (§8.6), so a 256-slot tile would
compute 51× the arithmetic the chunk needs. §8.6.2 sweeps 16, 32, 64 and 128 and
finds 64 optimal — and the reason is not the padding, which a shorter tile would
cut, but the weight bank each tile has to read.

Per layer, on one chunk:

1. `router.gemm` writes `[T][n_expert]` logits, and `moe_topk_rows_f32` — the
   same softmax / top-k / renormalise body as `moe_topk_f32`, one block per
   token — writes `[T][n_used]` ids and weights. The two paths cannot drift
   apart on a convention because it is one body.
2. `moe_bucket_f32` counts, scans and scatters: `off`, `xtok`, `yslot`.
3. `moe_prefill_gemm` runs the gate bank and `moe_prefill_gemm_silu` the up
   bank, whose epilogue is `silu(gate) * acc` and whose fp16 copy is the down
   projection's activation. Both write slot-major `[T*n_used][n_ff_exp]`.
4. `moe_prefill_gemm` runs the down bank with `ymap = yslot`, so each expert's
   rows land straight in the token-major `[T][n_used][n_embd]` layout
   llama.cpp writes. `moe_combine_rows_f32` is then the same left fold in
   rank order the decode path does, and `moe_shared_add_rows_f32` adds the
   shared expert's sigmoid outside it.

The slot-major layout is what makes steps 3 and 4 cheap: the fold never needs a
second pass, because the scatter index is the fold's own address.

**Below 64 tokens the token loop is still the path.**
`HIPML_MOE_PREFILL_MIN_TOKENS` moves the threshold, which is how `moe-gate`
forces the bucketed path onto a 9-token prompt. The gate is the measurement
that picked 64: on that fixture the bucketed path pads 72 slots into 2,624
(36.4x) across 41 experts, so it reads more than the 8-of-256 gather the token
loop already does. §8.6 has the padding factors that matter.

The token loop is unchanged and still the fallback: `forward_chunk_token_loop`
runs one `run_token` per prompt token, in order, with `head = false` for all but
the last. `head = false` still runs `output_norm` (so a capture has every row)
and skips only the 540 MB `lm_head` projection. That makes a MoE layer 17
launches a token and the whole token 1,103.

`capture` works on both paths: the post-`output_norm` rows are written by the
same block at the end of `forward_chunk`, which is what `ppl` reads, and `ppl`
on this file returns the same number through either path (§8.6).


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
whole prompt instead of once per token (§7.4). This section is the loop's
measurement and is kept as the baseline §8.6 compares against; a 169-token
prompt now takes the bucketed path (the threshold is 64), so the loop has to be
asked for with `HIPML_MOE_PREFILL_MIN_TOKENS`.

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
the ROCm build, `0.7.0-dev (build 11820, commit d7725699)`, with `-ngl 99`.

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

**Prefill: hipml lost by an order of magnitude.** 147 tok/s against 3,375 and
138 against 4,204. The cause was §7.4: a MoE chunk ran the decode path once per
prompt token, so prefill re-streamed 2.56 GiB per prompt token. At 147 tok/s that
is a ~405 GB/s effective stream, the same wall as decode — llama.cpp's grouped
GEMM reads each expert once per tile instead of once per token. It was a missing
kernel, not a missing convention, and §8.6 is the kernel: **2,849 tok/s at 512
and 3,620 at 8,184**. §8.6.1 also re-prices the 3,375 above: it is llama.cpp
with its cold first request charged, and against its warm 3,588.74 the 512 cell
is 79%, not 85%. The table is left as it was measured, so the delta in this
section is the before.

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

hipml's cell was `n/a` when this section was written, for a reason in the engine,
not a missing measurement: `BatchEngine::new` refused a MoE file — *"batch decode
has no MoE FFN yet (256 experts, 8 used): the batched engine runs a dense FFN per
block; the single-sequence decode path does have the expert gather"* — so every
request failed and there was nothing to compare. §8.7 is the batched FFN that
removed the refusal, and the cell measured against llama.cpp's faster backend:
**355.9 tok/s aggregate, 3.7% below the Vulkan build in the same hold and 6.1%
below the 379.0 above.**

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

### 8.6 The bucketed prefill, measured

Same box, same file, one GPU-lock hold per cell, q8 KV, `hipml` at this branch
with `built with hip feature: true`. Real text is the calibration corpus's first
ids (`C:\mx\matrix\corpus.ids`), fed through `--prefill-ids-file`, so the routing
is a prompt's and not a synthetic ramp's.

| cell | prompt | chunk | tok/s (3 reps) | median |
| --- | ---: | ---: | ---: | ---: |
| real text, 512 | 511 | 512 | 2863.99 / 2877.86 / 2870.95 | **2870.95** |
| real text, 8,192 | 8,184 | 2,048 | 3837.78 / 3830.13 / 3829.71 | **3830.13** |
| real text, 8,192 | 8,184 | 512 | 2838.95 | 2838.95 |
| synthetic, 512 | 512 | 512 | 2845.82 | 2845.82 |
| synthetic, 8,192 | 8,192 | 2,048 | 3811.18 / 3803.93 / 3807.45 | **3807.45** |
| synthetic, 512 @ depth 8,192 | 512 | 512 | 2592.45 / 2599.64 / 2599.00 | **2599.64** |

The token loop on the same 511-token real-text prompt is **146.89 tok/s**
(511 tokens in 3478.8 ms), which reproduces §8.4's 147.10 — so the 512 cell is a
**19.5x** improvement, and the id out of the prefill is the same `37986` on both
paths.

Peak VRAM, read from `hipMalloc`'s own accounting after the prefill (nothing is
freed before the read, so it is the peak): **22.02 GiB** at 512 with the
287.4 MiB chunk scratch, **22.94 GiB** at 8,184 with the 1,140.8 MiB one.
Decode is unchanged: **128.2 tok/s** at depth 0 (128.22 / 128.29 / 128.06
against the parent's 128.24), because the decode path is untouched.

**Where it landed, against §8.4's llama.cpp.** 2,871 against 3,375.05 is
**85.1%**, and 3,830 against 4,203.73 is **91.1%**. Both cells are the same
prompt on both engines, so the ratio is the engine, not the prompt. Neither
figure survives the harness check below. Measured in one lock hold against a
llama.cpp that is also warm, the same two cells are **2,849 against 3,588.74
(79.4%)** at 512 and **3,620 against 4,116.16 (88.0%)** at 8,184.

**A control, three hours later.** The same source path, rebuilt and re-measured
for this slice, reads 2,845–2,849 at 512 (2846.96 / 2844.99 / 2870.45 and
2856.23 / 2848.59 / 2846.39) and **3,612–3,620** at 8,184 (3612.37 / 3593.95 /
3619.69 and 3620.32 / 3653.19 / 3597.30). The 512 cell moved 0.9%; the 8,184
cell moved 5.7%. Nothing in the diff can explain the second — `prefill_moe.hip`
is byte-identical to the parent commit and the Rust change is a constant, an
array and a pure function — so it is the box. This box shows a 4–8%
session-to-session spread for identical source, which is why every comparison in
this note is within one lock hold and why the sweep below is read for its
ordering, not its absolute numbers.

The 512 cell is the short one and the reason is measured, not guessed. Every
expert's run is rounded up to a whole 64-slot tile, and `--bucket-stats` prints
the factor:

| chunk | distinct experts | slots | runs (min / median / max) | padded | tiles |
| ---: | ---: | ---: | --- | ---: | ---: |
| 512 real text | 216 of 256 | 4,088 | 0 / **5** / 163 | 15,040 (**3.68x**) | 235 |
| 2,048 real text | 253 of 256 | 16,320 | 0 / **30** / 674 | 25,920 (**1.59x**) | 405 |
| 9 (the gate fixture) | 41 of 256 | 72 | 0 / 0 / 5 | 2,624 (**36.44x**) | 41 |

A 512-token chunk has a **median run of 5 slots**: 216 experts each get a whole
64-slot tile for a handful of rows, so the grouped GEMM does 3.68x the chunk's
arithmetic. At 2,048 the median run is 30 and the factor falls to 1.59x. That is
part of why the 512 cell is the weaker one — but only part, and not the part that
can be fixed by shrinking the tile, which is what §8.6.2 measured.

The FFN's own rate at 512, from `--prefill-split`: `gemm ffn` is **103.1 ms of a
190 ms chunk** (§8.6.2). 15,040 padded slots x 3,145,728 MACs x 40 layers is
3.79 TFLOP, so the routed bank runs at **37 TFLOPS**, and its 235 tiles read
17.9 GB in that time, **173 GB/s**. Neither is a wall on this card (95 TFLOPS
fp16, ~633 GB/s), which is the same conclusion §8.1 and §8.5 reached from the
other direction. §8.6.2 says what the 173 GB/s is — the small-read wall, not the
padding. The rest of the 190 ms is the dense path (`seq ssm` 41.8, `seq attn`
10.1, `gemm ssm` 10.5, `elementwise` 6.7, `gemm attn` 3.4), which this slice does
not touch.

**What the recon got wrong.** Three things, all of them now measured:

1. It priced the grouped GEMM on the dense prefill's 256-token tile. At a median
   run of 5 slots that is 51x the arithmetic; the tile has to be 64, and even
   then the padding is 3.68x at 512.
2. It assumed the bucket had to compact the activation tile into slot order
   before the GEMM. It does not: `xtok` is a gather index, so the tile is read
   where it lies and the compaction is only the 4,096-entry sort.
3. It read "each expert is read once per tile" as the win. That is the win —
   §8.6.2 fits 87% of the FFN to the per-tile weight staging — but the same
   sentence says the padding was never the lever: a shorter tile cuts padding
   and multiplies tiles, and the sweep finds 64 already optimal.

### 8.6.1 Which llama.cpp number the 512 cell is against

The 34% is between two llama.cpp harnesses, not between llama.cpp and hipml.
Same file, same box, same 511-token fixture:

| harness | backend | protocol | tok/s |
| --- | --- | --- | ---: |
| `llama-bench -p 512 -r 5` | Vulkan | one process, 5 timed reps | 4,529.55 ± 115.06 (the figure this section used to quote) |
| `llama-bench -p 512 -r 5`, run five times | Vulkan | one process per run | 3,340 / 3,494 / 3,692 / 3,711 / **4,319 ± 50.58** |
| `llama-bench -p 512 -r 1`, run fifteen times | Vulkan | fresh process, one timed rep | **3,173.73** (2,965.71–3,197.49) |
| `llama-server`, 511 ids, q8 | ROCm | fresh process, cold rep | **1,160.01** |
| `llama-server`, 511 ids, q8 | ROCm | warm, median of five | **3,580.66** |
| `llama-server`, 511 ids, q8, paired | ROCm | one warm-up + 3 reps, one lock hold with hipml | **3,588.74** (3,249.11 / 3,588.74 / 3,606.67) |
| `llama-server`, 8,184 ids, q8, paired | ROCm | one warm-up + 3 reps, same hold | **4,116.16** (4,102.63 / 4,123.02 / 4,116.16) |
| `llama-server`, 511 ids, f16 | ROCm | warm, median of five | 3,643.88 |
| `llama-server`, 511 ids, q8 | Vulkan | warm, median of five | 3,347.24 |
| `llama-server`, 512 real-text tokens, q8 | ROCm | warm, median of five | 3,649.35 |
| `llama-server`, 512 real-text tokens, q8 | Vulkan | warm, median of five | 3,331.59 |
| `llama-server`, 511 ids, q8, `-ub 128` | ROCm | warm, median of five | 2,189.43 |

Ruled out, one at a time:

* **Cold start.** Real, and 3.1x — but it is 257 ms of a 397 ms first request and
  it is gone by rep 1. It explains the matrix cell's 1,528 minimum, not the
  4,529.
* **Tokenisation and cache init.** 511 ids and 512 real-text tokens agree to 0.3%
  on both backends (3,580.66 against 3,649.35 on ROCm, 3,347.24 against 3,331.59
  on Vulkan).
* **`-b` / `-ub`.** A real lever — `-ub 128` costs 39% on the server and 19% on
  the bench — but both tools default to `-ub 512`, and `-b 2048 -ub 512` against
  `-b 2048 -ub 2048` is 3,177 against 3,164, i.e. nothing. It is not the
  difference.
* **The fixture length.** `-p 512` generates 512 synthetic tokens and the fixture
  is 511. That is 0.2%.
* **Steady-state loop against request overhead.** This one is real, and it runs
  the other way from the guess: llama-bench's in-process reps are the fast ones
  (4,319 with a ±50 spread) and a fresh process per rep is 3,174.

What is left is llama.cpp's own spread. Identical `llama-bench` commands, run one
after another, read 3,340, 3,494, 3,692, 3,711 and 4,319 — a 29% range for a
command that differs in nothing. The 4,529.55 is the top of that distribution,
and a single-rep-per-process run never reaches it (3,197.49 in fifteen tries).

**The baseline this note uses for the 512 cell is 3,588.74**: `llama-server`,
ROCm, the same 511 ids, q8 KV, warm, measured in the same lock hold as the hipml
number it is compared against (2,848.59, median of three). A separate five-rep
sweep 1.5 hours earlier read 3,580.66, so the baseline is stable to 0.2% while
the harness is not. Not 3,375.05 and not 4,529.55. hipml's number is measured
with the model resident and the graph already built — its first rep and its third
agree to 0.4% — so charging llama.cpp for the first request's graph build while
not charging hipml is the asymmetry that makes 3,375 flatter us. 3,588.74 is the
harsher of the two defensible numbers, so it is the one the claim is made
against: **hipml is 79.4% of llama.cpp at 512, not 85.1%.** At 8,184 the same
paired run reads 3,620.32 against 4,116.16, **88.0%**. Against the 4,529.55
harness figure the 512 cell would be 63%, and that number is recorded here and
nowhere else in this note.

The same harness check also re-prices the decode win, and it is worth stating
plainly because it is the other half of §8.4. **llama.cpp's decode rate is a
backend property on this card**: 94.20–96.50 tok/s on ROCm and **128.95–132.18 on
Vulkan**, same file, same q8 KV, same prompt. hipml decodes at 128.2. So the
33.5% "win" at depth 0 is hipml against the ROCm backend, and against the Vulkan
backend hipml is level. That does not change what §8.4 measured — the ROCm build
is the one the matrix ran — but it does mean the decode headline is a llama.cpp
backend difference and should be read as one.

### 8.6.2 The padding is not the ceiling: the tile height, swept

The 512 cell pads 3.68x and the 8,184 cell 1.59x, so the obvious next move is a
shorter tile. It was built and swept: the same chunk at 16, 32, 64 and 128 slots
a tile, real text, medians of three, one lock hold. Tile 8 is not representable:
the tile is `MG_TT` slots and `MG_WT` slots a warp, `MG_WT` is 16, and the kernel
asserts it is a multiple of 16, so there is no 8-row MMA to build on.

Chunk 512, 511-token prompt:

| tile | tok/s (3 reps) | median | padding | tiles |
| ---: | --- | ---: | ---: | ---: |
| 16 | 2020.56 / 2018.43 / 2015.47 | **2,018.43** | 6,064 (1.48x) | 379 |
| 32 | 2008.82 / 2505.14 / 2388.03 | **2,388.03** | 8,928 (2.18x) | 279 |
| 64 | 2810.50 / 2820.11 / 2833.61 | **2,820.11** | 15,040 (3.68x) | 235 |
| 128 | 2594.95 / 2700.28 / 2702.32 | **2,700.28** | 28,160 (6.89x) | 220 |

Chunk 2,048, 8,184-token prompt:

| tile | tok/s (3 reps) | median | padding | tiles |
| ---: | --- | ---: | ---: | ---: |
| 16 | 2415.72 / 2416.48 / 2401.32 | **2,415.72** | 18,320 (1.12x) | 1,145 |
| 32 | 3162.97 / 3129.74 / 3102.78 | **3,129.74** | 20,480 (1.25x) | 640 |
| 64 | 3994.67 / 3983.35 / 4031.55 | **3,994.67** | 25,920 (1.59x) | 405 |
| 128 | 4182.73 / 4064.84 / 4026.14 | **4,064.84** | 39,552 (2.42x) | 309 |

The 128 row is from the sweep build, which carried a fourth dispatch path; the
shipped `--bucket-stats` prices the three heights the shipped binary can run.
The first rep of a path that has never run reads 2–25% low while the code object
loads, which is the 2,008 in tile 32's row; the median is taken over all three
anyway, so that row is if anything understated.

**A shorter tile is 28% slower at 512.** Cutting the padding from 3.68x to 1.48x
takes the cell from 2,820 to 2,018. The padding was never the ceiling. What is
the ceiling is the number of tiles, and the run-length distribution says why a
short tile cannot have few tiles:

| chunk | 1–16 | 17–32 | 33–64 | 65–128 | 129–256 | 257+ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | **154** | 29 | 18 | 11 | 4 | 0 |
| 2,048 | 68 | 65 | 63 | 30 | 15 | 12 |

154 of the 216 runs in a 512-token chunk are 16 slots or shorter, with a median
of 5. Those 154 runs are 154 tiles at any height, so a 16-slot tile buys padding
and pays tiles: 379 against 235.

`--prefill-split` prices the two ends of the sweep. At 511 tokens, `gemm ffn` is
**150.4 ms of a 230 ms chunk** at tile 16 (2,185 tok/s) and **103.1 ms of 190 ms**
at tile 64 (2,728 tok/s). Two points, two unknowns, and one fit that explains
both:

```
ffn_ms = 0.383 x tiles + 0.000873 x padded_slots
```

| tile | tiles x 0.383 | padded x 0.000873 | fitted ffn | measured ffn |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 145.2 | 5.3 | 150.5 | 150.4 |
| 32 | 106.9 | 7.8 | 114.7 | — |
| 64 | 90.0 | 13.1 | 103.1 | 103.1 |
| 128 | 84.3 | 24.6 | 108.9 | — |

**87% of the routed FFN is per-tile weight staging and 13% is the padded MMA**, so
the optimum is where the two terms balance, and it is 64. The model also explains
the rest of the sweep: 32 is 11% worse than 64 (114.7 against 103.1) and 128 is
5.6% worse, because 220 tiles x 0.383 ms of staging plus 24.6 ms of padding
arithmetic is still more than 235 x 0.383 plus 13.1. At 2,048 the two terms
nearly cancel between 64 and 128 — 405 tiles against 309, but 25,920 padded slots
against 39,552 — and the measurement agrees: 4,065 against 3,995, +1.7%, inside
the box's wobble.

Read as a rate, the staging term is 235 tiles x 1.9 MB x 40 layers = 17.9 GB in
the 103.1 ms, **173 GB/s**. A tile is not expensive because it is a tile; its
1.9 MB bank is read at a third of what §8.5 measured for a 0.59 MB read. The tile
height trades those bytes against padded MMA slots and nothing else, which is why
the fit is a straight line in both — and why the ceiling on this cell is the same
small-read wall §8.5 found, one level up in size.

**The rule is one height, 64.** No cell needs an adaptive tile: 64 is the fastest
at 511 tokens in both chunks and within 2% of the best at 8,184, and 128's
advantage there is smaller than the build-to-build spread the control above
records. A rule that is not needed is worse than none, so no adaptive rule and no
`HIPML_MOE_TILE` shipped.

**Merging short runs across experts is not expressible, so it was not built.** A
16-row MMA reads one weight tile for all 16 slot rows; putting expert A's 5 slots
and expert B's 5 slots in one tile needs a per-row weight pointer, which is the
decode GEMV and not the grouped GEMM. The arithmetic would not be unchanged: the
weight tile would stop being shared across the 16 rows, which is the reason the
tile exists. The fit above is a floor rather than a shortcut for the same reason —
the per-tile term *is* the weight read.

**What the sweep changed in the tree: nothing in the kernel.** `prefill_moe.hip`
on this branch is byte-identical to the parent commit and `MOE_PREFILL_TOKENS` is
still 64. What shipped is the diagnostic: `--bucket-stats` now prints the
run-length histogram and the padding at 16, 32 and 64 with the shipped height
marked, so the tables above can be reproduced from the binary instead of from a
sweep build. A runtime dispatch over 16/32/64 was built and measured at tile 64:
2,820 (build A) and 2,932 (build B) against **2,845 for the fixed-tile binary** —
−0.9% to +3.1%, inside the spread this box shows for identical source, with no
cell where a shorter tile won. It was not shipped: it costs two more kernel
variants to buy a measurement that is not there.

**Perplexity.** `ppl` on this file, `--ctx 512 --chunks 4`, is **2.2275** through
the token loop (unchanged from §8.2) and **2.2263** through the bucketed path, a
0.054% move over 1,020 scored positions. `oracle-gate` over 20 steps on the
bucketed path is 20/20, and `moe-gate --device` with the path forced onto its
9-token prompt is 64/64 with the routing diff at 34 identical / 3 rank-order /
3 set-difference against the token loop's 30 / 8 / 2.

### 8.7 Serving a mixture of experts

§8.4's serve cell was `n/a` because `BatchEngine::new` refused a MoE file
outright: every request failed before the model was reached and there was
nothing to compare. This section is the batched FFN that removed the refusal,
and what the cell measures now.

#### 8.7.1 The batched step is the decode gather with the slot on `grid.z`

`moe_gemv` already resolves `w + ids[j] * stride` on the device, so a batch needs
one more index, not another body: `moe_gemv_batch` is `moe_gemv`'s tile function
with `grid.z` as the slot. Slot `s` reads its own ids (`ids + s * gridDim.y`),
its own activation row (`x + s * x_slot`) and writes its own output
(`y + s * y_slot`); the dequant, the dot and the fold are `moe_gemv_tile`, shared
by both kernels, so there is no second copy of the arithmetic to drift. The ids
stay on the device, so the step is still capturable.

The rest of the block follows the same rule. The router and the shared expert's
gate dot are f32, so they take `gemv_f32_b` (`Proj::gemm_batched_f32`), whose
reduction is `gemv_f32`'s; the shared expert's three block-format projections
take the exact `_b` GEMV (`Proj::gemm_batched`); the top-k, the combine and the
shared add are the `_rows` kernels the prefill tile already uses, one row per
slot. `MoeBatch` is one scratch set shared by every layer, slot-major, sized
`[slots * n_used]` rows for the routed tiles and `[slots]` for the per-slot
scalars. `weight_bytes` counts what actually streams: the router and the shared
expert once, the routed banks once per slot per rank.

The rows path shares one body: `MoeBlock::forward_rows` is the bucketed grouped
GEMM over any `[n_tok][n_embd]` tile, and both the prompt chunk and the
multi-sequence row block call it, so a batched prefill and a single prompt cannot
drift apart.

Speculative verify had no MoE body in either verify pass: the single-stream
column path (`columns.rs`) returned *"the column path has no MoE body"* and the
batch verify (`batch/verify.rs`) returns *"speculative verify has no MoE FFN
(MTP is dense-only)"*. **The single-stream path now has one** — it runs
`MoeBlock::forward_batch` with one slot per verify column, so `spec-bench` and
`column-bench` work on this file; §8.8 has the numbers. The batch verify still
refuses, so `serve --drafts` on a MoE file remains unavailable (§10).

There was a second blocker behind the first, and it was not in the engine:
`chat.rs` required every marker in a chat template. Qwen3.6's template has no
`reasoning_instructions`, so every request 400'd before the model was reached;
that one marker was made optional, and then MiniCPM-V-4.7's template 400'd on
the next marker (`preserve_thinking`) — the same bug one marker later. The
handling was one-marker-at-a-time and would have broken on the next model, so it
is now fixed as a class.

`validate_template` requires only the two ChatML turn delimiters, because those
are what the renderer emits verbatim and a template without them is a different
format. Every other marker is a branch the renderer implements itself and is
optional; its absence changes that branch:

| marker | absent |
| --- | --- |
| `render_content` | ignored; it names the macro this renderer inlines |
| `add_generation_prompt` | the caller's flag still appends the turn |
| `enable_thinking` | thinking is not a knob: the closed block is emitted |
| `preserve_thinking` | pre-query thinking is dropped (no branch keeps it) |
| `reasoning_instructions` | no system reasoning block |
| the opening thinking tag | no thinking block at all |

A marker the engine has never seen is never inspected, so it cannot fail a
render. `cargo test -p hipml-core --test chat_render` is 12 passed, including a
template with an unknown marker and a template missing `preserve_thinking`.

The cell: `hipml serve --gguf MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf` answers
`/v1/chat/completions` with **200** (18 prompt tokens, 8 completion tokens,
"Hello there, how are you?"), where the parent binary answers **400** on the
same request. `/completion` is 200 on both.

#### 8.7.2 Both shapes were measured, and the gather won

A batch can route its FFN two ways: gather each slot's top-k experts (one launch
per bank, `grid.z` = slot, an expert read once per slot per rank) or bucket all
`slots * n_used` rows by expert and run the prefill's grouped GEMM over them (a
distinct expert read once). The second reuses `MoeBlock::forward_rows` and was
built and measured first, because "read each expert once" is what wins on a
prompt.

B=8, 32 tokens, a distinct stream per slot, the WMMA tile on (what `serve` runs
at B≥4), one lock hold:

| FFN shape | aggregate tok/s | ms/step | FFN stage of the step |
| --- | ---: | ---: | ---: |
| per-slot gather (`grid.z = B`) | **425.86** | **18.79** | 14.02 ms |
| bucket across slots (grouped GEMM) | 217.50 | 36.78 | 31.66 ms |

Both rows are one hold on one tree, which is why the bucket row exists at all; on
the shipped tree (the bucket route removed) the gather cell re-measures at 422.47
tok/s and 18.94 ms/step, the same path inside this box's spread.

The bucket route is 1.96× slower here, and the reason is §8.6.2's: 8 slots × 8
ranks is 64 rows over 256 experts, so the bucket is 41–64 runs of one or two
rows and the tile pads each of them to 64. A 9-token prompt already pads 36.44×;
64 scattered rows pad worse, and the grouped GEMM pays for it in tiles instead of
in reads. The gather's 2.56 GiB × slots is a dense stream per bank, which is the
shape this card's bandwidth likes. The bucket route is not behind a flag: it was
removed from the tree, and the numbers above are the whole of it.

#### 8.7.3 Scaling

`batch-bench --gguf <moe> --slots B --tokens 32`, the same token in every slot,
the exact `_b` dispatch (no `HIPML_BATCH_SKINNY`):

| B | aggregate tok/s | ms/step | per stream |
| ---: | ---: | ---: | ---: |
| 1 | 125.88 | 7.94 | 125.88 |
| 2 | 155.09 | 12.90 | 77.54 |
| 4 | 172.15 | 23.24 | 43.04 |
| 8 | 197.88 | 40.43 | 24.74 |
| 16 | 203.46 | 78.64 | 12.72 |

With the WMMA tile (`HIPML_BATCH_SKINNY=1`, which is what `serve` sets at B≥4):
B=8 419.05, B=16 533.72, and B=8 on distinct streams 422.47. The exact path is
the one the id gate runs; the tile is the one the server runs. At B=16 the exact
step streams 11.13 GiB at 152 GB/s, against 2.56 GiB at 346 GB/s at B=1: the
gather reads each expert once per slot, so the bytes grow with B and the wall
§8.5 measured is what the growth meets.

#### 8.7.4 Cell b, against the faster backend

Cell b is 8 slots at 16K context, a 2,384-token prompt, 64 tokens a request,
three reps, one lock hold, warm. The card's rule is llama.cpp's faster backend
per cell, so the baseline here is the Vulkan build; §8.4's 379.0 was the ROCm
build for the same cell and is quoted too, because it is the number the note
priced.

| engine | aggregate tok/s p50 (best rep) | per-stream p50 | TTFT p50 | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| llama.cpp Vulkan | 369.71 (369.94) | 46.29 | 190.0 ms | 22,971 MiB |
| hipml | 355.93 (356.11) | 49.27 | 232.6 ms | 25,213.5 MiB |
| llama.cpp ROCm | 328.86 (361.89) | 47.20 | 256.8 ms | 23,548 MiB |

**hipml serves this file at 355.9 tok/s aggregate: 3.7% below the faster backend
in the same hold and 6.1% below 379.0.** It is above the ROCm build's median by
8.2%, though not above its best rep (361.9), which is why the cell is priced
against the backend that is faster on its median. Per stream it is ahead of both
(49.27 against 46.29 and 47.20) because the aggregate is spread over eight
requests that finish at different times, and it pays for that in TTFT (232.6 ms
against 190.0, +22.4%) and in VRAM (+9.8% over Vulkan, +7.1% over ROCm — the
25,213.5 MiB is the routed FFN's scratch, `[slots * n_used]` tiles for 8 slots,
on top of the dense path's).

Nothing was tuned after the number was known: the cell is the engine as the gates
left it, with `HIPML_BATCH_SKINNY` set by `serve` exactly as it is for a dense
file. The spread across the three reps is 0.05% on hipml and 0.06% on Vulkan, so
the 3.7% is the engine, not the box.

#### 8.7.5 The per-slot id gate, and where it stops

`batch-bench --check` replays a distinct token stream per slot through the
batched engine and each slot's own stream through the single-sequence engine, and
requires every id to match. `--sweep` rebuilds the engine at B = 1..=slots and
replays each width, so a bug that only shows at one width cannot hide behind the
capacity the benchmark happened to run at. On a dense file that gate passes at 16
slots. On this file:

| width | tokens | result |
| --- | ---: | --- |
| B=1..4, `--sweep` | 32 | every slot 32/32 |
| B=5..16, `--sweep` | 16 | **slot 4 step 13: batch 382 vs single 290**, 15 of 16 slots clean; the same width at 32 tokens also breaks slot 15 at step 27 (362 vs 64) |

The routed FFN is not the cause, and `--diff-slot` shows where it is. Feeding one
slot's stream through both engines and diffing the whole logit vector per step
(Qwen3.6-35B-A3B, B=5, slot 4):

| step | logits differing (of 248,320) | max abs d | batch top | single top |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 470 | 470 |
| 1 | 244,744 | 4.4e-5 | 198 | 198 |
| 4 | 248,318 | 8.2e-2 | 279 | 279 |
| 13 | 248,319 | 1.3e-1 | 382 | 290 |

Step 0 is bit-identical: the embedding, 40 layers of routed FFN on the batch's own
scratch, the final norm and the lm_head all reproduce the single-stream path
exactly. Step 1 is the first step where a per-slot state is read (two keys in the
cache, one SSM step) and it already differs by 4.4e-5, so the seed is upstream of
the FFN. That is by design, not a bug in the gather: from B=2 the batched
attention is the staged `_b` split-K over the cache stride, while the single
stream uses the whole-group warp kernel at this file's head_dim — `batch_launch.rs`
says so at its `one` branch, where the two engines share a kernel only at one
slot (`HIPML_ATTN_STAGED` forces even that one onto the staged path).

A dense file shows the same seed and does not care: Qwen3.8-27B at B=5 differs by
2.2e-3 at step 1 and still matches 8/8 ids, Arete-27B by 2.6e-5. A MoE file
amplifies it, because 4.4e-5 of router input is enough to reorder the last rank
(§8.3 already reports 8 of 40 layers doing that between the CPU and GPU
references), and one flip puts the two streams 8.2e-2 apart by step 4. From there
it is a race between the divergence and the top-2 margin: at step 13 the margin
is 2.6e-2 against the single stream's 8.1e-2, and the argmax flips.

So the gate's honest statement: **the batched FFN is exact per slot — step 0 runs
every layer of it on the batch's own scratch and is bit-identical, and
`gemv-check` on the shared expert's three projections is 0 differ at 16 columns
(8,192 / 8,192 / 32,768 values) — but the end-to-end per-slot id gate holds only
at B=1..4, because from B=2 the batched step is not the single-stream step.**
Making it hold at B=16 means giving the batch back its attention kernel, which is
the batched engine's whole point on a dense file; that is the owner's call, and
it is written here rather than papered over with a narrower gate.

## 8.8 Speculative decode on the MoE file (2026-11-14)

The verify pass had no routed FFN, so no speculative decode was possible on any
mixture-of-experts file. Three changes, all reuse rather than new arithmetic:

* **The column body is `MoeBlock::forward_batch`.** The verify runs `k` columns
  of one token through attention, GDN and the FFN at once; the FFN needs the
  router over `k` rows, the top-k per column, the gather and the fold per
  column. That is exactly the batched decode step §8.7 built for serve, with
  one slot per verify column: the column index is `grid.z`, each column reads
  its own router row, ids, activation row and output row, and the tile body is
  the single-sequence one. `forward_rows` — the bucketed grouped GEMM — is
  **not** the reuse, because its arithmetic is not the single-token step's
  (§8.7.2 measured it 1.96x slower at B=8 as well).
* **The drafter's FFN is a `MoeBlock`** when the file has experts, built from
  the same `blk.<n>.ffn_*` names the decoder layers use. `blk.40` on this file
  carries the 256-expert banks, not `ffn_gate`.
* **BF16 is widened to f32 at upload.** `blk.40.ffn_gate_inp` and
  `ffn_gate_inp_shexp` are the file's only BF16 tensors and no GEMV reads BF16;
  they are 1 MB, so the loader expands them (2 MB resident) rather than adding a
  kernel.

`column-bench --cols 4 --check` is **0 differ** on 7,946,240 logits, hidden and
residual, at q8 and f32 over 32 columns. `spec-bench --gguf <this file> --mtp
<the same file>` is **128/128 token-exact at drafts 1, 2 and 3** on both KV
modes, against plain greedy decode of the same prompt.

| drafts | ms/step | tok/step | tok/s q8 | vs plain | tok/s f32 | acceptance | exact |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 7.86 | 1.000 | **127.30** | 1.00x | 124.89 | — | — |
| 1 | 14.37 | 1.954 | **135.94** | 1.07x | 133.58 | 0.954 | 128/128 |
| 2 | 19.90 | 2.822 | **141.84** | **1.11x** | 138.59 | 0.956, 0.953 | 128/128 |
| 3 | 26.67 | 3.528 | **132.28** | 1.04x | 129.82 | 0.917, 0.939, 0.935 | 128/128 |

The verify slope is the finding. A dense verify shares the FFN's weight stream
across columns (four columns cost 1.33x a one-column pass); a routed FFN cannot,
because column `c` reads its own 8 of 256 experts. `column-bench` on this file:

| columns | ms/pass | vs 1 col |
| ---: | ---: | ---: |
| 1 | 9.01 | 1.00x |
| 2 | 12.82 | 1.42x |
| 3 | 16.90 | 1.88x |
| 4 | 22.34 | **2.48x** |

So the second draft pays (+11.1%) and the third does not: from two drafts to
three the step buys 0.706 tokens for 6.77 ms, 9.6 ms per marginal token against
the 7.86 ms a plain token costs. `docs/notes/mtp.md` has the full table and the
step breakdown (drafts 3: draft 3.90 + verify 22.47 + rollback 0.30 ms).

**Verdict: speculation makes this file faster than plain decode, by 11.1% at two
drafts (141.84 against 127.30 tok/s), and it is token-exact.** It does not beat
the 3-draft dense result because the routed verify is the term that does not
amortise.

## 8.9 llama.cpp speculates too (2026-11-15)

The comparison in §8.4 and in `40629ff` is against llama.cpp's **plain** decode,
and that is not the honest baseline: both files carry an inline MTP block and
both llama.cpp builds on the box can drive it. Build 11820 (ROCm, `d7725699`)
and 11836 (Vulkan, `f5027fb21`) expose the same flag set:

```
--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,draft-dspark,...
--spec-draft-model, -md, --model-draft FNAME
--spec-draft-n-max N                 (default 3)
--spec-draft-n-min N
--spec-draft-p-min, --draft-p-min P
--spec-draft-sampling {greedy,probabilistic}
```

**The inline block needs no `-md`.** `--spec-type draft-mtp` alone makes the
server build the drafter out of the target's own MTP layer, and its log says so:

```
common_speculative_init_result: creating MTP draft context against the target model 'Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf'
```

With `-md` it loads a separate sidecar instead (that is how the dense 27B cell
in [`benchmarks.md`](../benchmarks.md) was run). So the reference command for a
file with an inline block is:

```powershell
llama-server -m <file> -ngl 99 -c 512 --parallel 1 -ctk q8_0 -ctv q8_0 `
  --no-webui --no-cache-idle-slots --spec-type draft-mtp --spec-draft-n-max 3
```

`p2_math_169.ids` as the prompt, 128 greedy tokens, `ignore_eos`, three reps per
cell, one process per cell, both backends. The same fixture and token count as
`spec-bench`, so the columns are comparable; hipml's rows are `spec-bench` on
this branch (q8 KV, same prompt).

### OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G (dense, inline `blk.64`)

| engine | plain | d1 | d2 | d3 | accept at d3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp ROCm 11820 | 33.94 | 53.08 | 66.76 | **75.75** | 0.885 (92/104) |
| llama.cpp Vulkan 11836 | 34.43 | 55.88 | 70.38 | **82.09** | 0.885 (92/104) |
| hipml | 34.06 | 53.67 | 59.85 | **68.12** | 0.971 / 0.912 / 0.903 |

### Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL (MoE, inline `blk.40`)

| engine | plain | d1 | d2 | d3 | accept at d3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp ROCm 11820 | 99.18 | 137.98 | 164.88 | **175.79** | 0.858 (91/106) |
| llama.cpp Vulkan 11836 | 132.02 | 169.99 | 198.93 | **213.35** | 0.885 (92/104) |
| hipml | 127.50 | 136.46 | 141.64 | **132.16** | 0.917 / 0.939 / 0.935 |

**We are behind llama.cpp speculating, on both files.** Orca: 68.12 against
82.09 (−17.0%) on its faster backend, −10.1% against the ROCm build. This file:
141.64 at our best depth against 213.35 (−33.6%), −19.4% against ROCm. Even
against the slower backend we lose, so the verdict does not depend on which
build is quoted.

The cause is the same term §8.8 names, and llama.cpp's numbers price it. On this
file at three drafts both engines emit almost the same number of tokens per
step — 3.528 for us, 3.53 for llama.cpp (`mean len`, acceptance 0.858) — so the
difference is the **step cost**: 26.69 ms for our 4-column verify against
16.7 ms for llama.cpp's (128 tokens in 0.6 s). llama.cpp's routed verify
amortises; ours is 2.48× a one-column pass at four columns (§8.8). Our plain
decode is faster than its ROCm plain (127.50 against 99.18) and level with its
Vulkan plain (132.02), so the whole deficit is the verify, not the token.

Two further observations, both about llama.cpp rather than us:

* **Its speculative ids are not its plain ids on this file.** The ROCm plain
  stream and the ROCm d3 stream diverge at token 53 (72 of 128 ids differ); the
  Vulkan d3 stream happens to equal the ROCm plain stream exactly. So its
  213.35 tok/s is a different token stream from its own plain decode, while
  every hipml row above is 128/128 identical to plain. Its Orca cells are
  identical across all eight (same sha), so the drift is MoE-specific — the same
  router amplification §8.7.5 measured in our own batch path.
* **The two drafter files were not worth taking.** Orca + the DFlash2 drafter
  (`--spec-type draft-dflash -md ...DFlash2-Q8_0.gguf --spec-draft-n-max 7`,
  Vulkan) reads 66.92 tok/s at acceptance 0.709 / mean len 5.77 — below the
  inline block's own d2 (70.38) and d3 (82.09). MiniCPM-V-4.7 + the DSpark
  drafter (`--spec-type draft-dspark`, n-max 7, Vulkan) reads 142.56 against
  159.32 plain (0.90×), acceptance 0.368. Declining both was right.

## 8.10 Speculative batch on the MoE file (2026-11-15)

`serve --drafts` on a MoE file already worked for a lone request, because the
server's speculative mode runs the single-stream `MtpGpu`, which has had a
routed body since §8.8; `probe_exact.py --ctx 32768` on this file is **PASS, 48
replies id-for-id across drafts 1,2,3**. What was missing was the *batch*
drafter and verify: `batch/verify.rs` returned *"speculative verify has no MoE
FFN"* and `batch/mtp.rs` hardcoded `blk.64` and a dense `ffn_gate`, so
`mtp-batch-bench` on this file died at setup with *"model has no tensor
blk.64.attn_q.weight"*.

Two edits, both reuse:

* `batch/verify.rs`: the `BatchFfn::Moe` arm now runs `MoeBlock::forward_batch`
  over the `[M]` verify tile with a `MoeBatch` sized for it, then the residual
  add — the same call the plain batched step makes on the same row.
* `batch/mtp.rs`: the drafter's block index is `mtp_blk(&cfg)` (= `n_layer`, so
  40 here and 64 on the dense files), its FFN is a `MtpFfn::{Dense,Moe}` built
  the way the single-stream drafter builds it, and the routed step is
  `MoeBlock::forward_batch` over the `B` slot rows.

### Where it is exact, and the bound

`mtp-batch-bench` (this file, `p2_math_169.ids` rotated per slot, 128 tokens,
q8 KV, serve's `--skinny` dispatch) compares every slot's speculative stream
with that slot's plain batched stream:

| B | drafts | exact | note |
| ---: | ---: | --- | --- |
| 1 | 1, 2 | **yes**, 1/1 slots | |
| 4 | 1, 2 | **yes**, 4/4 slots | |
| 8 | 1 | no | slot 5 matches to token 101, slot 6 to token 42 |
| 8 | 2 | no | slot 1 matches to token 52, slot 5 to token 101 |

**The per-slot bound is unchanged at B≤4, and speculative batch does not widen
it.** `batch-bench --slots 16 --check --sweep` on this file still reports the
documented `slot 4 step 13: batch 382 vs single 290`, 15 of 16 slots clean. At
B=8 the verify runs `M = 16` rows where the plain reference runs `B = 8`, which
crosses the 8-row `_b` group boundary; on a routed FFN that is enough to move a
last-rank expert and flip an argmax later, exactly the amplification §8.7.5
describes. A dense file at B=8 is exact (`mtp-batch.md`), so this is the MoE
router, not the verify.

### Throughput: the batch optimum is not the single-stream one

| B | plain agg | d1 agg | d2 agg | best |
| ---: | ---: | ---: | ---: | --- |
| 1 | 120.50 | 139.76 (**1.16×**) | 150.43 (**1.25×**) | d2 |
| 4 | 292.13 | 330.3 (**1.13×**) | 324.2 (1.11×) | d1 |
| 8 | 412.18 | 400.1 (0.97×) | 360.9 (0.88×) | **none (drafts 0)** |

The single-stream optimum is d2 (§8.8); at 4 slots it is d1 and at 8 slots there
is no win at all. The verify is the reason and the numbers are the same shape as
§8.8's: at B=8 one draft costs a 36.28 ms step against 19.41 ms plain (1.87×)
and two drafts 53.11 ms (2.73×); at B=1 two drafts cost 19.63 ms against 8.30 ms
(2.37×). `M = B(1+R)` crosses a weight-pass boundary as B grows, and the routed
FFN adds its per-column expert reads on top, so the schedule that wins alone
loses at 8 slots. **Drafts 2 winning single-stream and drafts 1 winning at 4
slots is the same finding as §8.8's "the routed verify does not amortise", not
an inconsistency.**

### The 8-slot serve cell, and why it is flat

`serve_bench.py --cell b` (8 slots, 16K per slot, 2,384-token prompt, 64 tokens
a request, 3 waves), this file, q8 KV:

| engine | drafts | agg p50 | per-stream p50 | TTFT p50 | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| hipml | 0 | **355.0** | 49.17 | 232.7 ms | 25,213.5 MiB |
| hipml | 1 | 354.9 | 49.10 | 232.4 ms | 25,284.1 MiB |
| hipml | 2 | 353.8 | 48.97 | 233.4 ms | 25,284.1 MiB |
| llama.cpp Vulkan | plain | 340.3 (best 352.9) | 44.52 | 271.1 ms | 22,971.0 MiB |
| llama.cpp Vulkan | `--spec-type draft-mtp --spec-draft-n-max 2` | 283.3 | 37.91 | 218.7 ms | 24,849.8 MiB |

The hipml rows are flat because the server only speculates a request that is
**alone** (`serve/sched/run.rs`: `Mode::Spec` iff no other active slot and an
empty queue). An 8-slot wave never takes that branch, so `--drafts 1/2` cost
the drafter's 70 MiB and change nothing; the gain is on cell a, not here. That
wiring is deliberate and unchanged — and §8.10's own B=8 row says wiring it
would not pay anyway (0.97× at d1, 0.88× at d2).

So the honest 8-slot baseline is llama.cpp's **plain** number: 340.3 in this
session (369.71 in the census) against hipml's 355.0, which is +4.3% here. Its
own speculative batch is 17% *below* its plain batch on this cell, so quoting
llama.cpp speculating would flatter us; it is not the right comparison either.

## 9. Gates

On the binary this branch ships, `--features hip`, model paths passed with
`--gguf`, `HIPML_MTP_GGUF` removed from the process environment (not blanked)
and `--mtp` passed literally where a drafter is wanted.

| gate | expected | observed |
| --- | --- | --- |
| `moe-gate` CPU (Qwen3.6-35B-A3B) | 64/64 ids | **PASS 64/64**, 0.69 s/token |
| `moe-gate --device` | 64/64 ids | **PASS 64/64**, 126.53 tok/s, 21.69 GiB |
| `moe-gate --device --bucket-stats`, `HIPML_MOE_PREFILL_MIN_TOKENS=1` | 64/64 ids, bucketed path | **PASS 64/64**, 195.8 tok/s prefill, 41 of 256 experts, 2,624 padded (36.44x) |
| `oracle-gate` (Qwen3.6-35B-A3B), bucketed chunk 2048 | 20/20 ids | **PASS 20/20** |
| `row-dequant` | 10 formats exact | **10/10 exact**, 2048/2048 values each |
| `load` (Qwen3.6-35B-A3B) | 0 unsupported, 256/8/512/512 | **0 unsupported**, 21.41 GiB, 20/20 round-trip, Q8_0 rows exact |
| `gpu-bench`, 511 ids, bucketed | same id as the token loop | **2,849.11 tok/s prefill**, last id 37986 |
| `gpu-bench`, 511 ids, token loop | same id as the bucketed path | **140.72 tok/s** (was 146.89 when §8.6 was written), last id 37986 |
| `gpu-bench --bucket-stats`, 511 ids | histogram + padding at 16/32/64 | **216 of 256, 4,088 slots, median 5**, 3.68x at the shipped 64 |
| `ppl` (Qwen3.6-35B-A3B) | close to llama | **2.2263** bucketed, vs 2.2228 (+0.16%) |
| `ppl`, `moe-matrix` corpus, ctx 512 (Qwen3.6 / MiniCPM / Sage / Arete) | 2.0982 / 2.2761 / 2.1804 / 2.0384 | **2.0982 / 2.2761 / 2.1804 / 2.0384** |
| MoE decode, depth 0, 200 tokens | 128.2 tok/s, unchanged | **128.18 tok/s** (7.802 ms/token); the single-sequence decode path is untouched by this slice, and the 125.89 recorded here before is the same path on a slower run |
| `gpu-bench`, 511 ids, chunk 512 | 2,849 tok/s | **2,853.14 tok/s**, last id 248046 |
| `gpu-bench`, 511 real-text ids, chunk 512 | 3,001.84 tok/s | **2,877.21 tok/s** (2,874.62 / 2,877.21 / 2,877.51), last id 37986; §8.6's same cell reads 2,870.95 |
| `gpu-bench`, 8,184 ids, chunk 2048 | 3,620 tok/s | **3,777.16 tok/s**, last id 284 |
| `batch-bench --slots 4 --tokens 32 --check --sweep` | every slot, every width | **B=1..=4 all reproduce the B=1 path** (4 slots × 32 ids) |
| `batch-bench --slots 16 --tokens 16 --check --sweep` | every slot, every width | **15 of 16 slots; slot 4 step 13 (382 vs 290)** — §8.7.5 |
| serve cell b, 8 slots, drafts 0/1/2 | llama.cpp's faster backend, same hold | **355.0 / 354.9 / 353.8 tok/s** (flat: a wave does not take `Mode::Spec`); against Vulkan plain 340.3 in the same session (+4.3%), against the census 369.71 (−4.0%); §8.10 |

Dense path, same binary, so the new `Ffn` enum and the row gather did not move
anything that was already gated:

| gate | expected | observed |
| --- | --- | --- |
| `oracle-gate` UD, 20 steps, chunk 2048, q8 | PASS 20/20, 0.1086 at 101748 | PASS 20/20, **0.1086 at 101748** |
| `oracle-gate` stock, 20 steps | PASS 20/20, 0.1481 at 34114 | PASS 20/20, **0.1481 at 34114** |
| greedy 12 ids, stock | `11751 13 198 760 6511 314 9564 369 19241 13 198 760` | identical |
| `parity --deep` | 74 passed | **74 passed**, 0 failed, 1 skipped (`sample_top_p_f32`) |
| `parity prefill --deep` | 15 passed | **15 passed**, 0 failed, 0 skipped |
| `batch-bench --slots 16 --check` | 16/16 on every slot | 16 slots, **all slots reproduce the B=1 path** over the full stream |
| `spec-bench --drafts 3 --kv q8` | 128/128 exact every row | **128/128** on all four rows |
| `spec-bench --drafts 3 --kv f32` | 128/128 exact every row | **128/128** on all four rows |
| `column-bench --check` | 0 differ | **0 differ** on 7,946,240 logits, hidden and residual |
| `column-bench --check` (Qwen3.6-35B-A3B, routed FFN) | 0 differ | **0 differ** on 7,946,240 logits, hidden and residual, q8 and f32, 32 columns |
| `spec-bench` Qwen3.6-35B-A3B, drafts 1/2/3, q8 | 128/128 exact, faster than plain | **128/128**, 136.46 / 141.64 / 132.16 tok/s against 127.50 |
| `spec-bench` Qwen3.6-35B-A3B, drafts 1/2/3, f32 | 128/128 exact | **128/128**, 134.74 / 139.43 / 129.79 tok/s against 125.05 |
| `mtp-batch-bench` Qwen3.6-35B-A3B, drafts 1/2 | exact to B≤4 | **B=1 and B=4 exact**, 4/4 slots; B=8 not exact (slot 5 @101, slot 6 @42 at d1) — §8.10 |
| `serve` `/v1/chat/completions`, MiniCPM-V-4.7 | 200, not the reported 400 | **200**, 18 prompt / 8 completion tokens; the base binary is still **400** |
| `probe_exact.py --ctx 32768` | PASS | PASS, 48 replies id-for-id across drafts 1,2,3 |
| `ppl` (Arete-27B) | 2.1824 vs llama 2.1791 | unchanged: **2.0384** on `C:\mm\moe-matrix\corpus.ids` on both this binary and the parent, where 2.1824 is the `calib_longctx` 4-window corpus (no ids dump for it on the box) |
| dense decode, depth 0, 200 tokens | ~33.2 tok/s | **33.37 tok/s** (29.967 ms/token) on the UD file this session; the census cell is 33.44 |
| `cargo test --workspace` | clean | **101 passed, 0 failed** (local and on the GPU host) |
| `cargo fmt --all -- --check` | clean | clean |
| `scripts/check-public-tree.sh` | clean | clean |
| `scripts/check-doc-links.py` | all links resolve | 283 links in 53 files |

The two oracle maxima are the same numbers every slice since the prefill-tile
note reports, which is the expected result for a change that does not touch the
dense forward pass.

## 10. What is not verified, and what is out of scope

* **MTP.** `blk.40` is a MoE block; `mtp.rs` now builds the drafter's FFN as
  a `MoeBlock` when the file has experts, so the block loads and runs. The two
  BF16 router tensors are widened to f32 at upload (2 MB) because no GEMV reads
  BF16. §8.8 is the single-stream measurement and §8.10 the batch one; the
  batch drafter and verify are built and exact at B≤4.
* **Grouped GEMM / bucketed prefill: shipped, at 79% of llama.cpp's warm 512
  cell.** §7.4 and §8.6 are the path and the numbers; §8.6.1 is why the baseline
  is 3,588.74 and not 3,375.05 or 4,529.55, and §8.6.2 is the tile sweep. What is
  left out of scope here: the tile height, which the sweep left at 64; a fused
  gate/up launch (two launches and two weight streams per expert now); and the
  dense path's `seq ssm` 41.8 ms of a 512-token chunk, which is 23% of that cell
  and belongs to the SSM note.
* **The 512 cell's next lever is the bank read, not the tile.** §8.6.2 prices the
  per-tile staging at 173 GB/s for a 1.9 MB bank, against the 495 GB/s §8.5
  measured for 0.59 MB. Re-tiling the grouped GEMM's *read* — not its MMA — is
  the open item, and nothing in this slice touched it.
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
* **Batch/serve.** The batched decode engine now serves a MoE file (§8.7): the
  routed FFN runs as one gather launch per bank with the slot on `grid.z`. The
  per-slot id gate holds at B=1..4 and not above, because the batched step is
  not the single-stream step from B=2 (the batched attention kernel is a
  different one, by design) and routing amplifies the difference — §8.7.5 is
  the measurement and the decision that is owed.
* **`serve --drafts` on a MoE file: the batch path is built, the server does not
  use it in a wave.** A lone request has speculated on a MoE file since §8.8
  (`MtpGpu` has a routed body; `probe_exact.py` is 48 replies id-for-id). Since
  §8.10 the batch drafter (`batch/mtp.rs`) and verify (`batch/verify.rs`) load
  and run on the file too, and are token-exact at B≤4. What is still true is
  that `serve` chooses `Mode::Spec` only when a request is alone, so the 8-slot
  cell does not speculate; §8.10 measures that and prices it (0.97× at d1,
  0.88× at d2 at B=8), so the wiring is not the missing win.
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
