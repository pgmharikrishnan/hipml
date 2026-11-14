# The MoE FFN: the conventions, settled against llama.cpp's ids

> *Engineering log from development; internal work-package IDs and paths have
> been sanitized or removed where they were machine-specific.*

`Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf` is the first mixture-of-experts file the
engine has opened: 753 tensors, 22,853,663,008 B, `qwen35moe`, 40 decoder blocks
plus one inline MTP block, 256 experts with 8 used. Every attention and
gated-delta-net tensor in it is shaped exactly like the 27B's; the FFN is the
only new thing.

This slice is **CPU only**: no kernel, no forward-pass change, no prefill, no
MTP. It settles the numerical conventions against llama.cpp before anyone
writes a GPU kernel, because four independent conventions (softmax domain,
top-k order, renormalisation, where the router weight is applied) can each leave
the ids subtly wrong, and a kernel built on a wrong one is the wrong kernel.

**Result: the CPU reference reproduces llama.cpp's 64 greedy ids on a fixed
prompt, 64/64, first try.** The file also loads: 753 tensors, 0 unsupported,
21.41 GiB resident.

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
| read + upload | 7.12 s (3.21 GB/s) |
| resident delta | 22,991,470,592 B (21.41 GiB) |
| BF16 tensors | 2, 1,052,672 B, supported |
| unsupported | **none** |
| round-trip | 20/20 byte-identical |

The embedding probe is skipped for this file and says so: its
`token_embd.weight` is Q8_0 and the device dequantisation probe has kernels for
Q4_K and Q6_K only. The payload check above already covers it byte for byte.
`lm_head` on the resident Q8_0 head runs at 124.9 GB/s.

## 7. Gates

Dense path, on the binary this branch ships, `--features hip`, model paths
passed with `--gguf`, `HIPML_MTP_GGUF` removed from the process environment
(not blanked) and `--mtp` passed literally where a drafter is wanted.

| gate | expected | observed |
| --- | --- | --- |
| `moe-gate` (Qwen3.6-35B-A3B) | 64/64 ids | **PASS 64/64**, 0.69 s/token |
| `load` (Qwen3.6-35B-A3B) | 0 unsupported, 256/8/512/512 | **0 unsupported**, 21.41 GiB, 20/20 round-trip |
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
| `cargo test --workspace` | clean | clean (local and on the GPU host) |
| `cargo fmt --all -- --check` | clean | clean |
| `scripts/check-public-tree.sh` | clean | clean |

The two oracle maxima are the same numbers every slice since the prefill-tile
note reports, which is the expected result for a change that does not touch the
forward pass.

## 8. What is not verified, and what is out of scope

* **MTP.** `blk.40` is a MoE block and `mtp.rs` requires a dense `ffn_gate`, so
  the drafter cannot load this file. The two BF16 tensors are MTP-only; the type
  exists so the file opens, not so the block runs.
* **GPU decode and prefill.** No kernel, no forward-pass change, no
  `GpuEngine::build` path for the expert bank. `Proj::of` would still reject a
  3-D tensor, and `gpu.rs` still builds a dense FFN per block, so a GPU load of
  this file stops there — by design.
* **One prompt, one seed, 64 ids.** The gate says the conventions are right; it
  does not bound how often a near-tie would flip a token between this f32 CPU
  reference and llama's GPU kernels. The margin print in `moe-gate` is the tool
  for that if it ever happens.
* **The load-bearing assumption** is that llama.cpp's MoE path on the ROCm
  backend and its CPU path agree on these ids. Only the GPU path was run; the
  box did not have the RAM for a CPU llama-server beside the loaded GPU model.
  If a later slice sees a divergence, that is the first control to repeat.
