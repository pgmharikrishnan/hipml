# FP4: `gfx1201` has no FP4 instruction, and the competitor's format is not FP4

Base `1473a9c` (origin/main). Box: R9700 (`gfx1201`, 64 CU, 32 GB, 640 GB/s
spec). ROCm SDK 7.2.53211, AMD LLVM 22.0.0git. Follows
[`external-comparison.md`](external-comparison.md), which recorded two external
projects' published numbers and concluded that a 4-bit format is the one
remaining decode lever. This note answers the question that leaves open: is
there hardware-mapped FP4 on this chip.

CPU-only. No kernel was launched and no GPU lock was taken: everything here is
`llvm-mc`, header inspection, a string scan of `clang.exe`, and `-c` / `-S`
compiles.

## 0. Question and answer

Question: does `gfx1201` do native 4-bit matrix math, because fewer bytes per
weight is the only lever left on decode?

Answer: **there is no FP4 instruction of any kind on `gfx1201`, and the
competitor's `ROCmFP4` is not an FP4 hardware format either — it is an integer
codebook read by an integer dot product.** So there is nothing to port and
nothing to map. The lever is still bytes per weight, it is worth about **+7%**
decode rather than the +13.8% a pure 4.00-bit format would imply, and it needs
no new ISA at all.

Three answers, in the order the brief asked:

| question | answer | where |
| --- | --- | --- |
| (a) supported by the ISA? | **No.** Zero FP4 instructions on `gfx1201`. | §1 |
| (b) reachable from HIP without inline asm? | **No** for FP4; **yes** for FP8 and int4, which this chip does have | §3 |
| (c) worth the engineering? | **No**, and the premise is a category error: decode is a bandwidth wall, so a matrix instruction cannot raise tok/s | §8 |

Two findings turned out to be worth more than the question asked, and they are
in §2 because they change what we do next: **our card has native FP8 and int4
WMMA that the competitor's Strix Halo does not**, and **FP4 WMMA exists in this
LLVM but only on `gfx1250`**.

## 1. The ISA: no FP4 on `gfx1201`

Method: each mnemonic assembled in isolation with
`llvm-mc -triple=amdgcn -mcpu=gfx1201 -show-encoding`, one candidate per file, so
a rejection cannot mask the next line. The message distinguishes three cases:
`invalid instruction` means the mnemonic does not exist in the AMDGPU target at
all; `instruction not supported on this GPU` means it exists but is gated to
another `mcpu`; `invalid operand` means the mnemonic exists for this GPU and the
operand width was wrong in that probe.

### 1.1 FP4 forms — every candidate rejected

| mnemonic | assembles | assembler message |
| --- | --- | --- |
| `v_wmma_f32_16x16x64_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_wmma_f16_16x16x64_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_wmma_f32_16x16x128_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_wmma_f16_16x16x128_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_wmma_f32_16x16x32_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_wmma_f16_16x16x32_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_wmma_f32_16x16x16_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_mfma_f32_16x16x128_fp4` | **no** | `error: invalid instruction` |
| `v_mfma_f32_32x32x64_fp4` | **no** | `error: invalid instruction` |
| `v_mfma_f16_16x16x128_fp4` | **no** | `error: invalid instruction` |
| `v_swmmac_f32_16x16x64_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_swmmac_f16_16x16x64_fp4_fp4` | **no** | `error: invalid instruction` |
| `v_dot2_f32_fp4`, `v_dot4_f32_fp4` | **no** | `error: invalid instruction` |
| `v_pk_fma_fp4`, `v_cmp_fp4`, `v_mul_f32_fp4` | **no** | `error: invalid instruction` |

### 1.2 FP4 converts — also absent

| mnemonic | assembles | note |
| --- | --- | --- |
| `v_cvt_f32_fp4` | **no** | `error: invalid instruction, did you mean: v_cvt_f32_f64, v_cvt_f32_fp8?` |
| `v_cvt_f16_fp4` | **no** | `error: invalid instruction` |
| `v_cvt_pk_f32_fp4` | **no** | `error: invalid instruction, did you mean: v_cvt_pk_f32_fp8?` |
| `v_cvt_pk_fp4_f32` | **no** | `error: invalid instruction` |
| `v_cvt_scalef32_pk_fp4_f32` | **no** | `error: instruction not supported on this GPU` (gfx950-only) |

**There is not one FP4 instruction of any kind on `gfx1201`** — no matrix op, no
dot, no convert. FP4 is entirely absent from this chip's ISA.

### 1.3 What the chip does have: FP8 and int4 WMMA

| mnemonic | assembles | encoding |
| --- | --- | --- |
| `v_wmma_f32_16x16x16_fp8_fp8` | **yes** | `[0x00,0x40,0x46,0xcc,0x08,0x15,0x02,0x1c]` |
| `v_wmma_f32_16x16x16_fp8_bf8` | **yes** | `[0x00,0x40,0x47,0xcc,0x08,0x15,0x02,0x1c]` |
| `v_wmma_f32_16x16x16_bf8_fp8` | **yes** | `[0x00,0x40,0x48,0xcc,0x08,0x15,0x02,0x1c]` |
| `v_wmma_f32_16x16x16_bf8_bf8` | **yes** | `[0x00,0x40,0x49,0xcc,0x08,0x15,0x02,0x1c]` |
| `v_wmma_i32_16x16x16_iu4` | **yes** | `[0x00,0x40,0x45,0xcc,0x08,0x13,0x02,0x1c]` |
| `v_wmma_i32_16x16x32_iu4` | **yes** | `[0x00,0x40,0x4a,0xcc,0x08,0x15,0x02,0x1c]` |
| `v_wmma_i32_16x16x16_iu8` | **yes** | `[0x00,0x40,0x44,0xcc,0x08,0x15,0x02,0x1c]` |
| `v_dot8_i32_iu4` | **yes** | `[0x00,0x40,0x18,0xcc,0x01,0x05,0x02,0x1c]` |
| `v_dot4_i32_iu8` | **yes** | `[0x00,0x40,0x16,0xcc,0x01,0x05,0x02,0x1c]` |
| `v_dot8c_i32_i4` | **no** | `error: instruction not supported on this GPU` |
| `v_wmma_f32_16x16x32_fp8_fp8` | **no** | `error: invalid instruction` (no K=32 FP8 form) |
| `v_wmma_f16_16x16x16_fp8_fp8` | **no** | `error: invalid instruction` (no f16-accumulate FP8) |

Operand shape, for FP8: `v[0:7]` destination (8 VGPRs), `v[8:9]` and `v[10:11]`
sources — 2 VGPRs each = 8 FP8 per lane × 32 lanes = 256 = 16×16. The width was
pinned down by probing: `v[8:11]` gives `invalid operand for instruction`.

So `gfx1201` **does** have a native 4-bit matrix operand path — `iu4`, packed
nibbles straight into WMMA — but it is **integer** (int4 × int4 → int32), not
FP4. That distinction is the whole finding.

### 1.4 Converts the chip does have

| mnemonic | assembles | note |
| --- | --- | --- |
| `v_cvt_f32_fp8` | **yes** | emits as `v_cvt_f32_fp8_e32`, `[0x01,0xd9,0x00,0x7e]` |
| `v_cvt_pk_f32_fp8` | **yes** | `v_cvt_pk_f32_fp8_e32`, `[0x02,0xdd,0x00,0x7e]` |
| `v_cvt_pk_fp8_f32` | **yes** | `[0x00,0x00,0x69,0xd7,0x01,0x05,0x02,0x00]` |
| `v_cvt_f32_bf8` | **yes** | `v_cvt_f32_bf8_e32` |
| `v_cvt_pk_bf8_f32` | **yes** | `[0x00,0x00,0x6a,0xd7,0x01,0x05,0x02,0x00]` |
| `v_cvt_f16_fp8` | **no** | `error: instruction not supported on this GPU` |
| `v_cvt_scalef32_pk_fp8_f32` | **no** | `error: instruction not supported on this GPU` (gfx950-only) |

### 1.5 Control — direct-to-LDS, reproducing `prefill-tile-ceiling.md` §5

| mnemonic | assembles | assembler message |
| --- | --- | --- |
| `global_load_lds_dword v0, s[0:1]` | **no** | `error: instruction not supported on this GPU` |
| `buffer_load_dword v0, s[0:3], 0 offen lds` | **no** | `error: invalid operand for instruction` |
| `s_buffer_load_dword s0, s[0:3], 0x0` | **yes** | canonicalised to `s_buffer_load_b32`, `[0x00,0x00,0x02,0xf4,0x00,0x00,0x00,0xf8]` |

These reproduce both lines recorded in
[`prefill-tile-ceiling.md`](prefill-tile-ceiling.md) §5, which read the same pair
as "the gfx9 `buffer_load ... lds` form does not assemble for this target". The
`s_buffer_load_dword` row is a positive control: the syntax and target are fine,
so the two rejections above are the instruction, not the harness.

## 2. Two findings worth more than the question asked

Neither of these was asked for. Both change what we do next.

### 2.1 Our card has matrix capability the competitor's Strix Halo does not

FP8 WMMA and int4 WMMA are **`gfx12`-only**. Probed across five `mcpu` values:

| instruction | gfx1100 (RDNA3) | gfx1151 (RDNA3.5, Strix Halo) | gfx1200 | **gfx1201 (ours)** | gfx950 (CDNA4) |
| --- | --- | --- | --- | --- | --- |
| `v_wmma_i32_16x16x32_iu4` | not supported | **not supported** | OK | **OK** | not supported |
| `v_wmma_i32_16x16x16_iu4` | operands invalid | **operands invalid** | OK | **OK** | not supported |
| `v_wmma_f32_16x16x16_fp8_fp8` | not supported | **not supported** | OK | **OK** | not supported |
| `v_dot8_i32_iu4` | OK | OK | OK | **OK** | not supported |
| `v_cvt_f32_fp8` | not supported | **not supported** | OK | **OK** | OK |

The competitor's published numbers come from `gfx1151`
([`external-comparison.md`](external-comparison.md) §1). On that part, FP8 WMMA,
int4 WMMA and hardware FP8 convert are all **absent**. So the project whose FP4
format prompted this probe is running on a chip with *less* matrix capability
than the one we are trying to catch up on — which is itself evidence that its
gain cannot be a matrix-instruction gain.

This is reachable without `rocWMMA` and without inline asm. No `rocWMMA` is
installed in this SDK at all (a recursive search for `*wmma*` across the ROCm
tree returns nothing), but it is not needed: the compiler carries the builtins,
and two lines of HIP emit the instructions (§3.1).

### 2.2 FP4 WMMA does exist in this LLVM — on `gfx1250`, not `gfx1201`

| mnemonic | gfx1201 | gfx1250 |
| --- | --- | --- |
| `v_wmma_f32_32x16x128_f4` | `error: instruction not supported on this GPU` | **assembles**, `[0x00,0x40,0x88,0xcc,0x10,0x41,0x02,0x1c]` |
| `v_wmma_f32_16x16x128_f8f6f4` | `error: instruction not supported on this GPU` | recognised (`wrong register tuple size for MATRIX_FMT_FP8` at the probed width) |

So "FP4 matrix math" is a real thing in this toolchain, and `gfx1201` is not a
downclocked version of the chip that has it. The compiler agrees: the builtin
`__builtin_amdgcn_wmma_f32_32x16x128_f4` is gated on `gfx1250-insts`
(§3.2). Anyone reviving this idea later should start by asking which silicon,
because the answer changed between `gfx1201` and `gfx1250`.

**What §2 changes.** The decode lever is unchanged — bytes per weight, §6. But
if prefill ever becomes the constraint, this chip has FP8 and int4 tiles that
the previous roofline work assumed were unavailable, and they are two builtins
away. That is a separate note; it is not needed for decode.

## 3. Reachability from HIP

The builtins were enumerated directly out of the compiler binary: 67 distinct
`__builtin_amdgcn_wmma_*` names.

Builtins tagged `_gfx12`, i.e. the ones this chip can execute:

```text
__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12
__builtin_amdgcn_wmma_f32_16x16x16_fp8_bf8_w32_gfx12
__builtin_amdgcn_wmma_f32_16x16x16_bf8_fp8_w32_gfx12
__builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12
__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12
__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12
__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12
(+ f16/bf16 variants, and _w64 equivalents)
```

Builtins mentioning FP4 — **none carries a `_gfx12` tag**:

```text
__builtin_amdgcn_wmma_f32_32x16x128_f4
__builtin_amdgcn_wmma_f32_16x16x128_f8f6f4
__builtin_amdgcn_wmma_scale_f32_32x16x128_f4
__builtin_amdgcn_wmma_scale16_f32_32x16x128_f4
__builtin_amdgcn_cvt_scalef32_pk_fp4_f32   (and 17 more cvt_*_fp4)
```

### 3.1 Proof: plain HIP compiles and emits the instruction

Source, with no asm and no intrinsics headers:

```cpp
typedef float f8v __attribute__((ext_vector_type(8)));
typedef int   i2v __attribute__((ext_vector_type(2)));
typedef int   i8v __attribute__((ext_vector_type(8)));

__global__ void k_fp8(f8v *out, i2v a, i2v b, f8v c) {
  out[0] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}
__global__ void k_int4(i8v *out, i2v a, i2v b, i8v c) {
  out[0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(1, a, 1, b, c, 0);
}
```

`hipcc --cuda-device-only -S --offload-arch=gfx1201` emits, verbatim:

```text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1201"
	v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[10:11], v[0:7]
	v_wmma_i32_16x16x32_iu4 v[0:7], v[8:9], v[10:11], v[0:7] neg_lo:[1,1,0]
```

Note `neg_lo:[1,1,0]`: the int4 form carries per-operand signedness, which is
what a symmetric quantised format needs. Two lines of HIP, no asm, no `rocWMMA`.

### 3.2 Proof: FP4 is impossible on this chip

The same source shape with the FP4 builtin:

```cpp
out[0] = __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, (short)0, out[1]);
```

```text
error: '__builtin_amdgcn_wmma_f32_32x16x128_f4' needs target feature
       gfx1250-insts,wavefrontsize32
1 error generated when compiling for gfx1201.
```

A hard front-end rejection naming the feature. No flag makes this legal on
`gfx1201`, and the ISA probe of §1 independently agrees.

## 4. What the ROCm headers expose

| file | size | finding |
| --- | --- | --- |
| `include/hip/hip_fp4.h` | 32 lines | thin wrapper, includes `amd_detail/amd_hip_fp4.h` |
| `include/hip/amd_detail/amd_hip_fp4.h` | 460 lines | `__hip_fp4_e2m1`, `__hip_fp4x2_e2m1`, `__hip_fp4x4_e2m1` |
| `include/hip/amd_detail/amd_hip_fp8.h` | 3208 lines | `__hip_fp8_e4m3`, `__hip_fp8_e5m2` plus fast paths |
| `include/hip/amd_detail/amd_hip_mx_common.h` | 30 lines | only `enum hipRoundMode` |

**FP4 has no hardware path on `gfx1201`.** Every conversion in
`amd_hip_fp4.h` tests `#if __gfx950__` for the hardware route and falls back to
software otherwise:

```c
// amd_hip_fp4.h:162-176
__hip_cvt_float_to_fp4(const float x, ...) {
  ...
#if __gfx950__
  u.ui32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(u.ui32, x, 0.0f, 1.0f, 0);
  return u.fp4[0];
#else
  u.ui32 = fcbx::from_float<float, fcbx::Encoding::E2M1, true>(x, 0);
  return u.fp4[0];
#endif
}
```

`gfx1201` takes the `#else`: bit-twiddling in `fcbx::`, not silicon.
`__gfx950__` is the only architecture name anywhere in the file — no `gfx12`
branch exists to take.

**FP8 does have a hardware path on `gfx1201`** — `amd_hip_fp8.h:33-39`:

```c
#if (defined(__gfx942__) || defined(__gfx1200__) || defined(__gfx1201__) || \
     defined(__gfx950__)) && __HIP_DEVICE_COMPILE__
#define HIP_FP8_CVT_FAST_PATH 1
```

and `amd_hip_fp8.h:44-47` sets `HIP_FP8_TYPE_OCP 1` for `gfx12`. The fast path
calls `__builtin_amdgcn_cvt_f32_fp8` and `cvt_pk_fp8_f32` — the instructions
§1.4 accepted. So **FP8 is native and OCP-conformant on this chip; FP4 is not.**
The ROCm stack matches the ISA exactly.

## 5. What `ROCmFP4` actually is

Sources read (Apache-2.0; GitHub's licence detector reports `NOASSERTION` because
the file is the short-form notice rather than the full text, the same caveat
[`external-comparison.md`](external-comparison.md) §1 records):

- `charlie12345/ROCmFPX` — `ggml/rocmfp4/README.md`, `ggml/rocmfp4/rocmfp4.h`,
  `ggml/rocmfp4/rocmfp4.c`
- `julianmb/halofpx` — `docs/QUANTIZATION_RECIPES.md`, `LICENSE`

`julianmb/halofpx` is a Python wrapper only: its `engine/` directory contains
nothing but `.gitkeep`. All the format work is in `charlie12345/ROCmFPX`.

### 5.1 Block layout (`ggml/rocmfp4/rocmfp4.h`)

```c
#define QK_ROCMFP4 32          // 32 weights per block
#define QR_ROCMFP4 2

typedef struct {               // dual-scale, "pure 4.50 BPW"
    uint8_t qs[QK_ROCMFP4/2];  // 16 B: 32 packed 4-bit codes
    uint8_t e[2];              // 2 B: one unsigned E4M3 scale per 16-weight half
} block_rocmfp4;               // 18 B / 32 weights = 4.50 bpw

typedef struct {               // FAST, single-scale, 4.25 BPW
    uint8_t qs[QK_ROCMFP4/2];  // 16 B
    uint8_t e;                 // 1 B: one unsigned E4M3 scale per 32-weight block
} block_rocmfp4_fast;          // 17 B / 32 weights = 4.25 bpw
```

### 5.2 The codebook is integer, not IEEE FP4 (`rocmfp4.c:19-33`)

```c
// "signed integer FP4-like codebook at half-scale ... E2M1-derived, but the
//  largest magnitude is retuned from 12 to 10 after sampling Qwen3 dense tensors"
static const int8_t rocmfp4_codebook[16] = {
     0,  1,  2,  3,  4,  6,  8, 10,
     0, -1, -2, -3, -4, -6, -8,-10,
};
```

Bit 3 is the sign; the low 3 bits decode as `mag3 <= 4 ? mag3 : 2*mag3 - 4`,
giving magnitudes `{0,1,2,3,4,6,8,10}` at half-scale, i.e.
`{0,0.5,1,1.5,2,3,4,5} × scale`. This is deliberately **not** OCP E2M1
(`{0,0.5,1,1.5,2,3,4,6}`): the top level was pulled from 6 to 5 to cut outlier
pull, and the levels are integers so they can feed an integer dot product.

### 5.3 How a block is consumed — integer dot product, not a matrix operand

`rocmfp4.c:670-673`, the x86 reference, which is the same shape as the HIP path:

```c
// Decode one 32-weight block's low and high nibble streams through the
// Codebook10 table with a single PSHUFB, then integer-dot each against its half
// of the q8_0 block. Integer sums are order-independent...
```

The HIP entry point is `rocmfp4_vec_dot_q4_0_q8_0(...)` (`rocmfp4.h`) — a
**vector dot against `q8_0` activations**, i.e. `dp4a`-class integer arithmetic,
then one FP8 scale multiply per block. No matrix instruction is involved on any
backend. That is the same computational shape as the `Q4_K` path we already run.

### 5.4 Their own assessment of the matrix path

From `ggml/rocmfp4/README.md`:

> `:674` — "ROCmFP4 K/V is forced to the scalar FA path because the current
> custom decode is **not a coopmat/native matrix-core FP4 path**."

> `:994-996` — "it is **not yet a native rocWMMA FP4 tensor-core
> implementation**. Current speed gains come from the compact block layout and
> backend decode paths; deeper rocWMMA/cooperative-matrix work is future
> optimization work."

> `:1003-1007` — "The bundled rocWMMA 7.1.0 headers expose gfx12 WMMA paths for
> FP8/BF8 and integer 8-bit inputs, but **no native FP4 input type or FP4
> WMMA/MFMA builtin is visible locally**. A true matrix-core ROCmFP4 path
> therefore needs a measured unpack/convert strategy first, such as ROCmFP4
> Codebook10 to int8 WMMA or FP8 WMMA tiles, before claiming native FP4
> tensor-core execution."

That is an independent confirmation of §1–§4 from the project whose README claim
the brief quotes. The claim is that the blocks "map directly to RDNA 3.5
cooperative-matrix (`KHR_coopmat`) operands"; the code is an integer codebook and
an integer dot product, and their own notes say the matrix path is future work.
Their gain is the block layout — 4.25–4.50 bpw instead of 4.55 — which is exactly
the byte-count conclusion [`external-comparison.md`](external-comparison.md) §5
reached from their published numbers alone.

## 6. The ceiling: +7.1%, not +13.8%

Our figures, from this tree: 15.48 GB streamed per token, 30.02 ms per token,
33.31 tok/s at depth 0, `UD-Q4_K_M` at 4.55 bits/weight effective, 640 GB/s
spec, 585.6 GB/s pure read ([`external-comparison.md`](external-comparison.md) §4,
[`decode-roofline.md`](decode-roofline.md) §2).

Byte reduction is linear in the weight-stream time, because the stream is
bandwidth-bound. Scaling the whole token by the byte ratio is therefore an upper
bound:

```text
tok/s_new = 33.31 × (4.55 / bpw_new)
```

| format | bpw | file | GB/token | ms/token | tok/s | vs now |
| --- | --- | --- | --- | --- | --- | --- |
| `UD-Q4_K_M` (now) | 4.55 | 15.42 GiB | 15.48 | 30.02 | **33.31** | — |
| `ROCmFP4` dual-scale | 4.50 | 15.25 GiB | 15.31 | 29.69 | 33.68 | +1.1% |
| **`ROCmFP4_FAST`** | **4.25** | **14.40 GiB** | **14.46** | **28.04** | **35.66** | **+7.1%** |
| hypothetical pure 4-bit | 4.00 | 13.56 GiB | 13.61 | 26.39 | 37.89 | +13.8% |

Worked example, at 4.00 bpw:

```text
bytes/token = 15.48 GB × (4.00 / 4.55) = 13.61 GB
time        = 13.61 GB ÷ 515.66 GB/s   = 26.39 ms
tok/s       = 1000 / 26.39             = 37.89
cross-check = 13.61 GB ÷ (640 × 0.806) = 26.39 ms
```

**A pure 4.00 bpw is not reachable with per-block scales.** One scale byte per
32 weights costs 8/32 = 0.25 bpw on its own, so **4.25 is the floor for a
block-32 format**. Getting to 4.06 bpw needs block-128 scales, which trades
quality for the last ~1.7%. So **+7% decode is the honest prize; +14% is the
arithmetic ceiling of an idealised format that no real block layout achieves.**

**Refinement, and it lowers the number.** The table above scales the whole token,
but about **3.48 ms of the 30.02 ms token is not the weight stream**
([`decode-ablation.md`](decode-ablation.md) §2: `gemv_only` 28.524 ms against a
30.0 ms token). If that fixed cost does not shrink with the format, only the
~26.5 ms stream scales:

```text
4.25 bpw: 26.5 × 0.9341 + 3.48 = 28.25 ms → 35.4 tok/s (+6.3%)
4.00 bpw: 26.5 × 0.8791 + 3.48 = 26.80 ms → 37.3 tok/s (+12.0%)
```

So the realisable band is **+6% to +12%**, not +7% to +14%. The byte ratio is a
bound on the stream, and the stream is not the whole token.

### 6.1 Quality caveat

- 4.25 bpw is **below** the 4.55 we run. `Q4_K_M`'s extra ~0.55 bpw buys
  super-block scales plus asymmetric min offsets; `ROCmFP4` replaces that with
  one symmetric scale per 32 weights and a symmetric integer codebook. That is a
  genuine accuracy risk, not a rounding detail.
- The competitor had to rescue it, by their own notes: the pure FAST path "gives
  up too much PPL to be the default quality target"; `_LEAN` protects token
  embeddings with `Q5_K`, `_COHERENT` with `Q6_K`, and `_STRIX` additionally
  spends the dual-scale layout on attention K/V (`ggml/rocmfp4/README.md:12-30`).
  **Their quality-biased presets climb to 4.38–4.49 bpw, at which point the byte
  win over 4.55 collapses to +1–3%** — the +1.1% row above, i.e. nothing.
- **We have no quality harness.** No perplexity or KL gate exists in this tree, so
  none of this is measurable on our side today. A format change cannot land
  without one, and building it is a prerequisite, not a follow-up.

## 7. Corrections to the brief

Two figures in the brief did not survive contact with this tree's measurements.
They are recorded as corrections rather than quietly restated, because both were
load-bearing in the decision.

**7.1 The size headline mixes two baselines.** The brief gives "−14.9% size,
+13.5% decode". Those are the competitor's own numbers and they are internally
consistent — `13.55 / 15.92 = 0.851` against *their* stock `Q4_K_M`
([`external-comparison.md`](external-comparison.md) §2a). But our file is not
their file: 15.42 GiB against their 15.92 GiB. Against **our** file the same
format is **12.1% fewer bytes**, which is the figure
[`external-comparison.md`](external-comparison.md) §5 already uses. The bits
comparison is smaller still: 4.25 against 4.55 bpw is **6.6%**. So the headline
is −14.9% against their baseline, −12.1% against ours, and −6.6% on bits per
weight — three different numbers for one change. The related inconsistency: their
13.55 GiB at 4.26 bpw implies ~27.3 G weights, while our 15.42 GiB at 4.55
implies ~29.1 G, so the two files cannot both be the same parameter count at
those bpw figures. The byte-ratio model in §6 is unaffected, but the −14.9%
should not be quoted as a property of our file.

**7.2 "96.6% of a pure read" and "640 GB/s" are different measurements.** The
brief applies both to the same token-level calculation, and they cannot both hold
there. They describe different scopes:

| figure | what it measures | value |
| --- | --- | --- |
| 96.6% | the routed GEMV against a **pure read of the same bytes at the same shapes** — 26.98 ms against 26.08 ms ([`decode-roofline.md`](decode-roofline.md) §2) | 96.6% |
| 585.6 GB/s | the **dedicated read pass**, best of three ([`decode-ablation.md`](decode-ablation.md) §3) | 585.6 GB/s |
| 640 GB/s | the **GDDR6 spec peak** | 640 GB/s |
| 515.7 GB/s | **the whole token**, 15.48 GB ÷ 30.02 ms | 80.6% of spec |

So decode does not sustain "96.6% of 640 GB/s". The weight stream runs at about
**583 GB/s** — near the 585.6 GB/s read, which is what the 96.6% says — and the
**whole token** runs at **515.7 GB/s end-to-end, i.e. 80.6% of the 640 GB/s
spec**, because ~3.48 ms of the token is not the weight stream. That 80.6% is
the same efficiency-per-byte figure [`external-comparison.md`](external-comparison.md)
§5 reports as "81% for us". The byte-ratio scaling in §6 is independent of which
figure is used, and the cross-check in §6 reproduces it exactly.

## 8. Verdict

**(a) Supported by the ISA? — No.** Zero FP4 instructions on `gfx1201`: no
`v_wmma_*fp4*`, no `v_mfma_*fp4*`, no `v_swmmac_*fp4*`, no `v_dot*fp4*`, no FP4
convert. FP4 WMMA exists in this LLVM as `v_wmma_f32_32x16x128_f4` and is gated
to `gfx1250-insts`. The chip does have native FP8 WMMA (4 operand-type
combinations, K=16) and native int4 WMMA (`16x16x16`, `16x16x32`), both
`gfx12`-only.

**(b) Reachable from HIP without inline asm? — No for FP4, yes for FP8 and
int4.** `__builtin_amdgcn_wmma_f32_32x16x128_f4` is a hard front-end error
(`needs target feature gfx1250-insts`). FP8 and int4 are ordinary builtins, and
both compiled for `gfx1201` and emitted real instructions from two lines of HIP.
`rocWMMA` is not installed and is not needed for those two.

**(c) Worth the engineering? — No, and the framing should change.** Decode is a
bandwidth wall: the stream runs at ~583 GB/s against a 585.6 GB/s read, and the
whole token at 80.6% of spec peak, with compute at low single-digit percent of
the part's capability. A matrix instruction cannot move a bandwidth wall. Even a
perfect FP4 WMMA would raise prefill, not decode. The competitor demonstrates
this: they have no FP4 matrix path either, and their entire gain is 4.25 bpw
instead of 4.55.

### 8.1 What to do instead

The lever is bytes per weight, and it needs no new ISA:

1. **First step — measure, do not port.** Before any kernel work, build the
   perplexity/KL gate this tree does not have, then quantise to a 4.25 bpw
   block-32 int4 codebook layout and compare against the current `UD-Q4_K_M`.
   Without the gate, +6% decode is unfalsifiable against an unknown quality loss,
   and the competitor's own preset ladder (§6.1) says the loss is real.
2. **If it passes**, the compute side already exists here: the current `Q4_K`
   `vec_dot` path is an integer dot product, and `gfx1201` additionally offers
   `v_dot8_i32_iu4` and `v_wmma_i32_16x16x32_iu4` with `neg_lo` per-operand
   signedness if prefill ever needs the wider tile. No FP4 hardware is required
   to capture the whole win.
3. **Expect +6%, not +14%**, and expect to spend most of the quality budget on
   embeddings and attention K/V, exactly as the competitor did.

### 8.2 What a GPU run would still have to settle

Two questions remain, both about **int4**, not FP4, and both needing the GPU:
does `v_wmma_i32_16x16x32_iu4` reach full rate on `gfx1201`, and does
`v_dot8_i32_iu4` beat the current `v_dot4_i32_iu8` sequence for the decode GEMV.
Neither affects the verdict above, which rests only on bytes per weight. They
were not run here because another work package held the GPU.

## 9. What this note does not establish

- It does not measure any kernel. No instruction above was executed; "assembles"
  and "emits" mean the toolchain accepts it, not that it runs at any given rate.
- It does not reproduce the competitor's numbers, and no external binary was
  built or run. §5 is a reading of their source and their README.
- It does not predict the quality cost of a 4.25 bpw format on this model. §6.1
  says only that the cost is real, that they had to rescue it, and that we cannot
  currently measure it.
- It does not claim the +6% is available today. §6 bounds it, and §8.1 makes the
  quality gate a prerequisite.
