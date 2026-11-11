# What is inside a dynamic quant: nine tensor types in one Q4_K_M file

*The `UD-` label names a bit budget, not a format. `UD-Q4_K_M` carries IQ4_XS, Q5_K, Q4_K, Q6_K, IQ4_NL, Q3_K and IQ3_S, and an engine that knows only the K-quants can size it without running it.*

`Qwen3.8-27B-UD-Q4_K_M.gguf` is 16,564,259,744 bytes and the name promises Q4_K, medium. Of its 866 tensors, 142 are Q4_K; IQ4_XS is the joint-largest group at 142 tensors and 5.30 GiB, and the file also carries Q5_K, Q6_K, Q8_0, IQ4_NL, Q3_K and IQ3_S. That is what `UD-` means.

## The inventory

The repository publishes 31 `.gguf` files: 25 quantised builds, two BF16 shards, two vision projectors, one importance matrix and the MTP sidecar. Three of the 25 are classic quants and 22 carry `UD-`. Files are `Qwen3.8-27B<name>.gguf`.

| name | GiB |
| --- | ---: |
| `-Q4_0` | 14.95 |
| `-Q4_1` | 16.34 |
| `-Q8_0` | 27.05 |
| `-UD-IQ1_S` | 5.77 |
| `-UD-IQ1_M` | 6.27 |
| `-UD-IQ2_XXS` | 6.71 |
| `-UD-IQ2_XS` | 7.10 |
| `-UD-IQ2_S` | 7.84 |
| `-UD-Q2_K_XL` | 9.27 |
| `-UD-IQ3_XXS` | 10.18 |
| `-UD-IQ3_S` | 11.21 |
| `-UD-Q3_K_XL` | 12.24 |
| `-UD-IQ4_XS` | 13.37 |
| `-UD-Q4_K_S` | 14.30 |
| `-UD-Q4_K_M` | 15.43 |
| `-UD-Q4_K_XL` | 16.45 |
| `-UD-Q5_K_S` | 17.48 |
| `-UD-Q5_K_M` | 18.51 |
| `-UD-Q5_K_XL` | 19.54 |
| `-UD-Q6_K` | 20.47 |
| `-UD-Q6_K_M` | 21.50 |
| `-UD-Q6_K_L` | 22.52 |
| `-UD-Q6_K_XL` | 23.56 |
| `-UD-Q8_K_L` | 26.12 |
| `-UD-Q8_K_XL` | 29.30 |

Type counts run from 2 in the uniform `-Q8_0` to 15 in `-UD-IQ3_S`. `-UD-Q8_K_XL` holds 2.57B parameters in 53 BF16 tensors, including the head and every fourth layer's q, k and v; `-UD-Q6_K` has no IQ format at all.

## What the header says

A GGUF header is self-describing: magic, version, a metadata list, then one record per tensor with name, dimensions, type id and offset. No payload is needed: each row below parses the first 4 MiB, then 16 MiB, over HTTP range requests. Cells are tensors / GiB of tensor data, F32 excluded; every file also holds 360 F32 tensors at 0.01 GiB, and totals run 0.01 GiB under the file size.

| type | `UD-Q4_K_S` | `UD-Q4_K_M` | `UD-Q6_K` | `UD-Q8_K_L` |
| --- | --- | --- | --- | --- |
| IQ4_XS | 172 / 5.86 | 142 / 5.30 | — | — |
| Q5_K | 80 / 2.34 | 142 / 4.87 | 115 / 5.12 | 11 / 0.32 |
| Q4_K | 95 / 2.73 | 74 / 2.93 | — | — |
| Q6_K | 18 / 1.42 | 28 / 1.68 | 241 / 12.83 | 57 / 2.56 |
| Q8_0 | 99 / 0.04 | 107 / 0.08 | 150 / 2.50 | 438 / 23.21 |
| IQ4_NL | 7 / 0.21 | 9 / 0.40 | — | — |
| Q3_K | 13 / 0.94 | 2 / 0.07 | — | — |
| IQ3_S | 15 / 0.54 | 2 / 0.07 | — | — |
| IQ3_XXS | 5 / 0.16 | — | — | — |
| IQ2_XS | 1 / 0.02 | — | — | — |
| IQ2_S | 1 / 0.03 | — | — | — |
| total | 866 / 14.29 | 866 / 15.42 | 866 / 20.46 | 866 / 26.11 |

Every build holds 27.32B parameters; only the distribution changes. `UD-Q4_K_S` is 1.13 GiB smaller and pushes 1.34B into IQ3_S and 0.45B into IQ3_XXS, while `UD-Q6_K` and `UD-Q8_K_L` stay inside the K-quant alphabet. In `UD-Q4_K_M` the FFN holds 17.38B and 99 of its 195 tensors are IQ4_XS, 8.82B in all; attention spreads 5.81B over Q4_K and Q5_K, and the SSM alpha and beta tensors are Q8_0 at 0.02 GiB, each 48 rows wide.

## Why IQ is not a K-quant with a different number

The K-quants are affine. A Q4_K super-block is 144 bytes for 256 values: fp16 `d`, fp16 `dmin`, twelve bytes of six-bit scale and min per 32-value sub-block, and 128 bytes of 4-bit quants. Reconstruction is `y = d * scale * q - dmin * min`. The IQ formats index a table instead:

| format | values | bytes | bits/value | value is |
| --- | ---: | ---: | ---: | --- |
| Q4_K | 256 | 144 | 4.50 | affine: scale and min per super-block, 6-bit scale per 32 |
| IQ4_XS | 256 | 136 | 4.25 | fp16 `d`, 6-bit scale per 32, 4-bit index into 16 entries |
| IQ4_NL | 32 | 18 | 4.50 | fp16 `d`, 4-bit index into 16 entries |
| IQ3_S | 256 | 110 | 3.44 | 4-bit scale per 32, 9-bit index into 4-byte grids, sign bit per value |

The 136 bytes of IQ4_XS are an fp16 `d`, a 16-bit `scales_h`, four bytes of `scales_l` and 128 bytes of indices. The reference dequantiser's inner loop is the specification:

```c
const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf)
             | (((x[i].scales_h >> 2*ib) & 3) << 4);
const float dl = d * (ls - 32);
for (int j = 0; j < 16; ++j) {
    y[j+ 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
    y[j+16] = dl * kvalues_iq4nl[qs[j] >>  4];
}
```

No min, no per-sub-block fp16 scale, no second operand: six bits of scale packed across two fields and biased by 32, and sixteen constants chosen once for every model using the format. A wrong implementation does not announce itself, because every index is in range by construction; a transposed nibble yields values of plausible magnitude, with no NaN and no fault. The text stays fluent and the model is wrong. The defence is a hand-computed reference vector per format, as `crates/hipml-core/tests/quant_reference.rs` pins the K-quant layouts.

## Why the block form has to survive on the device

The 142 IQ4_XS tensors hold 10.722B values in 5.30 GiB. Expanded: 42.9 GB as f32, 39.9 GiB; 21.4 GB as f16, 20.0 GiB. Neither fits beside the rest of a 27B model on a 32 GiB card, so it cannot be expanded at load: it stays blocks on the device, and three paths must know the layout — a decode GEMV unpacking scales in its inner loop, a batched GEMV reading them once for N streams, and a prefill GEMM dequantising a 256-value block into shared memory once per tile. The 107 Q8_0 tensors, at 0.08 GiB, are 0.15 GiB expanded.

## The trap that is not about formats

`UD-Q6_K` is 20.46 GiB of tensor data, which fits on 32 GiB with room for a cache. Its 115 Q5_K tensors hold 8.001B values. An engine that materialises a resident fp16 copy for its prefill GEMM pays 16.0 GB for those, or 14.9 GiB, taking the file to 35.4 GiB before a single KV page. The mechanism is not exotic: the engine in this repository keeps exactly that copy for Q5_K, because the GEMV reads blocks but the tiled prefill GEMM needs an fp16 operand (`crates/hipml-model/src/gpu_weights.rs`).

`UD-Q4_K_M` is the near miss: 14.2 GiB of copies on 15.42 GiB of blocks, 29.6 GiB before the cache. A file can pass a size check, load, and fail on the first long prompt, worse than failing at load.

## The sidecar, whose name is also wrong

`MTP/mtp-Qwen3.8-27B-Q4_0.gguf` is 1,369,590,656 bytes, SHA-256 `50d9ce5a6da381bbcfb31061cf73df94a90e6faf8efeddee379a9cb8f1501c6e`, and holds 18 tensors: 8 F32, 4 Q4_K, 4 Q6_K and 2 Q3_K. It contains no Q4_0.

The two Q3_K tensors are `token_embd.weight` and `output.weight`, 1.271B parameters each, 1.02 GiB of its 1.27 GiB. A sidecar drafter never reads them: it borrows the target's embedding and output head, which is what lets 1.4 GB draft for a 16.5 GB target. Skipping them by name before reading, as `load_sidecar` does, means they are never dequantised: the one place where not supporting a format is free.

## What supporting these files requires

- **A complete type table before upload.** Every id needs its block geometry, which turns tensor records into a byte budget.
- **A reference dequantiser per format.** IQ4_XS, IQ4_NL, IQ3_S and Q3_K need hand-computed vectors, not the code under test.
- **Block-form kernels on all three paths.** Decode GEMV, batched GEMV, and a prefill GEMM dequantising into shared memory.
- **A residency policy with an explicit number.** Which formats get an expanded copy, how large it is for this file, and the total before the cache.

## Where the numbers come from

- Inventory, sizes, SHA-256 and every tensor and parameter count: the Hugging Face model API for [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and a header-only parse of each `resolve/main` URL over HTTP range requests, for the four columns and all 25 builds.
- Block sizes and the IQ dequantisers: [`../reference/ggml-quants.c`](../reference/ggml-quants.c). The Q4_K layout, fp16 copy and sidecar skipping: [`../../crates/hipml-core/src/quant.rs`](../../crates/hipml-core/src/quant.rs), [`../../crates/hipml-model/src/gpu_weights.rs`](../../crates/hipml-model/src/gpu_weights.rs) and [`../../crates/hipml-model/src/lib.rs`](../../crates/hipml-model/src/lib.rs).
