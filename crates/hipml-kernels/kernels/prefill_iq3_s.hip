// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's IQ3_S layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Tiled prefill GEMM over IQ3_S weights.
//
// y[t, row] = W[row, :] . x[t, :] for `tokens` rows of x at once. Same geometry
// and shared staging as `prefill_q4_k`: a weight tile is dequantised ONCE into
// shared and reused for every prompt token in the block.
//
//   PFI3_TPB 256 tokens per block, PFI3_RPB 64 output rows, PFI3_KT 32 k/tile.
//   256 threads: 32 token groups x 8 row groups, 8x8 accumulators per thread.
//
// Each thread dequantises 8 contiguous values: the four values of one grid
// lookup and the four of the next, both under one sign byte and one scale
// nibble. The 512-entry codebook is in constant memory.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PFI3_BLOCK_BYTES 110
#define PFI3_TPB 256
#define PFI3_RPB 64
#define PFI3_KT 32
#define PFI3_TT 8
#define PFI3_RR 8
#define PFI3_TG (PFI3_TPB / PFI3_TT)
#define PFI3_RG (PFI3_RPB / PFI3_RR)
#define PFI3_THREADS (PFI3_TG * PFI3_RG)
#define PFI3_XSTRIDE (PFI3_KT + 1)
#define PFI3_WSTRIDE (PFI3_KT + 1)
#define PFI3_HALF_TOKENS (PFI3_TPB / 2)

// ggml-common.h `iq3s_grid`, four packed codebook values per index.
static __device__ __constant__ const unsigned int PFI3_GRID[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010Bu, 0x0101010Fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030Du, 0x01010501u, 0x01010503u, 0x0101050Bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090Bu, 0x0101090Fu, 0x01010B03u, 0x01010B07u, 0x01010D01u, 0x01010D05u, 0x01010F03u, 0x01010F09u,
    0x01010F0Fu, 0x01030101u, 0x01030103u, 0x01030105u, 0x01030109u, 0x01030301u, 0x01030303u, 0x0103030Bu,
    0x01030501u, 0x01030507u, 0x0103050Fu, 0x01030703u, 0x0103070Bu, 0x01030909u, 0x01030D03u, 0x01030D0Bu,
    0x01030F05u, 0x01050101u, 0x01050103u, 0x0105010Bu, 0x0105010Fu, 0x01050301u, 0x01050307u, 0x0105030Du,
    0x01050503u, 0x0105050Bu, 0x01050701u, 0x01050709u, 0x01050905u, 0x0105090Bu, 0x0105090Fu, 0x01050B03u,
    0x01050B07u, 0x01050F01u, 0x01050F07u, 0x01070107u, 0x01070303u, 0x0107030Bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070Du, 0x01070909u, 0x01070B01u, 0x01070B05u, 0x01070D0Fu, 0x01070F03u,
    0x01070F0Bu, 0x01090101u, 0x01090307u, 0x0109030Fu, 0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090B03u, 0x01090F01u, 0x010B0105u, 0x010B0109u, 0x010B0501u, 0x010B0505u, 0x010B050Du,
    0x010B0707u, 0x010B0903u, 0x010B090Bu, 0x010B090Fu, 0x010B0D0Du, 0x010B0F07u, 0x010D010Du, 0x010D0303u,
    0x010D0307u, 0x010D0703u, 0x010D0B05u, 0x010D0F03u, 0x010F0101u, 0x010F0105u, 0x010F0109u, 0x010F0501u,
    0x010F0505u, 0x010F050Du, 0x010F0707u, 0x010F0B01u, 0x010F0B09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u, 0x0301030Bu, 0x0301030Fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070Du, 0x03010B09u, 0x03010B0Du, 0x03010D03u, 0x03010F05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010Du, 0x03030301u, 0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030B01u, 0x03030B05u, 0x03030F01u, 0x03030F0Du, 0x03050101u, 0x03050305u, 0x0305030Bu,
    0x0305030Fu, 0x03050501u, 0x03050509u, 0x03050705u, 0x03050901u, 0x03050907u, 0x03050B0Bu, 0x03050D01u,
    0x03050F05u, 0x03070103u, 0x03070109u, 0x0307010Fu, 0x03070301u, 0x03070307u, 0x03070503u, 0x0307050Fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070D05u, 0x03070F01u, 0x03090107u, 0x0309010Bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u, 0x0309090Du, 0x03090B01u, 0x03090B09u, 0x030B0103u,
    0x030B0301u, 0x030B0307u, 0x030B0503u, 0x030B0701u, 0x030B0705u, 0x030B0B03u, 0x030D0501u, 0x030D0509u,
    0x030D050Fu, 0x030D0909u, 0x030D090Du, 0x030F0103u, 0x030F0107u, 0x030F0301u, 0x030F0305u, 0x030F0503u,
    0x030F070Bu, 0x030F0903u, 0x030F0D05u, 0x030F0F01u, 0x05010101u, 0x05010103u, 0x05010107u, 0x0501010Bu,
    0x0501010Fu, 0x05010301u, 0x05010305u, 0x05010309u, 0x0501030Du, 0x05010503u, 0x05010507u, 0x0501050Fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u, 0x0501090Bu, 0x05010B01u, 0x05010B05u, 0x05010D0Fu,
    0x05010F01u, 0x05010F07u, 0x05010F0Bu, 0x05030101u, 0x05030105u, 0x05030301u, 0x05030307u, 0x0503030Fu,
    0x05030505u, 0x0503050Bu, 0x05030703u, 0x05030709u, 0x05030905u, 0x05030B03u, 0x05050103u, 0x05050109u,
    0x0505010Fu, 0x05050503u, 0x05050507u, 0x05050701u, 0x0505070Fu, 0x05050903u, 0x05050B07u, 0x05050B0Fu,
    0x05050F03u, 0x05050F09u, 0x05070101u, 0x05070105u, 0x0507010Bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070B01u, 0x05070D0Du, 0x05090103u, 0x0509010Fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070Bu, 0x05090903u, 0x05090F05u, 0x05090F0Bu, 0x050B0109u, 0x050B0303u,
    0x050B0505u, 0x050B070Fu, 0x050B0901u, 0x050B0B07u, 0x050B0F01u, 0x050D0101u, 0x050D0105u, 0x050D010Fu,
    0x050D0503u, 0x050D0B0Bu, 0x050D0D03u, 0x050F010Bu, 0x050F0303u, 0x050F050Du, 0x050F0701u, 0x050F0907u,
    0x050F0B01u, 0x07010105u, 0x07010303u, 0x07010307u, 0x0701030Bu, 0x0701030Fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070Bu, 0x07010905u, 0x07010909u, 0x0701090Fu, 0x07010B03u, 0x07010D07u, 0x07010F03u,
    0x07030103u, 0x07030107u, 0x0703010Bu, 0x07030309u, 0x07030503u, 0x07030507u, 0x07030901u, 0x07030D01u,
    0x07030F05u, 0x07030F0Du, 0x07050101u, 0x07050305u, 0x07050501u, 0x07050705u, 0x07050709u, 0x07050B01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u, 0x07070507u, 0x0707050Fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090Fu, 0x07070B0Bu, 0x07070F07u, 0x07090107u, 0x07090303u, 0x0709030Du, 0x07090505u,
    0x07090703u, 0x07090B05u, 0x07090D01u, 0x07090D09u, 0x070B0103u, 0x070B0301u, 0x070B0305u, 0x070B050Bu,
    0x070B0705u, 0x070B0909u, 0x070B0B0Du, 0x070B0F07u, 0x070D030Du, 0x070D0903u, 0x070F0103u, 0x070F0107u,
    0x070F0501u, 0x070F0505u, 0x070F070Bu, 0x09010101u, 0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050Fu, 0x09010705u, 0x09010903u, 0x09010B01u, 0x09010F01u, 0x09030105u, 0x0903010Fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070Bu, 0x09030907u, 0x09030B03u, 0x09030B0Bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030Bu, 0x09050503u, 0x09050707u, 0x09050901u, 0x09050B0Fu, 0x09050D05u,
    0x09050F01u, 0x09070109u, 0x09070303u, 0x09070307u, 0x09070501u, 0x09070505u, 0x09070703u, 0x0907070Bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070Fu, 0x09090901u, 0x09090F03u, 0x090B010Bu, 0x090B010Fu,
    0x090B0503u, 0x090B0D05u, 0x090D0307u, 0x090D0709u, 0x090D0D01u, 0x090F0301u, 0x090F030Bu, 0x090F0701u,
    0x090F0907u, 0x090F0B03u, 0x0B010105u, 0x0B010301u, 0x0B010309u, 0x0B010505u, 0x0B010901u, 0x0B010909u,
    0x0B01090Fu, 0x0B010B05u, 0x0B010D0Du, 0x0B010F09u, 0x0B030103u, 0x0B030107u, 0x0B03010Bu, 0x0B030305u,
    0x0B030503u, 0x0B030705u, 0x0B030F05u, 0x0B050101u, 0x0B050303u, 0x0B050507u, 0x0B050701u, 0x0B05070Du,
    0x0B050B07u, 0x0B070105u, 0x0B07010Fu, 0x0B070301u, 0x0B07050Fu, 0x0B070909u, 0x0B070B03u, 0x0B070D0Bu,
    0x0B070F07u, 0x0B090103u, 0x0B090109u, 0x0B090501u, 0x0B090705u, 0x0B09090Du, 0x0B0B0305u, 0x0B0B050Du,
    0x0B0B0B03u, 0x0B0B0B07u, 0x0B0D0905u, 0x0B0F0105u, 0x0B0F0109u, 0x0B0F0505u, 0x0D010303u, 0x0D010307u,
    0x0D01030Bu, 0x0D010703u, 0x0D010707u, 0x0D010D01u, 0x0D030101u, 0x0D030501u, 0x0D03050Fu, 0x0D030D09u,
    0x0D050305u, 0x0D050709u, 0x0D050905u, 0x0D050B0Bu, 0x0D050D05u, 0x0D050F01u, 0x0D070101u, 0x0D070309u,
    0x0D070503u, 0x0D070901u, 0x0D09050Bu, 0x0D090907u, 0x0D090D05u, 0x0D0B0101u, 0x0D0B0107u, 0x0D0B0709u,
    0x0D0B0D01u, 0x0D0D010Bu, 0x0D0D0901u, 0x0D0F0303u, 0x0D0F0307u, 0x0F010101u, 0x0F010109u, 0x0F01010Fu,
    0x0F010501u, 0x0F010505u, 0x0F01070Du, 0x0F010901u, 0x0F010B09u, 0x0F010D05u, 0x0F030105u, 0x0F030303u,
    0x0F030509u, 0x0F030907u, 0x0F03090Bu, 0x0F050103u, 0x0F050109u, 0x0F050301u, 0x0F05030Du, 0x0F050503u,
    0x0F050701u, 0x0F050B03u, 0x0F070105u, 0x0F070705u, 0x0F07070Bu, 0x0F070B07u, 0x0F090103u, 0x0F09010Bu,
    0x0F090307u, 0x0F090501u, 0x0F090B01u, 0x0F0B0505u, 0x0F0B0905u, 0x0F0D0105u, 0x0F0D0703u, 0x0F0F0101u,
};

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8) into `out`.
static __device__ __forceinline__ void pfi3_dequant8(const unsigned char* __restrict__ rp,
                                                     int v, float* out)
{
    const int blk = v >> 8;
    const int within = v & 255;
    const int pair = within >> 6;
    const int half = (within >> 5) & 1;
    const int l = (within & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * PFI3_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const unsigned int sc = b[106 + pair];
    const int nib = half ? (int)(sc >> 4) : (int)(sc & 0xF);
    const float db = d * (float)(1 + 2 * nib);

    const unsigned int qh = b[66 + pair * 2 + half];
    const unsigned char* qp = b + 2 + pair * 16 + half * 8;
    const int i1 = (int)qp[2 * l] | (int)((qh << (8 - 2 * l)) & 256u);
    const int i2 = (int)qp[2 * l + 1] | (int)((qh << (7 - 2 * l)) & 256u);
    const unsigned int g1 = PFI3_GRID[i1];
    const unsigned int g2 = PFI3_GRID[i2];
    const unsigned int sign = b[74 + pair * 8 + half * 4 + l];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const float s1 = (sign & (1u << t)) ? -1.0f : 1.0f;
        const float s2 = (sign & (1u << (t + 4))) ? -1.0f : 1.0f;
        out[t] = db * (float)((g1 >> (8 * t)) & 0xFFu) * s1;
        out[4 + t] = db * (float)((g2 >> (8 * t)) & 0xFFu) * s2;
    }
}

extern "C" __global__ void prefill_iq3_s(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pfi3_x[PFI3_TPB * PFI3_XSTRIDE];
    __shared__ __align__(16) float pfi3_w[PFI3_RPB * PFI3_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PFI3_TG;   // token group: tokens tg + 32*tt
    const int rg = tid / PFI3_TG;   // row group: rows rg*8 .. rg*8+7
    const int tok_base = blockIdx.y * PFI3_TPB;
    const int row_base = blockIdx.x * PFI3_RPB;
    const int nblocks = k / 256;

    const int d_row = tid >> 2;    // 64 rows
    const int d_q = tid & 3;       // 4 chunks of 8 values per 32-value k-tile

    float acc[PFI3_TT][PFI3_RR];
#pragma unroll
    for (int tt = 0; tt < PFI3_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PFI3_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PFI3_TPB * (PFI3_KT / 4) / PFI3_THREADS; // 8 float4 per thread

    for (int base = 0; base < k; base += PFI3_KT) {
        __syncthreads(); // previous multiply finished before shared is reused

        // Weight tile: 64 rows x 32 values, dequantised once.
        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PFI3_BLOCK_BYTES;
                float v[8];
                pfi3_dequant8(rp, base + d_q * 8, v);
                float* dst = pfi3_w + d_row * PFI3_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        // x tile: 256 tokens x 32 values, token-major.
        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PFI3_THREADS;
            const int tl = i >> 3;         // token within the block
            const int c4 = (i & 7) * 4;    // k offset, multiple of 4
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pfi3_x + tl * PFI3_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PFI3_XSTRIDE;
        const int w0 = rg * PFI3_RR * PFI3_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PFI3_KT; ++kk) {
            float xv[PFI3_TT];
            float wv[PFI3_RR];
#pragma unroll
            for (int tt = 0; tt < PFI3_TT; ++tt)
                xv[tt] = pfi3_x[x0 + tt * PFI3_TG * PFI3_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PFI3_RR; ++rr)
                wv[rr] = pfi3_w[w0 + rr * PFI3_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PFI3_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PFI3_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    // Epilogue in two 128-token halves so the x buffer can hold the transpose
    // of the 8 KB output half; global stores are then coalesced along `row`.
    float* sbuf = pfi3_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PFI3_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PFI3_RR; ++rr)
                sbuf[(tg + PFI3_TG * u) * PFI3_RPB + (rg * PFI3_RR + rr)] =
                    acc[half * (PFI3_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PFI3_HALF_TOKENS * PFI3_RPB; i += PFI3_THREADS) {
            const int tl = i / PFI3_RPB;
            const int rl = i - tl * PFI3_RPB;
            const int gt = tok_base + half * PFI3_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
