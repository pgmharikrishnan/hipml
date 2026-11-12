// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's IQ3_S layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// y[row] = dequant_iq3_s(W[row,:]) . x
//
// Same row-tile GEMV as `gemv_q4_k`: one block owns IQ3S_ROW_TILE output rows
// and stages the x k-tile in shared once, so a value read from shared feeds
// IQ3S_ROWS_PER_WARP weight rows before it is discarded.
//
// Each lane owns IQ3S_VPL (16) consecutive columns, i.e. half of one 32-value
// sub-block: two 8-value grid groups, one shared scale nibble. The 9-bit grid
// index is the low 8 bits from `qs` plus one `qh` bit, and `signs` carries one
// sign bit per value.
//
// IQ3_S super-block (110 bytes, 256 values):
//   d(2) qs(64) qh(8) signs(32) scales(4)
// Eight 32-value sub-blocks in four (low nibble, high nibble) scale pairs;
// db = d * (1 + 2*nibble), so every scale is positive.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define IQ3S_VALUES 256
#define IQ3S_BLOCK_BYTES 110
#define IQ3S_BLOCK 256
#define IQ3S_WARPS (IQ3S_BLOCK / 32)
#define IQ3S_ROWS_PER_WARP 2
#define IQ3S_ROW_TILE (IQ3S_WARPS * IQ3S_ROWS_PER_WARP)
#define IQ3S_VPL 16
#define IQ3S_GROUP (IQ3S_VPL * 32)
#define IQ3S_K_TILE 2048

// Mirrors the manifest's KernelKind::RowTile { rows = IQ3S_ROW_TILE, block =
// IQ3S_BLOCK }; change both together.

// ggml-common.h `iq3s_grid`, four packed codebook values per index.
static __device__ __constant__ const unsigned int IQ3S_GRID[512] = {
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

// One scale nibble pair as `1 + 2*nibble` (never negative).
static __device__ __forceinline__ float iq3s_scale(const unsigned char* __restrict__ b,
                                                   int pair, int half)
{
    const unsigned int sc = b[106 + pair];
    const int nib = half ? (int)(sc >> 4) : (int)(sc & 0xF);
    return (float)(1 + 2 * nib);
}

static __device__ __forceinline__ void iq3s_stage_group(const float* __restrict__ x,
                                                        float* __restrict__ sx,
                                                        int base, int j)
{
    const int g = j >> 5; // 32 chunks of 16 values per 512-value group
    const int lane = j & 31;
    float* s = sx + g * IQ3S_GROUP + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + IQ3S_VPL * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

// Dequantise this lane's 16 values of one row into `wv`.
static __device__ __forceinline__ void iq3s_load(const unsigned char* __restrict__ rp,
                                                 int p, float* wv)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int pair = within >> 6;
    const int half = (within >> 5) & 1;
    const int l0 = (within & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * IQ3S_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float db = d * iq3s_scale(b, pair, half);
    const unsigned int qh = b[66 + pair * 2 + half];
    const unsigned char* qp = b + 2 + pair * 16 + half * 8;
    const unsigned char* sp = b + 74 + pair * 8 + half * 4;

#pragma unroll
    for (int ll = 0; ll < 2; ++ll) {
        const int l = l0 + ll;
        const int i1 = (int)qp[2 * l] | (int)((qh << (8 - 2 * l)) & 256u);
        const int i2 = (int)qp[2 * l + 1] | (int)((qh << (7 - 2 * l)) & 256u);
        const unsigned int g1 = IQ3S_GRID[i1];
        const unsigned int g2 = IQ3S_GRID[i2];
        const unsigned int sign = sp[l];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const float s1 = (sign & (1u << t)) ? -1.0f : 1.0f;
            const float s2 = (sign & (1u << (t + 4))) ? -1.0f : 1.0f;
            wv[ll * 8 + t] = db * (float)((g1 >> (8 * t)) & 0xFFu) * s1;
            wv[ll * 8 + 4 + t] = db * (float)((g2 >> (8 * t)) & 0xFFu) * s2;
        }
    }
}

// Sixteen-value dot in the order the single-row kernel accumulates: the two
// grid groups of one `l` interleave (g1[0], g2[0], g1[1], ...), not 0..15.
// Both the single-row and the batched kernel go through this, so row b of the
// batched result is the same 32 bits as a one-column decode.
static __device__ __forceinline__ float iq3s_dot16(const float* wv, const float* xv, float a)
{
#pragma unroll
    for (int ll = 0; ll < 2; ++ll)
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            a = fmaf(wv[ll * 8 + t], xv[ll * 8 + t], a);
            a = fmaf(wv[ll * 8 + 4 + t], xv[ll * 8 + 4 + t], a);
        }
    return a;
}

// Sixteen-column contribution of one IQ3_S row to `acc`.
static __device__ __forceinline__ void iq3s_acc(const unsigned char* __restrict__ rp,
                                                int p, const float* xv, float* acc)
{
    float wv[IQ3S_VPL];
    iq3s_load(rp, p, wv);
    *acc = iq3s_dot16(wv, xv, *acc);
}

// One row/warp when RPW is 1, two when it is IQ3S_ROWS_PER_WARP.
template <int RPW>
static __device__ __forceinline__ void iq3s_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / IQ3S_VALUES;
    const int row_tile = IQ3S_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ3S_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += IQ3S_K_TILE) {
        const int n = min(IQ3S_K_TILE, k - base);
        const int nchunks = n / IQ3S_VPL;
        for (int j = tid; j < nchunks; j += IQ3S_BLOCK) iq3s_stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * IQ3S_VPL; c < n; c += IQ3S_GROUP) {
            const int g = c / IQ3S_GROUP;
            const int s = g * IQ3S_GROUP + lane;
            float xv[IQ3S_VPL];
#pragma unroll
            for (int comp = 0; comp < IQ3S_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq3s_acc(rp[rr], p, xv, &acc[rr]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        float a = acc[rr];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0 && row[rr] < m) y[row[rr]] = a;
    }
}

extern "C" __global__ void gemv_iq3_s(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[IQ3S_K_TILE];
    iq3s_tile<IQ3S_ROWS_PER_WARP>(w, x, y, m, k, sx, blockIdx.x);
}

// ---------------------------------------------------------------------------
// Batched GEMV: B token rows share one pass over the weight. Same reuse and
// same per-lane k order as `gemv_iq3_s`, so row b is bit-identical to a
// single-sequence launch on x[b].
// ---------------------------------------------------------------------------

#define IQ3SB_K_TILE_BIG 1024
#define IQ3SB_K_TILE_SMALL 512
#define IQ3SB_ROWS_PER_WARP 2

template <int B, int RPW, int TILE>
static __device__ __forceinline__ void iq3s_tile_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / IQ3S_VALUES;
    const int row_tile = IQ3S_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * IQ3S_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += TILE) {
        const int n = min(TILE, k - base);
        const int nchunks = n / IQ3S_VPL;
        for (int j = tid; j < nchunks; j += IQ3S_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                iq3s_stage_group(x + (size_t)bi * k, sx + bi * TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * IQ3S_VPL; c < n; c += IQ3S_GROUP) {
            const int g = c / IQ3S_GROUP;
            const int s = g * IQ3S_GROUP + lane;
            const int p = base + c;
            float wv[RPW][IQ3S_VPL];
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) iq3s_load(rp[rr], p, wv[rr]);
#pragma unroll
            for (int bi = 0; bi < B; ++bi) {
                const float* src = sx + bi * TILE + s;
                float xv[IQ3S_VPL];
#pragma unroll
                for (int i = 0; i < IQ3S_VPL; ++i) xv[i] = src[i * 32];
#pragma unroll
                for (int rr = 0; rr < RPW; ++rr)
                    acc[rr][bi] = iq3s_dot16(wv[rr], xv, acc[rr][bi]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
#pragma unroll
        for (int bi = 0; bi < B; ++bi) {
            float a = acc[rr][bi];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
            if (lane == 0 && row[rr] < m) y[(size_t)bi * m + row[rr]] = a;
        }
    }
}

template <int B, int TILE>
static __device__ __forceinline__ void iq3s_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    iq3s_tile_b<B, IQ3SB_ROWS_PER_WARP, TILE>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_iq3_s_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    if (batch <= 4) {
        switch (batch) {
        case 1: iq3s_entry_b<1, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        case 2: iq3s_entry_b<2, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        case 3: iq3s_entry_b<3, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        case 4: iq3s_entry_b<4, IQ3SB_K_TILE_BIG>(w, x, y, m, k); break;
        }
    } else {
        switch (batch) {
        case 5: iq3s_entry_b<5, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 6: iq3s_entry_b<6, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        case 7: iq3s_entry_b<7, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        default: iq3s_entry_b<8, IQ3SB_K_TILE_SMALL>(w, x, y, m, k); break;
        }
    }
}
