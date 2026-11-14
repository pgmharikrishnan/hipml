// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's K-quant / q8_0 / IQ layouts
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//
// One row of an embedding table, dequantised, for every format the engine can
// dequantise.
//
// `token_embd.weight` is a row gather, not a matvec: the token id selects one
// row and nothing else is read. The per-format kernels this replaces
// (`embed_q4_k_row`, `embed_q6_k_row`) existed because the two files that were
// loaded first happened to use those formats, and `GpuEngine::new` rejected
// everything else — which is how a Q8_0 embedding table, the single most common
// choice for a large-vocabulary model, stopped a 22.85 GB file from opening at
// all. A row gather has no shape constraint that could justify a per-type
// allowlist, so this kernel is one entry with a `kind` selector and the host
// asks for a kind by name, not for a kernel.
//
// Geometry: one 256-thread block per `DR_VALUES` (256) values of the row, one
// value per thread. The K-quant super-block is 256 values, so for those one
// block is exactly one super-block. Q8_0 and IQ4_NL block at 32 values, so one
// block spans eight of them; F32 and BF16 are not blocked at all and one block
// spans 1 KiB / 512 B of the row. `dr_unit_bytes` is that mapping, and
// `row_kind` in `graph_launch.rs` mirrors the kind ids.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "iq3s_grid.inc"

// Values one block of this kernel covers.
#define DR_VALUES 256

// Kind ids. Mirrored by `row_kind` in `launch/device/graph_launch.rs`.
#define DR_F32 0
#define DR_BF16 1
#define DR_Q8_0 2
#define DR_Q3K 3
#define DR_Q4K 4
#define DR_Q5K 5
#define DR_Q6K 6
#define DR_IQ4NL 7
#define DR_IQ4XS 8
#define DR_IQ3S 9

// ggml-common.h `kvalues_iq4nl`, the 16-entry signed codebook both IQ4 formats
// index.
static __device__ __constant__ const signed char DR_KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// Bytes of the row one block covers, per kind.
static __device__ __forceinline__ int dr_unit_bytes(int kind)
{
    switch (kind) {
    case DR_F32: return DR_VALUES * 4;
    case DR_BF16: return DR_VALUES * 2;
    case DR_Q8_0: return (DR_VALUES / 32) * 34;
    case DR_Q3K: return 110;
    case DR_Q4K: return 144;
    case DR_Q5K: return 176;
    case DR_Q6K: return 210;
    case DR_IQ4NL: return (DR_VALUES / 32) * 18;
    case DR_IQ4XS: return 136;
    case DR_IQ3S: return 110;
    }
    return 0;
}

// The 6-bit scale and min of sub-block `j` from the 12 header bytes shared by
// Q4_K and Q5_K.
static __device__ __forceinline__ void dr_q4k_scale_min(int j,
                                                        const unsigned char* q,
                                                        int* sc, int* mn)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// The 16 six-bit scales of Q3_K, unpacked exactly as `dequantize_row_q3_K` does
// it: three 32-bit words, then a nibble/2-bit shuffle into four words of four
// signed bytes. Bytewise loads because a 110-byte block leaves `scales` at an
// offset that is not 4-byte aligned.
static __device__ __forceinline__ int dr_q3k_scale(const unsigned char* sc, int i)
{
    unsigned int aux0 = (unsigned int)sc[0] | ((unsigned int)sc[1] << 8)
                      | ((unsigned int)sc[2] << 16) | ((unsigned int)sc[3] << 24);
    unsigned int aux1 = (unsigned int)sc[4] | ((unsigned int)sc[5] << 8)
                      | ((unsigned int)sc[6] << 16) | ((unsigned int)sc[7] << 24);
    const unsigned int tmp = (unsigned int)sc[8] | ((unsigned int)sc[9] << 8)
                           | ((unsigned int)sc[10] << 16) | ((unsigned int)sc[11] << 24);
    const unsigned int KMASK1 = 0x03030303u;
    const unsigned int KMASK2 = 0x0f0f0f0fu;
    unsigned int aux2 = ((aux0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    const unsigned int aux3 = ((aux1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux0 = (aux0 & KMASK2) | ((tmp & KMASK1) << 4);
    aux1 = (aux1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    const unsigned int w = (i < 4) ? aux0 : (i < 8) ? aux1 : (i < 12) ? aux2 : aux3;
    return (int)(signed char)((w >> (8 * (i & 3))) & 0xFFu);
}

// Value `l` of the unit at `bp`, in the same mapping the matching GEMV kernel
// walks.
static __device__ __forceinline__ float dr_value(const unsigned char* __restrict__ bp,
                                                 int kind, int l)
{
    switch (kind) {
    case DR_F32:
        return reinterpret_cast<const float*>(bp)[l];

    case DR_BF16: {
        const unsigned int bits = (unsigned int)reinterpret_cast<const unsigned short*>(bp)[l];
        return __uint_as_float(bits << 16);
    }

    case DR_Q8_0: {
        const unsigned char* b = bp + (size_t)(l >> 5) * 34;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        return d * (float)reinterpret_cast<const signed char*>(b + 2)[l & 31];
    }

    case DR_Q3K: {
        const float d_all = __half2float(*reinterpret_cast<const __half*>(bp + 108));
        const int n = l >> 7;
        const int ph = l & 127;
        const int j = ph >> 5;
        const int within = ph & 31;
        const int half = within >> 4;
        const int li = within & 15;
        const float dl = d_all * (float)(dr_q3k_scale(bp + 96, n * 8 + j * 2 + half) - 32);
        const int qv = (int)((bp[32 + n * 32 + half * 16 + li] >> (2 * j)) & 3);
        const int m = 1 << (n * 4 + j);
        const int sub = (bp[half * 16 + li] & m) ? 0 : 4;
        return dl * (float)(qv - sub);
    }

    case DR_Q4K: {
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(bp + 2));
        const int sub = l >> 5;
        const int within = l & 31;
        int sc, mn;
        dr_q4k_scale_min(sub, bp + 4, &sc, &mn);
        const unsigned char qb = bp[16 + (sub >> 1) * 32 + within];
        const int q = (sub & 1) ? (int)(qb >> 4) : (int)(qb & 0xF);
        return d * (float)sc * (float)q - dmin * (float)mn;
    }

    case DR_Q5K: {
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(bp + 2));
        const int j64 = l >> 6;
        const int w = l & 63;
        const int w32 = w & 31;
        int sc, mn;
        dr_q4k_scale_min(j64 * 2 + (w >> 5), bp + 4, &sc, &mn);
        const unsigned char ql = bp[48 + j64 * 32 + w32];
        const int q = (w < 32) ? (int)(ql & 0xF) : (int)(ql >> 4);
        const int hi = (int)((bp[16 + w32] >> (2 * j64 + (w >> 5))) & 1u);
        return d * (float)sc * (float)(q + 16 * hi) - dmin * (float)mn;
    }

    case DR_Q6K: {
        const int half = l >> 7;
        const int ph = l & 127;
        const int group = ph >> 5;
        const int li = ph & 31;
        const int sub = group * 2 + (li >> 4);
        const float d = __half2float(*reinterpret_cast<const __half*>(bp + 208));
        const float scale =
            d * (float)reinterpret_cast<const signed char*>(bp + 192)[half * 8 + sub];
        const unsigned char qb = bp[half * 64 + ((group & 1) ? 32 : 0) + li];
        const unsigned char hb = bp[128 + half * 32 + li];
        const int nib = (group >= 2) ? (int)(qb >> 4) : (int)(qb & 0xF);
        const int hi = (int)((hb >> (2 * group)) & 3u);
        return scale * (float)((nib | (hi << 4)) - 32);
    }

    case DR_IQ4NL: {
        const unsigned char* b = bp + (size_t)(l >> 5) * 18;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const int j = l & 31;
        const int idx = (j < 16) ? (int)(b[2 + j] & 0xF) : (int)(b[2 + j - 16] >> 4);
        return d * (float)DR_KVALUES_IQ4NL[idx];
    }

    case DR_IQ4XS: {
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const unsigned int scales_h =
            (unsigned int)reinterpret_cast<const unsigned short*>(bp + 2)[0];
        const int ib = l >> 5;
        const int within = l & 31;
        const int ls = (int)((bp[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xF)
                     | ((int)((scales_h >> (2 * ib)) & 3u) << 4);
        const unsigned char q = bp[8 + ib * 16 + (within & 15)];
        const int idx = (within < 16) ? (int)(q & 0xF) : (int)(q >> 4);
        return d * (float)(ls - 32) * (float)DR_KVALUES_IQ4NL[idx];
    }

    case DR_IQ3S: {
        const float d = __half2float(*reinterpret_cast<const __half*>(bp));
        const int sb = l >> 5;
        const int pair = sb >> 1;
        const int half = sb & 1;
        const unsigned char sc = bp[106 + pair];
        const int nib = half ? (int)(sc >> 4) : (int)(sc & 0xF);
        const float db = d * (float)(1 + 2 * nib);
        const int within = l & 31;
        const int li = within >> 3;
        const int jj = within & 7;
        const unsigned int hi = (unsigned int)bp[66 + sb];
        const int qo = sb * 8;
        const int so = sb * 4;
        const int j = (jj < 4) ? jj : (jj - 4);
        const int idx = (jj < 4)
            ? (int)((unsigned int)bp[2 + qo + 2 * li] | ((hi << (8 - 2 * li)) & 256u))
            : (int)((unsigned int)bp[2 + qo + 2 * li + 1] | ((hi << (7 - 2 * li)) & 256u));
        const int byte = (int)((IQ3S_GRID[idx] >> (8 * j)) & 0xFFu);
        // The sign bit is `jj` in both halves: the host's second group uses
        // bit `j + 4`, and its `j` is `jj - 4`.
        const float sign = (bp[74 + so + li] & (1u << jj)) ? -1.0f : 1.0f;
        return db * (float)byte * sign;
    }
    }
    return 0.0f;
}

// Embedding row for a graph that is replayed: the row index lives in device
// memory (`step[2]`), so the captured launch needs no host round-trip.
extern "C" __global__ void embed_row_any(
    const unsigned char* __restrict__ w,
    float* __restrict__ out,
    int nblocks,
    int row_bytes,
    const int* __restrict__ step,
    int kind)
{
    const int b = blockIdx.x;
    const int l = threadIdx.x;
    if (b >= nblocks || l >= DR_VALUES) return;

    const unsigned char* bp = w + (size_t)step[2] * (size_t)row_bytes
                            + (size_t)b * (size_t)dr_unit_bytes(kind);
    out[(size_t)b * DR_VALUES + l] = dr_value(bp, kind, l);
}

// The same decode with the row address on the host: the loader's
// `embedding_row` probe and the chunked prefill's row-at-a-time embedding.
extern "C" __global__ void dequant_row_any(
    const unsigned char* __restrict__ w,
    float* __restrict__ out,
    int nblocks,
    int kind)
{
    const int b = blockIdx.x;
    const int l = threadIdx.x;
    if (b >= nblocks || l >= DR_VALUES) return;

    const unsigned char* bp = w + (size_t)b * (size_t)dr_unit_bytes(kind);
    out[(size_t)b * DR_VALUES + l] = dr_value(bp, kind, l);
}
