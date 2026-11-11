// SPDX-License-Identifier: Apache-2.0
// Block format decoding follows ggml's K-quant / q8_0 layout
// (https://github.com/ggml-org/llama.cpp, tag b12126),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// Dequantise Q6_K super-blocks to f32.
//
// Q6_K super-block (210 bytes, 256 values):
//   ql(128) qh(64) scales(16) d(2)
//   y = d*scale*(q-32),  q = (ql nibble) | ((qh plane) << 4)
//
// One 256-thread block per super-block, one output value per thread. The
// value's 128-value half picks the ql/qh/scale bank, its 32-value group picks
// the nibble and the qh plane, and its position inside the group picks the
// byte and the scale sub-block -- the same mapping `gemv_q6_k` walks.
//
// This is the embedding-row path (Unsloth UD dynamic quants store
// `token_embd.weight` as Q6_K), so the loader launches it over one row's
// super-blocks only. It is not on the decode hot path; the GEMV kernels
// dequantise inline.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define DQ6K_VALUES 256
#define DQ6K_BLOCK_BYTES 210

static __device__ __forceinline__ float dq6k_value(const unsigned char* __restrict__ bp,
                                                   int i)
{
    const int half = i >> 7;
    const int ph = i & 127;
    const int group = ph >> 5;
    const int l = ph & 31;
    const int sub = group * 2 + (l >> 4);

    const float d = __half2float(*reinterpret_cast<const __half*>(bp + 208));
    const float scale =
        d * (float)reinterpret_cast<const signed char*>(bp + 192)[half * 8 + sub];

    const unsigned char qb = bp[half * 64 + ((group & 1) ? 32 : 0) + l];
    const unsigned char hb = bp[128 + half * 32 + l];
    const int nib = (group >= 2) ? (int)(qb >> 4) : (int)(qb & 0xF);
    const int hi = (int)((hb >> (2 * group)) & 3u);
    return scale * (float)((nib | (hi << 4)) - 32);
}

// Embedding row for a graph that is replayed: the row index lives in device
// memory (`step[2]`), so the captured launch needs no host round-trip.
extern "C" __global__ void embed_q6_k_row(
    const unsigned char* __restrict__ w,
    float* __restrict__ out,
    int nblocks,
    int row_bytes,
    const int* __restrict__ step)
{
    const int b = blockIdx.x;
    const int l = threadIdx.x;
    if (b >= nblocks || l >= DQ6K_VALUES) return;

    const unsigned char* bp = w + (size_t)step[2] * (size_t)row_bytes
                            + (size_t)b * DQ6K_BLOCK_BYTES;
    out[(size_t)b * DQ6K_VALUES + l] = dq6k_value(bp, l);
}

extern "C" __global__ void dequant_q6_k_f32(
    const unsigned char* __restrict__ w,
    float* __restrict__ out,
    int nblocks)
{
    const int b = blockIdx.x;
    const int l = threadIdx.x;
    if (b >= nblocks || l >= DQ6K_VALUES) return;

    out[(size_t)b * DQ6K_VALUES + l] = dq6k_value(w + (size_t)b * DQ6K_BLOCK_BYTES, l);
}
