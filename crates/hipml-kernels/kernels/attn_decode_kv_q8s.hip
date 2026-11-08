// q8 decode attention, two alternatives to the f32 staging in
// `attn_decode_kv.hip`, built to find what the 12 ms/step at 100K is bound by.
//
//   attn_decode_kv_q8s   stage the *quantized* 272 B rows in shared (TILE = 16
//                        rows, 8.7 KB), dequantize inside each query-head warp.
//                        Shared traffic per row drops from 12 KB (6 warps x 2
//                        arrays x 1 KB f32) to 2.3 KB, and the tile is 4x
//                        taller, so there are 4x fewer barriers and barriers
//                        are the thing a 6-warp block waits on.
//
//   attn_decode_kv_q8g   no shared at all: each warp reads and dequantizes its
//                        own copy of the row from global. The per-split working
//                        set is ~100 KB, so the six GQA reads of a row should
//                        hit L2 and the kernel pays no staging pass and no
//                        __syncthreads. This is the control that says whether
//                        the f32 staging pass is the cost at all.
//
// Both keep the f32-staging kernel's ABI and numerics: scores in log2 units,
// (max, sum) partials consumed by `attn_decode_kv_reduce`, which is reused.
//
// Requires head_dim % 128 == 0 so a q8 row is a whole number of 16 B copies
// and a lane's eight components are two four-value groups (head_dim 256 here).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q8S_BLOCK 34
#define Q8S_LOG2E 1.4426950408889634f
#define Q8S_TILE 16

// One lane's four values from one q8 block. lane/8 selects the block that
// holds this group, (lane&7)*4 the byte offset inside its 32-value payload.
static __device__ __forceinline__ void q8s_ld4(
    float* o, const unsigned char* row, int c, int lane)
{
    const unsigned char* b = row + (size_t)(c * 4 + (lane >> 3)) * Q8S_BLOCK;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const unsigned w = *reinterpret_cast<const unsigned*>(b + 2 + ((lane & 7) << 2));
    o[0] = d * (float)((int)(w << 24) >> 24);
    o[1] = d * (float)((int)(w << 16) >> 24);
    o[2] = d * (float)((int)(w << 8) >> 24);
    o[3] = d * (float)((int)w >> 24);
}

// A lane's DPL components of one row: groups c*128 + lane*4, four values each.
static __device__ __forceinline__ void q8s_row(
    float* o, const unsigned char* row, int lane, int ngrp)
{
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < ngrp) q8s_ld4(o + c * 4, row, c, lane);
    }
}

// ---- quantized tile in shared ---------------------------------------------
extern "C" __global__ void attn_decode_kv_q8s(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    int seq,
    float scale,
    int splits)
{
    extern __shared__ __align__(16) unsigned char sh[];
    const int group = n_heads / n_kv;
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= group) return;
    const int tid = threadIdx.x;

    const int h = kvh * group + warp;
    const int ngrp = head_dim >> 7;              // four-value groups per lane
    const int nb = head_dim >> 5;                // q8 blocks per row
    const int row_bytes = nb * Q8S_BLOCK;
    const int per = (seq + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;
    const float sl2 = scale * Q8S_LOG2E;

    float qv[8];
    float acc[8];
    // q is f32, not q8: read it directly.
    {
        const float* qp = q + (size_t)h * head_dim;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < ngrp) {
                const float4 a = *reinterpret_cast<const float4*>(qp + c * 128 + lane * 4);
                qv[c * 4 + 0] = a.x;
                qv[c * 4 + 1] = a.y;
                qv[c * 4 + 2] = a.z;
                qv[c * 4 + 3] = a.w;
            }
        }
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[d] = 0.0f;
    }

    int bidx[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) bidx[j] = ((lane ^ (1 << j)) << 2);

    unsigned char* ks = sh;
    unsigned char* vs = sh + Q8S_TILE * row_bytes;
    const unsigned char* kbase = (const unsigned char*)k + (size_t)kvh * seq * row_bytes;
    const unsigned char* vbase = (const unsigned char*)v + (size_t)kvh * seq * row_bytes;
    const int chunks = row_bytes >> 4;           // 16 B copies per row

    float m = -INFINITY;
    float l = 0.0f;

    for (int pos = start; pos < end; pos += Q8S_TILE) {
        const int rows = (pos + Q8S_TILE <= end) ? Q8S_TILE : (end - pos);
        __syncthreads();
        for (int i = tid; i < 2 * Q8S_TILE * chunks; i += blockDim.x) {
            const int arr = i / (Q8S_TILE * chunks);
            const int rem = i - arr * Q8S_TILE * chunks;
            const int r = rem / chunks;
            const int c = rem - r * chunks;
            const int rr = (r < rows) ? r : (rows - 1);
            const unsigned char* src =
                (arr == 0 ? kbase : vbase) + (size_t)(pos + rr) * row_bytes + c * 16;
            unsigned char* dst = (arr == 0 ? ks : vs) + (size_t)r * row_bytes + c * 16;
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        }
        __syncthreads();

        float s[Q8S_TILE];
#pragma unroll
        for (int t = 0; t < Q8S_TILE; ++t) {
            float kd[8];
            q8s_row(kd, ks + (size_t)t * row_bytes, lane, ngrp);
            float dot = 0.0f;
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                if (d < ngrp * 4) dot = fmaf(qv[d], kd[d], dot);
            }
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                dot += __int_as_float(
                    __builtin_amdgcn_ds_bpermute(bidx[j], __float_as_int(dot)));
            }
            s[t] = (t < rows) ? dot * sl2 : -INFINITY;
        }

        float mt = m;
#pragma unroll
        for (int t = 0; t < Q8S_TILE; ++t) mt = fmaxf(mt, s[t]);
        const float corr = exp2f(m - mt);
        l *= corr;
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[d] *= corr;
#pragma unroll
        for (int t = 0; t < Q8S_TILE; ++t) {
            const float p = exp2f(s[t] - mt);
            l += p;
            float vd[8];
            q8s_row(vd, vs + (size_t)t * row_bytes, lane, ngrp);
#pragma unroll
            for (int d = 0; d < 8; ++d) {
                if (d < ngrp * 4) acc[d] = fmaf(p, vd[d], acc[d]);
            }
        }
        m = mt;
    }

    {
        float* dst = part_acc + ((size_t)h * splits + split) * head_dim;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < ngrp) {
                float4 a;
                a.x = acc[c * 4 + 0];
                a.y = acc[c * 4 + 1];
                a.z = acc[c * 4 + 2];
                a.w = acc[c * 4 + 3];
                *reinterpret_cast<float4*>(dst + c * 128 + lane * 4) = a;
            }
        }
        if (lane == 0) {
            float* ml = part_ml + ((size_t)h * splits + split) * 2;
            ml[0] = m;
            ml[1] = l;
        }
    }
}

// ---- no staging: each warp reads its own copy from global -----------------
extern "C" __global__ void attn_decode_kv_q8g(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    int seq,
    float scale,
    int splits)
{
    const int group = n_heads / n_kv;
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= group) return;

    const int h = kvh * group + warp;
    const int ngrp = head_dim >> 7;
    const int nb = head_dim >> 5;
    const int row_bytes = nb * Q8S_BLOCK;
    const int per = (seq + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;
    const float sl2 = scale * Q8S_LOG2E;

    float qv[8];
    float acc[8];
    {
        const float* qp = q + (size_t)h * head_dim;
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < ngrp) {
                const float4 a = *reinterpret_cast<const float4*>(qp + c * 128 + lane * 4);
                qv[c * 4 + 0] = a.x;
                qv[c * 4 + 1] = a.y;
                qv[c * 4 + 2] = a.z;
                qv[c * 4 + 3] = a.w;
            }
        }
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[d] = 0.0f;
    }

    int bidx[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) bidx[j] = ((lane ^ (1 << j)) << 2);

    const unsigned char* kbase = (const unsigned char*)k + (size_t)kvh * seq * row_bytes;
    const unsigned char* vbase = (const unsigned char*)v + (size_t)kvh * seq * row_bytes;

    float m = -INFINITY;
    float l = 0.0f;

    for (int pos = start; pos < end; ++pos) {
        float kd[8];
        q8s_row(kd, kbase + (size_t)pos * row_bytes, lane, ngrp);
        float dot = 0.0f;
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d < ngrp * 4) dot = fmaf(qv[d], kd[d], dot);
        }
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            dot += __int_as_float(
                __builtin_amdgcn_ds_bpermute(bidx[j], __float_as_int(dot)));
        }
        const float sc = dot * sl2;
        const float mt = fmaxf(m, sc);
        const float corr = exp2f(m - mt);
        const float p = exp2f(sc - mt);
        l = l * corr + p;
        float vd[8];
        q8s_row(vd, vbase + (size_t)pos * row_bytes, lane, ngrp);
#pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d < ngrp * 4) acc[d] = fmaf(p, vd[d], acc[d] * corr);
        }
        m = mt;
    }

    float* dst = part_acc + ((size_t)h * splits + split) * head_dim;
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < ngrp) {
            float4 a;
            a.x = acc[c * 4 + 0];
            a.y = acc[c * 4 + 1];
            a.z = acc[c * 4 + 2];
            a.w = acc[c * 4 + 3];
            *reinterpret_cast<float4*>(dst + c * 128 + lane * 4) = a;
        }
    }
    if (lane == 0) {
        float* ml = part_ml + ((size_t)h * splits + split) * 2;
        ml[0] = m;
        ml[1] = l;
    }
}
