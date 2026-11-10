// SPDX-License-Identifier: Apache-2.0
// Attribution lab for the q8 decode-attention kernel.
//
// `attn_decode_kv.hip` is the shipped kernel; this file is the same shape with
// phases compiled out, so one 100K pass can say where the 13 ms goes:
//
//   kv_lab_full   stage K+V, score, PV   (same work as the shipped kernel)
//   kv_lab_nov    stage K, score, no V   (V fetch + PV removed)
//   kv_lab_nok    stage V, no score, PV  (K fetch + score removed)
//   kv_lab_stage  stage K+V, no compute  (both compute phases removed)
//   kv_lab_nostage score + PV, no staging (reads uninitialised shared)
//
// Only the timing matters here: the entries are not in the parity table and
// are not used by the engine. Geometry is fixed to the Qwen3.8-27B decode
// case (head_dim 256, GQA group 6, TILE 4) so the body stays readable.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define LAB_TILE 4
#define LAB_LOG2E 1.4426950408889634f
#define LAB_BLOCK_BYTES 34

// One four-value group of a q8 block (scale + 32 signed bytes).
__device__ __forceinline__ void lab_ld4p(
    unsigned& w, float& d, const char* pl, const char* sc)
{
    d = __half2float(*(const __half*)sc);
    w = *(const unsigned*)pl;
}

__device__ __forceinline__ void lab_st4(float* o, unsigned w, float d) {
    float4 t;
    t.x = d * (float)((int)(w << 24) >> 24);
    t.y = d * (float)((int)(w << 16) >> 24);
    t.z = d * (float)((int)(w << 8) >> 24);
    t.w = d * (float)((int)w >> 24);
    *(float4*)__builtin_assume_aligned(o, 16) = t;
}

// Stage one array's tile: 64 groups, one per active thread, TILE rows each.
template <int TILE>
__device__ __forceinline__ void lab_stage(
    float* dst, const char* row0, int rows, int row_bytes, int local, bool active)
{
    if (!active) return;
    const char* pl = row0 + (local >> 3) * LAB_BLOCK_BYTES + 2 + ((local & 7) << 2);
    const char* sc = row0 + (local >> 3) * LAB_BLOCK_BYTES;
    float* q = dst + local * 4;
    unsigned w[TILE];
    float d[TILE];
    #pragma unroll
    for (int r = 0; r < TILE; ++r) {
        lab_ld4p(w[r], d[r], pl, sc);
        const int adv = (r + 1 < rows) ? row_bytes : 0;
        pl += adv;
        sc += adv;
    }
    #pragma unroll
    for (int r = 0; r < TILE; ++r) {
        lab_st4(q, w[r], d[r]);
        q += (r + 1 < rows) ? 256 : 0;
    }
}

// MODE bits: 1 = K fetch + score, 2 = V fetch + PV, 4 = staging.
template <int MODE>
__device__ __forceinline__ void lab_body(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_kv,
    const int* __restrict__ step,
    int stride,
    float scale,
    int splits)
{
    const int seq = step[1];
    extern __shared__ __align__(16) float sh[];
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = kvh * 6 + warp;
    const int per = (seq + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;
    const float sl2 = scale * LAB_LOG2E;
    const int row_bytes = 8 * LAB_BLOCK_BYTES;

    float qv[8];
    float acc[8];
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        const float4 t = *(const float4*)(q + (size_t)h * 256 + (c * 32 + lane) * 4);
        qv[c * 4 + 0] = t.x; qv[c * 4 + 1] = t.y;
        qv[c * 4 + 2] = t.z; qv[c * 4 + 3] = t.w;
    }
    #pragma unroll
    for (int d = 0; d < 8; ++d) acc[d] = 0.0f;

    int bidx[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) bidx[j] = ((lane ^ (1 << j)) << 2);

    float* ks = sh;
    float* vs = sh + LAB_TILE * 256;
    const char* kbase = (const char*)k + (size_t)kvh * stride * row_bytes;
    const char* vbase = (const char*)v + (size_t)kvh * stride * row_bytes;

    float m = -INFINITY;
    float l = 0.0f;

    for (int pos = start; pos < end; pos += LAB_TILE) {
        const int rows = (pos + LAB_TILE <= end) ? LAB_TILE : (end - pos);
        __syncthreads();
        if (MODE & 4) {
            // The block's first 64 threads stage K and the next 64 V, as in
            // the shipped kernel (the remaining 64 sit out the staging pass).
            const int tid = threadIdx.x;
            if (MODE & 1)
                lab_stage<LAB_TILE>(ks, kbase + (size_t)pos * row_bytes, rows,
                                    row_bytes, tid, tid < 64);
            if (MODE & 2)
                lab_stage<LAB_TILE>(vs, vbase + (size_t)pos * row_bytes, rows,
                                    row_bytes, tid - 64, tid >= 64 && tid < 128);
        }
        __syncthreads();

        float s[LAB_TILE];
        #pragma unroll
        for (int t = 0; t < LAB_TILE; ++t) {
            if (!(MODE & 1)) { s[t] = 0.0f; continue; }
            float kd[8];
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const float4 x = *(const float4*)(ks + t * 256 + (c * 32 + lane) * 4);
                kd[c * 4 + 0] = x.x; kd[c * 4 + 1] = x.y;
                kd[c * 4 + 2] = x.z; kd[c * 4 + 3] = x.w;
            }
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < 8; ++d) dot = fmaf(qv[d], kd[d], dot);
            #pragma unroll
            for (int j = 0; j < 5; ++j)
                dot += __int_as_float(
                    __builtin_amdgcn_ds_bpermute(bidx[j], __float_as_int(dot)));
            s[t] = (t < rows) ? dot * sl2 : -INFINITY;
        }

        float mt = m;
        #pragma unroll
        for (int t = 0; t < LAB_TILE; ++t) mt = fmaxf(mt, s[t]);
        const float corr = exp2f(m - mt);
        l *= corr;
        #pragma unroll
        for (int d = 0; d < 8; ++d) acc[d] *= corr;
        #pragma unroll
        for (int t = 0; t < LAB_TILE; ++t) {
            if (t >= rows) break;
            const float p = exp2f(s[t] - mt);
            l += p;
            if (!(MODE & 2)) continue;
            float vd[8];
            #pragma unroll
            for (int c = 0; c < 2; ++c) {
                const float4 x = *(const float4*)(vs + t * 256 + (c * 32 + lane) * 4);
                vd[c * 4 + 0] = x.x; vd[c * 4 + 1] = x.y;
                vd[c * 4 + 2] = x.z; vd[c * 4 + 3] = x.w;
            }
            #pragma unroll
            for (int d = 0; d < 8; ++d) acc[d] = fmaf(p, vd[d], acc[d]);
        }
        m = mt;
    }

    float* dst = part_acc + ((size_t)h * splits + split) * 256;
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        float4 t;
        t.x = acc[c * 4 + 0]; t.y = acc[c * 4 + 1];
        t.z = acc[c * 4 + 2]; t.w = acc[c * 4 + 3];
        *(float4*)(dst + (c * 32 + lane) * 4) = t;
    }
    if (lane == 0) {
        float* ml = part_ml + ((size_t)h * splits + split) * 2;
        ml[0] = m;
        ml[1] = l;
    }
}

#define LAB_ENTRY(NAME, MODE)                                                 \
    extern "C" __global__ void NAME(                                          \
        const float* __restrict__ q, const void* __restrict__ k,              \
        const void* __restrict__ v, float* __restrict__ part_acc,             \
        float* __restrict__ part_ml, int n_heads, int n_kv, int head_dim,     \
        const int* step, int stride, float scale, int splits)                 \
    {                                                                         \
        (void)n_heads; (void)head_dim;                                        \
        lab_body<MODE>(q, k, v, part_acc, part_ml, n_kv, step, stride,        \
                       scale, splits);                                        \
    }

LAB_ENTRY(attn_decode_kv_lab_full, 7)
LAB_ENTRY(attn_decode_kv_lab_nov, 5)
LAB_ENTRY(attn_decode_kv_lab_nok, 6)
LAB_ENTRY(attn_decode_kv_lab_stage, 4)
LAB_ENTRY(attn_decode_kv_lab_nostage, 3)

#undef LAB_ENTRY
