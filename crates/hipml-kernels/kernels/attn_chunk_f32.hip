// SPDX-License-Identifier: Apache-2.0
// Full attention over a prefill chunk: four launches per layer instead of eight
// per token.
//
// The decode chain is qg_split -> head norm -> rope -> kv append -> attention,
// all one token at a time. For a 512-token chunk that is 4,096 launches per
// attention layer, and most of the wall time is launch ramp rather than work.
// Here the same stages run over every row of the chunk:
//
//   qg_split_chunk_f32    q|gate -> q, gate                    [T][n_head][hd]
//   norm_rope_chunk_f32   per-head RMS norm then M-RoPE        q and k rows
//   kv_append_chunk_f32   append T rows to the position cache
//   attn_chunk_f32        causal online softmax per (token, head)
//
// The attention kernel is the decode kernel's block-per-head shape with the
// causal bound `seq = pos0 + t + 1`: the chunk has already appended all its
// rows to the cache, so future rows are masked by the loop bound, not by a
// branch. Numerics match `attn_decode_f32` (online softmax, f32 accumulate).

#include <hip/hip_runtime.h>

extern "C" __global__ void qg_split_chunk_f32(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ gate,
    int n_heads,
    int head_dim,
    int tokens)
{
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    const float* src = qg + ((size_t)t * n_heads + h) * 2 * head_dim;
    float* qd = q + ((size_t)t * n_heads + h) * head_dim;
    float* gd = gate + ((size_t)t * n_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        qd[d] = src[d];
        gd[d] = src[head_dim + d];
    }
}

// One block per (token, head): RMS norm over the head, then rope in place.
// `blockIdx.x` is the q head when < n_heads, otherwise the k head.
extern "C" __global__ void norm_rope_chunk_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    const float* __restrict__ qw,
    const float* __restrict__ kw,
    int n_heads,
    int n_kv,
    int head_dim,
    int n_rot,
    float theta_base,
    float eps,
    int pos0,
    int tokens)
{
    const int t = blockIdx.y;
    const int hb = blockIdx.x;
    const bool is_q = hb < n_heads;
    const int h = is_q ? hb : hb - n_heads;
    const int stride = is_q ? n_heads : n_kv;
    if (h >= stride) return;
    float* x = (is_q ? q : k) + ((size_t)t * stride + h) * head_dim;
    const float* w = is_q ? qw : kw;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    float acc = 0.0f;
    for (int i = tid; i < head_dim; i += nthreads) acc += x[i] * x[i];

    __shared__ float red[256];
    __shared__ float scale;
    red[tid] = acc;
    __syncthreads();
    for (int step = nthreads / 2; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        __syncthreads();
    }
    if (tid == 0) scale = 1.0f / sqrtf(red[0] / (float)head_dim + eps);
    __syncthreads();
    for (int i = tid; i < head_dim; i += nthreads) x[i] = x[i] * scale * w[i];
    __syncthreads();

    // Text input: every M-RoPE section uses the token position, so this is
    // plain NeoX rope. Kept in double, matching `mrope_f32`.
    const int half = n_rot >> 1;
    const int pos = pos0 + t;
    for (int p = tid; p < half; p += nthreads) {
        const double freq = pow((double)theta_base, -2.0 * (double)p / (double)n_rot);
        const double ang = (double)pos * freq;
        double sn, cs;
        sincos(ang, &sn, &cs);
        const float x1 = x[p];
        const float x2 = x[p + half];
        x[p] = x1 * (float)cs - x2 * (float)sn;
        x[p + half] = x1 * (float)sn + x2 * (float)cs;
    }
}

extern "C" __global__ void kv_append_chunk_f32(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int pos0,
    int width,
    int tokens)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * tokens * width) return;
    const int t = i / (2 * width);
    const int j = i - t * (2 * width);
    const int off = pos0 + t;
    if (j < width) {
        k_cache[(size_t)off * width + j] = k[(size_t)t * width + j];
    } else {
        v_cache[(size_t)off * width + (j - width)] = v[(size_t)t * width + j - width];
    }
}

// Tiled causal attention: one block per (query head, 64-token query tile).
// Sixteen warps, four consecutive queries per warp, and each lane holds 8 of
// the 256 head components. A KV row is loaded once per warp and reused by its
// four queries; the only reductions are 5-step shuffles.
//
// The tile length is the whole cost model: every block streams the KV prefix
// once, so the layer's KV traffic is (tokens / tile) x prefix x 8 B per head.
// At 8K with a 32-token tile that is 1.6 TB per prompt and the chain measured
// 4.3 s — 384 GB/s of re-reads, not arithmetic. 64 tokens halves it, and the
// register budget sets the ceiling: 4 queries x 8 components x 2 (q and acc)
// f32 per lane is ~80 VGPRs, so 512 threads is one block per CU.
#define ACQ_TOKENS 64
#define ACQ_PER_WARP 4
#define ACQ_D 8
#define ACQ_THREADS ((ACQ_TOKENS / ACQ_PER_WARP) * 32)

extern "C" __global__ void attn_chunk_tile_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int n_heads,
    int n_kv,
    int head_dim,
    int width,
    int pos0,
    int tokens,
    float scale)
{
    const int h = blockIdx.x;
    const int tile = blockIdx.y * ACQ_TOKENS;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int group = n_heads / n_kv;
    const int kvh = h / group;

    const int tq[ACQ_PER_WARP] = {
        tile + warp * ACQ_PER_WARP + 0,
        tile + warp * ACQ_PER_WARP + 1,
        tile + warp * ACQ_PER_WARP + 2,
        tile + warp * ACQ_PER_WARP + 3,
    };
    if (tq[0] >= tokens) return;
    const bool live[ACQ_PER_WARP] = {
        tq[0] < tokens, tq[1] < tokens, tq[2] < tokens, tq[3] < tokens,
    };

    // lane*8 .. lane*8+7 of each query head, and the same slice of the
    // accumulator.
    float qv[ACQ_PER_WARP][ACQ_D];
    float acc[ACQ_PER_WARP][ACQ_D];
#pragma unroll
    for (int i = 0; i < ACQ_PER_WARP; ++i) {
        const float* qp = q + ((size_t)tq[i] * n_heads + h) * head_dim + lane * ACQ_D;
#pragma unroll
        for (int j = 0; j < ACQ_D; ++j) {
            qv[i][j] = live[i] ? qp[j] : 0.0f;
            acc[i][j] = 0.0f;
        }
    }

    float run_max[ACQ_PER_WARP] = {-3.402823466e+38f, -3.402823466e+38f,
                                   -3.402823466e+38f, -3.402823466e+38f};
    float run_sum[ACQ_PER_WARP] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Rows up to the last query of this warp; earlier queries mask the tail.
    int seq_max = pos0 + tq[ACQ_PER_WARP - 1] + 1;
    if (seq_max > pos0 + tokens) seq_max = pos0 + tokens;
    const float* kb = k_cache + (size_t)kvh * head_dim + (size_t)lane * ACQ_D;
    const float* vb = v_cache + (size_t)kvh * head_dim + (size_t)lane * ACQ_D;

    // Two cache rows per iteration: the second dot product is an independent
    // 5-step shuffle chain, so the warp has 2x the reduction ILP, the online
    // max is lifted once for both rows, and the accumulator is rescaled once
    // instead of twice.
    const int rows2 = seq_max & ~1;
    for (int row = 0; row < rows2; row += 2) {
        const float4 k0 = *reinterpret_cast<const float4*>(kb + (size_t)row * width);
        const float4 k1 = *reinterpret_cast<const float4*>(kb + (size_t)row * width + 4);
        const float4 v0 = *reinterpret_cast<const float4*>(vb + (size_t)row * width);
        const float4 v1 = *reinterpret_cast<const float4*>(vb + (size_t)row * width + 4);
        const float4 k2 = *reinterpret_cast<const float4*>(kb + (size_t)(row + 1) * width);
        const float4 k3 = *reinterpret_cast<const float4*>(kb + (size_t)(row + 1) * width + 4);
        const float4 v2 = *reinterpret_cast<const float4*>(vb + (size_t)(row + 1) * width);
        const float4 v3 = *reinterpret_cast<const float4*>(vb + (size_t)(row + 1) * width + 4);

#pragma unroll
        for (int i = 0; i < ACQ_PER_WARP; ++i) {
            const int bound = pos0 + tq[i];
            if (!live[i] || row > bound) continue;
            const bool two = row + 1 <= bound;
            float da = qv[i][0] * k0.x + qv[i][1] * k0.y + qv[i][2] * k0.z +
                       qv[i][3] * k0.w + qv[i][4] * k1.x + qv[i][5] * k1.y +
                       qv[i][6] * k1.z + qv[i][7] * k1.w;
            float db = qv[i][0] * k2.x + qv[i][1] * k2.y + qv[i][2] * k2.z +
                       qv[i][3] * k2.w + qv[i][4] * k3.x + qv[i][5] * k3.y +
                       qv[i][6] * k3.z + qv[i][7] * k3.w;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                da += __shfl_xor_sync(0xffffffffffffffffULL, da, off);
                db += __shfl_xor_sync(0xffffffffffffffffULL, db, off);
            }
            const float sa = da * scale;
            const float sb = two ? db * scale : -3.402823466e+38f;
            const float m_new = fmaxf(run_max[i], fmaxf(sa, sb));
            const float corr = run_max[i] == m_new ? 1.0f : expf(run_max[i] - m_new);
            const float pa = expf(sa - m_new);
            const float pb = two ? expf(sb - m_new) : 0.0f;
            run_max[i] = m_new;
            run_sum[i] = run_sum[i] * corr + pa + pb;
            acc[i][0] = acc[i][0] * corr + pa * v0.x + pb * v2.x;
            acc[i][1] = acc[i][1] * corr + pa * v0.y + pb * v2.y;
            acc[i][2] = acc[i][2] * corr + pa * v0.z + pb * v2.z;
            acc[i][3] = acc[i][3] * corr + pa * v0.w + pb * v2.w;
            acc[i][4] = acc[i][4] * corr + pa * v1.x + pb * v3.x;
            acc[i][5] = acc[i][5] * corr + pa * v1.y + pb * v3.y;
            acc[i][6] = acc[i][6] * corr + pa * v1.z + pb * v3.z;
            acc[i][7] = acc[i][7] * corr + pa * v1.w + pb * v3.w;
        }
    }
    if (seq_max & 1) {
        const int row = seq_max - 1;
        const float4 k0 = *reinterpret_cast<const float4*>(kb + (size_t)row * width);
        const float4 k1 = *reinterpret_cast<const float4*>(kb + (size_t)row * width + 4);
        const float4 v0 = *reinterpret_cast<const float4*>(vb + (size_t)row * width);
        const float4 v1 = *reinterpret_cast<const float4*>(vb + (size_t)row * width + 4);
#pragma unroll
        for (int i = 0; i < ACQ_PER_WARP; ++i) {
            if (!live[i] || row > pos0 + tq[i]) continue;
            float dot = qv[i][0] * k0.x + qv[i][1] * k0.y + qv[i][2] * k0.z +
                        qv[i][3] * k0.w + qv[i][4] * k1.x + qv[i][5] * k1.y +
                        qv[i][6] * k1.z + qv[i][7] * k1.w;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                dot += __shfl_xor_sync(0xffffffffffffffffULL, dot, off);
            const float score = dot * scale;
            const float m_new = fmaxf(run_max[i], score);
            const float corr = run_max[i] == m_new ? 1.0f : expf(run_max[i] - m_new);
            const float p = expf(score - m_new);
            run_max[i] = m_new;
            run_sum[i] = run_sum[i] * corr + p;
            acc[i][0] = acc[i][0] * corr + p * v0.x;
            acc[i][1] = acc[i][1] * corr + p * v0.y;
            acc[i][2] = acc[i][2] * corr + p * v0.z;
            acc[i][3] = acc[i][3] * corr + p * v0.w;
            acc[i][4] = acc[i][4] * corr + p * v1.x;
            acc[i][5] = acc[i][5] * corr + p * v1.y;
            acc[i][6] = acc[i][6] * corr + p * v1.z;
            acc[i][7] = acc[i][7] * corr + p * v1.w;
        }
    }

#pragma unroll
    for (int i = 0; i < ACQ_PER_WARP; ++i) {
        if (!live[i]) continue;
        float* op = out + ((size_t)tq[i] * n_heads + h) * head_dim + lane * ACQ_D;
        const float inv = 1.0f / run_sum[i];
#pragma unroll
        for (int j = 0; j < ACQ_D; ++j) op[j] = acc[i][j] * inv;
    }
}

#define AC_MAX_BLOCK 256

extern "C" __global__ void attn_chunk_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int n_heads,
    int n_kv,
    int head_dim,
    int width,
    int pos0,
    int tokens,
    float scale)
{
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;
    const int group = n_heads / n_kv;
    const int kvh = h / group;
    const int seq = pos0 + t + 1;

    const float* qv = q + ((size_t)t * n_heads + h) * head_dim;
    float* ov = out + ((size_t)t * n_heads + h) * head_dim;
    const float* kb = k_cache + (size_t)kvh * head_dim;
    const float* vb = v_cache + (size_t)kvh * head_dim;

    __shared__ float red[AC_MAX_BLOCK];
    __shared__ float run_max;
    __shared__ float run_sum;

    const float qa = qv[tid];
    if (tid == 0) {
        run_max = -3.402823466e+38f;
        run_sum = 0.0f;
    }
    __syncthreads();

    float acc = 0.0f;
    for (int row = 0; row < seq; ++row) {
        const float* kp = kb + (size_t)row * width;
        red[tid] = qa * kp[tid];
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (tid < off) red[tid] += red[tid + off];
            __syncthreads();
        }
        const float score = red[0] * scale;
        const float m_old = run_max;
        const float m_new = fmaxf(m_old, score);
        const float corr = expf(m_old - m_new);
        const float p = expf(score - m_new);
        if (tid == 0) {
            run_max = m_new;
            run_sum = run_sum * corr + p;
        }
        const float* vp = vb + (size_t)row * width;
        acc = acc * corr + p * vp[tid];
        __syncthreads();
    }

    ov[tid] = acc / run_sum;
}
