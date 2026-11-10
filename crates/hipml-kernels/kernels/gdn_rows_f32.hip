// SPDX-License-Identifier: Apache-2.0
// Gated-delta-net recurrence over a prefill chunk, split by value rows.
//
// `gdn_chunk_f32` gives each value head one 1024-thread block: 48 blocks on a
// 64-CU card, and every token's q, k, v, g and beta come straight from global
// memory inside the sequential loop, so each step waits on a load. The rows of
// a head's state are independent (row j only needs v[j] and the head's shared
// q, k, decay, beta), so here a block owns 32 rows of one head:
//
//   grid (n_v, sd / 32), block 256 = 8 waves x 4 rows
//
// and a 16-token tile of q, k, the block's 32 v values, g and beta is staged
// into shared memory one tile ahead (register prefetch, two shared buffers,
// one barrier per tile). Each row does exactly the arithmetic `gdn_chunk_f32`
// does, element for element and in the same order, so the output and the
// carried state are bit-identical; only where the operands come from changed.
//
// Requires head_k == head_v == 128 (the host checks and falls back).

#include <hip/hip_runtime.h>

#define GR_LANES 32
#define GR_SD 128
#define GR_ROWS 32
#define GR_BLOCK 256
#define GR_WAVES (GR_BLOCK / GR_LANES)
#define GR_RPW (GR_ROWS / GR_WAVES)
#define GR_TPL (GR_SD / GR_LANES)
#define GR_TT 16

__device__ __forceinline__ float gr_wave_sum(float v)
{
    for (int off = GR_LANES / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, GR_LANES);
    }
    return v;
}

// One tile's operands in flight: q and k are TT x 32 float4 each (two per
// thread), v is TT x 8 float4 (threads 0..127), g and beta one per token.
struct GrPrefetch {
    float4 q[2];
    float4 k[2];
    float4 v;
    float g;
    float b;
};

__device__ __forceinline__ void gr_load(GrPrefetch& p,
                                        const float* __restrict__ qkv,
                                        const float* __restrict__ g,
                                        const float* __restrict__ beta,
                                        int tok0, int tokens, int n_v, int h,
                                        int hk, int jb, int conv_ch, int key_dim)
{
    const int tid = threadIdx.x;
    const float4 z = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + GR_BLOCK * i;
        const int tok = tok0 + (idx >> 5);
        const int c = idx & 31;
        if (tok < tokens) {
            const float* row = qkv + (size_t)tok * conv_ch + (size_t)hk * GR_SD;
            p.q[i] = reinterpret_cast<const float4*>(row)[c];
            p.k[i] = reinterpret_cast<const float4*>(row + key_dim)[c];
        } else {
            p.q[i] = z;
            p.k[i] = z;
        }
    }
    p.v = z;
    if (tid < GR_TT * (GR_ROWS / 4)) {
        const int tok = tok0 + (tid >> 3);
        if (tok < tokens) {
            const float* row = qkv + (size_t)tok * conv_ch + 2 * (size_t)key_dim +
                               (size_t)h * GR_SD + jb;
            p.v = reinterpret_cast<const float4*>(row)[tid & 7];
        }
    }
    p.g = 0.0f;
    p.b = 0.0f;
    if (tid < GR_TT && tok0 + tid < tokens) {
        const int gi = (tok0 + tid) * n_v + h;
        p.g = g[gi];
        p.b = beta[gi];
    }
}

__device__ __forceinline__ void gr_store(const GrPrefetch& p, float* sq, float* sk,
                                         float* sv, float* sg, float* sb)
{
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int idx = tid + GR_BLOCK * i;
        reinterpret_cast<float4*>(sq)[idx] = p.q[i];
        reinterpret_cast<float4*>(sk)[idx] = p.k[i];
    }
    if (tid < GR_TT * (GR_ROWS / 4)) reinterpret_cast<float4*>(sv)[tid] = p.v;
    if (tid < GR_TT) {
        sg[tid] = p.g;
        sb[tid] = p.b;
    }
}

extern "C" __global__ void __launch_bounds__(GR_BLOCK) gdn_rows_f32(
    float* __restrict__ s,
    const float* __restrict__ qkv,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int n_v,
    int n_k,
    int conv_ch,
    int key_dim,
    int tokens)
{
    __shared__ __align__(16) float sq[2][GR_TT * GR_SD];
    __shared__ __align__(16) float sk[2][GR_TT * GR_SD];
    __shared__ __align__(16) float sv[2][GR_TT * GR_ROWS];
    __shared__ float sg[2][GR_TT];
    __shared__ float sb[2][GR_TT];

    const int h = blockIdx.x;
    const int jb = blockIdx.y * GR_ROWS;
    const int lane = threadIdx.x % GR_LANES;
    const int wave = threadIdx.x / GR_LANES;
    const int hk = h % n_k;
    const float scale = rsqrtf((float)GR_SD);

    // r[u][t] is element lane + 32t of row jb + wave + 8u.
    float r[GR_RPW][GR_TPL];
    float* head = s + (size_t)h * GR_SD * GR_SD;
#pragma unroll
    for (int u = 0; u < GR_RPW; ++u) {
        const float* row = head + (size_t)(jb + wave + GR_WAVES * u) * GR_SD;
#pragma unroll
        for (int t = 0; t < GR_TPL; ++t) r[u][t] = row[lane + GR_LANES * t];
    }

    GrPrefetch p;
    gr_load(p, qkv, g, beta, 0, tokens, n_v, h, hk, jb, conv_ch, key_dim);
    gr_store(p, sq[0], sk[0], sv[0], sg[0], sb[0]);
    __syncthreads();

    for (int tok0 = 0, buf = 0; tok0 < tokens; tok0 += GR_TT, buf ^= 1) {
        const bool more = tok0 + GR_TT < tokens;
        if (more)
            gr_load(p, qkv, g, beta, tok0 + GR_TT, tokens, n_v, h, hk, jb, conv_ch,
                    key_dim);

        const int nt = min(GR_TT, tokens - tok0);
        for (int i = 0; i < nt; ++i) {
            const float decay = expf(sg[buf][i]);
            const float b = sb[buf][i];
            const float* qrow = sq[buf] + i * GR_SD;
            const float* krow = sk[buf] + i * GR_SD;
            const float* vrow = sv[buf] + i * GR_ROWS;

            float qq[GR_TPL];
            float kk[GR_TPL];
#pragma unroll
            for (int t = 0; t < GR_TPL; ++t) {
                qq[t] = qrow[lane + GR_LANES * t];
                kk[t] = krow[lane + GR_LANES * t];
            }
            const size_t obase = (size_t)(tok0 + i) * (size_t)n_v * GR_SD +
                                 (size_t)h * GR_SD + jb;
#pragma unroll
            for (int u = 0; u < GR_RPW; ++u) {
                const int jl = wave + GR_WAVES * u;
                float dot = 0.0f;
#pragma unroll
                for (int t = 0; t < GR_TPL; ++t) dot += r[u][t] * decay * kk[t];
                dot = gr_wave_sum(dot);
                const float delta = (vrow[jl] - dot) * b;

                float acc = 0.0f;
#pragma unroll
                for (int t = 0; t < GR_TPL; ++t) {
                    const float nv = r[u][t] * decay + kk[t] * delta;
                    r[u][t] = nv;
                    acc += nv * qq[t];
                }
                acc = gr_wave_sum(acc);
                if (lane == 0) out[obase + jl] = acc * scale;
            }
        }

        if (more) gr_store(p, sq[buf ^ 1], sk[buf ^ 1], sv[buf ^ 1], sg[buf ^ 1],
                           sb[buf ^ 1]);
        __syncthreads();
    }

#pragma unroll
    for (int u = 0; u < GR_RPW; ++u) {
        float* row = head + (size_t)(jb + wave + GR_WAVES * u) * GR_SD;
#pragma unroll
        for (int t = 0; t < GR_TPL; ++t) row[lane + GR_LANES * t] = r[u][t];
    }
}
