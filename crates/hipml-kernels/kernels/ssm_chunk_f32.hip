// SPDX-License-Identifier: Apache-2.0
// Gated-delta-net layer over a whole prefill chunk.
//
// The decode path launches seven kernels per token (gates, conv, silu, two
// L2 norms, recurrence, gated RMS). For a 256-token chunk that is 1,792
// launches per layer. These five kernels cover the same chain for every token
// in one launch each:
//
//   ssm_gates_chunk_f32     softplus(alpha+dt)*a, sigmoid(beta)   [T][n_v]
//   ssm_conv1d_chunk_f32    causal conv + silu, initial state      [T][C]
//   l2norm_chunk_f32        per-head L2 of q and k                 [T][2*n_k]
//   gdn_chunk_f32           the recurrence, state held in registers
//   gate_norm_chunk_f32     rmsnorm(o, ssm_norm) * silu(z)         [T][inner]
//
// The recurrence is the important one: `gdn_ar_f32` re-reads and rewrites the
// 64 KiB state matrix per head per token, so a chunk streams 73 GB through the
// state for a 512-token prompt. Here one block owns a value head for the whole
// chunk and keeps its state rows in registers (16 f32 per lane at 1024
// threads), so the state is read once and written once.
//
// The conv is parallel over (token, channel) rather than sequential over the
// window: tap `k` of output `t` reads token `t-d_conv+1+k`, which is in the
// chunk, or the carried state when that index is negative.

#include <hip/hip_runtime.h>

#define CH_LANES 32
#define CH_MAXV 4

__device__ __forceinline__ float ch_wave_sum(float v)
{
    for (int off = CH_LANES / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, CH_LANES);
    }
    return v;
}

extern "C" __global__ void ssm_gates_chunk_f32(
    const float* __restrict__ alpha,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ g,
    float* __restrict__ beta,
    int n_v,
    int tokens)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * n_v) return;
    const int h = i % n_v;
    const float z = alpha[i] + dt_bias[h];
    const float sp = z > 20.0f ? z : logf(1.0f + expf(z));
    g[i] = sp * a[h];
    beta[i] = 1.0f / (1.0f + expf(-beta_raw[i]));
}

extern "C" __global__ void copy_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

extern "C" __global__ void ssm_conv1d_chunk_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ state_in,
    float* __restrict__ state_out,
    float* __restrict__ y,
    int channels,
    int d_conv,
    int tokens)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * channels) return;
    const int t = i / channels;
    const int c = i - t * channels;
    const int hist = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;

    float acc = 0.0f;
    for (int k = 0; k < hist; ++k) {
        const int g = t - hist + k;
        // t-hist+k < 0 means the tap predates this chunk: state index t+k.
        const float v = g >= 0 ? x[(size_t)g * channels + c]
                               : state_in[(size_t)(t + k) * channels + c];
        acc += v * wc[k];
    }
    acc += x[(size_t)t * channels + c] * wc[hist];
    y[i] = acc / (1.0f + expf(-acc));

    // Carry the last `hist` rows into the next chunk. One writer per element:
    // token t >= tokens-hist owns state row t-(tokens-hist).
    if (t >= tokens - hist && hist > 0) {
        const int row = t - (tokens - hist);
        state_out[(size_t)row * channels + c] = x[(size_t)t * channels + c];
    }
}

extern "C" __global__ void l2norm_chunk_f32(
    const float* x,
    float* y,
    int n_k,
    int head_k,
    int key_dim,
    int conv_ch,
    float eps,
    int tokens)
{
    // One block per (token, q|k head).
    const int t = blockIdx.y;
    const int half = blockIdx.x >= n_k ? 1 : 0;
    const int h = blockIdx.x - half * n_k;
    const int off = t * conv_ch + half * key_dim + h * head_k;
    const float* src = x + off;
    float* dst = y + off;
    const int lane = threadIdx.x;

    float acc = 0.0f;
    for (int i = lane; i < head_k; i += blockDim.x) acc += src[i] * src[i];

    __shared__ float red[128];
    red[lane] = acc;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (lane < step) red[lane] += red[lane + step];
        __syncthreads();
    }
    __shared__ float scale;
    if (lane == 0) scale = rsqrtf(red[0] + eps);
    __syncthreads();
    for (int i = lane; i < head_k; i += blockDim.x) dst[i] = src[i] * scale;
}

extern "C" __global__ void gdn_chunk_f32(
    float* __restrict__ s,
    const float* __restrict__ qkv,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int n_v,
    int n_k,
    int sd,
    int conv_ch,
    int key_dim,
    int tokens)
{
    // 32-lane groups; group `grp` owns value rows grp + 32*u.
    const int h = blockIdx.x;
    const int lane = threadIdx.x % CH_LANES;
    const int grp = threadIdx.x / CH_LANES;
    const int ngroups = blockDim.x / CH_LANES;
    const int rows_per_group = (sd + ngroups - 1) / ngroups;
    if (rows_per_group > CH_MAXV) return;

    const int hk = h % n_k;
    const float scale = rsqrtf((float)sd);

    // Load the state once: r[u][t] is element lane + t*32 of row grp+32*u.
    float r[CH_MAXV][CH_MAXV];
    float* head = s + (size_t)h * sd * sd;
#pragma unroll
    for (int u = 0; u < CH_MAXV; ++u) {
        const int j = grp + ngroups * u;
        if (j >= sd) continue;
        float* row = head + (size_t)j * sd;
#pragma unroll
        for (int t = 0; t < CH_MAXV; ++t) {
            const int i = lane + CH_LANES * t;
            r[u][t] = i < sd ? row[i] : 0.0f;
        }
    }

    for (int tok = 0; tok < tokens; ++tok) {
        const int gi = tok * n_v + h;
        const float decay = expf(g[gi]);
        const float b = beta[gi];

        const float* qrow = qkv + (size_t)tok * conv_ch + (size_t)hk * sd;
        const float* krow = qrow + key_dim;
        const float* vrow = qkv + (size_t)tok * conv_ch + 2 * (size_t)key_dim +
                            (size_t)h * sd;

        float qq[CH_MAXV];
        float kk[CH_MAXV];
#pragma unroll
        for (int t = 0; t < CH_MAXV; ++t) {
            const int i = lane + CH_LANES * t;
            if (i < sd) {
                qq[t] = qrow[i];
                kk[t] = krow[i];
            } else {
                qq[t] = 0.0f;
                kk[t] = 0.0f;
            }
        }

#pragma unroll
        for (int u = 0; u < CH_MAXV; ++u) {
            const int j = grp + ngroups * u;
            if (j >= sd) continue;
            float dot = 0.0f;
#pragma unroll
            for (int t = 0; t < CH_MAXV; ++t) dot += r[u][t] * decay * kk[t];
            dot = ch_wave_sum(dot);
            const float delta = (vrow[j] - dot) * b;

            float acc = 0.0f;
#pragma unroll
            for (int t = 0; t < CH_MAXV; ++t) {
                const float nv = r[u][t] * decay + kk[t] * delta;
                r[u][t] = nv;
                acc += nv * qq[t];
            }
            acc = ch_wave_sum(acc);
            if (lane == 0) out[(size_t)tok * (size_t)n_v * sd + (size_t)h * sd + j] =
                acc * scale;
        }
    }

    // Write the state back once.
#pragma unroll
    for (int u = 0; u < CH_MAXV; ++u) {
        const int j = grp + ngroups * u;
        if (j >= sd) continue;
        float* row = head + (size_t)j * sd;
#pragma unroll
        for (int t = 0; t < CH_MAXV; ++t) {
            const int i = lane + CH_LANES * t;
            if (i < sd) row[i] = r[u][t];
        }
    }
}

#define CH_GN_BLOCK 128

extern "C" __global__ void gate_norm_chunk_f32(
    const float* __restrict__ o,
    const float* __restrict__ norm,
    const float* __restrict__ z,
    float* __restrict__ y,
    int n_v,
    int dim,
    float eps)
{
    // One block per (token, value head).
    const int t = blockIdx.y;
    const int h = blockIdx.x;
    const size_t base = ((size_t)t * n_v + h) * dim;
    const float* src = o + base;
    float* dst = y + base;
    const float* zz = z + base;
    const int lane = threadIdx.x;

    float acc = 0.0f;
    for (int i = lane; i < dim; i += CH_GN_BLOCK) acc += src[i] * src[i];

    __shared__ float red[CH_GN_BLOCK];
    red[lane] = acc;
    __syncthreads();
    for (int step = CH_GN_BLOCK / 2; step > 0; step >>= 1) {
        if (lane < step) red[lane] += red[lane + step];
        __syncthreads();
    }
    __shared__ float scale;
    if (lane == 0) scale = rsqrtf(red[0] / (float)dim + eps);
    __syncthreads();

    for (int i = lane; i < dim; i += CH_GN_BLOCK) {
        const float gv = zz[i];
        const float silu = gv / (1.0f + expf(-gv));
        dst[i] = src[i] * scale * norm[i] * silu;
    }
}
