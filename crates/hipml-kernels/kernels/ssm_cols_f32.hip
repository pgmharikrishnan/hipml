// SPDX-License-Identifier: Apache-2.0
// The GDN per-token chain over a chunk of `rows` verify columns.
//
// The recurrence is sequential in token order, so the columns are walked inside
// one kernel instead of one launch per column. Every stage keeps the same
// arithmetic order as its single-token kernel:
//
//   ssm_gates_cols_f32     per (row, head), identical expressions
//   ssm_conv1d_cols_f32    one thread per channel, the window slides row by row;
//                          the SiLU is fused (it is elementwise on the result)
//   l2norm_cols_f32        one block per (row, head), same 256-lane tree
//   gdn_cols_f32           one block per value head; each lane group keeps its
//                          slice of a state row in registers across the chunk
//   ssm_gate_norm_cols_f32 one block per (row, head)
//
// `gdn_cols_f32` and `ssm_conv1d_cols_f32` also write the recurrent state after
// column `r < rows-1` into a snapshot scratch, which is what a rejected draft
// restores. The snapshots are `[columns][layers][state_len]`: a layer's base
// pointer plus `snap_stride` floats per column. The value written is the same
// state the previous path memcpy'd after each column, written where it is
// produced.

#include <hip/hip_runtime.h>

#define SC_L2_BLOCK 256
#define SC_MAXV 4
#define SC_LANES 32

__device__ __forceinline__ float sc_wave_sum(float v)
{
    for (int off = SC_LANES / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, SC_LANES);
    }
    return v;
}

extern "C" __global__ void ssm_gates_cols_f32(
    const float* __restrict__ alpha,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ a,
    float* __restrict__ g,
    float* __restrict__ beta,
    int n,
    int rows)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const int h = i % n;
    const float z = alpha[i] + dt_bias[h];
    const float sp = z > 20.0f ? z : logf(1.0f + expf(z));
    g[i] = sp * a[h];
    beta[i] = 1.0f / (1.0f + expf(-beta_raw[i]));
}

// Depthwise causal conv1d over `rows` consecutive tokens, fused with the SiLU
// activation. Thread `c` owns channel `c` of every row, so the carried window
// needs no barrier. `snap`, when non-null, receives the window after each row
// but the last.
extern "C" __global__ void ssm_conv1d_cols_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ state,
    float* __restrict__ y,
    float* __restrict__ snap,
    int snap_stride,
    int channels,
    int d_conv,
    int rows)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const int hist = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;

    for (int tok = 0; tok < rows; ++tok) {
        const float* xr = x + (size_t)tok * channels;

        float acc = 0.0f;
        for (int k = 0; k < hist; ++k) {
            acc = __fmaf_rn(state[(size_t)k * channels + c], wc[k], acc);
        }
        acc = __fmaf_rn(xr[c], wc[hist], acc);
        const float v = acc;

        // Slide left, then append the current token (read-before-write is per
        // thread, so the copy cannot race another channel).
        for (int k = 0; k + 1 < hist; ++k) {
            state[(size_t)k * channels + c] = state[(size_t)(k + 1) * channels + c];
        }
        if (hist > 0) {
            state[(size_t)(hist - 1) * channels + c] = xr[c];
        }

        y[(size_t)tok * channels + c] = v / (1.0f + expf(-v));

        if (snap != 0 && tok + 1 < rows) {
            float* sp = snap + (size_t)tok * (size_t)snap_stride;
            for (int k = 0; k < hist; ++k) {
                sp[(size_t)k * channels + c] = state[(size_t)k * channels + c];
            }
        }
    }
}

extern "C" __global__ void l2norm_cols_f32(
    const float* x,
    float* y,
    int heads,
    int dim,
    float eps,
    int rows,
    int stride)
{
    const int row = blockIdx.x / heads;
    const int h = blockIdx.x - row * heads;
    const int t = threadIdx.x;
    const float* src = x + (size_t)row * (size_t)stride + (size_t)h * dim;
    float* dst = y + (size_t)row * (size_t)stride + (size_t)h * dim;

    float acc = 0.0f;
    for (int i = t; i < dim; i += SC_L2_BLOCK) {
        acc = __fmaf_rn(src[i], src[i], acc);
    }

    __shared__ float s[SC_L2_BLOCK];
    s[t] = acc;
    __syncthreads();
    for (int off = SC_L2_BLOCK / 2; off > 0; off >>= 1) {
        if (t < off) s[t] += s[t + off];
        __syncthreads();
    }

    __shared__ float scale;
    if (t == 0) scale = rsqrtf(s[0] + eps);
    __syncthreads();

    for (int i = t; i < dim; i += SC_L2_BLOCK) {
        dst[i] = src[i] * scale;
    }
}

// The recurrence over `rows` columns. `q`, `k`, `v` are rows of the conv output
// (`qkv_stride` wide, q at 0, k at `key_dim`, v at `2*key_dim`); `g` and `beta`
// are `[rows][n_v]`. A lane group owns one value row and keeps its `sd/32`
// components in registers across the chunk, so the state is read once and
// written once per head, and the snapshot rows are written between tokens.
extern "C" __global__ void gdn_cols_f32(
    float* __restrict__ s,
    const float* __restrict__ qkv,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ snap,
    int snap_stride,
    int n_v,
    int n_k,
    int qkv_stride,
    int key_dim,
    int sd,
    int rows)
{
    const int h = blockIdx.x;
    const int lane = threadIdx.x % SC_LANES;
    const int wave = threadIdx.x / SC_LANES;
    const int nwave = blockDim.x / SC_LANES;
    const int hk = h % n_k;
    const float scale = rsqrtf((float)sd);

    float* head = s + (size_t)h * sd * sd;
    const size_t head_len = (size_t)sd * sd;

    for (int j = wave; j < sd; j += nwave) {
        float* row = head + (size_t)j * sd;
        float r[SC_MAXV];
        #pragma unroll
        for (int t = 0; t < SC_MAXV; ++t) {
            const int i = lane + t * SC_LANES;
            r[t] = i < sd ? row[i] : 0.0f;
        }

        for (int tok = 0; tok < rows; ++tok) {
            const float* qh = qkv + (size_t)tok * qkv_stride + (size_t)hk * sd;
            const float* kh = qh + key_dim;
            const float* vh = qkv + (size_t)tok * qkv_stride +
                              2 * (size_t)key_dim + (size_t)h * sd;
            const float decay = expf(g[(size_t)tok * n_v + h]);
            const float b = beta[(size_t)tok * n_v + h];

            float dot = 0.0f;
            #pragma unroll
            for (int t = 0; t < SC_MAXV; ++t) {
                const int i = lane + t * SC_LANES;
                if (i < sd) {
                    dot = __fmaf_rn(__fmul_rn(r[t], decay), kh[i], dot);
                }
            }
            dot = sc_wave_sum(dot);

            const float delta = (vh[j] - dot) * b;

            float acc = 0.0f;
            #pragma unroll
            for (int t = 0; t < SC_MAXV; ++t) {
                const int i = lane + t * SC_LANES;
                if (i < sd) {
                    const float nv =
                        __fmaf_rn(r[t], decay, __fmul_rn(kh[i], delta));
                    r[t] = nv;
                    acc = __fmaf_rn(nv, qh[i], acc);
                }
            }
            acc = sc_wave_sum(acc);

            if (lane == 0) {
                out[(size_t)tok * n_v * sd + (size_t)h * sd + j] = acc * scale;
            }

            if (snap != 0 && tok + 1 < rows) {
                float* sp = snap + (size_t)tok * (size_t)snap_stride +
                            (size_t)h * head_len + (size_t)j * sd;
                #pragma unroll
                for (int t = 0; t < SC_MAXV; ++t) {
                    const int i = lane + t * SC_LANES;
                    if (i < sd) sp[i] = r[t];
                }
            }
        }

        #pragma unroll
        for (int t = 0; t < SC_MAXV; ++t) {
            const int i = lane + t * SC_LANES;
            if (i < sd) row[i] = r[t];
        }
    }
}

extern "C" __global__ void ssm_gate_norm_cols_f32(
    const float* __restrict__ o,
    const float* __restrict__ norm,
    const float* __restrict__ z,
    float* __restrict__ y,
    int heads,
    int dim,
    float eps,
    int rows)
{
    const int row = blockIdx.x / heads;
    const int h = blockIdx.x - row * heads;
    const int t = threadIdx.x;
    const size_t base = ((size_t)row * heads + h) * dim;
    const float* src = o + base;
    float* dst = y + base;
    const float* zz = z + base;

    float acc = 0.0f;
    for (int i = t; i < dim; i += SC_L2_BLOCK) {
        acc = __fmaf_rn(src[i], src[i], acc);
    }

    __shared__ float s[SC_L2_BLOCK];
    s[t] = acc;
    __syncthreads();
    for (int off = SC_L2_BLOCK / 2; off > 0; off >>= 1) {
        if (t < off) s[t] += s[t + off];
        __syncthreads();
    }

    __shared__ float scale;
    if (t == 0) {
        scale = rsqrtf(s[0] / (float)dim + eps);
    }
    __syncthreads();

    for (int i = t; i < dim; i += SC_L2_BLOCK) {
        const float g = zz[i];
        const float silu = g / (1.0f + expf(-g));
        dst[i] = src[i] * scale * norm[i] * silu;
    }
}
