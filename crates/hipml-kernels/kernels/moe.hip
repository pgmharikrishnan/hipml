// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, commit d7725699 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
//
// The mixture-of-experts FFN on the device: router top-k, the expert gather,
// the weighted combine and the shared-expert scalar.
//
// The conventions are the ones `hipml_core::moe` states and `moe-gate` proves
// against llama.cpp's ids (`llama-graph.cpp::build_moe_ffn`). They are repeated
// here because this is the second implementation and the one that ships:
//
//   1. softmax over **all** experts, not over the selected k;
//   2. top-k by value descending, ties by expert id ascending;
//   3. the k weights are the unbiased probabilities renormalised by their sum,
//      clamped at 6.103515625e-5 (the smallest normal f16);
//   4. per expert: up, gate, silu(gate)*up, down, then multiply by the weight;
//   5. the combine is a left fold in top-k rank order;
//   6. the shared expert is one dense SwiGLU scaled by
//      sigmoid(dot(x, ffn_gate_inp_shexp)), added outside the routed sum.
//
// Nothing here reads back to the host. The token graph is captured once
// (`gpu/graph.rs`), so a host readback of the ids per layer would break capture;
// the ids and weights stay in device buffers and the expert GEMV resolves
// `w + ids[j] * stride` inside the kernel.

#include <hip/hip_runtime.h>
#include <math.h>

// The expert GEMV reuses the tile bodies of `gemv_jobs.hip`, declared there and
// defined in the per-format files later in the translation unit. Redeclared
// here so this file does not depend on include order.
template <int RPW, bool WIDE, bool STREAM>
__device__ __forceinline__ void q4k_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile);
template <int RPW>
__device__ __forceinline__ void q6k_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile);
__device__ __forceinline__ void q5k_rows(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile);
__device__ __forceinline__ void q80_row(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row);
template <int RPW>
static __device__ __forceinline__ void iq4x_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    float* __restrict__ scb,
    int tile);

// Same kind ids as `gemv_jobs.hip` (`HipmlGemvJob::kind`).
#define MOE_GJ_Q4K 0
#define MOE_GJ_Q6K 1
#define MOE_GJ_Q5K 2
#define MOE_GJ_Q80 3
#define MOE_GJ_IQ4XS 4

// Tile geometry, mirroring `gemv_jobs.hip`. The host computes the grid with
// `gemv_job_tiles`, which is the same function the job-list launch uses, so the
// two cannot drift apart silently.
#define MOE_ROWS_PER_TILE 8
#define MOE_IQ4X_RPW 2
#define MOE_IQ4X_ROWS_PER_TILE (8 * MOE_IQ4X_RPW)

#define MOE_BLOCK 256
// Upper bound on the expert count this kernel can hold in shared memory. 256
// experts is what every Qwen MoE file uses; the host refuses a wider router
// rather than reading past the array.
#define MOE_TOP_K_MAX 1024

// The router: softmax over every expert, top-k, renormalise.
//
// One block. `sp` holds the probabilities so a selected expert can be retired
// (set to -inf) without losing the value the weight is computed from, which is
// why the weights are written at selection time and normalised afterwards.
extern "C" __global__ void moe_topk_f32(
    const float* __restrict__ logits,
    int n_expert,
    int k,
    unsigned int* __restrict__ ids,
    float* __restrict__ weights)
{
    __shared__ float sp[MOE_TOP_K_MAX];
    __shared__ float sv[MOE_BLOCK];
    __shared__ int si[MOE_BLOCK];
    __shared__ double sd[MOE_BLOCK];

    const int tid = threadIdx.x;
    const int nt = blockDim.x;

    // 1. max, so the exponential cannot overflow.
    float mx = -INFINITY;
    for (int i = tid; i < n_expert; i += nt) mx = fmaxf(mx, logits[i]);
    sv[tid] = mx;
    __syncthreads();
    for (int off = MOE_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sv[tid] = fmaxf(sv[tid], sv[tid + off]);
        __syncthreads();
    }
    const float m = sv[0];
    __syncthreads();

    // 2. exp, summed in f64 as `ggml_vec_soft_max_f32` accumulates
    // (`ggml_float` is a double).
    double s = 0.0;
    for (int i = tid; i < n_expert; i += nt) {
        const float p = expf(logits[i] - m);
        sp[i] = p;
        s += (double)p;
    }
    sd[tid] = s;
    __syncthreads();
    for (int off = MOE_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) sd[tid] += sd[tid + off];
        __syncthreads();
    }
    const float scale = (float)(1.0 / sd[0]);
    __syncthreads();
    for (int i = tid; i < n_expert; i += nt) sp[i] *= scale;
    __syncthreads();

    // 3. k passes of (value desc, id asc). The reduction operator is the same
    // one `argmax_part_f32` uses, applied k times over the survivors.
    for (int j = 0; j < k; ++j) {
        float bv = -INFINITY;
        int bi = 0x7FFFFFFF;
        for (int i = tid; i < n_expert; i += nt) {
            const float v = sp[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        sv[tid] = bv;
        si[tid] = bi;
        __syncthreads();
        for (int off = MOE_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) {
                const float ov = sv[tid + off];
                const int oi = si[tid + off];
                if (ov > sv[tid] || (ov == sv[tid] && oi < si[tid])) {
                    sv[tid] = ov;
                    si[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            ids[j] = (unsigned int)si[0];
            weights[j] = sv[0];
            sp[si[0]] = -INFINITY;
        }
        __syncthreads();
    }

    // 4. Renormalise by the selected sum, clamped at the smallest normal f16.
    if (tid == 0) {
        double sum = 0.0;
        for (int j = 0; j < k; ++j) sum += (double)weights[j];
        const float denom = fmaxf((float)sum, 6.103515625e-5f);
        for (int j = 0; j < k; ++j) weights[j] = weights[j] / denom;
    }
}

// One GEMV per selected expert, one launch: `grid.y` is the rank, `grid.x` the
// row tile inside that expert's matrix.
//
// This is `gemv_jobs.hip` with the job's weight pointer resolved on the device:
// the rank's expert id is only known after `moe_topk_f32`, so the base is
// `w + ids[j] * stride` and the whole launch is capturable. `x_stride` is 0 for
// gate/up (every rank reads the same activation row) and `n_ff_exp` for down
// (rank j reads its own expert's SwiGLU output).
extern "C" __global__ void moe_gemv(
    const unsigned char* __restrict__ w,
    const int* __restrict__ ids,
    long long stride,
    const float* __restrict__ x,
    long long x_stride,
    float* __restrict__ y,
    int m,
    int k,
    int kind)
{
    const int j = blockIdx.y;
    const int tile = blockIdx.x;

    const unsigned char* wj = w + (long long)ids[j] * stride;
    const float* xj = x + (long long)j * x_stride;
    float* yj = y + (long long)j * m;

    const int rows_per_tile = (kind == MOE_GJ_IQ4XS) ? MOE_IQ4X_ROWS_PER_TILE
                                                     : MOE_ROWS_PER_TILE;
    const int need = (kind == MOE_GJ_Q80) ? m : (m + rows_per_tile - 1) / rows_per_tile;
    if (tile >= need) return;

    __shared__ __align__(16) float sx[2048];
    __shared__ __align__(16) float scb[16];
    switch (kind) {
    case MOE_GJ_Q4K:
        q4k_tile<1, false, false>(wj, xj, yj, m, k, sx, tile);
        break;
    case MOE_GJ_Q6K:
        q6k_tile<1>(wj, xj, yj, m, k, sx, tile);
        break;
    case MOE_GJ_Q5K:
        q5k_rows(wj, xj, yj, m, k, sx, tile);
        break;
    case MOE_GJ_Q80:
        for (int row = tile; row < m; row += gridDim.x) {
            q80_row(wj, xj, yj, m, k, row);
        }
        break;
    case MOE_GJ_IQ4XS:
        iq4x_tile<MOE_IQ4X_RPW>(wj, xj, yj, m, k, sx, scb, tile);
        break;
    }
}

// `y[i] = ((w[0]*d[0][i] + w[1]*d[1][i]) + w[2]*d[2][i]) + …`, a left fold in
// top-k rank order — `llama-graph.cpp` adds the expert outputs in that order,
// not in expert-id order.
extern "C" __global__ void moe_combine_f32(
    float* __restrict__ y,
    const float* __restrict__ down,
    const float* __restrict__ weights,
    int n,
    int n_used)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = weights[0] * down[i];
    for (int j = 1; j < n_used; ++j) a += weights[j] * down[(size_t)j * n + i];
    y[i] = a;
}

// `y += shared * scale`, the shared expert's single sigmoid scalar applied to
// its whole output and added outside the routed sum.
extern "C" __global__ void moe_shared_add_f32(
    float* __restrict__ y,
    const float* __restrict__ shared,
    const float* __restrict__ scale,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] += shared[i] * scale[0];
}

// `out[i] = 1 / (1 + exp(-x[i]))`. The shared expert's gate is one dot product,
// so this runs over a single value; it exists as a kernel so the whole token
// stays inside the captured graph.
extern "C" __global__ void sigmoid_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 1.0f / (1.0f + expf(-x[i]));
}
