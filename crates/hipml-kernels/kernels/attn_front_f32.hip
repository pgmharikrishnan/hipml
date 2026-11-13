// SPDX-License-Identifier: Apache-2.0
// Portions of this file are derived from llama.cpp
// (https://github.com/ggml-org/llama.cpp, tag b12126 unless noted),
// Copyright (c) the ggml authors, used under the MIT License.
// See NOTICE and docs/reference/LICENSE.
// The decode attention front end in one launch: q/gate split, head RMS norm on
// q and k, and M-RoPE on both.
//
// The decode path used to issue five launches here -- `qg_split_f32`,
// `head_rmsnorm_f32` twice and `mrope_f32` twice -- for 40 + 8 head rows of
// work. They are one pass: block `h < n_heads` owns query head `h` (read from
// the joint q|gate projection, write `q` and `gate`), block `h >= n_heads` owns
// key head `h - n_heads` (in place in `k`).
//
// Every stage is the arithmetic of the kernel it replaces, element for element:
//
//   * the sum of squares is the same `__fmaf_rn` chain, in the same thread
//     order and the same shared-memory tree, with block size 256 and one
//     element per thread for a 128-wide head;
//   * the normalised value is `x * scale * w[d]`, the same two multiplies;
//   * the rotation is the same NeoX pair over the same doubles.
//
// So the fused pass is bit-identical to the five launches it replaces: the only
// intermediate it drops is the `qg -> q` copy, which is exact.
//
// One warp of 32 lanes cannot cover a 128-wide head here because the norm needs
// a block-wide reduction first; the block is 256 threads like `head_rmsnorm_f32`.

#include <hip/hip_runtime.h>

#define AF_BLOCK 256
#define AF_MAXD 256

extern "C" __global__ void attn_front_f32(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ gate,
    float* k,
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    int n_heads,
    int n_kv,
    int head_dim,
    int n_rot,
    int s0,
    int s1,
    int s2,
    int s3,
    const int* __restrict__ pos,
    float theta_base,
    float eps)
{
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const bool is_q = h < n_heads;
    const int hr = is_q ? h : h - n_heads;
    if (hr >= (is_q ? n_heads : n_kv) || head_dim > AF_MAXD) return;

    const float* src = is_q ? qg + (size_t)hr * (size_t)(2 * head_dim)
                            : k + (size_t)hr * (size_t)head_dim;
    float* dst = is_q ? q + (size_t)hr * (size_t)head_dim
                      : k + (size_t)hr * (size_t)head_dim;
    const float* w = is_q ? q_norm : k_norm;

    // The gate half of the joint projection is a straight copy, and it reads
    // the same cache line the norm is about to want: do it first.
    if (is_q) {
        float* gd = gate + (size_t)hr * (size_t)head_dim;
        for (int d = t; d < head_dim; d += AF_BLOCK) {
            gd[d] = src[head_dim + d];
        }
    }

    float acc = 0.0f;
    for (int d = t; d < head_dim; d += AF_BLOCK) {
        acc = __fmaf_rn(src[d], src[d], acc);
    }

    __shared__ float red[AF_BLOCK];
    __shared__ float buf[AF_MAXD];
    __shared__ float inv;
    red[t] = acc;
    __syncthreads();
    for (int off = AF_BLOCK / 2; off > 0; off >>= 1) {
        if ((int)t < off) red[t] += red[t + off];
        __syncthreads();
    }
    if (t == 0) {
        const float mean_sq = red[0] / (float)head_dim;
        inv = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float scale = inv;
    for (int d = t; d < head_dim; d += AF_BLOCK) {
        buf[d] = src[d] * scale * w[d];
    }
    __syncthreads();

    // M-RoPE (GGML_ROPE_TYPE_MROPE) over the first `n_rot` channels.
    const int p0 = pos[0];
    const int p1 = pos[1];
    const int p2 = pos[2];
    const int p3 = pos[3];
    const int half = n_rot >> 1;
    const int sect = s0 + s1 + s2 + s3;

    for (int c = t; c < half; c += AF_BLOCK) {
        int p = p0;
        if (sect > 0) {
            const int s = c % sect;
            if (s < s0) {
                p = p0;
            } else if (s < s0 + s1) {
                p = p1;
            } else if (s < s0 + s1 + s2) {
                p = p2;
            } else {
                p = p3;
            }
        }
        const double freq = pow((double)theta_base, -2.0 * (double)c / (double)n_rot);
        const double ang = (double)p * freq;
        double sn, cs;
        sincos(ang, &sn, &cs);

        const float x1 = buf[c];
        const float x2 = buf[c + half];
        dst[c] = x1 * (float)cs - x2 * (float)sn;
        dst[c + half] = x1 * (float)sn + x2 * (float)cs;
    }
    for (int d = n_rot + t; d < head_dim; d += AF_BLOCK) {
        dst[d] = buf[d];
    }
}
