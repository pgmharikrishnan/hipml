// The attention front end over `rows` verify columns, one launch per stage.
//
// These are the same kernels as `qg_split_f32`, `head_rmsnorm_f32` and
// `mrope_f32` with a row index folded into the block index. The arithmetic per
// (row, head) is unchanged; only the position, which the single-row path read
// from the device step buffer, becomes an argument (`pos0 + row`) so the verify
// path issues no per-layer uploads.

#include <hip/hip_runtime.h>

extern "C" __global__ void qg_split_cols_f32(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ gate,
    int n_heads,
    int head_dim,
    int rows)
{
    const int row = blockIdx.x / n_heads;
    const int h = blockIdx.x - row * n_heads;
    if (row >= rows) return;

    const float* src =
        qg + ((size_t)row * n_heads + h) * (size_t)(2 * head_dim);
    float* qd = q + ((size_t)row * n_heads + h) * (size_t)head_dim;
    float* gd = gate + ((size_t)row * n_heads + h) * (size_t)head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        qd[d] = src[d];
        gd[d] = src[head_dim + d];
    }
}

#define HF_BLOCK 256

extern "C" __global__ void head_rmsnorm_cols_f32(
    const float* x,
    const float* __restrict__ w,
    float* y,
    int n_heads,
    int head_dim,
    float eps,
    int rows)
{
    const int row = blockIdx.x / n_heads;
    const int h = blockIdx.x - row * n_heads;
    if (row >= rows) return;

    const float* xr = x + ((size_t)row * n_heads + h) * (size_t)head_dim;
    float* yr = y + ((size_t)row * n_heads + h) * (size_t)head_dim;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += HF_BLOCK) {
        acc = __fmaf_rn(xr[d], xr[d], acc);
    }

    __shared__ float red[HF_BLOCK];
    __shared__ float inv;
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int off = HF_BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float mean_sq = red[0] / (float)head_dim;
        inv = 1.0f / sqrtf(mean_sq + eps);
    }
    __syncthreads();

    const float scale = inv;
    for (int d = threadIdx.x; d < head_dim; d += HF_BLOCK) {
        yr[d] = xr[d] * scale * w[d];
    }
}

// M-RoPE on every head of every row. All four section positions are the token's
// position for text, exactly as the single-row launch writes them into the step
// buffer.
extern "C" __global__ void mrope_cols_f32(
    const float* x,
    float* y,
    int n_heads,
    int head_dim,
    int n_rot,
    int s0,
    int s1,
    int s2,
    int s3,
    int pos0,
    float theta_base,
    int rows)
{
    const int row = blockIdx.x / n_heads;
    const int h = blockIdx.x - row * n_heads;
    if (row >= rows) return;
    const int p0 = pos0 + row;
    const int p1 = p0;
    const int p2 = p0;
    const int p3 = p0;

    const int half = n_rot >> 1;
    const int sect = s0 + s1 + s2 + s3;
    const size_t base = ((size_t)row * n_heads + h) * (size_t)head_dim;

    for (int k = threadIdx.x; k < half; k += blockDim.x) {
        int p = p0;
        if (sect > 0) {
            const int s = k % sect;
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
        const double freq = pow((double)theta_base, -2.0 * (double)k / (double)n_rot);
        const double ang = (double)p * freq;
        double sn, cs;
        sincos(ang, &sn, &cs);

        const float x1 = x[base + k];
        const float x2 = x[base + half + k];
        y[base + k] = x1 * (float)cs - x2 * (float)sn;
        y[base + half + k] = x1 * (float)sn + x2 * (float)cs;
    }
}
