// Batched row-wise helpers: slot on `blockIdx.y`, the existing per-row axis on
// `blockIdx.x`. Arithmetic is the single-row kernel's, unchanged.

#include <hip/hip_runtime.h>

// --- f32 GEMV over `rows` x-vectors (alpha/beta projections) ---------------

#define BGV_BLOCK 256

extern "C" __global__ void gemv_f32_b(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int x_stride)
{
    const int row = blockIdx.x;
    if (row >= m) return;
    const int slot = blockIdx.y;

    const float* r = w + (size_t)row * (size_t)k;
    const float* xs = x + (size_t)slot * x_stride;
    float acc = 0.0f;
    for (int j = threadIdx.x; j < k; j += BGV_BLOCK) {
        acc += r[j] * xs[j];
    }

    __shared__ float s[BGV_BLOCK];
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = BGV_BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) s[threadIdx.x] += s[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[(size_t)slot * m + row] = s[0];
}

// --- q|gate split ----------------------------------------------------------

extern "C" __global__ void qg_split_b(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ gate,
    int n_heads,
    int head_dim)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int slot = blockIdx.y;
    const size_t src0 = ((size_t)slot * n_heads + h) * (size_t)(2 * head_dim);
    const size_t dst0 = ((size_t)slot * n_heads + h) * (size_t)head_dim;
    const float* src = qg + src0;
    float* qd = q + dst0;
    float* gd = gate + dst0;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        qd[d] = src[d];
        gd[d] = src[head_dim + d];
    }
}

// --- per-head RMS norm -----------------------------------------------------

#define BHR_BLOCK 256

extern "C" __global__ void head_rmsnorm_b(
    const float* x,
    const float* __restrict__ w,
    float* y,
    int n_heads,
    int head_dim,
    float eps,
    int slot_stride)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int slot = blockIdx.y;
    const float* xr = x + (size_t)slot * slot_stride + (size_t)h * (size_t)head_dim;
    float* yr = y + (size_t)slot * slot_stride + (size_t)h * (size_t)head_dim;

    float acc = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += BHR_BLOCK) {
        acc += xr[d] * xr[d];
    }

    __shared__ float red[BHR_BLOCK];
    __shared__ float inv;
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int off = BHR_BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] += red[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        inv = 1.0f / sqrtf(red[0] / (float)head_dim + eps);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += BHR_BLOCK) {
        yr[d] = xr[d] * inv * w[d];
    }
}

// --- M-RoPE with a per-slot position --------------------------------------

extern "C" __global__ void mrope_b(
    const float* x,
    float* y,
    int n_heads,
    int head_dim,
    int n_rot,
    int s0,
    int s1,
    int s2,
    int s3,
    const int* __restrict__ pos,
    float theta_base)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;
    const int slot = blockIdx.y;
    // pos is [slots][2] = (position, seq); text uses the token position for
    // every section.
    const int p0 = pos[slot * 2];
    const int p1 = p0;
    const int p2 = p0;
    const int p3 = p0;

    const int half = n_rot >> 1;
    const int sect = s0 + s1 + s2 + s3;
    const size_t base = ((size_t)slot * n_heads + h) * (size_t)head_dim;

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
