// NeoX (half-split) rotary embedding at one position.
//
// One warp per head; thread t handles rotations t, t+32, ... below half.

#include <hip/hip_runtime.h>

extern "C" __global__ void rope_neox_f32(
    const float* __restrict__ x,
    float* __restrict__ y,
    int n_heads,
    int head_dim,
    int pos,
    float theta_base)
{
    const int head = blockIdx.x;
    if (head >= n_heads) return;

    const int half = head_dim / 2;
    const size_t base = (size_t)head * (size_t)head_dim;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float freq = powf(theta_base, -2.0f * (float)i / (float)head_dim);
        const float ang = (float)pos * freq;
        float sn, cs;
        sincosf(ang, &sn, &cs);

        const float x1 = x[base + i];
        const float x2 = x[base + half + i];
        y[base + i] = x1 * cs - x2 * sn;
        y[base + half + i] = x1 * sn + x2 * cs;
    }
}
