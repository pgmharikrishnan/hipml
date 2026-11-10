// SPDX-License-Identifier: Apache-2.0
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
        // Angle in double, matching the reference. In f32 the argument
        // reduction for angles up to ~37 rad diverges between libms by ~7e-5.
        const double freq = pow((double)theta_base, -2.0 * (double)i / (double)head_dim);
        const double ang = (double)pos * freq;
        double sn, cs;
        sincos(ang, &sn, &cs);

        const float x1 = x[base + i];
        const float x2 = x[base + half + i];
        y[base + i] = x1 * (float)cs - x2 * (float)sn;
        y[base + half + i] = x1 * (float)sn + x2 * (float)cs;
    }
}
