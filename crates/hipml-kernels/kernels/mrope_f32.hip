// M-RoPE (GGML_ROPE_TYPE_MROPE) over the first `n_rot` channels of each head.
//
// NeoX pair `k` is `(k, k + n_rot/2)`. Pair `k` takes its position id from
// section `k % (s0+s1+s2+s3)`: temporal for [0,s0), height for [s0,s0+s1),
// width for the next s2, extra after that. For text all four ids are the token
// position, so this degenerates to plain NeoX rope; the mapping is kept so a
// vision batch uses the same kernel.
//
// Angles are built in double, matching ops::rope_multi_f32 (and rope_neox_f32):
// at 100K positions an f32 argument differs from the reference libm.
//
// One warp per head; x and y may alias (each thread owns both halves of a pair).
//
// The four position ids come from device memory (`pos[0..3]`) so one captured
// graph can be replayed at every step: the token's position is the only argument
// that changes between them.

#include <hip/hip_runtime.h>

extern "C" __global__ void mrope_f32(
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
    const int p0 = pos[0];
    const int p1 = pos[1];
    const int p2 = pos[2];
    const int p3 = pos[3];

    const int half = n_rot >> 1;
    const int sect = s0 + s1 + s2 + s3;
    const size_t base = (size_t)h * (size_t)head_dim;

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
