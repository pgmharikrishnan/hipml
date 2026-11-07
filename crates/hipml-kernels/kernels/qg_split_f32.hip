// Split the joint q|gate projection into two [n_heads][head_dim] tensors.
//
// Row layout of attn_q is per head: head_dim query values followed by head_dim
// gate values. One block per head.

#include <hip/hip_runtime.h>

extern "C" __global__ void qg_split_f32(
    const float* __restrict__ qg,
    float* __restrict__ q,
    float* __restrict__ gate,
    int n_heads,
    int head_dim)
{
    const int h = blockIdx.x;
    if (h >= n_heads) return;

    const float* src = qg + (size_t)h * (size_t)(2 * head_dim);
    float* qd = q + (size_t)h * (size_t)head_dim;
    float* gd = gate + (size_t)h * (size_t)head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        qd[d] = src[d];
        gd[d] = src[head_dim + d];
    }
}
