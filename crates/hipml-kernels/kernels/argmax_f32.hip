// Greedy argmax on the device: 248,320 logits in, one index out.
//
// The graph used to copy the whole logit vector back (993 KB) and walk it on
// the host, which costs a blocking copy plus ~0.33 ms of host work per token.
// Two launches replace that: `argmax_part_f32` reduces `parts` contiguous
// slices to (value, index) pairs, `argmax_final_f32` folds those.
//
// Ties resolve to the lowest index, matching `hipml_core::gen::argmax_f32`
// (strict `>` in index order), so greedy ids cannot change.

#include <hip/hip_runtime.h>
#include <math.h>

#define ARGMAX_BLOCK 256

extern "C" __global__ void argmax_part_f32(
    const float* __restrict__ x,
    float* __restrict__ best_v,
    int* __restrict__ best_i,
    int n,
    int parts)
{
    const int b = blockIdx.x;
    const int chunk = (n + parts - 1) / parts;
    const int start = b * chunk;
    const int end = min(n, start + chunk);

    float v = -INFINITY;
    int idx = start;
    for (int i = start + threadIdx.x; i < end; i += ARGMAX_BLOCK) {
        const float xi = x[i];
        if (xi > v) {
            v = xi;
            idx = i;
        }
    }

    __shared__ float sv[ARGMAX_BLOCK];
    __shared__ int si[ARGMAX_BLOCK];
    sv[threadIdx.x] = v;
    si[threadIdx.x] = idx;
    __syncthreads();
    for (int off = ARGMAX_BLOCK / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) {
            const float ov = sv[threadIdx.x + off];
            const int oi = si[threadIdx.x + off];
            if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) {
                sv[threadIdx.x] = ov;
                si[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        best_v[b] = sv[0];
        best_i[b] = si[0];
    }
}

extern "C" __global__ void argmax_final_f32(
    const float* __restrict__ best_v,
    const int* __restrict__ best_i,
    int* __restrict__ out,
    int parts)
{
    if (threadIdx.x != 0) return;
    float v = -INFINITY;
    int idx = 0;
    for (int b = 0; b < parts; ++b) {
        const float vb = best_v[b];
        const int ib = best_i[b];
        if (vb > v || (vb == v && ib < idx)) {
            v = vb;
            idx = ib;
        }
    }
    out[0] = idx;
}
