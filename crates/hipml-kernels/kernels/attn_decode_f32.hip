// Decode-time attention over the whole KV cache, online softmax.
//
// One block per query head; block width equals head_dim, so every thread owns
// one component of the query and of the accumulator. The score for a cache row
// is a block-wide reduction, which is the cost this kernel trades for keeping
// the cache streamed exactly once.
//
// GQA is a pointer offset: n_heads / n_kv query heads share one KV head.

#include <hip/hip_runtime.h>

#define MAX_BLOCK 256

extern "C" __global__ void attn_decode_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    int n_heads,
    int n_kv,
    int head_dim,
    int seq,
    float scale)
{
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    const int group = n_heads / n_kv;
    const int kvh = h / group;

    __shared__ float red[MAX_BLOCK];
    __shared__ float run_max;
    __shared__ float run_sum;

    const float qv = q[(size_t)h * head_dim + t];
    if (t == 0) {
        run_max = -3.402823466e+38f;
        run_sum = 0.0f;
    }
    __syncthreads();

    const float* kb = k + (size_t)kvh * seq * head_dim;
    const float* vb = v + (size_t)kvh * seq * head_dim;

    float acc = 0.0f;

    for (int row = 0; row < seq; ++row) {
        const float* kp = kb + (size_t)row * head_dim;
        red[t] = qv * kp[t];
        __syncthreads();
        for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
            if (t < off) red[t] += red[t + off];
            __syncthreads();
        }

        // All threads compute the same scalars, so the online update stays
        // consistent without another reduction.
        const float score = red[0] * scale;
        const float m_old = run_max;
        const float m_new = fmaxf(m_old, score);
        const float corr = expf(m_old - m_new);
        const float p = expf(score - m_new);

        if (t == 0) {
            run_max = m_new;
            run_sum = run_sum * corr + p;
        }
        const float* vp = vb + (size_t)row * head_dim;
        acc = acc * corr + p * vp[t];
        __syncthreads();
    }

    out[(size_t)h * head_dim + t] = acc / run_sum;
}
