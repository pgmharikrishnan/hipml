// Gated delta net, one token, autoregressive form.
//
// For value head h (key head hk = h % n_k) and every value axis j:
//
//   S[:,j] <- S[:,j] * exp(g[h]) + k * (v[h][j] - S[:,j] . k * exp(g[h])) * beta[h]
//   o[h][j] <- S[:,j] . q * 1/sqrt(sd)
//
// State is `s[h][j][i]`, i (the key axis) innermost, matching
// gated_delta_net_f32.c (`s_out[j*S_v + i] = S[i][j]`).
//
// One block per value head. A 32-lane group owns a value row `j`: each lane
// holds sd/32 elements of that row in registers, and the two dots are group
// reductions. No barrier is needed because rows never overlap.
//
// Kernel: gdn_ar_f32. Reference: delta::gdn_ar_f32.

#include <hip/hip_runtime.h>

// sd <= 128 for qwen35 (state size 128); four values per lane covers 32 lanes.
#define GDN_MAXV 4
// Reduction groups are pinned to 32 lanes. That is one hardware wave on RDNA
// in wave32 mode and one half-wave in wave64 mode, so the same code is correct
// either way; using `warpSize` as the shuffle width is not (it exceeds the
// supported segment width when the compiler emits wave32).
#define GDN_LANES 32

__device__ __forceinline__ float wave_sum(float v)
{
    for (int off = GDN_LANES / 2; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffffffffffULL, v, off, GDN_LANES);
    }
    return v;
}

extern "C" __global__ void gdn_ar_f32(
    float* __restrict__ s,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ out,
    int n_v,
    int n_k,
    int sd)
{
    const int h = blockIdx.x;
    const int lane = threadIdx.x % GDN_LANES;
    const int wave = threadIdx.x / GDN_LANES;
    const int nwave = blockDim.x / GDN_LANES;

    const int hk = h % n_k;
    const float decay = expf(g[h]);
    const float b = beta[h];
    const float scale = rsqrtf((float)sd);

    const float* qh = q + (size_t)hk * sd;
    const float* kh = k + (size_t)hk * sd;
    const float* vh = v + (size_t)h * sd;
    float* head = s + (size_t)h * sd * sd;

    // q and k stay in registers: they are read for every value row.
    float qq[GDN_MAXV];
    float kk[GDN_MAXV];
    #pragma unroll
    for (int t = 0; t < GDN_MAXV; ++t) {
        const int i = lane + t * GDN_LANES;
        if (i < sd) {
            qq[t] = qh[i];
            kk[t] = kh[i];
        } else {
            qq[t] = 0.0f;
            kk[t] = 0.0f;
        }
    }

    for (int j = wave; j < sd; j += nwave) {
        float* row = head + (size_t)j * sd;

        float r[GDN_MAXV];
        float dot = 0.0f;
        #pragma unroll
        for (int t = 0; t < GDN_MAXV; ++t) {
            const int i = lane + t * GDN_LANES;
            r[t] = i < sd ? row[i] : 0.0f;
            dot += r[t] * decay * kk[t];
        }
        dot = wave_sum(dot);

        const float delta = (vh[j] - dot) * b;

        float acc = 0.0f;
        #pragma unroll
        for (int t = 0; t < GDN_MAXV; ++t) {
            const int i = lane + t * GDN_LANES;
            if (i < sd) {
                const float nv = r[t] * decay + kk[t] * delta;
                row[i] = nv;
                acc += nv * qq[t];
            }
        }
        acc = wave_sum(acc);

        if (lane == 0) {
            out[(size_t)h * sd + j] = acc * scale;
        }
    }
}
