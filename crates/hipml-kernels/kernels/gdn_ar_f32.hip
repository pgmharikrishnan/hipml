// SPDX-License-Identifier: Apache-2.0
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
// GDN_ROWS rows of a head are in flight at once. Rows are independent, so this
// is a scheduling change only: every row's dot is the same fma chain in the
// same order and the same butterfly, and its update is the same expression.
// With one row in flight the wave chain is load -> reduce -> fma -> reduce ->
// store per row, and the state row it waits on is a full memory latency away;
// four rows give the wave four independent latencies to hide behind.
//
// The grid's y axis splits one head's value rows into `gridDim.y` stripes, so a
// single value head can fill more than one CU. Rows never interact, so the
// split changes which block runs a row and nothing about what it computes.
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
// Independent state rows one wave keeps in flight.
#define GDN_ROWS 4

__device__ __forceinline__ float wave_sum(float v)
{
    // Butterfly (xor) reduction: every lane ends with the same total. A
    // down-shuffle would leave it only in lane 0, but every lane needs `dot`
    // to update its slice of the state row.
    for (int off = GDN_LANES / 2; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, GDN_LANES);
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

    // This block owns value rows [lo, hi) of head h.
    const int stripe = gridDim.y;
    const int seg = (sd + stripe - 1) / stripe;
    const int lo = blockIdx.y * seg;
    const int hi = min(sd, lo + seg);

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

    for (int j0 = lo + wave; j0 < hi; j0 += GDN_ROWS * nwave) {
        // Row index of slot u, uniform across the wave.
        int jr[GDN_ROWS];
        float rows[GDN_ROWS][GDN_MAXV];
        float dot[GDN_ROWS];
        #pragma unroll
        for (int u = 0; u < GDN_ROWS; ++u) {
            jr[u] = j0 + u * nwave;
        }

        // Load every row first, then reduce: the loads are independent, so the
        // wave has GDN_ROWS memory latencies in flight instead of one.
        #pragma unroll
        for (int u = 0; u < GDN_ROWS; ++u) {
            dot[u] = 0.0f;
            const bool on = jr[u] < hi;
            const float* row = head + (size_t)jr[u] * sd;
            #pragma unroll
            for (int t = 0; t < GDN_MAXV; ++t) {
                const int i = lane + t * GDN_LANES;
                rows[u][t] = (on && i < sd) ? row[i] : 0.0f;
                // Explicit fma so the batched kernel compiles to the same
                // rounding; contraction is otherwise per-file.
                dot[u] = __fmaf_rn(__fmul_rn(rows[u][t], decay), kk[t], dot[u]);
            }
        }
        #pragma unroll
        for (int u = 0; u < GDN_ROWS; ++u) {
            dot[u] = wave_sum(dot[u]);
        }

        #pragma unroll
        for (int u = 0; u < GDN_ROWS; ++u) {
            if (jr[u] >= hi) continue;
            const float delta = (vh[jr[u]] - dot[u]) * b;
            float* row = head + (size_t)jr[u] * sd;

            float acc = 0.0f;
            #pragma unroll
            for (int t = 0; t < GDN_MAXV; ++t) {
                const int i = lane + t * GDN_LANES;
                if (i < sd) {
                    const float nv =
                        __fmaf_rn(rows[u][t], decay, __fmul_rn(kk[t], delta));
                    row[i] = nv;
                    acc = __fmaf_rn(nv, qq[t], acc);
                }
            }
            acc = wave_sum(acc);

            if (lane == 0) {
                out[(size_t)h * sd + jr[u]] = acc * scale;
            }
        }
    }
}
