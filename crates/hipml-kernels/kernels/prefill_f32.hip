// Tiled prefill GEMM over raw f32 weights (the SSM alpha/beta pair).
//
// y[t, row] = W[row, :] . x[t, :] for `tokens` rows of x at once. This is
// `prefill_q4_k` with the weight tile copied straight into shared instead of
// dequantised, for the two projections the loader keeps as f32. The previous
// path, `gemm_rows_f32`, is warp-per-(token,row): correct, but it reads each
// weight row once per token, and that redundant stream cost 5% of a 100K
// prefill once its grid was fixed. Staging the tile once serves all 256 tokens.
//
// Geometry matches `prefill_q4_k` (see the manifest's KernelKind::Prefill):
//   PF32_TPB 256 tokens per block, PF32_RPB 64 output rows, PF32_KT 32 k per
//   tile. RR=2 rows per thread gives 1024 threads: 32 token groups x 32 row
//   groups; each thread owns 8 tokens x 2 rows.

#include <hip/hip_runtime.h>

#define PF32_TPB 256
#define PF32_RPB 64
#define PF32_KT 32
#define PF32_TT 8
#define PF32_RR 2
#define PF32_TG (PF32_TPB / PF32_TT)
#define PF32_RG (PF32_RPB / PF32_RR)
#define PF32_THREADS (PF32_TG * PF32_RG)
#define PF32_XSTRIDE (PF32_KT + 1)
#define PF32_WSTRIDE (PF32_KT + 1)
#define PF32_HALF_TOKENS (PF32_TPB / 2)

extern "C" __global__ void prefill_f32(
    const float* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pf32_x[PF32_TPB * PF32_XSTRIDE];
    __shared__ __align__(16) float pf32_w[PF32_RPB * PF32_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PF32_TG;   // token group: tokens tg + 32*tt
    const int rg = tid / PF32_TG;   // row group: rows rg*8 .. rg*8+7
    const int tok_base = blockIdx.y * PF32_TPB;
    const int row_base = blockIdx.x * PF32_RPB;

    const int d_row = tid >> 2;    // 64 rows
    const int d_q = tid & 3;       // 4 chunks of 8 values per 32-value k-tile

    float acc[PF32_TT][PF32_RR];
#pragma unroll
    for (int tt = 0; tt < PF32_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PF32_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PF32_TPB * (PF32_KT / 4) / PF32_THREADS; // 8 float4 per thread

    for (int base = 0; base < k; base += PF32_KT) {
        __syncthreads(); // previous multiply finished before shared is reused

        // Weight tile: 64 rows x 32 values, copied once for all tokens.
        {
            const int r = row_base + d_row;
            if (r < m) {
                const float* rp = w + (size_t)r * (size_t)k + base + d_q * 8;
                float4 a = *reinterpret_cast<const float4*>(rp);
                float4 b = *reinterpret_cast<const float4*>(rp + 4);
                float* dst = pf32_w + d_row * PF32_WSTRIDE + d_q * 8;
                dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
                dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
            }
        }

        // x tile: 256 tokens x 32 values, token-major.
        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PF32_THREADS;
            const int tl = i >> 3;         // token within the block
            const int c4 = (i & 7) * 4;    // k offset, multiple of 4
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pf32_x + tl * PF32_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PF32_XSTRIDE;
        const int w0 = rg * PF32_RR * PF32_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PF32_KT; ++kk) {
            float xv[PF32_TT];
            float wv[PF32_RR];
#pragma unroll
            for (int tt = 0; tt < PF32_TT; ++tt)
                xv[tt] = pf32_x[x0 + tt * PF32_TG * PF32_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PF32_RR; ++rr)
                wv[rr] = pf32_w[w0 + rr * PF32_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PF32_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PF32_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    // Epilogue in two 128-token halves, coalesced along `row` (y is row-major).
    float* sbuf = pf32_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PF32_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PF32_RR; ++rr)
                sbuf[(tg + PF32_TG * u) * PF32_RPB + (rg * PF32_RR + rr)] =
                    acc[half * (PF32_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PF32_HALF_TOKENS * PF32_RPB; i += PF32_THREADS) {
            const int tl = i / PF32_RPB;
            const int rl = i - tl * PF32_RPB;
            const int gt = tok_base + half * PF32_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
