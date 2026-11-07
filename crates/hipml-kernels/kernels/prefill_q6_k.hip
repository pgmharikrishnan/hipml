// Tiled prefill GEMM over Q6_K weights. Structure identical to prefill_q4_k:
// one block owns 256 tokens x 64 output rows, the weight tile is dequantised
// once into shared and reused for every token, and the f32 accumulators stay in
// registers (8 tokens x 8 rows per thread, 256 threads, 32-value k-tiles).
//
// Q6_K super-block (210 bytes, 256 values): ql(128) qh(64) scales(16) d(2).
// A thread's 8 contiguous values are half of one 16-value sub-block, so one
// int8 scale covers them: 8 ql bytes and 8 qh bytes.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PF6_BLOCK_BYTES 210
#define PF6_TPB 256
#define PF6_RPB 64
#define PF6_KT 32
#define PF6_TT 8
#define PF6_RR 8
#define PF6_TG (PF6_TPB / PF6_TT)
#define PF6_RG (PF6_RPB / PF6_RR)
#define PF6_THREADS (PF6_TG * PF6_RG)
#define PF6_XSTRIDE (PF6_KT + 1)
#define PF6_WSTRIDE (PF6_KT + 1)
#define PF6_HALF_TOKENS (PF6_TPB / 2)

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8) into `out`.
static __device__ __forceinline__ void pf6_dequant8(const unsigned char* __restrict__ rp,
                                                    int v, float* out)
{
    const int blk = v >> 8;
    const int within = v & 255;
    const int j = within >> 4; // 16-value sub-block, and the scale index
    const int l0 = within & 15; // 8-value half of that sub-block

    const unsigned char* b = rp + (size_t)blk * PF6_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
    const float scale =
        d * (float)reinterpret_cast<const signed char*>(b + 192)[j];

    const int half = j >> 3;
    const int s = j & 7;
    const int group = s >> 1;
    const int is = s & 1;
    const int ql_off = half * 64 + (group & 1) * 32 + is * 16 + l0;
    const int qh_off = 128 + half * 32 + is * 16 + l0;
    const int shift = 2 * group;
    const bool high = group >= 2;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int qb = b[ql_off + i];
        const int hb = (b[qh_off + i] >> shift) & 3;
        const int nib = high ? (qb >> 4) : (qb & 0xF);
        out[i] = scale * (float)((nib | (hb << 4)) - 32);
    }
}

extern "C" __global__ void prefill_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pf6_x[PF6_TPB * PF6_XSTRIDE];
    __shared__ __align__(16) float pf6_w[PF6_RPB * PF6_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PF6_TG;
    const int rg = tid / PF6_TG;
    const int tok_base = blockIdx.y * PF6_TPB;
    const int row_base = blockIdx.x * PF6_RPB;
    const int nblocks = k / 256;

    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    float acc[PF6_TT][PF6_RR];
#pragma unroll
    for (int tt = 0; tt < PF6_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PF6_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PF6_TPB * (PF6_KT / 4) / PF6_THREADS;

    for (int base = 0; base < k; base += PF6_KT) {
        __syncthreads();

        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PF6_BLOCK_BYTES;
                float v[8];
                pf6_dequant8(rp, base + d_q * 8, v);
                float* dst = pf6_w + d_row * PF6_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PF6_THREADS;
            const int tl = i >> 3;
            const int c4 = (i & 7) * 4;
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pf6_x + tl * PF6_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PF6_XSTRIDE;
        const int w0 = rg * PF6_RR * PF6_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PF6_KT; ++kk) {
            float xv[PF6_TT];
            float wv[PF6_RR];
#pragma unroll
            for (int tt = 0; tt < PF6_TT; ++tt)
                xv[tt] = pf6_x[x0 + tt * PF6_TG * PF6_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PF6_RR; ++rr)
                wv[rr] = pf6_w[w0 + rr * PF6_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PF6_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PF6_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    // Two 128-token halves so the x buffer can hold the transposed output and
    // the global stores stay coalesced along `row`.
    float* sbuf = pf6_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PF6_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PF6_RR; ++rr)
                sbuf[(tg + PF6_TG * u) * PF6_RPB + (rg * PF6_RR + rr)] =
                    acc[half * (PF6_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PF6_HALF_TOKENS * PF6_RPB; i += PF6_THREADS) {
            const int tl = i / PF6_RPB;
            const int rl = i - tl * PF6_RPB;
            const int gt = tok_base + half * PF6_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
