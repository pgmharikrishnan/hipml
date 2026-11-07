// Tiled prefill GEMM over Q4_K weights.
//
// y[t, row] = W[row, :] . x[t, :] for `tokens` rows of x at once. The whole win
// versus `tokens` GEMVs is that a weight tile is dequantised ONCE into shared
// memory and then multiplied against every token in the block.
//
// Geometry (mirrored by the manifest's KernelKind::Prefill):
//   PF4_TPB 256 tokens per block, PF4_RPB 64 output rows, PF4_KT 32 k per tile.
//   256 threads: 32 token groups x 8 row groups. Each thread owns 8 tokens x
//   8 rows = 64 f32 accumulators. Twelve shared dwords per 32 FMAs is what the
//   4x8 tile could afford; 8x8 cuts that to 16 per 64, which is the shared
//   bandwidth a 128-FMA/cycle CU can actually feed.
//
// The tokens a thread owns are strided (tg, tg+32, ... tg+224), not adjacent.
// With the x tile stored token-major at stride KT+1, the 32 lanes of a warp then
// read 32 different banks instead of 8, which is what makes the inner loop
// conflict-free. Weight rows are read as a broadcast (all lanes of a warp share
// one row group).
//
// Q4_K super-block (144 bytes, 256 values): d(2) dmin(2) scales(12) qs(128).
// Every thread dequantises 8 contiguous values, a quarter of a 32-value
// sub-block, so one d/dmin/scale/min set covers all 8.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PF4_BLOCK_BYTES 144
#define PF4_TPB 256
#define PF4_RPB 64
#define PF4_KT 32
#define PF4_TT 8
#define PF4_RR 8
#define PF4_TG (PF4_TPB / PF4_TT)
#define PF4_RG (PF4_RPB / PF4_RR)
#define PF4_THREADS (PF4_TG * PF4_RG)
#define PF4_XSTRIDE (PF4_KT + 1)
#define PF4_WSTRIDE (PF4_KT + 1)
#define PF4_HALF_TOKENS (PF4_TPB / 2)

static __device__ __forceinline__ void pf4_scale_min(int j, const unsigned char* q,
                                                     int* sc, int* mn)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Dequantise 8 contiguous values of one row, starting at value index `v`
// (a multiple of 8) into `out`.
static __device__ __forceinline__ void pf4_dequant8(const unsigned char* __restrict__ rp,
                                                    int v, float* out)
{
    const int blk = v >> 8;
    const int within = v & 255;
    const int sub = within >> 5;
    const int part = (within & 31) >> 3;

    const unsigned char* b = rp + (size_t)blk * PF4_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    int sc, mn;
    pf4_scale_min(sub, b + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;
    const bool high = (sub & 1) != 0;
    const unsigned char* qp = b + 16 + (sub >> 1) * 32 + part * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = high ? (qp[i] >> 4) : (qp[i] & 0xF);
        out[i] = scale * (float)q - minv;
    }
}

extern "C" __global__ void prefill_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) float pf4_x[PF4_TPB * PF4_XSTRIDE];
    __shared__ __align__(16) float pf4_w[PF4_RPB * PF4_WSTRIDE];

    const int tid = threadIdx.x;
    const int tg = tid % PF4_TG;   // token group: tokens tg + 32*tt
    const int rg = tid / PF4_TG;   // row group: rows rg*8 .. rg*8+7
    const int tok_base = blockIdx.y * PF4_TPB;
    const int row_base = blockIdx.x * PF4_RPB;
    const int nblocks = k / 256;

    const int d_row = tid >> 2;    // 64 rows
    const int d_q = tid & 3;       // 4 chunks of 8 values per 32-value k-tile

    float acc[PF4_TT][PF4_RR];
#pragma unroll
    for (int tt = 0; tt < PF4_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PF4_RR; ++rr) acc[tt][rr] = 0.0f;

    const int xq = PF4_TPB * (PF4_KT / 4) / PF4_THREADS; // 8 float4 per thread

    for (int base = 0; base < k; base += PF4_KT) {
        __syncthreads(); // previous multiply finished before shared is reused

        // Weight tile: 64 rows x 32 values, dequantised once.
        {
            const int r = row_base + d_row;
            if (r < m) {
                const unsigned char* rp =
                    w + (size_t)r * (size_t)nblocks * PF4_BLOCK_BYTES;
                float v[8];
                pf4_dequant8(rp, base + d_q * 8, v);
                float* dst = pf4_w + d_row * PF4_WSTRIDE + d_q * 8;
#pragma unroll
                for (int i = 0; i < 8; ++i) dst[i] = v[i];
            }
        }

        // x tile: 256 tokens x 32 values, token-major.
        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PF4_THREADS;
            const int tl = i >> 3;         // token within the block
            const int c4 = (i & 7) * 4;    // k offset, multiple of 4
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            float* dst = pf4_x + tl * PF4_XSTRIDE + c4;
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        }

        __syncthreads();

        const int x0 = tg * PF4_XSTRIDE;
        const int w0 = rg * PF4_RR * PF4_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PF4_KT; ++kk) {
            float xv[PF4_TT];
            float wv[PF4_RR];
#pragma unroll
            for (int tt = 0; tt < PF4_TT; ++tt)
                xv[tt] = pf4_x[x0 + tt * PF4_TG * PF4_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PF4_RR; ++rr)
                wv[rr] = pf4_w[w0 + rr * PF4_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PF4_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PF4_RR; ++rr)
                    acc[tt][rr] = fmaf(xv[tt], wv[rr], acc[tt][rr]);
        }
    }

    // Epilogue in two 128-token halves so the x buffer (33.8 KB) can hold the
    // transpose of the 8 KB output half. Global stores are then coalesced along
    // `row`: y is row-major [tokens, m], so consecutive rows are consecutive.
    float* sbuf = pf4_x;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PF4_TT / 2; ++u)
#pragma unroll
            for (int rr = 0; rr < PF4_RR; ++rr)
                sbuf[(tg + PF4_TG * u) * PF4_RPB + (rg * PF4_RR + rr)] =
                    acc[half * (PF4_TT / 2) + u][rr];
        __syncthreads();
        for (int i = tid; i < PF4_HALF_TOKENS * PF4_RPB; i += PF4_THREADS) {
            const int tl = i / PF4_RPB;
            const int rl = i - tl * PF4_RPB;
            const int gt = tok_base + half * PF4_HALF_TOKENS + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sbuf[i];
        }
    }
}
