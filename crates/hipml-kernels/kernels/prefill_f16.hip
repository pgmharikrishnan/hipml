// fp16 prefill GEMM over Q4_K / Q6_K / Q8_0 weights.
//
// Same tiling as `prefill_q4_k` (a weight tile is dequantised once into shared
// and reused for every prompt token in the block) with two changes:
//
//   1. shared holds `half` instead of `float`, so an LDS dword carries two k
//      values and the bytes per FMA drop by half;
//   2. the inner product is `v_dot2_f32_f16` (`amd_mixed_dot`): two fp16
//      products per instruction, accumulated in f32. That is the packed-fp16
//      rate the card quotes as 95.7 TFLOPS, twice the fp32 FMA rate, without
//      the accuracy loss of a half accumulator.
//
// Geometry: 256 prompt tokens x 128 output rows per block, 512 threads. A
// thread owns 8 tokens (strided by 32) x 8 rows = 64 f32 accumulators. Per
// 16 half2 k-steps it reads 8 x-half2 + 8 w-half2 (64 B) and issues 64 dot2
// (128 FMAs): 4 FLOP/byte, which matches a 128 B/cycle LDS at the fp16 rate.
// The token stride is `PFH_XSTRIDE = 17` half2 (68 B), coprime with the 32
// banks, so the 32 lanes of a warp hit 32 banks; weight rows are a broadcast.
//
// Q4_K/Q6_K/Q8_0 differ only in how 8 contiguous values are dequantised, so
// the three entry points share one templated body.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define PFH_TPB 256
#define PFH_RPB 128
#define PFH_KT 32
#define PFH_KH2 (PFH_KT / 2)
#define PFH_TT 8
#define PFH_RR 8
#define PFH_TG (PFH_TPB / PFH_TT)
#define PFH_RG (PFH_RPB / PFH_RR)
#define PFH_THREADS (PFH_TG * PFH_RG)
#define PFH_XSTRIDE (PFH_KH2 + 1)
#define PFH_WSTRIDE (PFH_KH2 + 1)
#define PFH_QUARTER (PFH_TPB / 4)

#define PF4H_BLOCK_BYTES 144
#define PF6H_BLOCK_BYTES 210
#define PF8H_BLOCK_BYTES 34

static __device__ __forceinline__ void pf4h_scale_min(int j, const unsigned char* q,
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
// (a multiple of 8). KIND: 0 Q4_K, 1 Q6_K, 2 Q8_0.
template <int KIND>
static __device__ __forceinline__ void pfh_dequant8(const unsigned char* __restrict__ rp,
                                                    int v, float* out)
{
    if (KIND == 0) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int sub = within >> 5;
        const int part = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF4H_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        int sc, mn;
        pf4h_scale_min(sub, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;
        const bool high = (sub & 1) != 0;
        const unsigned char* qp = b + 16 + (sub >> 1) * 32 + part * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = high ? (qp[i] >> 4) : (qp[i] & 0xF);
            out[i] = scale * (float)q - minv;
        }
    } else if (KIND == 1) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int j = within >> 4;
        const int l0 = within & 15;
        const unsigned char* b = rp + (size_t)blk * PF6H_BLOCK_BYTES;
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
    } else {
        const int blk = v >> 5;
        const int part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF8H_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const signed char* qs = reinterpret_cast<const signed char*>(b + 2 + part * 8);
#pragma unroll
        for (int i = 0; i < 8; ++i) out[i] = d * (float)qs[i];
    }
}

template <int KIND>
static __device__ __forceinline__ void pfh_body(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) __half2 sfh_x[PFH_TPB * PFH_XSTRIDE];
    __shared__ __align__(16) __half2 sfh_w[PFH_RPB * PFH_WSTRIDE];
    // Transposed epilogue, one quarter of the token tile at a time:
    // 64 tokens x 128 rows x 4 B = 32 KiB.
    __shared__ __align__(16) float sfh_o[PFH_QUARTER * PFH_RPB];

    const int tid = threadIdx.x;
    const int tg = tid % PFH_TG;
    const int rg = tid / PFH_TG;
    const int tok_base = blockIdx.y * PFH_TPB;
    const int row_base = blockIdx.x * PFH_RPB;
    // Q4_K/Q6_K pack 256 values per super-block; Q8_0 packs 32.
    const int nblocks = k / (KIND == 2 ? 32 : 256);

    // 128 rows x 32 values = 4096 values per k-tile; 512 threads x 8.
    const int d_row = tid >> 2;
    const int d_q = tid & 3;
    // x tile: 256 tokens x 32 values = 8192 floats; 512 threads x 4 floats,
    // two passes.
    const int xq = PFH_TPB * (PFH_KT / 4) / PFH_THREADS;

    float acc[PFH_TT][PFH_RR];
#pragma unroll
    for (int tt = 0; tt < PFH_TT; ++tt)
#pragma unroll
        for (int rr = 0; rr < PFH_RR; ++rr) acc[tt][rr] = 0.0f;

    for (int base = 0; base < k; base += PFH_KT) {
        __syncthreads(); // previous multiply finished before shared is reused

        {
            const int r = row_base + d_row;
            if (r < m && d_row < PFH_RPB) {
                __half2* dst = sfh_w + d_row * PFH_WSTRIDE + d_q * 4;
                if (KIND == 3) {
                    // Already half (Q5_K, dequantised once when the model was
                    // loaded). Copy straight to shared, no unpack.
                    const __half* hp = reinterpret_cast<const __half*>(w) +
                                       (size_t)r * k + base + d_q * 8;
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        dst[i] = *reinterpret_cast<const __half2*>(hp + 2 * i);
                } else {
                    const unsigned char* rp =
                        w + (size_t)r * (size_t)nblocks *
                                (KIND == 0 ? PF4H_BLOCK_BYTES
                                           : KIND == 1 ? PF6H_BLOCK_BYTES : PF8H_BLOCK_BYTES);
                    float v[8];
                    pfh_dequant8<KIND>(rp, base + d_q * 8, v);
#pragma unroll
                    for (int i = 0; i < 4; ++i)
                        dst[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
                }
            }
        }

        for (int s = 0; s < xq; ++s) {
            const int i = tid + s * PFH_THREADS;
            const int tl = i >> 3;
            const int c4 = (i & 7) * 4;
            const int gt = tok_base + tl;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (gt < tokens) {
                v = *reinterpret_cast<const float4*>(x + (size_t)gt * k + base + c4);
            }
            __half2* dst = sfh_x + tl * PFH_XSTRIDE + c4 / 2;
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }

        __syncthreads();

        const int x0 = tg * PFH_XSTRIDE;
        const int w0 = rg * PFH_RR * PFH_WSTRIDE;
#pragma unroll 2
        for (int kk = 0; kk < PFH_KH2; ++kk) {
            __half2 xv[PFH_TT];
            __half2 wv[PFH_RR];
#pragma unroll
            for (int tt = 0; tt < PFH_TT; ++tt)
                xv[tt] = sfh_x[x0 + tt * PFH_TG * PFH_XSTRIDE + kk];
#pragma unroll
            for (int rr = 0; rr < PFH_RR; ++rr)
                wv[rr] = sfh_w[w0 + rr * PFH_WSTRIDE + kk];
#pragma unroll
            for (int tt = 0; tt < PFH_TT; ++tt)
#pragma unroll
                for (int rr = 0; rr < PFH_RR; ++rr)
                    acc[tt][rr] = amd_mixed_dot(xv[tt], wv[rr], acc[tt][rr], false);
        }
    }

    // Transpose through shared in four 64-token quarters so the global stores
    // are coalesced along `row` (y is row-major [tokens, m]).
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < PFH_TT / 4; ++u)
#pragma unroll
            for (int rr = 0; rr < PFH_RR; ++rr)
                sfh_o[(tg + PFH_TG * u) * PFH_RPB + (rg * PFH_RR + rr)] =
                    acc[q * (PFH_TT / 4) + u][rr];
        __syncthreads();
        for (int i = tid; i < PFH_QUARTER * PFH_RPB; i += PFH_THREADS) {
            const int tl = i / PFH_RPB;
            const int rl = i - tl * PFH_RPB;
            const int gt = tok_base + q * PFH_QUARTER + tl;
            const int gr = row_base + rl;
            if (gt < tokens && gr < m) y[(size_t)gt * m + gr] = sfh_o[i];
        }
    }
}

extern "C" __global__ void prefill_q4_k_h(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pfh_body<0>(w, x, y, m, k, tokens);
}

extern "C" __global__ void prefill_q6_k_h(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pfh_body<1>(w, x, y, m, k, tokens);
}

extern "C" __global__ void prefill_q8_0_h(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pfh_body<2>(w, x, y, m, k, tokens);
}

// Weights already dequantised to half, row-major [m][k]. Used for Q5_K, which
// has no unpack kernel: the loader dequantises it once, and this keeps the
// prefill path's fp16 tile and dot2 inner loop unchanged.
extern "C" __global__ void prefill_halfw_h(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int tokens)
{
    pfh_body<3>(w, x, y, m, k, tokens);
}
