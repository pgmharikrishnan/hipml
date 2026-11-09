// Skinny quantized GEMM on the matrix unit: R <= 16 activation rows against a
// K x N quantized weight, `y[r, :] = W @ x[r, :]`.
//
// Why this kernel exists: the f32-from-LDS batch GEMV multiplies every column
// separately, so its cost grows with R (1.90x at 4 columns, 2.95x at 8 while
// it is still streaming the same 17 GB). Padding R to one 16-wide WMMA tile
// makes the columns nearly free: the weight stream is read once and the matrix
// unit does the column fan-out, so the step cost is the tile, not R.
//
// The layout fact this rests on: in `wmma_f32_16x16x16_f16` (wave32, gfx12) a
// lane's B fragment element e is weight row `lane % 16`, k = `8*(lane/16) + e`
// — eight consecutive k of one row, exactly one Q4_K sub-block row run, which
// is what `pf_dequant8` produces. So a lane dequantises its own 8 values into
// its B fragment, and the same 8 values serve all 16 token rows.
//
// Per block: PWS_RT output rows, staged as fp16 in shared one PWS_KW-wide k
// tile at a time; warp w owns output rows [16w, 16w+16) and one 16x16
// accumulator for the 16 (padded) token rows. Per k16 step a warp issues two
// LDS.128 (one A fragment, one B fragment) and one WMMA. Weight bytes read are
// independent of R; the only R-dependent work is the token-row staging and the
// epilogue.
//
// Staging is the whole cost, so it is written for the memory system: four
// threads per row, eight values each, and the eight payload bytes fetched with
// the widest load the address alignment allows (one 64-bit load for Q4_K). As
// eight scalar byte loads a warp instruction touched eight rows with 4 of 32
// bytes used and the kernel ran at 32 GB/s; see `pws_load8_bytes`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "prefill_common.inc"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "gemm_skinny: the gfx12 WMMA intrinsics are wave32 only"
#endif

// Shipped geometry. Every value is a template argument so the sweep entries
// below can be one build; the defaults are the measured best.
#ifndef PWS_RT
#define PWS_RT 32      // output rows per block
#endif
#ifndef PWS_KW
#define PWS_KW 256     // k values per shared tile
#endif
#ifndef PWS_THREADS
#define PWS_THREADS 128
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) pws_h8;
typedef float __attribute__((ext_vector_type(8))) pws_f8;

// The eight quant payload bytes of one row run, fetched with the widest load the
// address alignment allows.
//
// This is the difference between 32 GB/s and the memory system's number: as
// eight scalar byte loads, one warp instruction touches eight rows x 4 of 32
// bytes, so a block issues 8x too many sector requests and every one is mostly
// waste. As one 64-bit load per eight values the same instruction requests
// eight sectors, each fully used.
static __device__ __forceinline__ void pws_load8_bytes(const unsigned char* __restrict__ p,
                                                       unsigned int* w)
{
    const size_t a = (size_t)p;
    if ((a & 7) == 0) {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        w[0] = v.x;
        w[1] = v.y;
    } else if ((a & 3) == 0) {
        const unsigned int* d = reinterpret_cast<const unsigned int*>(p);
        w[0] = d[0];
        w[1] = d[1];
    } else if ((a & 1) == 0) {
        const unsigned short* s = reinterpret_cast<const unsigned short*>(p);
        w[0] = (unsigned int)s[0] | ((unsigned int)s[1] << 16);
        w[1] = (unsigned int)s[2] | ((unsigned int)s[3] << 16);
    } else {
        w[0] = 0;
        w[1] = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) w[i >> 2] |= (unsigned int)p[i] << (8 * (i & 3));
    }
}

// `pf_dequant8` with wide payload loads: same arithmetic, same block layouts,
// eight contiguous values per lane. Only the load width differs, which is why
// the parity cases compare it to the same CPU reference.
template <int KIND>
static __device__ __forceinline__ void pws_dequant8(const unsigned char* __restrict__ rp,
                                                    int v, float* out)
{
    unsigned int w0[2];
    unsigned int w1[2];
    if (KIND == 0) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int sub = within >> 5;
        const int part = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q4K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
        int sc, mn;
        pf_scale_min(sub, b + 4, &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;
        const bool high = (sub & 1) != 0;
        pws_load8_bytes(b + 16 + (sub >> 1) * 32 + part * 8, w0);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int byte = (int)((w0[i >> 2] >> (8 * (i & 3))) & 0xFF);
            const int q = high ? (byte >> 4) : (byte & 0xF);
            out[i] = scale * (float)q - minv;
        }
    } else if (KIND == 1) {
        const int blk = v >> 8;
        const int within = v & 255;
        const int j = within >> 4;
        const int l0 = within & 15;
        const unsigned char* b = rp + (size_t)blk * PF_Q6K_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        const float scale = d * (float)(signed char)b[192 + j];
        const int hf = j >> 3;
        const int s = j & 7;
        const int group = s >> 1;
        const int is = s & 1;
        const int ql_off = hf * 64 + (group & 1) * 32 + is * 16 + l0;
        const int qh_off = 128 + hf * 32 + is * 16 + l0;
        const int shift = 2 * group;
        const bool high = group >= 2;
        pws_load8_bytes(b + ql_off, w0);
        pws_load8_bytes(b + qh_off, w1);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int qb = (int)((w0[i >> 2] >> (8 * (i & 3))) & 0xFF);
            const int hb = ((int)((w1[i >> 2] >> (8 * (i & 3))) & 0xFF) >> shift) & 3;
            const int nib = high ? (qb >> 4) : (qb & 0xF);
            out[i] = scale * (float)((nib | (hb << 4)) - 32);
        }
    } else {
        const int blk = v >> 5;
        const int part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q8_0_BLOCK_BYTES;
        const float d = __half2float(*reinterpret_cast<const __half*>(b));
        pws_load8_bytes(b + 2 + part * 8, w0);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int q = (int)((w0[i >> 2] >> (8 * (i & 3))) & 0xFF);
            out[i] = d * (float)(signed char)q;
        }
    }
}

// PROBE splits the block's cost for the bench: 0 runs everything, 1 skips the
// staging stores (fragment reads still happen), 2 skips the fragment reads and
// the WMMA. Probes compute garbage; they exist only to name the bottleneck.
template <int KIND, int RT, int KW, int THR, int PROBE = 0>
static __device__ __forceinline__ void pws_body(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    constexpr int XS = KW + 8;              // fp16 row stride, 16 B aligned
    constexpr int WS = KW + 8;
    constexpr int NW = RT / 16;             // warp n-tiles (= compute warps)
    constexpr int XROW = KW / 4;            // threads per token row (float4)
    constexpr int XRP = THR / XROW;         // token rows staged per pass
    // Weight staging: one warp reads one row's k-tile as one contiguous run,
    // 8 values per lane. GPR is how many lanes cover a row (KW/8), so a warp
    // covers 32/GPR rows per pass. Coalescing is the reason: 8 rows x 32 B per
    // instruction (the rows-first layout) ran the staging at 24 GB/s, one row x
    // 256 B per instruction is what the GEMV already does at 630 GB/s.
    constexpr int GPR = KW / 8;             // lanes per weight row
    constexpr int RPW = 32 / GPR;           // weight rows per warp per pass
    constexpr int PRP = (THR / 32) * RPW;   // weight rows staged per pass
    static_assert(RT % 16 == 0, "RT must be a multiple of the 16 tile");
    static_assert(KW % 32 == 0 && KW <= 256, "KW must be 32-aligned and <= 256");
    static_assert(KW >= 64 || THR / 32 == 1, "small KW needs one warp per row");
    static_assert(16 % XRP == 0, "token staging must cover 16 rows");
    static_assert(RT % PRP == 0 || RT < PRP, "ragged weight staging");

    __shared__ __align__(16) _Float16 shx[16 * XS];
    __shared__ __align__(16) _Float16 shw[RT * WS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_base = blockIdx.x * RT;
    const size_t wrow = pf_row_bytes(k, KIND);

    // x staging: XROW threads per token row, four floats each.
    const int xt = tid / XROW;
    const int xc = (tid % XROW) * 4;
    // weight staging: lane l covers values [l*8, l*8+8) of its row.
    const int wcol = (lane % GPR) * 8;
    const int wrow_l = warp * RPW + lane / GPR;

    auto stage = [&](int base) {
        if (PROBE == 1) return;
#pragma unroll
        for (int p = 0; p < 16 / XRP; ++p) {
            const int rl = p * XRP + xt;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            // k is a multiple of 32, so a float4 is wholly inside or outside the
            // last (partial) k-tile.
            if (rl < r && base + xc < k)
                v = *reinterpret_cast<const float4*>(x + (size_t)rl * k + base + xc);
            __half2* dst = reinterpret_cast<__half2*>(shx + rl * XS + xc);
            dst[0] = __floats2half2_rn(v.x, v.y);
            dst[1] = __floats2half2_rn(v.z, v.w);
        }
#pragma unroll
        for (int p = 0; p < RT; p += PRP) {
            const int rl = p + wrow_l;
            if (rl >= RT) continue;
            const int gr = row_base + rl;
            const int kc = wcol;
            pws_h8 hv;
            if (gr < n && base + kc < k) {
                float v[8];
                pws_dequant8<KIND>(w + (size_t)gr * wrow, base + kc, v);
#pragma unroll
                for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) hv[i] = (_Float16)0.0f;
            }
            *reinterpret_cast<pws_h8*>(shw + rl * WS + kc) = hv;
        }
    };

    // Independent WMMA chains. A single accumulator per warp runs the matrix
    // unit at 1/8 of what the prefill kernel gets, because consecutive WMMA
    // into the same accumulator are a dependency chain. Rotating the k16 steps
    // through NACC accumulators costs one fold at the end and gives NACC
    // instructions in flight.
    constexpr int NACC = (KW / 16) < 8 ? (KW / 16) : 8;
    pws_f8 acc[NACC];
#pragma unroll
    for (int t = 0; t < NACC; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[t][e] = 0.0f;

    // Fragment offsets: A row is the token row, B row is the output row.
    const int frow = lane & 15;
    const int fcol = (lane >> 4) * 8;
    const int xoff = frow * XS + fcol;
    const int boff = (warp * 16 + frow) * WS + fcol;

    for (int base = 0; base < k; base += KW) {
        __syncthreads(); // the previous multiply has finished with the tiles
        stage(base);
        __syncthreads();
        if (warp < NW) {
            if (PROBE == 2) continue;
#pragma unroll
            for (int t = 0; t < KW / 16; ++t) {
                const int kk = t * 16;
                const pws_h8 a =
                    *reinterpret_cast<const pws_h8*>(shx + xoff + kk);
                const pws_h8 b =
                    *reinterpret_cast<const pws_h8*>(shw + boff + kk);
                acc[t % NACC] =
                    __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[t % NACC]);
            }
        }
    }

    // Fold the k-split accumulators into one (the reduction order differs from
    // a single chain; that is the one numerics difference of this kernel).
#pragma unroll
    for (int t = 1; t < NACC; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[0][e] += acc[t][e];

    // Accumulator -> f32: this lane holds column (output row) `lane & 15` of
    // token rows 8*(lane>>4) + e.
    if (warp < NW) {
        const int gr = row_base + warp * 16 + (lane & 15);
        if (gr < n) {
            const int t0 = 8 * (lane >> 4);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = t0 + e;
                if (t < r) y[(size_t)t * n + gr] = acc[0][e];
            }
        }
    }
}

// The shipped entries: PWS_RT x PWS_KW, one per resident block format.
extern "C" __global__ void gemm_skinny_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    pws_body<0, PWS_RT, PWS_KW, PWS_THREADS>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q6_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    pws_body<1, PWS_RT, PWS_KW, PWS_THREADS>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q8_0(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int n,
    int k,
    int r)
{
    pws_body<2, PWS_RT, PWS_KW, PWS_THREADS>(w, x, y, n, k, r);
}

// Geometry sweep on the Q4_K entry: same body, one build. Each is a manifest
// row (`RowTile { rows = RT, block = THR }`) and runs the same parity case, so
// a sweep cannot pick a geometry that no test covers. The winning geometry is
// baked into PWS_* and the rest are removed.
extern "C" __global__ void gemm_skinny_q4_kg1( // RT 16, KW 256
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 16, 256, 128>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg2( // RT 32, KW 256, 8 warps
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 32, 256, 256>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg3( // RT 32, KW 128
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 32, 128, 128>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg4( // RT 64, KW 256
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 64, 256, 128>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg5( // RT 64, KW 256, 8 warps
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 64, 256, 256>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_kg6( // RT 128, KW 256, 8 warps
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 128, 256, 256>(w, x, y, n, k, r);
}

// Cost probes at the shipped geometry: `_ns` skips the staging stores, `_nc`
// skips the fragment reads and the WMMA. Garbage results, timing only.
extern "C" __global__ void gemm_skinny_q4_ns(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 32, 256, 128, 1>(w, x, y, n, k, r);
}

extern "C" __global__ void gemm_skinny_q4_nc(
    const unsigned char* __restrict__ w, const float* __restrict__ x,
    float* __restrict__ y, int n, int k, int r)
{
    pws_body<0, 32, 256, 128, 2>(w, x, y, n, k, r);
}
