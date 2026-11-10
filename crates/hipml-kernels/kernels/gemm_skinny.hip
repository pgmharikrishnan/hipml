// SPDX-License-Identifier: Apache-2.0
// Skinny quantized GEMM on the matrix unit: R <= 16 activation rows against a
// K x N quantized weight, `y[r, :] = W @ x[r, :]`.
//
// Cost curve: padding R to one 16-wide WMMA tile makes the columns nearly free,
// because the weight stream is read once and the matrix unit does the column
// fan-out. Measured on the five production shapes, R=4 costs 0.99x and R=8
// 1.00x of R=1 (see docs/review.md, WP33). That is the whole point: the
// f32-from-LDS batch GEMV grows 1.90x at 4 columns and 2.95x at 8.
//
// It is *not* a throughput win, and it must not be used to replace a GEMV:
// staging the dequantised weight through shared memory runs at ~40-68 GB/s of
// weight against the single-column GEMV's 540-980 GB/s. The existing WMMA
// prefill kernel has the same staging rate and hides it behind 256-token
// compute; at 16 rows there is nothing to hide it behind. See the WP33 section
// of docs/review.md for the probe breakdown.
//
// The layout fact this rests on: in `wmma_f32_16x16x16_f16` (wave32, gfx12) a
// lane's B fragment element e is weight row `lane % 16`, k = `8*(lane/16) + e`
// -- eight consecutive k of one row, exactly one Q4_K sub-block row run, which
// is what `pws_prep`/`pws_finish` produce. So a lane dequantises its own 8
// values into its B fragment, and the same 8 values serve all 16 token rows.
//
// Per block: PWS_RT output rows, staged as fp16 in shared one PWS_KW-wide k
// tile at a time; warp w owns output rows [16w, 16w+16) and one 16x16
// accumulator for the 16 (padded) token rows. Per k16 step a warp issues two
// LDS.128 (one A fragment, one B fragment) and one WMMA. Weight bytes read are
// independent of R; the only R-dependent work is the token-row staging and the
// epilogue.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "prefill_common.inc"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "gemm_skinny: the gfx12 WMMA intrinsics are wave32 only"
#endif

// Shipped geometry: 32 output rows, a 128-value k tile, 4 warps. Measured
// against the alternatives (16/32/48/64 rows, 128/256 k, 4 vs 8 warps) on the
// Q4_K shapes; the best of them is within 1% of this one and the rest are 10%
// to 40% worse. Note `rows` in the manifest must equal PWS_RT: the launch grid
// is what turns a mismatch into silently half-computed output.
#ifndef PWS_RT
#define PWS_RT 32      // output rows per block
#endif
#ifndef PWS_KW
#define PWS_KW 128     // k values per shared tile
#endif
#ifndef PWS_THREADS
#define PWS_THREADS 128
#endif
// Independent WMMA chains per warp. More than a handful buys nothing once
// blocks are resident and costs a register per accumulator element.
#ifndef PWS_NACC
#define PWS_NACC 4
#endif

typedef _Float16 __attribute__((ext_vector_type(8))) pws_h8;
typedef float __attribute__((ext_vector_type(8))) pws_f8;

// Branch-free Q4_K scale/min unpack from the three dwords at b+4, b+8, b+12.
//
// `pf_scale_min` branches on the sub-block index, which is per-lane here (a
// lane's 8 values sit in one sub-block), so the compiler split every group into
// two exec-masked blocks and drained the memory pipe at each boundary: the ISA
// showed 80 `s_wait_loadcnt` in the loop body. Same arithmetic, three dword
// arguments, no divergent branch.
static __device__ __forceinline__ void pws_scale_min(int j, unsigned w0, unsigned w1,
                                                     unsigned w2, int* sc, int* mn)
{
    const int k = j & 3;
    const unsigned ba = (w0 >> (8 * k)) & 0xFF; // b[4+k]
    const unsigned bb = (w1 >> (8 * k)) & 0xFF; // b[8+k]
    const unsigned bc = (w2 >> (8 * k)) & 0xFF; // b[12+k]
    const bool lo = j < 4;
    *sc = (int)(lo ? (ba & 63) : ((bc & 0xF) | ((ba >> 6) << 4)));
    *mn = (int)(lo ? (bb & 63) : ((bc >> 4) | ((bb >> 6) << 4)));
}

// The raw words for eight contiguous values of one row, loaded by `pws_prep`.
//
// The load and the arithmetic are separate calls on purpose. Written as one
// function, every pass ended up in its own exec-masked block with an
// `s_wait_loadcnt` after each load, so a warp had exactly one 8-byte load in
// flight. Hoisting the loads into an array of these gives several in flight.
//
// Layout per KIND (the byte-level truth is `pf_dequant8`):
//   Q4_K  r[0] = d | dmin, r[1..3] = the 12 scale bytes, r[4..5] = 8 nibbles
//   Q6_K  r[0..1] = ql, r[2..3] = qh, r[4] = d, r[5] = sub-block scale
//   Q8_0  r[0..1] = the 8 quants, r[2] = d
struct pws_raw {
    unsigned r[6];
};

static __device__ __forceinline__ unsigned pws_u16pair(const unsigned char* p)
{
    const unsigned short* s = reinterpret_cast<const unsigned short*>(p);
    return (unsigned)s[0] | ((unsigned)s[1] << 16);
}

// Eight values starting at value index `v` (a multiple of 8): load only.
template <int KIND>
static __device__ __forceinline__ void pws_prep(const unsigned char* __restrict__ rp,
                                                int v, pws_raw& out)
{
    if (KIND == 0) {
        // 144-byte blocks, 16-aligned rows: the 16-byte header and the 8-byte
        // nibble run are both naturally aligned, so each is one load.
        const int blk = v >> 8;
        const int within = v & 255;
        const int sub = within >> 5;
        const int part = (within & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q4K_BLOCK_BYTES;
        const uint4 h = *reinterpret_cast<const uint4*>(b);
        const uint2 q =
            *reinterpret_cast<const uint2*>(b + 16 + (sub >> 1) * 32 + part * 8);
        out.r[0] = h.x;
        out.r[1] = h.y;
        out.r[2] = h.z;
        out.r[3] = h.w;
        out.r[4] = q.x;
        out.r[5] = q.y;
    } else if (KIND == 1) {
        // 210-byte blocks: only 2-aligned, so the runs are read as u16 pairs.
        const int blk = v >> 8;
        const int within = v & 255;
        const int j = within >> 4;
        const int l0 = within & 15;
        const unsigned char* b = rp + (size_t)blk * PF_Q6K_BLOCK_BYTES;
        const int hf = j >> 3;
        const int s = j & 7;
        const int group = s >> 1;
        const int is = s & 1;
        const int ql_off = hf * 64 + (group & 1) * 32 + is * 16 + l0;
        const int qh_off = 128 + hf * 32 + is * 16 + l0;
        out.r[0] = pws_u16pair(b + ql_off);
        out.r[1] = pws_u16pair(b + ql_off + 4);
        out.r[2] = pws_u16pair(b + qh_off);
        out.r[3] = pws_u16pair(b + qh_off + 4);
        out.r[4] = (unsigned)*(reinterpret_cast<const unsigned short*>(b + 208));
        out.r[5] = (unsigned)b[192 + j];
    } else {
        // 34-byte blocks: payload starts 2 bytes in, so u16 pairs again.
        const int blk = v >> 5;
        const int part = (v & 31) >> 3;
        const unsigned char* b = rp + (size_t)blk * PF_Q8_0_BLOCK_BYTES;
        const unsigned char* p = b + 2 + part * 8;
        out.r[0] = pws_u16pair(p);
        out.r[1] = pws_u16pair(p + 4);
        out.r[2] = (unsigned)*(reinterpret_cast<const unsigned short*>(b));
    }
}

// The same eight values, arithmetic only. `within` is `(base + wcol) & 255`.
template <int KIND>
static __device__ __forceinline__ void pws_finish(const pws_raw& in, int within,
                                                  float* out)
{
    if (KIND == 0) {
        const int sub = within >> 5;
        const float d =
            __half2float(__ushort_as_half((unsigned short)(in.r[0] & 0xFFFFu)));
        const float dmin =
            __half2float(__ushort_as_half((unsigned short)(in.r[0] >> 16)));
        int sc, mn;
        pws_scale_min(sub, in.r[1], in.r[2], in.r[3], &sc, &mn);
        const float scale = d * (float)sc;
        const float minv = dmin * (float)mn;
        const bool high = (sub & 1) != 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned word = in.r[4 + (i >> 2)];
            const int byte = (int)((word >> (8 * (i & 3))) & 0xFF);
            const int q = high ? (byte >> 4) : (byte & 0xF);
            out[i] = scale * (float)q - minv;
        }
    } else if (KIND == 1) {
        const int j = within >> 4;
        const int s = j & 7;
        const int group = s >> 1;
        const float d =
            __half2float(__ushort_as_half((unsigned short)(in.r[4] & 0xFFFFu)));
        const float scale = d * (float)(signed char)in.r[5];
        const int shift = 2 * group;
        const bool high = group >= 2;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned qword = in.r[i >> 2];
            const unsigned hword = in.r[2 + (i >> 2)];
            const int qb = (int)((qword >> (8 * (i & 3))) & 0xFF);
            const int hb = ((int)((hword >> (8 * (i & 3))) & 0xFF) >> shift) & 3;
            const int nib = high ? (qb >> 4) : (qb & 0xF);
            out[i] = scale * (float)((nib | (hb << 4)) - 32);
        }
    } else {
        const float d =
            __half2float(__ushort_as_half((unsigned short)(in.r[2] & 0xFFFFu)));
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const unsigned word = in.r[i >> 2];
            const int q = (int)((word >> (8 * (i & 3))) & 0xFF);
            out[i] = d * (float)(signed char)q;
        }
    }
}

template <int KIND, int RT, int KW, int THR>
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
    // Weight staging: one warp reads one row's k tile as one contiguous run,
    // 8 values per lane. GPR is how many lanes cover a row (KW/8).
    constexpr int GPR = KW / 8;             // lanes per weight row
    constexpr int RPW = 32 / GPR;           // weight rows per warp per pass
    constexpr int PRP = (THR / 32) * RPW;   // weight rows staged per pass
    static_assert(RT % 16 == 0, "RT must be a multiple of the 16 tile");
    static_assert(KW % 32 == 0 && KW <= 256, "KW must be 32-aligned and <= 256");
    static_assert(16 % XRP == 0, "token staging must cover 16 rows");
    static_assert(RT % PRP == 0, "ragged weight staging");

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
        // Token rows: clamp to the last real row instead of predicating the
        // load. Rows >= r are computed and discarded, which is cheaper than a
        // per-lane branch around every global load. Hoisted in small groups:
        // the point is several loads in flight, not all of them at once.
        {
            constexpr int XP = 16 / XRP;            // one float4 per pass
            constexpr int XB = XP < 4 ? XP : 4;
            for (int p0 = 0; p0 < XP; p0 += XB) {
                float4 xv[XB];
#pragma unroll
                for (int p = 0; p < XB; ++p) {
                    const int rl = (p0 + p) * XRP + xt;
                    const int gt = rl < r ? rl : r - 1;
                    xv[p] = *reinterpret_cast<const float4*>(
                        x + (size_t)gt * k + base + xc);
                }
#pragma unroll
                for (int p = 0; p < XB; ++p) {
                    const int rl = (p0 + p) * XRP + xt;
                    __half2* dst =
                        reinterpret_cast<__half2*>(shx + rl * XS + xc);
                    dst[0] = __floats2half2_rn(xv[p].x, xv[p].y);
                    dst[1] = __floats2half2_rn(xv[p].z, xv[p].w);
                }
            }
        }
        // Weight rows: same clamp. `k` is a multiple of 256 and KW <= 256, so
        // every k tile is whole and no k bound is needed.
        {
            constexpr int WP = RT / PRP;
            constexpr int WB = (WP % 4 == 0) ? 4 : ((WP % 2 == 0) ? 2 : 1);
            static_assert(WP % WB == 0, "weight hoist group must divide the passes");
            for (int p0 = 0; p0 < WP; p0 += WB) {
                pws_raw raw[WB];
#pragma unroll
                for (int p = 0; p < WB; ++p) {
                    const int g = row_base + (p0 + p) * PRP + wrow_l;
                    const int gr = g < n ? g : n - 1;
                    pws_prep<KIND>(w + (size_t)gr * wrow, base + wcol, raw[p]);
                }
                const int within = (base + wcol) & 255;
#pragma unroll
                for (int p = 0; p < WB; ++p) {
                    const int rl = (p0 + p) * PRP + wrow_l;
                    float v[8];
                    pws_finish<KIND>(raw[p], within, v);
                    pws_h8 hv;
#pragma unroll
                    for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
                    *reinterpret_cast<pws_h8*>(shw + rl * WS + wcol) = hv;
                }
            }
        }
    };

    // Independent WMMA chains. A single accumulator per warp runs the matrix
    // unit at a fraction of its throughput, because consecutive WMMA into the
    // same accumulator are a dependency chain. Rotating the k16 steps through
    // NACC accumulators costs one fold at the end.
    constexpr int NACC = (KW / 16) < PWS_NACC ? (KW / 16) : PWS_NACC;
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

// One entry per resident block format; `skinny_resident` checks the geometry
// against `KernelKind::RowTile { rows = PWS_RT, block = PWS_THREADS }`.
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
