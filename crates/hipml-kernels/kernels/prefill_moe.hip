// SPDX-License-Identifier: Apache-2.0
// Grouped expert GEMM: the routed FFN of a mixture-of-experts prefill chunk.
//
// A token reads 8 of 256 experts, so a chunk of T tokens reads
// 256 * (1 - (1 - 1/256)^(8T)) distinct experts — 255.9 at T = 2,048, i.e. the
// whole bank. Running the decode step once per prompt token therefore streams
// the bank once per token; this kernel streams each expert once per chunk.
//
// The shape is llama.cpp's `ggml_mul_mat_id`: `n_expert` blocks, each one
// compacting the token slots that selected its expert and running one dense
// GEMM. The compaction is a counting sort done by `moe_bucket_f32` (moe.hip):
// `off[e]..off[e+1]` is expert `e`'s run in a slot-major list, `xtok` maps a
// slot to its activation row and `yslot` to its output row. Both maps are
// gather indices, so the activation tile is read where it lies and no pass
// over `T x n_embd` is spent compacting it.
//
// Geometry: 64 tokens x 128 rows per block, 256 threads (8 warps: 4 token
// groups x 2 row groups, 4 accumulators a warp), 32-value k tiles. The token
// tile is 64 rather than the dense tile's 256 because an expert's run is short:
// at T = 2,048 a run averages 64 slots, and at T = 512 it averages 16. A
// 256-token tile would compute 4-16x the arithmetic the chunk needs.
//
// The fragment layout, the dequantiser and the k order are the shipped
// `prefill_wmma2.hip` tile's, so an expert row is the same 32-value-tile sum
// the dense prefill takes; only the row indices move.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "prefill_common.inc"

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "prefill_moe: the gfx12 WMMA intrinsics are wave32 only"
#endif

#define MG_TT 64        // prompt tokens (slots) per block
#define MG_RT 128       // output rows per block
#define MG_WT 16        // slots per warp
#define MG_WR 64        // rows per warp
#define MG_KW 32        // k values per shared k-tile
#define MG_THREADS 256
#define MG_XS (MG_KW + 8)   // x row stride in halfs (16 B aligned)
#define MG_WS (MG_KW + 8)   // weight row stride in halfs
#define MG_KQ (MG_KW / 32)  // 32-value staging passes per row
#define MG_XROW_H (MG_THREADS / 4)  // slot rows staged per pass
#define MG_DPROW (MG_THREADS / 4)   // weight rows dequantised per pass
#define MG_WARPS (MG_THREADS / 32)
#define MG_TG (MG_TT / MG_WT)
#define MG_RG (MG_RT / MG_WR)
#define MG_NTG (MG_WT / 16)
#define MG_NRG (MG_WR / 16)
#define MG_ACC (MG_NTG * MG_NRG)
#define MG_RP ((MG_RT + MG_DPROW - 1) / MG_DPROW)
#define MG_XP ((MG_TT + MG_XROW_H - 1) / MG_XROW_H)

static_assert(MG_TG * MG_RG == MG_WARPS, "warp grid must cover the block");
static_assert(MG_TT % MG_WT == 0 && MG_RT % MG_WR == 0, "ragged warp tiles");
static_assert(MG_WT % 16 == 0 && MG_WR % 16 == 0, "warp tile must be a 16 tile");
static_assert(MG_TT % MG_XROW_H == 0, "ragged x staging");
static_assert(MG_KW % 32 == 0, "k tile must be a multiple of 32");

typedef _Float16 __attribute__((ext_vector_type(8))) mg_h8;
typedef float __attribute__((ext_vector_type(8))) mg_f8;
typedef unsigned int __attribute__((ext_vector_type(4))) mg_u4;

// One expert's run of `[tok0, tok1)` slots through `[m, k]` weights.
//
// `x` is the chunk's activation tile in fp16, row-major `[slots][k]`; `xmap`
// gives the activation row of slot `s`, `ymap` its output row. `SILU` is the
// ffn `up` projection's epilogue: `y = silu(gate) * acc`, with `gate` read at
// the same output index and `yh` the fp16 copy of `y` (the `down` projection's
// activation tile).
template <int KIND, bool SILU>
static __device__ __forceinline__ void mg_body(
    const unsigned char* __restrict__ w,
    const void* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ gate,
    _Float16* __restrict__ yh,
    const int* __restrict__ xmap,
    const int* __restrict__ ymap,
    int m,
    int k,
    int tok0,
    int tok1,
    _Float16* __restrict__ shx,
    _Float16* __restrict__ shw)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wtg = warp / MG_RG;
    const int wrg = warp % MG_RG;

    const int row_base = blockIdx.x * MG_RT;
    const size_t wrow = pf_row_bytes(k, KIND);

    const int xt = tid >> 2;
    const int xc = (tid & 3) * 8;
    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    const int frow = lane & 15;
    const int fcol = (lane >> 4) * 8;
    const int xoff = (wtg * MG_WT + frow) * MG_XS + fcol;
    const int boff = (wrg * MG_WR + frow) * MG_WS + fcol;

    // Weight rows this block owns, resolved once: rows past `m` clamp onto the
    // last valid one and are masked at the store, which keeps every guard out
    // of the k loop.
    const unsigned char* wptr[MG_RP];
#pragma unroll
    for (int p = 0; p < MG_RP; ++p) {
        int gr = row_base + p * MG_DPROW + d_row;
        if (gr >= m) gr = m - 1;
        wptr[p] = w + (size_t)gr * wrow;
    }

    const int mrow = 8 * (lane >> 4);
    const int gr0 = row_base + wrg * MG_WR + (lane & 15);

    for (int t0 = tok0; t0 < tok1; t0 += MG_TT) {
        const int t1 = (t0 + MG_TT < tok1) ? t0 + MG_TT : tok1;

        // Activation rows of this tile, one offset per staging pass.
        size_t xofs[MG_XP];
#pragma unroll
        for (int p = 0; p < MG_XP; ++p) {
            int gt = t0 + p * MG_XROW_H + xt;
            if (gt >= t1) gt = t1 - 1;
            xofs[p] = (size_t)xmap[gt] * k;
        }

        mg_f8 acc[MG_ACC];
#pragma unroll
        for (int j = 0; j < MG_ACC; ++j)
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

        for (int base = 0; base < k; base += MG_KW) {
            __syncthreads();

            // Activation tile: fp16, `MG_XROW_H` slots per pass, 8 halfs a
            // thread. Copied as 16 raw bytes so the tile is bit-identical to
            // the one the dense prefill staged.
#pragma unroll
            for (int q = 0; q < MG_KQ; ++q)
#pragma unroll
                for (int p = 0; p < MG_XP; ++p) {
                    const int rl = p * MG_XROW_H + xt;
                    const int kc = base + q * 32 + xc;
                    const unsigned char* sp =
                        reinterpret_cast<const unsigned char*>(x) + (xofs[p] + kc) * 2;
                    mg_u4 v;
                    __builtin_memcpy(&v, sp, 16);
                    *reinterpret_cast<mg_u4*>(shx + rl * MG_XS + q * 32 + xc) = v;
                }

            // Weight tile: `MG_DPROW` rows per pass, 8 values a thread.
#pragma unroll
            for (int q = 0; q < MG_KQ; ++q)
#pragma unroll
                for (int p = 0; p < MG_RP; ++p) {
                    const int rl = p * MG_DPROW + d_row;
                    float v[8];
                    pf_dequant8<KIND>(wptr[p], base + q * 32 + d_q * 8, v);
                    mg_h8 hv;
#pragma unroll
                    for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
                    *reinterpret_cast<mg_h8*>(shw + rl * MG_WS + q * 32 + d_q * 8) = hv;
                }
            __syncthreads();

#pragma unroll
            for (int kk = 0; kk < MG_KW / 16; ++kk) {
                mg_h8 a[MG_NTG];
                mg_h8 b[MG_NRG];
#pragma unroll
                for (int i = 0; i < MG_NTG; ++i)
                    a[i] = *reinterpret_cast<const mg_h8*>(shx + xoff + i * 16 * MG_XS +
                                                          kk * 16);
#pragma unroll
                for (int j = 0; j < MG_NRG; ++j)
                    b[j] = *reinterpret_cast<const mg_h8*>(shw + boff + j * 16 * MG_WS +
                                                          kk * 16);
#pragma unroll
                for (int i = 0; i < MG_NTG; ++i)
#pragma unroll
                    for (int j = 0; j < MG_NRG; ++j) {
                        const int idx = i * MG_NRG + j;
                        acc[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
                            a[i], b[j], acc[idx]);
                    }
            }
        }

#pragma unroll
        for (int i = 0; i < MG_NTG; ++i)
#pragma unroll
            for (int j = 0; j < MG_NRG; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int gt = t0 + wtg * MG_WT + i * 16 + mrow + e;
                    const int r = gr0 + j * 16;
                    if (gt < t1 && r < m) {
                        const size_t o = (size_t)ymap[gt] * m + r;
                        if (SILU) {
                            const float g = gate[o];
                            const float a = (g / (1.0f + expf(-g))) * acc[i * MG_NRG + j][e];
                            y[o] = a;
                            yh[o] = (_Float16)a;
                        } else {
                            y[o] = acc[i * MG_NRG + j][e];
                        }
                    }
                }
        __syncthreads();
    }
}

// One launch for every expert with at least one slot. `blockIdx.y` is the
// expert; an expert with no slots returns before touching shared memory.
#define MG_DEFINE_ENTRY(NAME, SILU)                                                  \
    extern "C" __global__ void __launch_bounds__(MG_THREADS) NAME(                    \
        const unsigned char* __restrict__ w, long long stride,                        \
        const int* __restrict__ off, const void* __restrict__ x,                      \
        const int* __restrict__ xmap, float* __restrict__ y,                          \
        const int* __restrict__ ymap, const float* __restrict__ gate,                 \
        _Float16* __restrict__ yh, int m, int k, int kind)                            \
    {                                                                                \
        const int e = blockIdx.y;                                                     \
        const int s0 = off[e];                                                        \
        const int s1 = off[e + 1];                                                    \
        if (s0 >= s1) return;                                                         \
        const unsigned char* we = w + (long long)e * stride;                          \
        __shared__ __align__(16) _Float16 shx[MG_TT * MG_XS];                         \
        __shared__ __align__(16) _Float16 shw[MG_RT * MG_WS];                         \
        switch (kind) {                                                               \
        case 0:                                                                       \
            mg_body<0, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        case 1:                                                                       \
            mg_body<1, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        case 2:                                                                       \
            mg_body<2, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        case 4:                                                                       \
            mg_body<4, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        case 5:                                                                       \
            mg_body<5, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        case 6:                                                                       \
            mg_body<6, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        case 7:                                                                       \
            mg_body<7, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        case 8:                                                                       \
            mg_body<8, SILU>(we, x, y, gate, yh, xmap, ymap, m, k, s0, s1, shx, shw); \
            break;                                                                    \
        default:                                                                      \
            break;                                                                    \
        }                                                                             \
    }

// `moe_prefill_gemm`: the gate and down projections (raw store).
MG_DEFINE_ENTRY(moe_prefill_gemm, false)
// `moe_prefill_gemm_silu`: the `up` projection, SwiGLU in the epilogue.
MG_DEFINE_ENTRY(moe_prefill_gemm_silu, true)
