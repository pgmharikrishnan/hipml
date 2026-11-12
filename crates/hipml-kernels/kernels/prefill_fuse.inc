// SPDX-License-Identifier: Apache-2.0
// Fused ffn gate+up tile: one launch produces `act = silu(gate) * up` and its
// fp16 copy, from one staged activation tile and two weight streams.
//
// Included from the end of `prefill_wmma2.hip`, so it reuses that file's block
// decoders (`pww2_loadraw` / `pww2_dequant8`), fragment types and activation
// staging geometry. Nothing here is visible to the shipped tiles: the existing
// bodies are untouched, and every function below is new.
//
// ## Why two matrices in one block needs a new tile
//
// A block's accumulator footprint is `TT * RT` f32 values over 256 threads, and
// the shipped tile is pinned by it: 256 tokens x 128 rows = 32,768 floats =
// 128 VGPRs per thread, inside a 256-VGPR file with one resident block.
//
// The gate/up pair is two matrices of the same [m, k], and SwiGLU needs both
// values for one output element, so a fused block must hold both. Doubling the
// accumulators (256 tokens x 128 rows x 2) is 256 VGPRs before a single staging
// register, i.e. the whole file: it cannot fit. The fusion therefore keeps the
// accumulator budget at one shipped block's - `TT * 2 * RT == 32,768` floats -
// by trading tokens for rows, which leaves exactly two shapes:
//
//   PFU_TT=128, PFU_RT=128   x stream halves, weight sweeps double
//   PFU_TT=256, PFU_RT=64    streams unchanged, launch count halves (shipped)
//
// Both give the same LDS footprint (30,720 B: the x tile plus two weight tiles)
// and the same per-warp accumulator count (16 f8 = 128 VGPRs), because the warp
// grid is `TT/32` token groups x `RT/64` row groups and a warp always covers
// 32 tokens x 64 rows of each matrix. The compiled kernel is 186-208 VGPRs (the
// shipped `_w2h` tile is 181-211) with no spills, so it keeps the one block per
// CU the tile is pinned at.
//
// Measured at 2,048 tokens (`probe_prefill_tiles.cpp`, one lock hold,
// `docs/notes/prefill-tile-ceiling.md` section 9): the 128/128 shape is 33-165%
// *slower* than the two launches it replaces - halving the activation reads per
// output row doubles the weight sweeps, and the weight stream is the larger one
// once a pair is resident - while 256/64 is within +-3% of the pair, i.e. the
// activation staging cannot be halved inside the register file. The shipped
// 256/64 shape wins end to end only through the launch count.
//
// ## Numerics
//
// The k order is the shipped one (32-value k-tiles, `KW/16` WMMA rounds each),
// so a given output element accumulates the same fragments in the same order;
// the fused kernel's gate and up values are bit-identical to the two separate
// launches. The SwiGLU expression is `silu_mul_f32`'s verbatim. The gate tile is
// never written to global memory: it is in registers at the epilogue.

#ifndef PFU_TT
#define PFU_TT 256          // prompt tokens per block
#endif
#ifndef PFU_RT
#define PFU_RT 64           // output rows per matrix per block (gate and up)
#endif
#ifndef PFU_KW
#define PFU_KW 32           // k values per shared k-tile
#endif

#define PFU_THREADS 256
#define PFU_XS (PFU_KW + 8)            // x row stride in halfs
#define PFU_WS (PFU_KW + 8)            // weight row stride in halfs
#define PFU_KQ (PFU_KW / 32)           // 32-value staging passes per row
#define PFU_XROW (PFU_THREADS / 4)     // fp16 x rows staged per pass
#define PFU_DPROW (PFU_THREADS / 4)    // weight rows dequantised per pass
#define PFU_XP (PFU_TT / PFU_XROW)
#define PFU_RP (PFU_RT / PFU_DPROW)
#define PFU_WT 32                      // tokens per warp
#define PFU_WR 64                      // rows per warp, per matrix
#define PFU_TG (PFU_TT / PFU_WT)
#define PFU_RG (PFU_RT / PFU_WR)
#define PFU_NTG (PFU_WT / 16)
#define PFU_NRG (PFU_WR / 16)
#define PFU_ACC (PFU_NTG * PFU_NRG)    // accumulators per matrix per warp
#define PFU_WN (PFU_KQ * PFU_RP)       // staged weight runs per thread

static_assert(PFU_TG * PFU_RG == PFU_THREADS / 32, "warp grid must cover the block");
static_assert(PFU_TT * 2 * PFU_RT == 32768, "one shipped block's accumulator budget");
static_assert(PFU_TT % PFU_XROW == 0 && PFU_RT % PFU_DPROW == 0, "ragged staging");
static_assert(PFU_RT % PFU_WR == 0 && PFU_TT % PFU_WT == 0, "ragged warp tile");
static_assert(PFU_KW % 32 == 0 && PFU_KW <= 128, "k tile must be 32..128");

// Row bytes of one weight block row, for a format tag.
template <int KIND>
static __device__ __forceinline__ size_t pfu_row_bytes(int k)
{
    return pf_row_bytes(k, KIND);
}

// One k-tile of weight staging: fetch every run, then dequantise. Free
// functions rather than lambdas because the format is a template parameter and
// the two matrices have different ones.
template <int KIND>
static __device__ __forceinline__ void pfu_load_w(const unsigned char* const* wp,
                                                 pww2_raw* raw, int base, int dq)
{
#pragma unroll
    for (int q = 0; q < PFU_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PFU_RP; ++p)
            pww2_loadraw<KIND>(wp[p], base + q * 32 + dq, raw[q * PFU_RP + p]);
}

template <int KIND>
static __device__ __forceinline__ void pfu_commit_w(_Float16* ws, const pww2_raw* raw, int base,
                                                    int dq, int drow)
{
#pragma unroll
    for (int q = 0; q < PFU_KQ; ++q)
#pragma unroll
        for (int p = 0; p < PFU_RP; ++p) {
            const int rl = p * PFU_DPROW + drow;
            float v[8];
            pww2_dequant8<KIND>(raw[q * PFU_RP + p], base + q * 32 + dq, v);
            pww2_h8 hv;
#pragma unroll
            for (int i = 0; i < 8; ++i) hv[i] = (_Float16)v[i];
            *reinterpret_cast<pww2_h8*>(ws + rl * PFU_WS + q * 32 + dq) = hv;
        }
}

// `act[t, :] = silu(gate[t, :]) * (Wu @ x[t, :])`, `acth` its fp16 copy. The gate
// product is (Wg @ x[t, :]) and never leaves this block.
template <int KG, int KU>
static __device__ __forceinline__ void pfu_body(
    const unsigned char* __restrict__ wg,
    const unsigned char* __restrict__ wu,
    const void* __restrict__ x,
    float* __restrict__ act,
    _Float16* __restrict__ acth,
    int m,
    int k,
    int tokens)
{
    __shared__ __align__(16) _Float16 shx[PFU_TT * PFU_XS];
    __shared__ __align__(16) _Float16 shwg[PFU_RT * PFU_WS];
    __shared__ __align__(16) _Float16 shwu[PFU_RT * PFU_WS];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wtg = warp / PFU_RG;
    const int wrg = warp % PFU_RG;

    const int tok_base = blockIdx.y * PFU_TT;
    const int row_base = blockIdx.x * PFU_RT;
    const size_t wrowg = pfu_row_bytes<KG>(k);
    const size_t wrowu = pfu_row_bytes<KU>(k);

    const int xt = tid >> 2;
    const int xc = (tid & 3) * 8;
    const int d_row = tid >> 2;
    const int d_q = tid & 3;

    const int frow = lane & 15;
    const int fcol = (lane >> 4) * 8;
    const int xoff = (wtg * PFU_WT + frow) * PFU_XS + fcol;
    const int boff = (wrg * PFU_WR + frow) * PFU_WS + fcol;

    pww2_f8 accg[PFU_ACC];
    pww2_f8 accu[PFU_ACC];
#pragma unroll
    for (int j = 0; j < PFU_ACC; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            accg[j][e] = 0.0f;
            accu[j][e] = 0.0f;
        }

    // Rows outside the prompt (or outside `m`) are clamped onto the last valid
    // one, as in the shipped tile: the store loop masks them.
    size_t xofs[PFU_XP];
#pragma unroll
    for (int p = 0; p < PFU_XP; ++p) {
        int gt = tok_base + p * PFU_XROW + xt;
        if (gt >= tokens) gt = tokens - 1;
        xofs[p] = (size_t)gt * k;
    }
    const unsigned char* wptg[PFU_RP];
    const unsigned char* wptu[PFU_RP];
#pragma unroll
    for (int p = 0; p < PFU_RP; ++p) {
        int gr = row_base + p * PFU_DPROW + d_row;
        if (gr >= m) gr = m - 1;
        wptg[p] = wg + (size_t)gr * wrowg;
        wptu[p] = wu + (size_t)gr * wrowu;
    }

    pww2_raw rg[PFU_WN];
    pww2_raw ru[PFU_WN];
    pww2_u4 xh[PFU_XP];

    auto load_x = [&](int base) {
#pragma unroll
        for (int p = 0; p < PFU_XP; ++p) {
            const unsigned char* sp =
                reinterpret_cast<const unsigned char*>(x) + (xofs[p] + base + xc) * 2;
            __builtin_memcpy(&xh[p], sp, 16);
        }
    };

    auto commit_x = [&]() {
#pragma unroll
        for (int p = 0; p < PFU_XP; ++p) {
            const int rl = p * PFU_XROW + xt;
            *reinterpret_cast<pww2_u4*>(shx + rl * PFU_XS + xc) = xh[p];
        }
    };

    for (int base = 0; base < k; base += PFU_KW) {
        __syncthreads();
        // Both weight streams are fetched before either is converted, so their
        // global latency overlaps the first dequantise pass. The activation run
        // is already fp16, so it is a copy.
        load_x(base);
        pfu_load_w<KG>(wptg, rg, base, d_q * 8);
        pfu_load_w<KU>(wptu, ru, base, d_q * 8);
        commit_x();
        pfu_commit_w<KG>(shwg, rg, base, d_q * 8, d_row);
        pfu_commit_w<KU>(shwu, ru, base, d_q * 8, d_row);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < PFU_KW / 16; ++kk) {
            pww2_h8 a[PFU_NTG];
            pww2_h8 bg[PFU_NRG];
            pww2_h8 bu[PFU_NRG];
#pragma unroll
            for (int i = 0; i < PFU_NTG; ++i)
                a[i] = *reinterpret_cast<const pww2_h8*>(shx + xoff + i * 16 * PFU_XS + kk * 16);
#pragma unroll
            for (int j = 0; j < PFU_NRG; ++j) {
                bg[j] = *reinterpret_cast<const pww2_h8*>(shwg + boff + j * 16 * PFU_WS + kk * 16);
                bu[j] = *reinterpret_cast<const pww2_h8*>(shwu + boff + j * 16 * PFU_WS + kk * 16);
            }
#pragma unroll
            for (int i = 0; i < PFU_NTG; ++i)
#pragma unroll
                for (int j = 0; j < PFU_NRG; ++j) {
                    const int idx = i * PFU_NRG + j;
                    accg[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], bg[j],
                                                                                accg[idx]);
                    accu[idx] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], bu[j],
                                                                                accu[idx]);
                }
        }
    }

    // SwiGLU in the epilogue: `silu_mul_f32`'s expression on values this block
    // already holds, so the gate tile is neither stored nor re-read. The row
    // masks are the shipped store loop's.
    const int mrow = 8 * (lane >> 4);
    const int gr = row_base + wrg * PFU_WR + (lane & 15);
#pragma unroll
    for (int i = 0; i < PFU_NTG; ++i)
#pragma unroll
        for (int j = 0; j < PFU_NRG; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int gt = tok_base + wtg * PFU_WT + i * 16 + mrow + e;
                const int r = gr + j * 16;
                if (gt < tokens && r < m) {
                    const size_t o = (size_t)gt * m + r;
                    const float g = accg[i * PFU_NRG + j][e];
                    const float a = (g / (1.0f + expf(-g))) * accu[i * PFU_NRG + j][e];
                    act[o] = a;
                    acth[o] = (_Float16)a;
                }
            }
        }
}

// One entry per format pair. The gate and up tensors of one layer are chosen
// independently by the UD dynamic quant, so the pair is the template argument
// rather than the single format the other tiles take: a runtime tag inside the
// decoder costs 317 spill slots (see the note). The list is the pairs the
// shipped mixes use; a pair that is not here falls back to the two-launch path.
#define PFU_ENTRY(G, U, NAME)                                                          \
    extern "C" __global__ void __launch_bounds__(PFU_THREADS, 1) NAME(                 \
        const unsigned char* __restrict__ wg, const unsigned char* __restrict__ wu,     \
        const void* __restrict__ x, float* __restrict__ act,                           \
        _Float16* __restrict__ acth, int m, int k, int tokens)                          \
    {                                                                                  \
        pfu_body<G, U>(wg, wu, x, act, acth, m, k, tokens);                             \
    }

// Same format on both sides.
PFU_ENTRY(0, 0, prefill_gu_q4_k_q4_k)
PFU_ENTRY(1, 1, prefill_gu_q6_k_q6_k)
PFU_ENTRY(2, 2, prefill_gu_q8_0_q8_0)
PFU_ENTRY(3, 3, prefill_gu_halfw_halfw)
PFU_ENTRY(4, 4, prefill_gu_q5_k_q5_k)
PFU_ENTRY(5, 5, prefill_gu_iq4_xs_iq4_xs)
PFU_ENTRY(6, 6, prefill_gu_iq4_nl_iq4_nl)
PFU_ENTRY(7, 7, prefill_gu_q3_k_q3_k)
PFU_ENTRY(8, 8, prefill_gu_iq3_s_iq3_s)

// Mismatched pairs: UD-Q4_K_M (16 pairs), UD-Q6_K (5) and the stock file (1).
PFU_ENTRY(0, 4, prefill_gu_q4_k_q5_k)
PFU_ENTRY(0, 5, prefill_gu_q4_k_iq4_xs)
PFU_ENTRY(0, 1, prefill_gu_q4_k_q6_k)
PFU_ENTRY(4, 0, prefill_gu_q5_k_q4_k)
PFU_ENTRY(4, 1, prefill_gu_q5_k_q6_k)
PFU_ENTRY(5, 0, prefill_gu_iq4_xs_q4_k)
PFU_ENTRY(5, 4, prefill_gu_iq4_xs_q5_k)
PFU_ENTRY(5, 7, prefill_gu_iq4_xs_q3_k)
PFU_ENTRY(5, 8, prefill_gu_iq4_xs_iq3_s)
PFU_ENTRY(6, 4, prefill_gu_iq4_nl_q5_k)
PFU_ENTRY(6, 5, prefill_gu_iq4_nl_iq4_xs)
PFU_ENTRY(7, 5, prefill_gu_q3_k_iq4_xs)
PFU_ENTRY(1, 2, prefill_gu_q6_k_q8_0)
PFU_ENTRY(1, 4, prefill_gu_q6_k_q5_k)
