// Matrix-unit (gfx12 WMMA) flash attention over a prefill chunk.
//
// The f32 tiled kernel (`attn_chunk_tile_f32`) reduces every (query, key)
// score with a five-step warp shuffle, so its cost is the shuffle chain, not
// the arithmetic: 8K prefill measured 3.58 s of a 9.93 s prompt at 3.7
// TFLOPS. Here QK^T and PV are `wmma_f32_16x16x16_f16` (the instruction the
// shipped prefill GEMM already uses at 171-188 TFLOPS), and the only
// cross-lane work left is the 16-key online-softmax reduction: four
// `xor` steps over a whole 16x16 score tile, not five per score.
//
// ## Block shape
//
// One block owns one KV head and AW_TILES * 16 query rows:
//
//   AW_TILES warps per query head, one per 16-query fragment
//   n_heads / n_kv warps per query tile (the GQA group shares its K/V)
//   => 32 * (group * AW_TILES) threads
//
// K^T and V are staged to shared once per block, so the whole GQA group reads
// each cache row once instead of once per query head; that is n_heads-wide
// traffic turned n_kv-wide. Q fragments stay in registers (16 x 16 x 16 halfs
// = 64 VGPRs), the 16 output fragments are 128 VGPRs, and the score fragment
// is 8 more.
//
// ## Fragment layouts (measured, same probe as prefill_wmma.hip)
//
//   A/B operand:    lane l, element e -> (row l % 16, col 8*(l/16) + e)
//   accumulator C:  lane l, element e -> (col l % 16, row 8*(l/16) + e)
//
// so for QK^T:  A = Q [query][d],  B = K^T [d][key]  (staged transposed)
//     for PV:   A = P [query][key] (transposed in shared, see below)
//               B = V [key][d]     (staged as-is)
// The score tile is column-major across lanes (one lane owns one key), which
// is why the per-row max/sum is a 4-step xor over the 16 lanes of each
// half-wave, and why P is written back to shared transposed before the PV
// pass turns it into the A operand.
//
// ## Numerics
//
// Scores are scaled by `scale * log2(e)` so the softmax is a bare exp2f; the
// running max and the correction term are in log2 units, the running sum is
// linear (the reduce step is the same as `attn_decode_kv`). Operands are f16,
// accumulation is f32, so a chunk differs from the f32 kernel in the last
// bits the same way the fp16 projections already do.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "attn_chunk_wmma: the gfx12 WMMA intrinsics are wave32 only"
#endif

// head width this kernel is built for: 16 output fragments + 16 Q fragments
// are what the VGPR budget is sized against.
#define AW_HD 256
// keys per iteration. 16 is one wmma n-tile.
#ifndef AW_KT
#define AW_KT 16
#endif
// query fragments (16 rows each) per block: warps = group * AW_TILES.
#ifndef AW_TILES
#define AW_TILES 2
#endif
// shared strides, in halfs. K^T is [d][key], V is [key][d]; the pad moves each
// row off the previous bank group.
#ifndef AW_KST
#define AW_KST 18
#endif
#ifndef AW_VST
#define AW_VST 264
#endif
// Warps a block can hold: the GQA group cap is 8 and AW_TILES <= 2.
#define AW_MAXW 16

typedef _Float16 aw_h8 __attribute__((ext_vector_type(8)));
typedef float aw_f8 __attribute__((ext_vector_type(8)));

#define AW_LOG2E 1.4426950408889634f
#define AW_NEG_INF (-3.402823466e+38f)

extern "C" __global__ void attn_chunk_wmma_f16(
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int n_heads,
    int n_kv,
    int head_dim,
    int width,
    int pos0,
    int tokens,
    float scale)
{
    // K^T: 256 rows of d, AW_KT key columns.
    __shared__ __align__(16) _Float16 sk[AW_HD * AW_KST];
    // V: AW_KT rows of key, 256 columns of d.
    __shared__ __align__(16) _Float16 sv[AW_KT * AW_VST];
    // P, one 16x16 tile per warp, row-major [query][key].
    __shared__ __align__(16) _Float16 sp[AW_MAXW * 16 * 16];

    const int kvh = blockIdx.x;
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = kvh * group + warp / AW_TILES;
    const int qi = warp % AW_TILES;
    const int tq = blockIdx.y * (AW_TILES * 16) + qi * 16;

    const int frow = lane & 15;        // operand row this lane feeds
    const int koff = (lane >> 4) * 8;  // 8-value half of the 16-wide inner dim

    // ---- Q fragments, once per block -------------------------------------
    aw_h8 qf[16];
    {
        const bool live = tq + frow < tokens;
        const float* qrow = q + ((size_t)(tq + frow) * n_heads + h) * head_dim;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            aw_h8 v;
            if (live) {
                const float* p = qrow + dd * 16 + koff;
                const float4 a = *reinterpret_cast<const float4*>(p);
                const float4 b = *reinterpret_cast<const float4*>(p + 4);
                v[0] = __float2half_rn(a.x);
                v[1] = __float2half_rn(a.y);
                v[2] = __float2half_rn(a.z);
                v[3] = __float2half_rn(a.w);
                v[4] = __float2half_rn(b.x);
                v[5] = __float2half_rn(b.y);
                v[6] = __float2half_rn(b.z);
                v[7] = __float2half_rn(b.w);
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) v[e] = (_Float16)0.0f;
            }
            qf[dd] = v;
        }
    }

    aw_f8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    float run_max[8];
    float run_sum[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        run_max[e] = AW_NEG_INF;
        run_sum[e] = 0.0f;
    }

    // Rows up to the last query of the block; each warp masks the tail of its
    // own 16 queries. The cache already holds every row of the chunk.
    int seq_block = pos0 + blockIdx.y * (AW_TILES * 16) + AW_TILES * 16;
    if (seq_block > pos0 + tokens) seq_block = pos0 + tokens;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float* kb = k_cache + (size_t)kvh * head_dim;
    const float* vb = v_cache + (size_t)kvh * head_dim;

    for (int base = 0; base < seq_block; base += AW_KT) {
        const int rows = (base + AW_KT <= seq_block) ? AW_KT : (seq_block - base);

        __syncthreads();

        // K^T: every (key, d) value lands at sk[d * AW_KST + key]. One thread
        // loads four consecutive d of one key, so the global read is a float4
        // and the four shared stores are 2 B each.
        for (int idx = tid; idx < AW_KT * (AW_HD / 4); idx += nthreads) {
            const int key = idx / (AW_HD / 4);
            const int dq = (idx - key * (AW_HD / 4)) * 4;
            const int rr = (key < rows) ? key : (rows - 1);
            const float4 v = *reinterpret_cast<const float4*>(
                kb + (size_t)(base + rr) * width + dq);
            sk[(dq + 0) * AW_KST + key] = __float2half_rn(v.x);
            sk[(dq + 1) * AW_KST + key] = __float2half_rn(v.y);
            sk[(dq + 2) * AW_KST + key] = __float2half_rn(v.z);
            sk[(dq + 3) * AW_KST + key] = __float2half_rn(v.w);
        }
        // V: row-major [key][d], one half4 store per four values.
        for (int idx = tid; idx < AW_KT * (AW_HD / 4); idx += nthreads) {
            const int key = idx / (AW_HD / 4);
            const int dq = (idx - key * (AW_HD / 4)) * 4;
            const int rr = (key < rows) ? key : (rows - 1);
            const float4 v = *reinterpret_cast<const float4*>(
                vb + (size_t)(base + rr) * width + dq);
            _Float16* dst = sv + key * AW_VST + dq;
            dst[0] = __float2half_rn(v.x);
            dst[1] = __float2half_rn(v.y);
            dst[2] = __float2half_rn(v.z);
            dst[3] = __float2half_rn(v.w);
        }
        __syncthreads();

        // ---- QK^T: two independent accumulator chains over 8 d-steps each.
        aw_f8 s0, s1;
#pragma unroll
        for (int e = 0; e < 8; ++e) { s0[e] = 0.0f; s1[e] = 0.0f; }
#pragma unroll
        for (int dd = 0; dd < 8; ++dd) {
            const aw_h8 kb0 =
                *reinterpret_cast<const aw_h8*>(sk + (dd * 16 + frow) * AW_KST + koff);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qf[dd], kb0, s0);
        }
#pragma unroll
        for (int dd = 8; dd < 16; ++dd) {
            const aw_h8 kb1 =
                *reinterpret_cast<const aw_h8*>(sk + (dd * 16 + frow) * AW_KST + koff);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qf[dd], kb1, s1);
        }

        // ---- causal mask + online softmax (log2 units) -------------------
        // acc col = key, row = query; the row reduction is over the 16 lanes
        // of this half-wave (xor 1..8), four steps for a whole 16x16 tile.
        const float sl2 = scale * AW_LOG2E;
        const int kglob = pos0 + base + (lane & 15);
        float p[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float s = (s0[e] + s1[e]) * sl2;
            p[e] = (kglob <= pos0 + tq + 8 * (lane >> 4) + e) ? s : AW_NEG_INF;
        }
        // `psum` becomes the tile row max; the score stays in `p` so the
        // exponential does not need a second dot product.
        float psum[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) psum[e] = p[e];
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                psum[e] = fmaxf(psum[e], __shfl_xor_sync(0xffffffffffffffffULL, psum[e], off));
            }
        }
        float corr[8];
        bool grow = false;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            psum[e] = fmaxf(run_max[e], psum[e]);      // m_new
            corr[e] = exp2f(run_max[e] - psum[e]);
            grow |= (psum[e] > run_max[e]);
            run_max[e] = psum[e];
            p[e] = exp2f(p[e] - psum[e]);
        }
        // Row sum over the tile: `psum` is free again (m_new is in run_max).
#pragma unroll
        for (int e = 0; e < 8; ++e) psum[e] = p[e];
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                psum[e] += __shfl_xor_sync(0xffffffffffffffffULL, psum[e], off);
            }
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) run_sum[e] = run_sum[e] * corr[e] + psum[e];

        // Scale the output accumulators when the running max grew. The
        // per-row factor differs inside the warp, so the fast path is a
        // warp-wide vote that it did not grow (true for almost every tile
        // after the first few).
        if (__any(grow)) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j][e] *= corr[e];
            }
        }

        // ---- P to shared, transposed into the A-fragment layout ----------
        // lane l holds key (lane & 15); query 8*(lane>>4)+e.
        _Float16* pw = sp + warp * 256;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            pw[(8 * (lane >> 4) + e) * 16 + (lane & 15)] = __float2half_rn(p[e]);
        }

        // ---- PV: 16 independent column fragments ------------------------
        const aw_h8 pf = *reinterpret_cast<const aw_h8*>(pw + frow * 16 + koff);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const aw_h8 vf =
                *reinterpret_cast<const aw_h8*>(sv + frow * AW_VST + j * 16 + koff);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf, vf, acc[j]);
        }
    }

    // ---- epilogue ---------------------------------------------------------
    const int row = tq + 8 * (lane >> 4);
    const int dcol = lane & 15;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = row + e;
            if (t < tokens) {
                out[((size_t)t * n_heads + h) * head_dim + j * 16 + dcol] =
                    acc[j][e] / run_sum[e];
            }
        }
    }
}
