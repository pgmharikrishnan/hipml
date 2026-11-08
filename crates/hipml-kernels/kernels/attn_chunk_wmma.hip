// Matrix-unit (gfx12 WMMA) flash attention over a prefill chunk.
//
// The f32 tiled kernel (`attn_chunk_tile_f32`) reduces every (query, key)
// score with a five-step warp shuffle, so its cost is the shuffle chain, not
// the arithmetic: 8K prefill measured 3.58 s of a 9.93 s prompt at 3.7
// TFLOPS. Here QK^T and PV are `wmma_f32_16x16x16_f16` (the instruction the
// shipped prefill GEMM already uses at 171-188 TFLOPS), and the only
// cross-lane work left is one `xor 16` per 16x16 score tile.
//
// ## Two warps per (query head, 16-query tile)
//
// A single warp holding both the 16 Q fragments (64 VGPRs) and the 16 output
// fragments (128 VGPRs) needs ~240 VGPRs; the compiler caps this kernel's
// allocation at 192 and spills the output fragments to scratch *inside the key
// loop*, which costs more than the shuffle chain it replaced (measured in the
// generated assembly: four scratch_store_b128 / scratch_load_b128 per tile).
//
// So the work is split by role:
//
//   score warp  QK^T + online softmax + P to shared   (no output fragments)
//   value warp  P from shared + PV                    (no Q fragments)
//
// The two run concurrently, so a key tile costs max(QK, PV) rather than their
// sum, and the largest live set is the value warp's 16 output fragments
// (128 VGPRs) or the score warp's 16 Q fragments + score (88). One block is
// one KV head, all of its GQA query heads, and 16 query rows: 32 * 2 * group
// threads, which is 12 warps for Qwen3.8-27B's 6-way GQA.
//
// ## Orientation: S^T = K Q^T
//
// The score tile is computed transposed so both cache tiles stage row-major
// and 16-byte aligned, and so the softmax reduction is one shuffle:
//
//   QK^T:  A = K [key][d] (shared), B = Q^T [d][query] (registers)
//          C = S^T [key][query]
//   PV:    A = P [query][key] (shared, transposed on write), B = V [key][d]
//          C = O [query][d]
//
// Probe layouts (`prefill_wmma.hip`):
//
//   A/B operand:    lane l, element e -> (row l % 16, col 8*(l/16) + e)
//   accumulator C:  lane l, element e -> (col l % 16, row 8*(l/16) + e)
//
// In S^T each lane holds one query (col) and eight keys (rows 8*(l/16)+e), so
// the tiled max/sum is seven fmax/add over e plus one `xor 16`. The output
// accumulator is [query][d] (eight queries per lane), so the epilogue takes
// the running sums from a 16-entry shared table instead of shuffling eight
// values. The score warp writes that table once, at the end.
//
// ## Numerics
//
// Scores are scaled by `scale * log2(e)` so the softmax is a bare exp2f; max
// and correction are in log2 units, the running sum is linear (same convention
// as `attn_decode_kv`). Operands are f16, accumulation f32, so a chunk differs
// from the f32 kernel in the last bits the way the fp16 projections already do.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "attn_chunk_wmma: the gfx12 WMMA intrinsics are wave32 only"
#endif

// Head width this kernel is built for.
#define AW_HD 256
// Keys per iteration: one wmma n-tile (rows of S^T).
#ifndef AW_KT
#define AW_KT 16
#endif
// Row stride of the K/V shared tiles, in halfs. 264 keeps every row 16 B
// aligned (a half8 fragment load) and moves consecutive rows off the same
// bank group.
#ifndef AW_KVST
#define AW_KVST 264
#endif
// Query rows per block, and warps per query head (score + value).
#define AW_QUERIES 16
#define AW_WARPS_PER_HEAD 2
// GQA group cap: the launcher checks it.
#define AW_MAXGROUP 8

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
    __shared__ __align__(16) _Float16 sk[AW_KT * AW_KVST];      // [key][d]
    __shared__ __align__(16) _Float16 sv[AW_KT * AW_KVST];      // [key][d]
    __shared__ __align__(16) _Float16 sp[AW_MAXGROUP * 16 * 16]; // [query][key]
    __shared__ float ssum[AW_MAXGROUP * 16];
    __shared__ float scorr[AW_MAXGROUP * 16];

    const int kvh = blockIdx.x;
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int role = warp & 1;              // 0 = score, 1 = value
    const int hidx = warp >> 1;             // query head within the GQA group
    const int h = kvh * group + hidx;
    const int tq = blockIdx.y * AW_QUERIES;

    const int frow = lane & 15;        // operand row this lane feeds
    const int koff = (lane >> 4) * 8;  // 8-value half of the 16-wide inner dim
    const int lrow = 8 * (lane >> 4);  // first accumulator row of this lane

    // ---- Q^T fragments (score warp only), once per block -----------------
    aw_h8 qt[16];
    if (role == 0) {
        const float* qb = q + (size_t)(tq + lrow) * n_heads * head_dim +
                          (size_t)h * head_dim;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            aw_h8 v;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = tq + lrow + e;
                v[e] = (_Float16)((t < tokens)
                                      ? qb[(size_t)e * n_heads * head_dim + dd * 16 + frow]
                                      : 0.0f);
            }
            qt[dd] = v;
        }
    }

    // Output fragments: the value warp owns all 16 d-tiles of the score warp's
    // 16 queries (columns d = j*16 + l%16).
    aw_f8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    float run_max = AW_NEG_INF;  // score warp only
    float run_sum = 0.0f;

    int seq_block = pos0 + blockIdx.y * AW_QUERIES + AW_QUERIES;
    if (seq_block > pos0 + tokens) seq_block = pos0 + tokens;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float* kb = k_cache + (size_t)kvh * head_dim;
    const float* vb = v_cache + (size_t)kvh * head_dim;
    const int qglob = pos0 + tq + (lane & 15);

    for (int base = 0; base < seq_block; base += AW_KT) {
        const int rows = (base + AW_KT <= seq_block) ? AW_KT : (seq_block - base);

        // Wait for the previous tile's readers before overwriting K/V.
        __syncthreads();

        // K and V: both [key][d]. One thread loads four consecutive d of one
        // key (a float4, coalesced across the warp) and stores a half4.
        for (int idx = tid; idx < AW_KT * (AW_HD / 4); idx += nthreads) {
            const int key = idx / (AW_HD / 4);
            const int dq = (idx - key * (AW_HD / 4)) * 4;
            const int rr = (key < rows) ? key : (rows - 1);
            const size_t src = (size_t)(base + rr) * width + dq;
            const float4 a = *reinterpret_cast<const float4*>(kb + src);
            const float4 b = *reinterpret_cast<const float4*>(vb + src);
            _Float16* kd = sk + key * AW_KVST + dq;
            _Float16* vd = sv + key * AW_KVST + dq;
            kd[0] = __float2half_rn(a.x);
            kd[1] = __float2half_rn(a.y);
            kd[2] = __float2half_rn(a.z);
            kd[3] = __float2half_rn(a.w);
            vd[0] = __float2half_rn(b.x);
            vd[1] = __float2half_rn(b.y);
            vd[2] = __float2half_rn(b.z);
            vd[3] = __float2half_rn(b.w);
        }
        __syncthreads();

        if (role == 0) {
            // ---- S^T = K Q^T: A = K (one row per lane), B = Q^T -----------
            aw_f8 s0, s1;
#pragma unroll
            for (int e = 0; e < 8; ++e) { s0[e] = 0.0f; s1[e] = 0.0f; }
#pragma unroll
            for (int dd = 0; dd < 8; ++dd) {
                const aw_h8 ka =
                    *reinterpret_cast<const aw_h8*>(sk + frow * AW_KVST + dd * 16 + koff);
                s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s0);
            }
#pragma unroll
            for (int dd = 8; dd < 16; ++dd) {
                const aw_h8 ka =
                    *reinterpret_cast<const aw_h8*>(sk + frow * AW_KVST + dd * 16 + koff);
                s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s1);
            }

            // ---- causal mask + online softmax (log2 units) ---------------
            // Lane l owns query l%16; its eight rows are keys 8*(l/16)+e.
            const float sl2 = scale * AW_LOG2E;
            float p[8];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kg = pos0 + base + lrow + e;
                const float s = (s0[e] + s1[e]) * sl2;
                p[e] = (kg <= qglob) ? s : AW_NEG_INF;
            }
            // Row max: seven fmax over e, then the other half of the keys.
            float mreg = p[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) mreg = fmaxf(mreg, p[e]);
            mreg = fmaxf(mreg, __shfl_xor_sync(0xffffffffffffffffULL, mreg, 16));
            const float old_max = run_max;
            run_max = fmaxf(run_max, mreg);
            const float corr = exp2f(old_max - run_max);
            run_sum *= corr;
#pragma unroll
            for (int e = 0; e < 8; ++e) p[e] = exp2f(p[e] - run_max);
            float sreg = p[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) sreg += p[e];
            sreg += __shfl_xor_sync(0xffffffffffffffffULL, sreg, 16);
            run_sum += sreg;

            // P[query][key]; lane l owns query l%16, keys 8*(l/16)+e.
            _Float16* pw = sp + hidx * 256;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                pw[(lrow + e) * 16 + (lane & 15)] = __float2half_rn(p[e]);
            }
            // The value warp owns the output fragments, so the softmax
            // correction has to travel to it through shared.
            scorr[hidx * 16 + (lane & 15)] = corr;
        }
        __syncthreads();

        if (role == 1) {
            // ---- PV: A = P [query][key], B = V [key][d] ------------------
            float c[8];
            bool grow = false;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                c[e] = scorr[hidx * 16 + lrow + e];
                grow |= (c[e] != 1.0f);
            }
            if (grow) {
#pragma unroll
                for (int j = 0; j < 16; ++j)
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[j][e] *= c[e];
            }
            const aw_h8 pf =
                *reinterpret_cast<const aw_h8*>(sp + hidx * 256 + frow * 16 + koff);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const aw_h8 vf =
                    *reinterpret_cast<const aw_h8*>(sv + frow * AW_KVST + j * 16 + koff);
                acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf, vf, acc[j]);
            }
        }
    }

    // ---- epilogue ---------------------------------------------------------
    if (role == 0) ssum[hidx * 16 + (lane & 15)] = 1.0f / run_sum;
    __syncthreads();

    if (role == 1) {
        const int row = tq + lrow;
        const int dcol = lane & 15;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = row + e;
                if (t < tokens) {
                    out[((size_t)t * n_heads + h) * head_dim + j * 16 + dcol] =
                        acc[j][e] * ssum[hidx * 16 + lrow + e];
                }
            }
        }
    }
}
