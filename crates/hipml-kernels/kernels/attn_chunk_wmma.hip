// Matrix-unit (gfx12 WMMA) flash attention over a prefill chunk.
//
// The f32 tiled kernel (`attn_chunk_tile_f32`) reduces every (query, key)
// score with a five-step warp shuffle, so its cost is the shuffle chain, not
// the arithmetic: 8K prefill measured 3.58 s of a 9.93 s prompt at 3.7
// TFLOPS. Here QK^T and PV are `wmma_f32_16x16x16_f16` (the instruction the
// shipped prefill GEMM already uses at 171-188 TFLOPS), and the only
// cross-lane work left is one `xor 16` per 16x16 score tile.
//
// ## Block shape: 32 queries x 128 output dims
//
// A warp that holds both the 16 Q fragments (64 VGPRs) and 16 output
// fragments (128 VGPRs) plus the softmax state needs ~240 VGPRs. The compiler
// caps this kernel at 192 and spills the fragments to scratch *inside the key
// loop*; the generated assembly showed four `scratch_store_b128`/
// `scratch_load_b128` pairs per tile, which is worse than the shuffle chain.
//
// `blockIdx.z` therefore selects half of the head width: a warp keeps 8 output
// fragments (64 VGPRs) and 128 d of V, so the live set is 64 + 64 + 24 + temps
// ~= 167 VGPRs, one block per CU at 12 warps and no scratch. QK^T still runs
// over all 256 d (`qt` is the full 16 fragments) because the score needs it;
// both d-half blocks re-read K, which is the 1.5x traffic this buys.
//
//   blockIdx.x  KV head
//   blockIdx.y  32 queries (two warps, one 16-query fragment each)
//   blockIdx.z  d half: output columns z*128 .. z*128+127
//
// K and V are staged once per block, so all six query heads of the GQA group
// share them; K is staged in full, V only for this block's d half.
//
// ## Orientation: S^T = K Q^T
//
// Computing the score tile transposed is what makes both cache tiles stage
// row-major and 16-byte aligned, and cuts the softmax to one shuffle:
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
// accumulator is [query][d] (eight queries per lane), so the epilogue reads
// the eight running sums from the lanes that hold them (eight shuffles, once).
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
// Output dims a block computes (blockIdx.z picks the half).
#define AW_DHALF 128
// Keys per iteration: one wmma n-tile (rows of S^T).
#ifndef AW_KT
#define AW_KT 16
#endif
// Row stride of the K/V shared tiles, in halfs. Keeps every row 16 B aligned
// (a half8 fragment load) and moves consecutive rows off the same bank group.
#ifndef AW_KST
#define AW_KST 264
#endif
#ifndef AW_VST
#define AW_VST 136            // 128 d + 8 pad
#endif
// Queries per block, warps, and the GQA group cap.
#define AW_QUERIES 32
#define AW_MAXGROUP 8
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
    __shared__ __align__(16) _Float16 sk[AW_KT * AW_KST];       // [key][d]
    __shared__ __align__(16) _Float16 sv[AW_KT * AW_VST];       // [key][d half]
    __shared__ __align__(16) _Float16 sp[AW_MAXW * 16 * 16];    // [query][key]

    const int kvh = blockIdx.x;
    const int dh = blockIdx.z;                    // 0 or 1
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = kvh * group + warp / 2;
    const int qi = warp & 1;
    const int tq = blockIdx.y * AW_QUERIES + qi * 16;

    const int frow = lane & 15;        // operand row this lane feeds
    const int koff = (lane >> 4) * 8;  // 8-value half of the 16-wide inner dim
    const int lrow = 8 * (lane >> 4);  // first accumulator row of this lane

    // ---- Q^T fragments, once per block -----------------------------------
    // B operand of S^T: element (d, query) = Q[query][d], i.e. lane l holds
    // d = dd*16 + (l%16), queries 8*(l/16)+e. Eight scalar reads per fragment,
    // paid once; the key loop re-reads no Q.
    aw_h8 qt[16];
    {
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

    // Output fragments: columns d = dh*128 + j*16 + (l%16).
    aw_f8 acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    float run_max = AW_NEG_INF;   // one query per lane (col = l%16)
    float run_sum = 0.0f;

    int seq_block = pos0 + blockIdx.y * AW_QUERIES + AW_QUERIES;
    if (seq_block > pos0 + tokens) seq_block = pos0 + tokens;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float* kb = k_cache + (size_t)kvh * head_dim;
    const float* vb = v_cache + (size_t)kvh * head_dim + dh * AW_DHALF;
    const int qglob = pos0 + tq + (lane & 15);

    for (int base = 0; base < seq_block; base += AW_KT) {
        const int rows = (base + AW_KT <= seq_block) ? AW_KT : (seq_block - base);

        __syncthreads();

        // K: full head width, [key][d]. One thread loads four consecutive d
        // of one key (a float4, coalesced across the warp) and stores a half4.
        for (int idx = tid; idx < AW_KT * (AW_HD / 4); idx += nthreads) {
            const int key = idx / (AW_HD / 4);
            const int dq = (idx - key * (AW_HD / 4)) * 4;
            const int rr = (key < rows) ? key : (rows - 1);
            const float4 a = *reinterpret_cast<const float4*>(
                kb + (size_t)(base + rr) * width + dq);
            _Float16* kd = sk + key * AW_KST + dq;
            kd[0] = __float2half_rn(a.x);
            kd[1] = __float2half_rn(a.y);
            kd[2] = __float2half_rn(a.z);
            kd[3] = __float2half_rn(a.w);
        }
        // V: only this block's d half.
        for (int idx = tid; idx < AW_KT * (AW_DHALF / 4); idx += nthreads) {
            const int key = idx / (AW_DHALF / 4);
            const int dq = (idx - key * (AW_DHALF / 4)) * 4;
            const int rr = (key < rows) ? key : (rows - 1);
            const float4 b = *reinterpret_cast<const float4*>(
                vb + (size_t)(base + rr) * width + dq);
            _Float16* vd = sv + key * AW_VST + dq;
            vd[0] = __float2half_rn(b.x);
            vd[1] = __float2half_rn(b.y);
            vd[2] = __float2half_rn(b.z);
            vd[3] = __float2half_rn(b.w);
        }
        __syncthreads();

        // ---- S^T = K Q^T: A = K (one row per lane), B = Q^T ---------------
        aw_f8 s0, s1;
#pragma unroll
        for (int e = 0; e < 8; ++e) { s0[e] = 0.0f; s1[e] = 0.0f; }
#pragma unroll
        for (int dd = 0; dd < 8; ++dd) {
            const aw_h8 ka =
                *reinterpret_cast<const aw_h8*>(sk + frow * AW_KST + dd * 16 + koff);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s0);
        }
#pragma unroll
        for (int dd = 8; dd < 16; ++dd) {
            const aw_h8 ka =
                *reinterpret_cast<const aw_h8*>(sk + frow * AW_KST + dd * 16 + koff);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s1);
        }

        // ---- causal mask + online softmax (log2 units) -------------------
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

        // Rescale only when this tile raised the max (rare after the first few
        // tiles); the factor is one scalar per lane, not one per output row.
        if (run_max > old_max) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j][e] *= corr;
        }

        // ---- P to shared, transposed into the A-fragment layout ----------
        // Writes P[query][key]; lane l owns query l%16, keys 8*(l/16)+e.
        _Float16* pw = sp + warp * 256;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            pw[(lrow + e) * 16 + (lane & 15)] = __float2half_rn(p[e]);
        }

        // ---- PV: A = P [query][key], B = V [key][d] ----------------------
        const aw_h8 pf = *reinterpret_cast<const aw_h8*>(pw + frow * 16 + koff);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const aw_h8 vf =
                *reinterpret_cast<const aw_h8*>(sv + frow * AW_VST + j * 16 + koff);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf, vf, acc[j]);
        }
    }

    // ---- epilogue ---------------------------------------------------------
    // The output accumulator holds eight queries per lane; fetch each query's
    // running sum from the lane that owns it (col = query = 0..15).
    const int row = tq + lrow;
    float inv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float sum = __shfl_sync(0xffffffffffffffffULL, run_sum, lrow + e);
        inv[e] = 1.0f / sum;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = row + e;
            if (t < tokens) {
                out[((size_t)t * n_heads + h) * head_dim + dh * AW_DHALF + j * 16 +
                    (lane & 15)] = acc[j][e] * inv[e];
            }
        }
    }
}
