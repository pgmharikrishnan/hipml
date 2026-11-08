// Matrix-unit (gfx12 WMMA) flash attention over a prefill chunk.
//
// The f32 tiled kernel (`attn_chunk_tile_f32`) reduces every (query, key)
// score with a five-step warp shuffle, so its cost is the shuffle chain, not
// the arithmetic: 8K prefill measured 3.58 s of a 9.93 s prompt at 3.7
// TFLOPS. Here QK^T and PV are `wmma_f32_16x16x16_f16` (the instruction the
// shipped prefill GEMM already uses at 171-188 TFLOPS), and the only
// cross-lane work left is one `xor 16` per 16x16 score tile.
//
// ## Block shape
//
// One block owns one KV head and AW_TILES * 16 query rows:
//
//   AW_TILES warps per query head, one per 16-query fragment
//   n_heads / n_kv warps per query tile (the GQA group shares its K/V)
//   => 32 * (group * AW_TILES) threads
//
// K and V are staged to shared once per block, so the whole GQA group reads
// each cache row once instead of once per query head: n_heads-wide traffic
// becomes n_kv-wide. Q fragments stay in registers (16 x 16 x 16 halfs = 64
// VGPRs), the 16 output fragments are 128, and the score fragment 8.
//
// ## Orientation: S^T = K Q^T
//
// The score tile is computed transposed, which is what makes both cache tiles
// load row-major and 16-byte aligned:
//
//   QK^T:  A = K [key][d] (staged as-is), B = Q^T [d][query] (registers)
//          C = S^T [key][query]
//   PV:    A = P [query][key] (written transposed to shared), B = V [key][d]
//          C = O [query][d]
//
// The shipped probe's layouts (`prefill_wmma.hip`):
//
//   A/B operand:    lane l, element e -> (row l % 16, col 8*(l/16) + e)
//   accumulator C:  lane l, element e -> (col l % 16, row 8*(l/16) + e)
//
// so in S^T each lane holds one query (col) and eight keys (rows 8*(l/16)+e),
// and the tiled max/sum for that query is: seven fmax/add over e plus one
// `xor 16` to fold keys 8..15 in. The output accumulator is [query][d], i.e.
// eight queries per lane, so the epilogue reads the eight running sums from
// the lanes that hold them (eight shuffles, once).
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

// Head width this kernel is built for: 16 output fragments + 16 Q fragments
// are what the VGPR budget is sized against.
#define AW_HD 256
// Keys per iteration: one wmma n-tile (rows of S^T).
#ifndef AW_KT
#define AW_KT 16
#endif
// 16-query fragments per block: warps = group * AW_TILES.
#ifndef AW_TILES
#define AW_TILES 2
#endif
// Row stride of the K/V shared tiles, in halfs. 264 keeps each row 16 B
// aligned (a half8 fragment load) and moves consecutive rows off the same
// bank group; 256 would put every row on the same banks.
#ifndef AW_KVST
#define AW_KVST 264
#endif
// Warps a block can hold (GQA group cap 8, AW_TILES <= 2).
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
    __shared__ __align__(16) _Float16 sk[AW_KT * AW_KVST];  // [key][d]
    __shared__ __align__(16) _Float16 sv[AW_KT * AW_KVST];  // [key][d]
    __shared__ __align__(16) _Float16 sp[AW_MAXW * 16 * 16]; // [query][key]

    const int kvh = blockIdx.x;
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = kvh * group + warp / AW_TILES;
    const int qi = warp % AW_TILES;
    const int tq = blockIdx.y * (AW_TILES * 16) + qi * 16;

    const int frow = lane & 15;        // operand row this lane feeds
    const int koff = (lane >> 4) * 8;  // 8-value half of the 16-wide inner dim

    // ---- Q^T fragments, once per block -----------------------------------
    // B operand of S^T: element (d, query) = Q[query][d], i.e. lane l holds
    // d = dd*16 + (l%16), queries 8*(l/16)+e. Eight scalar reads per fragment,
    // paid once; the key loop re-reads no Q.
    aw_h8 qt[16];
    {
        const float* qb = q + (size_t)(tq + 8 * (lane >> 4)) * n_heads * head_dim +
                          (size_t)h * head_dim;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            aw_h8 v;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int t = 8 * (lane >> 4) + e;
                v[e] = (_Float16)((tq + t < tokens)
                                      ? qb[(size_t)e * n_heads * head_dim +
                                           dd * 16 + frow]
                                      : 0.0f);
            }
            qt[dd] = v;
        }
    }

    aw_f8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    // One query per lane (col = l%16); the running max/sum are scalars.
    float run_max = AW_NEG_INF;
    float run_sum = 0.0f;

    int seq_block = pos0 + blockIdx.y * (AW_TILES * 16) + AW_TILES * 16;
    if (seq_block > pos0 + tokens) seq_block = pos0 + tokens;

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const float* kb = k_cache + (size_t)kvh * head_dim;
    const float* vb = v_cache + (size_t)kvh * head_dim;

    // The chunk has already appended every row, so the causal bound is the
    // per-lane key/query comparison, not a loop bound.
    const int qglob = pos0 + tq + (lane & 15);

    for (int base = 0; base < seq_block; base += AW_KT) {
        const int rows = (base + AW_KT <= seq_block) ? AW_KT : (seq_block - base);

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

        // ---- S^T = K Q^T: A = K (one row per lane), B = Q^T ---------------
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

        // ---- causal mask + online softmax (log2 units) -------------------
        // Lane l owns query l%16; its eight rows are keys 8*(l/16)+e.
        const float sl2 = scale * AW_LOG2E;
        float p[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int kg = pos0 + base + 8 * (lane >> 4) + e;
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
            for (int j = 0; j < 16; ++j) {
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j][e] *= corr;
            }
        }

        // ---- P to shared, transposed into the A-fragment layout ----------
        // Writes P[query][key]; lane l owns query l%16, keys 8*(l/16)+e.
        _Float16* pw = sp + warp * 256;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            pw[(8 * (lane >> 4) + e) * 16 + (lane & 15)] = __float2half_rn(p[e]);
        }

        // ---- PV: A = P [query][key], B = V [key][d] ----------------------
        const aw_h8 pf = *reinterpret_cast<const aw_h8*>(pw + frow * 16 + koff);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const aw_h8 vf =
                *reinterpret_cast<const aw_h8*>(sv + frow * AW_KVST + j * 16 + koff);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(pf, vf, acc[j]);
        }
    }

    // ---- epilogue ---------------------------------------------------------
    // The output accumulator holds eight queries per lane; fetch each query's
    // running sum from the lane that owns it (col = query = 0..15).
    const int row = tq + 8 * (lane >> 4);
    const int dcol = lane & 15;
    float inv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float sum = __shfl_sync(0xffffffffffffffffULL, run_sum, 8 * (lane >> 4) + e);
        inv[e] = 1.0f / sum;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int t = row + e;
            if (t < tokens) {
                out[((size_t)t * n_heads + h) * head_dim + j * 16 + dcol] =
                    acc[j][e] * inv[e];
            }
        }
    }
}
