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
//   QK^T:  A = K [key][d] (shared),      B = Q [query][d] (registers)
//          C = S^T [key][query]
//   PV:    A = P [query][key] (registers, same layout as S^T), B = V^T [d][key]
//          C = O [query][d]
//
// Both operands are fed to the matrix unit the way the shipped prefill GEMM
// feeds its weights: B is stored row-major by *output column*, so a B
// fragment is eight contiguous inner-dimension values of one output column
// (B_hw[n][k], verified by `prefill_wmma.hip`'s accumulator/operand probe).
// That is why V is staged transposed and Q is not.
//
//   A operand:      lane l, element e -> (row l % 16, k 8*(l/16) + e)
//   B operand:      lane l, element e -> (col n = l % 16, k 8*(l/16) + e)
//   accumulator C:  lane l, element e -> (col l % 16, row 8*(l/16) + e)
//
// In S^T each lane holds one query (col) and eight keys (rows 8*(l/16)+e), so
// the tiled max/sum is seven fmax/add over e plus one `xor 16`. The output
// accumulator is [query][d] (eight queries per lane), so the epilogue reads
// the eight running sums from the lanes that hold them (eight shuffles, once).
//
// ## KV source: mirror or quantized cache
//
// The staging pass can fill those same two shared tiles from either the f32
// position-major mirror or the q8_0/q4_0 head-major decode cache. A quantized
// row is 272 B at head_dim 256 instead of 1 KB, and turning a 4-value group
// into halfs costs one f16 scale load and four multiply-converts, so a deep
// prompt reads ~3.8x less from global and still feeds the matrix unit the
// identical f16 operands. That is what lets chunked prefill run at 100K
// without a 12 GiB mirror; the only difference from the mirror is the 8-bit
// cache's own rounding, which the decode path already carries.
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
// V is [d][key] (the PV B operand is row-major by output column): 16 keys
// plus 2 pad halfs. 18 keeps 4 B rows, so a fragment read is four 32-bit
// loads, and makes the staging stores walk all 32 banks (stride 9 words).
#ifndef AW_VST
#define AW_VST 18
#endif
// Queries per block and the GQA group cap the launcher enforces.
#define AW_QUERIES 32
#define AW_MAXGROUP 8

// Cache dtypes the staging pass can read: 0 f32, 1 q8_0, 2 q4_0.
#define AW_DT_F32 0
#define AW_DT_Q8 1
#define AW_DT_Q4 2

typedef _Float16 aw_h8 __attribute__((ext_vector_type(8)));
typedef float aw_f8 __attribute__((ext_vector_type(8)));

#define AW_LOG2E 1.4426950408889634f
#define AW_NEG_INF (-3.402823466e+38f)

// Four consecutive head components of one cache row, as halfs.
//
// `dtype` selects how `row` is addressed: f32 rows are `width` floats wide and
// `row` points at component 0, while a quantized row is a run of 34 B (q8_0) or
// 18 B (q4_0) blocks and `d` picks block and offset. The dequantized values go
// straight to f16 — the matrix unit takes f16 operands either way, so the only
// difference from the f32 mirror is the 8-bit cache's own rounding.
static __device__ __forceinline__ void aw_ld4(
    _Float16* o, const void* row, int d, int dtype)
{
    if (dtype == AW_DT_F32) {
        const float4 a = *reinterpret_cast<const float4*>(
            reinterpret_cast<const float*>(row) + d);
        o[0] = __float2half_rn(a.x);
        o[1] = __float2half_rn(a.y);
        o[2] = __float2half_rn(a.z);
        o[3] = __float2half_rn(a.w);
    } else if (dtype == AW_DT_Q8) {
        const unsigned char* b =
            reinterpret_cast<const unsigned char*>(row) + (size_t)(d >> 5) * 34;
        const float s = __half2float(*reinterpret_cast<const __half*>(b));
        const unsigned w = *reinterpret_cast<const unsigned*>(b + 2 + (d & 31));
        o[0] = __float2half_rn(s * (float)((int)(w << 24) >> 24));
        o[1] = __float2half_rn(s * (float)((int)(w << 16) >> 24));
        o[2] = __float2half_rn(s * (float)((int)(w << 8) >> 24));
        o[3] = __float2half_rn(s * (float)((int)w >> 24));
    } else {
        // q4_0: value j in the low nibble of byte j, j+16 in the high nibble.
        const unsigned char* b =
            reinterpret_cast<const unsigned char*>(row) + (size_t)(d >> 5) * 18;
        const float s = __half2float(*reinterpret_cast<const __half*>(b));
        const unsigned w = *reinterpret_cast<const unsigned*>(b + 2 + (d & 15));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = (int)((w >> (8 * j)) & 0x0f) - 8;
            o[j] = __float2half_rn(s * (float)q);
        }
    }
}

extern "C" __global__ void attn_chunk_wmma_f16(
    const float* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    float* __restrict__ out,
    int n_heads,
    int n_kv,
    int head_dim,
    int width,
    int pos0,
    int tokens,
    float scale,
    int kv_dtype,
    int kv_stride)
{
    __shared__ __align__(16) _Float16 sk[AW_KT * AW_KST];       // [key][d]
    __shared__ __align__(16) _Float16 sv[AW_DHALF * AW_VST];    // [d][key]

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

    // ---- Q fragments, once per block -------------------------------------
    // B operand of S^T is stored row-major by output column, i.e. B_hw is
    // (query, d): lane l holds query l%16 and k = 8*(l/16)+e. That is Q's
    // natural layout, so a fragment is one row of eight contiguous d.
    aw_h8 qt[16];
    {
        const float* qrow = q + (size_t)(tq + frow) * n_heads * head_dim +
                            (size_t)h * head_dim;
        const bool live = tq + frow < tokens;
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

    // One row's byte stride and the gap between KV heads. The f32 mirror is
    // position-major `[pos][n_kv*head_dim]` with the heads packed inside a row;
    // the quantized cache is head-major `[kvh][pos][blocks]`, so `kv_stride`
    // (the cache's per-head row count) is what separates two heads there.
    const int nblock = head_dim >> 5;
    const size_t row_bytes = (kv_dtype == AW_DT_F32)
        ? (size_t)width * 4
        : (size_t)nblock * (kv_dtype == AW_DT_Q8 ? 34 : 18);
    const size_t head_gap = (kv_dtype == AW_DT_F32)
        ? (size_t)head_dim * 4
        : (size_t)kv_stride * row_bytes;
    const char* kb = reinterpret_cast<const char*>(k_cache) + (size_t)kvh * head_gap;
    const char* vb = reinterpret_cast<const char*>(v_cache) + (size_t)kvh * head_gap;
    const int qglob = pos0 + tq + (lane & 15);

    for (int base = 0; base < seq_block; base += AW_KT) {
        const int rows = (base + AW_KT <= seq_block) ? AW_KT : (seq_block - base);

        __syncthreads();

        // K: full head width, [key][d]. One thread fetches four consecutive d
        // of one key (16 B from the f32 mirror, one 4-value group from a q8/q4
        // block) and stores a half4.
        for (int idx = tid; idx < AW_KT * (AW_HD / 4); idx += nthreads) {
            const int key = idx / (AW_HD / 4);
            const int dq = (idx - key * (AW_HD / 4)) * 4;
            const int rr = (key < rows) ? key : (rows - 1);
            aw_ld4(sk + key * AW_KST + dq, kb + (size_t)(base + rr) * row_bytes,
                   dq, kv_dtype);
        }
        // V: this block's d half, transposed to [d][key] because the PV B
        // operand is row-major by output column with eight keys contiguous.
        // A lane fetches four consecutive d of one key and writes four halfs
        // into four d rows; the 18-half stride walks the banks four ways, which
        // is cheaper than four times as many scalar loads.
        for (int idx = tid; idx < AW_KT * (AW_DHALF / 4); idx += nthreads) {
            const int key = idx / (AW_DHALF / 4);
            const int dq = (idx - key * (AW_DHALF / 4)) * 4;
            const int rr = (key < rows) ? key : (rows - 1);
            _Float16 t[4];
            aw_ld4(t, vb + (size_t)(base + rr) * row_bytes,
                   dh * AW_DHALF + dq, kv_dtype);
            sv[(dq + 0) * AW_VST + key] = t[0];
            sv[(dq + 1) * AW_VST + key] = t[1];
            sv[(dq + 2) * AW_VST + key] = t[2];
            sv[(dq + 3) * AW_VST + key] = t[3];
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
        // The cache row index is the global position: the chunk appended its
        // rows at pos0.. and the prefix before them. Only the query carries
        // pos0 (it is at chunk row tq, global pos0 + tq).
        for (int e = 0; e < 8; ++e) {
            const int kg = base + lrow + e;
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
        // tiles). `corr` belongs to this lane's query (col = l%16), while the
        // output rows are queries 8*(l/16)+e, so the factor is broadcast from
        // the lane that owns each row before it touches the accumulators.
        float c[8];
        bool grow = false;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            c[e] = __shfl_sync(0xffffffffffffffffULL, corr, lrow + e);
            grow |= (c[e] != 1.0f);
        }
        if (grow) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j][e] *= c[e];
        }

        // ---- PV: A = P [query][key], B = V^T [d][key] --------------------
        // The A-fragment layout is (row = l%16, k = 8*(l/16)+e), i.e. query
        // l%16 and keys 8*(l/16)+e -- bit-for-bit the score tile's layout, so
        // P goes straight from the softmax registers into the matrix unit.
        aw_h8 ph;
#pragma unroll
        for (int e = 0; e < 8; ++e) ph[e] = __float2half_rn(p[e]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const aw_h8 vf =
                *reinterpret_cast<const aw_h8*>(sv + (j * 16 + frow) * AW_VST + koff);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ph, vf, acc[j]);
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
