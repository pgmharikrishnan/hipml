// Split-K chunk attention with the whole head width in one block.
//
// The shipped matrix-unit kernel (`attn_chunk_wmma.hip`, and its split copy)
// gives each block half of the 256-d output width (`blockIdx.z`), because a
// warp holding all 16 output fragments (128 VGPRs) plus the 16 Q fragments
// (64) does not fit the register budget at the occupancy it wants. The cost is
// that both halves recompute the full 256-d score tile: 288 matrix instructions
// per (head, 32 queries, 256 d, key tile) where 192 are necessary, i.e. 1.5x
// duplicated QK work. A probe that skipped QK on the z=1 half measured the
// attention stage at 95.0 ms instead of 113.3 ms, so deduplicating it is worth
// ~16% of the stage even though the score tiles stay in registers.
//
// This kernel removes the duplication instead of the registers: one block owns
// the whole head width, so the score tile is computed once and consumed by all
// 16 output fragments. The register ceiling is paid for with *fewer warps per
// query row*, not with a shared Q mirror:
//
//   - Each warp is still 16 queries x 256 d: 16 Q fragments + 16 output
//     fragments, ~232 VGPRs. At 232 the compiler's occupancy is 6 waves/SIMD
//     (49152 VGPRs/SIMD / (32 * 232)).
//   - A block is `group * 2` warps (two 16-query tiles per head), i.e. the
//     shipped 12 warps for the shipped group of six. LDS is 17,664 B (K full
//     + V full), so two blocks fit per CU and the register bound is the same
//     two blocks (24 warps/CU).
//   - The K/V tile is no longer re-read per d half, so the block streams
//     K + V once per 64 queries where the shipped kernel streams 1.5x that
//     per 32: staged bytes drop by ~3x.
//
// Geometry:
//
//   blockIdx.x  KV head
//   blockIdx.y  64 queries (four 16-query fragments per head)
//   blockIdx.z  split
//
// The score orientation, the softmax in log2 units, the causal/slice mask, the
// running-max seed and the partial layout are the split kernel's; see the long
// note in `attn_chunk_split.hip`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "attn_chunk_full: the gfx12 WMMA intrinsics are wave32 only"
#endif

#define FC_HD 256
#define FC_KT 16
#define FC_KST 264
#define FC_VST 18
// 16-query tiles per block (per head). Two keeps the 12-warp block of the
// shipped kernel for the shipped GQA group of six, so occupancy is identical
// (two blocks per CU) and the only change is the removed duplicate QK.
#define FC_QTILES 2
#define FC_QUERIES (FC_QTILES * 16)
#define FC_SEED (-1e30f)

#define FC_DT_F32 0
#define FC_DT_Q8 1
#define FC_DT_Q4 2

#define FC_LOG2E 1.4426950408889634f
#define FC_NEG_INF (-3.402823466e+38f)

typedef _Float16 fc_h8 __attribute__((ext_vector_type(8)));
typedef float fc_f8 __attribute__((ext_vector_type(8)));

// Four consecutive head components of one cache row, as halfs. Identical to
// `aw_ld4`: f32 mirror, q8_0 or q4_0 blocks.
template <int DT>
static __device__ __forceinline__ void fc_ld4(_Float16* o, const char* row, int d)
{
    if (DT == FC_DT_F32) {
        const float4 a = *reinterpret_cast<const float4*>(
            reinterpret_cast<const float*>(row) + d);
        o[0] = __float2half_rn(a.x);
        o[1] = __float2half_rn(a.y);
        o[2] = __float2half_rn(a.z);
        o[3] = __float2half_rn(a.w);
    } else if (DT == FC_DT_Q8) {
        const unsigned char* b =
            reinterpret_cast<const unsigned char*>(row) + (size_t)(d >> 5) * 34;
        const float s = __half2float(*reinterpret_cast<const __half*>(b));
        const unsigned w = *reinterpret_cast<const unsigned*>(b + 2 + (d & 31));
        o[0] = __float2half_rn(s * (float)((int)(w << 24) >> 24));
        o[1] = __float2half_rn(s * (float)((int)(w << 16) >> 24));
        o[2] = __float2half_rn(s * (float)((int)(w << 8) >> 24));
        o[3] = __float2half_rn(s * (float)((int)w >> 24));
    } else {
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

// Stage the 16-key tile: K in full, V in full (this block owns every output
// dim). One thread fetches four consecutive d of one key.
template <int DT>
static __device__ __forceinline__ void fc_stage(
    _Float16* sk, _Float16* sv, const char* kb, const char* vb,
    size_t row_bytes, int base, int rows, int tid, int nthreads)
{
    for (int idx = tid; idx < FC_KT * (FC_HD / 4); idx += nthreads) {
        const int key = idx / (FC_HD / 4);
        const int dq = (idx - key * (FC_HD / 4)) * 4;
        const int rr = (key < rows) ? key : (rows - 1);
        fc_ld4<DT>(sk + key * FC_KST + dq, kb + (size_t)(base + rr) * row_bytes, dq);
    }
    // V rides transposed, [d][key], so the PV B fragment (row-major by output
    // column, eight keys contiguous) reads it directly.
    for (int idx = tid; idx < FC_KT * (FC_HD / 4); idx += nthreads) {
        const int key = idx / (FC_HD / 4);
        const int dq = (idx - key * (FC_HD / 4)) * 4;
        const int rr = (key < rows) ? key : (rows - 1);
        _Float16 t[4];
        fc_ld4<DT>(t, vb + (size_t)(base + rr) * row_bytes, dq);
        sv[(dq + 0) * FC_VST + key] = t[0];
        sv[(dq + 1) * FC_VST + key] = t[1];
        sv[(dq + 2) * FC_VST + key] = t[2];
        sv[(dq + 3) * FC_VST + key] = t[3];
    }
}

// Two blocks per CU is the shape: 12 warps each (group 6 x two 16-query
// tiles) and ~235 VGPRs with no spill. Without the bound the compiler caps at
// 192 VGPRs, spills ~56 registers inside the key loop and keeps the same
// 8-wave occupancy, which is strictly worse.
extern "C" __global__ __launch_bounds__(384, 2) void attn_chunk_full_f16(
    const float* __restrict__ q,
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    int width,
    int pos0,
    int tokens,
    float scale,
    int kv_dtype,
    int kv_stride,
    int splits,
    int qslots)
{
    __shared__ __align__(16) _Float16 sk[FC_KT * FC_KST];   // [key][d]
    __shared__ __align__(16) _Float16 sv[FC_HD * FC_VST];   // [d][key]

    const int kvh = blockIdx.x;
    const int sp = blockIdx.z;
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = kvh * group + warp % group;
    const int tq = blockIdx.y * FC_QUERIES + (warp / group) * 16;

    const int frow = lane & 15;
    const int koff = (lane >> 4) * 8;
    const int lrow = 8 * (lane >> 4);

    // ---- Q fragments, once per block -------------------------------------
    fc_h8 qt[16];
    {
        const float* qrow = q + (size_t)(tq + frow) * n_heads * head_dim +
                            (size_t)h * head_dim;
        const bool live = tq + frow < tokens;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            fc_h8 v;
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

    // Output fragments: all 256 d, columns d = j*16 + (l%16).
    fc_f8 acc[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    float run_max = FC_SEED;   // one query per lane (col = l%16)
    float run_sum = 0.0f;

    int seq_block = pos0 + blockIdx.y * FC_QUERIES + FC_QUERIES;
    if (seq_block > pos0 + tokens) seq_block = pos0 + tokens;

    // This block's slice of the live key range, [lo, hi).
    const int lo = (int)(((long)seq_block * sp) / splits);
    const int hi = (int)(((long)seq_block * (sp + 1)) / splits);

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int nblock = head_dim >> 5;
    const size_t row_bytes = (kv_dtype == FC_DT_F32)
        ? (size_t)width * 4
        : (size_t)nblock * (kv_dtype == FC_DT_Q8 ? 34 : 18);
    const size_t head_gap = (kv_dtype == FC_DT_F32)
        ? (size_t)head_dim * 4
        : (size_t)kv_stride * row_bytes;
    const char* kb = reinterpret_cast<const char*>(k_cache) + (size_t)kvh * head_gap;
    const char* vb = reinterpret_cast<const char*>(v_cache) + (size_t)kvh * head_gap;
    const int qglob = pos0 + tq + (lane & 15);

    for (int base = lo; base < hi; base += FC_KT) {
        const int rows = (base + FC_KT <= hi) ? FC_KT : (hi - base);

        __syncthreads();

        if (kv_dtype == FC_DT_F32) {
            fc_stage<FC_DT_F32>(sk, sv, kb, vb, row_bytes, base, rows, tid, nthreads);
        } else if (kv_dtype == FC_DT_Q8) {
            fc_stage<FC_DT_Q8>(sk, sv, kb, vb, row_bytes, base, rows, tid, nthreads);
        } else {
            fc_stage<FC_DT_Q4>(sk, sv, kb, vb, row_bytes, base, rows, tid, nthreads);
        }
        __syncthreads();

        // ---- S^T = K Q^T: the score tile, computed once ------------------
        fc_f8 s0, s1;
#pragma unroll
        for (int e = 0; e < 8; ++e) { s0[e] = 0.0f; s1[e] = 0.0f; }
#pragma unroll
        for (int dd = 0; dd < 8; ++dd) {
            const fc_h8 ka =
                *reinterpret_cast<const fc_h8*>(sk + frow * FC_KST + dd * 16 + koff);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s0);
        }
#pragma unroll
        for (int dd = 8; dd < 16; ++dd) {
            const fc_h8 ka =
                *reinterpret_cast<const fc_h8*>(sk + frow * FC_KST + dd * 16 + koff);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s1);
        }

        // ---- causal mask + online softmax (log2 units) -------------------
        const float sl2 = scale * FC_LOG2E;
        const bool interior =
            (base + FC_KT <= hi) && (base + FC_KT - 1 <= pos0 + tq);
        float p[8];
        if (interior) {
#pragma unroll
            for (int e = 0; e < 8; ++e) p[e] = (s0[e] + s1[e]) * sl2;
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kg = base + lrow + e;
                const float s = (s0[e] + s1[e]) * sl2;
                p[e] = (kg <= qglob && kg < hi) ? s : FC_NEG_INF;
            }
        }
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

        float c[8];
        bool grow = false;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            c[e] = __shfl_sync(0xffffffffffffffffULL, corr, lrow + e);
            grow |= (c[e] != 1.0f);
        }
        if (grow) {
#pragma unroll
            for (int j = 0; j < 16; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j][e] *= c[e];
        }

        // ---- PV: A = P [query][key], B = V^T [d][key] --------------------
        fc_h8 ph;
#pragma unroll
        for (int e = 0; e < 8; ++e) ph[e] = __float2half_rn(p[e]);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const fc_h8 vf =
                *reinterpret_cast<const fc_h8*>(sv + (j * 16 + frow) * FC_VST + koff);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ph, vf, acc[j]);
        }
    }

    // ---- epilogue: unnormalised partials ----------------------------------
    // Same layout as the split kernel, without the d-half index. This block
    // spans 64 queries while the partial scratch is 32-rounded, so the last
    // block of a chunk must not write past `qslots`.
    const int row = tq + lrow;
    const int col = lane & 15;
    const size_t qbase = (size_t)sp * qslots + row;
    const size_t accbase = (qbase * n_heads + h) * head_dim + col;
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            if (row + e < qslots)
                part_acc[accbase + (size_t)e * n_heads * head_dim + j * 16] = acc[j][e];

    if (lane < 16 && tq + lane < qslots) {
        const size_t mlbase = (((size_t)sp * qslots + tq + lane) * n_heads + h) * 2;
        part_ml[mlbase + 0] = run_max;
        part_ml[mlbase + 1] = run_sum;
    }
}
