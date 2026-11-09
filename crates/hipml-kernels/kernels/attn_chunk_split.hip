// Split-K chunk attention: the matrix-unit kernel of `attn_chunk_wmma.hip`
// with the key axis cut into `splits` contiguous slices across blocks, plus the
// combine pass that folds them (flash-decoding, for prefill).
//
// ## Why
//
// The mono kernel gives each (KV head, 32-query tile, d half) block the whole
// causal prefix: at 100K the last chunk's blocks stream ~6,300 key tiles each,
// one after another. A block owns the CU for its whole life (168 VGPRs at 12
// warps is exactly one block per CU), so the staging barrier of every tile sits
// on the critical path with nothing else resident to run. Splitting the key
// axis multiplies the block count by `splits` and divides each block's loop by
// it, which is the only lever that shortens that serial chain without touching
// the tile geometry, the shared traffic or the arithmetic.
//
// This file is a deliberate copy of the mono kernel rather than a template: the
// split path needs a different loop bound, an extra bound in the causal mask
// (a slice must not admit the next slice's keys), a finite seed for the running
// max (a slice can sit entirely above the diagonal) and an epilogue that writes
// unnormalised partials instead of the output. Keeping the shipped kernel
// byte-for-byte unchanged is worth the duplication.
//
// ## Geometry
//
//   blockIdx.x  KV head
//   blockIdx.y  32 queries (two warps, one 16-query fragment each)
//   blockIdx.z  split * 2 + d half
//
// Block shape, tile shapes, the S^T = K Q^T orientation and the staging pass
// are the mono kernel's; see the long note in `attn_chunk_wmma.hip`.
//
// ## Partials
//
//   part_acc[(split * qslots + row) * n_heads + head][head_dim]   fp32
//   part_ml [(split * qslots + row) * n_heads + head][2]          fp32 (max, sum)
//
// `qslots` is the chunk rounded up to a 32-query tile. Max is in log2 units and
// the accumulator is unnormalised against it, exactly the decode split-K
// convention (`attn_decode_kv`), so the combine is a weighted sum.
//
// The running max starts at -1e30 rather than -inf: a slice wholly above the
// diagonal would otherwise compute exp2f(-inf - -inf) = NaN.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#if defined(__AMDGCN_WAVEFRONT_SIZE) && __AMDGCN_WAVEFRONT_SIZE != 32
#error "attn_chunk_split: the gfx12 WMMA intrinsics are wave32 only"
#endif

// Must match the mono kernel: 256-wide heads, 128 output dims per block.
#define SC_HD 256
#define SC_DHALF 128
#define SC_KT 16
#define SC_KST 264
#define SC_VST 18
#define SC_QUERIES 32
// Finite seed for the running max; see the header note.
#define SC_SEED (-1e30f)

#define SC_DT_F32 0
#define SC_DT_Q8 1
#define SC_DT_Q4 2

#define SC_LOG2E 1.4426950408889634f
#define SC_NEG_INF (-3.402823466e+38f)

typedef _Float16 sc_h8 __attribute__((ext_vector_type(8)));
typedef float sc_f8 __attribute__((ext_vector_type(8)));

// Four consecutive head components of one cache row, as halfs. Identical to
// `aw_ld4`: f32 mirror, q8_0 or q4_0 blocks.
template <int DT>
static __device__ __forceinline__ void sc_ld4(_Float16* o, const char* row, int d)
{
    if (DT == SC_DT_F32) {
        const float4 a = *reinterpret_cast<const float4*>(
            reinterpret_cast<const float*>(row) + d);
        o[0] = __float2half_rn(a.x);
        o[1] = __float2half_rn(a.y);
        o[2] = __float2half_rn(a.z);
        o[3] = __float2half_rn(a.w);
    } else if (DT == SC_DT_Q8) {
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

// Stage the 16-key tile: K in full, V for this block's d half. Identical to
// `aw_stage`.
template <int DT>
static __device__ __forceinline__ void sc_stage(
    _Float16* sk, _Float16* sv, const char* kb, const char* vb,
    size_t row_bytes, int base, int rows, int dh, int tid, int nthreads)
{
    for (int idx = tid; idx < SC_KT * (SC_HD / 4); idx += nthreads) {
        const int key = idx / (SC_HD / 4);
        const int dq = (idx - key * (SC_HD / 4)) * 4;
        const int rr = (key < rows) ? key : (rows - 1);
        sc_ld4<DT>(sk + key * SC_KST + dq, kb + (size_t)(base + rr) * row_bytes, dq);
    }
    for (int idx = tid; idx < SC_KT * (SC_DHALF / 4); idx += nthreads) {
        const int key = idx / (SC_DHALF / 4);
        const int dq = (idx - key * (SC_DHALF / 4)) * 4;
        const int rr = (key < rows) ? key : (rows - 1);
        _Float16 t[4];
        sc_ld4<DT>(t, vb + (size_t)(base + rr) * row_bytes, dh * SC_DHALF + dq);
        sv[(dq + 0) * SC_VST + key] = t[0];
        sv[(dq + 1) * SC_VST + key] = t[1];
        sv[(dq + 2) * SC_VST + key] = t[2];
        sv[(dq + 3) * SC_VST + key] = t[3];
    }
}

extern "C" __global__ void attn_chunk_split_f16(
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
    __shared__ __align__(16) _Float16 sk[SC_KT * SC_KST];       // [key][d]
    __shared__ __align__(16) _Float16 sv[SC_DHALF * SC_VST];    // [d][key]

    const int kvh = blockIdx.x;
    const int dh = blockIdx.z & 1;
    const int sp = blockIdx.z >> 1;
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int h = kvh * group + warp / 2;
    const int qi = warp & 1;
    const int tq = blockIdx.y * SC_QUERIES + qi * 16;

    const int frow = lane & 15;
    const int koff = (lane >> 4) * 8;
    const int lrow = 8 * (lane >> 4);

    // ---- Q fragments, once per block --------------------------------------
    sc_h8 qt[16];
    {
        const float* qrow = q + (size_t)(tq + frow) * n_heads * head_dim +
                            (size_t)h * head_dim;
        const bool live = tq + frow < tokens;
#pragma unroll
        for (int dd = 0; dd < 16; ++dd) {
            sc_h8 v;
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

    sc_f8 acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[j][e] = 0.0f;

    float run_max = SC_SEED;   // one query per lane (col = l%16)
    float run_sum = 0.0f;

    int seq_block = pos0 + blockIdx.y * SC_QUERIES + SC_QUERIES;
    if (seq_block > pos0 + tokens) seq_block = pos0 + tokens;

    // This block's slice of the live key range, [lo, hi). Slices are equal in
    // key count, not in tile count, so an unaligned boundary just makes the
    // first tile short.
    const int lo = (int)(((long)seq_block * sp) / splits);
    const int hi = (int)(((long)seq_block * (sp + 1)) / splits);

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int nblock = head_dim >> 5;
    const size_t row_bytes = (kv_dtype == SC_DT_F32)
        ? (size_t)width * 4
        : (size_t)nblock * (kv_dtype == SC_DT_Q8 ? 34 : 18);
    const size_t head_gap = (kv_dtype == SC_DT_F32)
        ? (size_t)head_dim * 4
        : (size_t)kv_stride * row_bytes;
    const char* kb = reinterpret_cast<const char*>(k_cache) + (size_t)kvh * head_gap;
    const char* vb = reinterpret_cast<const char*>(v_cache) + (size_t)kvh * head_gap;
    const int qglob = pos0 + tq + (lane & 15);

    for (int base = lo; base < hi; base += SC_KT) {
        const int rows = (base + SC_KT <= hi) ? SC_KT : (hi - base);

        __syncthreads();

        if (kv_dtype == SC_DT_F32) {
            sc_stage<SC_DT_F32>(sk, sv, kb, vb, row_bytes, base, rows, dh, tid, nthreads);
        } else if (kv_dtype == SC_DT_Q8) {
            sc_stage<SC_DT_Q8>(sk, sv, kb, vb, row_bytes, base, rows, dh, tid, nthreads);
        } else {
            sc_stage<SC_DT_Q4>(sk, sv, kb, vb, row_bytes, base, rows, dh, tid, nthreads);
        }
        __syncthreads();

        // ---- S^T = K Q^T --------------------------------------------------
        sc_f8 s0, s1;
#pragma unroll
        for (int e = 0; e < 8; ++e) { s0[e] = 0.0f; s1[e] = 0.0f; }
        if (dh == 0) { /* PROBE: dh=1 skips QK */
#pragma unroll
        for (int dd = 0; dd < 8; ++dd) {
            const sc_h8 ka =
                *reinterpret_cast<const sc_h8*>(sk + frow * SC_KST + dd * 16 + koff);
            s0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s0);
        }
#pragma unroll
        for (int dd = 8; dd < 16; ++dd) {
            const sc_h8 ka =
                *reinterpret_cast<const sc_h8*>(sk + frow * SC_KST + dd * 16 + koff);
            s1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ka, qt[dd], s1);
        }
        }

        // ---- causal mask + online softmax (log2 units) --------------------
        // `kg < hi` keeps a slice from admitting the next slice's keys: the
        // last tile stages duplicated rows when `hi - base < 16`. Both bounds
        // are warp-uniform, so an interior tile skips the mask entirely: every
        // query of the warp is at or after the tile's last key.
        const float sl2 = scale * SC_LOG2E;
        const bool interior =
            (base + SC_KT <= hi) && (base + SC_KT - 1 <= pos0 + tq);
        float p[8];
        if (interior) {
#pragma unroll
            for (int e = 0; e < 8; ++e) p[e] = (s0[e] + s1[e]) * sl2;
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int kg = base + lrow + e;
                const float s = (s0[e] + s1[e]) * sl2;
                p[e] = (kg <= qglob && kg < hi) ? s : SC_NEG_INF;
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
            for (int j = 0; j < 8; ++j)
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[j][e] *= c[e];
        }

        // ---- PV: A = P [query][key], B = V^T [d][key] --------------------
        sc_h8 ph;
#pragma unroll
        for (int e = 0; e < 8; ++e) ph[e] = __float2half_rn(p[e]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const sc_h8 vf =
                *reinterpret_cast<const sc_h8*>(sv + (j * 16 + frow) * SC_VST + koff);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ph, vf, acc[j]);
        }
    }

    // ---- epilogue: unnormalised partials ----------------------------------
    // No division here: the combine weights each slice by exp2(max - m) and
    // normalises once, so this pass is the mono epilogue minus the reciprocal.
    const int row = tq + lrow;
    const int col = lane & 15;
    const size_t qbase = (size_t)sp * qslots + row;
    const size_t accbase = (qbase * n_heads + h) * head_dim + dh * SC_DHALF + col;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 8; ++e)
            part_acc[accbase + (size_t)e * n_heads * head_dim + j * 16] = acc[j][e];

    if (lane < 16) {
        const size_t mlbase = (((size_t)sp * qslots + tq + lane) * n_heads + h) * 2;
        part_ml[mlbase + 0] = run_max;
        part_ml[mlbase + 1] = run_sum;
    }
}

// Fold the key splits: one block per (chunk row, query head), one thread per
// head component. Every weight is exp2(max_s - max), then one division.
extern "C" __global__ void attn_chunk_combine_f32(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int tokens,
    int qslots,
    int splits)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (t >= tokens || d >= head_dim) return;

    const float* ml = part_ml + ((size_t)t * n_heads + h) * 2;
    const float* ac = part_acc + ((size_t)t * n_heads + h) * head_dim + d;
    const size_t ml_step = (size_t)qslots * n_heads * 2;
    const size_t ac_step = (size_t)qslots * n_heads * head_dim;

    float m = SC_SEED;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, ml[s * ml_step]);

    // Four splits in flight: the accumulators are independent of the sums, so
    // a dependent load per split would leave the block latency-bound.
    float sum = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float w = exp2f(ml[s * ml_step] - m);
        sum += ml[s * ml_step + 1] * w;
        acc += ac[s * ac_step] * w;
    }
    out[((size_t)t * n_heads + h) * head_dim + d] = acc / sum;
}
