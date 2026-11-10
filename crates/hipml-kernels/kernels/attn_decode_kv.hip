// SPDX-License-Identifier: Apache-2.0
// Flash-style decode attention over a *quantized* KV cache.
//
//   attn_decode_kv_q8 / _q4   grid = n_kv * splits, one block per
//                             (kv head, contiguous slice of positions)
//   attn_decode_kv_reduce     grid = n_heads, merges the splits
//
// Block shape is 32 * (n_heads / n_kv) threads: one warp per query head of the
// KV group, so the six Qwen3.8-27B query heads that share a KV head sit in the same
// block and work from the same cache row.
//
// A block stages a tile of rows into shared memory once, *dequantized to f32*.
// Reading from global per warp instead shares nothing: the six reads of every
// byte measured 108 GB/s of unique KV traffic while DRAM was at ~648 GB/s.
// Dequantizing once is also what makes q4 cheaper than q8 per byte.
//
// ## Instruction budget (what this file is actually about)
//
// At 100K the old loop issued 243 instructions per cached value per warp, of
// which ~30 were arithmetic: 16 scalar shared loads plus address math, and a
// __shfl_xor_sync expansion of ~12 instructions per butterfly step (60 per row).
// Staging was worse: ~43 instructions per dequantized value because each value
// recomputed a 64-bit address and re-loaded the f16 scale. Four changes:
//
//   1. Vector ownership: a lane owns `head_dim / 32` components as float4 chunks
//      at (c*32 + lane)*4. Two LDS.128 per K row and two per V row replace
//      sixteen LDS.32, and each 8-lane 128-bit group covers 128 contiguous
//      bytes, so the loads are conflict-free without a swizzle.
//   2. Tile-wide online softmax: the tile's scores are computed first (the
//      dot/reduce chains overlap), the rescale runs once per tile instead of
//      once per row, and p needs one exp2 per row.
//   3. Butterfly reduce through __builtin_amdgcn_ds_bpermute with the five
//      lane^bit byte offsets hoisted: five bpermute + five add, no exec-mask
//      bookkeeping, instead of ~60 instructions.
//   4. Staging issues a whole tile's loads before the first is consumed (four
//      values per 32-bit payload load, one LDS.128 store) and walks the rows
//      with pointers instead of a fresh 64-bit scaled index.
//
// ## Numerics / layout
//
// Scores are kept in log2 units (`s * scale * log2e`) so the softmax is a bare
// exp2f (gfx12's v_exp_f32; expf would expand to a range reduction). `part_ml`
// therefore carries (max, sum) with the max in log2 units, and the reduce kernel
// uses exp2f directly. Partials: part_acc[n_heads][splits][head_dim],
// part_ml[n_heads][splits][2], fp32. The butterfly is 5 steps, every lane on
// either wave size because (lane^bit)*4 is a valid bpermute byte offset <= 255.
// Format and encoding: crates/hipml-core/src/kv.rs. `head_dim` is a multiple of
// 32 and `head_dim / 32` a power of two.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Tile sizes. q8 stages twice the bytes per row, so it wants the smaller tile
// (8 KB per block, 8 blocks in a 64 KB CU); q4 fits 16 KB and trades that
// occupancy for half as many barriers. The launcher sizes shared memory to
// match the entry point it calls.
#define KV_TILE_Q8 4
#define KV_TILE_Q4 8

// exp2 of the score already in log2 units. One hardware v_exp_f32.
#define KV_LOG2E 1.4426950408889634f

// Each loader splits its work so the staging pass can put every global load of
// a tile in flight before the first one is consumed: `ld4p` fetches the scale
// and the 32-bit payload of a four-value group, `st4` dequantizes and stores it.

// One block: f16 scale followed by 32 signed bytes; `i` is a multiple of four,
// so a four-value group never crosses a block.
struct KvQ8 {
    enum { BLOCK_BYTES = 34 };
    static __device__ __forceinline__ int scale_off(int i) { return (i >> 5) * BLOCK_BYTES; }
    static __device__ __forceinline__ int payload_off(int i) {
        return (i >> 5) * BLOCK_BYTES + 2 + (i & 31);
    }
    enum { PAIR_OFF = 0 };   // one dword covers four values
    static __device__ __forceinline__ int ngroups(int head_dim) { return head_dim >> 2; }
    static __device__ __forceinline__ int group_i(int l, int) { return l << 2; }
    static __device__ __forceinline__ void ld4p(
        unsigned& w, float& d, const char* pl, const char* sc) {
        d = __half2float(*(const __half*)sc);
        w = *(const unsigned*)pl;
    }
    static __device__ __forceinline__ void st4(float* o, int i, unsigned w, float d) {
        (void)i;
        float4 t;
        t.x = d * (float)((int)(w << 24) >> 24);
        t.y = d * (float)((int)(w << 16) >> 24);
        t.z = d * (float)((int)(w << 8) >> 24);
        t.w = d * (float)((int)w >> 24);
        *(float4*)__builtin_assume_aligned(o, 16) = t;
    }
};

// One block: f16 scale followed by 16 bytes, low nibble = value j, high = j+16.
struct KvQ4 {
    enum { BLOCK_BYTES = 18 };
    static __device__ __forceinline__ int scale_off(int i) { return (i >> 5) * BLOCK_BYTES; }
    static __device__ __forceinline__ int payload_off(int i) {
        return (i >> 5) * BLOCK_BYTES + 2 + (i & 15);
    }
    // One payload dword holds four low nibbles and the four high nibbles that
    // live sixteen values later, so a thread produces eight values from one
    // load: the second group is staged at PAIR_OFF with the same dword.
    enum { PAIR_OFF = 16 };
    static __device__ __forceinline__ int ngroups(int head_dim) { return head_dim >> 3; }
    static __device__ __forceinline__ int group_i(int l, int) {
        return (l >> 2) * 32 + (l & 3) * 4;   // low nibbles only; the pair is +16
    }
    static __device__ __forceinline__ void ld4p(
        unsigned& w, float& d, const char* pl, const char* sc) {
        d = __half2float(*(const __half*)sc);
        w = *(const unsigned*)pl;
    }
    static __device__ __forceinline__ void st4(float* o, int i, unsigned w, float d) {
        const int sh = (i & 16) >> 2;   // 0 for the low nibbles, 4 for the high
        const float md = -8.0f * d;
        float4 t;
        t.x = fmaf((float)((w >> sh) & 15), d, md);
        t.y = fmaf((float)((w >> (sh + 8)) & 15), d, md);
        t.z = fmaf((float)((w >> (sh + 16)) & 15), d, md);
        t.w = fmaf((float)((w >> (sh + 24)) & 15), d, md);
        *(float4*)__builtin_assume_aligned(o, 16) = t;
    }
};

// Which vector width a lane can use, and how many chunks of that width cover
// its head_dim/32 components. 256 -> two float4, 64 -> one float2, odd
// head_dims fall back to the scalar chunk-strided ownership.
template <int DPL>
struct Lanes {
    enum { W = (DPL % 4 == 0) ? 4 : ((DPL % 2 == 0) ? 2 : 1) };
    enum { NC = DPL / W };
};

// A lane's W components, W in {1,2,4}: W is a template constant, so only one
// branch survives, and the float4 form turns 4 LDS.32 into one LDS.128.
template <int W>
__device__ __forceinline__ void vget(float* o, const float* p) {
    if (W == 4) {
        const float4 t = *(const float4*)__builtin_assume_aligned(p, 16);
        o[0] = t.x; o[1] = t.y; o[2] = t.z; o[3] = t.w;
    } else if (W == 2) {
        const float2 t = *(const float2*)p;
        o[0] = t.x; o[1] = t.y;
    } else {
        o[0] = p[0];
    }
}

template <int W>
__device__ __forceinline__ void vput(float* p, const float* o) {
    if (W == 4) {
        float4 t;
        t.x = o[0]; t.y = o[1]; t.z = o[2]; t.w = o[3];
        *(float4*)__builtin_assume_aligned(p, 16) = t;
    } else if (W == 2) {
        float2 t;
        t.x = o[0]; t.y = o[1];
        *(float2*)p = t;
    } else {
        p[0] = o[0];
    }
}

// A lane's components of one row, at (c*32 + lane)*W .. +W.
template <int DPL>
__device__ __forceinline__ void ld_row(float* o, const float* base, int lane) {
    const int W = Lanes<DPL>::W;
    const int NC = Lanes<DPL>::NC;
    #pragma unroll
    for (int c = 0; c < NC; ++c) vget<W>(o + c * W, base + (c * 32 + lane) * W);
}

template <int DPL>
__device__ __forceinline__ void st_row(float* base, const float* o, int lane) {
    const int W = Lanes<DPL>::W;
    const int NC = Lanes<DPL>::NC;
    #pragma unroll
    for (int c = 0; c < NC; ++c) vput<W>(base + (c * 32 + lane) * W, o + c * W);
}

// Staging, per thread: one group of one array (K for the first half of the
// block, V for the second) for every row of the tile. Two per-thread pointers
// advance by row_bytes / head_dim, so a row costs one add instead of a fresh
// 64-bit scaled index, and the whole tile's loads issue before the first store.
template <typename Loader, int TILE>
__device__ __forceinline__ void stage_tile(
    float* ks,
    float* vs,
    const char* krow,
    const char* vrow,
    int rows,
    int row_bytes,
    int head_dim)
{
    const int tid = threadIdx.x;
    const int half = blockDim.x >> 1;
    const bool is_k = tid < half;
    const int local = is_k ? tid : (tid - half);
    const char* row0 = is_k ? krow : vrow;
    float* dst0 = is_k ? ks : vs;
    const int ng = Loader::ngroups(head_dim);
    if (local >= ng) return;
    const int i = Loader::group_i(local, head_dim);

    const int po = Loader::payload_off(i);
    const int so = Loader::scale_off(i);
    const char* pl = row0 + po;
    const char* sc = row0 + so;
    float* q = dst0 + i;
    unsigned w[TILE];
    float d[TILE];

    // Walk the rows with pointers; once the last valid row is reached the
    // advance is zero, so a short tile re-reads that row instead of branching.
    #pragma unroll
    for (int r = 0; r < TILE; ++r) {
        Loader::ld4p(w[r], d[r], pl, sc);
        const int adv = (r + 1 < rows) ? row_bytes : 0;
        pl += adv;
        sc += adv;
    }
    #pragma unroll
    for (int r = 0; r < TILE; ++r) {
        Loader::st4(q, i, w[r], d[r]);
        if (Loader::PAIR_OFF) {
            Loader::st4(q + Loader::PAIR_OFF, i + Loader::PAIR_OFF, w[r], d[r]);
        }
        q += (r + 1 < rows) ? head_dim : 0;
    }

    // Blocks with fewer staging threads than groups stage the rest fused; only
    // small GQA groups take this path.
    for (int l2 = local + half; l2 < ng; l2 += half) {
        const int i2 = Loader::group_i(l2, head_dim);
        const char* p2 = row0 + Loader::payload_off(i2);
        const char* s2 = row0 + Loader::scale_off(i2);
        #pragma unroll
        for (int r = 0; r < TILE; ++r) {
            const int rr = (r < rows) ? r : (rows - 1);
            unsigned ww;
            float dd;
            Loader::ld4p(ww, dd, p2 + rr * row_bytes, s2 + rr * row_bytes);
            float* o = dst0 + rr * head_dim + i2;
            Loader::st4(o, i2, ww, dd);
            if (Loader::PAIR_OFF) Loader::st4(o + Loader::PAIR_OFF, i2 + Loader::PAIR_OFF, ww, dd);
        }
    }
}

template <typename Loader, int DPL, int TILE>
__device__ __forceinline__ void kv_decode_body(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    const int* __restrict__ step,
    int stride,
    float scale,
    int splits)
{
    // `step[1]` is how many positions are valid (device memory: one graph is
    // replayed at every step); `stride` is the cache's per-head row count and is
    // fixed at graph build.
    const int seq = step[1];
    // 16-byte aligned so a lane's float4 chunk really becomes one ds_load_b128;
    // with plain float alignment clang splits it into ds_load_2addr_b32 pairs.
    extern __shared__ __align__(16) float sh[];
    const int group = n_heads / n_kv;
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= group) return;

    const int h = kvh * group + warp;
    const int nb = head_dim >> 5;
    // The split partition is fixed in cache positions (`stride`, the cache's
    // per-head row count) rather than in `seq`, so a launch at position `p` and
    // the verify pass's launch for its column share the same (max, sum, acc)
    // grouping. Splits past `seq` are skipped, and rows past `seq` in the last
    // tile are folded to -inf, exactly as a shorter launch would.
    const int per = (stride + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;
    const float sl2 = scale * KV_LOG2E;

    float qv[DPL];
    float acc[DPL];
    ld_row<DPL>(qv, q + (size_t)h * head_dim, lane);
    #pragma unroll
    for (int d = 0; d < DPL; ++d) acc[d] = 0.0f;

    // Byte offsets for the five butterfly steps: each is lane^bit, times four
    // because ds_bpermute indexes bytes. Constant for the whole kernel.
    int bidx[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) bidx[j] = ((lane ^ (1 << j)) << 2);

    float* ks = sh;
    float* vs = sh + TILE * head_dim;
    const int row_bytes = nb * Loader::BLOCK_BYTES;
    const char* kbase = (const char*)k + (size_t)kvh * stride * row_bytes;
    const char* vbase = (const char*)v + (size_t)kvh * stride * row_bytes;

    float m = -INFINITY;   // log2 units
    float l = 0.0f;

    for (int pos = start; pos < end; pos += TILE) {
        const int rows = (pos + TILE <= end) ? TILE : (end - pos);

        // Wait for the previous tile's readers before overwriting the buffer.
        __syncthreads();
        stage_tile<Loader, TILE>(
            ks, vs,
            kbase + (size_t)pos * row_bytes,
            vbase + (size_t)pos * row_bytes,
            rows, row_bytes, head_dim);
        __syncthreads();

        // Pass 1: the tile's scores, one dot/reduce chain per row. Short rows
        // are folded to -inf so pass 2 needs no row predicate.
        float s[TILE];
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            float kd[DPL];
            ld_row<DPL>(kd, ks + t * head_dim, lane);
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < DPL; ++d) dot = fmaf(qv[d], kd[d], dot);
            #pragma unroll
            for (int j = 0; j < 5; ++j) {
                dot += __int_as_float(
                    __builtin_amdgcn_ds_bpermute(bidx[j], __float_as_int(dot)));
            }
            s[t] = (t < rows) ? dot * sl2 : -INFINITY;
        }

        // Pass 2: one rescale per tile, one exp2 per row, one fma per component.
        float mt = m;
        #pragma unroll
        for (int t = 0; t < TILE; ++t) mt = fmaxf(mt, s[t]);
        const float corr = exp2f(m - mt);
        l *= corr;
        #pragma unroll
        for (int d = 0; d < DPL; ++d) acc[d] *= corr;
        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            // A short final tile leaves shared rows >= `rows` uninitialized.
            // Their score is -INFINITY (p == 0), but 0 * uninitialized can be
            // NaN, so do not read them at all.
            if (t >= rows) break;
            const float p = exp2f(s[t] - mt);
            l += p;
            float vd[DPL];
            ld_row<DPL>(vd, vs + t * head_dim, lane);
            #pragma unroll
            for (int d = 0; d < DPL; ++d) acc[d] = fmaf(p, vd[d], acc[d]);
        }
        m = mt;
    }

    float* dst = part_acc + ((size_t)h * splits + split) * head_dim;
    st_row<DPL>(dst, acc, lane);
    if (lane == 0) {
        float* ml = part_ml + ((size_t)h * splits + split) * 2;
        ml[0] = m;
        ml[1] = l;
    }
}

// head_dim/32 is the per-lane component count; one instantiation per supported
// head_dim so every shared access stays a vector.
template <typename Loader, int TILE>
__device__ __forceinline__ void kv_dispatch(
    const float* q,
    const void* k,
    const void* v,
    float* part_acc,
    float* part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    const int* __restrict__ step,
    int stride,
    float scale,
    int splits)
{
#define KV_ARGS q, k, v, part_acc, part_ml, n_heads, n_kv, head_dim, step, stride, scale, splits
#define KV_CASE(D) case D: kv_decode_body<Loader, D, TILE>(KV_ARGS); break;
    switch (head_dim >> 5) {
    KV_CASE(1) KV_CASE(2) KV_CASE(3) KV_CASE(4)
    KV_CASE(5) KV_CASE(6) KV_CASE(7)
    default: KV_CASE(8)
    }
#undef KV_ARGS
#undef KV_CASE
}

#define KV_ENTRY(NAME, LOADER, TILE)                                          \
    extern "C" __global__ void NAME(                                          \
        const float* __restrict__ q,                                          \
        const void* __restrict__ k,                                           \
        const void* __restrict__ v,                                           \
        float* __restrict__ part_acc,                                         \
        float* __restrict__ part_ml,                                          \
        int n_heads, int n_kv, int head_dim, const int* step, int stride,     \
        float scale, int splits)                                              \
    {                                                                         \
        kv_dispatch<LOADER, TILE>(                                            \
            q, k, v, part_acc, part_ml,                                       \
            n_heads, n_kv, head_dim, step, stride, scale, splits);            \
    }

KV_ENTRY(attn_decode_kv_q8, KvQ8, KV_TILE_Q8)
KV_ENTRY(attn_decode_kv_q4, KvQ4, KV_TILE_Q4)

#undef KV_ENTRY

// Batched form: one grid.y step per slot, one query row / cache per slot. The
// slot strides are base pointers advanced here, so the body above is reused
// unchanged and slot b computes exactly the single-slot arithmetic.
#define KV_ENTRY_B(NAME, LOADER, TILE)                                        \
    extern "C" __global__ void NAME(                                          \
        const float* __restrict__ q,                                          \
        const void* __restrict__ k,                                           \
        const void* __restrict__ v,                                           \
        float* __restrict__ part_acc,                                         \
        float* __restrict__ part_ml,                                          \
        int n_heads, int n_kv, int head_dim, const int* pos, int stride,      \
        float scale, int splits, int kv_slot_bytes, int acc_slot, int ml_slot,\
        int q_slot)                                                           \
    {                                                                         \
        const int slot = blockIdx.y;                                          \
        kv_dispatch<LOADER, TILE>(                                            \
            q + (size_t)slot * q_slot,                                        \
            (const char*)k + (size_t)slot * kv_slot_bytes,                    \
            (const char*)v + (size_t)slot * kv_slot_bytes,                    \
            part_acc + (size_t)slot * acc_slot,                               \
            part_ml + (size_t)slot * ml_slot,                                 \
            n_heads, n_kv, head_dim, pos + slot * 2, stride, scale, splits);  \
    }

KV_ENTRY_B(attn_decode_kv_q8_b, KvQ8, KV_TILE_Q8)
KV_ENTRY_B(attn_decode_kv_q4_b, KvQ4, KV_TILE_Q4)

#undef KV_ENTRY_B

// Batched reduce: one grid.y step per slot.
extern "C" __global__ void attn_decode_kv_reduce_b(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    int head_dim,
    int splits,
    int acc_slot,
    int ml_slot,
    int out_slot)
{
    const int h = blockIdx.x;
    const int slot = blockIdx.y;
    const int t = threadIdx.x;
    if (t >= head_dim) return;

    const float* ml = part_ml + (size_t)slot * ml_slot + (size_t)h * splits * 2;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, ml[s * 2]);

    float acc = 0.0f;
    float l = 0.0f;
    const float* pa = part_acc + (size_t)slot * acc_slot
                    + (size_t)h * splits * head_dim + t;
    for (int s = 0; s < splits; ++s) {
        const float w = exp2f(ml[s * 2] - m);
        l += ml[s * 2 + 1] * w;
        acc += pa[(size_t)s * head_dim] * w;
    }
    out[(size_t)slot * out_slot + (size_t)h * head_dim + t] =
        (l > 0.0f) ? acc / l : 0.0f;
}

// Merge per-split (acc, m, l) into out. One block per query head, one thread per
// head component; every thread walks the splits itself because splits is small
// (a few tens) and the per-split (m, l) pair is a broadcast load.
//
// `m` is carried in log2 units by both kernels, so the weight is a bare exp2.
extern "C" __global__ void attn_decode_kv_reduce(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    int head_dim,
    int splits)
{
    const int h = blockIdx.x;
    const int t = threadIdx.x;
    if (t >= head_dim) return;

    const float* ml = part_ml + (size_t)h * splits * 2;
    float m = -INFINITY;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, ml[s * 2]);

    // Eight loads in flight, same arithmetic order: with 128 splits a
    // dependent load per iteration leaves the 24 blocks of this grid with one
    // 4-byte request per thread outstanding, which is latency, not bandwidth.
    float acc = 0.0f;
    float l = 0.0f;
    const float* pa = part_acc + (size_t)h * splits * head_dim + t;
    int s = 0;
    for (; s + 8 <= splits; s += 8) {
        float v[8];
        float w[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = pa[(size_t)(s + j) * head_dim];
#pragma unroll
        for (int j = 0; j < 8; ++j) w[j] = exp2f(ml[(s + j) * 2] - m);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            l += ml[(s + j) * 2 + 1] * w[j];
            acc += v[j] * w[j];
        }
    }
    for (; s < splits; ++s) {
        const float w = exp2f(ml[s * 2] - m);
        l += ml[s * 2 + 1] * w;
        acc += pa[(size_t)s * head_dim] * w;
    }
    out[(size_t)h * head_dim + t] = (l > 0.0f) ? acc / l : 0.0f;
}
