// SPDX-License-Identifier: Apache-2.0
// Decode attention over the q8 cache, one warp per (KV head, key range, head
// subset).
//
// The six query heads of a GQA group share one K/V row. Holding all six in one
// warp costs 48 query + 48 accumulator registers before any K/V is loaded, which
// caps occupancy at 10 warps per CU (192 VGPRs) and leaves the kernel
// latency-bound: measured 408 GB/s at 100K, against a 194-instruction/row floor
// and an IPC of 2.2. The same body with `HPW` heads per warp splits the group
// over `ceil(G/HPW)` warps that share a block, so the row is read once into the
// CU's L1 and both warps hit it:
//
//   HPW = G   one warp per (KV head, split): the original shape
//   HPW = 3   two warps per (KV head, split): 24 query + 24 accumulator
//             registers, ~20 warps per CU, and the same row from L1
//
// Per key row a warp issues two dword K payload loads and two f16 scale loads
// (a lane owns four dims in each 128-dim half, so eight lanes cover one block's
// 32-byte payload), HPW eight-deep f32 dot chains each reduced with a five-step
// bpermute butterfly, two dword V loads and HPW PV fma chains.
//
// Nothing is staged and there is no __syncthreads. ROWS key rows are loaded
// before the first is consumed; the score phase needs no V, so both arrays stay
// in flight across the unroll. The online softmax is tile-wide: one rescale of
// the accumulators per tile, one exp2 per (row, head).
//
// ## Numerics / layout
//
// Identical to `attn_decode_kv.hip`: scores in log2 units, per-head (max, sum)
// partials, same split boundaries, same `attn_decode_kv_reduce`. The dot is f32
// throughout (the q8 payload is dequantised to f32 in registers); only the
// lane-to-dimension map differs, so the accumulation order changes at the last
// few bits.
//
// head_dim must be 256 (8 components per lane over 32 lanes) and the GQA group
// at most 8. Both are checked by the launcher; other shapes keep the staged
// kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// exp2 of a score already in log2 units (same constant as the staged kernel).
#define KVW_LOG2E 1.4426950408889634f

// Warps per block. Independent tasks share a block only for scheduling
// granularity; there is no shared memory and no barrier. With HPW < G the
// warps of one task must land in one block so they share the L1 line.
#define KVW_WARPS 4

// The only head width this map covers: 32 lanes x 8 components.
#define KVW_DIM 256

// Lane ownership: dims (c*32 + lane)*4 for c in {0,1}, the same map the staged
// kernel's `ld_row` uses, so partials are read and written identically.
__device__ __forceinline__ int kvw_dim(int c, int lane) { return (c * 32 + lane) << 2; }

// q8 row: 8 blocks of {f16 d; signed char qs[32]}. A lane's four values of
// half `c` live in block (c*4 + lane/8) at payload offset 2 + (lane%8)*4.
__device__ __forceinline__ int kvw_blk(int c, int lane) { return c * 4 + (lane >> 3); }
__device__ __forceinline__ int kvw_off(int lane) { return 2 + ((lane & 7) << 2); }

// Four dequantised values of one q8 block.
__device__ __forceinline__ void kvw_q8x4(
    float* o, const char* row, int c, int lane, int off)
{
    const char* b = row + kvw_blk(c, lane) * 34;
    const float d = __half2float(*(const __half*)b);
    const unsigned w = *(const unsigned*)(b + off);
    o[0] = d * (float)((int)(w << 24) >> 24);
    o[1] = d * (float)((int)(w << 16) >> 24);
    o[2] = d * (float)((int)(w << 8) >> 24);
    o[3] = d * (float)((int)w >> 24);
}

// A lane's eight components of one row (two q8 blocks, four values each).
__device__ __forceinline__ void kvw_row(float* o, const char* row, int lane, int off) {
    kvw_q8x4(o + 0, row, 0, lane, off);
    kvw_q8x4(o + 4, row, 1, lane, off);
}

// Write a lane's eight accumulator components into a [head_dim] partial row.
__device__ __forceinline__ void kvw_put(float* row, const float* o, int lane) {
    #pragma unroll
    for (int c = 0; c < 2; ++c) {
        float4 t;
        t.x = o[c * 4 + 0]; t.y = o[c * 4 + 1];
        t.z = o[c * 4 + 2]; t.w = o[c * 4 + 3];
        *(float4*)__builtin_assume_aligned(row + kvw_dim(c, lane), 16) = t;
    }
}

template <int G, int HPW, int ROWS>
__device__ __forceinline__ void kvw_body(
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
    (void)n_heads;
    (void)head_dim;
    // `step[1]` is the number of valid positions; `stride` is the cache's
    // per-head row count, fixed when the graph is captured.
    const int seq = step[1];
    const int wpr = (G + HPW - 1) / HPW;      // warps per (KV head, split)
    const int gw = blockIdx.x * KVW_WARPS + (threadIdx.x >> 5);
    const int task = gw / wpr;
    const int sub = gw - task * wpr;
    const int lane = threadIdx.x & 31;
    if (task >= n_kv * splits) return;

    const int kvh = task / splits;
    const int split = task - kvh * splits;
    const int g0 = sub * HPW;                 // first head this warp owns
    const int h0 = kvh * G + g0;
    const int row_bytes = 8 * 34;
    const int per = (seq + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;
    const float sl2 = scale * KVW_LOG2E;
    const int off = kvw_off(lane);

    float qv[HPW][8];
    float acc[HPW][8];
    float m[HPW], l[HPW];
    #pragma unroll
    for (int g = 0; g < HPW; ++g) {
        if (g0 + g >= G) break;
        const float* qp = q + (size_t)(h0 + g) * head_dim;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            const float4 t = *(const float4*)(qp + kvw_dim(c, lane));
            qv[g][c * 4 + 0] = t.x;
            qv[g][c * 4 + 1] = t.y;
            qv[g][c * 4 + 2] = t.z;
            qv[g][c * 4 + 3] = t.w;
        }
        #pragma unroll
        for (int d = 0; d < 8; ++d) acc[g][d] = 0.0f;
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    // Butterfly byte offsets (lane^bit, times four) for the five reduce steps.
    int bidx[5];
    #pragma unroll
    for (int j = 0; j < 5; ++j) bidx[j] = ((lane ^ (1 << j)) << 2);

    const char* kbase = (const char*)k + (size_t)kvh * stride * row_bytes;
    const char* vbase = (const char*)v + (size_t)kvh * stride * row_bytes;

    for (int pos = start; pos < end; pos += ROWS) {
        const int rows = (pos + ROWS <= end) ? ROWS : (end - pos);

        // Load the whole tile first: K and V of every row are independent of
        // the score chain, so 4 * ROWS loads are in flight before the first
        // value is consumed. A short final tile re-reads its last valid row
        // and its score is folded to -inf below.
        float kd[ROWS][8], vd[ROWS][8];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int rr = (r < rows) ? r : (rows - 1);
            kvw_row(kd[r], kbase + (size_t)(pos + rr) * row_bytes, lane, off);
            kvw_row(vd[r], vbase + (size_t)(pos + rr) * row_bytes, lane, off);
        }

        // Pass 1: one dot chain per (row, head), each reduced over the warp.
        float s[ROWS][HPW];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            #pragma unroll
            for (int g = 0; g < HPW; ++g) {
                if (g0 + g >= G) break;
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < 8; ++d) dot = fmaf(qv[g][d], kd[r][d], dot);
                #pragma unroll
                for (int j = 0; j < 5; ++j)
                    dot += __int_as_float(
                        __builtin_amdgcn_ds_bpermute(bidx[j], __float_as_int(dot)));
                s[r][g] = (r < rows) ? dot * sl2 : -INFINITY;
            }
        }

        // Pass 2: per-head tile max, one rescale of the accumulators, then one
        // exp2 and one fma chain per (row, head).
        float mt[HPW];
        #pragma unroll
        for (int g = 0; g < HPW; ++g) {
            if (g0 + g >= G) break;
            float t = m[g];
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) t = fmaxf(t, s[r][g]);
            mt[g] = t;
            const float corr = exp2f(m[g] - t);
            l[g] *= corr;
            #pragma unroll
            for (int d = 0; d < 8; ++d) acc[g][d] *= corr;
            m[g] = t;
        }
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (r >= rows) break;
            #pragma unroll
            for (int g = 0; g < HPW; ++g) {
                if (g0 + g >= G) break;
                const float p = exp2f(s[r][g] - mt[g]);
                l[g] += p;
                #pragma unroll
                for (int d = 0; d < 8; ++d) acc[g][d] = fmaf(p, vd[r][d], acc[g][d]);
            }
        }
    }

    #pragma unroll
    for (int g = 0; g < HPW; ++g) {
        if (g0 + g >= G) break;
        const int h = h0 + g;
        kvw_put(part_acc + ((size_t)h * splits + split) * KVW_DIM, acc[g], lane);
        if (lane == 0) {
            float* ml = part_ml + ((size_t)h * splits + split) * 2;
            ml[0] = m[g];
            ml[1] = l[g];
        }
    }
}

// Dispatch: `ALL` keeps the whole GQA group in one warp (HPW = G); otherwise
// three heads per warp, or the group when it is smaller.
template <int ROWS, bool ALL>
__device__ __forceinline__ void kvw_dispatch(
    const float* q,
    const void* k,
    const void* v,
    float* part_acc,
    float* part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    const int* step,
    int stride,
    float scale,
    int splits)
{
#define KVW_ARGS q, k, v, part_acc, part_ml, n_heads, n_kv, head_dim, step, stride, scale, splits
#define KVW_CASE(G)                                                           \
    case G:                                                                   \
        kvw_body<G, (ALL ? G : (G < 3 ? G : 3)), ROWS>(KVW_ARGS);             \
        break;
    switch (n_heads / n_kv) {
    KVW_CASE(1) KVW_CASE(2) KVW_CASE(3) KVW_CASE(4)
    KVW_CASE(5) KVW_CASE(6) KVW_CASE(7) default: KVW_CASE(8)
    }
#undef KVW_ARGS
#undef KVW_CASE
}

#define KVW_ENTRY(NAME, ROWS, ALL)                                            \
    extern "C" __global__ void NAME(                                          \
        const float* __restrict__ q,                                          \
        const void* __restrict__ k,                                           \
        const void* __restrict__ v,                                           \
        float* __restrict__ part_acc,                                         \
        float* __restrict__ part_ml,                                          \
        int n_heads, int n_kv, int head_dim, const int* step, int stride,     \
        float scale, int splits)                                              \
    {                                                                         \
        kvw_dispatch<ROWS, ALL>(q, k, v, part_acc, part_ml, n_heads, n_kv,    \
                                head_dim, step, stride, scale, splits);       \
    }

// Whole group in one warp.
KVW_ENTRY(attn_decode_kv_w2, 2, true)
KVW_ENTRY(attn_decode_kv_w4, 4, true)
// Three heads per warp, two warps per (KV head, split).
KVW_ENTRY(attn_decode_kv_s1, 1, false)
KVW_ENTRY(attn_decode_kv_s2, 2, false)
KVW_ENTRY(attn_decode_kv_s4, 4, false)

#undef KVW_ENTRY

// Batched form: one grid.y step per verify column. The columns share the
// cache, so the K/V base is unchanged; each column carries its own query row,
// its own split scratch slice and its own (position, seq) pair. Slot b runs the
// single-column body at its own position, so a column's partials are the ones
// the one-column launch would have written.
#define KVW_ENTRY_B(NAME, ROWS, ALL)                                          \
    extern "C" __global__ void NAME(                                          \
        const float* __restrict__ q,                                          \
        const void* __restrict__ k,                                           \
        const void* __restrict__ v,                                           \
        float* __restrict__ part_acc,                                         \
        float* __restrict__ part_ml,                                          \
        int n_heads, int n_kv, int head_dim, const int* pos, int stride,      \
        float scale, int splits, int acc_slot, int ml_slot, int q_slot)       \
    {                                                                         \
        const int slot = blockIdx.y;                                          \
        kvw_dispatch<ROWS, ALL>(                                              \
            q + (size_t)slot * q_slot, k, v,                                  \
            part_acc + (size_t)slot * acc_slot,                               \
            part_ml + (size_t)slot * ml_slot,                                 \
            n_heads, n_kv, head_dim, pos + slot * 2, stride, scale, splits);  \
    }

KVW_ENTRY_B(attn_decode_kv_w2_b, 2, true)
KVW_ENTRY_B(attn_decode_kv_w4_b, 4, true)

#undef KVW_ENTRY_B
