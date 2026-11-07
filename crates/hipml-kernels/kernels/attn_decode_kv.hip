// Flash-style decode attention over a *quantized* KV cache.
//
// Two kernels, because one block per query head cannot saturate the card at
// 100K depth:
//
//   attn_decode_kv_q8 / _q4   grid = n_kv * splits, one block per
//                             (kv head, contiguous slice of positions)
//   attn_decode_kv_reduce     grid = n_heads, merges the splits
//
// Block shape is 32 * (n_heads / n_kv) threads: one warp per query head of the
// KV group, so the six Qwen3.8-27B query heads that share a KV head sit in the same
// block and work from the same cache row.
//
// ## Why a shared-memory tile
//
// Reading the cache straight from global in each of the six warps does not
// share anything: the warps drift apart, the row falls out of L1, and DRAM sees
// six reads of every byte. Measured on gfx1201 that pinned the kernel at
// 108 GB/s of *unique* KV traffic while DRAM was already at ~648 GB/s, i.e. the
// six-fold duplication was the whole cost.
//
// So the block stages KV_TILE rows into shared memory once, *dequantized to
// f32*, and the warps then read f32 from shared:
//
//   stage:   192 threads dequantize 2 x KV_TILE x head_dim values
//   compute: each warp does its own q head against the same f32 tile
//
// Dequantizing once instead of once per warp is also what makes q4 cheaper than
// q8 per byte rather than more expensive: at one row per warp per iteration the
// nibble unpacking ran six times.
//
// Layout and the format itself: crates/hipml-core/src/kv.rs. Per value:
//   q8: d * qs[j]                        block = f16 d | 32 i8
//   q4: d * (nibble - 8)                 block = f16 d | 16 B, low = j, high = j+16
// `head_dim` is a multiple of 32 and `head_dim / 32` a power of two.
//
// Partials are fp32: part_acc[n_heads][splits][head_dim], part_ml[n_heads][splits][2]
// with (running max, running sum). The reduce recomputes the merge weights from
// the per-split maxima, so the merge is exact regardless of split order.
//
// gfx1201 is wave32; the reduction walks warpSize so it is correct either way.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define KV_MAX_CHUNKS 8
#define KV_TILE 8

// One block: f16 scale followed by 32 signed bytes.
struct KvQ8 {
    enum { BLOCK_BYTES = 34 };
    // Value `i` of a row, dequantized. i>>5 is the block, i&31 the element.
    static __device__ __forceinline__ float at(const void* row, int i) {
        const char* b = (const char*)row + (i >> 5) * BLOCK_BYTES;
        const float d = __half2float(*(const __half*)b);
        return d * (float)((const signed char*)b)[2 + (i & 31)];
    }
};

// One block: f16 scale followed by 16 bytes, low nibble = value j, high = j+16.
struct KvQ4 {
    enum { BLOCK_BYTES = 18 };
    static __device__ __forceinline__ float at(const void* row, int i) {
        const char* b = (const char*)row + (i >> 5) * BLOCK_BYTES;
        const float d = __half2float(*(const __half*)b);
        const int e = i & 31;
        const int byte = (int)((const unsigned char*)b)[2 + (e & 15)];
        return d * (float)(((e < 16) ? (byte & 15) : (byte >> 4)) - 8);
    }
};

// Butterfly sum: every lane ends with the total, so no broadcast is needed.
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 1; off < warpSize; off <<= 1) {
        v += __shfl_xor_sync(0xffffffffffffffffull, v, off);
    }
    return v;
}

// Dequantize KV_TILE rows of `k` and `v` into shared f32. Every thread walks the
// row with stride blockDim, so a warp reads 32 consecutive payload bytes.
template <typename Loader>
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
    const int nthreads = blockDim.x;
    for (int r = 0; r < rows; ++r) {
        const char* kr = krow + (size_t)r * row_bytes;
        const char* vr = vrow + (size_t)r * row_bytes;
        float* kd = ks + r * head_dim;
        float* vd = vs + r * head_dim;
        for (int i = tid; i < head_dim; i += nthreads) {
            kd[i] = Loader::at(kr, i);
            vd[i] = Loader::at(vr, i);
        }
    }
}

template <typename Loader>
__device__ __forceinline__ void kv_decode_body(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    int seq,
    float scale,
    int splits)
{
    extern __shared__ float sh[];
    const int group = n_heads / n_kv;
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= group) return;

    const int h = kvh * group + warp;
    const int nb = head_dim >> 5;
    const int per = (seq + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;

    // This lane's components: dims c*32 + lane for c in [0, nb).
    float qv[KV_MAX_CHUNKS];
    float acc[KV_MAX_CHUNKS];
    #pragma unroll
    for (int c = 0; c < KV_MAX_CHUNKS; ++c) {
        const int cc = (c < nb) ? c : 0;
        const float x = q[h * head_dim + cc * 32 + lane];
        qv[c] = (c < nb) ? x : 0.0f;
        acc[c] = 0.0f;
    }

    float* ks = sh;
    float* vs = sh + KV_TILE * head_dim;
    const int row_bytes = nb * Loader::BLOCK_BYTES;
    const char* kbase = (const char*)k + (size_t)kvh * seq * row_bytes;
    const char* vbase = (const char*)v + (size_t)kvh * seq * row_bytes;

    float m = -INFINITY;
    float l = 0.0f;

    for (int pos = start; pos < end; pos += KV_TILE) {
        const int rows = (pos + KV_TILE <= end) ? KV_TILE : (end - pos);
        const char* krow = kbase + (size_t)pos * row_bytes;
        const char* vrow = vbase + (size_t)pos * row_bytes;

        // Wait for the previous tile's readers before overwriting the buffer.
        __syncthreads();
        stage_tile<Loader>(ks, vs, krow, vrow, rows, row_bytes, head_dim);
        __syncthreads();

        // One row per iteration: unrolling the whole tile in registers cost
        // 144 VGPRs (two blocks per CU) and lost more to occupancy than it
        // gained. The loads stay branch-free; the 8-row dependency chain is
        // hidden by resident warps instead.
        #pragma unroll 1
        for (int t = 0; t < KV_TILE; ++t) {
            const int r = (t < rows) ? t : 0;
            const float* kr = ks + r * head_dim;
            const float* vr = vs + r * head_dim;
            float dot = 0.0f;
            float vv[KV_MAX_CHUNKS];
            #pragma unroll
            for (int c = 0; c < KV_MAX_CHUNKS; ++c) {
                // Unconditional shared loads: a `(c < nb) ? load : 0` turns into
                // a branch per chunk, which is what made the first version slow.
                // qv is already 0 past nb, so only V needs the mask.
                const int cc = (c < nb) ? c : 0;
                const float mask = (c < nb) ? 1.0f : 0.0f;
                dot += qv[c] * kr[cc * 32 + lane];
                vv[c] = mask * vr[cc * 32 + lane];
            }
            const float s = (t < rows) ? warp_sum(dot) * scale : -INFINITY;
            const float m_new = fmaxf(m, s);
            const float corr = expf(m - m_new);
            const float p = expf(s - m_new);
            l = l * corr + p;
            m = m_new;
            #pragma unroll
            for (int c = 0; c < KV_MAX_CHUNKS; ++c) acc[c] = acc[c] * corr + p * vv[c];
        }
    }

    float* dst = part_acc + ((size_t)h * splits + split) * head_dim;
    #pragma unroll
    for (int c = 0; c < KV_MAX_CHUNKS; ++c) {
        if (c < nb) dst[c * 32 + lane] = acc[c];
    }
    if (lane == 0) {
        float* ml = part_ml + ((size_t)h * splits + split) * 2;
        ml[0] = m;
        ml[1] = l;
    }
}

extern "C" __global__ void attn_decode_kv_q8(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads, int n_kv, int head_dim, int seq, float scale, int splits)
{
    kv_decode_body<KvQ8>(q, k, v, part_acc, part_ml, n_heads, n_kv, head_dim, seq, scale, splits);
}

extern "C" __global__ void attn_decode_kv_q4(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads, int n_kv, int head_dim, int seq, float scale, int splits)
{
    kv_decode_body<KvQ4>(q, k, v, part_acc, part_ml, n_heads, n_kv, head_dim, seq, scale, splits);
}

// Merge per-split (acc, m, l) into out. One block per query head, one thread per
// head component; every thread walks the splits itself because splits is small
// (a few tens) and the per-split (m, l) pair is a broadcast load.
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

    float acc = 0.0f;
    float l = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float w = expf(ml[s * 2] - m);
        l += ml[s * 2 + 1] * w;
        acc += part_acc[((size_t)h * splits + s) * head_dim + t] * w;
    }
    out[(size_t)h * head_dim + t] = (l > 0.0f) ? acc / l : 0.0f;
}
