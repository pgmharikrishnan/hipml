// Split-K decode attention over the position-major f32 KV cache.
//
// The single-block-per-head kernel (`attn_decode_f32`) reduces each row with
// five block barriers, so at 100K it is latency-bound: 24 blocks replay the
// same cache rows six times (GQA) and pay ~1.6M barriers per layer. This kernel
// keeps the f32 layout but copies the structure of `attn_decode_kv`:
//
//   attn_decode_split_f32   grid = n_kv * splits
//                           block = (n_heads / n_kv) warps, one per query head
//                           of the KV group; each warp lane owns head_dim/32
//                           components, so the dot is a warp shuffle (no
//                           __syncthreads in the row loop)
//   attn_reduce_f32         grid = n_heads, merges the splits
//
// A tile of TILE rows is staged once per block in shared memory, so the six
// query heads of a KV group read each K/V row once, not six times. Partials are
// part_acc[n_heads][splits][head_dim] and part_ml[n_heads][splits][2] (max and
// sum, natural log units, matching the reference's expf).
//
// Requires head_dim % 32 == 0 and head_dim / 32 <= 8 (head_dim <= 256). The
// launcher falls back to attn_decode_f32 otherwise.

#include <hip/hip_runtime.h>

#define SPLIT_TILE 8
#define MAX_DPL 8

extern "C" __global__ void attn_decode_split_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    int seq,
    float scale,
    int splits,
    int width)
{
    extern __shared__ float sh[];
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int dpl = head_dim >> 5;
    const int h = kvh * group + warp;

    const int per = (seq + splits - 1) / splits;
    const int start = split * per;
    const int end = (start + per < seq) ? start + per : seq;

    if (start >= seq) {
        #pragma unroll
        for (int c = 0; c < MAX_DPL; ++c) {
            if (c < dpl) part_acc[((size_t)h * splits + split) * head_dim + c * 32 + lane] = 0.0f;
        }
        if (lane == 0) {
            float* ml = part_ml + ((size_t)h * splits + split) * 2;
            ml[0] = -3.402823466e+38f;
            ml[1] = 0.0f;
        }
        return;
    }

    float qv[MAX_DPL];
    float acc[MAX_DPL];
    #pragma unroll
    for (int c = 0; c < MAX_DPL; ++c) {
        qv[c] = (c < dpl) ? q[(size_t)h * head_dim + c * 32 + lane] : 0.0f;
        acc[c] = 0.0f;
    }

    float* ks = sh;
    float* vs = sh + SPLIT_TILE * head_dim;
    float m = -3.402823466e+38f;
    float lsum = 0.0f;

    for (int pos = start; pos < end; pos += SPLIT_TILE) {
        const int rows = (pos + SPLIT_TILE <= end) ? SPLIT_TILE : (end - pos);

        __syncthreads();
        // Stage K and V for the tile. Rows past `rows` re-read the last valid
        // row; their scores are folded to -inf below.
        const int nload = 2 * SPLIT_TILE * head_dim;
        for (int idx = threadIdx.x; idx < nload; idx += blockDim.x) {
            const int arr = idx / (SPLIT_TILE * head_dim);
            const int rem = idx - arr * SPLIT_TILE * head_dim;
            const int r = rem / head_dim;
            const int d = rem - r * head_dim;
            const int rr = (r < rows) ? r : (rows - 1);
            const float* base = (arr == 0) ? k : v;
            float* dst = (arr == 0) ? ks : vs;
            dst[r * head_dim + d] = base[(size_t)(pos + rr) * width + kvh * head_dim + d];
        }
        __syncthreads();

        float s[SPLIT_TILE];
        #pragma unroll
        for (int t = 0; t < SPLIT_TILE; ++t) {
            float dot = 0.0f;
            #pragma unroll
            for (int c = 0; c < MAX_DPL; ++c) {
                if (c < dpl) dot = fmaf(qv[c], ks[t * head_dim + c * 32 + lane], dot);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot += __shfl_xor_sync(0xffffffffffffffffULL, dot, off);
            }
            s[t] = (t < rows) ? dot * scale : -3.402823466e+38f;
        }

        float mt = m;
        #pragma unroll
        for (int t = 0; t < SPLIT_TILE; ++t) mt = fmaxf(mt, s[t]);
        const float corr = expf(m - mt);
        lsum *= corr;
        #pragma unroll
        for (int c = 0; c < MAX_DPL; ++c) acc[c] *= corr;

        #pragma unroll
        for (int t = 0; t < SPLIT_TILE; ++t) {
            const float p = expf(s[t] - mt);
            lsum += p;
            #pragma unroll
            for (int c = 0; c < MAX_DPL; ++c) {
                if (c < dpl) {
                    acc[c] = fmaf(p, vs[t * head_dim + c * 32 + lane], acc[c]);
                }
            }
        }
        m = mt;
    }

    #pragma unroll
    for (int c = 0; c < MAX_DPL; ++c) {
        if (c < dpl) {
            part_acc[((size_t)h * splits + split) * head_dim + c * 32 + lane] = acc[c];
        }
    }
    if (lane == 0) {
        float* ml = part_ml + ((size_t)h * splits + split) * 2;
        ml[0] = m;
        ml[1] = lsum;
    }
}

// Merge split partials. One block per query head, one thread per component.
extern "C" __global__ void attn_reduce_f32(
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
    float m = -3.402823466e+38f;
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
