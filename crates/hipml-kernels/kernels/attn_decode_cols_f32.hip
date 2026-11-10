// SPDX-License-Identifier: Apache-2.0
// R-column split-K decode attention over the position-major f32 KV cache.
//
// One launch verifies every speculative column. The split partition is fixed
// in *positions*, not in sequence length: split `s` owns
// `[s*per, (s+1)*per)` and the launcher passes `per = ceil(n_ctx / splits)`.
// That makes the partition independent of `seq`, which is what lets a column
// with a shorter sequence be folded into the same sweep: rows at or past its
// own bound are folded to -inf, and a folded row contributes exactly zero
// (`expf(-inf) == 0`), so the (max, sum, acc) updates are the very same
// operations a one-column launch with that `seq` performs. The single-column
// decode path calls this kernel with `rows = 1`, so both paths share one
// arithmetic.
//
// Column `r` is at absolute position `pos0 + r` and may attend to positions
// `0..=pos0 + r` (its own token is already in the cache: the append happens
// first).
//
// Partials are `part_acc[rows][n_heads][splits][head_dim]` and
// `part_ml[rows][n_heads][splits][2]`; `attn_reduce_cols_f32` merges them.
//
// Requires head_dim % 32 == 0 and head_dim / 32 <= 8 (head_dim <= 256).

#include <hip/hip_runtime.h>

#define AC_TILE 8
#define AC_MAX_DPL 8
#define AC_MAX_COLS 4
#define AC_NEG_INF -3.402823466e+38f

extern "C" __global__ void attn_decode_cols_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ part_acc,
    float* __restrict__ part_ml,
    int n_heads,
    int n_kv,
    int head_dim,
    int pos0,
    int rows,
    float scale,
    int splits,
    int per,
    int width)
{
    // Last column's bound: it reads positions 0..pos0+rows-1.
    const int seq_max = pos0 + rows;
    extern __shared__ float sh[];
    const int kvh = blockIdx.x / splits;
    const int split = blockIdx.x - kvh * splits;
    const int group = n_heads / n_kv;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int dpl = head_dim >> 5;
    const int h = kvh * group + warp;

    const int start = split * per;
    const int end = (start + per < seq_max) ? start + per : seq_max;

    if (start >= seq_max) {
        // Nothing in this slice: the reduce folds an empty partial away.
        for (int r = 0; r < rows; ++r) {
            const size_t hb = ((size_t)r * n_heads + h) * splits + split;
            #pragma unroll
            for (int c = 0; c < AC_MAX_DPL; ++c) {
                if (c < dpl) {
                    part_acc[hb * head_dim + c * 32 + lane] = 0.0f;
                }
            }
            if (lane == 0) {
                float* ml = part_ml + hb * 2;
                ml[0] = AC_NEG_INF;
                ml[1] = 0.0f;
            }
        }
        return;
    }

    float qv[AC_MAX_COLS][AC_MAX_DPL];
    float acc[AC_MAX_COLS][AC_MAX_DPL];
    float m[AC_MAX_COLS];
    float lsum[AC_MAX_COLS];
    #pragma unroll
    for (int r = 0; r < AC_MAX_COLS; ++r) {
        m[r] = AC_NEG_INF;
        lsum[r] = 0.0f;
        #pragma unroll
        for (int c = 0; c < AC_MAX_DPL; ++c) {
            qv[r][c] = 0.0f;
            acc[r][c] = 0.0f;
            if (r < rows && c < dpl) {
                qv[r][c] = q[((size_t)r * n_heads + h) * head_dim + c * 32 + lane];
            }
        }
    }

    float* ks = sh;
    float* vs = sh + AC_TILE * head_dim;

    for (int pos = start; pos < end; pos += AC_TILE) {
        const int rows_t = (pos + AC_TILE <= end) ? AC_TILE : (end - pos);

        __syncthreads();
        // Stage K and V for the tile once for every column. Rows past `rows_t`
        // re-read the last valid row; their scores are folded to -inf below.
        const int nload = 2 * AC_TILE * head_dim;
        for (int idx = threadIdx.x; idx < nload; idx += blockDim.x) {
            const int arr = idx / (AC_TILE * head_dim);
            const int rem = idx - arr * AC_TILE * head_dim;
            const int r0 = rem / head_dim;
            const int d = rem - r0 * head_dim;
            const int rr = (r0 < rows_t) ? r0 : (rows_t - 1);
            const float* base = (arr == 0) ? k : v;
            float* dst = (arr == 0) ? ks : vs;
            dst[r0 * head_dim + d] =
                base[(size_t)(pos + rr) * width + kvh * head_dim + d];
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < AC_MAX_COLS; ++r) {
            if (r >= rows) break;
            const int seq_r = pos0 + 1 + r;
            // Once every row of this column is behind us the column's partial
            // is final for this split; the tile is skipped for it.
            if (pos >= seq_r) continue;

            float s[AC_TILE];
            #pragma unroll
            for (int t = 0; t < AC_TILE; ++t) {
                float dot = 0.0f;
                #pragma unroll
                for (int c = 0; c < AC_MAX_DPL; ++c) {
                    if (c < dpl) {
                        dot = fmaf(qv[r][c], ks[t * head_dim + c * 32 + lane], dot);
                    }
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    dot += __shfl_xor_sync(0xffffffffffffffffULL, dot, off);
                }
                s[t] = (t < rows_t && pos + t < seq_r) ? dot * scale : AC_NEG_INF;
            }

            float mt = m[r];
            #pragma unroll
            for (int t = 0; t < AC_TILE; ++t) mt = fmaxf(mt, s[t]);
            const float corr = expf(m[r] - mt);
            lsum[r] *= corr;
            #pragma unroll
            for (int c = 0; c < AC_MAX_DPL; ++c) acc[r][c] *= corr;

            #pragma unroll
            for (int t = 0; t < AC_TILE; ++t) {
                const float p = expf(s[t] - mt);
                lsum[r] += p;
                #pragma unroll
                for (int c = 0; c < AC_MAX_DPL; ++c) {
                    if (c < dpl) {
                        acc[r][c] = fmaf(p, vs[t * head_dim + c * 32 + lane], acc[r][c]);
                    }
                }
            }
            m[r] = mt;
        }
    }

    #pragma unroll
    for (int r = 0; r < AC_MAX_COLS; ++r) {
        if (r >= rows) break;
        const size_t hb = ((size_t)r * n_heads + h) * splits + split;
        #pragma unroll
        for (int c = 0; c < AC_MAX_DPL; ++c) {
            if (c < dpl) part_acc[hb * head_dim + c * 32 + lane] = acc[r][c];
        }
        if (lane == 0) {
            float* ml = part_ml + hb * 2;
            ml[0] = m[r];
            ml[1] = lsum[r];
        }
    }
}

// Merge the split partials of every column. One block per (query head, column),
// one thread per component; the reduction order matches attn_reduce_f32.
extern "C" __global__ void attn_reduce_cols_f32(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_ml,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int splits)
{
    const int h = blockIdx.x;
    const int r = blockIdx.y;
    const int t = threadIdx.x;
    if (t >= head_dim) return;

    const size_t hb = ((size_t)r * n_heads + h) * splits;
    const float* ml = part_ml + hb * 2;
    float m = AC_NEG_INF;
    for (int s = 0; s < splits; ++s) m = fmaxf(m, ml[s * 2]);

    float acc = 0.0f;
    float l = 0.0f;
    for (int s = 0; s < splits; ++s) {
        const float w = expf(ml[s * 2] - m);
        l += ml[s * 2 + 1] * w;
        acc += part_acc[(hb + s) * head_dim + t] * w;
    }
    out[((size_t)r * n_heads + h) * head_dim + t] = (l > 0.0f) ? acc / l : 0.0f;
}
