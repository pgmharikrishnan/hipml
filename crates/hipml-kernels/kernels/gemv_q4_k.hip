// y[row] = dequant_q4_k(W[row,:]) . x
//
// Row-tile GEMV: one block owns Q4K_ROW_TILE output rows and stages the x k-tile in
// shared memory once. Each warp owns Q4K_ROWS_PER_WARP rows and keeps one
// accumulator per row, so a value read from shared is applied to
// Q4K_ROWS_PER_WARP weight rows before it is discarded. With 4-bit weights x is 8x
// the weight bytes, so this reuse is what removes the shared/L2 ceiling.
//
// Each lane owns Q4K_VPL (16) consecutive columns, i.e. half of one
// 32-value Q4_K sub-block, so one d/dmin/scale/min set covers all 16 values and
// their 16 quant bytes are four aligned 32-bit loads.
//
// The shared tile is de-interleaved by 512-value group: component `c` of a group
// lives at sx[g*512 + c*32 + lane]. A lane reads its 16 components as 16 scalar
// shared loads, one bank per lane, so there are no bank conflicts.
//
// Q4_K super-block (144 bytes, 256 values):
//   d(2) dmin(2) scales(12) qs(128)
// Math is unchanged: y = d*scale[j]*q - dmin*min[j] for sub-block j.
//
// Two barriers per k-tile; the only reduction is a warp shuffle tree per row.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define Q4K_VALUES 256
#define Q4K_BLOCK_BYTES 144
#define Q4K_BLOCK 256
#define Q4K_WARPS (Q4K_BLOCK / 32)
#define Q4K_ROWS_PER_WARP 2
#define Q4K_ROW_TILE (Q4K_WARPS * Q4K_ROWS_PER_WARP)
#define Q4K_VPL 16
#define Q4K_GROUP (Q4K_VPL * 32)
#define Q4K_K_TILE 2048

// Mirrors the manifest's KernelKind::RowTile { rows = Q4K_ROW_TILE, block = Q4K_BLOCK };
// change both together.

static __device__ __forceinline__ void q4k_scale_min(int j, const unsigned char* q,
                                                     int* sc, int* mn)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ void stage_group(const float* __restrict__ x,
                                                   float* __restrict__ sx,
                                                   int base, int j)
{
    const int g = j >> 5; // 32 chunks of 16 values per 512-value group
    const int lane = j & 31;
    float* s = sx + g * Q4K_GROUP + lane;
    const float4* src = reinterpret_cast<const float4*>(x + base + Q4K_VPL * j);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        const float4 v = src[q];
        s[(4 * q + 0) * 32] = v.x;
        s[(4 * q + 1) * 32] = v.y;
        s[(4 * q + 2) * 32] = v.z;
        s[(4 * q + 3) * 32] = v.w;
    }
}

// Sixteen-column contribution of one Q4_K row to `acc`.
//
// WIDE loads the 16 quant bytes as one 128-bit `buffer_load_dwordx4` instead of
// four dwords. The payload is 16-byte aligned inside the block (offset 16 plus a
// multiple of 16), so this is the same bytes and the same arithmetic; it is an
// A/B switch because the kernel may be load-issue-bound rather than DRAM-bound.
template <bool WIDE, bool STREAM = false>
static __device__ __forceinline__ void q4k_acc(const unsigned char* __restrict__ rp,
                                               int p, const float* xv, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int sub = within >> 5;
    const int part = (within & 31) >> 4;

    const unsigned char* b = rp + (size_t)blk * Q4K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    int sc, mn;
    q4k_scale_min(sub, b + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;

    const unsigned int* qp = reinterpret_cast<const unsigned int*>(
        b + 16 + (sub >> 1) * 32 + part * 16);
    union Q4Bytes {
        unsigned int w[4];
        unsigned char b[16];
    } q;
    if (WIDE && STREAM) {
        // sc1 (streaming): the 17 GB stream is read once per token and never
        // reused, so it should not be kept in L2. Same bytes, same arithmetic.
        // `__builtin_nontemporal_load` does not take HIP vector types, so this
        // is four dword loads carrying the streaming cache policy.
#pragma unroll
        for (int i = 0; i < 4; ++i) q.w[i] = __builtin_nontemporal_load(qp + i);
    } else if (WIDE) {
        const uint4 wide = *reinterpret_cast<const uint4*>(qp);
        q.w[0] = wide.x;
        q.w[1] = wide.y;
        q.w[2] = wide.z;
        q.w[3] = wide.w;
    } else {
        q.w[0] = qp[0];
        q.w[1] = qp[1];
        q.w[2] = qp[2];
        q.w[3] = qp[3];
    }
    const bool high = (sub & 1) != 0;

    float a = *acc;
#pragma unroll
    for (int i = 0; i < Q4K_VPL; ++i) {
        const unsigned int byte = q.b[i];
        const int v = high ? (int)(byte >> 4) : (int)(byte & 0xF);
        a = fmaf(scale * (float)v - minv, xv[i], a);
    }
    *acc = a;
}

// One row/warp when RPW is 1, two when it is Q4K_ROWS_PER_WARP. The lane owns
// the same columns either way, so the per-row accumulation order and the warp
// reduction tree are unchanged: the two entries are bit-identical, and the
// 1-row grid is twice as long, which is what a 5120-row tensor needs.
// `tile` is blockIdx.x in the standalone entry points and blockIdx.x of the
// job's own grid in gemv_jobs. External linkage so the job kernel can call it.
template <int RPW, bool WIDE, bool STREAM = false>
__device__ __forceinline__ void q4k_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / Q4K_VALUES;
    const int row_tile = Q4K_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q4K_BLOCK_BYTES;
        acc[rr] = 0.0f;
    }

    for (int base = 0; base < k; base += Q4K_K_TILE) {
        const int n = min(Q4K_K_TILE, k - base);
        const int nchunks = n / Q4K_VPL;
        for (int j = tid; j < nchunks; j += Q4K_BLOCK) stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * Q4K_VPL; c < n; c += Q4K_GROUP) {
            const int g = c / Q4K_GROUP;
            const int s = g * Q4K_GROUP + lane;
            float xv[Q4K_VPL];
#pragma unroll
            for (int comp = 0; comp < Q4K_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr) q4k_acc<WIDE, STREAM>(rp[rr], p, xv, &acc[rr]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        float a = acc[rr];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
        if (lane == 0 && row[rr] < m) y[row[rr]] = a;
    }
}

extern "C" __global__ void gemv_q4_k(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q4K_K_TILE];
    q4k_tile<Q4K_ROWS_PER_WARP, false>(w, x, y, m, k, sx, blockIdx.x);
}

extern "C" __global__ void gemv_q4_k_r1(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q4K_K_TILE];
    q4k_tile<1, false>(w, x, y, m, k, sx, blockIdx.x);
}

// One row per warp, one 128-bit load per 16 values: the A/B entry for the
// load-issue hypothesis. Bit-identical to `gemv_q4_k_r1`.
extern "C" __global__ void gemv_q4_k_u4(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q4K_K_TILE];
    q4k_tile<1, true>(w, x, y, m, k, sx, blockIdx.x);
}

// Streaming-load A/B: `gemv_q4_k_r1` with `buffer_load_dwordx4` + sc1 on the
// weight payload (the brief's cache-hint item).
extern "C" __global__ void gemv_q4_k_nt(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    __shared__ __align__(16) float sx[Q4K_K_TILE];
    q4k_tile<1, true, true>(w, x, y, m, k, sx, blockIdx.x);
}

// Split-K partial pass: grid (row tiles, splits), each block accumulates a
// contiguous 512-value-aligned slice of k and writes `part[split][row]`.
//
// The probe for the one explanation the measurements keep pointing at: a
// 5120-row tensor is 640 blocks of 8 warps, and every block streams its rows
// through a single accumulator chain, so the kernel's in-flight bytes scale with
// the number of threads, not with the grid. Splitting k gives the same bytes to
// more threads doing less work each — the shape `gpu-sweep`'s calibration buffer
// likes (267 GB/s at 4096 blocks, 604 at 16384).
//
// This is *not* bit-identical to `gemv_q4_k`: the per-lane accumulation over k
// is partitioned differently, so the last bits of every output can move. It is
// measured as an A/B, and adopted only if the greedy tokens survive.
extern "C" __global__ void gemv_q4_k_split(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ part,
    int m,
    int k,
    int splits)
{
    __shared__ __align__(16) float sx[Q4K_K_TILE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / Q4K_VALUES;
    const int row = blockIdx.x * Q4K_WARPS + warp;
    const int r = min(row, m - 1);
    const unsigned char* rp = w + (size_t)r * (size_t)nblocks * Q4K_BLOCK_BYTES;

    // 512-value units so every slice starts on a group boundary (the shared
    // staging and the lane mapping are per 512-value group).
    const int per = ((k / splits) + Q4K_GROUP - 1) / Q4K_GROUP * Q4K_GROUP;
    const int begin = blockIdx.y * per;
    const int end = min(begin + per, k);

    float acc = 0.0f;
    for (int base = begin; base < end; base += Q4K_K_TILE) {
        const int n = min(Q4K_K_TILE, end - base);
        const int nchunks = n / Q4K_VPL;
        for (int j = tid; j < nchunks; j += Q4K_BLOCK) stage_group(x, sx, base, j);
        __syncthreads();

        for (int c = lane * Q4K_VPL; c < n; c += Q4K_GROUP) {
            const int g = c / Q4K_GROUP;
            const int s = g * Q4K_GROUP + lane;
            float xv[Q4K_VPL];
#pragma unroll
            for (int comp = 0; comp < Q4K_VPL; ++comp) xv[comp] = sx[s + comp * 32];
            q4k_acc<true>(rp, base + c, xv, &acc);
        }
        __syncthreads();
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    if (lane == 0 && row < m) part[(size_t)blockIdx.y * m + row] = acc;
}

// `y[row] = sum_s part[s][row]`, one thread per row, splits in order.
extern "C" __global__ void gemv_split_reduce(
    const float* __restrict__ part,
    float* __restrict__ y,
    int m,
    int splits)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;
    float a = 0.0f;
    for (int s = 0; s < splits; ++s) a += part[(size_t)s * m + row];
    y[row] = a;
}

// ---------------------------------------------------------------------------
// Batched GEMV: B token rows share one pass over the weight.
//
//   y[b][row] = dequant_q4k(W[row,:]) . x[b][:]
//
// The weight block is decoded once per row and applied to all B x rows, so the
// decode step's 17 GB weight stream serves B tokens. Each row keeps its own
// accumulator and walks k in exactly the order `gemv_q4_k` does, so row b here
// is bit-identical to a single-sequence launch on x[b].
//
// K_TILE drops to 512 (one 512-value group) so the shared x tile is B*512 f32:
// 16 KB at B=8. The c order is unchanged, only the tile boundary moves.
// ---------------------------------------------------------------------------

#define Q4KB_K_TILE 512

// Same decode as q4k_acc, applied to B x rows. `sxb` points at this lane's
// de-interleaved x slot for b=0; row b is `sxb + b*stride`.
template <int B>
static __device__ __forceinline__ void q4k_acc_b(const unsigned char* __restrict__ rp,
                                                 int p, const float* __restrict__ sxb,
                                                 int stride, float* acc)
{
    const int blk = p >> 8;
    const int within = p & 255;
    const int sub = within >> 5;
    const int part = (within & 31) >> 4;

    const unsigned char* b = rp + (size_t)blk * Q4K_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(b + 2));
    int sc, mn;
    q4k_scale_min(sub, b + 4, &sc, &mn);
    const float scale = d * (float)sc;
    const float minv = dmin * (float)mn;

    const unsigned int* qp = reinterpret_cast<const unsigned int*>(
        b + 16 + (sub >> 1) * 32 + part * 16);
    union Q4BytesB {
        unsigned int w[4];
        unsigned char b[16];
    } q;
    q.w[0] = qp[0];
    q.w[1] = qp[1];
    q.w[2] = qp[2];
    q.w[3] = qp[3];
    const bool high = (sub & 1) != 0;

    // Dequantise the 16 values once; the batch columns then only pay FMAs.
    float wv[Q4K_VPL];
#pragma unroll
    for (int i = 0; i < Q4K_VPL; ++i) {
        const unsigned int byte = q.b[i];
        const int v = high ? (int)(byte >> 4) : (int)(byte & 0xF);
        wv[i] = scale * (float)v - minv;
    }
#pragma unroll
    for (int bi = 0; bi < B; ++bi) {
        const float* xv = sxb + bi * stride;
        float a = acc[bi];
#pragma unroll
        for (int i = 0; i < Q4K_VPL; ++i) a = fmaf(wv[i], xv[i * 32], a);
        acc[bi] = a;
    }
}

template <int B, int RPW>
static __device__ __forceinline__ void q4k_tile_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nblocks = k / Q4K_VALUES;
    const int row_tile = Q4K_WARPS * RPW;

    int row[RPW];
    int r[RPW];
    const unsigned char* rp[RPW];
    float acc[RPW][B];
#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
        row[rr] = tile * row_tile + warp * RPW + rr;
        r[rr] = min(row[rr], m - 1);
        rp[rr] = w + (size_t)r[rr] * (size_t)nblocks * Q4K_BLOCK_BYTES;
#pragma unroll
        for (int bi = 0; bi < B; ++bi) acc[rr][bi] = 0.0f;
    }

    for (int base = 0; base < k; base += Q4KB_K_TILE) {
        const int n = min(Q4KB_K_TILE, k - base);
        const int nchunks = n / Q4K_VPL;
        for (int j = tid; j < nchunks; j += Q4K_BLOCK) {
#pragma unroll
            for (int bi = 0; bi < B; ++bi)
                stage_group(x + (size_t)bi * k, sx + bi * Q4KB_K_TILE, base, j);
        }
        __syncthreads();

        for (int c = lane * Q4K_VPL; c < n; c += Q4K_GROUP) {
            const int g = c / Q4K_GROUP;
            const int s = g * Q4K_GROUP + lane;
            const int p = base + c;
#pragma unroll
            for (int rr = 0; rr < RPW; ++rr)
                q4k_acc_b<B>(rp[rr], p, sx + s, Q4KB_K_TILE, acc[rr]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int rr = 0; rr < RPW; ++rr) {
#pragma unroll
        for (int bi = 0; bi < B; ++bi) {
            float a = acc[rr][bi];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) a += __shfl_down(a, off);
            if (lane == 0 && row[rr] < m) y[(size_t)bi * m + row[rr]] = a;
        }
    }
}

template <int B>
static __device__ __forceinline__ void q4k_entry_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k)
{
    extern __shared__ __align__(16) float hipml_sx_b[];
    q4k_tile_b<B, Q4K_ROWS_PER_WARP>(w, x, y, m, k, hipml_sx_b, blockIdx.x);
}

extern "C" __global__ void gemv_q4_k_b(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int batch)
{
    switch (batch) {
    case 1: q4k_entry_b<1>(w, x, y, m, k); break;
    case 2: q4k_entry_b<2>(w, x, y, m, k); break;
    case 3: q4k_entry_b<3>(w, x, y, m, k); break;
    case 4: q4k_entry_b<4>(w, x, y, m, k); break;
    case 5: q4k_entry_b<5>(w, x, y, m, k); break;
    case 6: q4k_entry_b<6>(w, x, y, m, k); break;
    case 7: q4k_entry_b<7>(w, x, y, m, k); break;
    default: q4k_entry_b<8>(w, x, y, m, k); break;
    }
}
