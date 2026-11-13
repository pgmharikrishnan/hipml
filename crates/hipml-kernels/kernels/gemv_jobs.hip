// SPDX-License-Identifier: Apache-2.0
// One launch over a device-side job list of independent GEMVs.
//
// The decode step issues 336 GEMVs a token and each one pays its own memory
// pipeline fill/drain: `gpu-sweep` reads the same 16.5 GB at 477 GB/s as 399
// single-tensor launches and at 636 GB/s as one launch that walks a job list.
// The projections inside one layer that share an input have no ordering between
// them (qkv/gate/alpha/beta, q/k/v, ffn gate/up), so they can share one kernel
// and one fill: grid.y selects the job, grid.x the row tile inside it.
//
// The bodies are the standalone entries' tile functions, so a row is dequantised
// and reduced by exactly the same code and writes the same bits. `gemv_q4_k`,
// `gemv_q6_k`, `gemv_q5_k` and `gemv_q8_0` call those functions with
// blockIdx.x; here blockIdx.x is the tile inside the job's own row space.
//
// Job ABI: `HipmlGemvJob` is mirrored by `GemvJob` in `graph_launch.rs`.

#include <hip/hip_runtime.h>

#define GJ_Q4K 0
#define GJ_Q6K 1
#define GJ_Q5K 2
#define GJ_Q80 3
#define GJ_IQ4XS 4

// One row per warp, 8 rows per block for the three K-quant formats: the `_r1`
// geometry WP16 measured as the faster one at these row counts. Q8_0 keeps its
// one-row-per-block shape, so its tile count is its row count.
#define GJ_ROWS_PER_TILE 8

// IQ4_XS brings its own tile geometry from gemv_iq4_xs.hip (16 rows a block,
// two a warp) and a 2,048-float x stage, which is exactly `sx` below. The host's
// `gemv_job_tiles` must use the same row count: a mismatch either re-streams a
// tail row from idle blocks or leaves rows unrun.
#define GJ_IQ4X_RPW 2
#define GJ_IQ4X_ROWS_PER_TILE (8 * GJ_IQ4X_RPW)
#ifdef IQ4X_ROW_TILE
static_assert(GJ_IQ4X_ROWS_PER_TILE == IQ4X_ROW_TILE,
              "the IQ4_XS job tile must be the entry's own row tile");
#endif

struct HipmlGemvJob {
    const unsigned char* w;
    const float* x;
    float* y;
    int m;
    int k;
    int kind;
    int pad;
};

// Defined in the format files later in this translation unit.
template <int RPW, bool WIDE, bool STREAM>
__device__ __forceinline__ void q4k_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile);
template <int RPW>
__device__ __forceinline__ void q6k_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile);
__device__ __forceinline__ void q5k_rows(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    int tile);
__device__ __forceinline__ void q80_row(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    int row);
template <int RPW>
static __device__ __forceinline__ void iq4x_tile(
    const unsigned char* __restrict__ w,
    const float* __restrict__ x,
    float* __restrict__ y,
    int m,
    int k,
    float* __restrict__ sx,
    float* __restrict__ scb,
    int tile);

extern "C" __global__ void gemv_jobs(
    const HipmlGemvJob* __restrict__ jobs,
    int n_jobs)
{
    const int j = blockIdx.y;
    if (j >= n_jobs) return;
    const HipmlGemvJob job = jobs[j];
    const int tile = blockIdx.x;

    // Every job shares the grid, so a block past its own job's row space must
    // retire before it touches memory: the clamped tail row would otherwise be
    // re-streamed by thousands of idle blocks of a wider neighbour.
    const int rows_per_tile = (job.kind == GJ_IQ4XS) ? GJ_IQ4X_ROWS_PER_TILE
                                                      : GJ_ROWS_PER_TILE;
    const int need = (job.kind == GJ_Q80)
        ? job.m
        : (job.m + rows_per_tile - 1) / rows_per_tile;
    if (tile >= need) return;

    __shared__ __align__(16) float sx[2048];
    __shared__ __align__(16) float scb[16];
    switch (job.kind) {
    case GJ_Q4K:
        q4k_tile<1, false, false>(job.w, job.x, job.y, job.m, job.k, sx, tile);
        break;
    case GJ_Q6K:
        q6k_tile<1>(job.w, job.x, job.y, job.m, job.k, sx, tile);
        break;
    case GJ_Q5K:
        q5k_rows(job.w, job.x, job.y, job.m, job.k, sx, tile);
        break;
    case GJ_Q80:
        for (int row = tile; row < job.m; row += gridDim.x) {
            q80_row(job.w, job.x, job.y, job.m, job.k, row);
        }
        break;
    case GJ_IQ4XS:
        iq4x_tile<GJ_IQ4X_RPW>(job.w, job.x, job.y, job.m, job.k, sx, scb, tile);
        break;
    }
}
