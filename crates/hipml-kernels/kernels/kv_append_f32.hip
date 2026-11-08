// Append one token row to a position-major KV cache pair.
//
// k_cache and v_cache are [n_ctx][width]; the row at `step[0]` is overwritten
// (the position lives in device memory so a captured graph is reusable).
// One linear grid: threads [0,width) write K, [width,2*width) write V.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_append_f32(
    float* __restrict__ k_cache,
    float* __restrict__ v_cache,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int* __restrict__ step,
    int width)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * width) return;
    const size_t base = (size_t)step[0] * (size_t)width;
    if (i < width) {
        k_cache[base + i] = k[i];
    } else {
        v_cache[base + (i - width)] = v[i - width];
    }
}
