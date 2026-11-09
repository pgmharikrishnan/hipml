// Causal depthwise conv1d with the last d_conv-1 columns carried in `state`.
//
// Port of ggml_compute_forward_ssm_conv_f32: one channel per thread, the kernel
// taps are contiguous per channel (`w[c*d_conv + k]`), and the history is
// `[(d_conv-1)][channels]`, channel innermost. Thread `c` only ever reads and
// writes column `c`, so sliding the window inside the kernel needs no barrier.
//
// Kernel: ssm_conv1d_f32. Reference: delta::ssm_conv1d_f32.

#include <hip/hip_runtime.h>

extern "C" __global__ void ssm_conv1d_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ state,
    float* __restrict__ y,
    int channels,
    int d_conv)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const int hist = d_conv - 1;
    const float* wc = w + (size_t)c * d_conv;

    float acc = 0.0f;
    for (int k = 0; k < hist; ++k) {
        acc = __fmaf_rn(state[(size_t)k * channels + c], wc[k], acc);
    }
    acc = __fmaf_rn(x[c], wc[hist], acc);
    y[c] = acc;

    // Slide left, then append the current token. Read-before-write is per
    // thread, so the copy cannot race another channel.
    for (int k = 0; k + 1 < hist; ++k) {
        state[(size_t)k * channels + c] = state[(size_t)(k + 1) * channels + c];
    }
    if (hist > 0) {
        state[(size_t)(hist - 1) * channels + c] = x[c];
    }
}
