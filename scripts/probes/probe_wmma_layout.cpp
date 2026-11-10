// Fragment layout of the gfx12 matrix-unit instruction used by
// `prefill_wmma.hip`, measured rather than assumed.
//
//   hipcc -O2 --offload-arch=gfx1201 probe_wmma_layout.cpp -o probe_layout
//   probe_layout 0|1|2
//
// One warp, one 16x16x16 f16 WMMA. The operands are loaded with the mapping the
// kernel uses (index = lane % 16, k = 8*(lane/16) + e) and the 32x8 accumulator
// registers are dumped, so each dump slot shows which output element it holds:
//
//   mode 0  A[m][k] = m*16+k+1, B = I   -> C = A          (all 256 entries)
//   mode 1  A[m][k] = m,       B = 1    -> C[m][n] = 16*m (rows)
//   mode 2  A[m][k] = 1,       B[k][n]=n-> C[m][n] = 16*n (columns)
//
// Result on gfx1201: operands are index = lane % 16, k = 8*(lane / 16) + e, and
// accumulator slot (lane, e) is column lane % 16 of row 8*(lane / 16) + e.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>

typedef _Float16 __attribute__((ext_vector_type(8))) half8;
typedef float __attribute__((ext_vector_type(8))) float8;

__global__ void probe(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ C) {
    const int l = threadIdx.x;
    const int r = l % 16;
    const int k0 = 8 * (l / 16);
    half8 a, b;
    float8 c;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        a[e] = (_Float16)A[r * 16 + k0 + e];
        b[e] = (_Float16)B[(k0 + e) * 16 + r];
        c[e] = 0.0f;
    }
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#pragma unroll
    for (int e = 0; e < 8; ++e) C[l * 8 + e] = c[e];
}

int main(int argc, char** argv) {
    const int mode = argc > 1 ? atoi(argv[1]) : 0;
    float A[256], B[256], C[256];
    for (int m = 0; m < 16; ++m)
        for (int j = 0; j < 16; ++j) {
            A[m * 16 + j] = mode == 0 ? (float)(m * 16 + j + 1) : (mode == 1 ? (float)m : 1.0f);
            B[j * 16 + m] = mode == 0 ? (j == m ? 1.0f : 0.0f) : (mode == 1 ? 1.0f : (float)m);
        }
    float *dA, *dB, *dC;
    hipMalloc(&dA, sizeof(A));
    hipMalloc(&dB, sizeof(B));
    hipMalloc(&dC, sizeof(C));
    hipMemcpy(dA, A, sizeof(A), hipMemcpyHostToDevice);
    hipMemcpy(dB, B, sizeof(B), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(probe, dim3(1), dim3(32), 0, 0, dA, dB, dC);
    if (hipDeviceSynchronize() != hipSuccess) {
        printf("launch failed\n");
        return 1;
    }
    hipMemcpy(C, dC, sizeof(C), hipMemcpyDeviceToHost);
    for (int l = 0; l < 32; ++l) {
        printf("lane %2d:", l);
        for (int e = 0; e < 8; ++e) {
            const float v = C[l * 8 + e];
            const int iv = (int)(v + 0.5f);
            if (mode == 0) {
                if (iv < 1 || iv > 256) printf("  ---  ");
                else printf(" %2d,%-2d", (iv - 1) / 16, (iv - 1) % 16);
            } else {
                printf(" %5.0f", v / 16.0f);
            }
        }
        printf("\n");
    }
    return 0;
}
