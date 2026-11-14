// Empty-launch curve: what one more node in the graph costs.
//
// The decode token issues ~1,100 launches in one captured hipGraph. Two
// different things could make that expensive: the host submitting each launch,
// or the GPU paying a fixed price per node whether or not the node does work.
// This probe separates them by launching N kernels that do nothing at all, and
// N kernels that each read one 4 KiB page, both eager and captured into one
// graph, for N = 64 … 4096.
//
// Build: hipcc -O3 --offload-arch=$env:HIPML_ARCH probe_launch.cpp -o probe_launch.exe

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    hipError_t e_ = (x);                                                       \
    if (e_ != hipSuccess) {                                                    \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                 \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void noop_kernel() {}

// One 4 KiB page read per block, so the node has real (if tiny) work.
__global__ void touch_kernel(const float* __restrict__ p, float* __restrict__ out) {
  float v = p[threadIdx.x];
  if (v == 1234.5f) out[threadIdx.x] = v;
}

static double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

int main() {
  int dev = 0;
  CK(hipGetDevice(&dev));
  hipDeviceProp_t prop{};
  CK(hipGetDeviceProperties(&prop, dev));
  std::printf("device: %s, %d CUs\n", prop.name, prop.multiProcessorCount);

  float *p = nullptr, *out = nullptr;
  CK(hipMalloc(&p, 4096));
  CK(hipMalloc(&out, 4096));
  std::vector<float> host(1024, 0.0f);
  CK(hipMemcpy(p, host.data(), 4096, hipMemcpyHostToDevice));

  const std::vector<int> sizes = {64, 256, 512, 1024, 2048, 4096};
  const int reps = 200;

  // Warm the runtime and the card.
  for (int i = 0; i < 200; i++) noop_kernel<<<1, 32>>>();
  CK(hipDeviceSynchronize());

  std::printf("\n%-6s %-12s %-12s %-12s %-12s %-12s %-12s\n", "N", "noop eager",
              "noop graph", "touch eager", "touch graph", "graph launch", "eager submit");
  std::printf("%-6s %-12s %-12s %-12s %-12s %-12s %-12s\n", "", "us/node",
              "us/node", "us/node", "us/node", "us", "us/node");

  for (int n : sizes) {
    // --- noop, eager: host submits every node
    auto t = std::chrono::steady_clock::now();
    double submit = 0.0;
    for (int r = 0; r < reps; r++) {
      auto h = std::chrono::steady_clock::now();
      for (int i = 0; i < n; i++) noop_kernel<<<1, 32>>>();
      submit += ms_since(h) * 1000.0;
    }
    double eager_noop_total = ms_since(t) * 1000.0;
    CK(hipDeviceSynchronize());
    double eager_noop = eager_noop_total / (double)(reps * n);

    // --- noop, captured into one graph, replayed reps times
    hipGraph_t g1{};
    hipStream_t s{};
    CK(hipStreamCreate(&s));
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; i++) noop_kernel<<<1, 32, 0, s>>>();
    CK(hipStreamEndCapture(s, &g1));
    hipGraphExec_t e1{};
    CK(hipGraphInstantiate(&e1, g1, nullptr, nullptr, 0));
    for (int r = 0; r < 20; r++) CK(hipGraphLaunch(e1, s));
    CK(hipDeviceSynchronize());

    t = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) CK(hipGraphLaunch(e1, s));
    double graph_noop_total = ms_since(t) * 1000.0;
    CK(hipDeviceSynchronize());
    double graph_noop = graph_noop_total / (double)(reps * n);

    // --- touch, eager
    t = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++)
      for (int i = 0; i < n; i++) touch_kernel<<<1, 256>>>(p, out);
    double eager_touch_total = ms_since(t) * 1000.0;
    CK(hipDeviceSynchronize());
    double eager_touch = eager_touch_total / (double)(reps * n);

    // --- touch, captured
    hipGraph_t g2{};
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    for (int i = 0; i < n; i++) touch_kernel<<<1, 256, 0, s>>>(p, out);
    CK(hipStreamEndCapture(s, &g2));
    hipGraphExec_t e2{};
    CK(hipGraphInstantiate(&e2, g2, nullptr, nullptr, 0));
    for (int r = 0; r < 20; r++) CK(hipGraphLaunch(e2, s));
    CK(hipDeviceSynchronize());
    t = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) CK(hipGraphLaunch(e2, s));
    double graph_touch_total = ms_since(t) * 1000.0;
    CK(hipDeviceSynchronize());
    double graph_touch = graph_touch_total / (double)(reps * n);

    // --- the cost of the one hipGraphLaunch itself: same graph, N = 1
    hipGraph_t g3{};
    CK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
    noop_kernel<<<1, 32, 0, s>>>();
    CK(hipStreamEndCapture(s, &g3));
    hipGraphExec_t e3{};
    CK(hipGraphInstantiate(&e3, g3, nullptr, nullptr, 0));
    // The first launch of a fresh exec uploads it (~170 ms here), so it must not
    // be inside the timed loop.
    for (int r = 0; r < 20; r++) CK(hipGraphLaunch(e3, s));
    CK(hipDeviceSynchronize());
    t = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) CK(hipGraphLaunch(e3, s));
    double one_launch = ms_since(t) * 1000.0 / (double)reps * 1000.0;
    CK(hipDeviceSynchronize());

    std::printf("%-6d %-12.3f %-12.3f %-12.3f %-12.3f %-12.3f %-12.3f\n", n, eager_noop,
                graph_noop, eager_touch, graph_touch, one_launch, submit / (double)(reps * n));

    CK(hipGraphExecDestroy(e1));
    CK(hipGraphDestroy(g1));
    CK(hipGraphExecDestroy(e2));
    CK(hipGraphDestroy(g2));
    CK(hipGraphExecDestroy(e3));
    CK(hipGraphDestroy(g3));
    CK(hipStreamDestroy(s));
  }

  CK(hipFree(p));
  CK(hipFree(out));
  return 0;
}
