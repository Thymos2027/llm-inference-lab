#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do {                                            \
  cudaError_t error = (call);                                            \
  if (error != cudaSuccess) {                                            \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,           \
                 cudaGetErrorString(error));                            \
    std::exit(EXIT_FAILURE);                                             \
  }                                                                     \
} while (0)

constexpr int TILE = 32;
constexpr int ROWS = 8;

// Input is row-major A[height,width]; transpose output is B[width,height].
__global__ void matrix_copy(const float* a, float* b, int width, int height) {
  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (x < width && y + j < height)
      b[(y + j) * width + x] = a[(y + j) * width + x];
}

__global__ void transpose_naive(const float* a, float* b, int width, int height) {
  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (x < width && y + j < height)
      b[x * height + y + j] = a[(y + j) * width + x];
}

__global__ void transpose_shared(const float* a, float* b, int width, int height) {
  // Extra column avoids a 32-way shared-memory bank conflict on transposed reads.
  __shared__ float tile[TILE][TILE + 1];
  const int x = blockIdx.x * TILE + threadIdx.x;
  const int y = blockIdx.y * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (x < width && y + j < height)
      tile[threadIdx.y + j][threadIdx.x] = a[(y + j) * width + x];
  __syncthreads();

  const int out_x = blockIdx.y * TILE + threadIdx.x;
  const int out_y = blockIdx.x * TILE + threadIdx.y;
  for (int j = 0; j < TILE; j += ROWS)
    if (out_x < height && out_y + j < width)
      b[(out_y + j) * height + out_x] = tile[threadIdx.x][threadIdx.y + j];
}

enum class Variant { Copy, Naive, Shared };
const char* name(Variant v) {
  return v == Variant::Copy ? "copy" : v == Variant::Naive ? "naive" : "shared";
}

void launch(Variant v, const float* a, float* b, int width, int height) {
  const dim3 block(TILE, ROWS);
  const dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
  if (v == Variant::Copy) matrix_copy<<<grid, block>>>(a, b, width, height);
  else if (v == Variant::Naive) transpose_naive<<<grid, block>>>(a, b, width, height);
  else transpose_shared<<<grid, block>>>(a, b, width, height);
  CUDA_CHECK(cudaGetLastError());
}

float benchmark(Variant v, const float* a, float* b, int width, int height) {
  constexpr int warmups = 5, repetitions = 100;
  for (int i = 0; i < warmups; ++i) launch(v, a, b, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repetitions; ++i) launch(v, a, b, width, height);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / repetitions;
}

bool run_case(int width, int height, bool measure) {
  const size_t count = static_cast<size_t>(width) * height;
  const size_t bytes = count * sizeof(float);
  std::vector<float> input(count), output(count);
  for (size_t i = 0; i < count; ++i) input[i] = static_cast<float>(i % 1009) * 0.125f;
  float *a = nullptr, *b = nullptr;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMemcpy(a, input.data(), bytes, cudaMemcpyHostToDevice));
  bool passed = true;
  std::printf("shape=%dx%d (height x width)\n", height, width);
  for (Variant v : {Variant::Copy, Variant::Naive, Variant::Shared}) {
    CUDA_CHECK(cudaMemset(b, 0xff, bytes));
    launch(v, a, b, width, height);
    CUDA_CHECK(cudaMemcpy(output.data(), b, bytes, cudaMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (int row = 0; row < height; ++row)
      for (int col = 0; col < width; ++col) {
        const float expected = input[static_cast<size_t>(row) * width + col];
        const size_t index = v == Variant::Copy
            ? static_cast<size_t>(row) * width + col
            : static_cast<size_t>(col) * height + row;
        if (output[index] != expected) ++mismatches;
      }
    std::printf("  %-6s mismatches=%zu  %s", name(v), mismatches,
                mismatches ? "FAIL" : "PASS");
    passed &= mismatches == 0;
    std::puts("");
  }
  if (measure && passed) {
    const Variant variants[] = {Variant::Copy, Variant::Naive, Variant::Shared};
    float samples[3][3];
    for (int round = 0; round < 3; ++round)
      for (int variant = 0; variant < 3; ++variant)
        samples[variant][round] = benchmark(variants[variant], a, b, width, height);
    for (int variant = 0; variant < 3; ++variant) {
      std::sort(samples[variant], samples[variant] + 3);
      const float ms = samples[variant][1];
      const double gbps = 2.0 * bytes / (ms * 1.0e6);
      std::printf("  %-6s median %.4f ms  %.1f GB/s (3 x 100 launches; semantic read+write)\n",
                  name(variants[variant]), ms, gbps);
    }
  }
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  return passed;
}

int main(int argc, char** argv) {
  if (argc == 3 && std::strcmp(argv[1], "--profile") == 0) {
    Variant v;
    if (std::strcmp(argv[2], "copy") == 0) v = Variant::Copy;
    else if (std::strcmp(argv[2], "naive") == 0) v = Variant::Naive;
    else if (std::strcmp(argv[2], "shared") == 0) v = Variant::Shared;
    else return EXIT_FAILURE;
    constexpr int width = 4096, height = 4096;
    const size_t bytes = static_cast<size_t>(width) * height * sizeof(float);
    float *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 0, bytes));
    launch(v, a, b, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    return EXIT_SUCCESS;
  }
  if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--profile copy|naive|shared]\n", argv[0]);
    return EXIT_FAILURE;
  }
  return run_case(53, 37, false) && run_case(257, 129, false) &&
         run_case(4096, 4096, true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
