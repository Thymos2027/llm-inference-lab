#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

constexpr int kThreads = 256;
constexpr size_t kElementCount = (1u << 24) + 123;

__global__ void vector_add(const float* a, const float* b, float* c, size_t n) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n) c[index] = a[index] + b[index];
}

void launch(const float* a, const float* b, float* c, size_t n) {
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  vector_add<<<blocks, kThreads>>>(a, b, c, n);
  CUDA_CHECK(cudaGetLastError());
}

void allocate(float** a, float** b, float** c, size_t bytes) {
  CUDA_CHECK(cudaMalloc(a, bytes));
  CUDA_CHECK(cudaMalloc(b, bytes));
  CUDA_CHECK(cudaMalloc(c, bytes));
}

void release(float* a, float* b, float* c) {
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
}

int profile_once() {
  const size_t bytes = kElementCount * sizeof(float);
  float *a = nullptr, *b = nullptr, *c = nullptr;
  allocate(&a, &b, &c, bytes);
  CUDA_CHECK(cudaMemset(a, 0, bytes));
  CUDA_CHECK(cudaMemset(b, 0, bytes));

  launch(a, b, c, kElementCount);
  CUDA_CHECK(cudaDeviceSynchronize());
  release(a, b, c);
  return EXIT_SUCCESS;
}

int validate_and_benchmark() {
  const size_t bytes = kElementCount * sizeof(float);
  std::vector<float> host_a(kElementCount);
  std::vector<float> host_b(kElementCount);
  std::vector<float> host_c(kElementCount);
  for (size_t i = 0; i < kElementCount; ++i) {
    host_a[i] = static_cast<float>(static_cast<int>(i % 251) - 125) * 0.25f;
    host_b[i] = static_cast<float>(static_cast<int>(i % 127) - 63) * 0.5f;
  }

  float *a = nullptr, *b = nullptr, *c = nullptr;
  allocate(&a, &b, &c, bytes);
  CUDA_CHECK(cudaMemcpy(a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  launch(a, b, c, kElementCount);
  CUDA_CHECK(cudaMemcpy(host_c.data(), c, bytes, cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  for (size_t i = 0; i < kElementCount; ++i) {
    const float expected = host_a[i] + host_b[i];
    max_error = std::fmax(max_error, std::fabs(host_c[i] - expected));
  }
  const bool passed = max_error <= 1e-6f;

  constexpr int warmups = 5;
  constexpr int repetitions = 100;
  for (int i = 0; i < warmups; ++i) launch(a, b, c, kElementCount);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repetitions; ++i) launch(a, b, c, kElementCount);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  const double average_ms = total_ms / repetitions;
  const double transferred_bytes = 3.0 * static_cast<double>(bytes);
  const double effective_gb_per_second = transferred_bytes / (average_ms * 1.0e6);
  std::printf("Vector Add: n=%zu, blocks=%zu, threads/block=%d\n",
              kElementCount, (kElementCount + kThreads - 1) / kThreads, kThreads);
  std::printf("Correctness: max error %.8f  %s\n", max_error,
              passed ? "PASS" : "FAIL");
  std::printf("Kernel time: %.3f ms\n", average_ms);
  std::printf("Effective bandwidth: %.1f GB/s (read A + read B + write C)\n",
              effective_gb_per_second);

  release(a, b, c);
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char** argv) {
  if (argc == 2 && std::strcmp(argv[1], "--profile") == 0) {
    return profile_once();
  }
  if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--profile]\n", argv[0]);
    return EXIT_FAILURE;
  }
  return validate_and_benchmark();
}
