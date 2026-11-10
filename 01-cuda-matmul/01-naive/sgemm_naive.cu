#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error = (call);                                                  \
    if (error != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(error));                                   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

// Row-major: A[M,K] * B[K,N] = C[M,N].
__global__ void sgemm_naive(const float* A, const float* B, float* C, int M,
                            int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

bool run_benchmark_case(int M, int N, int K);

int main(int argc, char** argv) {
  const bool observe = argc == 2 && std::strcmp(argv[1], "--observe") == 0;
  const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
  if (argc > 1 && !observe && !benchmark) {
    std::fprintf(stderr, "Usage: %s [--observe|--benchmark]\n", argv[0]);
    return EXIT_FAILURE;
  }
  if (benchmark) {
    if (!run_benchmark_case(1, 1024, 1024)) return EXIT_FAILURE;
    if (!run_benchmark_case(64, 1024, 1024)) return EXIT_FAILURE;
    return run_benchmark_case(1024, 1024, 1024) ? EXIT_SUCCESS : EXIT_FAILURE;
  }
  const int M = observe ? 512 : 37;
  const int N = observe ? 512 : 53;
  const int K = observe ? 512 : 29;

  std::vector<float> hA(M * K), hB(K * N), hC(M * N);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (float& value : hA) value = distribution(rng);
  for (float& value : hB) value = distribution(rng);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, hC.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 block(16, 16);
  const dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);
  if (observe) {
    std::puts("Running repeated CUDA kernels for about 10 seconds...");
    std::fflush(stdout);
    const auto start = std::chrono::steady_clock::now();
    int launches = 0;
    do {
      for (int i = 0; i < 16; ++i) {
        sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
      }
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      launches += 16;
    } while (std::chrono::steady_clock::now() - start <
             std::chrono::seconds(10));
    std::printf("Finished %d kernel launches.\n", launches);
  } else {
    sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  float max_error = 0.0f;
  bool passed = true;
  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      double reference = 0.0;
      for (int k = 0; k < K; ++k) {
        reference += static_cast<double>(hA[row * K + k]) * hB[k * N + col];
      }
      const float expected = static_cast<float>(reference);
      const float actual = hC[row * N + col];
      const float error = std::fabs(actual - expected);
      max_error = std::fmax(max_error, error);
      if (!std::isfinite(actual) || error > 1e-4f + 1e-4f * std::fabs(expected)) {
        passed = false;
      }
    }
  }

  std::printf("shape=%dx%d; C[%d,%d] = %.6f; max absolute error = %.8f; %s\n",
              M, N, M - 1, N - 1, hC.back(), max_error,
              passed ? "PASS" : "FAIL");
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Optional timing mode. The kernel above is unchanged from the first lesson.
bool run_benchmark_case(int M, int N, int K) {
  const size_t a_count = static_cast<size_t>(M) * K;
  const size_t b_count = static_cast<size_t>(K) * N;
  const size_t c_count = static_cast<size_t>(M) * N;
  std::vector<float> hA(a_count), hB(b_count), hC(c_count);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (float& value : hA) value = distribution(rng);
  for (float& value : hB) value = distribution(rng);

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CUDA_CHECK(cudaMalloc(&dA, a_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dB, b_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dC, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), a_count * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), b_count * sizeof(float),
                        cudaMemcpyHostToDevice));

  const dim3 block(16, 16);
  const dim3 grid((N + 15) / 16, (M + 15) / 16);
  sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  const int sample_rows[] = {0, M / 2, M - 1};
  const int sample_cols[] = {0, N / 2, N - 1};
  bool passed = true;
  float max_sample_error = 0.0f;
  for (int row : sample_rows) {
    for (int col : sample_cols) {
      double reference = 0.0;
      for (int k = 0; k < K; ++k) {
        reference += static_cast<double>(hA[static_cast<size_t>(row) * K + k]) *
                     hB[static_cast<size_t>(k) * N + col];
      }
      const float expected = static_cast<float>(reference);
      const float actual = hC[static_cast<size_t>(row) * N + col];
      const float error = std::fabs(actual - expected);
      max_sample_error = std::fmax(max_sample_error, error);
      if (!std::isfinite(actual) || error > 1e-4f + 1e-4f * std::fabs(expected)) {
        passed = false;
      }
    }
  }

  std::printf("shape=%dx%dx%d  max sample error=%.8f  %s\n",
              M, N, K, max_sample_error, passed ? "PASS" : "FAIL");
  if (passed) {
    constexpr int repetitions = 50;
    for (int i = 0; i < 3; ++i) {
      sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < repetitions; ++i) {
      sgemm_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    const float ms = elapsed_ms / repetitions;
    const double gflops = 2.0 * M * N * K / (ms * 1e6);
    std::printf("  naive  %.3f ms  %.1f GFLOP/s\n", ms, gflops);
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return passed;
}
