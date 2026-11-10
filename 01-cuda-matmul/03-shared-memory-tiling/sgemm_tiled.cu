#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                   cudaGetErrorString(error));                                   \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                           \
  } while (0)

constexpr int TILE = 16;

// Baseline: one thread computes one element of C.
__global__ void sgemm_naive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;

  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

// Each 16x16 block of threads computes one 16x16 tile of C.
__global__ void sgemm_tiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
  __shared__ float a_tile[TILE][TILE];
  __shared__ float b_tile[TILE][TILE];

  const int local_row = threadIdx.y;
  const int local_col = threadIdx.x;
  const int row = blockIdx.y * TILE + local_row;
  const int col = blockIdx.x * TILE + local_col;
  float sum = 0.0f;

  for (int start = 0; start < K; start += TILE) {
    // Every thread loads one A element and one B element.
    // Out-of-range entries of the last tiles are zero-filled.
    const int a_col = start + local_col;
    const int b_row = start + local_row;
    a_tile[local_row][local_col] =
        (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    b_tile[local_row][local_col] =
        (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads();  // Wait until the whole block finishes loading.

    for (int k = 0; k < TILE; ++k) {
      sum += a_tile[local_row][k] * b_tile[k][local_col];
    }
    __syncthreads();  // Wait before any thread overwrites the tiles.
  }

  if (row < M && col < N) C[row * N + col] = sum;
}

enum class Variant { Naive, Tiled };

const char* name(Variant variant) {
  return variant == Variant::Naive ? "naive" : "tiled";
}

void launch(Variant variant, const float* A, const float* B, float* C,
            int M, int N, int K) {
  const dim3 block(TILE, TILE);
  const dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
  if (variant == Variant::Naive) {
    sgemm_naive<<<grid, block>>>(A, B, C, M, N, K);
  } else {
    sgemm_tiled<<<grid, block>>>(A, B, C, M, N, K);
  }
  CUDA_CHECK(cudaGetLastError());
}

float benchmark(Variant variant, const float* A, const float* B, float* C,
                int M, int N, int K) {
  constexpr int repetitions = 50;
  for (int i = 0; i < 3; ++i) launch(variant, A, B, C, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repetitions; ++i) launch(variant, A, B, C, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed_ms / repetitions;
}

float cpu_reference(const std::vector<float>& A, const std::vector<float>& B,
                    int row, int col, int N, int K) {
  double sum = 0.0;
  for (int k = 0; k < K; ++k) {
    sum += static_cast<double>(A[static_cast<size_t>(row) * K + k]) *
           B[static_cast<size_t>(k) * N + col];
  }
  return static_cast<float>(sum);
}

bool close_enough(float actual, float expected) {
  return std::isfinite(actual) &&
         std::fabs(actual - expected) <= 1e-4f + 1e-4f * std::fabs(expected);
}

bool run_case(int M, int N, int K, bool measure) {
  const size_t a_count = static_cast<size_t>(M) * K;
  const size_t b_count = static_cast<size_t>(K) * N;
  const size_t c_count = static_cast<size_t>(M) * N;
  std::vector<float> hA(a_count), hB(b_count), hNaive(c_count), hTiled(c_count);
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

  launch(Variant::Naive, dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaMemcpy(hNaive.data(), dC, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));
  launch(Variant::Tiled, dA, dB, dC, M, N, K);
  CUDA_CHECK(cudaMemcpy(hTiled.data(), dC, c_count * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool passed = true;
  float max_difference = 0.0f;
  for (size_t i = 0; i < c_count; ++i) {
    max_difference = std::fmax(max_difference, std::fabs(hTiled[i] - hNaive[i]));
    if (!close_enough(hTiled[i], hNaive[i])) passed = false;
  }

  // Fully check small irregular shapes. For large shapes, check representative
  // elements against the CPU and compare every GPU output against the baseline.
  const int sample_rows[] = {0, M / 2, M - 1};
  const int sample_cols[] = {0, N / 2, N - 1};
  for (int row = 0; row < M; ++row) {
    if (measure && row != sample_rows[0] && row != sample_rows[1] &&
        row != sample_rows[2]) continue;
    for (int col = 0; col < N; ++col) {
      if (measure && col != sample_cols[0] && col != sample_cols[1] &&
          col != sample_cols[2]) continue;
      const float expected = cpu_reference(hA, hB, row, col, N, K);
      const size_t index = static_cast<size_t>(row) * N + col;
      if (!close_enough(hNaive[index], expected) ||
          !close_enough(hTiled[index], expected)) passed = false;
    }
  }

  std::printf("shape=%dx%dx%d  max GPU difference=%.8f  %s\n",
              M, N, K, max_difference, passed ? "PASS" : "FAIL");
  if (passed && measure) {
    for (Variant variant : {Variant::Naive, Variant::Tiled}) {
      const float ms = benchmark(variant, dA, dB, dC, M, N, K);
      const double gflops = 2.0 * M * N * K / (ms * 1e6);
      std::printf("  %-6s %.3f ms  %.1f GFLOP/s\n", name(variant), ms, gflops);
    }
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return passed;
}

int main() {
  if (!run_case(37, 53, 29, false)) return EXIT_FAILURE;
  if (!run_case(129, 257, 65, false)) return EXIT_FAILURE;
  if (!run_case(1, 1024, 1024, true)) return EXIT_FAILURE;
  if (!run_case(64, 1024, 1024, true)) return EXIT_FAILURE;
  return run_case(1024, 1024, 1024, true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
