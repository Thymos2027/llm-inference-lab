#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// All three kernels compute row-major C[M,N] = A[M,K] * B[K,N].
// Original lesson: a warp spans two groups of 16 adjacent columns.
__global__ void sgemm_original(const float* A, const float* B, float* C,
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

// One warp covers 32 adjacent columns of one output row.
__global__ void sgemm_columns(const float* A, const float* B, float* C,
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

// Same 32x8 block and arithmetic, but adjacent lanes cover different rows.
// B reads are broadcast; A reads and C writes are separated by row strides.
__global__ void sgemm_rows(const float* A, const float* B, float* C,
                           int M, int N, int K) {
  const int row = blockIdx.y * blockDim.x + threadIdx.x;
  const int col = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float sum = 0.0f;
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

enum class Variant { Original, Columns, Rows };

const char* name(Variant variant) {
  switch (variant) {
    case Variant::Original: return "original 16x16";
    case Variant::Columns: return "columns 32x8";
    case Variant::Rows: return "rows 32x8";
  }
  return "unknown";
}

void launch(Variant variant, const float* A, const float* B, float* C,
            int M, int N, int K) {
  if (variant == Variant::Original) {
    const dim3 block(16, 16);
    const dim3 grid((N + 15) / 16, (M + 15) / 16);
    sgemm_original<<<grid, block>>>(A, B, C, M, N, K);
  } else if (variant == Variant::Columns) {
    const dim3 block(32, 8);
    const dim3 grid((N + 31) / 32, (M + 7) / 8);
    sgemm_columns<<<grid, block>>>(A, B, C, M, N, K);
  } else {
    const dim3 block(32, 8);
    const dim3 grid((N + 7) / 8, (M + 31) / 32);
    sgemm_rows<<<grid, block>>>(A, B, C, M, N, K);
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

bool run_case(int M, int N, int K, bool measure) {
  const size_t a_count = static_cast<size_t>(M) * K;
  const size_t b_count = static_cast<size_t>(K) * N;
  const size_t c_count = static_cast<size_t>(M) * N;
  std::vector<float> hA(a_count), hB(b_count), hC(c_count), baseline(c_count);
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

  bool passed = true;
  const Variant variants[] = {Variant::Original, Variant::Columns, Variant::Rows};
  std::printf("shape=%dx%dx%d\n", M, N, K);
  for (Variant variant : variants) {
    launch(variant, dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, c_count * sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (variant == Variant::Original) baseline = hC;

    bool variant_passed = true;
    float max_error = 0.0f;
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        const size_t index = static_cast<size_t>(row) * N + col;
        float expected = baseline[index];
        const bool check_cpu = !measure ||
            ((row == 0 || row == M / 2 || row == M - 1) &&
             (col == 0 || col == N / 2 || col == N - 1));
        if (check_cpu) {
          double reference = 0.0;
          for (int k = 0; k < K; ++k) {
            reference += static_cast<double>(hA[static_cast<size_t>(row) * K + k]) *
                         hB[static_cast<size_t>(k) * N + col];
          }
          expected = static_cast<float>(reference);
        }
        const float actual = hC[index];
        const float error = std::fabs(actual - expected);
        max_error = std::fmax(max_error, error);
        if (!std::isfinite(actual) || error > 1e-4f + 1e-4f * std::fabs(expected)) {
          variant_passed = false;
        }
      }
    }
    std::printf("  %-16s max error %.8f  %s\n", name(variant), max_error,
                variant_passed ? "PASS" : "FAIL");
    passed &= variant_passed;
  }

  if (passed && measure) {
    for (Variant variant : variants) {
      const float ms = benchmark(variant, dA, dB, dC, M, N, K);
      const double gflops = 2.0 * M * N * K / (ms * 1e6);
      std::printf("  %-16s %.3f ms  %.1f GFLOP/s\n", name(variant), ms, gflops);
    }
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return passed;
}

int main() {
  const bool small_passed = run_case(37, 53, 29, false);
  if (!small_passed) return EXIT_FAILURE;
  if (!run_case(1, 1024, 1024, true)) return EXIT_FAILURE;
  if (!run_case(64, 1024, 1024, true)) return EXIT_FAILURE;
  return run_case(1024, 1024, 1024, true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
