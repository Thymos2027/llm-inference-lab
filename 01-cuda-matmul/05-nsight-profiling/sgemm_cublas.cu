#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Compile the exact lesson 04 kernels into this comparison, without its main().
#define SGEMM_KERNELS_ONLY
#include "../04-register-tiling/sgemm_register.cu"

#define CUBLAS_CHECK(call) do {                                         \
  cublasStatus_t status = (call);                                       \
  if (status != CUBLAS_STATUS_SUCCESS) {                                \
    std::fprintf(stderr, "%s:%d: cuBLAS status %d\n", __FILE__,        \
                 __LINE__, static_cast<int>(status));                  \
    std::exit(EXIT_FAILURE);                                            \
  }                                                                    \
} while (0)

enum class Method { Naive, Tiled, Register, Cublas };
const char* method_name(Method m) {
  switch (m) {
    case Method::Naive: return "naive";
    case Method::Tiled: return "tiled";
    case Method::Register: return "register";
    case Method::Cublas: return "cuBLAS";
  }
  return "unknown";
}

void launch_method(Method method, cublasHandle_t handle,
                   const float* a, const float* b, float* c,
                   int M, int N, int K) {
  if (method != Method::Cublas) {
    const dim3 block(TILE, TILE);
    const int rows = method == Method::Register ? TM * TILE : TILE;
    const dim3 grid((N + TILE - 1) / TILE, (M + rows - 1) / rows);
    if (method == Method::Naive)
      sgemm_naive<<<grid, block>>>(a, b, c, M, N, K);
    else if (method == Method::Tiled)
      sgemm_tiled<<<grid, block>>>(a, b, c, M, N, K);
    else sgemm_register<<<grid, block>>>(a, b, c, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // Row-major C=A*B is column-major C^T=B^T*A^T. The same FP32 arrays
  // are passed to cuBLAS with swapped operands and leading dimensions.
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha,
                            b, CUDA_R_32F, N, a, CUDA_R_32F, K,
                            &beta, c, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F_PEDANTIC,
                            CUBLAS_GEMM_DEFAULT));
}

float benchmark(Method method, cublasHandle_t handle,
                const float* a, const float* b, float* c,
                int M, int N, int K) {
  constexpr int warmups = 3, repetitions = 50;
  for (int i = 0; i < warmups; ++i)
    launch_method(method, handle, a, b, c, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repetitions; ++i)
    launch_method(method, handle, a, b, c, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / repetitions;
}

float cpu_reference(const std::vector<float>& a, const std::vector<float>& b,
                    int row, int col, int N, int K) {
  double sum = 0;
  for (int k = 0; k < K; ++k)
    sum += static_cast<double>(a[static_cast<size_t>(row) * K + k]) *
           b[static_cast<size_t>(k) * N + col];
  return static_cast<float>(sum);
}

bool close_enough(float actual, float expected) {
  return std::isfinite(actual) &&
         std::fabs(actual - expected) <= 1e-3f + 2e-4f * std::fabs(expected);
}

bool run_case(int M, int N, int K, bool measure, cublasHandle_t handle) {
  const size_t a_count = static_cast<size_t>(M) * K;
  const size_t b_count = static_cast<size_t>(K) * N;
  const size_t c_count = static_cast<size_t>(M) * N;
  std::vector<float> a(a_count), b(b_count), c(c_count), cublas_reference(c_count);
  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
  for (float& x : a) x = distribution(rng);
  for (float& x : b) x = distribution(rng);
  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CUDA_CHECK(cudaMalloc(&da, a_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&db, b_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dc, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a.data(), a_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b.data(), b_count * sizeof(float), cudaMemcpyHostToDevice));
  bool passed = true;
  std::printf("shape=%dx%dx%d (M,N,K)\n", M, N, K);
  const Method methods[] = {Method::Cublas, Method::Naive, Method::Tiled, Method::Register};
  for (Method method : methods) {
    launch_method(method, handle, da, db, dc, M, N, K);
    CUDA_CHECK(cudaMemcpy(c.data(), dc, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    if (method == Method::Cublas) cublas_reference = c;
    float max_error = 0;
    size_t mismatches = 0;
    // All outputs for the irregular case; nine representative outputs on
    // measured cases, matching the earlier SGEMM lessons' CPU check scope.
    for (int row = 0; row < M; ++row) {
      if (measure && row != 0 && row != M / 2 && row != M - 1) continue;
      for (int col = 0; col < N; ++col) {
        if (measure && col != 0 && col != N / 2 && col != N - 1) continue;
        const float expected = cpu_reference(a, b, row, col, N, K);
        const float actual = c[static_cast<size_t>(row) * N + col];
        max_error = std::fmax(max_error, std::fabs(actual - expected));
        if (!close_enough(actual, expected)) ++mismatches;
      }
    }
    size_t gpu_mismatches = 0;
    if (method != Method::Cublas)
      for (size_t i = 0; i < c_count; ++i)
        if (!close_enough(c[i], cublas_reference[i])) ++gpu_mismatches;
    std::printf("  %-8s CPU-check max error %.6f  CPU mismatches=%zu  cuBLAS mismatches=%zu  %s\n",
                method_name(method), max_error, mismatches, gpu_mismatches,
                mismatches || gpu_mismatches ? "FAIL" : "PASS");
    passed &= mismatches == 0 && gpu_mismatches == 0;
  }
  if (passed && measure) {
    const Method timed[] = {Method::Naive, Method::Tiled, Method::Register, Method::Cublas};
    float samples[4][3];
    for (int round = 0; round < 3; ++round)
      for (int method = 0; method < 4; ++method)
        samples[method][round] = benchmark(timed[method], handle, da, db, dc, M, N, K);
    for (int method = 0; method < 4; ++method) {
      std::sort(samples[method], samples[method] + 3);
      const float ms = samples[method][1];
      std::printf("  %-8s median %.5f ms  %.1f GFLOP/s  (3 x 50 launches)\n",
                  method_name(timed[method]), ms, 2.0 * M * N * K / (ms * 1e6));
    }
  }
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return passed;
}

int main(int argc, char** argv) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
  CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
  if (argc == 3 && std::strcmp(argv[1], "--profile") == 0) {
    Method method;
    if (std::strcmp(argv[2], "naive") == 0) method = Method::Naive;
    else if (std::strcmp(argv[2], "tiled") == 0) method = Method::Tiled;
    else if (std::strcmp(argv[2], "register") == 0) method = Method::Register;
    else if (std::strcmp(argv[2], "cublas") == 0) method = Method::Cublas;
    else return EXIT_FAILURE;
    constexpr int M = 1024, N = 1024, K = 1024;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CUDA_CHECK(cudaMalloc(&a, static_cast<size_t>(M) * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, static_cast<size_t>(K) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c, static_cast<size_t>(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(a, 0, static_cast<size_t>(M) * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(b, 0, static_cast<size_t>(K) * N * sizeof(float)));
    launch_method(method, handle, a, b, c, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUBLAS_CHECK(cublasDestroy(handle));
    return EXIT_SUCCESS;
  }
  if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--profile naive|tiled|register|cublas]\n", argv[0]);
    return EXIT_FAILURE;
  }
  const bool passed = run_case(37, 53, 29, false, handle) &&
      run_case(1, 1024, 1024, true, handle) &&
      run_case(64, 1024, 1024, true, handle) &&
      run_case(1024, 1024, 1024, true, handle);
  CUBLAS_CHECK(cublasDestroy(handle));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
