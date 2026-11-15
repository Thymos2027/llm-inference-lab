#include <cublas_v2.h>

#include <algorithm>
#include <cstring>

// Compile the exact multi-tile WMMA kernel from the reference, without its main().
#define WMMA_KERNELS_ONLY
#include "wmma_multitile_reference.cu"

// Same-precision comparison: A, B are FP16 row-major, C is FP32 row-major,
// and both methods accumulate in FP32.
//   - wmma:   one warp per 16x16 C tile, A/B loaded straight from global memory.
//   - cuBLAS: cublasGemmEx with CUDA_R_16F inputs, CUDA_R_32F output and
//             CUBLAS_COMPUTE_32F (FP32 accumulation, Tensor Cores allowed).

#define CUBLAS_CHECK(call) do {                                         \
  cublasStatus_t status = (call);                                       \
  if (status != CUBLAS_STATUS_SUCCESS) {                                \
    std::fprintf(stderr, "%s:%d: cuBLAS status %d\n", __FILE__,        \
                 __LINE__, static_cast<int>(status));                  \
    std::exit(EXIT_FAILURE);                                            \
  }                                                                    \
} while (0)

struct Shape { int m, n, k; };
enum class Method { Wmma, Cublas };
const char* method_name(Method m) { return m == Method::Wmma ? "wmma" : "cuBLAS"; }

void launch_method(Method method, cublasHandle_t handle,
                   const __half* a, const __half* b, float* c,
                   int M, int N, int K) {
  if (method == Method::Wmma) {
    wmma_multi_tile<<<dim3(N / TILE, M / TILE), 32>>>(a, b, c, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  // Row-major C=A*B is column-major C^T=B^T*A^T: swap operands and M/N.
  const float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha,
                            b, CUDA_R_16F, N, a, CUDA_R_16F, K,
                            &beta, c, CUDA_R_32F, N,
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

// Average milliseconds per launch over `repetitions` back-to-back launches.
float benchmark(Method method, cublasHandle_t handle,
                const __half* a, const __half* b, float* c,
                int M, int N, int K) {
  constexpr int warmups = 3, repetitions = 50;
  for (int i = 0; i < warmups; ++i) launch_method(method, handle, a, b, c, M, N, K);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repetitions; ++i) launch_method(method, handle, a, b, c, M, N, K);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsed = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return elapsed / repetitions;
}

// Double-precision reference from the FP16-converted inputs, plus sum|a*b|
// so the tolerance scales with the magnitude of the accumulated terms.
void cpu_reference(const std::vector<float>& a_q, const std::vector<float>& b_q,
                   int row, int col, int N, int K, double* value, double* abs_sum) {
  double sum = 0, mag = 0;
  for (int k = 0; k < K; ++k) {
    const double p = double(a_q[size_t(row) * K + k]) * b_q[size_t(k) * N + col];
    sum += p;
    mag += std::fabs(p);
  }
  *value = sum;
  *abs_sum = mag;
}

bool within(double actual, double expected, double abs_sum) {
  return std::isfinite(actual) && std::fabs(actual - expected) <= 1e-5 + 1e-4 * abs_sum;
}

bool run_case(Shape s, bool full_check, cublasHandle_t handle) {
  const int M = s.m, N = s.n, K = s.k;
  std::printf("\n== M=%d N=%d K=%d ==\n", M, N, K);
  const size_t a_count = size_t(M) * K, b_count = size_t(K) * N, c_count = size_t(M) * N;

  std::mt19937 rng(2026);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__half> a_h(a_count), b_h(b_count);
  std::vector<float> a_q(a_count), b_q(b_count);  // FP16 values seen by the GPU, as float
  for (size_t i = 0; i < a_count; ++i) { a_h[i] = __float2half(dist(rng)); a_q[i] = __half2float(a_h[i]); }
  for (size_t i = 0; i < b_count; ++i) { b_h[i] = __float2half(dist(rng)); b_q[i] = __half2float(b_h[i]); }

  __half *da = nullptr, *db = nullptr;
  float* dc = nullptr;
  CUDA_CHECK(cudaMalloc(&da, a_count * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&db, b_count * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dc, c_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(da, a_h.data(), a_count * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, b_h.data(), b_count * sizeof(__half), cudaMemcpyHostToDevice));

  // Rows/columns checked against the CPU: all of them for small shapes,
  // first / middle / last otherwise (9 positions).
  std::vector<int> rows, cols;
  for (int r = 0; r < M; ++r) if (full_check || r == 0 || r == M / 2 || r == M - 1) rows.push_back(r);
  for (int c = 0; c < N; ++c) if (full_check || c == 0 || c == N / 2 || c == N - 1) cols.push_back(c);

  std::vector<float> c_out[2];
  bool passed = true;
  for (Method method : {Method::Cublas, Method::Wmma}) {
    CUDA_CHECK(cudaMemset(dc, 0xFF, c_count * sizeof(float)));  // NaN: catch unwritten outputs
    launch_method(method, handle, da, db, dc, M, N, K);
    std::vector<float>& c = c_out[int(method)];
    c.resize(c_count);
    CUDA_CHECK(cudaMemcpy(c.data(), dc, c_count * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0;
    size_t mismatches = 0;
    for (int r : rows)
      for (int col : cols) {
        double expected, abs_sum;
        cpu_reference(a_q, b_q, r, col, N, K, &expected, &abs_sum);
        const double actual = c[size_t(r) * N + col];
        max_err = std::max(max_err, std::fabs(actual - expected));
        if (!within(actual, expected, abs_sum)) ++mismatches;
      }
    std::printf("  %-6s vs FP16-input CPU reference: %zu points, max abs err %.3e, %s\n",
                method_name(method), rows.size() * cols.size(), max_err,
                mismatches ? "FAIL" : "PASS");
    passed &= mismatches == 0;
  }
  // Whole-matrix GPU-vs-GPU check. Both accumulate in FP32 but in different
  // orders, so they agree only up to rounding: use a K-scaled tolerance.
  double max_diff = 0;
  size_t gpu_mismatches = 0;
  const double gpu_tol = 1e-5 * K;  // |a|,|b| <= 1, so sum|a*b| <= K
  for (size_t i = 0; i < c_count; ++i) {
    const double d = std::fabs(double(c_out[0][i]) - c_out[1][i]);
    if (!(d <= gpu_tol)) ++gpu_mismatches;  // also counts NaN
    if (std::isfinite(d)) max_diff = std::max(max_diff, d);
  }
  std::printf("  wmma vs cuBLAS, all %zu outputs: max abs diff %.3e (tol %.1e), %s\n",
              c_count, max_diff, gpu_tol, gpu_mismatches ? "FAIL" : "PASS");
  passed &= gpu_mismatches == 0;

  if (passed && !full_check) {
    // Interleave methods across 3 rounds; report the median round.
    float samples[2][3];
    for (int round = 0; round < 3; ++round)
      for (Method method : {Method::Wmma, Method::Cublas})
        samples[int(method)][round] = benchmark(method, handle, da, db, dc, M, N, K);
    float ms[2];
    for (Method method : {Method::Wmma, Method::Cublas}) {
      float* x = samples[int(method)];
      std::sort(x, x + 3);
      ms[int(method)] = x[1];
      std::printf("  %-6s median %.5f ms  %8.1f GFLOP/s  (rounds %.5f / %.5f / %.5f ms, 3 x 50 launches)\n",
                  method_name(method), x[1], 2.0 * M * N * K / (x[1] * 1e6), x[0], x[1], x[2]);
    }
    std::printf("  cuBLAS / wmma speed ratio: %.2fx\n", ms[int(Method::Wmma)] / ms[int(Method::Cublas)]);
  }
  CUDA_CHECK(cudaFree(da));
  CUDA_CHECK(cudaFree(db));
  CUDA_CHECK(cudaFree(dc));
  return passed;
}

int main(int argc, char** argv) {
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));

  // --profile wmma|cublas [M N K]: one launch with zero inputs, for ncu (default 1024^3).
  if ((argc == 3 || argc == 6) && std::strcmp(argv[1], "--profile") == 0) {
    Method method;
    if (std::strcmp(argv[2], "wmma") == 0) method = Method::Wmma;
    else if (std::strcmp(argv[2], "cublas") == 0) method = Method::Cublas;
    else return EXIT_FAILURE;
    int M = 1024, N = 1024, K = 1024;
    if (argc == 6) { M = std::atoi(argv[3]); N = std::atoi(argv[4]); K = std::atoi(argv[5]); }
    if (M <= 0 || N <= 0 || K <= 0 || M % TILE || N % TILE || K % TILE) return EXIT_FAILURE;
    __half *a = nullptr, *b = nullptr;
    float* c = nullptr;
    CUDA_CHECK(cudaMalloc(&a, size_t(M) * K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b, size_t(K) * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&c, size_t(M) * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(a, 0, size_t(M) * K * sizeof(__half)));
    CUDA_CHECK(cudaMemset(b, 0, size_t(K) * N * sizeof(__half)));
    launch_method(method, handle, a, b, c, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));
    CUBLAS_CHECK(cublasDestroy(handle));
    return EXIT_SUCCESS;
  }
  if (argc != 1) {
    std::fprintf(stderr, "Usage: %s [--profile wmma|cublas [M N K]]\n", argv[0]);
    return EXIT_FAILURE;
  }

  bool passed = run_case({64, 48, 32}, true, handle);  // small: every output vs CPU, no timing
  const Shape timed[] = {{16, 1024, 1024}, {64, 1024, 1024}, {1024, 1024, 1024},
                         {2048, 2048, 2048}, {4096, 4096, 4096}};
  for (const Shape& s : timed) passed = run_case(s, false, handle) && passed;
  std::printf("\nOverall: %s\n", passed ? "PASS" : "FAIL");
  CUBLAS_CHECK(cublasDestroy(handle));
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
