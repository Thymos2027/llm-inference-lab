#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call) do {                                       \
  const cudaError_t error = (call);                                 \
  if (error != cudaSuccess) {                                       \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,         \
                 cudaGetErrorString(error));                        \
    std::exit(EXIT_FAILURE);                                        \
  }                                                                 \
} while (0)

namespace wmma = nvcuda::wmma;
constexpr int TILE = 16;

// C[M,N] = A[M,K] * B[K,N]. A and B are row-major FP16; C is row-major FP32.
// M, N, K must be multiples of TILE in this step.
//
// Mapping: one block = one warp = one 16x16 tile of C.
//   blockIdx.y -> which tile row of C  (rows    tile_row*16 .. +15)
//   blockIdx.x -> which tile column    (columns tile_col*16 .. +15)
// The warp walks K in steps of 16 and accumulates in one FP32 fragment.
__global__ void wmma_multi_tile(const __half* a, const __half* b, float* c,
                                int m, int n, int k) {
  const int tile_row = blockIdx.y;
  const int tile_col = blockIdx.x;

  wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k0 = 0; k0 < k; k0 += TILE) {
    // A tile: rows tile_row*16.., columns k0... One row of A has k elements.
    const __half* a_tile = a + (tile_row * TILE) * k + k0;
    // B tile: rows k0.., columns tile_col*16... One row of B has n elements.
    const __half* b_tile = b + k0 * n + tile_col * TILE;
    wmma::load_matrix_sync(a_frag, a_tile, k);
    wmma::load_matrix_sync(b_frag, b_tile, n);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  // C tile: rows tile_row*16.., columns tile_col*16... One row of C has n elements.
  float* c_tile = c + (tile_row * TILE) * n + tile_col * TILE;
  wmma::store_matrix_sync(c_tile, c_frag, n, wmma::mem_row_major);
}

// Lessons that reuse this kernel define WMMA_KERNELS_ONLY to skip the test driver.
#ifndef WMMA_KERNELS_ONLY

struct Shape { int m, n, k; };

// Returns true when every element passes.
bool run_case(Shape s, std::mt19937& rng) {
  const int m = s.m, n = s.n, k = s.k;
  std::printf("\n== M=%d N=%d K=%d ==\n", m, n, k);
  if (m % TILE || n % TILE || k % TILE || m <= 0 || n <= 0 || k <= 0) {
    std::printf("skip: this step only supports positive multiples of %d\n", TILE);
    return false;
  }

  // Raw FP32 inputs, then the FP16 copies the GPU actually receives.
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> a_raw(size_t(m) * k), b_raw(size_t(k) * n);
  for (float& x : a_raw) x = dist(rng);
  for (float& x : b_raw) x = dist(rng);
  std::vector<__half> a_h(a_raw.size()), b_h(b_raw.size());
  std::vector<float> a_q(a_raw.size()), b_q(b_raw.size());  // FP16 values as float
  for (size_t i = 0; i < a_raw.size(); ++i) {
    a_h[i] = __float2half(a_raw[i]);
    a_q[i] = __half2float(a_h[i]);
  }
  for (size_t i = 0; i < b_raw.size(); ++i) {
    b_h[i] = __float2half(b_raw[i]);
    b_q[i] = __half2float(b_h[i]);
  }

  __half *d_a = nullptr, *d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_h.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_b, b_h.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_c, size_t(m) * n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a_h.data(), a_h.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b_h.data(), b_h.size() * sizeof(__half), cudaMemcpyHostToDevice));
  // NaN fill: any C element the kernel forgets to write will fail the check.
  CUDA_CHECK(cudaMemset(d_c, 0xFF, size_t(m) * n * sizeof(float)));

  const dim3 grid(n / TILE, m / TILE);
  const dim3 block(32);
  std::printf("grid=(%d,%d) block=%d: %d warps, each owns one 16x16 C tile, %d K steps\n",
              grid.x, grid.y, block.x, grid.x * grid.y, k / TILE);
  wmma_multi_tile<<<grid, block>>>(d_a, d_b, d_c, m, n, k);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> c(size_t(m) * n);
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, c.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Two CPU references in double:
  //   ref_q   uses the FP16-quantized inputs -> judges the kernel (PASS/FAIL).
  //   ref_raw uses the original FP32 inputs  -> shows FP16 input quantization error only.
  // Tolerance scales with sum_k |a*b|, the size of the numbers being accumulated.
  int mismatches = 0, non_finite = 0;
  double max_kernel_err = 0.0, max_quant_err = 0.0, max_kernel_rel = 0.0;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double ref_q = 0.0, ref_raw = 0.0, abs_sum = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        const double pq = double(a_q[size_t(row) * k + kk]) * double(b_q[size_t(kk) * n + col]);
        ref_q += pq;
        abs_sum += std::fabs(pq);
        ref_raw += double(a_raw[size_t(row) * k + kk]) * double(b_raw[size_t(kk) * n + col]);
      }
      const float actual = c[size_t(row) * n + col];
      const double err = std::fabs(double(actual) - ref_q);
      const double tol = 1e-5 + 1e-4 * abs_sum;
      if (!std::isfinite(actual) || err > tol) {
        if (mismatches < 5) {
          std::printf("C[%d,%d]: got %.6f, expected %.6f (tol %.2e)\n",
                      row, col, actual, ref_q, tol);
        }
        ++mismatches;
      }
      if (!std::isfinite(actual)) {
        ++non_finite;
      } else {
        if (err > max_kernel_err) max_kernel_err = err;
        if (abs_sum > 0 && err / abs_sum > max_kernel_rel) max_kernel_rel = err / abs_sum;
      }
      const double qerr = std::fabs(ref_q - ref_raw);
      if (qerr > max_quant_err) max_quant_err = qerr;
    }
  }

  if (non_finite) {
    std::printf("non-finite outputs (NaN/Inf): %d of %d -- were these C elements ever written?\n",
                non_finite, m * n);
  }
  if (non_finite < m * n) {
    std::printf("kernel vs FP16-input reference (finite outputs): max abs err %.3e, max err/sum|a*b| %.3e\n",
                max_kernel_err, max_kernel_rel);
  }
  std::printf("FP16 input quantization (ref_q vs ref_raw): max abs diff %.3e\n",
              max_quant_err);
  std::printf("C[0,0]=%.6f  C[%d,%d]=%.6f\n", c[0], m - 1, n - 1, c[size_t(m) * n - 1]);
  std::printf("%s (%d mismatches of %d)\n", mismatches ? "FAIL" : "PASS", mismatches, m * n);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return mismatches == 0;
}

int main(int argc, char** argv) {
  std::vector<Shape> shapes;
  if (argc == 4) {
    shapes.push_back({std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])});
  } else {
    // 16^3: same size as the single-tile lesson.
    // 64x48x32: M != N != K, so a swapped index or wrong leading dimension shows up.
    // 256x512x128: many tiles and 8 K steps.
    shapes = {{16, 16, 16}, {64, 48, 32}, {256, 512, 128}};
  }

  std::mt19937 rng(2026);
  bool all_pass = true;
  for (const Shape& s : shapes) all_pass = run_case(s, rng) && all_pass;
  std::printf("\nOverall: %s\n", all_pass ? "PASS" : "FAIL");
  return all_pass ? EXIT_SUCCESS : EXIT_FAILURE;
}

#endif  // WMMA_KERNELS_ONLY
