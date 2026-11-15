#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// One 32-thread warp computes C[16,16] = A[16,16] * B[16,16].
// A and B are row-major FP16; C is row-major FP32.
__global__ void wmma_one_tile(const __half* a, const __half* b, float* c) {
  // TODO 1: Declare matrix_a, matrix_b, and accumulator fragments.
  //         Use a 16x16x16 tile, FP16 inputs, and FP32 accumulation.
  wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, __half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;
  // TODO 2: Initialize the accumulator to zero.
  wmma::fill_fragment(c_frag, 0.0f);
  // TODO 3: Load A and B. Both leading dimensions are TILE.
  wmma::load_matrix_sync(a_frag, a, TILE);
  wmma::load_matrix_sync(b_frag, b, TILE);
  // TODO 4: Multiply and accumulate with mma_sync.
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  // TODO 5: Store the accumulator to row-major C.
  wmma::store_matrix_sync(c, c_frag, TILE, wmma::mem_row_major);
}

int main() {
  constexpr int count = TILE * TILE;
  std::vector<__half> a(count), b(count);
  std::vector<float> c(count);

  // A is diagonal: A[i,i] = i+1. B[k,j] = 16*k+j.
  // First predict C[1,2] and C[2,1] by hand.

  for (int row = 0; row < TILE; ++row) {
    for (int col = 0; col < TILE; ++col) {
      float value = float(row * TILE + col);
      if (row == 1 && col == 2) value = 1.0001f;
      b[row * TILE + col] = __float2half(value);
      a[row * TILE + col] = __float2half(row == col ? float(row + 1) : 0.0f);
    }
  }
  std::printf("B[1,2]: raw=%.7f, after FP16=%.7f\n",
            1.0001f, __half2float(b[TILE + 2]));

  __half *d_a = nullptr, *d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, count * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_b, count * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_c, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), count * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_c, 0, count * sizeof(float)));

  wmma_one_tile<<<1, 32>>>(d_a, d_b, d_c);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, count * sizeof(float), cudaMemcpyDeviceToHost));

  int mismatches = 0;
  for (int row = 0; row < TILE; ++row) {
    for (int col = 0; col < TILE; ++col) {
      double expected = 0.0;
      for (int k = 0; k < TILE; ++k) {
        expected += double(__half2float(a[row * TILE + k])) *
                    double(__half2float(b[k * TILE + col]));
      }
      const float actual = c[row * TILE + col];
      if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-5) {
        if (mismatches < 5) {
          std::printf("C[%d,%d]: got %.6f, expected %.6f\n",
                      row, col, actual, expected);
        }
        ++mismatches;
      }
    }
  }
  std::printf("C[1,2]=%.1f, C[2,1]=%.1f; %s (%d mismatches)\n",
              c[TILE + 2], c[2 * TILE + 1],
              mismatches ? "FAIL" : "PASS", mismatches);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  return mismatches ? EXIT_FAILURE : EXIT_SUCCESS;
}
