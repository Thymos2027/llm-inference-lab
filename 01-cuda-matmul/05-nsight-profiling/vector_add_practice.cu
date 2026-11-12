#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error = (call);                                           \
    if (error != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(error));                                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

// 第一关：一个线程负责一个数组元素。
//
// 现在只修改这个 kernel 中标有 TODO 的两处。main 函数暂时作为测试框架，
// 不要求你现在理解其中的 cudaMalloc、cudaMemcpy 等代码。
__global__ void vector_add_practice(const float* a, const float* b, float* c,
                                    int n) {
  // TODO 1：用 blockIdx.x、blockDim.x 和 threadIdx.x 算出全局线程编号。
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  c[index] = a[index] + b[index];
  // if (index < n)
  // {
  //   c[index] = a[index] + b[index];
  // }
  

  // TODO 2：只有 index 对应有效元素时，才计算 c[index] = a[index] + b[index]。
}

int main() {
  constexpr int n = 10;
  constexpr int threads_per_block = 4;
  constexpr int blocks = (n + threads_per_block - 1) / threads_per_block;

  const float host_a[n] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const float host_b[n] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
  float host_c[n] = {};

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  const size_t bytes = n * sizeof(float);

  CUDA_CHECK(cudaMalloc(&device_a, bytes));
  CUDA_CHECK(cudaMalloc(&device_b, bytes));
  CUDA_CHECK(cudaMalloc(&device_c, bytes));
  CUDA_CHECK(cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(device_c, 0, bytes));

  vector_add_practice<<<blocks, threads_per_block>>>(device_a, device_b, device_c,
                                                     n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost));

  bool passed = true;
  std::printf("blocks=%d, threads/block=%d, total threads=%d, elements=%d\n",
              blocks, threads_per_block, blocks * threads_per_block, n);
  for (int i = 0; i < n; ++i) {
    const float expected = host_a[i] + host_b[i];
    const bool correct = host_c[i] == expected;
    passed = passed && correct;
    std::printf("i=%2d  %.1f + %.1f = %.1f  expected %.1f  %s\n", i,
                host_a[i], host_b[i], host_c[i], expected,
                correct ? "OK" : "WRONG");
  }
  std::printf("Result: %s\n", passed ? "PASS" : "NOT PASS YET");

  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));

  // 练习未完成时也正常退出，方便反复构建、观察输出。
  return EXIT_SUCCESS;
}
