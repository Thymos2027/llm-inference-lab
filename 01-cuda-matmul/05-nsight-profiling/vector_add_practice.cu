#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
// 实验 05 的真实规模：2^24 + 123 = 16,777,339 个元素。
// 多出的 123 让元素数不能被 256 整除，最后一个 block 会有越界线程。
constexpr int kElementCount = (1 << 24) + 123;

// ============================================================================
// 第 1 关：CUDA kernel（已完成）
//
// 三个内置变量：
//   threadIdx.x  当前线程在 block 内的编号，0..blockDim.x-1
//   blockDim.x   每个 block 有多少线程（这里是 256）
//   blockIdx.x   当前 block 的编号
//
// 一个线程负责一个元素，所以“全局编号”= block 编号 * block 大小 + 线程编号。
// 本轮总线程数 65537 * 256 = 16,777,472，比元素数多 133 个，
// 靠 if (index < n) 把这 133 个越界线程挡掉，否则会读写非法显存。
// ============================================================================
__global__ void vector_add_practice(const float* a, const float* b, float* c,
                                    int n) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

// ============================================================================
// 第 2 关：显存申请与数据传输（已完成）
//
// 主机内存（host）和显存（device）是两块不同的物理内存：
//   std::vector 在主机内存里，GPU 的 kernel 读不到；
//   cudaMalloc 在显存里申请一段空间，返回一个“设备指针”，只有 GPU 能用。
// 所以一次完整计算固定是四步：
//   H2D 拷入 → kernel 计算 → D2H 拷出 → cudaFree 释放。
//
// 下面保留了本关的 TODO 提示，方便回顾每次申请、拷贝和释放的作用。
// ============================================================================
int main() {
  const size_t bytes = static_cast<size_t>(kElementCount) * sizeof(float);
  const int blocks = (kElementCount + kThreads - 1) / kThreads;

  std::printf("elements=%d, blocks=%d, threads/block=%d\n", kElementCount,
              blocks, kThreads);
  std::printf("each buffer=%.1f MB, total=%.1f MB\n",
              bytes / (1024.0 * 1024.0), 3.0 * bytes / (1024.0 * 1024.0));

  // 主机端输入：用可复现的公式生成，避免依赖随机数库。（这部分已给出）
  std::vector<float> host_a(kElementCount);
  std::vector<float> host_b(kElementCount);
  std::vector<float> host_c(kElementCount);
  for (int i = 0; i < kElementCount; ++i) {
    host_a[i] = static_cast<float>(i % 251 - 125) * 0.25f;
    host_b[i] = static_cast<float>(i % 127 - 63) * 0.5f;
  }

  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;

  // --------------------------------------------------------------------------
  // TODO 2-1：申请三块显存，大小都是 bytes。
  //   函数：cudaError_t cudaMalloc(void** address, size_t bytes)
  //   形参是“设备指针的地址”，所以传 &device_a，不是 device_a。
  //   用 CUDA_CHECK 包起来，失败会打印文件名、行号和原因。
  // --------------------------------------------------------------------------
  CUDA_CHECK(cudaMalloc(&device_a, bytes));
  CUDA_CHECK(cudaMalloc(&device_b, bytes));
  CUDA_CHECK(cudaMalloc(&device_c, bytes));
  // --------------------------------------------------------------------------
  // TODO 2-2：把 host_a、host_b 拷到 device_a、device_b。
  //   函数：cudaError_t cudaMemcpy(void* dst, const void* src,
  //                               size_t bytes, cudaMemcpyKind kind)
  //   主机内存 → 显存，kind 用 cudaMemcpyHostToDevice。
  //   拷多少个字节？用变量 bytes，不要手写 64MB。
  // --------------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));
  // 以下三行（启动 + 查错 + 拷回）保持原样，先别改。
  vector_add_practice<<<blocks, kThreads>>>(device_a, device_b, device_c,
                                            kElementCount);
  CUDA_CHECK(cudaGetLastError());

  // --------------------------------------------------------------------------
  // TODO 2-3：把 device_c 拷回 host_c。
  //   函数同上，方向反过来：显存 → 主机内存，kind 用 cudaMemcpyDeviceToHost。
  //   这一句同时有一个隐藏作用：它必须等 kernel 算完才可能开始拷，
  //   所以它替你完成了“等 GPU 结束”这件事，后面计时要重新处理这一点。
  // --------------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));
  // ============================================================================
  // 第 3 关：正确性验证（已完成）
  // 遍历全部元素，比较 GPU 输出和 CPU 计算的期望值，记录最大绝对误差。
  // ============================================================================
  float max_error = 0.0f;
  for (size_t i = 0; i < kElementCount; ++i) {
    const float expected = host_a[i] + host_b[i];
    max_error = std::fmax(max_error, std::fabs(host_c[i] - expected));
  }
  const bool passed = max_error <= 1e-6f;
  std::printf("Correctness: max error %.8f  %s\n", max_error,
          passed ? "PASS" : "FAIL");

  // ============================================================================
  // 第 4 关：CUDA Event 计时（自己做）
  // 此时输入仍在显存中。只计 kernel 的时间，不把 cudaMemcpy、CPU 校验或
  // cudaMalloc/cudaFree 放进两个 event 之间。CUDA Event 测的是 GPU 时间；
  // kernel 启动是异步的，不能只用 CPU 时钟包住启动语句。
  //
  // TODO 4-1：先启动 5 次 kernel 预热，再用 cudaDeviceSynchronize 等它们完成。
  //           每次启动后仍要检查 cudaGetLastError。
  constexpr int warmups = 5;
  constexpr int repetitions = 100;
  for (int i = 0; i < warmups; ++i) {
    vector_add_practice<<<blocks, kThreads>>>(device_a, device_b, device_c, kElementCount);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  // TODO 4-2：创建 start、stop 两个 cudaEvent_t，使用 cudaEventCreate。
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  // TODO 4-3：记录 start；连续启动 100 次相同的 kernel；记录 stop。
  //           用 cudaEventSynchronize(stop) 等待计时区间结束。
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repetitions; ++i) {
    vector_add_practice<<<blocks, kThreads>>>(device_a, device_b, device_c, kElementCount);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // TODO 4-4：用 cudaEventElapsedTime 取得总毫秒数，除以 100 得到
  //           每次 kernel 的平均毫秒数，并打印结果。
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  const double average_ms = total_ms / repetitions;

  // TODO 4-5：用 cudaEventDestroy 释放两个 event。
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::printf("Average kernel time: %.3f ms\n", average_ms);
  std::printf("Effective bandwidth: %.3f GB/s\n",
              3.0 * bytes / (average_ms * 1.0e6));
  std::printf("bytes=%zu, blocks=%d, threads/block=%d\n", bytes, blocks, kThreads);
  // 完成后运行：.\scripts\build-windows.ps1 -Lesson NsightPractice
  // ============================================================================

  // --------------------------------------------------------------------------
  // TODO 2-4：释放三块显存，顺序随意，参数是设备指针（不带 &）。
  //   函数：cudaError_t cudaFree(void* address)
  // --------------------------------------------------------------------------
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
  // 完整校验失败时返回非零状态，方便构建脚本发现错误。
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
