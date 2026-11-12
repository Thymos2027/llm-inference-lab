# 实验 05：第一次用 Nsight 分析 CUDA Kernel

前四个实验主要靠代码、正确性和 CUDA Event 计时理解 SGEMM。本实验先暂停继续优化矩阵乘法，用一个简单的 Vector Add 学会回答三个问题：

1. kernel 在 GPU 上实际执行了多久？
2. 它主要忙于计算，还是忙于搬运数据？
3. block、线程、寄存器和 occupancy 在报告中的什么位置？

完整路线是 Vector Add → Matrix Transpose → 分析现有 SGEMM → 对照 cuBLAS。本页先完成 Vector Add。

## Kernel 做了什么

每个线程处理一个位置：

```cpp
c[index] = a[index] + b[index];
```

一次计算只做一次浮点加法，却要读取 `a[index]`、读取 `b[index]`、写入 `c[index]`。FP32 每个元素 4 字节，所以每处理一个元素，按代码语义计算约搬运 12 字节、执行 1 次浮点运算。它适合用来认识**显存带宽瓶颈**。

程序使用 `2^24 + 123` 个元素。多出来的 123 让元素总数不能被 256 整除，最后一个 block 会出现越界线程，因此 kernel 仍需要 `index < n`。

## 第一步：正确性与计时

在项目根目录运行：

```powershell
.\scripts\build-windows.ps1 -Lesson Nsight
```

程序会完成：

- 生成可复现的输入；
- 与 CPU 计算的 `a[i] + b[i]` 逐元素比较；
- 预热 5 次，再用 CUDA Event 测量 100 次 kernel 的平均时间；
- 按 `读取 A + 读取 B + 写入 C`，也就是每元素 12 字节，计算有效带宽。

这里的有效带宽是根据程序语义估算的，不等于 Nsight 直接测得的 DRAM 字节数。缓存、写事务和硬件统计口径可能让两者不同。

## 第二步：生成两份 Nsight 报告

```powershell
.\scripts\profile-windows.ps1 -Experiment VectorAdd
```

报告保存在 `.local/profiles/05-vector-add/`，不会提交到 Git：

- `vector-add-systems.nsys-rep`：Nsight Systems 时间线；
- `vector-add-compute.ncu-repz`：Nsight Compute kernel 指标。

profiling 脚本会以 `--profile` 运行程序。在这个模式下只启动一次 Vector Add，不执行 100 次 benchmark，也不做大数组的 CPU 校验。这样 Nsight Compute 只需要分析一个目标 kernel。

## 两个工具分别看什么

### Nsight Systems：先看全局时间线

打开 `vector-add-systems.nsys-rep`，依次找到：

1. CUDA API 行中的 `cudaMalloc`、`cudaMemset`、kernel launch 和同步；
2. GPU 行中的 `vector_add`；
3. CPU 发起 kernel 与 GPU 执行 kernel 的时间关系。

Nsight Systems 适合回答“什么时候发生、持续多久、CPU 和 GPU 是否互相等待”。它不会替代 CUDA Event 的稳定 benchmark。

### Nsight Compute：再看单个 kernel

打开 `vector-add-compute.ncu-repz`，先只看四页：

| 页面 | 先回答的问题 |
| --- | --- |
| Launch Statistics | grid、block 是否与代码一致？每个 block 是否是 256 个线程？ |
| Occupancy | 实际和理论 occupancy 是多少？限制因素是什么？ |
| GPU Speed of Light | Memory Throughput 与 Compute Throughput 谁更高？ |
| Memory Workload Analysis | 全局内存访问量、吞吐和访问模式怎样？ |

不要把“occupancy 越高”直接等同于“kernel 越快”。本实验先学会找到指标；后面的 transpose 和 SGEMM 会用对照实验判断指标变化是否真的改善性能。

## 本机第一次采集结果

2026-11-11 在 RTX 5060 Ti 上运行三次普通 benchmark，kernel 平均时间分别为 `0.524 / 0.512 / 0.529 ms`，中位数 `0.524 ms`，按代码语义估算的有效带宽为 `384.5 GB/s`。三次均通过逐元素检查。

同一份代码的单次 profiler 结果如下：

| 来源 | 观察值 |
| --- | ---: |
| Nsight Systems kernel 时间 | `0.498 ms` |
| Nsight Compute kernel 时间 | `0.486 ms` |
| Nsight Compute Memory Throughput | `403.12 GB/s` |
| DRAM Throughput | `90.07%` |
| Compute (SM) Throughput | `9.34%` |
| 每线程寄存器 | `16` |
| 理论 / 实际 occupancy | `100% / 82.99%` |

这里最重要的不是几个时间之间相差多少，而是 **DRAM Throughput 约 90%，Compute Throughput 只有约 9%**。GPU 大部分压力来自读取和写入数据；增加更多普通浮点加法通常不能解决瓶颈。这就是“Vector Add 受显存带宽限制”的直接证据。

CUDA Event、Nsight Systems 和 Nsight Compute 来自不同运行，采集方式也不同，因此 `0.524`、`0.498` 和 `0.486 ms` 不要求完全相等。稳定比较速度仍以相同条件下多次 CUDA Event 计时为准，profiler 用于解释原因。

报告中的 Grid Size 是 `65,537`，Block Size 是 `256`，因此总共启动 `16,777,472` 个线程；元素只有 `16,777,339` 个，最后多出的 133 个线程被 `index < n` 挡住。这也把边界判断和报告中的启动配置对应了起来。

## 这节的检查题

完成运行和报告采集后，尝试回答：

1. 为什么每个元素只算 1 次加法，却按 12 字节计算有效带宽？
2. 为什么元素数不能被 256 整除时仍然能得到正确结果？
3. Nsight Compute 中 Memory Throughput 和 Compute Throughput 哪个更接近上限？这说明了什么？
4. Nsight Systems 的 kernel 时间为什么不必与 100 次 CUDA Event 的平均时间完全相同？

下一小节将实现矩阵复制、naive transpose 和 shared-memory transpose，让合并访存的差异直接出现在报告中。
