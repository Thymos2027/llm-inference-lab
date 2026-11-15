# 实验 05：第一次用 Nsight 分析 CUDA Kernel

前四个实验主要靠代码、正确性和 CUDA Event 计时理解 SGEMM。本实验先暂停继续优化矩阵乘法，用一个简单的 Vector Add 学会回答三个问题：

1. kernel 在 GPU 上实际执行了多久？
2. 它主要忙于计算，还是忙于搬运数据？
3. block、线程、寄存器和 occupancy 在报告中的什么位置？

完整路线是 Vector Add → Matrix Transpose → 分析现有 SGEMM → 对照 cuBLAS。本页保留每一步的代码、测量口径和本机结果。

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

## 矩阵复制与转置

[transpose.cu](transpose.cu) 中的三个 kernel 都用 `32×8` 线程块处理 `32×32` tile，输入为行主序 `A[height,width]`。`copy` 读写相同位置；`naive` 读取连续的行，但相邻线程写到相隔 `height` 个 FP32 元素的位置；`shared` 先把 tile 放进 `float tile[32][33]`，同步后交换线程的读写坐标，让全局读取和写入都连续。第 33 列用于避免转置读 shared memory 时的 32 路 bank conflict。边界条件覆盖非 32 倍数的宽和高。

```powershell
.\scripts\build-windows.ps1 -Lesson Transpose
.\scripts\profile-windows.ps1 -Experiment Transpose
```

2026-11-14 在本机 RTX 3060 8 GB（`sm_86`、驱动 `621.24`）上，CUDA `nvcc 13.3.73`、`-O2 -std=c++17 -arch=sm_86 -lineinfo`。`37×53`、`129×257`、`4096×4096` 的所有输出逐元素精确相等。下表只计 GPU kernel：每版预热 5 次，每轮 CUDA Event 连续计时 100 次，三版交错做 3 轮，列出每版的中位轮平均值。输入和输出各占 64 MiB；有效带宽按“读一遍 + 写一遍”的 128 MiB 除以时间估算，不等于实际 DRAM 流量。

| `4096×4096` | 中位 kernel 时间 | 语义有效带宽 | 相对 naive |
| --- | ---: | ---: | ---: |
| copy | `0.6233 ms` | `215.3 GB/s` | `2.64×` |
| naive transpose | `1.6427 ms` | `81.7 GB/s` | `1.00×` |
| shared transpose | `0.6616 ms` | `202.9 GB/s` | `2.48×` |

同一形状的 Nsight Compute 报告在 `.local/profiles/05-transpose/`；每个 kernel 各有 `.nsys-rep` 和 `.ncu-repz`。以下数字来自单次 profiler 运行，不能与上表的 Event 时间混算速度比。

| Kernel | NCU 时间 | DRAM Throughput | L1/TEX Throughput | 实际 occupancy |
| --- | ---: | ---: | ---: | ---: |
| copy | `1.26 ms` | `90.30%` | `17.19%` | `81.03%` |
| naive transpose | `1.72 ms` | `42.64%` | `78.45%` | `80.57%` |
| shared transpose | `0.630 ms` | `91.89%` | `16.87%` | `100.09%` |

NCU 的 Memory Workload Analysis Tables 对 naive 报告：全局 store 的每个 32 字节 sector 平均只利用 4 字节。这与代码中相邻线程跨 `height` 写入一致；shared 通过 tile 改变输出线程分工，消除了这一访问模式。naive 的 L1/TEX 吞吐较高不代表有效带宽高：大量内存事务只写入少量有用字节。shared 的性能接近 copy，且 DRAM 吞吐接近上限。Profiler 会反复重放 kernel，copy 的单次 NCU 时间与稳态 Event 时间差异尤其明显；以同条件 Event 结果比较速度，NCU 用来解释访存。

## 现有 SGEMM 与严格 FP32 cuBLAS

[sgemm_cublas.cu](sgemm_cublas.cu) 直接编译[实验 04 的 naive、tiled、register kernel](../04-register-tiling/sgemm_register.cu)，并用 `cublasGemmEx` 对照。cuBLAS 设置 `CUBLAS_COMPUTE_32F_PEDANTIC` 与 `CUBLAS_PEDANTIC_MATH`，A、B、C 全是 FP32，不使用 TF32。由于 cuBLAS 按列主序解释矩阵，调用时交换 A/B 与 M/N，使相同数组得到行主序 `C=A×B`。四版共用同一组输入、输出形状和 CUDA Event 测量范围；计时只包含 GPU 计算，不包含生成数据、分配、拷贝和正确性检查。每版预热 3 次，每轮连续计时 50 次，四版交错做 3 轮，列出中位轮平均值。小非整齐形状对全部输出与 CPU double 累加参考比较；计时形状对 9 个代表位置与 CPU 比较，并把自写 kernel 的全部输出与 cuBLAS 逐元素比较。容差 `1e-3 + 2e-4×|reference|`；全部通过。

本机补装了 NVIDIA conda 包 `libcublas-dev` / `libcublas` `13.5.1.27` 到 `.local/cuda-13.3`。其他机器若没有 cuBLAS，先安装开发库，再运行：

```powershell
.\scripts\build-windows.ps1 -Lesson SgemmCublas
.\scripts\profile-windows.ps1 -Experiment Sgemm
```

2026-11-14 本机 RTX 3060、`sm_86` 的结果（单位：ms；数值越小越好）：

| `M×N×K` | naive | tiled | register `TM=8` | cuBLAS pedantic FP32 | cuBLAS / register 速度比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1×1024×1024` | `0.04706` | `0.05773` | `0.13748` | `0.02173` | `6.33×` |
| `64×1024×1024` | `0.21131` | `0.16607` | `0.13300` | `0.04383` | `3.03×` |
| `1024×1024×1024` | `2.62214` | `1.99231` | `0.74646` | `0.34259` | `2.18×` |

在 `1024³` 上采集现有三个 kernel，报告在 `.local/profiles/05-sgemm/`。NCU 的单次时间分别为 `3.29 / 2.06 / 0.778 ms`，与 Event 表的稳态时间不直接相减。三版的理论/实际 occupancy 分别是 `100/98.45%`、`100/98.37%`、`66.67/62.48%`；register 更快，虽然 occupancy 更低。它把每线程累加的输出从 1 个增到 8 个，同一 B 值参与更多计算，grid 从 `4096` 个 block 降为 `512` 个；寄存器从每线程 `40`（naive）增到 `54`，每 block 静态 shared memory 为 `9.22 KiB`。这说明复用收益在此形状上盖过了资源占用成本。`M=1` 时，大部分 register kernel 的行工作无效，反而最慢。NCU 的 naive 报告还提示全局 load 每个 32 字节 sector 平均只用 18 字节；这是进一步检查访问模式的线索，不能只凭这条提示把全部性能差距归因于合并访存。

本次 cuBLAS 仍明显快于 register，但这是同形状、同输入/输出精度和同计时范围下的对照；它没有说明 cuBLAS 内部只采用了一种算法。进入[实验 06：Tensor Core 矩阵乘法入门](../06-tensor-core/README.md)时，FP16/BF16 输入与 FP32 累加另作一组，不能与这里严格 FP32 的速度混为一谈。
