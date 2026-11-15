# 实验 06：Tensor Core 矩阵乘法入门

实验 05 建立了正确性、计时和 profiler 分析方法。本实验沿用同一套方法，学习 Tensor Core 的矩阵乘加，并弄清输入精度、累加精度和性能数字各代表什么。2026-11-15 已完成单 tile 正确性、Tensor Core 路径验证、FP16 输入量化观察、多 tile 与 K 方向循环，以及同精度 cuBLAS 对照与计时。本次作为入门阶段收束，后续转入 nano-vLLM。

## 单 tile 练习文件

- [wmma_practice.cu](wmma_practice.cu)：已由学习者填完五处 WMMA TODO；在原始输入中把 `B[1,2]` 改成 `1.0001f` 来观察 FP16 量化。
- [wmma_reference.cu](wmma_reference.cu)：保持原始整数输入的完整参考实现，可对照 WMMA 调用。

两份代码都用一个 warp 计算 `16×16×16`，A、B 为行主序 FP16，C 为行主序 FP32。A 是对角矩阵 `A[i,i]=i+1`；参考版的 B 为 `B[k,j]=16k+j`，练习版只有 `B[1,2]` 的原始值改为 `1.0001f`。练习版的 CPU 检查以转换后的 FP16 输入为参考。

```powershell
.\scripts\build-windows.ps1 -Lesson TensorCorePractice -BuildOnly
.\scripts\build-windows.ps1 -Lesson TensorCorePractice
.\scripts\build-windows.ps1 -Lesson TensorCoreReference
```

2026-11-15 本机 RTX 3060（`sm_86`）上，练习版最初的整数输入得到 `C[1,2]=36.0`、`C[2,1]=99.0`、`PASS (0 mismatches)`。Nsight Compute 的 `smsp__inst_executed_pipe_tensor.sum=2`，确认执行了 Tensor Core 管线指令；该值不是 FLOP 数。将 `B[1,2]` 的原始 FP32 值改为 `1.0001f` 后，程序打印其 FP16 值为 `1.0000000`，`C[1,2]=2.0`、`C[2,1]=99.0`、`PASS (0 mismatches)`。该位置以原始输入计算为约 `2.0002`，以量化输入计算为 `2.0`，输入量化造成约 `0.0002` 的差；当前例子只覆盖一个改动的输入位置，尚未测量整块矩阵的误差分布。

## 多 tile 与 K 方向循环

单 tile 版只算一个 `16×16×16`。真实矩阵更大，要回答两个新问题：**C 被切成很多 16×16 小块后，谁负责哪一块？**以及**K 大于 16 时，同一块 C 怎样把多段乘积累加起来？**

- [wmma_multitile_practice.cu](wmma_multitile_practice.cu)：kernel 里留了三处 TODO——由 `blockIdx` 求出负责的 C tile、写出 K 方向循环与 A/B tile 的起始地址和 leading dimension、写回 C tile。未填写前 C 保持 NaN，程序报 `FAIL`。
- [wmma_multitile_reference.cu](wmma_multitile_reference.cu)：完整参考实现，先自己做练习版再对照。

线程分工：**一个 block 只有一个 warp，负责一个 16×16 的 C tile**；launch 为 `grid(N/16, M/16)`、`block(32)`。warp 沿 K 每次前进 16，加载 `A[tile_row*16 .., k0 ..]` 与 `B[k0 .., tile_col*16 ..]`，把乘积累加进同一个 FP32 accumulator fragment，循环结束后一次写回。和单 tile 版最大的区别是 leading dimension：A 一行有 `K` 个元素、B 和 C 一行有 `N` 个元素，不再是 `16`。

```powershell
.\scripts\build-windows.ps1 -Lesson TensorCoreMultiPractice
.\scripts\build-windows.ps1 -Lesson TensorCoreMultiReference
```

默认检查三种形状：`16×16×16`（与单 tile 版同尺寸）、`64×48×32`（`M≠N≠K`，行列下标写反或 leading dimension 写错会暴露）、`256×512×128`（多 tile、8 个 K 步）。也可以直接运行 exe 并传入 `M N K` 测其他形状，本步只接受 16 的整数倍。

正确性检查分开报告两种误差：

1. **kernel 误差**：与"先转换成 FP16 的输入、double 累加"的 CPU 参考比较，决定 `PASS/FAIL`。阈值为 `1e-5 + 1e-4 × Σ|a·b|`，随累加项的大小放缩。
2. **FP16 输入量化误差**：同一套 FP16 输入的参考值与原始 FP32 输入的参考值之差，只打印、不判定，用来区分"输入精度损失"与"kernel 算错"。

做练习前先在纸上回答：`M=64, N=48, K=32` 时一共几个 C tile？负责 C 第 32–47 行、第 16–31 列的是哪个 block？它在 `k0=16` 时 A tile 和 B tile 的起始下标是多少？

这一步只验证正确性，不计时：一个 block 只有一个 warp，A/B 直接从 global memory 加载，没有 shared memory 复用，性能不具代表性。

2026-11-15 本机练习版三处 TODO 已由学习者填完，三种默认形状均 `PASS`。

## 同精度 cuBLAS 对照与计时

[wmma_cublas.cu](wmma_cublas.cu) 直接编译[多 tile 参考版](wmma_multitile_reference.cu)的 `wmma_multi_tile` kernel（定义 `WMMA_KERNELS_ONLY` 跳过其 `main`），与 `cublasGemmEx` 对照。两边精度一致：A、B 为 `CUDA_R_16F` 行主序，C 为 `CUDA_R_32F` 行主序，cuBLAS 用 `CUBLAS_COMPUTE_32F`（FP32 累加）与默认 math mode，允许使用 Tensor Core。cuBLAS 按列主序解释矩阵，调用时交换 A/B 与 M/N。

```powershell
.\scripts\build-windows.ps1 -Lesson TensorCoreCublas
```

测量口径沿用实验 05：

- 输入为 `[-1,1]` 均匀分布，先转换成 FP16 再上传；两种方法共用同一组设备数组。
- **正确性**：小形状 `64×48×32` 检查全部输出，不计时；计时形状检查 9 个代表位置。参考值以 FP16 输入、double 累加计算，阈值 `1e-5 + 1e-4 × Σ|a·b|`。另外逐元素比较 wmma 与 cuBLAS 的全部输出，阈值 `1e-5 × K`。计算前把 C 填成 NaN，以发现漏写的输出。
- **计时**：只计 GPU 计算，不含数据生成、分配、拷贝和检查。每种方法预热 3 次、每轮连续 launch 50 次并用 CUDA Event 计时，两种方法交错跑 3 轮，取中位轮的单次平均。cuBLAS 若拆成多个 kernel（见下文的 split-K），计时包含全部 kernel。

2026-11-15 本机 RTX 3060（`sm_86`）结果，全部形状 `PASS`（单位 ms，越小越好）：

| `M×N×K` | wmma（1 warp / block） | cuBLAS FP16→FP32 | cuBLAS / wmma 速度比 | cuBLAS GFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| `16×1024×1024` | `0.02312` | `0.02929` | `0.79×` | `1146` |
| `64×1024×1024` | `0.02800` | `0.02877` | `0.97×` | `4665` |
| `1024×1024×1024` | `0.38648` | `0.12621` | `3.06×` | `17015` |
| `2048×2048×2048` | `2.74733` | `0.74938` | `3.67×` | `22925` |
| `4096×4096×4096` | `75.99187` | `5.94121` | `12.79×` | `23133` |

同日重跑一次，速度比依次为 `0.78× / 0.98× / 3.20× / 3.68× / 12.94×`，结论不变，单次运行之间相差约 5% 以内。

wmma 在 `1024³` 到 `2048³` 约为 `5.6`–`6.3` TFLOP/s，到 `4096³` 骤降到 `1.8` TFLOP/s。按 28 个 SM、每 SM 每周期 256 次 FP16→FP32 FMA、boost 频率 1777 MHz 估算，本卡 FP16 输入、FP32 累加的稠密峰值约 `25.5` TFLOP/s（按白皮书口径推算，非实测）；cuBLAS 在大方阵上约达其 90%。

### Nsight Compute 观察

用 `wmma_cublas.exe --profile wmma|cublas M N K` 单次 launch（零输入），以 `ncu --metrics` 采集。以下为单次 profiler 运行的数字，不与上表的 Event 时间混算：

| 形状 / 方法 | kernel | tensor 指令 | tensor 管线占用 | DRAM 读 | L2 命中率 |
| --- | --- | ---: | ---: | ---: | ---: |
| `1024³` wmma | `wmma_multi_tile` | `524,288` | `11.2%` | `23.3 MB` | `87.7%` |
| `1024³` cuBLAS | `ampere_s1688gemm_fp16_128x128…` | `1,048,576` | `45.2%` | `8.6 MB` | `82.3%` |
| `4096³` wmma | `wmma_multi_tile` | `33,554,432` | `3.75%` | `8.86 GB` | `31.4%` |
| `4096³` cuBLAS | `cutlass_80_tensorop_s16816gemm_f16_256x128…` | `33,554,432` | `46.8%` | `1.09 GB` | `42.0%` |
| `16×1024×1024` cuBLAS | `ampere_s16816gemm…` + `splitKreduce_kernel` | `65,536` | `33.7%` | `2.2 MB` | `24.1%` |

指标分别为 `smsp__inst_executed_pipe_tensor.sum`、`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active`、`dram__bytes_read.sum`、`lts__t_sector_hit_rate.pct`。

- **两边都走了 Tensor Core。**wmma 每个 `16×16×16` 执行 2 条 tensor 指令：`1024³` 有 `64³=262,144` 个，共 `524,288` 条。cuBLAS 在 `1024³` 选了 `m16n8k8`（`s1688`）kernel，指令数翻倍；在 `4096³` 选了 `m16n8k16`（`s16816`）kernel，指令数与 wmma 相同。指令数相同而时间差约 10 倍，说明差距不在算术本身，而在喂数据。
- **wmma 慢在没有数据复用。**每个 block 只算一个 `16×16` 的 C tile，要从 global memory 读入 A 的整条 `16×K` 行带和 B 的整条 `K×16` 列带，相邻 block 读的同一段数据只能靠 L2 共享。`1024³` 时 A、B 共 4 MB，L2 命中率达 88%。`4096³` 时 A、B 共 64 MB，远超 L2，命中率掉到 31%，DRAM 读取 8.86 GB，是 A、B 本身大小的约 140 倍，tensor 管线只有 3.75% 时间在工作。cuBLAS 用 `128×128` / `256×128` 的大 block tile，先放进 shared memory，让每个读入的元素参与更多次乘加，DRAM 读取只有 1.09 GB。
- **小 M 时 wmma 反而更快。**`M=16` 只有一行 C tile，wmma 的 grid 为 64 个 block；cuBLAS 选了 split-K：主 kernel 把 K 切成两半并行（grid `16×1×2`），再由 `splitKreduce_kernel` 合并，Event 时间包含两个 kernel 和额外的 launch 开销。这只说明 cuBLAS 在该形状上的启发式选择不占优，不说明 wmma 写法更好；`M=64` 时两者已基本持平。
- 在 `1024³` 及以上，wmma 与 cuBLAS 的全部输出逐元素完全相同（max abs diff 为 0），推测两者在 K 方向的 FP32 累加顺序一致；`M≤64` 时差约 `1e-4`，在阈值内，其中 `M=16` 可确认是 split-K 改变了累加顺序（`M=64` 未单独 profile）。

跨精度参考（**不是同精度加速比**）：实验 05 中严格 FP32 cuBLAS 在 `1024³` 为 `0.34259` ms，本页 FP16 输入、FP32 累加的 cuBLAS 为 `0.12621` ms。前者不使用 Tensor Core，输入精度也不同，这只说明换用 FP16 输入后能用上 Tensor Core 的工作负载取舍。

## 阶段小结与留待后续的问题

- 一个 warp 负责一个 `16×16` 输出 tile，沿 K 每 16 个元素累加；多 tile 练习的三种默认形状均通过全量正确性检查，Nsight 确认使用 Tensor Core 管线。
- 用转换后的 FP16 输入计算参考值，可以把输入量化误差与 kernel 计算误差分开；同精度 cuBLAS 对照也使用相同输入、输出与计时范围。
- 大方阵上，自写 WMMA 的主要限制是 A/B 数据缺少跨 warp 的 shared memory 复用。`4096³` 的 DRAM 读取和 tensor 管线占用与 cuBLAS 的差异支持这一解释；小 M 时还要考虑 cuBLAS 的 split-K 策略与 launch 开销。
- 本阶段只覆盖 M、N、K 均为 16 的倍数的 FP16 路线。非整齐形状、BF16 与多 warp 数据复用尚未实现，因此下方的完整完成标准仍有未勾选项。需要继续 Tensor Core 专题时，从非整齐形状的补零与全量正确性检查开始。

## 实验顺序

1. 确认当前 GPU 的计算能力、CUDA 版本，以及可用的 Tensor Core 数据类型和编程接口。先选 FP16 输入、FP32 累加；可从 CUDA WMMA 的矩阵乘加接口入手。
2. 从一个满足 tile 尺寸要求的小方阵开始，写出最小矩阵乘法，明确 A、B、C 的布局、leading dimension、tile 尺寸和边界条件。用能检查的输入确认结果与矩阵方向。
3. 将原始 FP32 输入**先转换成 FP16**，再用转换后的 A、B 计算 FP32 参考值。比较 Tensor Core 输出与该参考值，同时记录 FP16 量化相对于原始 FP32 输入带来的误差，避免把两种误差混在一起。
4. 在相同输入、精度、输出形状和计时范围下与 cuBLAS 对照；预热后用 CUDA Event 多次计时，记录中位数。与前面严格 FP32 SGEMM 的结果分开列出，跨精度比较只作为明确标注的工作负载取舍，不称为同精度加速比。
5. 再处理不能整除 tile 的形状，选择补零、边界 tile 或受控回退，并验证结果。FP16 路线成立后，再按当前硬件支持情况尝试 BF16。
6. 用 Nsight Compute 检查实际生成的矩阵乘加指令、计算与访存指标；结合 tile 利用率和数据搬运解释为什么某些形状快、某些形状慢。

## 完成标准

- 至少一个整齐形状和一个非整齐形状通过正确性检查，误差阈值注明输入转换与累加精度。
- 留下可复现的构建和运行命令、GPU / CUDA 环境、输入形状、dtype、计时范围及多次测量结果。
- 能指出 Tensor Core 路径确实被使用，并解释与同精度 cuBLAS 的性能差异及 FP16 量化误差。

实验 05 的 transpose、现有 SGEMM 分析和严格 FP32 cuBLAS 对照见[实验 05](../05-nsight-profiling/README.md)。
