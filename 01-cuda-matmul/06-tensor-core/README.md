# 实验 06：Tensor Core 矩阵乘法入门

实验 05 建立了正确性、计时和 profiler 分析方法。本实验沿用同一套方法，学习 Tensor Core 的矩阵乘加，并弄清输入精度、累加精度和性能数字各代表什么。目前已完成单 tile 正确性、Tensor Core 路径验证，以及一次 FP16 输入量化的单点观察；接下来扩展到多 tile，再进行同精度 cuBLAS 对照。

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

这一步只验证正确性，不计时：一个 block 只有一个 warp，A/B 直接从 global memory 加载，没有 shared memory 复用，性能不具代表性。通过后再做 FP16 输入、FP32 累加的 cuBLAS 对照与计时，之后处理非 16 倍数的形状。

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
