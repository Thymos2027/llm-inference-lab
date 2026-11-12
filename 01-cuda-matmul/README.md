# CUDA 矩阵乘法练习

这一阶段从线程如何计算一个输出元素开始，逐步研究访存和数据复用。

不同方法的实测时间和有效算力统一记录在[性能对比](BENCHMARKS.md)。

| 顺序 | 实验 | 关注点 |
| --- | --- | --- |
| 01 | [Naive FP32 矩阵乘法](01-naive/README.md) | grid、block、线程索引、边界、CPU 参考校验 |
| 02 | [合并访存与线程映射](02-coalesced-access/README.md) | warp 内的访问模式、正确性和计时 |
| 03 | [Shared Memory Tiling](03-shared-memory-tiling/README.md) | block 内的数据复用与同步 |
| 04 | [Register Tiling](04-register-tiling/README.md) | 一个线程累加多个输出、线程内的数据复用 |
| 05 | [Nsight 性能分析](05-nsight-profiling/README.md) | Vector Add 入门、时间线、kernel 指标与瓶颈判断 |

实验 05 暂停继续增加 SGEMM 优化，先用 Vector Add 和 Matrix Transpose 学会 Nsight，再分析现有 SGEMM 并与 cuBLAS 对照。Tensor Core 的 FP16/BF16 矩阵乘法会在这套分析方法建立后继续，并与严格 FP32 的 SGEMM 分开标注精度和性能数据。

每节实验保留自己的源码、输入形状、环境、测量结果和解释。跨实验共用的安装与构建脚本放在根目录 `scripts/`；编译产物和本机 CUDA 环境放在被 Git 忽略的 `.local/` 或对应实验目录中。
