# CUDA 矩阵乘法练习

这一阶段从线程如何计算一个输出元素开始，逐步研究访存和数据复用。

| 顺序 | 实验 | 关注点 |
| --- | --- | --- |
| 01 | [Naive FP32 矩阵乘法](01-naive/README.md) | grid、block、线程索引、边界、CPU 参考校验 |
| 02 | [合并访存与线程映射](02-coalesced-access/README.md) | warp 内的访问模式、正确性和计时 |

后续再加入 Shared Memory Tiling、寄存器分块等实验。Tensor Core 的 FP16/BF16 矩阵乘法属于这一阶段，但会与严格 FP32 的 SGEMM 分开标注精度和性能数据。

每节实验保留自己的源码、输入形状、环境、测量结果和解释。跨实验共用的安装与构建脚本放在根目录 `scripts/`；编译产物和本机 CUDA 环境放在被 Git 忽略的 `.local/` 或对应实验目录中。
