# Triton kernels：CUDA 之后的算子练习

这是[学习计划](../LEARNING_PLAN.md)中的第二个项目阶段。CUDA SGEMM 的 Nsight、cuBLAS 与 Tensor Core 入门已阶段性收束，现在从 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)的 Vector Addition、Fused Softmax 和 Matrix Multiplication 开始。

| 顺序 | 实验 | 状态与重点 |
| --- | --- | --- |
| 01 | [Vector Addition](01-vector-add/README.md) | 已在 Windows RTX 3060 上跑通；program、块内偏移、边界 mask、正确性与 CUDA Graph 计时 |
| 02 | Fused Softmax | 下一步：行内归约、数值稳定性和不同长度的边界处理 |
| 03 | Matrix Multiplication | 待开始：tile、矩阵布局与同精度参考比较 |

后续在这里实现 Fused RMSNorm、SiLU×Mul；理解融合减少的读写和 kernel launch 后，再尝试 Fused Attention。每个实验保存源码、参考实现、测试形状、dtype、正确性误差、计时方法和解释。真实开始实验时再建立具体子目录，不预先放空练习。

本机 Windows 环境已通过第一个 Triton kernel 的编译、运行和边界形状检查。使用社区维护的 `triton-windows`；具体版本、安装命令和实测结果见[实验 01](01-vector-add/README.md)。后续算子仍逐个验证，遇到 Windows 特有的兼容问题时再考虑 WSL / Linux。
