# Triton kernels：CUDA 之后的算子练习

这是[学习计划](../LEARNING_PLAN.md)中的第二个项目阶段。先完成 CUDA SGEMM 的 Nsight、cuBLAS 与 Tensor Core 入门，再从 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/)的 Vector Addition、Fused Softmax 和 Matrix Multiplication 开始。

后续在这里实现 Fused RMSNorm、SiLU×Mul；理解融合减少的读写和 kernel launch 后，再尝试 Fused Attention。每个实验保存源码、参考实现、测试形状、dtype、正确性误差、计时方法和解释。真实开始实验时再建立具体子目录，不预先放空练习。

Triton 的运行环境和 GPU 兼容性在进入本阶段时确认；优先复用现有 Windows 设备，必要时转到 WSL / Linux。
