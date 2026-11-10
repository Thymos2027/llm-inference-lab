# LLM 推理系统练习

这里按「一个概念、一个实验、一个可检查的结果」学习 GPU 编程与推理系统。

## 目录与学习顺序

目前的矩阵乘法练习统一放在 [01-cuda-matmul](01-cuda-matmul/README.md)，每节实验各自保存源码和说明：

各版本的实测数据见 [FP32 矩阵乘法性能记录](01-cuda-matmul/BENCHMARKS.md)。

```text
01-cuda-matmul/
├── 01-naive/             # 线程映射、边界与正确性
├── 02-coalesced-access/ # 访存模式与性能对照
└── 03-shared-memory-tiling/ # block 内的数据复用
scripts/                  # 共用的 Windows 安装与构建脚本
```

先做 [实验 01：CUDA 线程与最简单的矩阵乘法](01-cuda-matmul/01-naive/README.md)。它只解决三个问题：

1. CUDA 的 grid、block、thread 如何对应到输出矩阵。
2. 如何写出正确的矩阵乘法 kernel。
3. 如何用随机数据和非整齐尺寸检验结果。

完成后继续 [实验 02：合并访存与线程映射](01-cuda-matmul/02-coalesced-access/README.md)。它用相同的矩阵乘法和相同的计算量，比较相邻线程负责相邻列或相邻行的效果。接着做 [实验 03：Shared Memory Tiling](01-cuda-matmul/03-shared-memory-tiling/README.md)，研究一个 block 怎样合作保存和复用 A、B 的数据。
