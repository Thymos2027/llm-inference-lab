# 实验 01：CUDA 线程与矩阵乘法

## 这次要理解什么

设 `A` 的形状是 `M×K`，`B` 是 `K×N`，输出 `C` 是 `M×N`：

```text
C[row, col] = sum(A[row, k] * B[k, col])，k 从 0 到 K-1
```

本实验让**一个 CUDA 线程计算 C 的一个元素**。先区分两个层级：

- **grid** 是这次 Kernel 启动的所有 block 组成的网格。
- **block** 是一组线程；一个 block 中的线程共享这个 block 的编号 `blockIdx`，每个线程有自己的组内编号 `threadIdx`。

这里把 C 按 `16×16` 的小块划分：`block(16,16)` 表示每个 block 横向有 16 个线程、纵向有 16 个线程。`grid` 横向需要足够多的 block 覆盖 C 的 `N` 列，纵向需要覆盖 C 的 `M` 行。

下面两行计算的是**当前线程负责的 C 元素的行号和列号**，不是线程在硬件上的物理位置：

```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

这里约定 CUDA 的 `x` 方向对应矩阵的**列**（横向），`y` 方向对应矩阵的**行**（纵向）。代码先写 `row` 还是先写 `col` 都可以；两者都确定之后，才用它们计算 `C[row,col]`。

例如，暂时想象 C 只有 3 行、5 列，每个 block 有 `2×2` 个线程，就需要横向 3 个、纵向 2 个 block，即 `grid(3,2)`。其中 `blockIdx=(1,0)`、`threadIdx=(0,1)` 的线程负责：

```text
row = 0×2+1 = 1
col = 1×2+0 = 2
→ C[1,2] = sum(A[1,k] × B[k,2])
```

同一个 block 内的其他线程负责相邻的 C 元素。最边缘一些线程会落在 C 的范围外，由边界判断直接退出。

矩阵按行连续存储，因此 `A[row,k]` 对应 `A[row*K+k]`，`B[k,col]` 对应 `B[k*N+col]`。`row` 或 `col` 超出输出尺寸的线程必须跳过，避免越界访问。

## 在 Windows 上运行

本机已检测到 RTX 5060 Ti 和 Visual Studio 2022 Build Tools。项目使用位于 `.local/cuda-13.3` 的 CUDA 13.3 Conda 环境，避免修改系统 CUDA 配置。安装完成后，在 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-windows.ps1
```

脚本会先调用 Visual Studio 的 C++ 编译环境，再使用 `nvcc` 编译并运行程序。

若想在 Windows 任务管理器里观察 GPU 活动，打开「性能 → GPU」，把其中一张图切换到 `Compute_0`（如果列表中有 `CUDA`，也可以选它），然后运行：

```powershell
.\scripts\build-windows.ps1 -Observe
```

观察模式使用 `512×512×512` 矩阵，连续发射短 Kernel 约 10 秒。正常模式的 `37×53×29` 计算太快，任务管理器通常采样不到。观察模式结束后仍会进行 CPU 参考结果校验；这个模式用于看硬件活动，不用于性能比较。

## 在 Linux / WSL 上运行

在安装了 CUDA Toolkit 的 Ubuntu / WSL 终端中执行：

```bash
cd /mnt/e/Coder/LLM/llm-inference-lab/01-cuda-basics
nvcc -O2 -std=c++17 -arch=sm_120 sgemm_naive.cu -o sgemm_naive
./sgemm_naive
```

运行后应看到 `PASS` 和最大绝对误差。程序用固定种子的随机数生成输入，并以 CPU 计算作为参考。当前尺寸 `M=37, N=53, K=29` 故意不能被 `16×16` 的 block 整除，用来检验边界处理。

`nvidia-smi` 显示的 CUDA Version 是驱动支持的版本，不表示已安装 `nvcc`。编译前可用 `nvcc --version` 检查 Toolkit 是否可用。

## 亲手改三个地方

1. 从 `sgemm_naive.cu` 找到 `row` 和 `col` 的计算。手算 `blockIdx=(2,1)`、`threadIdx=(3,5)` 对应 `C` 的哪个元素。
2. 将 `M, N, K` 改为 `4, 5, 3`，再改为 `128, 129, 33`。两次都应输出 `PASS`。
3. 手算边缘线程：当 `blockIdx.x=3`、`threadIdx.x=5` 时，`col=53`，已经超出最后一列 `52`。据此解释边界判断的作用。不要运行去掉判断的版本；越界线程可能覆盖其他元素或访问矩阵范围外的内存。

完成标准：能够独立解释线程坐标、三处矩阵下标，以及为什么随机输入和非整齐尺寸比常量方阵更容易发现错误。
