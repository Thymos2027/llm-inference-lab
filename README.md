# LLM 推理系统练习

这里按「一个概念、一个实验、一个可检查的结果」学习 GPU 编程与推理系统。

## 目前的成果

- 从 naive、合并访存对照、shared memory tiling 到 register tiling，逐步实现并比较 FP32 CUDA 矩阵乘法 kernel；用非整齐尺寸和 CPU 参考结果检查正确性。
- 在 RTX 5060 Ti 上，对 `1024×1024×1024` 矩阵乘法进行同程序配对测量：register tiling 的中位 kernel 时间为 `0.402 ms`，相对 tiled 版本的 `1.076 ms` 为 `2.68×`。测量只包含 kernel，不包含数据准备、传输或校验；[完整方法和结果](01-cuda-matmul/BENCHMARKS.md)。
- 记录不同形状的限制：在 `1×1024×1024` 上，同一 register tiling 版本为 `0.060 ms`，慢于 tiled 版本的 `0.025 ms`。这组实验目前聚焦 FP32 矩阵乘法，尚不是完整 LLM 推理服务的吞吐测试。
- 使用 CUDA Event 计时，并通过 Nsight Compute、Nsight Systems 和 SASS 检查性能现象；[TM 参数与指令分析](01-cuda-matmul/04-register-tiling/TM_ANALYSIS.md)。

## 目录与学习顺序

目前的矩阵乘法练习统一放在 [01-cuda-matmul](01-cuda-matmul/README.md)，每节实验各自保存源码和说明：

各版本的实测数据见 [FP32 矩阵乘法性能记录](01-cuda-matmul/BENCHMARKS.md)。

```text
01-cuda-matmul/
├── 01-naive/             # 线程映射、边界与正确性
├── 02-coalesced-access/ # 访存模式与性能对照
├── 03-shared-memory-tiling/ # block 内的数据复用
├── 04-register-tiling/ # 每线程多个输出与寄存器累加
└── 05-nsight-profiling/ # 用 Vector Add 学习 Nsight 性能分析
scripts/                  # 共用的 Windows 安装与构建脚本
```

先做 [实验 01：CUDA 线程与最简单的矩阵乘法](01-cuda-matmul/01-naive/README.md)。它只解决三个问题：

1. CUDA 的 grid、block、thread 如何对应到输出矩阵。
2. 如何写出正确的矩阵乘法 kernel。
3. 如何用随机数据和非整齐尺寸检验结果。

完成后继续 [实验 02：合并访存与线程映射](01-cuda-matmul/02-coalesced-access/README.md)。它用相同的矩阵乘法和相同的计算量，比较相邻线程负责相邻列或相邻行的效果。接着做 [实验 03：Shared Memory Tiling](01-cuda-matmul/03-shared-memory-tiling/README.md)，研究一个 block 怎样合作保存和复用 A、B 的数据。然后做 [实验 04：Register Tiling](01-cuda-matmul/04-register-tiling/README.md)，研究一个线程怎样复用数据计算多个输出。完成这些 kernel 后，用 [实验 05：Nsight 性能分析](01-cuda-matmul/05-nsight-profiling/README.md) 建立“计时 → 采集 → 解释瓶颈”的方法。

## 本机工具与 `.local` 目录

以下 Windows 环境已在两台机器上核对：RTX 5060 Ti（2026-11-11）与 RTX 3060（2026-11-12）。`.local/` 被 Git 忽略，保存工具环境、安装包缓存、编译产物和性能分析报告；克隆仓库后不会自动拥有这些文件。CUDA 工具在 `.local` 中，Nsight 的实际可执行文件则位于本机其他目录。

| 工具 | 本机版本与位置 | 在项目中的用途 |
| --- | --- | --- |
| NVIDIA GPU 与驱动 | ① GeForce RTX 5060 Ti，驱动 610.88，compute capability 12.0（`sm_120`）；② GeForce RTX 3060，驱动 621.14，compute capability 8.6（`sm_86`） | 运行 CUDA kernel；[build-windows.ps1](scripts/build-windows.ps1) 通过 `nvidia-smi` 自动检测本机架构，也可用 `-Arch sm_86` / `-Arch sm_120` 显式指定。 |
| CUDA 编译器与运行库 | CUDA 13.3；`nvcc 13.3.73`，位于 `.local/cuda-13.3/Library/bin/` | 编译 `.cu`，提供 CUDA Runtime 和头文件；环境由 [setup-windows-cuda.ps1](scripts/setup-windows-cuda.ps1) 配置。[build-windows.ps1](scripts/build-windows.ps1) 会从该目录寻找 `nvcc.exe`。 |
| `cuobjdump`、`nvdisasm` | 分别为 13.3.73、13.3.29；均位于 `.local/cuda-13.3/Library/bin/` | 查看编译后的 SASS 机器指令，例如实验 04 中的 `LDS.64` 与 `LDS.128`。 |
| Nsight Compute (`ncu`) | 2026.4.0；可执行文件路径见 `.local/profilers/nsight-compute-path.txt` | 采集单个 kernel 的指令、occupancy、等待等指标，报告格式为 `.ncu-repz`。 |
| Nsight Systems (`nsys`) | 5060 Ti 机 2026.6.1、3060 机 2026.5.2；可执行文件路径见 `.local/profilers/nsight-systems-path.txt` | 查看 CUDA 调用与 kernel 执行时间线，报告格式为 `.nsys-rep`。 |
| aria2 (`aria2c`) | 1.37.0；位于 `.local/profilers/aria2/` | 下载 Nsight 安装包时支持断点续传；不参与 kernel 编译或计时。 |
| Visual Studio 2022 Build Tools | 安装在系统目录，由 `vswhere.exe` 定位 | 提供 Windows C++ 宿主编译器；构建脚本通过 `vcvars64.bat` 配置环境。 |
| Conda | 安装在系统目录；本项目环境位于 `.local/cuda-13.3/` | 安装 CUDA 编译器、运行库和反汇编工具；`.local/conda-pkgs/` 是安装包缓存。 |
| VS Code C/C++ 扩展 | `ms-vscode.cpptools` 1.35.6，安装在 VS Code 扩展目录 | 提供 CUDA C++ 源码的编辑器导航与 Outline；不影响命令行编译。 |

Nsight Compute 和 Nsight Systems 的程序在 5060 Ti 机上分别解压在 `E:\N`、`E:\S`，在 3060 机上经 conda 安装在 `D:\nsight`，都不是 `.local` 中的程序。`.local/profilers/` 保存路径文件、验证结果及示例报告；其中 `validation.json` 记录了 `ncu` 的 GPU 采样和 `nsys` 的 CUDA trace 均已通过。注意 `ncu` 访问 GPU 性能计数器默认需要管理员权限（报 `ERR_NVGPUCTRPERM`）：可在 NVIDIA App「系统 → 高级 → 开发者 → 管理 GPU 性能计数器」允许所有用户访问，或从管理员终端运行 `profile-windows.ps1`；`nsys` 普通权限即可。

其他实验目录：`.local/tm-sweep/` 保存 `TM` 参数扫描产物，`.local/tm-profile/` 保存 Nsight Compute 与 SASS 分析，`.local/tm-deep/` 保存 K 循环展开的受控计时 CSV 和报告。正文结论见[实验 04 的 TM 分析](01-cuda-matmul/04-register-tiling/TM_ANALYSIS.md)；这些本机原始文件不会随仓库同步。

在项目根目录可以检查版本、状态并运行课程：

```powershell
& .\.local\cuda-13.3\Library\bin\nvcc.exe --version
& .\.local\cuda-13.3\Library\bin\cuobjdump.exe --version
$ncu = (Get-Content .\.local\profilers\nsight-compute-path.txt -Raw).Trim()
$nsys = (Get-Content .\.local\profilers\nsight-systems-path.txt -Raw).Trim()
& $ncu --version
& $nsys --version
Get-Content .\.local\profilers\validation.json
.\scripts\build-windows.ps1 -Lesson Register
```

性能记录使用程序内的 CUDA Event 计时；`ncu`、`nsys` 用来解释瓶颈。Profiler 采样可能改变运行时长，比较各版速度时以相同条件下的 CUDA Event 结果为准。
