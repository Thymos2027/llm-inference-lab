# 实验 01：Triton Vector Addition

这是 Triton 阶段的第一个算子。给定两个连续的 FP32 向量 `x`、`y`，计算 `out[i] = x[i] + y[i]`。先理解 Triton 的 program、块内偏移和边界 mask，再和同精度的 PyTorch `torch.add` 对照正确性与 GPU 执行时间。源码见 [vector_add.py](vector_add.py)。

## 运行环境

2026-11-16 在 WIN-GG 的 Windows 原生环境验证：RTX 3060 8 GB（`sm_86`）、驱动 621.24、Python 3.12.16、PyTorch `2.9.1+cu128`、`triton-windows 3.5.1.post24`（`import triton` 显示 3.5.1）、NumPy 2.5.5。这里使用社区维护的 Windows Triton 构建版；Triton 官方列出的受支持平台是 Linux。此组合在本机跑通了本实验，不代表后续全部 Triton 教程都已通过。

项目根目录下的 `.local/triton-win/` 是被 Git 忽略的本机环境。若已存在，可直接运行：

```powershell
& .\.local\triton-win\Scripts\python.exe .\02-triton-kernels\01-vector-add\vector_add.py
& .\.local\triton-win\Scripts\python.exe .\02-triton-kernels\01-vector-add\vector_add.py --benchmark
```

在另一台 Windows 机器上复现时，可从项目根目录创建隔离环境（需要 NVIDIA CUDA 驱动与 `uv`）：

```powershell
uv venv --python 3.12 .local/triton-win
uv pip install --python .local/triton-win/Scripts/python.exe --index https://download.pytorch.org/whl/cu128 'torch==2.9.1+cu128'
uv pip install --python .local/triton-win/Scripts/python.exe 'triton-windows==3.5.1.post24' numpy
```

若下载直连失败，先检查本机代理是否运行，再仅为当前终端设置代理环境变量；不需要修改系统全局代理。

## Kernel 怎样分工

`block=256` 表示一个 Triton program 负责 256 个连续元素。`tl.program_id(0)` 选中第几个 program；`tl.arange(0, block)` 生成这个 program 内的偏移。启动的 program 数为 `ceil(n / block)`。例如 `n=1003` 时有 4 个 program，最后一个只处理 235 个有效元素，剩余 21 个位置由 `offsets < n` 的 mask 屏蔽。

`tl.load` 分别读 `x`、`y`，`tl.store` 写 `out`。边界 mask 同时用于读取和写入，避免最后一个 program 越界。PyTorch 参考用同样的输入、FP32 dtype 和独立的输出缓冲区执行 `torch.add(x, y, out=reference)`。Triton 输出先填 NaN，以便发现漏写位置。

## 正确性与计时方法

- 固定随机种子 2026；检查长度 `1, 255, 256, 257, 1003, 4096, 1048576`，覆盖单元素、块边界及非整齐长度。
- 每个输出元素与 PyTorch 参考逐元素比较，本实验只做一次 FP32 加法，因此要求 `rtol=0, atol=0`；同时打印最大绝对误差。
- `--benchmark` 先对每个计时长度再次做全量正确性检查，再测 `1003`、`1048576`、`16777339`。两个方法都写入已分配的输出缓冲区，计时不包括随机数据生成、分配或检查。
- 计时使用 `triton.testing.do_bench_cudagraph`：预热并估算时长，捕获重复调用的 CUDA Graph，再用 CUDA Event 测 10 次 graph replay，报告单次 kernel 的中位时间。PyTorch 与 Triton 使用同一计时方法；数值包含 graph 中的 GPU 工作，不包含 Python 逐次 launch 开销。
- 按程序语义，每个元素读取两个 FP32、写入一个 FP32，估算搬运量为 `12 × n` 字节。表中的有效带宽由此除以 Triton 时间得到，不是 profiler 测出的 DRAM 流量；缓存会影响解释。

2026-11-16 在上述 RTX 3060 环境、`block=256` 上运行，全部正确性检查 `PASS`，最大绝对误差均为 0。一次运行的计时结果如下；每个单元格是 10 次 replay 的中位数，不是单次最快值：

| 长度 `n` | PyTorch `torch.add` | Triton | PyTorch / Triton | Triton 有效带宽 |
| ---: | ---: | ---: | ---: | ---: |
| 1,003 | 0.00123 ms | 0.00086 ms | 1.42× | 13.9 GB/s |
| 1,048,576 | 0.06100 ms | 0.05914 ms | 1.03× | 212.8 GB/s |
| 16,777,339 | 0.95027 ms | 0.93677 ms | 1.01× | 214.9 GB/s |

大数组上两者接近：PyTorch 的加法本身已经是单个 GPU kernel，Triton 版本没有减少读写或 launch。短向量的时间不到 2 微秒，比例对时钟、缓存和计时条件敏感，不能从这一次结果推断 Triton 普遍快 42%。后面的融合实验会比较真正减少中间结果读写与 launch 的算子。

## 亲手检查

1. 手算 `n=1003, block=256` 时各 program 的起始偏移、最后一个 program 有效元素数，以及为什么读写两处都需要 mask。
2. 运行 `--block 128` 和 `--block 512` 的正确性检查。再各跑一次 `--benchmark`，记录大向量的变化；若差别很小，先考虑计时波动，不急着下优化结论。
3. 对照 [CUDA Vector Add](../../01-cuda-matmul/05-nsight-profiling/README.md)，说明 CUDA 线程索引与 Triton program/偏移的对应关系，以及为什么每元素都至少涉及两次读取和一次写入。

完成标准：能独立解释 `program_id`、`arange`、grid 和 mask；上述边界长度全部通过；能说清计时范围、参考实现和大数组上两种单 kernel 加法接近的原因。
