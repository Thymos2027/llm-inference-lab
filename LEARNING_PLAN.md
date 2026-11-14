# 学习计划：从 CUDA 基础到 LLM 推理系统

本项目沿着你分享的建议，先完成 **CUDA SGEMM 和 Triton 算子**，再以 **nano-vLLM** 为第二个完整项目学习推理引擎，最后进入 Attention / MoE 专项和 vLLM Ascend。目标是能独立写对、测准、解释 GPU kernel，也能追踪请求在推理引擎中的完整路径，并用实验判断系统改动的效果。

已有的 `01-cuda-matmul/` 同时承担分享建议里的“CUDA 基础”和“SGEMM”两部分，不为了对齐编号拆散已经完成的课程。默认先在 Windows 做 CUDA C++ 实验；Triton、nano-vLLM 的实际运行环境在开始该阶段时按其依赖确认，需要时再用 WSL / Linux。

## 怎么使用这份计划

- **仓库状态**表示代码和文档是否已经存在，不代表我已经学会。每节仍要自己完成该节 README 的练习与检查题。
- 按“预测现象 → 写或改代码 → 正确性检查 → 同条件计时 → 解释结果”推进。出现 `PASS` 只说明当前测试通过；性能结论还需要可比较的计时或 profiler 证据。
- 每完成一节，在该节 README 中记录输入形状、运行命令、结果和自己的解释；跨版本数据放在对应的性能记录中。不要只留下截图或单次最快成绩。
- 一次只推进下面的**当前任务**。后续目录先保存阶段目标；具体实验文件在开始做时增加。

## 仓库目录与分享建议的对应关系

```text
llm-inference-lab/
├── LEARNING_PLAN.md               # 本路线与阶段验收
├── 01-cuda-matmul/                # CUDA 基础、FP32 SGEMM 与 Tensor Core 入门
│   ├── 01-naive/
│   ├── 02-coalesced-access/
│   ├── 03-shared-memory-tiling/
│   ├── 04-register-tiling/
│   ├── 05-nsight-profiling/
│   └── 06-tensor-core/
├── 02-triton-kernels/             # Triton 算子与融合实验
├── 03-nano-vllm/                  # nano-vLLM 源码学习和受控改造
└── scripts/                       # 项目共用脚本
```

原建议把 CUDA 基础、SGEMM 分成两个顶层目录，因此 Triton 和 nano-vLLM 分别编号为 03、04；这里把前两者合并，编号相应前移。原建议的 `05-benchmarks/` 与 `notes/` 暂不另建：现有 [SGEMM 性能记录](01-cuda-matmul/BENCHMARKS.md)和每节 README 已在保存结果与解释。跨项目比较真正出现时，再建立根目录 `benchmarks/`，并注明统一的测试负载与环境。nano-vLLM 上游源码以固定版本单独获取，本仓库的 `03-nano-vllm/` 保存学习记录、实验代码和改动说明，不直接混入未注明版本的上游快照。

## 当前任务：完成实验 05 的性能分析方法

目前 01–04 的 CUDA 矩阵乘法代码、正确性检查和性能记录已经在仓库中；实验 05 已有 Vector Add 程序、练习文件和第一次 Nsight 采集结果。下一步是把这些结果变成自己能解释的知识，然后再做 transpose。

1. 运行 `./scripts/build-windows.ps1 -Lesson NsightPractice`，确认 [Vector Add 练习](01-cuda-matmul/05-nsight-profiling/vector_add_practice.cu)的线程编号、边界判断和输出。源码里的 TODO 注释仍需对照自己的实现检查。
2. 阅读[实验 05](01-cuda-matmul/05-nsight-profiling/README.md)，自己回答末尾四道检查题；能说明“每元素约 12 字节”是代码语义估算，以及为什么 profiler 时间不必等于 benchmark 时间。
3. 打开已有的 Nsight Systems / Nsight Compute 报告，或按实验 05 的命令重新采集；指出 kernel 时间、grid/block、Memory Throughput、Compute Throughput 和 occupancy 分别在哪里。重新采集若耗时较长，按后台任务方式运行。
4. 接着实现矩阵复制、naive transpose 和 shared memory transpose，比较合并访存、正确性和计时；再分析现有 SGEMM，最后与 cuBLAS 做相同形状、相同精度、相同计时范围的对照。
5. 完成实验 05 后进入[实验 06：Tensor Core 入门](01-cuda-matmul/06-tensor-core/README.md)：先确认当前 GPU 与 CUDA 工具链支持的精度和接口，再用 FP16 输入、FP32 累加实现最小矩阵乘法，对照相同数据类型与计算模式的 cuBLAS，以及由相同量化输入计算的 FP32 参考值。之后处理非整齐形状、尝试 BF16 并用 profiler 分析，完成后进入 `02-triton-kernels/`。

**本阶段完成标准：**能用计时和 profiler 数据解释一个 kernel 为什么慢，以及一次改动为什么快或慢；结论注明测试形状与测量范围。

## 路线与验收

| 阶段 | 仓库状态 | 实验主线 | 进入下一阶段前能交出的结果 |
| --- | --- | --- | --- |
| 0. C++ / CUDA 最小基础 | 贯穿 `01-cuda-matmul/` | 指针与数组、行主序、host/device、编译与启动、grid/block/thread、边界与报错 | 能在[实验 01](01-cuda-matmul/01-naive/README.md)中手算线程负责的 `C[row,col]`，解释 CPU 参考值与误差检查；哪里不懂就回到这里补。 |
| 1. CUDA 矩阵乘法项目 | 01–04 已有代码与记录；实验 05 的 Vector Add 已有代码和首次结果；实验 06 已有阶段说明 | Naive → 合并访存 → shared memory → register tiling → 实验 05：Nsight / 同精度 cuBLAS 对照 → 实验 06：Tensor Core 入门 | 非整齐形状通过检查；能解释[大方阵与 `M=1` 的差异](01-cuda-matmul/BENCHMARKS.md)，并用 profiler 证据解释至少一次优化。实验 06 用相同输入、dtype、形状和计时范围对照 Tensor Core 与 cuBLAS；FP16/BF16 输入、FP32 累加与严格 FP32 SGEMM 分开记录精度和速度。无需追求 cuBLAS 的某个固定百分比。 |
| 2. Triton kernels 项目 | [阶段说明已建](02-triton-kernels/README.md)，实验待开始 | Vector Add → Fused Softmax → Matmul → Fused RMSNorm / SiLU×Mul；再考虑 Fused Attention | 每个算子与可信参考比较，覆盖边界形状；记录输入、dtype、计时范围和速度。至少一个融合实验说明少了哪些读写或 launch，以及实测是否获益。 |
| 3. nano-vLLM 项目 | [阶段说明已建](03-nano-vllm/README.md)，源码实验待开始 | 跑通固定版本 → Scheduler / continuous batching → KV cache / block table / prefix cache → CUDA Graph 与算子融合 → chunked prefill | 能从请求到输出追踪一次 prefill 和 decode；留有日志或可视化、正确性检查，并在相同负载下比较 TTFT、TPOT、吞吐和显存。模型文件下载前确认版本和配套文件。 |
| 4. Attention / MoE 专项 | 待开始 | Decode attention、GQA / KV 复用、Grouped GEMM、offloading；FP8 先分清存储格式与硬件计算能力 | 选一个明确瓶颈做受控实验，记录正确性、负载、硬件和性能变化；不要同时引入多个未验证优化。 |
| 5. vLLM Ascend 迁移 | 待开始 | 对照已有 CUDA / nano-vLLM 经验阅读真实框架的调度、缓存和算子路径，再做 Profiling、Benchmark 与小改动 | 在可用设备上复现基线，对一个具体问题给出可检查的改动或分析报告；记录模型、精度、并发、延迟、吞吐和环境。 |

## 防止跑偏的规则

1. **先正确，再计时，再优化。** 每次改 kernel 都保留 CPU 或可信参考对照；边界形状不能省。
2. **先对照，再归因。** 不把不同精度、不同形状、不同计时范围或不同程序的数字直接当作加速比。
3. **一次只解决一个问题。** 先完成实验 05 的 transpose 和 profiler 分析，再进入实验 06；Tensor Core 对照时不要同时改线程映射、精度和计时方法。
4. **保留失败实验。** 记录预期、结果和原因；没有收益也能说明方法适用范围。
5. **以阶段验收为推进依据。** 已有源码不等于已经掌握；如果解释不了结果，就先回到该阶段复现和提问。

阶段 1 的统一结果在 [BENCHMARKS.md](01-cuda-matmul/BENCHMARKS.md)。后续阶段开始时，在对应目录增加源码、可复现命令和结果，并同步更新本页的仓库状态与当前任务。Triton 可从[官方教程](https://triton-lang.org/main/getting-started/tutorials/)开始；nano-vLLM 的上游项目见[官方仓库](https://github.com/GeeeekExplorer/nano-vllm)。
