# 学习计划：从 CUDA 基础到 LLM 推理系统

本项目沿着你分享的建议，学习 **PyTorch 推理运行时**、**CUDA SGEMM 与 Tensor Core**，再开展 **Triton 算子**、**Transformers 基础**与 **nano-vLLM** 的练习，之后进入 Attention / MoE 专项和 vLLM Ascend。目标是能独立写对、测准、解释 GPU kernel，也能追踪请求在推理引擎中的完整路径，并用实验判断系统改动的效果。

已有的 `01-cuda-matmul/` 同时承担分享建议里的“CUDA 基础”和“SGEMM”两部分，不为了对齐编号拆散已经完成的课程。CUDA C++、Transformers、nano-vLLM 单卡练习和 Triton 的首个算子均在 Windows 原生环境运行。目录编号沿用原计划；用户已过完阶段 00 与 01；当前推进阶段 03 nano-vLLM，Triton 保留现有进度。

## 怎么使用这份计划

- **仓库状态**表示代码和文档是否已经存在，不代表我已经学会。每节仍要自己完成该节 README 的练习与检查题。
- 按“预测现象 → 写或改代码 → 正确性检查 → 同条件计时 → 解释结果”推进。出现 `PASS` 只说明当前测试通过；性能结论还需要可比较的计时或 profiler 证据。
- 每完成一节，在该节 README 中记录输入形状、运行命令、结果和自己的解释；跨版本数据放在对应的性能记录中。不要只留下截图或单次最快成绩。
- 一次只推进下面的**当前任务**。后续目录先保存阶段目标；具体实验文件在开始做时增加。

## 仓库目录与分享建议的对应关系

```text
llm-inference-lab/
├── LEARNING_PLAN.md               # 本路线与阶段验收
├── 00-pytorch-inference-basics/  # 张量语义、CUDA 运行时、Profiler 与引擎接口
├── 00-transformers-basics/       # tokenizer、模型前向与生成循环的独立先修实验
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

## 当前任务：PyTorch 推理基础实验 01

2026-11-18：为“熟练使用 PyTorch 进行推理引擎开发维护”的求职目标增加 [PyTorch 推理基础](00-pytorch-inference-basics/README.md)。先完成实验 01，能够根据 shape、stride、data pointer 和写入后的联动判断 view 与 copy；再逐节进入 device/dtype、Module 状态、CUDA 计时、显存、Profiler、自定义算子、图捕获和 KV cache。每次只推进一节，并保留一次有意制造和修复的错误。

## 后续任务：nano-vLLM 的 prefill、decode 与 KV block 生命周期

2026-11-17：用户确认已过完目录 00 和 01，现在进入阶段 03。先完成 [第一课：请求到输出](03-nano-vllm/01-request-lifecycle.md)，用已有追踪脚本解释两请求的 prefill、decode、完成与缓存释放。随后依次学习跨 block 分配、前缀复用、调度与 chunked prefill，最后研究性能优化。一次只推进一课。

Transformers 基础与 nano-vLLM 的既有进度如下，继续学习时从对应实验接上：

Transformers 实验 01–05 已有可运行代码和结果：tokenizer、模型输入、logits、生成循环与 KV cache。实验 06 已将 Qwen3-0.6B 第 0 层的 Attention 与 FFN 展开，并与实际层输出核对；下一步亲手逐行解释这些张量及计算，然后再看缓存下的 decode Attention。

1. 已确认单独的 `D:\Coding\nano-vllm` 工作树、现有 Qwen3-0.6B 模型与 Windows `.venv`；单卡 eager 短生成已跑通。具体版本、兼容实现与测试范围见[nano-vLLM 阶段说明](03-nano-vllm/README.md)。
2. 已用 [逐步追踪脚本](03-nano-vllm/trace_steps.py) 走通 `LLMEngine.step` 中的 `Scheduler.schedule → ModelRunner.run → Scheduler.postprocess` 路径，并在[阶段说明](03-nano-vllm/README.md)记录 prefill、decode、请求完成和 KV block 释放。下一步让提示词跨越 block 边界，再检查分配与前缀缓存复用。
3. 已在不改变调度策略的前提下记录每步请求数、安排的 token 数和 KV block 使用情况，短生成检查通过；后续比较同一模型与输入的行为和性能。Windows 注意力替代实现的结果与上游 FlashAttention 路线分开记录。

**本阶段完成标准：**能从请求到输出解释一次 prefill 和 decode，并用可复现日志说明调度与 KV cache 的状态变化。性能比较注明模型版本、输入、并发、GPU 与计时范围。

## 路线与验收

| 阶段 | 仓库状态 | 实验主线 | 进入下一阶段前能交出的结果 |
| --- | --- | --- | --- |
| PyTorch 推理基础 | [实验 01–13](00-pytorch-inference-basics/README.md)已有首版可运行代码，待逐节学习和记录 | storage / stride → Module 状态 → inference mode → CUDA 计时与显存 → mixed precision → batching → Profiler → custom op → compile → KV cache | 能定位 device/dtype/layout 错误，正确测量 GPU，解释缓存分配器与图中断，并用 PyTorch 参考实现验收自定义算子和增量 decode。 |
| 0. C++ / CUDA 最小基础 | 贯穿 `01-cuda-matmul/` | 指针与数组、行主序、host/device、编译与启动、grid/block/thread、边界与报错 | 能在[实验 01](01-cuda-matmul/01-naive/README.md)中手算线程负责的 `C[row,col]`，解释 CPU 参考值与误差检查；哪里不懂就回到这里补。 |
| 1. CUDA 矩阵乘法项目 | 01–05 已有代码、正确性检查、计时和 profiler 记录；实验 06 的 Tensor Core 入门部分已阶段性收束 | Naive → 合并访存 → shared memory → register tiling → 实验 05：Nsight / 同精度 cuBLAS 对照 → 实验 06：Tensor Core 入门 | 能解释[大方阵与 `M=1` 的差异](01-cuda-matmul/BENCHMARKS.md)，并用 profiler 证据解释至少一次优化。实验 06 的非整齐形状与 BF16 尚未完成；重返该专题时再完成完整验收。 |
| 2. Triton kernels 项目 | [Vector Addition](02-triton-kernels/01-vector-add/README.md) 已在 Windows RTX 3060 上通过正确性与计时；Fused Softmax 待开始 | Vector Add → Fused Softmax → Matmul → Fused RMSNorm / SiLU×Mul；再考虑 Fused Attention | 每个算子与可信参考比较，覆盖边界形状；记录输入、dtype、计时范围和速度。至少一个融合实验说明少了哪些读写或 launch，以及实测是否获益。 |
| Transformers 基础先修 | [实验 01–06](00-transformers-basics/README.md)已有代码和运行结果；用户已过完本阶段 | tokenizer → 模型与张量 → 前向与 logits → 生成循环 → KV cache → Attention / FFN | 能解释输入如何变成 token ID，模型每层如何混合上下文与处理每个位置，以及生成循环怎样产出文本。 |
| 3. nano-vLLM 项目 | 已找到单独工作树、现有 Qwen3-0.6B 与 Windows `.venv`；单卡 eager 短生成通过 | 跑通固定版本 → Scheduler / continuous batching → KV cache / block table / prefix cache → CUDA Graph 与算子融合 → chunked prefill | 能从请求到输出追踪一次 prefill 和 decode；留有日志或可视化、正确性检查，并在相同负载下比较 TTFT、TPOT、吞吐和显存。 |
| 4. Attention / MoE 专项 | 待开始 | Decode attention、GQA / KV 复用、Grouped GEMM、offloading；FP8 先分清存储格式与硬件计算能力 | 选一个明确瓶颈做受控实验，记录正确性、负载、硬件和性能变化；不要同时引入多个未验证优化。 |
| 5. vLLM Ascend 迁移 | 待开始 | 对照已有 CUDA / nano-vLLM 经验阅读真实框架的调度、缓存和算子路径，再做 Profiling、Benchmark 与小改动 | 在可用设备上复现基线，对一个具体问题给出可检查的改动或分析报告；记录模型、精度、并发、延迟、吞吐和环境。 |

## 防止跑偏的规则

1. **先正确，再计时，再优化。** 每次改 kernel 都保留 CPU 或可信参考对照；边界形状不能省。
2. **先对照，再归因。** 不把不同精度、不同形状、不同计时范围或不同程序的数字直接当作加速比。
3. **一次只解决一个问题。** 当前先解释 nano-vLLM 第一课的请求生命周期；完成后再推进缓存块与前缀复用。
4. **保留失败实验。** 记录预期、结果和原因；没有收益也能说明方法适用范围。
5. **以阶段验收为推进依据。** 已有源码不等于已经掌握；如果解释不了结果，就先回到该阶段复现和提问。

阶段 1 的统一结果在 [BENCHMARKS.md](01-cuda-matmul/BENCHMARKS.md)。后续阶段开始时，在对应目录增加源码、可复现命令和结果，并同步更新本页的仓库状态与当前任务。Triton 可从[官方教程](https://triton-lang.org/main/getting-started/tutorials/)开始；nano-vLLM 的上游项目见[官方仓库](https://github.com/GeeeekExplorer/nano-vllm)。
