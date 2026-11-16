# nano-vLLM：推理引擎源码与改造

这是[学习计划](../LEARNING_PLAN.md)中的第三个项目阶段，目前优先于 Triton 推进。上游为 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。本机在单独的 `D:\Coding\nano-vllm` 工作树中修改和运行引擎；本目录记录学习过程。

## 两个仓库的分工

| 位置 | 用途 |
| --- | --- |
| `llm-inference-lab/03-nano-vllm/` | 学习路线、实验负载、运行记录、分析与结论。记录所用的 nano-vLLM 提交哈希，不保存引擎源码副本。 |
| [`Thymos2027/nano-vllm`](https://github.com/Thymos2027/nano-vllm)（本机 `D:\Coding\nano-vllm`） | 引擎源码、Windows 兼容实现、依赖锁定、测试和上游 PR 分支；是代码的唯一维护位置。 |
| `~/huggingface/Qwen3-0.6B/` | 模型权重与 tokenizer，保存在 Git 仓库之外。 |

修改引擎时在 nano-vLLM 仓库提交；做完实验后在本目录记录负载、结果和该提交。两边分别维护 Git 历史，不把引擎源码纳入学习仓库的 Git 跟踪。`llm-inference-lab/.local/nano-vllm-pr/` 只是为上游 PR 创建的、被忽略的临时 Git worktree，属于同一个 nano-vLLM 仓库。

## 本机状态：截至 2026-11-16

- `D:\Coding\nano-vllm` 的 `main` 为已推送的 `8db1d75`，基于上游 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（项目版本 `0.2.0`）。Windows 兼容代码也已提交为[上游 PR #300](https://github.com/GeeeekExplorer/nano-vllm/pull/300)；PR 为保持精简，没有包含 fork `main` 中的 `.python-version` 和 `uv.lock`。
- 模型位于 `C:\Users\gigappb\huggingface\Qwen3-0.6B\`，已有 `config.json`、tokenizer 文件和约 1.5 GB 的 `model.safetensors`；配置标明 `Qwen3ForCausalLM`、BF16。`example.py` 使用的 `~/huggingface/Qwen3-0.6B/` 在本机指向这份文件，无需再次下载。
- 本机 RTX 3060 8 GB（`sm_86`）；该仓库已有 Windows `.venv`，其中 PyTorch `2.14.1+cu130` 检测到 CUDA，Transformers 为 `5.23.0`。Windows 单卡路径用 Gloo、PyTorch SDPA 注意力和未编译的 PyTorch 算子替代上游的 NCCL、FlashAttention 与 Triton；它适合先学习调度流程，不可直接用于上游性能对照。
- `python -m unittest discover -s tests -v` 的四项 Windows 注意力检查通过。使用现有模型、`enforce_eager=True`、`tensor_parallel_size=1`、`max_model_len=max_num_batched_tokens=512`、`max_tokens=8` 的两请求短生成也成功输出，每个请求 8 个 token。尚未运行完整的 `example.py` 或基准。

## 在另一台 Windows 设备复现

以下命令在 **PowerShell** 中执行。需要 Windows x64、NVIDIA GPU 和支持 CUDA 13.0 的驱动；本机验证配置为 RTX 3060 8 GB。先安装 [Git](https://git-scm.com/download/win) 与 [uv](https://docs.astral.sh/uv/getting-started/installation/)，确认 `git --version`、`uv --version` 和 `nvidia-smi` 均可运行。准备本仓库 `llm-inference-lab` 的工作副本，从其根目录执行：

```powershell
$lab = (Resolve-Path .).Path
$nano = Join-Path (Split-Path $lab -Parent) 'nano-vllm'
git clone https://github.com/Thymos2027/nano-vllm.git $nano
git -C $nano checkout --detach 8db1d75134717bdee43b52d1ad33cd0d9898a7f4
Set-Location $nano
uv sync --locked --python 3.12
```

该提交包含 `.python-version`、`pyproject.toml` 和 `uv.lock`：Windows 使用官方 PyTorch CUDA 13.0 包，跳过 FlashAttention / Triton 依赖；单卡分布式初始化用 Gloo 并关闭 libuv rendezvous，注意力改由 PyTorch SDPA 计算。`uv sync` 会在新设备下载所需软件包并创建 `.venv`；本机已有环境，无须重建或重复下载。仓库不包含模型权重。

把现有 Qwen3-0.6B 模型**整目录**复制到新设备的 `$HOME\huggingface\Qwen3-0.6B\`，包括 `model.safetensors`、`config.json` 和 tokenizer 文件。若模型放在别处，下面的短生成通过 `$modelPath` 指向它，无须修改源码。先检查 GPU 包与注意力实现：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

预期 CUDA 检查输出 `True`，四项注意力检查均为 `ok`。然后运行短生成（将模型路径改成新设备上的实际位置）：

```powershell
$modelPath = Join-Path $HOME 'huggingface\Qwen3-0.6B'
$env:NANOVLLM_MODEL = (Resolve-Path -LiteralPath $modelPath).Path
@'
import os
from nanovllm import LLM, SamplingParams

llm = LLM(os.environ["NANOVLLM_MODEL"], enforce_eager=True,
          tensor_parallel_size=1, max_model_len=512,
          max_num_batched_tokens=512, max_num_seqs=8,
          gpu_memory_utilization=0.8)
print(llm.generate(["Hello, Nano-vLLM."],
                   SamplingParams(temperature=0.6, max_tokens=8),
                   use_tqdm=False))
'@ | .\.venv\Scripts\python.exe -
```

成功时会打印一个含 `text` 和 `token_ids` 的结果，`token_ids` 最多 8 个；随机采样的文字不必逐字相同。若出现显存不足，先关闭占用 GPU 的程序，再降低 `gpu_memory_utilization` 或 `max_model_len` 重新试。本路线目前只验证单卡 eager 短生成；Windows SDPA 路线与上游 FlashAttention 路线实现不同，不把两者的速度直接比较。

性能差异主要来自实现：Windows 路线按请求调用 SDPA、组装因果 mask、从分页 KV cache 收集 token，并跳过 `torch.compile`；Linux 原版使用 FlashAttention、Triton KV 写入和编译后的逐元素算子。Windows 路线在长上下文或高并发下可能产生更多同步与 kernel launch，但尚无同一 GPU、模型、负载及计时口径下的 Windows/Linux 实测倍率。单卡 Gloo 与 NCCL 的差别不是当前主要计算路径。

## 首个可观察任务

### 2026-11-16：单卡逐步追踪

在 WIN-GG 的 RTX 3060 8 GB 上，使用 nano-vLLM 提交 `8db1d75134717bdee43b52d1ad33cd0d9898a7f4`、本机 Qwen3-0.6B（BF16）与 Windows SDPA 路线，先确认两请求 `generate()` 短生成成功，再运行 [trace_steps.py](trace_steps.py)。脚本沿着 `LLMEngine.step()` 的相同顺序调用 `Scheduler.schedule()`、`ModelRunner.run()`、`Scheduler.postprocess()`，只读取并打印状态，不修改调度策略。两条原始提示词分别是 `The capital of France is` 与 `A GPU is a`，新 token 上限分别为 4 和 6，`temperature=0.6`、`ignore_eos=True`。配置为单卡 eager、`max_model_len=max_num_batched_tokens=512`、`max_num_seqs=8`、`gpu_memory_utilization=0.8`，随机种子为 0。

从本仓库根目录运行（可用 `--model` 指定其他模型目录）：

```powershell
& D:\Coding\nano-vllm\.venv\Scripts\python.exe .\03-nano-vllm\trace_steps.py
```

本次实际运行通过末尾的 token 数与 KV block 释放检查，关键状态如下；采样文字可能随环境变化：

| step | 阶段 | 安排的 token | 运行中请求 | 已用 KV block | 观察 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | prefill | 9（提示词 5+4） | 2 | 2 | 两个提示词一次进入模型，各产生第 1 个新 token。 |
| 2–3 | decode | 每步 2 | 2 | 2 | 每条请求每步处理上一步产生的 token，并采样下一个。 |
| 4 | decode | 2 | 1 | 1 | 请求 1 达到 4 个新 token，状态变 `FINISHED`，其 block 释放。 |
| 5–6 | decode | 每步 1 | 最后为 0 | 最后为 0 | 请求 2 达到 6 个新 token，block 全部释放。 |

本机该配置分配了 151 个 256-token 的 KV block；这是本次显存状态下的值。日志中 `cached` 是已经写入 KV cache 的 token 数：prefill 后它等于提示词长度，刚采样出的 token 要到下一次 decode 才写入。请求完成时 `deallocate()` 把 `cached` 归零并清空 `block_table`。这里验证了请求、调度和 block 生命周期；没有做性能计时，也没有覆盖跨 block 的分配或前缀缓存复用。

源码阅读路线为 `example.py → nanovllm/llm.py → engine/llm_engine.py`：`generate()` 添加请求并反复调用 `step()`；每步由 `Scheduler.schedule()` 选择 prefill 或 decode，`ModelRunner` 执行计算，再由 `Scheduler.postprocess()` 更新序列和 KV block 状态。下一步可让提示词跨越 256-token block 边界，观察新 block 分配与前缀缓存复用。若要比较性能，先固定现有 Windows 兼容代码的版本与负载，并单独标明注意力实现不同。

按以下顺序做可观察的小改动：

1. **Scheduler / continuous batching**：记录每个调度 step 的请求、token budget、prefill 与 decode。
2. **KV cache / block table / prefix cache**：跟踪物理块的分配、复用与释放，并检查引用关系。
3. **CUDA Graph / 算子融合**：在同一负载下比较 eager、Graph 和融合算子的延迟与正确性。
4. **Chunked prefill**：改变 chunk size 与并发，测 TTFT、TPOT、吞吐和尾延迟。

本目录保存实验脚本、日志说明、结果和对应的引擎提交哈希；引擎源码改动留在独立的 nano-vLLM 仓库。每次只改一个因素，先确认输出正确，再比较同一模型、同一输入和同一硬件上的性能。
