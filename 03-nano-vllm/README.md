# nano-vLLM：推理引擎源码与改造

这是[学习计划](../LEARNING_PLAN.md)中的第三个项目阶段，目前优先于 Triton 推进。上游为 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。本机已有单独的 `D:\Coding\nano-vllm` 工作树，当前先使用其中的 Windows 单卡兼容实现学习推理流程。

## 起点：2026-11-15 本机检查

- `D:\Coding\nano-vllm` 的 HEAD 为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（上游项目版本 `0.2.0`），`origin` 指向用户自己的 fork。Windows 兼容改动目前在该仓库的未提交工作树中；已将其完整改动保存为[windows-native.patch](windows-native.patch)，换机时使用“上游提交 + 补丁”复现。
- 模型位于 `C:\Users\gigappb\huggingface\Qwen3-0.6B\`，已有 `config.json`、tokenizer 文件和约 1.5 GB 的 `model.safetensors`；配置标明 `Qwen3ForCausalLM`、BF16。`example.py` 使用的 `~/huggingface/Qwen3-0.6B/` 在本机指向这份文件，无需再次下载。
- 本机 RTX 3060 8 GB（`sm_86`）；该仓库已有 Windows `.venv`，其中 PyTorch `2.14.1+cu130` 检测到 CUDA，Transformers 为 `5.23.0`。Windows 单卡路径用 Gloo、PyTorch SDPA 注意力和未编译的 PyTorch 算子替代上游的 NCCL、FlashAttention 与 Triton；它适合先学习调度流程，不可直接用于上游性能对照。
- `python -m unittest discover -s tests -v` 的三项 Windows 注意力检查通过。使用现有模型、`enforce_eager=True`、`tensor_parallel_size=1`、`max_model_len=max_num_batched_tokens=512`、`max_tokens=8` 的短生成也成功输出 8 个 token。尚未运行完整的 `example.py` 或基准。

## 在另一台 Windows 设备复现

以下命令在 **PowerShell** 中执行。需要 Windows x64、NVIDIA GPU 和支持 CUDA 13.0 的驱动；本机验证配置为 RTX 3060 8 GB。先安装 [Git](https://git-scm.com/download/win) 与 [uv](https://docs.astral.sh/uv/getting-started/installation/)，确认 `git --version`、`uv --version` 和 `nvidia-smi` 均可运行。准备本仓库 `llm-inference-lab` 的工作副本，从其根目录执行：

```powershell
$lab = (Resolve-Path .).Path
$nano = Join-Path (Split-Path $lab -Parent) 'nano-vllm'
$patch = Join-Path $lab '03-nano-vllm\windows-native.patch'
git clone https://github.com/GeeeekExplorer/nano-vllm.git $nano
git -C $nano checkout --detach bb823b3e06983d71485a8e1f23715ebd87d98ef8
git -C $nano apply --check $patch
git -C $nano apply $patch
Set-Location $nano
uv sync --locked --python 3.12
```

补丁包含 `.python-version`、`pyproject.toml` 和 `uv.lock`：Windows 使用官方 PyTorch CUDA 13.0 包，跳过 FlashAttention / Triton 依赖；单卡分布式初始化用 Gloo，注意力改由 PyTorch SDPA 计算。`uv sync` 会在新设备下载所需软件包并创建 `.venv`；本机已有环境，无须重建或重复下载。补丁仅适用于上面的上游提交，且不包含模型权重。

把现有 Qwen3-0.6B 模型**整目录**复制到新设备的 `$HOME\huggingface\Qwen3-0.6B\`，包括 `model.safetensors`、`config.json` 和 tokenizer 文件。若模型放在别处，下面的短生成通过 `$modelPath` 指向它，无须修改源码。先检查 GPU 包与注意力实现：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

预期 CUDA 检查输出 `True`，三项注意力检查均为 `ok`。然后运行短生成（将模型路径改成新设备上的实际位置）：

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

## 首个可观察任务

从已有的 Windows 单卡短生成继续，源码阅读路线为 `example.py → nanovllm/llm.py → engine/llm_engine.py`：`generate()` 添加请求并反复调用 `step()`；每步由 `Scheduler.schedule()` 选择 prefill 或 decode，`ModelRunner` 执行计算，再由 `Scheduler.postprocess()` 更新序列和 KV block 状态。下一步先记录每步的请求、token 与 KV block 变化，不改调度策略。若要比较性能，先固定现有 Windows 兼容代码的版本与负载，并单独标明注意力实现不同。

按以下顺序做可观察的小改动：

1. **Scheduler / continuous batching**：记录每个调度 step 的请求、token budget、prefill 与 decode。
2. **KV cache / block table / prefix cache**：跟踪物理块的分配、复用与释放，并检查引用关系。
3. **CUDA Graph / 算子融合**：在同一负载下比较 eager、Graph 和融合算子的延迟与正确性。
4. **Chunked prefill**：改变 chunk size 与并发，测 TTFT、TPOT、吞吐和尾延迟。

本目录保存自己的实验代码、日志说明、结果与上游改动记录。不要把不明版本的上游源码直接复制进来。每次只改一个因素，先确认输出正确，再比较同一模型、同一输入和同一硬件上的性能。
