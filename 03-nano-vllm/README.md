# nano-vLLM：推理引擎源码与改造

这是[学习计划](../LEARNING_PLAN.md)中的第三个项目阶段，目前优先于 Triton 推进。上游为 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。本机已有单独的 `D:\Coding\nano-vllm` 工作树，当前先使用其中的 Windows 单卡兼容实现学习推理流程。

## 起点：2026-11-15 本机检查

- `D:\Coding\nano-vllm` 的 HEAD 为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（上游项目版本 `0.2.0`），`origin` 指向用户自己的 fork。Windows 兼容改动目前在该仓库的未提交工作树中；本页的运行结果对应这份工作树，不能仅凭上游提交哈希复现。
- 模型位于 `C:\Users\gigappb\huggingface\Qwen3-0.6B\`，已有 `config.json`、tokenizer 文件和约 1.5 GB 的 `model.safetensors`；配置标明 `Qwen3ForCausalLM`、BF16。`example.py` 使用的 `~/huggingface/Qwen3-0.6B/` 在本机指向这份文件，无需再次下载。
- 本机 RTX 3060 8 GB（`sm_86`）；该仓库已有 Windows `.venv`，其中 PyTorch `2.14.1+cu130` 检测到 CUDA，Transformers 为 `5.23.0`。Windows 单卡路径用 Gloo、PyTorch SDPA 注意力和未编译的 PyTorch 算子替代上游的 NCCL、FlashAttention 与 Triton；它适合先学习调度流程，不可直接用于上游性能对照。
- `python -m unittest discover -s tests -v` 的三项 Windows 注意力检查通过。使用现有模型、`enforce_eager=True`、`tensor_parallel_size=1`、`max_model_len=max_num_batched_tokens=512`、`max_tokens=8` 的短生成也成功输出 8 个 token。尚未运行完整的 `example.py` 或基准。

## 首个可观察任务

从已有的 Windows 单卡短生成继续，源码阅读路线为 `example.py → nanovllm/llm.py → engine/llm_engine.py`：`generate()` 添加请求并反复调用 `step()`；每步由 `Scheduler.schedule()` 选择 prefill 或 decode，`ModelRunner` 执行计算，再由 `Scheduler.postprocess()` 更新序列和 KV block 状态。下一步先记录每步的请求、token 与 KV block 变化，不改调度策略。若要比较性能，先固定现有 Windows 兼容代码的版本与负载，并单独标明注意力实现不同。

按以下顺序做可观察的小改动：

1. **Scheduler / continuous batching**：记录每个调度 step 的请求、token budget、prefill 与 decode。
2. **KV cache / block table / prefix cache**：跟踪物理块的分配、复用与释放，并检查引用关系。
3. **CUDA Graph / 算子融合**：在同一负载下比较 eager、Graph 和融合算子的延迟与正确性。
4. **Chunked prefill**：改变 chunk size 与并发，测 TTFT、TPOT、吞吐和尾延迟。

本目录保存自己的实验代码、日志说明、结果与上游改动记录。不要把不明版本的上游源码直接复制进来。每次只改一个因素，先确认输出正确，再比较同一模型、同一输入和同一硬件上的性能。
