# nano-vLLM：推理引擎源码与改造

这是[学习计划](../LEARNING_PLAN.md)中的第三个项目阶段，目前优先于 Triton 推进。上游为 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。先固定上游提交、记录依赖和运行环境，在可用的 NVIDIA GPU 上运行原版示例与基准。

## 起点：2026-11-15 本机检查

- 上游源码固定在提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`（项目版本 `0.2.0`），只读学习副本放在被 Git 忽略的 `.local/nano-vllm-upstream/`。复现时以提交哈希为准，不依赖该本机副本。
- Windows 本机模型位于 `C:\Users\gigappb\huggingface\Qwen3-0.6B\`，已有 `config.json`、tokenizer 文件和约 1.5 GB 的 `model.safetensors`；配置标明 `Qwen3ForCausalLM`、BF16。无需再次下载模型。上游 `example.py` 默认查找 Linux 风格的 `~/huggingface/Qwen3-0.6B/`，在实际运行环境中需指向这份现有文件。
- 本机为 RTX 3060 8 GB（`sm_86`），Windows Python 为 3.12.16。当前未安装 `torch`、`triton`、`transformers`、`flash-attn`、`xxhash`，也没有可用的 WSL 发行版；尚未运行 nano-vLLM。上游 `pyproject.toml` 要求 Python 3.10–3.12 及上述依赖。先确定支持 CUDA 与这些依赖的 Linux 运行环境，再做原版基线。

## 首个可观察任务

先用 `enforce_eager=True`、`tensor_parallel_size=1` 跑通上游 `example.py` 的 Qwen3-0.6B 示例，记录环境、输入和输出。源码阅读路线为 `example.py → nanovllm/llm.py → engine/llm_engine.py`：`generate()` 添加请求并反复调用 `step()`；每步由 `Scheduler.schedule()` 选择 prefill 或 decode，`ModelRunner` 执行计算，再由 `Scheduler.postprocess()` 更新序列和 KV block 状态。运行基线通过后，先只加观察日志，不改调度策略。

按以下顺序做可观察的小改动：

1. **Scheduler / continuous batching**：记录每个调度 step 的请求、token budget、prefill 与 decode。
2. **KV cache / block table / prefix cache**：跟踪物理块的分配、复用与释放，并检查引用关系。
3. **CUDA Graph / 算子融合**：在同一负载下比较 eager、Graph 和融合算子的延迟与正确性。
4. **Chunked prefill**：改变 chunk size 与并发，测 TTFT、TPOT、吞吐和尾延迟。

本目录保存自己的实验代码、日志说明、结果与上游改动记录。不要把不明版本的上游源码直接复制进来。每次只改一个因素，先确认输出正确，再比较同一模型、同一输入和同一硬件上的性能。
