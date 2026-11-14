# nano-vLLM：推理引擎源码与改造

这是[学习计划](../LEARNING_PLAN.md)中的第三个项目阶段。上游为 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)。开始时先固定上游提交、记录依赖和运行环境，在可用的 NVIDIA GPU 上运行原版示例与基准；模型下载先核对版本、量化格式和配套文件，按项目的下载规则选择来源。

按以下顺序做可观察的小改动：

1. **Scheduler / continuous batching**：记录每个调度 step 的请求、token budget、prefill 与 decode。
2. **KV cache / block table / prefix cache**：跟踪物理块的分配、复用与释放，并检查引用关系。
3. **CUDA Graph / 算子融合**：在同一负载下比较 eager、Graph 和融合算子的延迟与正确性。
4. **Chunked prefill**：改变 chunk size 与并发，测 TTFT、TPOT、吞吐和尾延迟。

本目录保存自己的实验代码、日志说明、结果与上游改动记录。不要把不明版本的上游源码直接复制进来。每次只改一个因素，先确认输出正确，再比较同一模型、同一输入和同一硬件上的性能。
