# PyTorch 推理引擎基础

这组实验面向“使用 PyTorch 开发和维护推理引擎”，不按训练教程展开。重点是看懂张量的底层语义、写出可信的正确性检查、理解 CUDA 的异步执行和显存行为，并能把自定义算子、图捕获与 KV cache 串成一个最小运行时。

所有实验使用仓库根目录的 `.venv`：

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe .\00-pytorch-inference-basics\01-tensor-storage.py
```

除实验 11 的算子注册和实验 12 的图捕获外，各脚本只依赖 PyTorch，不依赖外部模型文件。实验 06–08、10、13 在 CUDA 可用时才给出完整的 GPU 结果。

## 学习顺序

### 第一层：张量与模型语义

1. [Tensor、Storage、view 与 stride](01-tensor-storage.py)：判断操作共享存储还是复制，解释非连续张量。
2. [形状、广播与批量矩阵乘法](02-shape-broadcast-matmul.py)：从 Q/K 形状推导 Attention score，并识别 stride 为 0 的广播视图。
3. [device、dtype 与 layout](03-device-dtype-layout.py)：检查设备转移、精度、内存占用与连续布局。
4. [`nn.Module`、buffer、hook 与 `state_dict`](04-module-state-dict.py)：掌握模型维护时常见的状态和观测入口。
5. [`eval`、`no_grad` 与 `inference_mode`](05-inference-mode.py)：避免把关闭 Dropout 和关闭 Autograd 混为一谈。

### 第二层：CUDA 运行时与性能诊断

6. [CUDA 异步执行与正确计时](06-cuda-async-timing.py)：比较 CPU 提交时间、同步时间和 CUDA Event 时间。
7. [CUDA 缓存分配器与显存峰值](07-cuda-memory.py)：观察 allocated、reserved、peak 与张量生命周期。
8. [FP32、FP16 与 BF16](08-mixed-precision.py)：同时记录输入字节数、数值误差和 kernel 时间。
9. [动态批处理、padding 与 mask](09-batching-and-masks.py)：量化 padding 浪费，并验证批处理结果等价于逐请求处理。
10. [PyTorch Profiler](10-profiler.py)：从算子表区分 CPU 调度、矩阵乘法和逐元素操作。

### 第三层：推理引擎接入

11. [自定义算子与 dispatcher](11-custom-operator/README.md)：注册 RMSNorm 算子、提供 FakeTensor 实现，并在 eager 与编译图中核对参考结果。这一课先学接口契约；后续再把实现替换为 C++/CUDA kernel。
12. [`torch.compile` 与 graph break](12-compile-and-graph-breaks.py)：用 `fullgraph=True` 暴露 `.item()` 引起的图中断，再改成纯张量控制流。
13. [最小推理运行时与 KV cache](13-mini-inference-runtime.py)：实现 causal self-attention，对比完整重算和增量 decode，核对生成 token 与送入模型的位置数。

## 每节实验的验收方式

运行脚本只是第一步。每节至少回答四个问题：

1. 哪些张量共享存储，哪些操作发生复制？
2. 正确性参考是什么，断言覆盖了哪些边界？
3. 计时是否包含同步、数据传输、预热或编译？
4. 如果输入换成非连续张量、不同 dtype 或不同批量，结论是否仍成立？

完成一节后，把本机输出、自己的解释和一次有意制造并修复的错误补到本页或对应实验说明。`PASS` 只代表当前断言通过，不代表已经掌握。

## 与其他目录的关系

- 本目录先学习 PyTorch 运行时与引擎接口。
- [`00-transformers-basics`](../00-transformers-basics/README.md)继续负责 tokenizer、真实模型前向、生成循环和 Qwen 层结构。
- [`01-cuda-matmul`](../01-cuda-matmul/README.md)负责 CUDA C++ kernel 与底层性能优化。
- [`02-triton-kernels`](../02-triton-kernels/README.md)负责把可信的 PyTorch 参考实现替换为 Triton 算子。
- [`03-nano-vllm`](../03-nano-vllm/README.md)负责真实推理引擎的调度、KV block 和请求生命周期。
