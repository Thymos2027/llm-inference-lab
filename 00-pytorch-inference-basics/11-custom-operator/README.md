# 实验 11：自定义算子与 PyTorch dispatcher

本实验先学习推理引擎接入自定义 kernel 时必须满足的接口契约：schema、device/dtype 行为、参考实现、FakeTensor/编译图支持和错误输入。当前实现仍由 PyTorch 算子组成，不把它误称为高性能融合 kernel；后续可在不改变调用方的前提下，把实现替换为 C++/CUDA 或 Triton。

```powershell
& .\.venv\Scripts\python.exe .\00-pytorch-inference-basics\11-custom-operator\rms_norm_op.py
```

脚本注册 `inference_lab::rms_norm`，用独立参考表达式检查 eager 输出，再通过 `torch.compile(..., backend="eager", fullgraph=True)` 验证 dispatcher 和 FakeTensor 契约能够被完整图捕获。

验收问题：

1. 为什么算子 schema 需要明确 Tensor 和标量参数？
2. FakeTensor 实现为什么不能读取真实数据？
3. 如果替换为 CUDA kernel，哪些 dtype、shape、stride 与 device 检查必须保留？
4. “注册成功”为什么不代表比普通 PyTorch 表达式更快？
