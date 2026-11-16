# Transformers 基础：从输入到生成

这是独立于 nano-vLLM 源码的先修实验。先弄清 Hugging Face Transformers 的 tokenizer、模型输入输出和生成循环，再回到推理引擎。实验代码只放在本目录；`D:\Coding\nano-vllm` 不做改动。

## 学习顺序

1. **文字与 token**：运行本页的实验 01，理解 token ID、解码和聊天模板。
2. **模型与张量**：读 `config.json`，认识 `input_ids`、形状、设备和数据类型。
3. **一次前向计算**：看 `logits` 的形状，理解“预测下一个 token”的含义。
4. **生成循环**：亲手写一个最小的“前向计算 → 选 token → 追加输入”循环，再与 `generate()` 对照。
5. **KV cache**：比较有无缓存时的输入和计算，再回到 nano-vLLM 的 block table。

每次只推进一个实验。当前是 **实验 01：tokenizer**。

## 根目录 PyTorch 环境检查

仓库根目录的 `pyproject.toml` 固定了 Windows CUDA 13.0 版 PyTorch。先在根目录运行以下命令，确认 PyTorch 能找到 GPU，并完成一个小矩阵乘法：

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe .\00-transformers-basics\check_torch.py
```

这只是环境检查。后续学习模型前向时会继续使用同一个根目录 `.venv`。

## 实验 01：文字如何进入模型

需要本机已有的 Qwen3-0.6B tokenizer 文件。学习仓库根目录使用统一的 `.venv`，由根目录的 `pyproject.toml` 和 `uv.lock` 固定依赖；不会修改 nano-vLLM 仓库。此实验只读取 tokenizer 文件，不加载模型权重或占用 GPU。

在本仓库根目录执行：

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe .\00-transformers-basics\01-tokenizer.py
```

模型放在别处时可加 `--model <模型目录>`。脚本做两件事：

- 把 `Hello, world!` 编成整数 ID，再解码回原文。`piece` 是 tokenizer 为人显示的 token 片段，不是模型直接接收的文字。
- 把 `[{"role": "user", "content": "What is a GPU?"}]` 套入 Qwen3 的聊天模板。`add_generation_prompt=True` 会在末尾放入 assistant 的起始标记，让模型接着写回答。

**读代码时先回答：**`tokenizer.encode()` 的返回值是什么类型？模型接收的是原文、`piece`，还是 token ID？为什么聊天输入比问题文字本身多出一些标记？

本机 2026-11-16 使用 Transformers 5.23.0 运行通过：普通文本得到 4 个 token ID `[9707, 11, 1879, 0]`，解码回 `Hello, world!`；其中显示为 `\u0120world` 的片段带有空格标记。聊天模板生成 17 个 token ID，包含用户和助手角色标记。本版本的 `apply_chat_template(..., tokenize=True)` 返回带 `input_ids` 和 `attention_mask` 的对象；这里的 mask 全为 1，表示这 17 个位置都是真实输入。具体 ID 与模板内容取决于 tokenizer 版本及模型文件。

## 用 VS Code 断点调试

用 VS Code 打开 `D:\Coding\llm-inference-lab` 文件夹，打开 [01-tokenizer.py](01-tokenizer.py)，在 `chat_text = ...` 这一行的左侧行号旁单击设置断点。按 **F5**，选择 **Transformers 01: tokenizer**；本仓库的 `.vscode/launch.json` 会使用根目录 `.venv` 运行此独立脚本。

停住后在左侧 **Variables** 看 `messages` 与 `options`。按一次 **F10**，`chat_text` 出现；再按一次，`chat_encoded` 出现；再按一次，`chat_token_ids` 出现。在 **Watch** 可输入 `tokenizer.chat_template` 或 `chat_encoded["input_ids"]`。**F10** 执行当前行并停在下一行，**F11** 进入函数内部，**Shift+F11** 返回调用处，**F5** 继续到下一个断点。配置允许 F11 进入 Transformers 库代码；初学时先用 F10 理清本脚本的变量变化。
