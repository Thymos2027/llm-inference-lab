# Transformers 基础：从输入到生成

这是独立于 nano-vLLM 源码的先修实验。先弄清 Hugging Face Transformers 的 tokenizer、模型输入输出和生成循环，再回到推理引擎。实验代码只放在本目录；`D:\Coding\nano-vllm` 不做改动。

## 学习顺序

1. **文字与 token**：运行本页的实验 01，理解 token ID、解码和聊天模板。
2. **模型与张量**：读 `config.json`，认识 `input_ids`、形状、设备和数据类型。
3. **一次前向计算**：看 `logits` 的形状，理解“预测下一个 token”的含义。
4. **生成循环**：亲手写一个最小的“前向计算 → 选 token → 追加输入”循环，再与 `generate()` 对照。
5. **KV cache**：比较有无缓存时的输入和计算，再回到 nano-vLLM 的 block table。

每次只推进一个实验。**实验 01–05 已完成**；下一步可回到 nano-vLLM 观察它的缓存块管理。

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

## 实验 02：模型与张量

读取本机 Qwen3-0.6B 的 `config.json`，再把两句长短不同的文字编码成同一个批次。此实验只加载配置和 tokenizer，不加载约 1.5 GB 的模型权重，也不运行模型前向。仍在仓库根目录执行：

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe .\00-transformers-basics\02-model-tensors.py
```

模型放在别处时可加 `--model <模型目录>`。代码先打印配置中的 `vocab_size`、`hidden_size`、层数、注意力头数、最大位置数与 `torch_dtype`，然后打印 `input_ids` 和 `attention_mask`：

- 两个张量的形状都是 `[批次大小, 序列长度]`。本机这次为 `(2, 5)`：两行分别对应两句文字，短句末尾补了一个位置。每一行的列数相同，方便组成一个批次。
- `input_ids` 存词表索引，数据类型是 `torch.int64`；补齐位置放 `pad_token_id`。`attention_mask` 中 1 表示真实 token，0 表示补齐位置。本次第一行的 mask 是 `[1, 1, 1, 1, 0]`，第二行全为 1。实际 ID 和长度取决于 tokenizer 文件。
- tokenizer 先在 CPU 生成张量；代码用 `.to(device)` 把**模型输入**移到可用的 CUDA 设备（没有 CUDA 时保留在 CPU）。两个张量移动后仍是整数，不会因设备变化而变成浮点数。
- 此模型配置的 `hidden_size=1024`、`torch_dtype=bfloat16`。前者表示若加载模型，词向量输出的形状可推为 `[2, 5, 1024]`；这里只推算形状，没有生成该张量。后者描述模型权重的存储精度，不是 `input_ids` 的数据类型。

**读代码时先回答：**`input_ids[0, 4]` 为什么不是第一句的真实 token？`attention_mask[0, 4]` 是多少？`(2, 5)` 的两个数字各代表什么？把输入移到 GPU 后，`input_ids.dtype` 会改变吗？

本机 2026-11-16 使用 Transformers 5.23.0 与 PyTorch 2.14.1+cu130 运行通过：`input_ids`、`attention_mask` 均为 `(2, 5)`、`torch.int64`，并从 CPU 移到 `cuda:0`。下一实验才会加载模型权重，观察实际前向输出的 `logits`。

## 实验 03：一次前向计算与 logits

这次加载本机 Qwen3-0.6B 的模型权重，用 `The capital of France is` 做**一次**前向计算。在仓库根目录执行：

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe .\00-transformers-basics\03-forward-logits.py
```

模型放在别处时可加 `--model <模型目录>`。脚本只读取本地模型文件；使用配置中的 BF16 权重和 PyTorch SDPA 注意力实现，有 CUDA 时在 GPU 上计算。`model.eval()` 与 `torch.inference_mode()` 用于推理，`use_cache=False` 使这次实验只看前向输出。

- 提示词被编码成 `(1, 5)` 的 `input_ids`：一个输入、五个 token。`model(**model_inputs)` 返回的 `outputs.logits` 是 `(1, 5, 151936)`，三个维度依次是**批次、输入位置、词表大小**。这里的每个位置都有一组覆盖整个词表的分数。
- `logits[0, -1, :]` 取第一个输入的最后一个位置，形状是 `(151936,)`。它给出**接在整段提示词之后**的下一个 token 的分数；`logits[0, 0, :]` 则只基于第一个输入 token 对其后一个 token 打分。原始 logits 不是概率。
- `torch.topk(..., k=5)` 只为观察分数最高的五个候选；它没有把 token 加回输入，也没有重复运行模型。本机这次最高分的 token 解码为 `' Paris'`（含前导空格），分数为 `17.5`。具体候选和分数可能随模型文件或运行环境变化。
- 模型权重与本次 logits 的数据类型都是 `torch.bfloat16`，而 `input_ids` 仍是 `torch.int64`。权重中的浮点数用于计算；token ID 是索引。

**读代码时先回答：**为什么 `logits` 比 `input_ids` 多一个维度？`logits[0, -1, :]` 中的 `-1` 和 `:` 各选中了什么？最高分 token 是完整回答吗？如果想得到第二个新 token，还需要做什么？

本机 2026-11-16 使用 Transformers 5.23.0、PyTorch 2.14.1+cu130 与 Qwen3-0.6B 运行通过，得到上述形状与候选。实验 04 才会把选出的 token 追加到输入，构造生成循环。

## 实验 04：亲手写生成循环

沿用实验 03 的 Qwen3-0.6B 和 `The capital of France is`，最多生成 8 个新 token。在仓库根目录执行：

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe .\00-transformers-basics\04-generation-loop.py
```

模型放在别处时可加 `--model <模型目录>`。脚本只读取本地模型文件，默认使用 CUDA（若可用），并在 `eval()` 和 `torch.inference_mode()` 下运行。

- 每一步都将**当前完整的** `input_ids` 和 `attention_mask` 送入模型，取 `logits[:, -1, :]`，用 `argmax` 选择分数最高的下一个 token。这是贪心选择，不是采样。
- `torch.cat` 把选中的 token ID 加到 `input_ids` 末尾，同时给 `attention_mask` 加一个 1。下一步输入长度因此增加 1；如果选中结束 token，则提前停下。这里 `use_cache=False`，每次重新计算完整输入，方便看清循环；缓存留到实验 05。
- 最后对同一提示词调用 `model.generate(max_new_tokens=8, do_sample=False, use_cache=False)`。脚本打印两边的新 token ID 和解码结果，并断言它们逐个相同。`generate()` 返回的序列包含原提示词，所以比较前先切出新增部分。

**读代码时先回答：**第一次与第二次前向的输入长度分别是多少？为什么只取最后一个位置的 logits？只追加 `input_ids` 而不追加 `attention_mask` 会怎样？`max_new_tokens=8` 是否意味着输出一定有 8 个 token？

本机 2026-11-16 使用 Transformers 5.23.0、PyTorch 2.14.1+cu130 与 Qwen3-0.6B 运行通过。手写循环的输入长度从 5 增至 12，依次选出 8 个新 token，解码为 `' Paris. The capital of France is also'`；`generate()` 给出完全相同的 8 个 ID。实际文字可能随模型文件或运行环境变化。

## 实验 05：KV cache 如何减少重复计算

沿用同一个模型和提示词，同时运行**不使用缓存**与**使用缓存**的贪心生成路径，逐步比较送入模型的输入长度。在仓库根目录执行：

```powershell
uv sync --locked --python 3.12
& .\.venv\Scripts\python.exe .\00-transformers-basics\05-kv-cache.py
```

模型放在别处时可加 `--model <模型目录>`。脚本最多生成 4 个新 token；两条路径每一步都取最后位置 logits 的 `argmax`，并断言选出的 token ID 相同。

- 第一步是 **prefill**：两条路径都送入完整的 5 个提示词 token。缓存路径设置 `use_cache=True`，从输出中取得 `past_key_values`；其中保存了已处理位置的 key/value 张量。
- 第二步起是 **decode**：无缓存路径每次重新送入越来越长的完整序列；缓存路径只送入刚生成的 1 个 token，同时传入之前的 `past_key_values`。`attention_mask` 仍覆盖**完整上下文**，其长度随生成增长。`past_key_values.get_seq_length()` 可查看调用前已有多少个缓存位置。
- 本机 2026-11-16 使用 Transformers 5.23.0、PyTorch 2.14.1+cu130 与 Qwen3-0.6B 运行通过。4 次前向中，无缓存输入长度为 `5, 6, 7, 8`，共送入 26 个 token 位置；缓存路径为 `5, 1, 1, 1`，共送入 8 个。两边生成的 ID 都是 `[12095, 13, 576, 6722]`，解码为 `' Paris. The capital'`。这些数字是**送入模型的 token 位置数**，不是运行时间或总计算量；缓存路径仍会对历史 key/value 做注意力计算。

**读代码时先回答：**为什么第二步缓存路径只输入 1 个 token，却能利用整个提示词？第二步的 `cached_before` 为什么是 5？为什么脚本仍传入覆盖完整上下文的 `attention_mask`？若共生成 4 个 token，为什么最后一个新 token 尚未写入 `past_key_values`？

回到 `D:\Coding\nano-vllm` 时，可先看 `nanovllm/engine/model_runner.py` 的 `prepare_decode()`：它也只把每条序列的 `last_token` 送入本轮模型计算。再看 `nanovllm/engine/block_manager.py` 中的块分配，以及序列的 `block_table`：这个表记录逻辑序列使用哪些物理 KV 缓存块，使旧位置的 key/value 可被继续读取。这里的 Transformers `past_key_values` 和 nano-vLLM 的块表服务于同一个复用目标，但内部数据结构不同；本实验没有运行或修改 nano-vLLM。

## 用 VS Code 断点调试

用 VS Code 打开 `D:\Coding\llm-inference-lab` 文件夹，打开 [01-tokenizer.py](01-tokenizer.py)，在 `chat_text = ...` 这一行的左侧行号旁单击设置断点。按 **F5**，选择 **Transformers 01: tokenizer**；本仓库的 `.vscode/launch.json` 会使用根目录 `.venv` 运行此独立脚本。

停住后在左侧 **Variables** 看 `messages` 与 `options`。按一次 **F10**，`chat_text` 出现；再按一次，`chat_encoded` 出现；再按一次，`chat_token_ids` 出现。在 **Watch** 可输入 `tokenizer.chat_template` 或 `chat_encoded["input_ids"]`。**F10** 执行当前行并停在下一行，**F11** 进入函数内部，**Shift+F11** 返回调用处，**F5** 继续到下一个断点。配置允许 F11 进入 Transformers 库代码；初学时先用 F10 理清本脚本的变量变化。

实验 02 可打开 [02-model-tensors.py](02-model-tensors.py)，在 `model_inputs = ...` 设断点，按 **F5** 选择 **Transformers 02: model tensors**。停住时比较 `batch["input_ids"]`、`batch["attention_mask"]`、`input_ids.shape` 和 `input_ids.dtype`；按一次 **F10** 后再看 `model_inputs["input_ids"].device`，确认变化的是设备，而形状与数据类型不变。

实验 03 可打开 [03-forward-logits.py](03-forward-logits.py)，在 `logits = outputs.logits` 设断点，按 **F5** 选择 **Transformers 03: forward logits**。停住时 `outputs` 已有值；用 **F10** 执行这一行后，在 **Debug Console** 输入 `logits.shape`，再用 **F10** 执行 `next_token_scores = ...`，输入 `next_token_scores.shape` 或 `next_token_scores.argmax().item()`。`logits` 很大，优先查看形状或单个位置的分数，不必展开整个张量。

实验 04 可打开 [04-generation-loop.py](04-generation-loop.py)，在 `next_token_id = ...` 设断点，按 **F5** 选择 **Transformers 04: generation loop**。停住时看 `step`、`input_ids.shape`、`outputs.logits.shape`；按 **F10** 观察 `next_token_id`，再继续到下一轮，比较输入长度。执行完 `model.generate(...)` 后，比较 `new_token_ids` 和 `generate_new_ids`。

实验 05 可打开 [05-kv-cache.py](05-kv-cache.py)，在 `cached_outputs = model(...)` 设断点，按 **F5** 选择 **Transformers 05: KV cache**。第一轮查看 `cached_input_ids.shape` 和 `past_key_values`；第二轮比较 `full_input_ids.shape`、`cached_input_ids.shape` 与 `cached_before`。执行该行后再看 `cached_outputs.past_key_values.get_seq_length()`，确认缓存长度增加。
