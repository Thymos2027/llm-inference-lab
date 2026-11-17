# 第一课：从请求到输出

目标：把 Transformers 的单条生成循环，对应到 nano-vLLM 的两请求执行过程。先读懂一次正常运行，再研究缓存复用与调度变化。

## 先预测，再运行

打开 [trace_steps.py](trace_steps.py)。两条提示词分别有 5、4 个 token，分别生成 4、6 个新 token。先预测：总共执行几轮模型计算？第一轮处理多少 token？较短请求结束后，每轮还处理多少 token？

源码仓库保持在学习仓库旁边。根目录已配置 `../nano-vllm` 为可编辑依赖，使用学习仓库的 `.venv`。首次运行先执行 `uv sync --locked`；换设备也需准备相邻的 fork 仓库。

从学习仓库根目录运行：

```powershell
uv run .\03-nano-vllm\trace_steps.py
```

也可在 VS Code 按 F5 选择 `nano-vLLM 01: request lifecycle`。在 `seqs, is_prefill = scheduler.schedule()`、`token_ids = ...` 和 `scheduler.postprocess(...)` 设断点，用 F10 比较执行前后变量，用 F11 进入引擎。

## 沿着这条调用路径读

源码位于独立仓库 `D:\Coding\nano-vllm\nanovllm`。

1. `engine/llm_engine.py`：`add_request()` 编码输入并建立 Sequence；`generate()` 循环调用 `step()`。
2. `engine/scheduler.py`：`schedule()` 选出本轮请求并决定 prefill 或 decode。观察 waiting、running 和 num_scheduled_tokens。
3. `engine/model_runner.py`：`prepare_prefill()` 打包提示词；`prepare_decode()` 为每条请求只取 last_token，同时准备 positions、context_lens、slot_mapping 和 block_tables。
4. `layers/attention_windows.py`：新 K/V 写入 slot_mapping 指定的位置，通过 block_table 收集历史 K/V，再做注意力计算。
5. `engine/scheduler.py`：`postprocess()` 更新缓存计数、追加采样 token、检查结束条件并释放 block。

追踪脚本把 step() 中的三个操作展开以便观察；它使用原有调度策略。

## 对照日志

2026-11-17 在引擎提交 `8db1d75134717bdee43b52d1ad33cd0d9898a7f4`、Qwen3-0.6B、Windows 单卡 eager 路径复跑通过。

| 轮次 | 阶段 | 送入模型的 token 数 | 执行后运行中请求数 | 执行后已用 block |
| --- | --- | ---: | ---: | ---: |
| 1 | prefill | 9 | 2 | 2 |
| 2–3 | decode | 每轮 2 | 2 | 2 |
| 4 | decode | 2 | 1 | 1 |
| 5 | decode | 1 | 1 | 1 |
| 6 | decode | 1 | 0 | 0 |

第一轮前向已能采样第一个新 token，因此生成 6 个 token 只需要一次 prefill 加五次 decode。cached 计数表示已计算并存入缓存的位置；刚采样出的 token 要等下一轮才送入模型。完成时缓存计数归零，是释放逻辑清空了序列状态。

本次 block_size=256，两条短提示词各占一个物理块；总块数为 151，随可用显存变化。采样文字不是验收标准。末尾 PASS 检查输出长度和所有块释放；这里没有覆盖跨块、前缀复用和运行中新请求到达。

## 自己解释后再进入下一课

- 为什么第一轮 scheduled=9，却只生成两个新 token？
- 为什么第二轮每条请求只输入一个 token，仍能利用整个提示词？指出历史 K/V 的读取位置。
- block_table=[0] 中的 0 表示 token ID、逻辑位置还是物理块编号？
- 为什么第四轮完成的请求显示 cached=0、blocks=[]？
- 对照 Transformers 实验 05，指出这里哪部分对应 past_key_values，哪部分负责多请求管理。

练习：只把第一个请求的 max_tokens 从 4 改成 2，预测每轮请求数与块数后再运行，最后恢复为 4。不要用随机输出文本是否相同来判断正确性。

下一课再用跨越 256-token 边界的输入，观察物理块分配与相同前缀的缓存复用。
