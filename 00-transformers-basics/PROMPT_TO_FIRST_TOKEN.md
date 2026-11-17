# Qwen3-0.6B：从 prompt 到第一个新 token

本文固定使用本地 `Qwen3-0.6B`、Transformers 5.23.0 和单条普通文本 `The capital of France is`。在仓库根目录运行 [实验 06](06-attention-ffn.py)：

```powershell
uv run .\00-transformers-basics\06-attention-ffn.py
```

实验使用 `attn_implementation="eager"` 展开第 0 层的计算，完整模型前向设置 `use_cache=False`，最后用 `argmax` 选择第一个新 token。这是一条**贪心、无聊天模板、无采样**的路径。聊天输入会先经过 `apply_chat_template`，加入角色与特殊标记，得到不同的 ID 和输出；不能把本文的 5 个 ID 当作聊天请求的输入。

## 0. 本模型的结构与记号

本地 `config.json` 给出：词表大小 `V=151936`、隐藏维度 `H=1024`、28 层、16 个 Q 头、8 个 K/V 头、每头 `D=128`、FFN 中间维度 `I=3072`、RMSNorm 的 `eps=1e-6`。权重为 BF16，输入 ID 为 `int64`。这 28 层都是完整因果注意力；**GQA（Q 与 KV 头数不同）不等于线性注意力与完整注意力混合**。

下面的形状写作 `[B, T, ...]`，其中批次 `B=1`、prompt 长度 `T=5`。每层有各自训练得到的 Q/K/V、`o_proj`、FFN 和归一化参数；同一层对不同 token 位置重复使用自己的参数。

## 1. 文字变成 token ID

脚本调用：

```python
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
```

Qwen3 的 tokenizer 根据词表和分词规则把文字切成 token，再映射到整数 ID；不是按空格把每个英文单词简单分开。本次结果是：

| 位置 | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| 可见片段 | `The` | `Ġcapital` | `Ġof` | `ĠFrance` | `Ġis` |
| ID | 785 | 6722 | 315 | 9625 | 374 |

`Ġ` 是 tokenizer 显示前导空格的记号。`return_tensors="pt"` 得到形状 `[1,5]` 的整数张量；`.to(device)` 把它移到 GPU，不改变 ID。单条无 padding 的输入，其 `attention_mask` 若由 tokenizer 返回，就是五个 1；实验 06 只传 `input_ids`，在这条无 padding 路径中模型仍会建立因果遮罩。

## 2. ID 查 embedding 表

`model.model.embed_tokens(input_ids)` 把 ID 用作**行号**，从 `[151936,1024]` 的训练权重表取出五行：

```text
x[0, 0, :] = embedding.weight[785, :]   # 1024 个 BF16 数
x[0, 1, :] = embedding.weight[6722, :]
...
x.shape = [1, 5, 1024]
```

ID `785` 本身不含可用于计算的“词义大小”；信息在对应的 1024 维向量中。这里只是索引取数，没有做 Q/K/V 矩阵乘法。位置编号 `0..4` 之后用于 RoPE；本模型没有在此处另加一张绝对位置 embedding 表。

## 3. 28 层依次更新表示

初始 `hidden_states = x`。第 0 层先处理五个位置，产生 `[1,5,1024]`；第 1 层以该结果为输入，依次直到第 27 层。**层与层有数据依赖，不能同时算完**。每一层内部按下列顺序执行；实验 06 将第 0 层逐项展开，并与模型真正运行时捕获的输出核对。

### 3.1 Attention 前的 RMSNorm

对每个位置的 1024 维向量分别计算均方根、缩放，再乘该层学到的逐维权重：

```text
residual = hidden_states                      [1, 5, 1024]
a = input_layernorm(hidden_states)           [1, 5, 1024]
```

其中 `RMSNorm(z) = z / sqrt(mean(z²) + eps) * gamma`，均值只沿最后的 1024 维计算，**不在 token 之间求平均**。`residual` 保留原输入，供 Attention 结束后相加。归一化帮助控制进入投影的数值尺度。

### 3.2 Q/K/V 投影、QK norm 与 RoPE

同一个 `a` 分别经过该层三个不同的线性投影：

| 张量 | 投影后的形状 | 重排为多头后的形状 |
| --- | --- | --- |
| Q | `[1,5,2048]` | `[1,16,5,128]` |
| K | `[1,5,1024]` | `[1,8,5,128]` |
| V | `[1,5,1024]` | `[1,8,5,128]` |

Q、K 各自再经过该层的 `q_norm`、`k_norm`：对**每个头的 128 维向量**做 RMSNorm。这与上一节对投影**输入**的 RMSNorm 不同；它控制进入 `QKᵀ` 点积的 Q/K 尺度。V 不参与打分，因此这里没有 `v_norm`。

接着，模型用位置 ID `0..4` 算出 RoPE 的 `cos/sin`，旋转 Q 和 K，V 不旋转。RoPE 让点积可以利用位置信息；在本模型实现中，同一组位置 `cos/sin` 供各层使用，Q/K 旋转后的形状不变。

### 3.3 GQA、因果打分与加权 V

每个 K/V 头服务两个 Q 头。为便于看矩阵形状，实验脚本把 8 份 K/V 按组扩展到 16 份；这表示**共享相同的 K/V 内容**，不意味着模型学了 16 套 K/V 投影。第 0 层此时计算：

```text
scores = Q @ Kᵀ / sqrt(128)                  [1, 16, 5, 5]
```

每个 Q 头有一张 `5×5` 分数表：行是查询位置，列是可读的键位置。因果遮罩禁止读取未来，例如第 2 行只能读取第 0、1、2 列。遮罩后沿**键的位置**做 softmax，每一行的可见权重之和约为 1。这个权重是“从哪些上下文位置取多少信息”，**不是词表上某个 token 的生成概率**。

实验打印的第 0 层、第 0 个头的权重如下（显示到小数点后三位；行、列都按五个输入 token 的位置排列）：

```text
          The   capital   of    France    is    ← 被读取的位置
The      1.000   0.000   0.000   0.000   0.000
capital  0.871   0.130   0.000   0.000   0.000
of       0.377   0.001   0.621   0.000   0.000
France   0.633   0.039   0.309   0.018   0.000
is       0.445   0.006   0.369   0.006   0.174
↑ 当前查询位置
```

例如这个头为位置 4（`is`）算出的 128 维向量，约为 `0.445·V₀ + 0.006·V₁ + 0.369·V₂ + 0.006·V₃ + 0.174·V₄`。其余 15 个头各有自己的权重；其余 27 层也会再次计算，不能据这一张表推断完整模型为什么选择 `' Paris'`。所示小数有 BF16 与打印舍入误差。

```text
weights @ V                                  输出 [1, 16, 5, 128]
拼接 16 个头                                 输入 o_proj：[1, 5, 2048]
o_proj.weight                               权重 [1024, 2048]
attention_output = o_proj(拼接结果)          输出 [1, 5, 1024]
after_attention = residual + attention_output  输出 [1, 5, 1024]
```

`[1024,2048]` 是 **`o_proj` 持有的权重矩阵形状**（输出维度 × 输入维度）；`[1,5,1024]` 是**把五个位置各自的 2048 维向量送入 `o_proj` 后得到的张量形状**。PyTorch 的 `nn.Linear` 对每个位置计算 `输出 = 输入 @ o_proj.weight.T`，所以维度是 `[1,5,2048] @ [2048,1024] → [1,5,1024]`。它把 16 个头的结果混合并投回隐藏维度，才能与原输入相加。Attention 是这一层中**跨 token 读取和混合信息**的步骤；第 4 个位置的表示此时可以包含前五个位置的信息。

### 3.4 FFN 与第二次残差相加

再对 Attention 后的每个位置分别做 RMSNorm，然后进入 Qwen3 的门控 FFN：

```text
b = post_attention_layernorm(after_attention)              [1, 5, 1024]
gate = gate_proj(b)                                         [1, 5, 3072]
up   = up_proj(b)                                           [1, 5, 3072]
ffn  = down_proj(SiLU(gate) * up)                           [1, 5, 1024]
layer_output = after_attention + ffn                        [1, 5, 1024]
```

FFN 对五个位置使用**同一层的同一套权重**，但各位置的输入不同；算某个位置的 FFN 时不会直接读取其他位置。它处理的输入已经过 Attention，因此可能包含上下文信息。至此完成一层；下一层重复 3.1–3.4，使用自己的权重。实验 06 的第 0 层 Attention 输出与整层输出，均与 Transformers 的对应结果核对通过。

## 4. 最后归一化、词表分数与选 token

第 27 层输出仍为 `[1,5,1024]`。模型做**最终 RMSNorm**，再经过 `lm_head`：

```text
28 层输出                           [1, 5, 1024]
final_hidden = model.model.norm(...) [1, 5, 1024]
lm_head.weight                      [151936, 1024]
logits = lm_head(final_hidden)       [1, 5, 151936]
最后输入位置的分数                    logits[0, 4, :] → [151936]
argmax                              12095 → tokenizer.decode([12095]) == ' Paris'
```

`lm_head` 是最后一个线性层，负责把**每个位置的 1024 维隐藏向量变成词表中 151936 个 token 的分数**。它与 `o_proj` 一样遵循 `输出 = 输入 @ weight.T`：`[1,5,1024] @ [1024,151936] → [1,5,151936]`。输出的最后一维，第 `j` 个数就是候选 token ID `j` 的 logit。这里“经过 `lm_head`”指执行这次线性投影，并不是立即把向量变成文字。

具体到最后一个输入位置 `is`：设它经过最终 RMSNorm 的向量为 `h = final_hidden[0,4,:]`，形状 `[1024]`。`lm_head.weight[j,:]` 是候选 token `j` 对应的一行 1024 维权重。两者做**点积并求和**，得到该候选的一个分数（此模型的 `lm_head` 没有 bias）：

```text
logits[0, 4, j] = h[0] * weight[j, 0]
                + h[1] * weight[j, 1]
                + ...
                + h[1023] * weight[j, 1023]
```

对 `j=0..151935` 的每一行都这样计算，就得到 `[151936]` 个分数；对五个输入位置同时计算，得到 `[1,5,151936]`。可以把它想成“一个 1024 维查询向量与整张词表权重表逐行做点积”，而不是把两个张量同位置逐元素相乘后直接得到文字。本例 `logits[0,4,12095]` 在贪心选择中最高，因此选出 ID `12095`。

本模型配置 `tie_word_embeddings=true`，所以 `lm_head` 与输入 embedding **共享同一个 `[151936,1024]` 权重张量**；作用方向不同：输入时按 ID 查一行向量，输出时用隐藏向量与词表中每一行做点积，形成一组分数。每个位置的 logits 都预测其**下一个** token；位置 4 对应 `is`，所以用它的分数来选位置 5 的第一个新 token。logits 是原始分数，不是概率；贪心 `argmax` 不必先做 softmax，因为 softmax 不改变最大值的位置。若选择采样，则还要根据温度、top-k/top-p 等规则处理分数后抽样，结果未必是 12095。

实验脚本的普通前向为五个输入位置都计算 logits，因此形状是 `[1,5,151936]`；实际生成只需要最后位置的分数，Transformers 的生成路径可选择只计算所需位置。得到 ID `12095` 后，才能将它追加到输入并解码显示 `' Paris'`。**这一次前向还没有把新 token 作为输入走过 28 层**；要生成第二个新 token，才继续进行下一次前向。若启用 KV cache，第一次 prefill 会保存这五个输入位置的 K/V，下一轮 decode 只需输入刚生成的 token，并读取先前的缓存；见[实验 05](05-kv-cache.py)。

## 对照代码与边界

| 问题 | 对照位置 |
| --- | --- |
| tokenizer 怎样得到 ID | [实验 01](01-tokenizer.py)、[实验 06](06-attention-ffn.py) 的 `tokenizer(...)` |
| embedding、一层 Attention 和 FFN 怎样算 | [实验 06](06-attention-ffn.py) 的 `embed_tokens` 至 `manual_layer_output` |
| 28 层和最终 `lm_head` 怎样连接 | Transformers 的 `Qwen3Model.forward`、`Qwen3ForCausalLM.forward`；[实验 03](03-forward-logits.py) 可查看 logits |
| 第一个 token 怎样选、后续怎样生成 | [实验 04](04-generation-loop.py)；缓存路径见[实验 05](05-kv-cache.py) |

以上形状和输出来自**此本地模型、此输入和贪心选择**。实验 06 为教学目的使用 eager Attention，并以 `use_cache=False` 对照；FlashAttention/SDPA、padding、批处理和 KV cache 的底层实现会有不同的中间张量或计算调度，但模型的层级数据流不因此改变。
