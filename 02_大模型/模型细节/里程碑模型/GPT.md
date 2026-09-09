# GPT

## 知识点解析

### 概述

GPT 是 [Decoder-only](<../../基础架构/Decoder-only vs Encoder-Decoder.md>) [Transformer](<../../基础架构/Transformer.md>) 自回归语言模型路线，核心是通过 next-token prediction 学习通用生成能力，并在规模化后形成 prompt、few-shot、in-context learning 和指令助手能力。

### 解决的问题

GPT 系列解决的问题按阶段变化：

- GPT-1：验证生成式预训练可以迁移到下游 NLP 任务。
- GPT-2：扩大模型和语料，展示 zero-shot 生成能力。
- GPT-3：用大规模 decoder-only 模型证明 few-shot 和 in-context learning。
- InstructGPT/ChatGPT：通过后训练让 base model 从续写器变成助手。

### 典型模型规格

| 版本 | 层数 | hidden size | heads | 参数量 | 重点 |
| --- | --- | --- | --- | --- | --- |
| GPT-1 | 12 | 768 | 12 | 约 117M | 生成式预训练 + 微调 |
| GPT-2 Small | 12 | 768 | 12 | 约 124M | 更大 WebText 预训练 |
| GPT-2 Medium | 24 | 1024 | 16 | 约 355M | 扩大容量 |
| GPT-2 Large | 36 | 1280 | 20 | 约 774M | 更强生成 |
| GPT-2 XL | 48 | 1600 | 25 | 约 1.5B | zero-shot 能力更明显 |
| GPT-3 | 96 | 12288 | 96 | 175B | in-context learning |

闭源 GPT-4 之后的具体层数和参数未公开，面试中不要编造。

### 完整架构

```text
input tokens
  -> BPE tokenization
  -> token embedding + positional embedding
  -> Transformer decoder block x N
       -> masked multi-head self-attention
       -> MLP / FFN
       -> residual + LayerNorm
  -> LM head
  -> next-token probability distribution
```

GPT 使用 causal mask，每个位置只能看见自己和之前的 token：

```text
P(x) = product_t P(x_t | x_1, ..., x_{t-1})
```

### 训练目标

GPT 的基础目标是 next-token prediction：

```text
maximize sum_t log P(x_t | x_<t)
```

后续助手模型通常继续做：

1. [SFT](<../../../03_训练优化与对齐/后训练与对齐/SFT 监督微调.md>)：学习指令-回答格式。
2. RLHF/RLAIF/DPO：对齐人类偏好和安全要求。
3. 工具/函数调用训练：学习输出结构化 tool call。
4. 推理增强训练：强化数学、代码和复杂规划。

### 做了什么改变

相比原始 Transformer：

- 去掉 Encoder 和 cross-attention，只保留 decoder causal self-attention。
- 用语言建模统一各种生成任务。
- 通过 prompt 把任务说明和样例写进上下文，不一定需要改参数。

现代 LLM 在 GPT 路线上常见改造：

- [RoPE](<../../基础架构/RoPE.md>) 替代绝对位置编码。
- [RMSNorm](<../../基础架构/RMSNorm.md>) 替代 LayerNorm。
- [SwiGLU](<../../../03_训练优化与对齐/参数/常见激活函数.md>) 替代 ReLU/GELU FFN。
- GQA/MQA 降低 [KV Cache](<../../../05_推理部署与系统/推理工程/KV_Cache与Prefill_Decode.md>)。
- [MoE](<../../基础架构/MoE.md>) 扩大参数容量但控制激活成本。

### 能力边界

GPT 的预训练目标只是预测下一个 token，不保证事实正确、推理可靠或安全合规。ChatGPT 式体验主要来自后训练、系统 prompt、工具调用和产品反馈闭环。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | GPT 为什么是 decoder-only | 因为目标是自回归生成，只需要 causal self-attention |
| 目标题 | next-token prediction 公式 | 写 `P(x_t \| x_<t)` 并说明 teacher forcing |
| 对比题 | GPT 和 [BERT](<BERT.md>) 区别 | 生成 vs 理解，causal mask vs bidirectional attention |
| 后训练题 | ChatGPT 为什么比 GPT-3 好用 | SFT + [RLHF](<../../../03_训练优化与对齐/后训练与对齐/RLHF 基于人类反馈的强化学习.md>) 让模型遵循指令和偏好 |

### 易错点

- 把 GPT-4 之后闭源模型的层数和参数说死。
- 把 in-context learning 说成模型参数更新；它只是上下文条件化。
- 只讲预训练，不讲 SFT/RLHF 对助手化体验的重要性。
- 忽略 causal mask，导致 GPT 和 BERT 区别说不清。

## 面试应对

### GPT 的完整架构是什么？

回答思路：按 decoder-only、causal mask、next-token prediction、后训练回答。

回答模板：

GPT 是 Decoder-only Transformer。输入经过 BPE tokenization 和 embedding 后，进入多层 masked self-attention block，每个 token 只能 attend 到自己和之前的 token，最后通过 LM head 预测下一个 token。GPT-1 是 12 层、768 hidden、12 heads，GPT-3 公开规格是 96 层、hidden size 12288、96 heads、175B 参数。它的基础训练目标是最大化 `P(x_t | x_<t)`。ChatGPT 这类助手模型还需要 SFT 和 RLHF 等后训练，让模型从单纯续写变成会遵循指令的助手。
