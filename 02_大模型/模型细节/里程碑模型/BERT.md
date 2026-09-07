# BERT

## 知识点解析

### 概述

BERT 是 Encoder-only Transformer 预训练语言模型，核心是通过双向上下文学习通用语言理解表示，并通过“预训练 + 微调”范式迁移到分类、匹配、抽取和序列标注任务。

### 解决的问题

BERT 解决了早期 NLP 模型的三个问题：

- 任务专用模型多，迁移能力弱。
- 单向语言模型无法同时利用左右上下文。
- 下游任务需要大量标注数据从头训练。

BERT 的目标不是做开放生成，而是得到强语义表示。

### 模型规格

| 版本 | 层数 | hidden size | attention heads | 参数量 |
| --- | --- | --- | --- | --- |
| BERT-Base | 12 | 768 | 12 | 约 110M |
| BERT-Large | 24 | 1024 | 16 | 约 340M |

输入长度通常为 512 tokens，词表使用 WordPiece。

### 完整架构

```text
input tokens
  -> WordPiece tokenization
  -> token embedding + segment embedding + position embedding
  -> Transformer Encoder block x N
       -> bidirectional self-attention
       -> FFN
       -> residual + LayerNorm
  -> [CLS] representation for sentence-level tasks
  -> token representations for token-level tasks
```

输入中常见特殊 token：

- `[CLS]`：句级分类向量。
- `[SEP]`：分隔句子 A 和句子 B。
- segment embedding：区分句子 A/B。
- position embedding：可学习绝对位置编码。

### 训练目标

MLM：随机 mask 一部分 token，让模型根据左右上下文预测原词。

```text
input:  the cat [MASK] on the mat
target: sat
```

NSP：判断句子 B 是否是句子 A 的下一句。后续 RoBERTa 发现 NSP 不一定必要，更多改进来自数据、batch 和训练策略。

### 做了什么改变

相比 GPT 类单向 LM，BERT 最大变化是 bidirectional encoder。每个 token 可以同时看左右上下文，因此更适合理解任务。

相比 ELMo 等早期表示模型，BERT 把深层 Transformer、预训练目标和下游微调整合成统一范式。

### 适用场景

- 文本分类：用 `[CLS]` 接分类头。
- 句子匹配：输入 `[CLS] sentence A [SEP] sentence B [SEP]`。
- 命名实体识别：每个 token 接分类头。
- 抽取式问答：预测 answer span 的 start/end。
- Reranker：对 query-doc pair 进行相关性打分。

### 能力边界

BERT 不适合直接做自回归长文本生成。它的预训练目标是补全 masked token，而不是 next-token prediction。现代生成式大模型更多采用 decoder-only 架构。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | BERT-Base 有多少层 | 12 层、768 hidden、12 heads、约 110M 参数 |
| 目标题 | MLM 和 NSP 是什么 | MLM 学双向上下文，NSP 学句间关系 |
| 对比题 | BERT 和 GPT 区别 | Encoder-only 双向理解 vs Decoder-only 自回归生成 |
| 应用题 | BERT 怎么做分类/NER/QA | 分类用 `[CLS]`，NER 用 token states，QA 预测 span |

### 易错点

- 说 BERT 是生成模型。BERT 主要用于理解表示，不是自回归生成。
- 说 MLM 等于普通语言模型。MLM 是 mask 预测，不是按顺序预测下一个 token。
- 忽略 segment embedding，导致句对任务输入解释不完整。
- 把 BERT 的双向理解和 GPT 的 causal attention 混淆。

## 面试应对

### BERT 的完整架构和核心创新是什么？

回答思路：按 Encoder-only、输入 embedding、训练目标和适用任务回答。

回答模板：

BERT 是 Encoder-only Transformer。BERT-Base 有 12 层 encoder、hidden size 768、12 个 attention heads，BERT-Large 有 24 层、hidden size 1024、16 个 heads。输入由 token embedding、segment embedding 和 position embedding 相加得到，经过双向 self-attention 编码后，用 `[CLS]` 做句级任务，用 token hidden states 做序列标注或抽取式问答。它的核心创新是用 MLM 学双向上下文表示，并通过预训练加微调把同一个模型迁移到多种理解任务。
