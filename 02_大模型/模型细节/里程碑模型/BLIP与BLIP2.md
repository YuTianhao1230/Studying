# BLIP 与 BLIP-2

## 知识点解析

### 概述

BLIP 是统一图文理解和生成的视觉语言预训练框架，BLIP-2 进一步用 Q-Former 连接冻结视觉编码器和冻结大语言模型，以低成本获得视觉到语言的生成能力。

### 解决的问题

BLIP 解决两个问题：

- web 图文数据噪声大，caption 质量不稳定。
- 图文理解和图文生成常被分开训练，框架不统一。

BLIP-2 解决的问题是：

- 端到端训练大视觉模型和大语言模型成本很高。
- 直接把大量视觉 token 喂给 LLM 成本高、对齐难。
- 需要一个轻量桥接模块把视觉表征转成 LLM 可用信息。

### 完整架构

#### BLIP 架构

BLIP 使用 Multimodal Mixture of Encoder-Decoder，简称 MED。它通过共享部分 Transformer 层，使模型同时支持三种功能：

```text
image
  -> image encoder, usually ViT-B or ViT-L
  -> visual tokens

text
  -> text encoder mode for understanding
  -> text decoder mode for generation
  -> multimodal encoder mode for image-text fusion
```

典型组件：

- image encoder：ViT-B/16 或 ViT-L/16。
- text encoder/decoder：BERT-base 风格 Transformer。
- multimodal encoder：在文本 self-attention 基础上加入 cross-attention 到 image tokens。
- captioner：为图像生成 caption。
- filter：过滤 noisy caption，形成更干净的 bootstrapped caption。

### BLIP 训练目标

- ITC：图文对比学习。
- ITM：图文匹配判断。
- LM：图像条件下自回归生成 caption。

BLIP 的 CapFilt 思路：

```text
web image-text pairs
  -> captioner generates synthetic captions
  -> filter removes noisy pairs
  -> train on cleaned / bootstrapped captions
```

#### BLIP-2 架构

BLIP-2 的核心是 Q-Former。

```text
image
  -> frozen image encoder, often EVA-CLIP/ViT-g style encoder
  -> frozen visual features

learnable query tokens, commonly 32 queries
  -> Q-Former, 12-layer Transformer, hidden size 768
  -> query attends to frozen visual features through cross-attention
  -> compact visual representation

projection
  -> align to frozen LLM embedding space
  -> frozen LLM, such as OPT or Flan-T5
  -> generated answer/caption
```

Q-Former 的作用是用少量 query token 从图像特征中抽取与语言相关的信息，避免把所有 patch token 都直接送入 LLM。

### BLIP-2 两阶段训练

第一阶段：视觉语言表示学习。

- 冻结 image encoder。
- 训练 Q-Former。
- 使用 ITC、ITM、image-grounded text generation 等目标。

第二阶段：视觉到语言生成学习。

- 冻结 LLM。
- 训练 Q-Former 和投影层，让视觉 query 输出能被 LLM 使用。
- 目标是让 LLM 在视觉条件下生成文本。

### 做了什么改变

BLIP 相比 ALBEF：

- 不只做理解，也做生成。
- 用 captioner/filter 改善 web caption 噪声。
- MED 统一 encoder、decoder 和 multimodal encoder。

BLIP-2 相比 BLIP：

- 冻结大视觉编码器和大语言模型。
- 只训练轻量 Q-Former 和投影层。
- 用少量 query token 压缩视觉信息，大幅降低训练成本。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | BLIP-2 的 Q-Former 是什么 | 12 层 Transformer，常用 32 个 learnable queries，从冻结视觉特征抽取信息 |
| 对比题 | BLIP 和 BLIP-2 区别 | BLIP 统一理解生成，BLIP-2 用 Q-Former 低成本连接冻结 LLM |
| 训练题 | BLIP 的 CapFilt 是什么 | captioner 生成 caption，filter 清洗 noisy pairs |
| 选型题 | BLIP-2 为什么训练便宜 | 冻结 image encoder 和 LLM，只训练桥接模块 |

### 易错点

- 把 BLIP-2 说成端到端训练视觉编码器和 LLM；它的关键恰恰是冻结大模块。
- 只说 Q-Former 是 projector，不讲 learnable query 和 cross-attention。
- 把 BLIP 的 ITC/ITM/LM 三个目标混在一起。
- 忽略 BLIP 用 CapFilt 处理 web 数据噪声。

## 面试应对

### BLIP-2 的完整架构是什么？

回答思路：按 frozen image encoder、Q-Former、projection、frozen LLM 回答。

回答模板：

BLIP-2 的核心是用轻量 Q-Former 连接冻结视觉编码器和冻结大语言模型。图像先经过冻结的视觉编码器得到 visual features；Q-Former 通常是 12 层 Transformer，hidden size 768，带 32 个 learnable query tokens，这些 query 通过 cross-attention 从视觉特征里抽取语言相关信息。之后再经过 projection 对齐到 LLM embedding 空间，送入冻结的 OPT 或 Flan-T5 等语言模型生成答案。它的优势是不用端到端训练视觉大模型和 LLM，只训练桥接模块就能获得视觉语言生成能力。
