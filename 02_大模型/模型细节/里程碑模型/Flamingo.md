# Flamingo

## 知识点解析

### 概述

Flamingo 是面向 few-shot 多模态学习的视觉语言模型，核心是在冻结视觉编码器和大语言模型之间加入 Perceiver Resampler 与 gated cross-attention，使语言模型能处理交错图文输入并完成少样本视觉任务。

### 解决的问题

Flamingo 之前，多模态模型通常需要针对每个任务微调，或者只能处理简单图文对。Flamingo 解决的是：

- 如何让大语言模型接收图像/视频信息。
- 如何支持交错的图文上下文。
- 如何像 GPT few-shot 那样，用少量图文示例适配新任务。

### 模型规格

Flamingo 论文中有 3B、9B、80B 等规模版本。典型组件：

| 组件 | 说明 |
| --- | --- |
| 视觉编码器 | 冻结的视觉模型，论文中使用 NFNet-F6 等强视觉编码器 |
| Perceiver Resampler | 将可变数量视觉特征压缩为固定数量 visual tokens |
| 语言模型 | 冻结或大部分冻结的 Chinchilla 风格 LM |
| Gated cross-attention | 插入到 LM 层间，让文本生成时读取 visual tokens |
| 输入形式 | 交错图像/视频帧和文本 |

### 完整架构

```text
images / video frames
  -> frozen vision encoder
  -> dense visual features
  -> Perceiver Resampler
  -> fixed number of visual tokens

text tokens
  -> language model blocks
       -> self-attention over text
       -> gated cross-attention to visual tokens
       -> FFN
  -> autoregressive text generation
```

Perceiver Resampler 的作用是把不同数量、不同尺寸的视觉特征压缩到固定数量 token，降低 LLM cross-attention 的成本。

Gated cross-attention 的作用是让模型在不破坏原语言模型能力的前提下逐步接入视觉信息。门控参数可以控制视觉信息注入强度。

### 做了什么改变

相比 CLIP：

- CLIP 只学习图文 embedding，不直接生成答案。
- Flamingo 可以基于图文上下文自回归生成。

相比 BLIP-2：

- BLIP-2 用 Q-Former 连接图像和 LLM。
- Flamingo 用 Perceiver Resampler 压缩视觉特征，并在 LM 层间插入 gated cross-attention。

相比 LLaVA：

- LLaVA 更像 projector + visual instruction tuning。
- Flamingo 更强调 few-shot 多模态上下文和交错图文序列。

### 训练方式

Flamingo 在大规模多模态网页数据和图文交错数据上训练，保留语言模型已有能力，同时学习视觉条件生成。训练目标本质上仍是图文条件下的自回归语言建模：

```text
maximize P(text tokens | previous text tokens, visual tokens)
```

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | Flamingo 如何接入视觉信息 | vision encoder -> Perceiver Resampler -> gated cross-attention |
| 对比题 | Flamingo 和 BLIP-2 区别 | Resampler + gated cross-attention vs Q-Former + frozen LLM |
| few-shot 题 | 为什么 Flamingo 支持多模态 few-shot | 可处理交错图文示例，上下文中学习任务格式 |
| 成本题 | 为什么要 resampler | 将可变视觉特征压缩为固定 tokens，降低 cross-attention 成本 |

### 易错点

- 把 Flamingo 说成普通图文检索模型；它是可生成的 few-shot VLM。
- 忽略 Perceiver Resampler，只说“图像特征接 LLM”不够具体。
- 忽略 gated cross-attention，它是保护原 LM 能力并注入视觉信息的关键。
- 把 Flamingo 和 LLaVA 的训练范式混为一谈。

## 面试应对

### Flamingo 的完整架构是什么？

回答思路：按视觉编码器、Perceiver Resampler、语言模型中的 gated cross-attention 回答。

回答模板：

Flamingo 的目标是让大语言模型具备多模态 few-shot 能力。它先用冻结视觉编码器提取图像或视频帧特征，再用 Perceiver Resampler 把可变数量的视觉特征压缩成固定数量的 visual tokens。文本侧是大语言模型，在若干层之间插入 gated cross-attention，让文本生成时可以 attend 到 visual tokens。这样模型可以接收交错的图文示例，并像语言模型做 few-shot 一样完成新的视觉问答、caption 或分类任务。
