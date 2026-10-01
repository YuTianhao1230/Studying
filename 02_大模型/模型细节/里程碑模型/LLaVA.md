# LLaVA

## 知识点解析

### 概述

LLaVA 是开源视觉指令模型路线的代表，核心是用 [CLIP](<CLIP.md>) 视觉编码器提取图像特征，通过 projector 对齐到 LLM embedding 空间，再用视觉指令数据微调大语言模型。

### 解决的问题

LLaVA 解决的是开源 LLM 不会看图的问题：

- LLM 只有文本输入，无法直接处理图像。
- 训练原生多模态大模型成本高。
- 视觉问答和多轮图文对话需要指令跟随数据。

LLaVA 的价值是用相对简单的工程结构，低成本把视觉能力接入开源 LLM。

### 典型模型规格

| 版本/组件 | 常见配置 |
| --- | --- |
| 视觉编码器 | CLIP ViT-L/14，常见输入分辨率 224 或 336 |
| 视觉特征 | patch tokens / selected layer features |
| projector | linear projector 或 2-layer MLP |
| 语言模型 | Vicuna/LLaMA 系列，常见 7B、13B |
| 训练阶段 | projector 预对齐 + 视觉指令微调 |

不同 LLaVA 版本会调整视觉编码器、分辨率、projector、数据规模和训练策略。面试中重点讲清“CLIP vision encoder + projector + LLM + visual instruction tuning”。

### 完整架构

```text
image
  -> CLIP vision encoder, often ViT-L/14
  -> visual patch features
  -> linear / MLP projector
  -> visual tokens in LLM embedding space

text instruction
  -> tokenizer
  -> text embeddings

visual tokens + text embeddings
  -> LLM, such as Vicuna/LLaMA
  -> autoregressive answer
```

### 训练流程

第一阶段：视觉-语言预对齐。

- 冻结视觉编码器和 LLM。
- 训练 projector。
- 目标是让 visual tokens 能被 LLM 理解。
- 常用图像-caption 数据做对齐。

第二阶段：视觉指令微调。

- 输入图像和多轮指令问答。
- 微调 projector 和 LLM 的部分或全部参数。
- 学习视觉问答、描述、推理和多轮对话格式。

### 做了什么改变

相比 CLIP：

- CLIP 输出图文相似度，不直接对话。
- LLaVA 把 CLIP 视觉特征接入 LLM，可以生成自然语言回答。

相比 BLIP-2：

- BLIP-2 使用 Q-Former 作为桥接模块，通常冻结 LLM。
- LLaVA 更简单，主要使用 projector，并依赖视觉指令微调获得对话能力。

相比 [Flamingo](<Flamingo.md>)：

- Flamingo 通过 gated cross-attention 接入视觉 tokens，强调 few-shot 交错图文。
- LLaVA 把视觉 tokens 直接作为 LLM 输入前缀，更容易复现和扩展。

### 能力边界

LLaVA 的瓶颈主要来自：

- 视觉编码器分辨率和特征层选择。
- projector 对齐能力。
- 视觉指令数据质量。
- OCR、小目标、空间关系、数量关系和细粒度 grounding。
- 幻觉：模型可能根据语言先验回答不存在的内容。

### 在项目中的意义

LLaVA 常用于多模态攻击评估，因为它代表开源 MLLM 的典型结构：

```text
CLIP-like visual encoder -> projector -> LLM
```

攻击如果能从 CLIP/ALBEF/TCL 迁移到 LLaVA，说明扰动可能影响了视觉编码器或图文语义对齐中较共享的部分，而不只是源模型的任务头。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | LLaVA 如何让 LLM 看图 | CLIP vision encoder 提特征，projector 对齐到 LLM embedding |
| 训练题 | 两阶段训练分别做什么 | 先训练 projector 对齐，再视觉指令微调 |
| 对比题 | LLaVA 和 BLIP-2 区别 | projector 简单直连 vs Q-Former 查询视觉信息 |
| 局限题 | LLaVA 为什么会幻觉 | 视觉信息不足、指令数据偏差、LLM 语言先验 |
| 项目题 | 为什么把 LLaVA 当攻击目标 | 代表开源 MLLM，结构中含 CLIP-like 视觉编码器 |

### 易错点

- 把 LLaVA 说成端到端从零训练的原生多模态模型。
- 忽略 projector 的作用。
- 只说 CLIP 接 LLM，不讲视觉指令微调。
- 把 LLaVA 的视觉能力完全归因于 LLM，忽略视觉编码器和数据。

## 面试应对

### LLaVA 的完整架构是什么？

回答思路：按 CLIP 视觉编码器、projector、LLM、两阶段训练回答。

回答模板：

LLaVA 的基本结构是 CLIP vision encoder 加 projector 加 LLM。图像先经过 CLIP ViT-L/14 等视觉编码器得到 patch-level visual features，再通过 linear projector 或 MLP projector 映射到 LLM 的 embedding 空间；文本指令经过 tokenizer 得到 text embeddings，视觉 token 和文本 token 一起送入 Vicuna/LLaMA 类语言模型自回归生成答案。训练一般分两阶段：先冻结视觉编码器和 LLM，只训练 projector 做视觉语言预对齐；再用视觉指令数据微调模型，让它学会图文问答和多轮对话。

### LLaVA 的两阶段训练分别优化什么？

回答思路：区分特征空间对齐和指令行为学习，并说明具体冻结范围随版本变化。

回答模板：

第一阶段用图像与描述数据训练视觉 projector，目标是在语言建模损失监督下把视觉编码器输出映射到 LLM 能消费的 embedding 空间，典型做法是冻结视觉编码器和 LLM。第二阶段使用视觉指令数据继续训练，让模型学习视觉问答、多轮对话和指令遵循，优化范围可以包含 projector 与 LLM 的部分或全部参数，具体取决于版本。第一阶段解决“视觉特征怎么接进来”，第二阶段解决“接入后怎样按指令回答”，不能把两者都概括成普通图文对齐。

### LLaVA、BLIP-2 和 Flamingo 的连接方式有什么区别？

回答思路：按连接器、视觉 token 注入方式和主要能力比较。

回答模板：

LLaVA 通常用线性层或 MLP projector 把视觉 patch 特征映射成 LLM 输入 token，结构简单，主要依赖视觉指令微调形成对话能力。BLIP-2 使用 Q-Former，通过可学习 Query 从冻结视觉编码器提取与语言相关的紧凑表示，再桥接冻结 LLM。Flamingo 使用 Perceiver Resampler 压缩视觉特征，并在语言模型层间插入 gated cross-attention，重点支持交错图文的 few-shot 上下文。三者都复用预训练视觉和语言主干，但信息压缩方式、注入位置和训练目标不同。

### LLaVA 适合哪些场景，主要局限是什么？

回答思路：先说明低成本视觉对话价值，再从视觉分辨率、连接器、数据与语言先验分析失败模式。

回答模板：

LLaVA 适合视觉问答、图片描述、多轮图文对话以及开源多模态研究，因为它用较简单的视觉编码器、projector 和 LLM 组合就能获得视觉指令能力。它的主要局限是视觉细节会受输入分辨率、视觉特征层和 token 数限制，简单 projector 的信息选择能力也弱于显式查询或多层 Cross-Attention；如果指令数据有偏差，LLM 还可能凭语言先验产生幻觉。因此 OCR、小目标、计数、空间关系和精确 grounding 等场景需要单独评测，不能只看通用对话表现。
