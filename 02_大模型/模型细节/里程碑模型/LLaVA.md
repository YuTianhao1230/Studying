# LLaVA

## 知识点解析

### 概述

LLaVA 是开源视觉指令模型路线的代表，核心是用 CLIP 视觉编码器提取图像特征，通过 projector 对齐到 LLM embedding 空间，再用视觉指令数据微调大语言模型。

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

相比 Flamingo：

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
