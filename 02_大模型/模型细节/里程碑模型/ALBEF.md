# ALBEF

## 知识点解析

### 概述

ALBEF 是视觉语言预训练模型，核心思想是 Align before Fuse：先用图文对比学习对齐图像和文本表示，再通过跨模态 encoder 融合两种模态，并用动量蒸馏缓解 web 图文数据噪声。

### 解决的问题

ALBEF 主要解决 CLIP 式双塔模型和早期融合模型的不足：

- 只做全局图文对比，细粒度理解不足。
- 直接融合 noisy web 图文对，容易被弱匹配或错误 caption 干扰。
- 下游 VQA、NLVR、图文检索等任务需要跨模态交互，而不只是两个 embedding 相似。

### 典型模型规格

ALBEF 常见公开实现：

| 组件 | 典型配置 |
| --- | --- |
| 视觉编码器 | ViT-B/16，12 层，hidden size 768，12 heads |
| 文本编码器 | BERT-base 前 6 层，hidden size 768，12 heads |
| 多模态融合编码器 | BERT-base 后 6 层改造，加入 cross-attention |
| 动量模型 | image encoder、text encoder、multimodal encoder 的 EMA 版本 |
| 主要预训练目标 | ITC、ITM、MLM |

不同代码实现可能在初始化 checkpoint、image resolution、queue size 上有差异，面试时重点讲清“ViT + BERT split + multimodal encoder + momentum distillation”。

### 完整架构

```text
image
  -> ViT image encoder
  -> visual embeddings

text
  -> BERT text encoder first 6 layers
  -> text embeddings

stage 1: contrastive alignment
  -> image-text contrastive loss

stage 2: multimodal fusion
  -> BERT multimodal encoder last 6 layers
  -> self-attention over text tokens
  -> cross-attention to visual embeddings

task heads
  -> ITM head
  -> MLM head
  -> downstream heads

momentum encoders
  -> generate soft pseudo targets
  -> maintain feature queues
```

### 训练目标

ITC：Image-Text Contrastive Learning。

- 拉近匹配图文对。
- 推远 batch/queue 中不匹配图文。
- 用于“align before fuse”的对齐阶段。

ITM：Image-Text Matching。

- 输入图文融合表示。
- 判断图像和文本是否匹配。
- 通常使用 hard negative 提升判别能力。

MLM：Masked Language Modeling。

- mask 文本 token。
- 结合图像信息恢复 token。
- 迫使模型学习跨模态融合表示。

Momentum Distillation：

- 维护动量模型作为 teacher。
- 生成 soft labels，降低 noisy caption 对训练的干扰。
- 类似 MoCo 的队列思想扩展负样本。

### 做了什么改变

相比 CLIP：

- CLIP 是双塔，只做图文 embedding 对齐。
- ALBEF 在对齐后增加 cross-modal fusion，可以建模 token 和 patch 的交互。
- ALBEF 有 ITM/MLM，更适合 VQA、NLVR、图文推理等理解任务。

相比直接 early fusion：

- ALBEF 先对齐再融合，减少 noisy web data 下直接融合的难度。
- 动量蒸馏提供软目标，缓解图文弱相关问题。

### 在对抗攻击中的意义

ALBEF 同时有图文对比和融合模块，很适合验证多模态攻击：

- 攻击 ITC 可以破坏全局图文对齐。
- 攻击 ITM 可以破坏图文匹配判断。
- 攻击 MLM 或融合表示可以影响细粒度语言理解。
- 作为源模型时，能测试扰动是否迁移到双塔和融合型 VLM。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | ALBEF 的 image/text/multimodal encoder 怎么组成 | ViT-B/16 + BERT 前 6 层 + BERT 后 6 层 cross-attention |
| 目标题 | ITC、ITM、MLM 分别做什么 | 对齐、匹配、跨模态补词 |
| 对比题 | ALBEF 和 CLIP 区别 | CLIP 双塔对比，ALBEF 先对齐再融合 |
| 噪声题 | Momentum distillation 为什么有用 | soft target 缓解 noisy web caption |

### 易错点

- 只说 ALBEF 是 CLIP 加 BERT，不讲 fusion encoder。
- 忽略 Align before Fuse 的顺序。
- 把 ITC 和 ITM 混淆：ITC 是 embedding 对比，ITM 是融合后匹配分类。
- 不讲动量模型，就少了 ALBEF 处理 noisy data 的关键。

## 面试应对

### ALBEF 的完整架构是什么？

回答思路：按 image encoder、text encoder、multimodal encoder、训练目标回答。

回答模板：

ALBEF 的核心是 Align before Fuse。典型实现中，图像侧使用 ViT-B/16，12 层、hidden size 768、12 heads；文本侧使用 BERT-base 的前 6 层作为 text encoder；融合侧使用 BERT-base 后 6 层改造成 multimodal encoder，通过 cross-attention 让文本 token 关注视觉特征。训练时先用 ITC 做图文对比对齐，再用 ITM 判断图文是否匹配，用 MLM 在图像条件下恢复 masked token。ALBEF 还维护动量模型生成 soft target，缓解 web 图文数据噪声。
