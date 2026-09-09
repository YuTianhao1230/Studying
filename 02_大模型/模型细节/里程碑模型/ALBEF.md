# ALBEF

## 知识点解析

### 概述

ALBEF 是视觉语言预训练模型，核心思想是 Align before Fuse：先用图文对比学习对齐图像和文本表示，再通过跨模态 encoder 融合两种模态，并用动量蒸馏缓解 web 图文数据噪声。

### 解决的问题

ALBEF 主要解决 [CLIP](<../../../06_视觉多模态与生成模型/多模态模型/CLIP.md>) 式双塔模型和早期融合模型的不足：

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

不同代码实现可能在初始化 checkpoint、image resolution、queue size 上有差异，面试时重点讲清“[ViT](<ViT.md>) + [BERT](<BERT.md>) split + multimodal encoder + momentum distillation”。

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

ALBEF 的关键不是简单把图像和文本拼到一起，而是明确分成两个阶段：

```text
阶段一：单模态编码 + ITC
  image encoder 和 text encoder 分别输出 embedding
  -> 先把图文表示对齐到同一个语义空间

阶段二：跨模态融合 + ITM/MLM
  text tokens 作为主序列
  -> self-attention 建模文本
  -> cross-attention 读取 visual tokens
  -> 做图文匹配和视觉条件下的 masked language modeling
```

典型的 BERT split 可以这样记：

```text
BERT-base 12 layers
  -> 前 6 层：text encoder，输入纯文本
  -> 后 6 层：multimodal encoder，加入 cross-attention
```

这里的 cross-attention 通常以文本 hidden states 作为 Query，以图像 encoder 输出作为 Key/Value，因此 ALBEF 的融合是“文本查询视觉”，而不是 CLIP 那种最后只算一个全局相似度。

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

### ITC、ITM、MLM 怎么配合

三种目标不是重复训练，而是分别约束不同层次的能力：

| 目标 | 输入方式 | 主要学到什么 |
| --- | --- | --- |
| ITC | 图像 embedding + 文本 embedding，双塔 | 全局图文语义对齐，适合检索 |
| ITM | 图像和文本进入融合 encoder | 判断这对图文是否真正匹配，学习细粒度交互 |
| MLM | mask 文本 token，同时提供图像和剩余文本 | 根据视觉证据恢复文本，学习跨模态理解 |

#### ITC 与动量队列

ALBEF 维护 image/text encoder 的 momentum teacher，并把历史 batch 的 embedding 放入 queue。这样当前 batch 不变大的情况下，也能获得更多负样本：

```text
当前 batch：
  online image/text encoder -> 当前 embedding

历史 queue：
  momentum image/text encoder -> 稳定的历史 embedding

当前 embedding
  + queue embedding
  -> 更大的图文对比候选集合
```

动量更新可以抽象成：

```text
theta_m = m * theta_m + (1 - m) * theta
```

其中 `theta` 是在线模型参数，`theta_m` 是动量模型参数。动量模型变化更平滑，生成的 soft target 不容易因为当前 batch 噪声突然抖动。

#### ITM 与 hard negative

ITM 不只使用随机负样本，还会根据 ITC 相似度挑选 hard negative：

```text
图像 i
  -> 从相似度高但实际不匹配的文本中采样 hard negative

文本 t
  -> 从相似度高但实际不匹配的图像中采样 hard negative
```

这样可以让融合 encoder 区分“主题相似但细节不一致”的图文对，而不是只学会识别完全无关的负样本。

### 为什么 ALBEF 需要先对齐再融合

如果一开始就让图像 patch 和文本 token 直接做深度融合，web 图文对里的错配、弱配对和标题噪声会直接进入跨模态 attention，训练容易学到错误对应关系。

ALBEF 先通过 ITC 建立较稳定的全局语义空间，再用 ITM/MLM 做细粒度融合：

```text
先对齐：
  这张图大概和哪些文本语义相关？

再融合：
  这句话中的哪个词，是否真的对应图像中的哪个视觉内容？
```

这就是 Align before Fuse 的工程含义。

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
- 作为源模型时，能测试扰动是否迁移到双塔和融合型 [VLM](<../../../06_视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>)。

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
- 把 hard negative 说成随机负样本；ALBEF 会利用 ITC 相似度挑选更难的负例。
- 把 cross-attention 方向说反；典型结构是文本 Query 读取视觉 Key/Value。

## 面试应对

### ALBEF 的完整架构是什么？

回答思路：按 image encoder、text encoder、multimodal encoder、训练目标回答。

回答模板：

ALBEF 的核心是 Align before Fuse。典型实现中，图像侧使用 ViT-B/16，12 层、hidden size 768、12 heads；文本侧使用 BERT-base 的前 6 层作为 text encoder；融合侧使用 BERT-base 后 6 层改造成 multimodal encoder，通过 cross-attention 让文本 token 关注视觉特征。训练时先用 ITC 做图文对比对齐，再用 ITM 判断图文是否匹配，用 MLM 在图像条件下恢复 masked token。ALBEF 还维护动量模型生成 soft target，缓解 web 图文数据噪声。

### ALBEF 的 ITC、ITM、MLM 分别是什么？

回答思路：按全局对齐、融合匹配、视觉条件补词三个层次区分。

回答模板：

ITC 是 Image-Text Contrastive Learning，让匹配图文的全局 embedding 接近，主要服务于跨模态对齐和检索；ITM 是 Image-Text Matching，把图像和文本送进 multimodal encoder，判断它们是否真正匹配，重点学习细粒度交互；MLM 是 Masked Language Modeling，随机 mask 文本 token，同时给模型图像和剩余文本，让它根据视觉证据恢复被 mask 的词，训练跨模态理解。三者分别对应全局对齐、融合判别和视觉条件语言建模。

### ALBEF 的动量蒸馏和 hard negative 有什么作用？

回答思路：动量模型解决 noisy target 稳定性，hard negative 提高图文匹配判别难度。

回答模板：

ALBEF 用 momentum encoder 维护一个变化更平滑的 teacher，并通过 feature queue 保存历史图文 embedding，从而在不显著增大当前 batch 的情况下获得更多稳定负样本和 soft target。Hard negative 则根据 ITC 相似度挑选“语义相近但实际不匹配”的图文对，给 ITM 训练更有区分度的负例。前者主要缓解 web 图文噪声和训练抖动，后者主要提升模型区分细粒度错配的能力。

### ALBEF 和 CLIP 的核心区别是什么？

回答思路：抓住双塔后融合 vs 先对齐再跨模态融合，落到能力边界。

回答模板：

CLIP 是双塔模型，图像和文本分别编码，最后只在 embedding 层计算相似度，因此检索和 zero-shot 分类高效，但缺少 token-patch 级别的交互。ALBEF 先用 ITC 把图文全局表示对齐，再把文本 token 和视觉 token 送入带 cross-attention 的 multimodal encoder，并用 ITM、MLM 学习细粒度理解；同时用 momentum distillation 和 hard negative 处理 noisy web data。所以 CLIP 更偏高效全局对齐，ALBEF 更偏对齐后的跨模态理解和匹配。
