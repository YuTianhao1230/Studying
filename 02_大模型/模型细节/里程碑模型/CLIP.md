# CLIP

## 知识点解析

### 概述

CLIP 是双塔图文对比学习模型，用图像编码器和文本编码器把图像、文本映射到同一 embedding 空间，通过大规模图文对训练获得 zero-shot 分类、图文检索和跨模态迁移能力。

### 解决的问题

CLIP 解决传统视觉监督学习的三个问题：

- 类别封闭：ImageNet 分类器只能预测固定类别。
- 标注成本高：人工类别标签覆盖不了开放世界概念。
- 迁移弱：换任务常需要重新训练分类头。

CLIP 把自然语言当作开放类别空间，用文本 prompt 表达类别，使视觉模型可以用自然语言做 zero-shot 分类。

### 完整架构

```text
image
  -> image encoder: Modified ResNet or Vision Transformer
  -> image feature
  -> linear projection
  -> normalized image embedding

text prompt
  -> BPE tokenizer, context length 77
  -> text Transformer
  -> [EOS] token representation
  -> linear projection
  -> normalized text embedding

image embeddings x text embeddings
  -> cosine similarity matrix / temperature scaling
  -> symmetric contrastive loss
```

### 视觉分支

CLIP 公开模型有两类视觉编码器。

Modified ResNet：

- 基于 ResNet-50、ResNet-101，也有更宽的 RN50x4、RN50x16、RN50x64。
- 做了三类关键改造：
  1. ResNet stem 使用三个较小卷积替代单个大卷积。
  2. 下采样使用 anti-aliased blur pooling，减少平移敏感。
  3. 最后不用全局平均池化，而是 attention pooling，让图像区域按注意力聚合。

Vision Transformer：

- 常见公开版本包括 ViT-B/32、ViT-B/16、ViT-L/14。
- ViT-B 通常是 12 层、hidden size 768、12 heads。
- ViT-L/14 通常是 24 层、hidden size 1024、16 heads。
- 图像被切成 patch token，经过 Transformer Encoder 得到全局图像表示。

### 文本分支

CLIP 文本编码器是 Transformer：

- tokenizer：lower-cased BPE。
- vocabulary size：约 49K。
- context length：77 tokens。
- 典型文本 Transformer：12 层、hidden size 512、8 heads。
- 使用 causal self-attention mask。
- 取 `[EOS]` token 的 hidden state 作为文本句子表示，再投影到共享 embedding 空间。

这里容易被问：CLIP 文本分支不是 BERT。它更接近 GPT 式 causal Transformer，但目标不是生成，而是为整句文本输出 embedding。

### 训练目标

给定 batch 内 `N` 个图文对，计算 `N x N` 相似度矩阵：

```text
logits = image_embeddings @ text_embeddings.T / temperature
```

目标是让第 `i` 张图和第 `i` 句文本相似度最高：

- image-to-text cross entropy
- text-to-image cross entropy
- 两者取平均

这就是双向 InfoNCE / contrastive loss。

### 做了什么改变

相比传统图像分类：

- 不用固定类别分类头，而是用文本 prompt 表达类别。
- 用图文对比学习替代人工类别监督。
- 学到开放词表的视觉语义空间。

相比单塔 VLM：

- CLIP 不做 early fusion，图像和文本分开编码。
- 适合检索和 zero-shot，但不直接生成自然语言答案。

### Zero-shot 分类怎么做

```text
image -> image embedding
class names -> prompt templates -> text embeddings
choose class with max cosine similarity
```

例如类别 `dog` 可以写成：

```text
a photo of a dog
```

使用多个 prompt template 做 ensemble，通常能提升 zero-shot 稳定性。

### 在多模态模型中的作用

CLIP 的视觉编码器和图文 embedding 空间影响很大：

- LLaVA 等模型常用 CLIP ViT 作为视觉编码器。
- 图文检索、开放词表分类、grounding 和 VLM 评测常用 CLIPScore。
- 多模态对抗攻击常攻击 CLIP 对齐空间，因为它代表图文语义绑定。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | CLIP 图像和文本分支分别是什么 | 图像是 Modified ResNet/ViT，文本是 12 层 causal Transformer |
| 目标题 | CLIP loss 怎么写 | batch 内 `N x N` 图文相似度，双向 cross entropy |
| 应用题 | CLIP 如何 zero-shot 分类 | 类别名写成 prompt，与图像 embedding 做相似度 |
| 对比题 | CLIP 和 ALBEF 区别 | CLIP 双塔对比，ALBEF 先对齐再融合 |
| 项目题 | 为什么攻击 CLIP 有意义 | CLIP 对齐空间是很多 VLM 的共享基础 |

### 易错点

- 把 CLIP 说成图像 caption 模型。CLIP 不直接生成文本。
- 把 CLIP 文本编码器说成 BERT。它是 Transformer 文本 encoder，但使用 causal mask，取 `[EOS]` 表示。
- 只说“大规模图文对”，不讲双向对比损失。
- 忽略 prompt template 对 zero-shot 分类结果的影响。

## 面试应对

### CLIP 的完整架构是什么？

回答思路：按图像分支、文本分支、投影、对比损失回答。

回答模板：

CLIP 是双塔图文对比模型。图像分支可以是改造版 ResNet 或 ViT：ResNet 版本改了 stem、加入 anti-aliased pooling，并用 attention pooling 替代平均池化；ViT 版本如 ViT-B/32、ViT-B/16、ViT-L/14，把图像切成 patch token 后用 Transformer Encoder 编码。文本分支是 Transformer，典型配置是 12 层、hidden size 512、8 heads，BPE 词表约 49K，context length 是 77，取 `[EOS]` token 表示文本。图像和文本分别投影并归一化到同一 embedding 空间，在 batch 内计算图文相似度矩阵，用 image-to-text 和 text-to-image 的双向对比损失训练。
