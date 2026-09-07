# ViT

## 知识点解析

### 概述

ViT 是把图像切成 patch token 后送入 Transformer Encoder 的视觉模型，它证明了在足够大规模预训练下，视觉任务也可以使用与语言模型高度一致的 Transformer 架构。

### 解决的问题

ViT 解决的是视觉模型长期依赖 CNN 归纳偏置的问题：

- CNN 强依赖局部卷积和平移等变，长距离关系需要堆很多层。
- 视觉和语言模型架构不统一，多模态连接成本较高。
- 大规模数据下，Transformer 的全局建模能力可能超过传统 CNN。

### 典型模型规格

| 版本 | patch size | 层数 | hidden size | heads | MLP size |
| --- | --- | --- | --- | --- | --- |
| ViT-Base | 16 或 32 | 12 | 768 | 12 | 3072 |
| ViT-Large | 16 或 32 | 24 | 1024 | 16 | 4096 |
| ViT-Huge | 14 | 32 | 1280 | 16 | 5120 |

常见写法如 `ViT-B/16` 表示 Base 规模、patch size 为 16。

### 完整架构

```text
image: H x W x C
  -> split into P x P patches
  -> flatten each patch
  -> linear projection to patch embeddings
  -> prepend [CLS] token
  -> add learnable position embeddings
  -> Transformer Encoder block x N
       -> multi-head self-attention
       -> MLP
       -> residual + LayerNorm
  -> [CLS] representation
  -> classification head
```

patch 数量：

```text
N = (H / P) * (W / P)
```

例如 `224 x 224` 图像使用 `16 x 16` patch，会得到 `14 x 14 = 196` 个 patch token，再加一个 `[CLS]` token。

### 做了什么改变

相比 CNN：

- CNN 用卷积核滑动提取局部特征。
- ViT 把图像 patch 当成 token，用 self-attention 做全局交互。
- CNN 的 inductive bias 更强，小数据更稳。
- ViT 更依赖大规模预训练，但与语言 Transformer 更容易统一。

相比原始 Transformer：

- 输入不是词 token，而是图像 patch。
- 通常只用 Encoder，不用 Decoder。
- 使用 learnable `[CLS]` token 汇聚整图信息。

### 训练与迁移

ViT 原论文强调大规模预训练的重要性。在小数据上从头训练，ViT 可能不如 ResNet；但在 JFT 等大规模数据上预训练后，再迁移到 ImageNet 等任务，效果很强。

常见训练策略：

- 图像分类监督预训练。
- MAE/BEiT/DINO 等自监督预训练。
- CLIP/SigLIP 图文对比预训练。

### 在多模态模型中的作用

ViT 很适合作为视觉编码器：

```text
image -> ViT patch tokens -> projector/Q-Former/cross-attention -> LLM
```

原因：

- 输出 token 序列，天然能接 Transformer 语言模型。
- patch token 可保留空间信息。
- 与 CLIP 结合后可获得强图文对齐能力。

LLaVA、BLIP-2、Qwen-VL、GPT-4V 类模型都可以从 ViT/CLIP-ViT 视觉编码器路线理解。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | ViT 怎么处理图像 | patch 切分、线性投影、位置编码、Transformer Encoder |
| 计算题 | patch token 数量 | `N=(H/P)*(W/P)` |
| 对比题 | ViT 和 CNN 区别 | 局部卷积 vs 全局 attention，归纳偏置和数据规模 |
| 多模态题 | 为什么 VLM 常用 ViT | token 输出形式和语言 Transformer 更容易连接 |

### 易错点

- 只说 ViT 用 Transformer，不讲图像如何变成 token。
- 忽略 patch size 对 token 数和计算量的影响。
- 认为 ViT 一定比 CNN 好；小数据场景 CNN 可能更稳。
- 把 `[CLS]` token 和图像 patch token 混淆。

## 面试应对

### ViT 的完整架构是什么？

回答思路：从 patch tokenization 到 Transformer Encoder 再到分类头。

回答模板：

ViT 会先把图像切成固定大小的 patch，比如 `224x224` 图像用 `16x16` patch 会得到 196 个 patch。每个 patch 展平后经过线性投影变成 token embedding，再加上一个 `[CLS]` token 和可学习位置编码，送入多层 Transformer Encoder。ViT-Base 通常是 12 层、hidden size 768、12 个 heads，ViT-Large 是 24 层、hidden size 1024、16 个 heads。最后用 `[CLS]` 表示接分类头。它的意义是把视觉建模转成 token 序列建模，也为后续 VLM 连接语言模型打下基础。
