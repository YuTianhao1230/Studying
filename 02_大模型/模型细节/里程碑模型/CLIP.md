# CLIP

## 知识点解析

### 概述

[CLIP](<../../../06_视觉多模态与生成模型/多模态模型/CLIP.md>) 是双塔图文对比学习模型，用图像编码器和文本编码器把图像、文本映射到同一 embedding 空间，通过大规模图文对训练获得 zero-shot 分类、图文检索和跨模态迁移能力。

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

CLIP 的关键是**双塔编码、后融合**：

```text
图像 -> image encoder -> image projection -> image embedding
文本 -> text encoder  -> text projection  -> text embedding

image embedding 和 text embedding 只在最后计算相似度，
中间没有 cross-attention，也没有 token-level early fusion。
```

所以 CLIP 的图像编码和文本编码可以分别离线计算、建立向量索引，在线只需要对 query 做一次编码和相似度搜索。这也是它适合图文检索和大规模 zero-shot 分类的原因。

### 视觉分支

CLIP 公开模型有两类视觉编码器。

Modified ResNet：

- 基于 ResNet-50、ResNet-101，也有更宽的 RN50x4、RN50x16、RN50x64。
- 做了三类关键改造：
  1. ResNet stem 使用三个较小卷积替代单个大卷积。
  2. 下采样使用 anti-aliased blur pooling，减少平移敏感。
  3. 最后不用全局平均池化，而是 attention pooling，让图像区域按注意力聚合。

Vision [Transformer](<../../基础架构/Transformer.md>)：

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

这里容易被问：CLIP 文本分支不是 [BERT](<BERT.md>)。它更接近 [GPT](<GPT.md>) 式 causal Transformer，但目标不是生成，而是为整句文本输出 embedding。

### CLIP 的典型配置怎么记

CLIP 不是一个只有单一尺寸的 checkpoint，面试时要区分“论文中的典型配置”和“所有模型都固定如此”：

| 分支 | 典型配置 | 输出用途 |
| --- | --- | --- |
| Image encoder | ResNet-50/101 或 ViT-B/32、ViT-B/16、ViT-L/14 | 生成全局 image embedding |
| ViT-B/32 | 12 层、hidden size 768、12 heads，patch size 32 | 计算成本较低 |
| ViT-L/14 | 24 层、hidden size 1024、16 heads，patch size 14 | 表达能力和 zero-shot 效果更强 |
| Text encoder | 12 层、hidden size 512、8 heads、context length 77 | 生成全局 text embedding |
| Projection | image/text 各自一个线性投影 | 把两种模态映射到相同维度 |

这里的“12 层文本 Transformer”是 OpenAI CLIP 常见配置，不代表所有后续 CLIP-like 模型都使用相同层数。

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

可以把目标写成：

```text
L_i2t = CrossEntropy(logits_per_image, target=[0, 1, ..., N-1])
L_t2i = CrossEntropy(logits_per_text, target=[0, 1, ..., N-1])
L_clip = (L_i2t + L_t2i) / 2
```

其中 `logits[i][j]` 表示第 `i` 张图和第 `j` 条文本的相似度，正确配对在对角线上。温度参数通常是可学习的 `logit_scale`，它控制 softmax 分布的尖锐程度：

- 温度太高或 `logit_scale` 太大，模型会过度强调最相似项，训练可能变得不稳定。
- 温度太低或 `logit_scale` 太小，对比信号变弱，正负样本难以拉开。

CLIP 的负样本主要来自 batch 内其他图文对，因此 batch size 和跨卡 all-gather 会直接影响负样本数量和对比学习质量。

### CLIP 为什么能从 noisy web data 学到能力

CLIP 使用的是互联网图文对，不要求每条文本都是严格人工标注的类别标签。大规模数据提供了丰富概念，但也带来错配、重复、广告和偏见。

它能工作的原因主要是：

1. 数据规模足够大，弱配对噪声在整体上可以被统计规律抵消。
2. 对比目标让图像和文本必须共享可区分的语义特征。
3. 大 batch 提供大量负样本，迫使模型学习更细的概念边界。
4. 文本 prompt 把开放词表类别映射到同一个语言空间。

但这不等于 CLIP 自动解决了数据质量问题。领域迁移、中文 prompt、细粒度 UI、计数和空间关系仍然可能明显掉点。

### 做了什么改变

相比传统图像分类：

- 不用固定类别分类头，而是用文本 prompt 表达类别。
- 用图文对比学习替代人工类别监督。
- 学到开放词表的视觉语义空间。

相比单塔 [VLM](<../../../06_视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>)：

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

工程上通常会把类别文本 embedding 预先算好并缓存：

```text
离线：
  class name + prompt template -> text embeddings -> cache

在线：
  image -> image embedding
  -> 与缓存的 class embeddings 做矩阵相似度
  -> 取最大值类别
```

图文检索也使用同一套 embedding：

```text
文本检索图片：text embedding -> ANN image index
图片检索文本：image embedding -> ANN text index
```

### 在多模态模型中的作用

CLIP 的视觉编码器和图文 embedding 空间影响很大：

- [LLaVA](<LLaVA.md>) 等模型常用 CLIP [ViT](<ViT.md>) 作为视觉编码器。
- 图文检索、开放词表分类、grounding 和 VLM 评测常用 CLIPScore。
- [多模态对抗攻击](<../../../13_AI安全与对抗攻击/多模态对抗攻击.md>)常攻击 CLIP 对齐空间，因为它代表图文语义绑定。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | CLIP 图像和文本分支分别是什么 | 图像是 Modified ResNet/ViT，文本是 12 层 causal Transformer |
| 目标题 | CLIP loss 怎么写 | batch 内 `N x N` 图文相似度，双向 cross entropy |
| 应用题 | CLIP 如何 zero-shot 分类 | 类别名写成 prompt，与图像 embedding 做相似度 |
| 对比题 | CLIP 和 [ALBEF](<ALBEF.md>) 区别 | CLIP 双塔对比，ALBEF 先对齐再融合 |
| 项目题 | 为什么攻击 CLIP 有意义 | CLIP 对齐空间是很多 VLM 的共享基础 |

### 易错点

- 把 CLIP 说成图像 caption 模型。CLIP 不直接生成文本。
- 把 CLIP 文本编码器说成 BERT。它是 Transformer 文本 encoder，但使用 causal mask，取 `[EOS]` 表示。
- 只说“大规模图文对”，不讲双向对比损失。
- 忽略 prompt template 对 zero-shot 分类结果的影响。
- 把 CLIP 的相似度矩阵说成逐样本二分类；它通常是 batch 内 N 类的双向交叉熵。
- 把 CLIP 的全局 embedding 能力等同于细粒度视觉定位能力。

## 面试应对

### CLIP 的完整架构是什么？

回答思路：按图像分支、文本分支、投影、对比损失回答。

回答模板：

CLIP 是双塔图文对比模型。图像分支可以是改造版 ResNet 或 ViT：ResNet 版本改了 stem、加入 anti-aliased pooling，并用 attention pooling 替代平均池化；ViT 版本如 ViT-B/32、ViT-B/16、ViT-L/14，把图像切成 patch token 后用 Transformer Encoder 编码。文本分支是 Transformer，典型配置是 12 层、hidden size 512、8 heads，BPE 词表约 49K，context length 是 77，取 `[EOS]` token 表示文本。图像和文本分别投影并归一化到同一 embedding 空间，在 batch 内计算图文相似度矩阵，用 image-to-text 和 text-to-image 的双向对比损失训练。

### CLIP 的对比损失怎么计算？

回答思路：用一个 batch 的 N 个图文对说明 N×N 相似度矩阵、对角线正样本和双向交叉熵。

回答模板：

一个 batch 里有 N 个图文对，图像编码器得到 N 个 image embeddings，文本编码器得到 N 个 text embeddings，归一化后做矩阵乘法得到 N×N 的相似度矩阵，再除以可学习的温度参数。第 i 行第 i 列是正确图文配对，其余位置是 batch 内负样本。然后分别对 image-to-text 和 text-to-image 计算交叉熵，最后取两者平均。CLIP 的核心不是逐对二分类，而是利用 batch 内其他样本构造多分类对比目标。

### CLIP 为什么适合图文检索，但不直接适合生成？

回答思路：区分“输出固定维度 embedding”和“自回归生成 token”。

回答模板：

CLIP 的图像和文本分支分别编码，最后只输出固定维度 embedding，再通过余弦相似度判断是否匹配。这个结构很适合离线建立向量索引，做图搜文、文搜图和 zero-shot 分类，但它没有 decoder 和 next-token prediction 目标，所以不会直接生成 caption 或回答问题。如果要生成文本，需要把 CLIP 视觉特征通过 projector、Q-Former 等桥接模块接入语言模型，再进行视觉指令微调。
