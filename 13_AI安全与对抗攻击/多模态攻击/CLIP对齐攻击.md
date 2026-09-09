# CLIP 对齐攻击

## 知识点解析

### 概述

[CLIP](<../../06_视觉多模态与生成模型/多模态模型/CLIP.md>) 对齐攻击针对图像和文本共享 embedding 空间，通过扰动图像或文本破坏正确图文对的相似度，或提高错误图文对的相似度。

### 解决的问题

CLIP 和大量 VLM/MLLM 使用图文对齐空间作为视觉语义基础。如果攻击能破坏 CLIP 对齐，就可能影响图文检索、zero-shot 分类、caption 评估和下游多模态模型。

### 模型结构回顾

```text
image -> image encoder -> normalized image embedding
text  -> text encoder  -> normalized text embedding
similarity = cosine(image_embedding, text_embedding)
```

攻击目标不一定是分类 logit，而是图文相似度和排序关系。

### 攻击目标

降低正确图文对相似度：

```text
minimize cosine(E_I(x_adv), E_T(t_pos))
```

提高错误图文对相似度：

```text
maximize cosine(E_I(x_adv), E_T(t_neg))
```

检索任务中还可以直接优化 ranking loss，让正确 caption 或正确 image 排名下降。

### 攻击流程

1. 固定文本或图像，选择源 CLIP/VLP 模型。
2. 对图像输入 `x` 求图文相似度损失梯度。
3. 在扰动预算内更新图像。
4. 用 Recall@K、rank drop、ASR 测试图文检索下降。
5. 迁移到 [ALBEF](<../../02_大模型/模型细节/里程碑模型/ALBEF.md>)、[TCL](<../../02_大模型/模型细节/里程碑模型/TCL.md>)、[BLIP](<../../06_视觉多模态与生成模型/多模态模型/BLIP.md>)、[LLaVA](<../../02_大模型/模型细节/里程碑模型/LLaVA.md>) 或商业 MLLM 观察效果。

### 和分类攻击的区别

| 对比项 | 分类攻击 | CLIP 对齐攻击 |
| --- | --- | --- |
| 输出 | 类别 logits | 图文 embedding 相似度 |
| 目标 | 真实类错误 | 正确图文不匹配或错误图文匹配 |
| 评价 | accuracy / ASR | Recall@K、rank、matching score |
| 迁移意义 | 分类器鲁棒性 | 跨模态语义对齐鲁棒性 |

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 机制题 | 为什么攻击 CLIP 有意义 | CLIP 对齐空间是很多 [VLM](<../../06_视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>) 的基础 |
| 公式题 | 对齐攻击 loss 怎么写 | 降低正对相似度，提高负对相似度 |
| 任务题 | 图文检索攻击怎么评估 | Recall@K、正确样本 rank 下降 |
| 项目题 | Syner-Attack 的 alignment loss 是什么作用 | 破坏图文共享空间，而非只攻击视觉分类 |

### 易错点

- 把 CLIP 攻击写成普通 ImageNet 分类攻击。
- 只看相似度下降，不看检索排序或任务级指标。
- 正负样本选择不清楚，导致攻击目标不可复现。
- 把 CLIPScore 下降直接等同于 MLLM 失败，缺少下游验证。

## 面试应对

### CLIP 对齐攻击怎么做？

回答思路：先讲 CLIP 共享空间，再讲正负图文相似度目标。

回答模板：

CLIP 对齐攻击不是直接攻击分类头，而是攻击图像和文本的共享 embedding 空间。给定正确图文对，可以优化扰动让图像 embedding 远离正确文本 embedding，或者靠近错误文本 embedding。形式上就是降低 `cos(E_I(x_adv), E_T(t_pos))`，或提高 `cos(E_I(x_adv), E_T(t_neg))`。评估时不能只看 loss，要看图文检索 Recall@K、正确匹配排名下降，以及是否能迁移到其他 VLM 或 MLLM。
