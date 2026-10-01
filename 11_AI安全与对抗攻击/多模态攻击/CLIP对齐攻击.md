# CLIP 对齐攻击

## 知识点解析

### 概述

[CLIP](<../../02_大模型/模型细节/里程碑模型/CLIP.md>) 对齐攻击针对图像和文本共享 embedding 空间，通过扰动图像或文本破坏正确图文对的相似度，或提高错误图文对的相似度。

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
5. 迁移到 [ALBEF](<../../02_大模型/模型细节/里程碑模型/ALBEF.md>)、[TCL](<../../02_大模型/模型细节/里程碑模型/TCL.md>)、[BLIP](<../../02_大模型/模型细节/里程碑模型/BLIP与BLIP2.md>)、[LLaVA](<../../02_大模型/模型细节/里程碑模型/LLaVA.md>) 或商业 MLLM 观察效果。

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
| 机制题 | 为什么攻击 CLIP 有意义 | CLIP 对齐空间是很多 [VLM](<../../02_大模型/视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>) 的基础 |
| 公式题 | 对齐攻击 loss 怎么写 | 降低正对相似度，提高负对相似度 |
| 任务题 | 图文检索攻击怎么评估 | Recall@K、正确样本 rank 下降 |
| 项目题 | Syner-Attack 的 alignment loss 是什么作用 | 破坏图文共享空间，而非只攻击视觉分类 |

### 易错点

- 把 CLIP 攻击写成普通 ImageNet 分类攻击。
- 只看相似度下降，不看检索排序或任务级指标。
- 正负样本选择不清楚，导致攻击目标不可复现。
- 把 CLIPScore 下降直接等同于 MLLM 失败，缺少下游验证。

## 面试应对

### 1. CLIP 对齐攻击的目标函数如何设计？

回答思路：先回顾归一化图文 embedding 和余弦相似度，再区分降低正对相似度、提高负对相似度和排序损失。

回答模板：

> CLIP 把图像和文本编码为归一化 embedding，并用余弦相似度衡量匹配。非定向对齐攻击可以最小化 `cos(E_I(x_adv),E_T(t_pos))`，使正确图文对分离；定向攻击可以最大化与错误文本 `t_neg` 的相似度；检索场景还可用 margin 或 ranking loss，让负样本分数超过正样本。优化时必须说明正负样本构造、攻击哪一模态以及扰动预算，否则目标不可复现。

### 2. CLIP 对齐攻击与普通分类攻击有什么不同，适用边界是什么？

回答思路：从攻击对象、输出和指标比较，指出对齐空间受损是下游风险信号，但不等于所有 MLLM 任务必然失败。

回答模板：

> 分类攻击针对固定类别 logits，通常用 accuracy 或 ASR 评估；CLIP 对齐攻击针对图文共享空间，关注正负图文相似度、Recall@K 和正确匹配排名。它更适合评估检索、zero-shot 分类或依赖 CLIP-like 表征的多模态系统。对齐分数下降只能证明代理表征受损，不能直接推出 caption、VQA 或商业 MLLM 一定失败，因为下游模型还可能有融合模块和生成模型纠错，所以必须补充真实任务指标。

### 3. 如何设计可信的 CLIP 对齐攻击实验？

回答思路：明确威胁模型和负样本，先验证源模型目标，再评估 hold-out VLM/任务，并用任务指标与消融区分代理 loss 改善。

回答模板：

> 我会固定图像扰动范数或文本编辑预算，明确正样本、随机负样本还是难负样本，并只在 clean 检索正确的样本上统计攻击。源模型报告相似度 margin、Recall@K 和 rank change，迁移实验使用未参与优化的 VLM，同时在 zero-shot 分类、检索或 VQA 等下游任务上报告原生指标。还要比较只降低正对相似度、只提高负对相似度和联合目标，避免仅凭优化 loss 下降宣称攻击有效。
