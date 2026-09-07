# 联合图文攻击与 Syner-Attack

## 知识点解析

### 概述

联合图文攻击同时扰动图像和文本，目标是破坏多模态模型的视觉表征、文本语义和图文对齐关系。Syner-Attack 可以从协同双流攻击和跨模态对齐破坏角度理解。

### 解决的问题

单模态攻击存在互补纠错问题：

- 只攻击图像时，文本仍可能提供正确语义。
- 只攻击文本时，图像仍可能纠正文本噪声。
- 多模态模型依赖图文互证，攻击单个模态不一定足够。

联合攻击的目标是让两个模态同时偏离，并破坏它们在共享语义空间中的绑定关系。

### 基本目标

```text
L_total = lambda_img * L_image_feature
        + lambda_align * L_image_text_alignment
        + lambda_txt * L_text_attack
```

典型分支：

- 图像特征扰动：让视觉表征偏离 clean image。
- 图文对齐扰动：降低正确图文相似度或匹配分数。
- 文本扰动：替换对图文对齐更关键的词。

### Syner-Attack 的表达框架

可以按四层讲：

1. 问题：VLM/MLLM 的黑盒迁移攻击仍不稳定。
2. 假设：多模态模型依赖视觉表征和图文对齐。
3. 方法：图像侧 feature/alignment loss + 文本侧 visual-guided attack。
4. 证据：image-only、text-only、joint、VGA、alignment loss、防御和 MLLM ASR 消融。

### 视觉引导文本攻击

视觉引导文本攻击不是随机替换词，而是结合图像和文本对齐关系选择词：

```text
word importance = language importance + visual relevance
```

优先替换：

- 物体词。
- 属性词。
- 动作词。
- 关系词。
- 数量词。

替换后仍需语义保持过滤，避免通过改变原始 caption 语义来“攻击成功”。

### 如何证明不是简单拼接

需要设计消融：

| 消融 | 证明什么 |
| --- | --- |
| image-only | 图像分支单独贡献 |
| text-only | 文本分支单独贡献 |
| image + text | 联合是否有增益 |
| w/o alignment loss | 图文对齐损失是否必要 |
| w/o VGA | 视觉引导文本攻击是否有效 |
| 不同预算 | 增益是否只来自更大扰动 |
| 防御下评估 | 扰动是否只依赖高频噪声 |

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 动机题 | 为什么同时攻击图文 | 单模态可能被另一模态纠正 |
| 公式题 | 联合 loss 怎么写 | feature + alignment + text attack |
| 创新题 | 如何回应拼接质疑 | 用机制解释和消融证据 |
| 约束题 | 文本扰动如何公平 | 替换率、语义相似度、实体关系保护 |
| 评估题 | MLLM 怎么评估 | 固定 prompt、ASR 规则、人审或 LLM judge |

### 易错点

- baseline 只能改图，自己改图又改文，却直接比较 ASR。
- 文本扰动改变 caption 语义，还声称语义保持。
- 联合攻击没有 image-only/text-only 消融。
- 只展示定性案例，没有样本量和 ASR。

## 面试应对

### Syner-Attack 的核心怎么讲？

回答思路：承认基础组件，强调跨模态机制和证据链。

回答模板：

Syner-Attack 的核心不是单独发明一个新的梯度算子，而是针对 VLM/MLLM 的跨模态对齐脆弱性，把图像侧特征扰动、图文对齐扰动和视觉引导文本攻击组织成协同双流框架。图像分支负责扰乱视觉表征和图文相似度，文本分支在语义保持约束下替换对图文匹配更关键的词。为了证明它不是简单拼接，需要用 image-only、text-only、joint、去掉 VGA、去掉 alignment loss 等消融，以及跨模型、MLLM 和防御下 ASR 来支撑。
