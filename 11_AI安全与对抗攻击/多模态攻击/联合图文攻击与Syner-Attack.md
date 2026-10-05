# 联合图文攻击与 Syner-Attack

## 知识点解析

### 概述

联合图文攻击同时扰动图像和文本，目标是破坏多模态模型的视觉表征、文本语义和图文对齐关系。Syner-Attack 采用先优化对抗图像、再执行离散文本替换的两阶段流程。

### 解决的问题

单模态攻击存在互补纠错问题：

- 只攻击图像时，文本仍可能提供正确语义。
- 只攻击文本时，图像仍可能纠正文本噪声。
- 多模态模型依赖图文互证，攻击单个模态不一定足够。

联合攻击的目标是让两个模态同时偏离，并破坏它们在共享语义空间中的绑定关系。

### 视觉优化目标与执行流程

$$
\mathcal{L}_{\mathrm{visual}}(I', T)
= \lambda \mathcal{L}_{\mathrm{feat}}(I', I)
+ (1 - \lambda)\mathcal{L}_{\mathrm{align}}(I', T),
\qquad 0 \le \lambda \le 1
$$

该目标只用于视觉阶段：

- `L_feat`：让对抗图像的视觉表征偏离干净图像。
- `L_align`：降低对抗图像与配对文本的跨模态对齐程度。
- `lambda`：平衡视觉特征破坏与图文对齐破坏。

VGA 在图像攻击之后执行离散词替换。可见代码流程是先调用 `Image_Attack` 得到 `adv_imgs`，再调用 `img_guided_attack` 生成 `adv_txts`。

### Syner-Attack 的表达框架

可以按四层讲：

1. 问题：VLM/MLLM 的[黑盒迁移攻击](<../迁移与通用攻击/黑盒迁移攻击.md>)仍不稳定。
2. 假设：多模态模型依赖视觉表征和图文对齐。
3. 方法：视觉阶段联合优化 $\lambda \mathcal{L}_{\mathrm{feat}} + (1-\lambda)\mathcal{L}_{\mathrm{align}}$，随后由 VGA 执行离散文本替换。
4. 证据：image-only、text-only、joint、VGA、alignment loss、防御和 MLLM ASR 消融。

### 视觉引导文本攻击

视觉引导文本攻击不是随机替换词，而是结合图像和文本对齐关系选择词：

$$
\begin{aligned}
S_{\mathrm{semantic\_norm}}(i)
&= \frac{S_{\mathrm{semantic}}(i)}{\max_j S_{\mathrm{semantic}}(j)}, \\
S_{\mathrm{visual\_norm}}(i)
&= \frac{S_{\mathrm{visual}}(i)}{\max_j S_{\mathrm{visual}}(j)}, \\
S_{\mathrm{VGA}}(i)
&= (1-\beta)S_{\mathrm{semantic\_norm}}(i)
+ \beta S_{\mathrm{visual\_norm}}(i).
\end{aligned}
$$

这里的最大值在当前候选词集合上计算；实现时要对分母为零的边界做保护。归一化用于消除两类分数的量纲和尺度差异，使 `beta` 表示可解释的融合权重。VGA 再按融合后的语言语义重要性和视觉相关性排序词，并从离散候选中选择替换词。优先替换：

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
| 完整两阶段流程 | 图像攻击后追加 VGA 是否有增益 |
| w/o alignment loss | 图文对齐损失是否必要 |
| w/o VGA | 视觉引导文本攻击是否有效 |
| 不同预算 | 增益是否只来自更大扰动 |
| 防御下评估 | 扰动是否只依赖高频噪声 |

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 动机题 | 为什么同时攻击图文 | 单模态可能被另一模态纠正 |
| 公式题 | 视觉阶段 loss 怎么写 | $\lambda \mathcal{L}_{\mathrm{feat}}+(1-\lambda)\mathcal{L}_{\mathrm{align}}$ |
| 创新题 | 如何回应拼接质疑 | 用机制解释和消融证据 |
| 约束题 | 文本扰动如何公平 | 替换率、语义相似度、实体关系保护 |
| 评估题 | MLLM 怎么评估 | 固定 prompt、ASR 规则、人审或 LLM judge |

### 易错点

- baseline 只能改图，自己改图又改文，却直接比较 ASR。
- 文本扰动改变 caption 语义，还声称语义保持。
- 联合攻击没有 image-only/text-only 消融。
- 只展示定性案例，没有样本量和 ASR。

## 面试应对

### 1. Syner-Attack 的优化目标和执行顺序是什么，协同体现在哪里？

回答思路：区分视觉阶段的双损失连续优化与后续 VGA 离散替换，说明两个阶段如何共同破坏视觉表征和跨模态对应关系。

回答模板：

> Syner-Attack 的可微联合目标属于视觉阶段，即 $\mathcal{L}_{\mathrm{visual}}=\lambda \mathcal{L}_{\mathrm{feat}}+(1-\lambda)\mathcal{L}_{\mathrm{align}}$。其中 `L_feat` 使对抗图像偏离干净视觉表征，`L_align` 削弱对抗图像与配对文本的对齐。视觉优化完成后，VGA 先分别对候选词的语义重要性和视觉相关性做最大值归一化，再按 `beta` 融合排序并执行离散候选替换。项目代码也按这个顺序先运行 `Image_Attack`，再运行 `img_guided_attack`。协同体现在双损失视觉优化与视觉引导文本替换按阶段配合，共同削弱视觉表征和图文对应关系。

### 2. 如何回应“Syner-Attack 只是图像攻击和文本攻击的拼接”？

回答思路：不回避基础组件，通过机制、交互消融和预算公平性建立证据链，避免引用未提供的论文数值。

回答模板：

> 我会把贡献表述为针对跨模态互补和对齐脆弱性的两阶段协同框架，而不是声称每个基础攻击算子都是新提出的。视觉阶段联合使用 `L_feat` 和 `L_align` 优化图像，随后 VGA 以视觉信息引导离散文本替换。证据上需要比较 image-only、text-only、仅双损失视觉攻击和完整两阶段流程，并分别去掉 alignment loss 与 VGA；只有在相同图像预算、文本预算和查询条件下观察到稳定增益，才能说明各组件存在互补作用。没有这样的消融时，不能夸大机制创新。

### 3. 联合攻击实验如何保证与单模态 baseline 公平，失败时怎么分析？

回答思路：按同威胁模型和能力边界分组比较，固定 prompt、模型和 Judge；按图像、文本、对齐、生成四层定位失败。

回答模板：

> 同时改图和文本的方法拥有更大的攻击能力，不能只凭 ASR 与只改图的 baseline 横向排名。我会分别设置 image-only、text-only 和 joint threat model，在各自预算内比较同类方法，并额外报告联合总成本、文本语义质量和图像范数。目标模型、prompt、解码参数及 Judge 规则保持一致。若攻击失败，我会检查图像特征是否被扰乱、文本候选是否真正影响视觉语义、图文相似度是否下降，以及生成模型是否通过上下文完成纠错，从而判断瓶颈位于单模态分支、对齐层还是下游生成层。
