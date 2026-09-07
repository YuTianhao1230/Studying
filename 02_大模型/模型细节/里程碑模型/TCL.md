# TCL

## 知识点解析

### 概述

TCL 是视觉语言预训练中的对比学习增强路线，关注图像和文本在全局、局部或语义层面的跨模态对齐质量，常用于理解 CLIP/ALBEF 之后多模态对齐目标如何继续细化。

### 解决的问题

CLIP 的图文对比通常是全局 image-text embedding 对齐，容易只学到粗粒度匹配。ALBEF 加入融合模块，但图文局部语义、hard negative 和跨模态 token 级关系仍然可能不足。TCL 这类方法关注：

- 全局图文匹配不等于细粒度区域-词对齐。
- web 图文对噪声会影响对比学习质量。
- 模型可能依赖 shortcut，只知道图文大概相关，但不理解局部实体、属性和关系。

### 完整架构

TCL 相关公开实现通常沿用视觉语言预训练的三段式结构：

```text
image
  -> visual encoder, often ViT-B/16
  -> patch-level visual tokens

text
  -> text encoder, often BERT-base style encoder
  -> token-level text representations

multimodal fusion
  -> cross-modal encoder
  -> fused image-text representation

contrastive objectives
  -> global image-text contrast
  -> local/token/semantic contrast
  -> matching or masked modeling objectives
```

如果以 ALBEF 风格实现理解，常见配置是：

- image encoder：ViT-B/16，12 层，hidden size 768。
- text encoder：BERT-base 风格，hidden size 768，12 heads。
- multimodal encoder：BERT-base 风格 cross-attention fusion。

不同论文或代码库中 TCL 的具体层数拆分可能不同，面试中不要把某个复现配置说成所有 TCL 的固定定义。应把重点放在“多粒度对比学习改善跨模态对齐”。

### 训练目标

TCL 的核心是让对比学习不只发生在全局图文对上，而是覆盖更多粒度：

- Global contrast：整图和整句对齐。
- Local contrast：图像 patch/region 与文本 token/phrase 对齐。
- Semantic contrast：语义相关但表面不同的图文表示更接近。
- Hard negative contrast：区分细微不匹配图文对。

常见目标可以抽象为：

```text
L_total = L_global_itc + lambda_1 * L_local_contrast
        + lambda_2 * L_semantic_contrast
        + lambda_3 * L_matching_or_mlm
```

具体项随论文实现变化，但核心都是强化跨模态对齐的粒度和判别性。

### 做了什么改变

相比 CLIP：

- CLIP 主要做全局双塔对比。
- TCL 强调更多粒度的对齐，缓解只学粗粒度匹配的问题。

相比 ALBEF：

- ALBEF 的代表性贡献是先对齐再融合和动量蒸馏。
- TCL 更关注对比目标本身如何构造得更细、更强。

### 在对抗攻击中的意义

TCL 类模型适合检验攻击是否真正破坏跨模态对齐：

- 如果扰动只影响全局 embedding，可能在 CLIP 上有效但迁移有限。
- 如果能破坏 TCL 学到的细粒度对齐，说明攻击更可能影响区域、属性、关系等语义绑定。
- 在 Syner-Attack 这类任务中，TCL 可作为比普通双塔模型更细的源模型或目标模型。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 概念题 | TCL 解决什么问题 | 全局对比不够细，强化局部/语义对齐 |
| 对比题 | TCL 和 CLIP/ALBEF 区别 | CLIP 双塔全局，ALBEF 先对齐再融合，TCL 强化对比粒度 |
| 项目题 | 为什么加入 TCL 做评估 | 检验攻击对细粒度跨模态对齐是否有效 |
| 架构题 | TCL 有哪些组件 | visual encoder、text encoder、multimodal encoder、contrastive objectives |

### 易错点

- 把 TCL 简化成“另一个 CLIP”，忽略局部/语义对比目标。
- 把某个代码实现的层数当成所有 TCL 的固定架构。
- 只讲对比学习，不讲它解决的是图文细粒度对齐不足。
- 和 ALBEF 对比时只说名字不同，不讲目标侧重点不同。

## 面试应对

### TCL 和 CLIP、ALBEF 有什么区别？

回答思路：按对齐粒度和融合方式比较。

回答模板：

CLIP 是典型双塔图文对比模型，主要学习整图和整句的全局 embedding 对齐。ALBEF 在对齐之后加入跨模态融合，通过 ITC、ITM、MLM 和动量蒸馏提升 noisy web data 下的理解能力。TCL 这类方法更关注对比目标本身的粒度，试图把对齐从全局图文对扩展到局部、token 或语义层面，让模型不只知道图文是否大致匹配，还能更好地区分区域、实体、属性和关系。
