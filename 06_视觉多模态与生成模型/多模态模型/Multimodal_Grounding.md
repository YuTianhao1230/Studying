# Multimodal_Grounding

## 知识点解析

### 概述

Multimodal Grounding 是把语言中的对象、动作、区域或时间片段，对齐到图像、视频、音频等模态中的具体位置或证据。

### 为什么重要

多模态模型不能只会“看图说话”，还要知道答案依据在哪里。

例如：

- “红色杯子在哪里？”需要输出图像区域。
- “视频里什么时候开始下雨？”需要输出时间段。
- “图中哪个按钮可以提交？”需要定位 UI 元素。
- “这个回答依据图片中哪一块？”需要给出视觉证据。

Grounding 是多模态可靠性和可解释性的基础。

### 常见任务

#### Referring Expression Comprehension

根据文本描述定位图像区域。

例子：

```text
输入：左边穿蓝衣服的人
输出：对应 bounding box
```

#### Visual Grounding

把回答中的实体映射到图像区域。

#### Phrase Grounding

把句子中的短语和视觉区域对齐。

#### Temporal Grounding

在视频中定位某个事件发生的时间段。

#### UI Grounding

在屏幕截图中定位按钮、输入框、菜单等交互元素。

这和 GUI Agent、Computer Use 关系很强。

### 关键能力

- 物体识别。
- 区域定位。
- 文本-视觉对齐。
- 空间关系理解。
- 时间关系理解。
- 多轮指代表达理解。

### 常见数据形式

- Image + Text + Bounding Box。
- Video + Text + Timestamp。
- UI Screenshot + Instruction + Element Box。
- Document Image + Question + Evidence Region。

### 评测指标

- IoU：预测框和真实框重叠度。
- Recall@K：Top-K 是否命中目标。
- Pointing Game Accuracy：点击点是否落在目标区域。
- Temporal IoU：预测时间段和真实时间段重叠。
- Answer Faithfulness：回答是否基于正确视觉证据。

### 和 Hallucination 的关系

多模态幻觉常见原因是模型生成了图中不存在的对象或属性。

Grounding 可以缓解这个问题：

- 要求模型给出视觉证据。
- 检查回答中的实体是否能定位。
- 对不能定位的内容降置信度。

## 面试应对

### Multimodal_Grounding 是什么？

回答思路：先说明处理的模态和任务，再讲输入输出。

回答模板：

Multimodal Grounding 是把语言中的对象、动作、区域或时间片段，对齐到图像、视频、音频等模态中的具体位置或证据。 它通常用于图像、文本、视频或区域级信息之间的表示、对齐、理解或生成。

### Multimodal_Grounding 的核心机制是什么？

回答思路：围绕编码、融合、对齐、生成或时序建模回答。

回答模板：

Multimodal Grounding 是把语言中的对象、动作、区域或时间片段，对齐到图像、视频、音频等模态中的具体位置或证据。 关键是说明不同模态的信息如何进入模型、如何交互，以及最终如何服务分类、检索、问答、生成或定位任务。

### Multimodal_Grounding 的场景和限制是什么？

回答思路：从数据、标注、评测和计算成本回答。

回答模板：

多模态模型不能只会“看图说话”，还要知道答案依据在哪里。例如： “红色杯子在哪里？”需要输出图像区域。 Multimodal Grounding 是把语言中的对象、动作、区域或时间片段，对齐到图像、视频、音频等模态中的具体位置或证据。 多模态任务尤其要注意数据质量、模态对齐、标注噪声和评测指标是否能反映真实体验。
