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

### 什么是多模态 Grounding？为什么重要？

回答思路：先给定义（把语言对齐到具体位置/证据），再说它对可靠性和可解释性的价值。

回答模板：

Multimodal Grounding 是把语言里的对象、动作、区域或时间片段，对齐到图像、视频里的具体位置或证据。它重要是因为多模态模型不能只会“看图说话”，还得知道答案的依据在哪。比如问“红杯子在哪”要输出图像区域，问“视频里什么时候下雨”要输出时间段，问“图里哪个按钮能提交”要定位 UI 元素。Grounding 让模型的回答可定位、可验证，是多模态可靠性和可解释性的基础，也是缓解幻觉的重要手段。

### Grounding 有哪些典型任务形式？

回答思路：按空间/时间/UI 分类，各举一个代表任务和输出形式。

回答模板：

按对齐的目标可以分几类。空间维度有 referring expression comprehension，根据文本描述定位图像里的 bounding box，还有 phrase grounding，把句子里的短语和区域一一对齐。时间维度有 temporal grounding，在视频里定位某个事件发生的时间段，输出时间区间。还有 UI grounding，在屏幕截图里定位按钮、输入框，输出是元素框，这个和 GUI Agent、Computer Use 关系很紧。它们的共同点是输出不只是文字，而是坐标框、时间段这类可定位的结构化结果。

### Grounding 类任务怎么评测？

回答思路：给出 IoU、Recall@K、Pointing Game、temporal IoU、faithfulness 等指标，并说明各自适用场景。

回答模板：

看输出形式选指标。输出 bounding box 的用 IoU 衡量预测框和真实框的重叠，还可以用 Recall@K 看 top-K 里有没有命中。只需要点中目标的用 Pointing Game，看预测点有没有落在目标区域内。视频时间定位用 temporal IoU，衡量预测时间段和真实事件段的重叠。如果目标是评价回答是否基于正确视觉证据，还要看 answer faithfulness。核心是指标要匹配任务的输出类型，别用一个指标套所有任务。

### Grounding 和多模态幻觉是什么关系？

回答思路：说明幻觉常见成因，再讲 grounding 如何提供“可验证依据”来缓解。

回答模板：

多模态幻觉常见的表现是模型说出图里根本不存在的对象或属性。Grounding 能缓解这个问题，思路是要求模型给出视觉证据：回答里提到的实体必须能在图上定位，定位不到的内容就降置信度甚至不输出。训练上可以加 grounding 监督，让模型学会“先定位再回答”；评测上可以检查回答里的实体是否可定位，作为幻觉的量化指标。本质是把“凭空生成”约束成“基于证据生成”。
