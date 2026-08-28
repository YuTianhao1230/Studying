# Multimodal Grounding

## 一句话解释

Multimodal Grounding 是把语言中的对象、动作、区域或时间片段，对齐到图像、视频、音频等模态中的具体位置或证据。

## 为什么重要

多模态模型不能只会“看图说话”，还要知道答案依据在哪里。

例如：

- “红色杯子在哪里？”需要输出图像区域。
- “视频里什么时候开始下雨？”需要输出时间段。
- “图中哪个按钮可以提交？”需要定位 UI 元素。
- “这个回答依据图片中哪一块？”需要给出视觉证据。

Grounding 是多模态可靠性和可解释性的基础。

## 常见任务

### 1. Referring Expression Comprehension

根据文本描述定位图像区域。

例子：

```text
输入：左边穿蓝衣服的人
输出：对应 bounding box
```

### 2. Visual Grounding

把回答中的实体映射到图像区域。

### 3. Phrase Grounding

把句子中的短语和视觉区域对齐。

### 4. Temporal Grounding

在视频中定位某个事件发生的时间段。

### 5. UI Grounding

在屏幕截图中定位按钮、输入框、菜单等交互元素。

这和 GUI Agent、Computer Use 关系很强。

## 关键能力

- 物体识别。
- 区域定位。
- 文本-视觉对齐。
- 空间关系理解。
- 时间关系理解。
- 多轮指代表达理解。

## 常见数据形式

- Image + Text + Bounding Box。
- Video + Text + Timestamp。
- UI Screenshot + Instruction + Element Box。
- Document Image + Question + Evidence Region。

## 评测指标

- IoU：预测框和真实框重叠度。
- Recall@K：Top-K 是否命中目标。
- Pointing Game Accuracy：点击点是否落在目标区域。
- Temporal IoU：预测时间段和真实时间段重叠。
- Answer Faithfulness：回答是否基于正确视觉证据。

## 和 Hallucination 的关系

多模态幻觉常见原因是模型生成了图中不存在的对象或属性。

Grounding 可以缓解这个问题：

- 要求模型给出视觉证据。
- 检查回答中的实体是否能定位。
- 对不能定位的内容降置信度。

## 面试可能怎么问

1. Multimodal Grounding 是什么？
2. Grounding 和普通图像分类有什么区别？
3. 如何评估 grounding 结果？
4. Grounding 如何减少多模态幻觉？
5. UI Agent 为什么需要 grounding？
