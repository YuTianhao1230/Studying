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
   - 回答思路：先给一句话定义，再说明它解决什么问题，最后补一个工程例子或常见风险。
   - 回答模板：Multimodal Grounding 是 Multimodal Grounding 里的核心概念，主要解决系统在能力、效率、稳定性或可控性上的问题。面试中要说明它的定义、适用场景、限制，以及在真实工程中如何验证它有效。
2. Grounding 和普通图像分类有什么区别？
   - 回答思路：先分别定义两个概念，再从目标、机制、适用场景和工程取舍四个角度对比。
   - 回答模板：先分别定义两个对象，再比较目标、机制、适用场景和风险。围绕 Multimodal Grounding，关键是说明它们解决的问题不同。
3. 如何评估 grounding 结果？
   - 回答思路：按“目标定义 -> 关键步骤 -> 风险控制 -> 指标验证”的顺序回答，避免只讲抽象原则。
   - 回答模板：我会把 评估 grounding 结果 拆成几个步骤：先明确目标和输入输出，再设计关键模块或策略，然后加入失败处理和权限/质量约束，最后用离线指标、线上指标或回归测试验证效果。在 Multimodal Grounding 场景里，重点是能落到真实工程流程。
4. Grounding 如何减少多模态幻觉？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 Multimodal Grounding 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
5. UI Agent 为什么需要 grounding？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 Multimodal Grounding 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
