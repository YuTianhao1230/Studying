# 关键帧检测：CoT-SFT 蒸馏方案设计

## 知识点解析

### 概述

CoT-SFT 的目标不是让模型输出更长的解释，而是把人工判断关键帧时隐含的边界证据显式化。[Direct SFT](<SFT 训练方案设计.md>) 只监督最终时间点，CoT-SFT 则额外监督任务理解、目标区域、状态变化、before/current/after 和 boundary check。关键帧项目中，CoT 数据必须先经过 GT 分流、时间校验、视觉一致性检查和长度过滤，再用于 Structured CoT SFT。

```text
Direct SFT checkpoint
  -> clean CoT 数据
  -> Structured CoT SFT
  -> 评估困难边界能力
  -> 可选 [RFT](<../../../03_训练优化与对齐/后训练与对齐/RFT 拒绝采样微调.md>) / [GRPO](<../../../03_训练优化与对齐/后训练与对齐/GRPO 组相对策略优化.md>)
```

### 1. 为什么 Direct SFT 不够

Direct SFT：

```text
视频 + 任务规则 -> {"time": 6.97}
```

它可以学会：

- 视频输入。
- task_type 遵循。
- 明显完成态。
- 时间输出格式。

但单点 GT 没有充分说明：

```text
前一段为什么不能选
当前为什么第一次满足
后续为什么只是稳定延续
哪个 UI 区域是必决证据
二次刷新是否推翻当前判断
```

当相邻帧视觉差异很小，或者页面存在局部异步加载时，模型容易依赖页面先验猜时间，而不是判断真正的视觉边界。

### 2. CoT 数据应该表达什么

CoT 不应等于完整视频 caption。完整 caption 回答：

```text
视频中发生了什么？
```

关键帧 CoT 回答：

```text
针对当前 task_type，哪个证据决定了第一次完成边界？
```

推荐结构：

```text
任务理解
  -> 目标区域
  -> 必须满足的证据
  -> 候选时间窗口
  -> before：未完成证据
  -> current：首次满足证据
  -> after：稳定性复核
  -> answer
```

### 3. CoT 数据来源

| 来源 | 作用 |
| --- | --- |
| 高质量人工 GT | 提供可靠完成态标准 |
| [Direct SFT](<SFT 训练方案设计.md>) 数据 | 保持业务覆盖和基础任务格式 |
| 低 ACC 指标 | 针对能力缺口采样 |
| bad case | 覆盖 early/late、二次刷新和证据幻觉 |
| GT 附近 hard negative | 区分接近完成和首次完成 |
| [Video Primitive](<Thinking with Visual Primitives 视觉原语推理.md>) | 提供客观时间轴 |
| 强教师模型 | 生成结构化状态、事件和边界解释 |

CoT 数据不是越多越好，优先保证：

```text
任务覆盖
  + 困难边界覆盖
  + 视觉证据真实
  + 时间标签可信
```

### 4. Video Primitive

详见[Thinking with Visual Primitives 视觉原语推理.md](<Thinking with Visual Primitives 视觉原语推理.md>)。

#### 4.1 State Primitive

描述一段相对稳定的页面状态：

```text
时间：1.00-2.00
状态：详情页主体已经出现，书封清晰，但热门书评仍为空或正在加载。
```

#### 4.2 Event Primitive

描述状态转移：

```text
时间：2.00-2.40
事件：热门书评区域从占位状态切换为完整文本。
before：区域为空或存在占位。
during：文本和头像逐步出现。
after：内容清晰，页面布局停止变化。
```

#### 4.3 Primitive 的约束

- 只描述视频中客观可见的状态和事件。
- 不直接输出 GT 或最终关键帧结论。
- 不围绕已知答案事后编造解释。
- 不为了凑长度重复静止画面。
- 时间段不能倒序、越界或大面积重叠。

### 5. 教师生成流程

#### 5.1 教师输入

教师至少需要：

```text
视频
task_type
完成态定义
排除条件
豁免条件
输出 schema
```

可见 [Prompt](<../../../02_大模型/应用与问题/Prompt调优.md>) 不应直接包含：

```text
GT 是多少
答案必须选哪个时间
人工标注认为哪一帧正确
```

否则容易形成答案驱动的事后合理化。

#### 5.2 两阶段生成

```text
第一步：任务无关的视频理解
  -> 生成 State/Event Primitive 时间轴

第二步：问题驱动的边界判断
  -> 选择相关 Primitive
  -> 生成 before/current/after
  -> 输出最终答案
```

这样可以分离：

```text
看懂视频
  vs
按业务规则判断完成态
```

### 6. 推荐数据格式

```json
{
  "task_understanding": "定位首屏完成加载并稳定的最早时间",
  "target_region": "首屏商品区域、权益栏和底部导航",
  "required_evidence": [
    "核心商品图文完成渲染",
    "页面主干停止位移",
    "后续没有核心内容二次替换"
  ],
  "candidate_observation": [
    {
      "time": 6.53,
      "frame_ref": "frame_196",
      "status": "not_satisfied",
      "evidence": "商品区域仍在刷新，部分图片为空白占位。"
    },
    {
      "time": 6.97,
      "frame_ref": "frame_209",
      "status": "satisfied",
      "evidence": "核心商品图文和权益栏清晰显示，页面停止位移。"
    },
    {
      "time": 10.00,
      "frame_ref": "frame_300",
      "status": "stable_after",
      "evidence": "后续没有核心区域二次刷新或内容替换。"
    }
  ],
  "boundary_check": "6.53 尚未完成，6.97 首次满足，10.00 只是后续稳定。",
  "answer": {
    "time": 6.97
  }
}
```

核心字段：

| 字段 | 作用 |
| --- | --- |
| `task_understanding` | 压缩任务规则 |
| `target_region` | 指定目标区域 |
| `required_evidence` | 拆解完成态条件 |
| `candidate_observation` | 记录候选时间和状态 |
| `frame_ref` | 绑定具体帧或抽帧 |
| `boundary_check` | 解释首次满足 |
| `answer` | 输出可解析时间 |

### 7. CoT 数据质量治理

#### 7.1 GT 分流

人工 GT 也可能有错误，不能强制教师围绕错误 GT 生成解释：

| GT 状态 | 处理 |
| --- | --- |
| GT 正确 | 围绕 GT 前后证据生成 CoT |
| GT 疑似错误 | 教师独立判断，人工抽样复核 |
| 无法确认 | 标记 ambiguous，不直接进入训练 |

#### 7.2 clean/ambiguous/reject

```text
clean：
  时间、证据、状态和答案一致。

ambiguous：
  边界模糊、GT 冲突或证据不足。

reject：
  视频损坏、格式错误、证据幻觉或 GT 泄漏。
```

#### 7.3 自动检查

视频和时间：

- 视频能解码。
- 时间不越界、不倒序。
- before/current/after 递增。
- answer 时间和首次满足段起点一致。

视觉证据：

- UI 元素确实出现在对应时间。
- 目标区域和 task_type 一致。
- 描述的状态变化真实存在。
- 没有把后续变化写到前面。

格式和来源：

- JSON/XML schema 可解析。
- answer 字段类型正确。
- 没有 Markdown 代码块和多余格式。
- 没有 GT、参考答案、人工标注等泄漏。

长度：

- 短任务保留一到两条关键证据。
- 边界任务保留 before/current/after。
- 删除重复 caption 和无关视频描述。

### 8. CoT-SFT 如何使用现有数据

#### 8.1 现有 SFT checkpoint

```text
M0 = Direct SFT checkpoint
```

从 M0 开始训练 CoT-SFT，而不是从 Base Model 直接训练：

```text
M0
  -> clean CoT SFT
  -> M_cot
```

第一轮可以只使用 clean CoT，先验证 CoT 是否提升困难指标。此时 M_cot 的输出协议会变成完整 CoT，不应默认当作原来的短答案模型直接上线。

#### 8.2 短答案和 CoT 的混合

不要把两种 target 在同一个 Prompt 下裸混：

```text
direct：
  <answer>{"time": ...}</answer>

structured_cot：
  状态/事件/边界证据 + answer
```

如果要联合训练，必须显式标记：

```text
output_mode=direct
output_mode=structured_cot
```

并分别评测格式和业务指标。

#### 8.3 线上只需要短答案

可选路线：

```text
CoT 模型离线生成和验证困难样本
  -> 训练 direct/cot 双模式模型
  -> 线上只使用 direct mode
```

或者：

```text
CoT 教师生成更可靠的困难样本
  -> 保留短答案 target
  -> answer-only distillation
```

不能简单把 CoT 文本截掉，就认为完成了能力蒸馏；这样会丢失大部分过程监督。

### 9. CoT-SFT 与 GRPO 的关系

```text
CoT-SFT：
  教模型学习结构化证据和基本推理路径。

  [GRPO](<../../../03_训练优化与对齐/后训练与对齐/GRPO 组相对策略优化.md>)：
  让模型生成多个候选，用 reward 选择和强化更好的路径。
```

进入 GRPO 前应确认：

- CoT 格式解析率稳定。
- answer.time 可提取。
- verifier 与人工判断一致，具体机制见 [GRPO 训练方案设计.md](<GRPO 训练方案设计.md>)。
- group 内存在正确和错误的回答差异。

### 10. 常见问题

| 问题 | 判断 |
| --- | --- |
| CoT 变长但 ACC 不升 | 学到解释模板，没有学到边界 |
| CoT 证据和视频不一致 | 教师幻觉或 verifier 缺失 |
| CoT 答案和原 GT 冲突 | GT 可能有误，需要人工复核 |
| CoT 输出被截断 | max length 不足或样本过长 |
| 混合训练后短答案格式退化 | direct/cot target 没有显式区分 |

## 面试应对

### CoT 数据怎么构造？

回答模板：

我不会直接让教师模型围绕 GT 编一段解释。首先让教师生成任务无关的 State/Event Primitive 时间轴，只描述视频中客观发生的状态和事件；然后根据 task_type 选择相关 Primitive，生成目标区域、必决证据、before/current/after 和 boundary check。最后对视频、时间、结构、视觉一致性、GT 泄漏和长度做过滤。这样 CoT 学到的是可验证的边界证据，而不是事后合理化。

### 为什么 CoT-SFT 要放在 Direct SFT 之后？

回答模板：

Direct SFT 先让模型学会视频输入、任务遵循、基础完成态和稳定的答案格式。CoT-SFT 的作用是补充单点 GT 没有表达的过程监督，例如前面为什么没完成、当前为什么第一次满足、后面为什么只是稳定延续。如果模型还不会处理视频和输出格式，直接做 CoT-SFT 会让训练目标变复杂，难以判断问题来自输入链路还是推理能力。
