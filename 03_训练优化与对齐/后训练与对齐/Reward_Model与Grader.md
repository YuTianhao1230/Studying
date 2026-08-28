# Reward Model 与 Grader

## 一句话解释

Reward Model 是给模型输出打偏好分的模型；Grader 是更泛化的评分器，可以是规则、脚本、测试、人工或 LLM，用来判断输出是否满足标准。

## 二者区别

| 概念 | 主要用途 | 形态 |
| --- | --- | --- |
| Reward Model | 为强化学习或偏好优化提供奖励信号 | 通常是训练出来的模型 |
| Grader | 为评测或训练样本打分 | 可以是规则、代码、测试、LLM、人 |

Reward Model 更偏训练；Grader 更偏评测和数据生产。

## 为什么重要

大模型和 Agent 的能力提升越来越依赖“反馈信号”。

反馈信号可以来自：

- 人类偏好。
- 单元测试。
- 格式校验。
- 可验证答案。
- LLM Judge。
- 用户线上行为。
- 专家审核。

如果反馈信号质量差，训练和评测都会被带偏。

## Reward Model 常见流程

```text
Prompt
  -> 采样多个回答
  -> 人类排序
  -> 构造成 preference pair
  -> 训练 Reward Model
  -> 给新回答打分
```

常见训练目标是让被偏好的回答得分高于不被偏好的回答。

## Grader 常见类型

### 1. Rule-based Grader

用确定性规则打分。

适合：

- JSON 格式。
- 数学答案。
- 字段匹配。
- 单元测试。

### 2. LLM Grader

用大模型按 rubric 打分。

适合：

- 开放问答。
- 摘要质量。
- 解释质量。
- 安全性判断。

风险：

- 评分不稳定。
- 被待评答案诱导。
- rubric 不清导致打分漂移。

### 3. Execution-based Grader

通过真实执行判断结果。

适合：

- 代码题。
- Agent 工具任务。
- 前端自动化测试。
- 数据处理任务。

## 好 Grader 的标准

- 标准明确。
- 可复现。
- 能覆盖关键错误。
- 能解释扣分原因。
- 对格式和内容分别评分。
- 能区分严重错误和轻微问题。

## 常见误区

- 把 LLM Judge 当绝对真值。
- rubric 写得太抽象。
- 只评最终答案，不评过程。
- Grader 训练数据和评测数据泄漏。
- 奖励信号只优化表面格式。

## 面试可能怎么问

1. Reward Model 和 Grader 有什么区别？
2. 如何设计一个 LLM Judge rubric？
3. 为什么代码任务适合 execution-based grading？
4. Reward Hacking 如何发生？
5. 如何评估 Grader 本身的可靠性？
