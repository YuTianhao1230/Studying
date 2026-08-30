# RLVR与Agentic_RL

## 知识点解析

### 概述

RLVR 是 Reinforcement Learning with Verifiable Rewards，即用可验证结果作为奖励信号训练模型；Agentic RL 是面向多步工具调用、搜索、代码执行等 Agent 行为的强化学习。

### 为什么 RLVR 重要

传统 RLHF 依赖人类偏好或 Reward Model，奖励信号可能主观、昂贵、不稳定。

RLVR 的核心优势是：某些任务结果可以自动验证。

例如：

- 数学题答案是否正确。
- 代码是否通过测试。
- 工具调用是否返回目标结果。
- 搜索任务是否找到指定证据。
- 格式化输出是否符合 schema。

可验证奖励让训练信号更清晰，也更容易规模化。

### RLVR 基本流程

```text
给定任务
  -> 模型生成答案或行动轨迹
  -> 执行验证器
  -> 得到 reward
  -> 用 RL 算法更新模型
```

其中验证器可以是：

- 单元测试。
- 数学判题器。
- 规则校验。
- 编译器。
- 搜索证据校验。
- 真实环境反馈。

### Agentic RL

Agentic RL 不只奖励最终文本，而是奖励多步行为。

它关注：

- 是否正确规划。
- 是否选对工具。
- 是否正确传参。
- 是否能根据工具结果调整。
- 是否能完成长任务。
- 是否遵守安全约束。

训练对象可以是：

- Tool-use agent。
- Code agent。
- Search agent。
- Computer-use agent。
- Multi-agent coordinator。

### 和普通 RLHF 的区别

| 维度 | RLHF | RLVR / Agentic RL |
| --- | --- | --- |
| 奖励来源 | 人类偏好、Reward Model | 可验证结果、执行反馈、环境反馈 |
| 任务形态 | 多为单轮回答质量 | 多步行动、工具调用、长任务 |
| 核心难点 | 偏好一致性、奖励模型质量 | 环境设计、奖励稀疏、轨迹归因 |
| 典型场景 | 对话质量、安全对齐 | 代码、数学、搜索、Agent 任务 |

### 常见挑战

- 奖励稀疏：只有最终成功/失败，中间步骤没信号。
- Credit Assignment：不知道哪一步导致失败。
- 环境成本高：每次训练都要真实执行工具或测试。
- Reward Hacking：模型钻验证器漏洞。
- 长任务不稳定：越长越容易偏离目标。

### 工程实践

- 用 trajectory 记录中间步骤。
- 对关键子目标设置中间 reward。
- 使用 deterministic grader 降低噪声。
- 把高风险操作限制在模拟环境。
- 用失败 case 反向构造训练数据。

## 面试应对

### RLVR与Agentic_RL 是什么？

回答思路：先放到 SFT、RLHF、DPO、RLVR/GRPO 的后训练链路里定位。

回答模板：

RLVR 是 Reinforcement Learning with Verifiable Rewards，即用可验证结果作为奖励信号训练模型；Agentic RL 是面向多步工具调用、搜索、代码执行等 Agent 行为的强化学习。 它的核心是改变预训练模型的行为分布，让模型更符合指令、偏好、任务目标或可验证结果。

### RLVR与Agentic_RL 的训练信号是什么？

回答思路：说明使用监督答案、偏好对、reward、verifier 还是 teacher 输出。

回答模板：

其中验证器可以是： 单元测试。 判断一个后训练方法，重点看数据形式、优化目标、是否在线采样、是否需要 Reference / Reward Model，以及如何防止模型偏离原始能力。

### RLVR与Agentic_RL 适合什么场景？

回答思路：结合数据条件、目标能力和工程成本回答。

回答模板：

传统 RLHF 依赖人类偏好或 Reward Model，奖励信号可能主观、昂贵、不稳定。RLVR 的核心优势是：某些任务结果可以自动验证。例如： 数学题答案是否正确。 如果数据质量不足、reward 不可靠或评测覆盖不完整，后训练可能带来表面收益但损害真实能力。

### RLVR与Agentic_RL 有哪些风险？

回答思路：重点讲数据、reward hacking、训练稳定性和评测污染。

回答模板：

RLVR 是 Reinforcement Learning with Verifiable Rewards，即用可验证结果作为奖励信号训练模型；Agentic RL 是面向多步工具调用、搜索、代码执行等 Agent 行为的强化学习。 实际使用时必须做对照实验、分桶评测、bad case 分析和能力回归，不能只看单一平均分。
