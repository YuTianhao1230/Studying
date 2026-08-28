# RLVR 与 Agentic RL

## 一句话解释

RLVR 是 Reinforcement Learning with Verifiable Rewards，即用可验证结果作为奖励信号训练模型；Agentic RL 是面向多步工具调用、搜索、代码执行等 Agent 行为的强化学习。

## 为什么 RLVR 重要

传统 RLHF 依赖人类偏好或 Reward Model，奖励信号可能主观、昂贵、不稳定。

RLVR 的核心优势是：某些任务结果可以自动验证。

例如：

- 数学题答案是否正确。
- 代码是否通过测试。
- 工具调用是否返回目标结果。
- 搜索任务是否找到指定证据。
- 格式化输出是否符合 schema。

可验证奖励让训练信号更清晰，也更容易规模化。

## RLVR 基本流程

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

## Agentic RL

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

## 和普通 RLHF 的区别

| 维度 | RLHF | RLVR / Agentic RL |
| --- | --- | --- |
| 奖励来源 | 人类偏好、Reward Model | 可验证结果、执行反馈、环境反馈 |
| 任务形态 | 多为单轮回答质量 | 多步行动、工具调用、长任务 |
| 核心难点 | 偏好一致性、奖励模型质量 | 环境设计、奖励稀疏、轨迹归因 |
| 典型场景 | 对话质量、安全对齐 | 代码、数学、搜索、Agent 任务 |

## 常见挑战

- 奖励稀疏：只有最终成功/失败，中间步骤没信号。
- Credit Assignment：不知道哪一步导致失败。
- 环境成本高：每次训练都要真实执行工具或测试。
- Reward Hacking：模型钻验证器漏洞。
- 长任务不稳定：越长越容易偏离目标。

## 工程实践

- 用 trajectory 记录中间步骤。
- 对关键子目标设置中间 reward。
- 使用 deterministic grader 降低噪声。
- 把高风险操作限制在模拟环境。
- 用失败 case 反向构造训练数据。

## 面试可能怎么问

1. RLVR 和 RLHF 的区别是什么？
2. 什么任务适合 verifiable reward？
3. Agentic RL 为什么比普通回答优化更难？
4. 如何给工具调用任务设计 reward？
5. 如何避免模型钻测试用例漏洞？
