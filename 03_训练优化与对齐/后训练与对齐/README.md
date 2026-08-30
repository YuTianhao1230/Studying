# 后训练与对齐

## 命名规则

这个目录下的名词型知识卡片优先采用“英文缩写或英文术语 + 中文翻译”的文件名，便于从文件列表直接看出含义。

例如：

- [DPO 直接偏好优化.md](<DPO 直接偏好优化.md>)
- [RLHF 基于人类反馈的强化学习.md](<RLHF 基于人类反馈的强化学习.md>)
- [GRPO 组相对策略优化.md](<GRPO 组相对策略优化.md>)
- [RLVR 可验证奖励强化学习.md](<RLVR 可验证奖励强化学习.md>)

如果一个文件包含的是方法体系、发展脉络或综合对比，而不是单个名词，可以保留中文专题名，例如 [后训练发展史与方法对比.md](<后训练发展史与方法对比.md>)。

## 核心概念

| 文件 | 核心含义 |
| --- | --- |
| [Post-training 后训练.md](<Post-training 后训练.md>) | 预训练之后用于提升指令遵循、偏好对齐、推理能力和业务适配的一组训练方法 |
| [SFT 监督微调.md](<SFT 监督微调.md>) | 用高质量指令-回答数据做监督训练，让模型学会任务格式和基础回答方式 |
| [RLHF 基于人类反馈的强化学习.md](<RLHF 基于人类反馈的强化学习.md>) | 用人类偏好训练奖励模型，再通过强化学习优化策略模型 |
| [DPO 直接偏好优化.md](<DPO 直接偏好优化.md>) | 直接用 chosen/rejected 偏好对优化模型，降低 RLHF 工程复杂度 |
| [PPO 近端策略优化.md](<PPO 近端策略优化.md>) | 通过限制策略更新幅度提升 RL 训练稳定性的经典算法 |
| [GRPO 组相对策略优化.md](<GRPO 组相对策略优化.md>) | 用同一 prompt 下多条回答的组内相对 reward 更新模型，常用于推理 RL |
| [RLVR 可验证奖励强化学习.md](<RLVR 可验证奖励强化学习.md>) | 用数学判题、单测、schema、工具结果等可验证信号作为 reward |
| [Agentic RL 智能体强化学习.md](<Agentic RL 智能体强化学习.md>) | 针对 Agent 多步工具调用、计划、观察和执行轨迹进行强化学习 |
| [Reward Model 与 Grader 奖励模型与评分器.md](<Reward Model 与 Grader 奖励模型与评分器.md>) | 负责给模型输出、候选答案或轨迹打分的偏好模型或规则评分器 |
| [Reward Collapse 奖励坍缩.md](<Reward Collapse 奖励坍缩.md>) | 模型通过钻 reward 漏洞获得高分，但真实质量下降的现象 |
| [LoRA 低秩适配.md](<LoRA 低秩适配.md>) | 参数高效微调方法，通过低秩矩阵适配大模型 |
| [Knowledge Distillation 知识蒸馏.md](<Knowledge Distillation 知识蒸馏.md>) | 让小模型学习强模型输出、推理轨迹或分布的能力迁移方法 |

## RLVR 和 Agentic RL 为什么拆开

RLVR 和 Agentic RL 经常一起出现，但它们不是同一个层级的概念。

- RLVR 关注 reward 来源：结果能否被自动验证。
- Agentic RL 关注训练对象：是否训练多步 Agent 行为轨迹。

数学和代码任务可以是 RLVR，但不一定是 Agentic RL；搜索、代码修复、GUI 操作这类 Agent 任务可以使用 RLVR，也可以使用人工偏好、LLM Judge 或环境反馈。拆开后更方便分别理解“奖励信号”和“行动范式”。

## 建议学习顺序

1. 先看 [Post-training 后训练.md](<Post-training 后训练.md>) 和 [后训练发展史与方法对比.md](<后训练发展史与方法对比.md>) 建立全局框架。
2. 再看 [SFT 监督微调.md](<SFT 监督微调.md>)，理解后训练冷启动。
3. 接着看 [RLHF 基于人类反馈的强化学习.md](<RLHF 基于人类反馈的强化学习.md>)、[DPO 直接偏好优化.md](<DPO 直接偏好优化.md>) 和 [PPO 近端策略优化.md](<PPO 近端策略优化.md>)，理解偏好对齐。
4. 再看 [GRPO 组相对策略优化.md](<GRPO 组相对策略优化.md>) 和 [RLVR 可验证奖励强化学习.md](<RLVR 可验证奖励强化学习.md>)，理解推理强化。
5. 最后看 [Agentic RL 智能体强化学习.md](<Agentic RL 智能体强化学习.md>)，把强化学习扩展到工具调用和长任务轨迹。

