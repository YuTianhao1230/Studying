# RLHF 基于人类反馈的强化学习

## 知识点解析

### 概述

RLHF，Reinforcement Learning from Human Feedback，是用人类偏好训练奖励模型，再用强化学习优化语言模型，使模型输出更符合人类偏好。

### 典型流程

```text
预训练模型
  -> SFT 得到初始助手模型
  -> 收集多答案偏好排序
  -> 训练 Reward Model
  -> 用 PPO 等算法优化策略模型
  -> 评测与安全对齐
```

### 为什么需要 RLHF

SFT 能让模型学会回答格式，但不一定能优化这些偏好：

- 有帮助。
- 真实可靠。
- 遵循指令。
- 不胡说。
- 不输出危险内容。
- 风格自然。

RLHF 试图把这些偏好转成可优化的奖励信号。

### Reward Model

Reward Model 输入 prompt 和 answer，输出一个分数，表示该回答有多符合偏好。

训练数据通常来自：

- 同一 prompt 的多个候选回答。
- 人类标注排序。
- 成对偏好数据。

### PPO 阶段

PPO 把语言模型看成策略模型，通过 Reward Model 给出的奖励进行优化。

实践中会加入 KL 约束，避免模型偏离 SFT 模型太远：

```text
reward = preference_reward - beta * KL(policy || reference_policy)
```

### 常见风险

- Reward Hacking：模型学会钻奖励模型漏洞。
- Reward Collapse：奖励信号失真导致输出质量崩坏。
- 标注偏差：人类偏好不一致。
- 成本高：需要大量采样和训练。
- 稳定性差：PPO 对超参数敏感。

### 和 DPO 的区别

- RLHF + PPO：显式训练 Reward Model，再强化学习优化。
- DPO：直接用偏好对优化策略，不单独训练 Reward Model。

DPO 更简单稳定，但表达能力和可控性取决于数据和目标设计。

## 面试应对

### RLHF 的三阶段是什么？

回答思路：按 SFT -> 训练 Reward Model -> PPO 优化的顺序讲，最后点出 KL 约束的作用。

回答模板：

RLHF 通常包括三个阶段：第一是 SFT，用高质量指令数据把预训练模型调成能对话、能遵循指令的助手模型；第二是训练 Reward Model，用人类偏好排序数据学习什么回答更好；第三是用 PPO 等强化学习算法优化策略模型，让模型在 Reward Model 打分下生成更符合人类偏好的回答，同时用 KL 约束防止偏离 SFT 模型太远。

### Reward Model 如何训练？

回答思路：讲清 pairwise 偏好数据的构造和"chosen 高分、rejected 低分"的目标，再补上线前检查。

回答模板：

Reward Model 通常用偏好排序数据训练。对同一个 prompt 采样多个回答，由人或更强模型标注哪个更好，再把这些回答组成 pairwise preference 数据。训练目标是让 Reward Model 给 chosen 更高分、给 rejected 更低分。上线前要检查标注一致性、长度偏见、领域分桶表现和 reward hacking 风险。

### PPO 阶段为什么需要 KL 约束？

回答思路：核心是"只追 reward 会跑飞、钻漏洞"，KL 把 policy 拉回 reference 附近做平衡。

回答模板：

PPO 阶段需要 KL 约束，是因为只最大化 Reward Model 分数会让策略模型偏离原来的语言分布，甚至学会钻奖励模型漏洞。KL 项把当前 policy 拉回 reference policy 附近，相当于在“追求更高 reward”和“保持原模型能力与语言质量”之间做平衡。KL 太小容易跑飞，太大则学不动。

### Reward Hacking 是什么？

回答思路：定义为"钻奖励漏洞拿高分而非真变好"，配长度偏好、过拟合单测的例子。

回答模板：

Reward Hacking 指模型没有真正变好，而是学会利用奖励函数或 Reward Model 的漏洞拿高分。例如奖励模型偏好长回答，模型就输出冗长内容；单测覆盖不足，代码模型就过拟合测试。排查时不能只看 reward 上升，还要看人工评测、分桶指标、bad case 和护栏指标。

### RLHF 和 DPO 有什么区别？

回答思路：从"是否显式训练 Reward Model、是否在线采样"这条主线对比。

回答模板：

核心区别是要不要显式的 Reward Model 和在线强化学习。RLHF+PPO 是先用偏好数据训练一个 Reward Model，再用 PPO 让策略模型在线采样、按 reward 优化，好处是能持续探索新回答、可控性强，但流程复杂、显存和调参成本高，PPO 还对超参很敏感。DPO 则跳过 Reward Model，直接用 (prompt, chosen, rejected) 偏好对，把偏好学习转成一个类似监督学习的损失，训练更简单稳定、成本低。代价是 DPO 是离线的，效果高度依赖偏好数据质量，探索能力和上限不如 RLHF。工程上数据质量好、追求稳定就用 DPO，需要更强对齐能力和在线优化时才上 RLHF。
