# GRPO 组相对策略优化

## 知识点解析

### 概述

GRPO，全称 **Group Relative Policy Optimization**，中文可以理解为“组相对策略优化”。它是一种用于大语言模型后训练阶段的强化学习算法，目标是在不额外训练 `Critic / Value Model` 的情况下，通过“同一个 prompt 下多条候选回答的组内相对优劣”来更新模型策略。

如果用一句话建立直觉：**[PPO](<PPO 近端策略优化.md>) 需要 Critic 来估计一个回答比预期好多少；GRPO 不训练 Critic，而是让同题的一组回答互相比较，用组内平均水平作为参照。**

GRPO 经常和 DeepSeek-R1、数学推理、代码推理、[RLVR](<RLVR 可验证奖励强化学习.md>) 一起出现。原因是这些任务往往可以给出相对明确的 reward，例如数学最终答案是否正确、代码是否通过单测、格式是否满足要求。只要 reward 相对可靠，模型就可以不断采样、比较、更新，从而强化更好的推理路径。

### 背景

传统 [RLHF](<RLHF 基于人类反馈的强化学习.md>) 中常见的 PPO 链路通常会维护 `Policy Model`、`Reference Model`、`Reward Model` 和 `Value Model / Critic`。其中 `Policy Model` 是正在训练的模型，`Reference Model` 用来约束模型不要偏离原始模型太远，`Reward Model` 或规则打分器负责评价回答质量，`Value Model / Critic` 负责估计状态价值，用来计算 advantage。

这个链路的问题在于 Critic 很重。在大模型训练里，Critic 往往和 policy 同规模，意味着额外的显存、额外的前向计算、额外的训练稳定性问题。PPO 本身也比较难调，学习率、KL 系数、clip range、reward scale、rollout 配置都会影响训练稳定性。GRPO 的提出就是为了降低这部分复杂度：**既保留强化学习的在线探索能力，又尽量去掉 Critic 带来的系统成本。**

### 方法原理

GRPO 的核心流程是：对同一个 prompt 采样一组回答，给每个回答计算 reward，然后用这一组 reward 的均值和标准差做归一化，得到每个回答的相对 advantage。简化公式是 `A_i = (r_i - mean(r)) / std(r)`。

这个公式背后的含义很直接：一个回答不再只看自己的绝对分数，而是看它在同题的一组回答里相对好不好。高于组内平均水平的回答，advantage 为正，模型会提高它的生成概率；低于组内平均水平的回答，advantage 为负，模型会降低它的生成概率。这样就不需要额外训练 Critic 来估计 baseline，因为组内平均 reward 本身就提供了一个相对基准。

GRPO 不是完全抛弃 PPO。它仍然保留 policy optimization、ratio clipping 和 KL 约束这些思想，用来限制策略更新幅度，防止模型一步更新太猛，或者偏离 reference model 太远。真正被替换的是 advantage 的来源：PPO 依赖 Critic 估计 baseline，GRPO 使用同组回答的相对 reward 构造 advantage。

### Rollout、Verifier 和 Reward 的分工

这三个概念经常一起出现，但职责不同：

```text
Rollout：
  当前 policy 实际生成的回答或行动轨迹。

Verifier：
  检查 rollout 是否满足任务标准。

Reward：
  把 verifier 的检查结果转换成可优化的分数。
```

GRPO 的完整链路是：

```text
prompt
  -> policy rollout：生成多个候选
  -> verifier：逐条检查
  -> reward：得到每条候选的分数
  -> group advantage：比较同题候选
  -> policy update
```

例如数学任务：

```text
rollout：
  模型生成多个解题过程和答案。

verifier：
  检查最终答案是否正确、格式是否合规。

reward：
  正确答案得高分，错误答案得低分。
```

Rollout 不是训练标签，而是模型当前策略在线探索出的候选；Verifier 也不一定是另一个大模型，可以是规则、程序、单元测试、环境反馈或它们的组合。Reward 可以是 verifier 的直接结果，也可以由多个分项加权得到。

### GRPO 的训练目标

对同一个 prompt `q` 采样 `G` 个回答：

```text
y_1, y_2, ..., y_G ~ π_old(y | q)
```

每个回答经过 reward function 得到 `r_i`，再计算组内相对 advantage：

```text
A_i = (r_i - mean(r_group))
      / (std(r_group) + epsilon)
```

策略更新可以简化表示为：

```text
L =
  - E[
      min(
        ρ_i,t * A_i,
        clip(ρ_i,t, 1 - ε, 1 + ε) * A_i
      )
    ]
  + β * KL(π_policy || π_reference)
```

其中：

- `ρ_i,t`：当前 policy 与旧 policy 在第 `t` 个 token 上的概率比。
- `ε`：限制单次策略更新幅度。
- `π_reference`：通常是 RL 开始前的 SFT checkpoint。
- `β`：KL 约束系数。

实际实现会因框架不同采用不同的 token-level loss 和 KL 估计，但核心思想不变：

```text
同题多答案
  -> reward 排序
  -> 组内相对 advantage
  -> clipped policy update
  -> KL 约束
```

### GRPO 为什么必须关注 reward 方差

GRPO 学习的是相对差异。如果同一个 group 中所有回答 reward 都相同：

```text
std(r_group) ≈ 0
```

那么组内没有有效的相对学习信号。常见原因：

- 所有回答都正确。
- 所有回答都错误。
- temperature 太低，回答几乎完全相同。
- parser 失败，所有回答被打成同一个分数。
- reward 过于粗糙，只返回 0/1。

训练时应监控：

```text
reward_mean
reward_std
group_zero_variance_ratio
answer_parse_rate
format_parse_rate
```

### GRPO 的 reward 设计原则

一个可用的 reward 通常由多个分项构成：

```text
R_total =
  R_answer
  + R_process
  + R_format
  + R_evidence
  - P_length
  - P_hallucination
```

设计时要遵循：

1. 最终业务目标的权重不能被格式奖励压过。
2. reward 要提供平滑差异，不能所有错误都得到同一个 0 分。
3. 必须抽检高 reward 样本，防止 reward hacking。
4. reward 分项要单独记录，不能只看总 reward。
5. reward 要与独立业务评测相关，否则 reward 上升不代表能力提升。

### GRPO 的工程训练循环

```text
1. 准备 SFT checkpoint 作为 policy 初始模型和 reference。
2. 从 RL-friendly 数据集中取一个 prompt。
3. 对同一个 prompt 采样 G 个回答。
4. 解析回答并运行 verifier。
5. 计算每个回答的分项 reward。
6. 计算组内均值、标准差和 advantage。
7. 使用 clipping 和 KL 约束更新 policy。
8. 记录 reward、KL、长度、解析率和业务指标。
9. 定期在固定回归集和困难集上评测。
```

GRPO 不应直接从 Base Model 开始。通常需要先经过 SFT 或 Reasoning SFT，让模型具备基本的任务理解、输出格式和可验证回答能力。

### 和 PPO、DPO 的区别

GRPO、PPO、[DPO](<DPO 直接偏好优化.md>) 都服务于模型对齐或能力提升，但它们的训练范式不同。

PPO 是典型在线强化学习方法，能力强但链路重，需要 Critic。DPO 更像离线偏好优化，直接使用 `(prompt, chosen, rejected)` 偏好对训练，不需要在线采样，也不需要 Critic，工程上更简单稳定。GRPO 介于两者之间：它保留了在线采样和 reward 优化，因此比 DPO 更有探索能力；同时去掉 Critic，因此比 PPO 更轻。

| 维度 | DPO | PPO | GRPO |
| --- | --- | --- | --- |
| 数据来源 | 离线 chosen/rejected | 在线采样 + reward | 在线采样 + group reward |
| 是否需要 Critic | 不需要 | 需要 | 不需要 |
| 是否需要 reward | 隐含在偏好对里 | 需要 | 需要 |
| 探索能力 | 较弱 | 强 | 强 |
| 工程复杂度 | 低 | 高 | 中 |
| 典型场景 | 风格偏好、通用对齐 | 通用 RLHF | 数学、代码、RLVR |

简单总结：DPO 是“给定好坏答案对，让模型学习偏好”；PPO 是“用 Reward 和 Critic 指导模型做强化学习”；GRPO 是“同题生成一组答案，根据组内相对 reward 更新模型”。

### 适用场景

GRPO 特别适合 **RLVR**，也就是 Reinforcement Learning with Verifiable Rewards。典型场景包括数学推理、代码生成、结构化输出、工具调用和部分 [Agent](<../../10_Agent/基础概念/Agent.md>) 任务。

数学题可以检查最终答案是否正确，代码题可以跑单测，格式任务可以做 JSON schema 校验，工具调用任务可以检查执行结果是否达成目标。这些任务的共同点是 reward 相对明确，不完全依赖人类主观偏好，因此更适合用 GRPO 这类在线 RL 方法强化模型的推理路径。

不太适合 GRPO 的场景，是那些 reward 很难定义或高度主观的任务，例如开放闲聊、创意写作、复杂价值判断。如果 reward 只能依赖一个不稳定的 LLM Judge，那么 GRPO 的收益会强依赖 Judge 质量。

### 优势与局限

GRPO 的主要优势有三点。第一，不需要 Critic，显存和计算成本更低，训练链路也少一个不稳定模块。第二，适合可验证任务，可以直接利用规则、单测、执行结果作为 reward。第三，相比 DPO，它保留了在线探索能力，模型可以生成新的推理路径，再通过 reward 强化有效路径。

它的局限也很明确。GRPO 去掉了 Critic，但没有解决 reward 质量问题。如果 reward 设计不完整，模型仍然会 reward hacking。例如数学题只看最终答案，模型可能学会猜答案；代码题单测太弱，模型可能过拟合测试；格式 reward 太重，模型可能牺牲内容质量。另一个问题是组内比较本身有方差，group size 太小会导致 advantage 估计不稳，group size 太大又会增加采样成本。此外，GRPO 仍然是 RL 训练，KL 系数、clipping、采样温度、reward scale 都会影响稳定性。

### 什么时候不应该做 GRPO

以下条件不满足时，应先修数据、SFT 或 verifier：

- 模型还不能稳定生成可解析回答。
- reward 无法区分正确和错误回答。
- group 内回答几乎没有差异。
- 高 reward 样本经人工抽检仍然大量错误。
- 真实业务指标与 reward 没有相关性。
- 训练成本无法支持 rollout 和验证。

如果 SFT 已经达到目标，也不需要为了完整训练流程强行加入 GRPO。GRPO 的价值在于利用在线探索继续优化 SFT 尚未解决的能力缺口。

### 相关概念

[PPO](<PPO 近端策略优化.md>) 是经典 policy optimization，GRPO 保留了它的策略更新和 KL 约束思想。[DPO](<DPO 直接偏好优化.md>) 是离线偏好优化，适合已有高质量偏好对的场景。[RLHF](<RLHF 基于人类反馈的强化学习.md>) 是更大的后训练框架，GRPO 可以作为其中的 RL 算法选择。[RLVR](<RLVR 可验证奖励强化学习.md>) 是 GRPO 常见的 reward 来源，尤其适合数学、代码和工具调用任务。[Agentic RL](<Agentic RL 智能体强化学习.md>) 则把 RL 目标扩展到多步工具调用和任务轨迹。[Reward Model 与 Grader](<Reward Model 与 Grader 奖励模型与评分器.md>) 决定了 GRPO 的 reward 是否可靠，也是项目落地时最需要警惕的部分。

## 面试应对

### GRPO 的核心训练目标是什么？

回答思路：按同题采样、reward、组内 advantage、策略更新和 KL 约束回答。

回答模板：

GRPO 对同一个 prompt 采样一组回答，并分别计算 reward。然后用组内 reward 的均值和标准差构造相对 advantage，高于组内平均水平的回答会被强化，低于平均水平的回答会被抑制。策略更新仍然使用 PPO 类的 clipping 和 KL 约束，防止 policy 一次更新过大或偏离 reference model 太远。它的核心不是简单多生成几次，而是把同一个任务下不同回答的相对质量转化为策略更新信号。

### GRPO 是什么？

回答思路：先给定义，再讲它为什么出现，最后讲核心机制和适用场景。

回答模板：

GRPO 是 Group Relative Policy Optimization，是一种用于大模型后训练的强化学习算法。它主要解决 PPO 训练中 Critic / Value Model 成本高、链路复杂的问题。GRPO 对同一个 prompt 采样一组回答，分别计算 reward，然后用组内 reward 的均值和标准差计算相对 advantage。高于组内平均的回答会被强化，低于平均的回答会被抑制。因为它不需要单独训练 Critic，所以显存和计算成本更低，特别适合数学、代码这类有可验证奖励的推理任务。

### GRPO 为什么可以去掉 Critic？

回答思路：先说明 PPO 里 Critic 的作用，再说明 GRPO 用组内均值替代 baseline。

回答模板：

PPO 里的 Critic 主要用于估计 baseline，从而计算 advantage，也就是某个回答相对预期好多少。GRPO 对同一个 prompt 采样多个回答，并计算这一组回答的 reward。然后它用组内 reward 的均值作为 baseline，用相对分数来近似 advantage。这样就不需要额外训练 Value Model。本质上，GRPO 用“同题多答案的组内竞争”替代了 Critic 的价值估计。

### GRPO 和 PPO 有什么区别？

回答思路：重点抓住 Critic、advantage 来源、训练成本三个维度。

回答模板：

PPO 和 GRPO 都属于 policy optimization，也都会控制策略更新幅度，并通常加入 KL 约束防止模型偏离 reference model。区别在于 PPO 需要训练 Critic 来估计 advantage，而 GRPO 不训练 Critic。GRPO 的 advantage 来自同一个 prompt 下多条回答的组内相对 reward。因此 GRPO 的训练链路更轻，显存和计算成本更低，但它对 reward 质量和采样组大小仍然很敏感。

### GRPO 和 DPO 有什么区别？

回答思路：DPO 是离线偏好优化，GRPO 是在线采样加 reward 优化。

回答模板：

DPO 和 GRPO 都可以用于对齐，但范式不同。DPO 使用离线偏好对 `(prompt, chosen, rejected)`，让模型提高 chosen 的概率、降低 rejected 的概率；GRPO 则让当前模型对同一个 prompt 生成一组回答，再根据 reward 做组内相对比较并更新策略。所以 DPO 更简单稳定，适合已有高质量偏好数据的场景；GRPO 更适合数学、代码、工具调用这类可以在线采样并用规则验证的任务。

### 为什么 GRPO 适合推理模型训练？

回答思路：强调可验证奖励、同题多解、在线探索和推理路径强化。

回答模板：

GRPO 适合推理模型训练，核心原因是数学、代码这类任务有比较可靠的 verifiable reward。同一个题目可以采样多个解法，然后用最终答案、单测、格式规则或执行结果给每个回答打分。GRPO 再用组内相对 reward 强化更好的解法。这样模型不只是模仿离线答案，而是在采样和验证中逐渐提高正确推理路径的概率，这也是它常和 RLVR、数学推理、代码推理一起讨论的原因。

### 如果一个 group 中所有回答的 reward 都一样，会发生什么？

回答思路：解释组内相对 advantage 和 reward 方差。

回答模板：

GRPO 依赖组内 reward 的相对差异。如果同一个 group 中所有回答的 reward 都一样，那么标准差接近零，所有回答的 advantage 也接近零，模型几乎得不到有效更新信号。常见原因是模型已经全部答对、全部答错、temperature 太低导致回答相同，或者 parser 和 reward 把不同回答错误地打成同一个分数。排查时要看 reward_std、group zero-variance ratio、解析率和采样多样性，并通过 Normal-Level 数据、提高探索或细化 reward 恢复差异。

### GRPO 有哪些风险？

回答思路：围绕 reward hacking、group size、KL 稳定性和开放任务不适配回答。

回答模板：

GRPO 最大风险仍然是 reward 质量。如果 reward 只看最终答案，模型可能学会猜答案；如果单测覆盖不足，模型可能过拟合测试；如果格式 reward 太强，模型可能牺牲内容质量。另一个风险是组内比较的方差，group size 太小会导致 advantage 不稳定，太大又增加采样成本。虽然 GRPO 去掉了 Critic，但它仍然是 RL 训练，KL 系数、clipping、采样温度和 reward scale 都会影响稳定性。

### 如果项目中使用 GRPO，怎么验证有效？

回答思路：回答要覆盖 baseline、分桶评测、reward hacking 检查、训练稳定性和成本。

回答模板：

我会先确认任务是否有可靠 reward，比如数学答案、代码单测或工具执行结果。然后设置 baseline，比如 [SFT](<SFT 监督微调.md>)、DPO 或不做 RL 的模型，对比 GRPO 是否提升目标能力。评测时不能只看平均分，还要看题型、难度、长度、领域的分桶结果，并抽查 bad case。同时要监控 KL、reward 分布、response 长度、pass rate、训练吞吐和显存成本，防止 reward hacking 或能力退化。只有目标指标提升、护栏指标稳定、成本可接受，才说明 GRPO 真的有效。
