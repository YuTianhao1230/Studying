# RFT 拒绝采样微调

## 知识点解析

### 概述

RFT 在不同资料中可能表示两种相关但不完全相同的方法：

- **Rejection Sampling Fine-Tuning**：拒绝采样微调，先让模型生成多个候选，再用 reward/verifier 过滤高质量样本，最后用筛选结果继续 SFT。
- **Reinforcement Fine-Tuning**：强化微调，有些平台用它泛指“用 grader/reward 反馈继续优化模型”的完整训练流程。

在大模型推理训练和视觉原语方案中，RFT 通常更接近 **Rejection Sampling Fine-Tuning**。它的本质是：

```text
模型生成候选
  -> reward / verifier 评分
  -> 筛选或重采样高质量回答
  -> 把筛选结果作为 SFT 数据
```

RFT 没有直接做 policy gradient，也不需要 Critic。它可以作为 SFT 和 GRPO 之间的低风险能力增强方法。

### 1. 为什么需要 RFT

普通 SFT 依赖预先准备好的高质量答案：

```text
问题 -> 固定答案
```

但高质量人工或教师答案通常有限。RFT 利用模型本身的生成能力扩大候选：

```text
问题
  -> 生成 N 个候选
  -> 选择 reward 最高或满足规则的候选
  -> 得到新的训练样本
```

适合解决：

- 人工标注成本高。
- 现有 SFT 数据覆盖不够。
- 模型已经有一定能力，但需要更多高质量轨迹。
- 任务有可靠 verifier 或 reward。
- 希望先用稳定的 SFT 方式验证 reward 是否有效。

### 2. RFT 的标准训练流程

#### 2.1 生成候选

给定 prompt `q`，由当前模型生成多个回答：

```text
y_1, y_2, ..., y_N ~ π_current(y | q)
```

为了获得有效候选，需要设置合理的：

- temperature。
- top-p。
- 最大生成长度。
- 采样数量。
- 推理模式和输出 schema。

如果采样过于确定，所有回答完全相同，RFT 没有筛选价值。

#### 2.2 评分和验证

每个候选回答经过：

```text
规则 verifier
  + Reward Model / Grader
  + 格式检查
  + 最终答案检查
  + 过程质量检查
```

得到：

```text
score(y_i)
```

评分不能只看表面文本相似度，应尽量与真实任务目标相关。

#### 2.3 筛选训练数据

常见筛选方式：

| 方式 | 规则 | 特点 |
| --- | --- | --- |
| Best-of-N | 每个 prompt 只保留最高 reward 回答 | 数据少，质量高，但可能缺乏多样性 |
| Threshold Filtering | 保留 reward 超过阈值的回答 | 样本量可控，依赖阈值 |
| Top-k Filtering | 保留每个 prompt 的前 k 个回答 | 保留一定多样性 |
| Normal-Level Filtering | 保留部分正确、部分错误的困难样本 | 适合后续 RL 或困难能力增强 |
| Diversity-aware Filtering | 质量合格后再去重 | 防止训练集被相似模板占满 |

筛选后的数据是：

```text
D_RFT = {(q, y_i) | score(y_i) >= threshold}
```

#### 2.4 继续 SFT

将筛选后的候选作为新的监督目标：

```text
原模型或 SFT checkpoint
  -> D_RFT
  -> 标准 SFT loss
```

所以 RFT 的最后一步仍然是普通监督微调：

```text
RFT = 生成和筛选阶段 + SFT 阶段
```

### 3. RFT 和其他方法的区别

| 方法 | 候选如何产生 | 如何更新模型 | 是否需要 policy gradient | 核心目标 |
| --- | --- | --- | --- | --- |
| SFT | 外部人工/教师提供 | 直接拟合 target | 否 | 学习示范 |
| RFT | 当前模型生成后筛选 | 筛选结果继续 SFT | 否 | 放大已有好行为 |
| GRPO | 当前模型在线生成 | reward 直接更新 policy | 是 | 优化策略和探索 |
| DPO | 已有 chosen/rejected | 偏好 loss | 否 | 学习离线偏好 |
| OPD | 学生自己生成 | 教师 logits/软分布 | 否 | 教师能力迁移 |

最关键的区别：

```text
RFT：
  生成 -> 过滤 -> SFT

GRPO：
  生成 -> reward -> policy gradient
```

RFT 更稳定，但探索能力较弱；GRPO 更有在线优化能力，但 reward、采样和训练稳定性要求更高。

### 4. RFT 的几种常见变体

#### 4.1 Rejection Sampling SFT

最典型形式：

```text
生成多个回答
  -> 选最高分
  -> SFT
```

适合答案可以被明确验证的任务，例如：

- 数学答案。
- 代码单测。
- JSON schema。
- 关键帧时间误差。

#### 4.2 Reward-Weighted SFT

不只保留样本，还用 reward 作为 loss 权重：

```text
L = w(y) * L_SFT
```

高 reward 样本权重大，低 reward 样本权重小。它比硬筛选保留更多数据，但要防止 reward scale 不稳定。

#### 4.3 Iterative RFT

重复多轮：

```text
M0
  -> rollout + filter
  -> SFT 得到 M1
  -> M1 rollout + filter
  -> SFT 得到 M2
```

优点是模型能力可以逐轮提高，缺点是错误会被自我复制，必须每轮保留独立评测集和人工抽检。

#### 4.4 Expert RFT

多个专家模型分别生成不同能力的数据：

```text
box expert
  + point expert
  + reasoning expert
  -> 统一筛选
  -> RFT
```

适合把不同专项能力合并到一个模型中，但要防止输出格式和能力之间发生冲突。

### 5. RFT 的优点

- 训练阶段仍然使用稳定的 SFT loss。
- 不需要 Critic、PPO 或复杂 policy update。
- 可以直接利用现有 verifier。
- 比 GRPO 更容易作为 baseline。
- 可以放大模型已经会生成的高质量轨迹。
- 能够将在线生成数据转成离线训练资产。

### 6. RFT 的局限和风险

#### 6.1 无法突破当前 policy 的生成上限

如果 N 个候选全部错误：

```text
RFT 没有正确轨迹可以学习。
```

它只能筛选模型已经生成出来的能力，不能像 GRPO 那样通过策略更新探索新的行为。

#### 6.2 自训练错误累积

如果 verifier 把错误回答判成高分：

```text
错误候选
  -> 被筛选
  -> 继续 SFT
  -> 模型更相信错误模式
```

因此必须抽检高 reward 样本，不能把 reward 直接等同于真实正确性。

#### 6.3 样本多样性下降

只保留 Best-of-N 容易让训练集被同一种模板占满，导致：

- 推理表达模式坍缩。
- 输出重复。
- 对不同任务泛化变差。

需要做去重、难度分桶和多样性采样。

#### 6.4 长度偏差

如果评分器偏好更长的解释，RFT 会筛选出大量冗长 CoT。应同时控制：

```text
正确性
  + 证据质量
  + 长度
  + 重复
```

### 7. 关键帧任务中的 RFT

关键帧任务可以采用：

```text
M1：Structured CoT SFT checkpoint
  -> 每个视频生成 N 个 CoT + time
  -> verifier 计算时间、格式、边界和证据分数
  -> 保留 clean/high-reward 结果
  -> 继续 CoT SFT
  -> M_RFT
```

筛选条件：

```text
时间误差小
  + answer 可解析
  + before/current/after 完整
  + UI 证据与视频一致
  + 没有 GT 泄漏
  + 没有重复和幻觉
```

优先使用：

- 低 ACC 指标。
- early/late bad case。
- 二次刷新样本。
- 局部异步加载样本。
- 当前模型部分正确、部分错误的 Normal-Level 样本。

RFT 可以作为 GRPO 前的低风险实验：

```text
CoT SFT
  -> RFT
  -> 对比 CoT SFT 和 RFT
  -> 如果 RFT 已经达到目标，停止
  -> 如果仍有策略缺口，再做 GRPO
```

如果线上只需要短答案，RFT 也必须区分：

```text
CoT RFT：
  训练目标包含完整 CoT。

Direct RFT：
  筛选后只保留短答案作为 target。
```

不能把完整 CoT 候选直接训练成线上短答案模型，否则输出协议会发生变化。

## 面试应对

### RFT 是什么？

回答思路：定义生成、筛选、再 SFT 三步，并和 GRPO 区分。

回答模板：

RFT 通常指 Rejection Sampling Fine-Tuning，也就是拒绝采样微调。它先让当前模型对同一个 prompt 生成多个候选回答，再用规则、Reward Model 或 verifier 对候选打分，保留高质量样本，最后把这些样本当作新的 SFT 数据继续训练。它和 GRPO 的区别是，RFT 通过筛选结果再做监督学习，不直接用 policy gradient 更新；因此更稳定，但探索能力也更弱，无法突破当前模型完全生成不出正确答案的情况。

### RFT 和 SFT 有什么关系？

回答模板：

RFT 的最后一步仍然是 SFT。区别在于普通 SFT 的 target 通常由人工或教师预先提供，而 RFT 的 target 是当前模型自己生成后经过 verifier 筛选得到的。RFT 可以理解为“rollout 数据生成和筛选 + 继续 SFT”，它不是完全独立于 SFT 的另一种 loss。

### RFT 和 GRPO 怎么选？

回答模板：

如果 reward/verifier 已经可靠，但希望用更稳定、更低成本的方式放大模型已有的正确行为，我会先尝试 RFT。如果模型已经能生成部分正确、部分错误的回答，且目标是通过在线探索进一步优化策略和困难边界，我会考虑 GRPO。RFT 的优点是训练稳定、实现简单，缺点是无法突破当前 policy 的生成上限；GRPO 有更强的探索能力，但 rollout 成本、reward 设计和训练稳定性要求更高。

### RFT 有哪些风险？

回答模板：

RFT 的主要风险是把错误候选筛进训练集。如果 verifier 不可靠，模型会通过迭代自训练放大错误。另一个风险是 Best-of-N 采样导致样本重复、推理模板坍缩和长度偏差。因此我会抽检高 reward 样本，做去重和难度分桶，保留独立评测集，并同时监控真实业务指标、输出长度和 bad case，而不是只看 reward 分数。
