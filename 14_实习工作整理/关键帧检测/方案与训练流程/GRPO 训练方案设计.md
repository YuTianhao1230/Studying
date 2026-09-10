# 关键帧检测：GRPO 训练方案

## 知识点解析

### 概述

关键帧检测中的 GRPO 不是为了让模型输出更长的思维链，而是为了在模型已经通过 Direct SFT 和 Structured CoT SFT 学会基本格式、视频理解和证据表达后，进一步优化困难样本上的边界决策。对同一个视频和 task_type，模型生成多个候选回答，系统通过时间误差、格式合法性、before/current/after 边界关系、视觉证据一致性和长度等 verifier 计算 reward，再用组内相对 reward 强化更好的回答。完整路线是：

```text
Direct SFT
  -> Structured CoT SFT
  -> 构造 RL-friendly 数据
  -> rollout 采样多条回答
  -> verifier 计算分项 reward
  -> 组内归一化 advantage
  -> GRPO 更新 policy
  -> 固定回归集和困难集验证
```

GRPO 不是必做阶段。如果 Structured CoT SFT 已经达到业务目标，或者 reward 与真实 ACC 没有可靠相关性，应停止在 SFT 或先修 verifier，而不是直接增加 RL。

### 1. 为什么关键帧任务适合 GRPO

关键帧任务具备比较好的可验证条件：

- 最终时间可以和 GT 计算误差。
- 输出 JSON/XML 可以做 schema 校验。
- `before/current/after` 可以做时间顺序和角色校验。
- 关键证据可以与参考帧、候选窗口或 UI 区域比对。
- 输出长度、重复和截断可以统计。

这使任务可以从：

```text
只模仿教师 CoT
```

进一步变成：

```text
模型自己生成候选判断
  -> 系统验证哪个判断更符合业务标准
  -> 强化高质量边界决策
```

### 2. GRPO 的核心原理

#### 2.1 组内相对比较

给定一个 prompt `q`，当前 policy 生成 `G` 个回答：

```text
y_1, y_2, ..., y_G ~ π_old(y | q)
```

每个回答经过 reward function 得到：

```text
r_1, r_2, ..., r_G
```

组内 advantage 为：

```text
A_i = (r_i - mean(r_group))
      / (std(r_group) + epsilon)
```

含义是：

- `A_i > 0`：这个回答比同题平均水平更好，增加其生成概率。
- `A_i < 0`：这个回答比同题平均水平更差，降低其生成概率。
- `A_i ≈ 0`：这个回答没有提供明显的相对学习信号。

GRPO 不训练单独的 Critic/Value Model，而是用同一 prompt 下其他回答的 reward 作为相对 baseline。

#### 2.2 为什么必须是同一个 prompt

组内比较必须发生在同一个任务条件下：

```text
同一个视频
  + 同一个 task_type
  + 同一套完成态规则
  -> 生成多个回答
```

不能把不同视频、不同指标或不同业务线的回答混在一个 group 里比较，否则 reward 差异可能来自任务难度，而不是回答质量。

#### 2.3 带 clipping 和 KL 约束的目标

GRPO 保留 PPO 类方法的策略更新约束。对回答中第 `t` 个 token，可以定义概率比：

```text
ρ_i,t(θ) =
  π_θ(y_i,t | q, y_i,<t)
  / π_old(y_i,t | q, y_i,<t)
```

策略目标可简化表示为：

```text
L_policy =
  - E[
      min(
        ρ_i,t * A_i,
        clip(ρ_i,t, 1 - ε, 1 + ε) * A_i
      )
    ]
  + β * KL(π_θ || π_ref)
```

其中：

- `π_θ`：当前正在更新的 policy。
- `π_old`：生成当前 rollout 的旧 policy。
- `π_ref`：参考模型，通常是 SFT checkpoint。
- `ε`：限制一次更新幅度。
- `β`：KL 惩罚系数。

不同框架对 KL 的具体估计和 loss 符号可能不同，但核心思想一致：

```text
强化高 reward 回答
  + 限制 policy 更新幅度
  + 防止偏离 SFT 能力太远
```

### 3. 关键帧任务的训练对象

#### 3.1 Policy

Policy 应从已经完成 Structured CoT SFT 的 checkpoint 开始，而不是从 Base Model 直接做 GRPO：

```text
Base Model
  -> Direct SFT
  -> Structured CoT SFT
  -> GRPO Policy
```

此时 policy 应该已经能：

- 接收视频。
- 遵循 task_type。
- 输出合法的 CoT 结构。
- 解析出最终 `time`。
- 初步表达 before/current/after 证据。

#### 3.2 Reference Model

Reference Model 通常使用 GRPO 开始前的 CoT SFT checkpoint：

```text
π_ref = Structured CoT SFT checkpoint
```

它的作用不是提供答案，而是约束 RL policy：

- 防止输出格式迅速漂移。
- 防止模型为了 reward 生成极端长文本。
- 防止模型遗忘 SFT 阶段的视觉和任务能力。

#### 3.3 Verifier 与 Reward

Verifier 负责把模型回答转换成可比较的 reward。关键帧任务至少需要：

```text
answer parser
  + format verifier
  + time verifier
  + boundary verifier
  + evidence verifier
  + length/repetition verifier
```

### 4. RL-friendly 数据怎么构造

GRPO 阶段不应该直接把 20w 短答案数据和 2.5w CoT target 混合作为 SFT 数据。RL 数据应该包含：

```text
视频
task_type
完成态定义
排除与豁免条件
GT time 或可验证答案
可选的 clean reference CoT
可选的候选窗口和 UI 区域
```

#### 4.1 数据来源

优先级：

1. 当前模型低 ACC 的指标。
2. 当前模型 early/late bad case。
3. GT 附近的 hard negative。
4. 存在局部异步或二次刷新的样本。
5. CoT 已通过质量校验的样本。
6. 少量普通样本，用于防止 policy 完全遗忘基础能力。

#### 4.2 Normal difficulty 筛选

可以先用 CoT SFT policy 对候选数据 rollout，再根据回答正确情况分桶：

```text
Easy：
  同一 prompt 的所有 rollout 都正确。

Normal：
  同一 prompt 有的 rollout 正确，有的错误。

Hard：
  所有 rollout 都错误。
```

GRPO 优先使用 Normal 数据：

- Easy 数据 reward 几乎相同，组内没有明显优势。
- Hard 数据可能完全没有正向回答，学习信号弱。
- Normal 数据同时包含成功和失败轨迹，最适合组内比较。

Hard 数据不是丢弃，而是用于：

```text
重新构造 CoT SFT
  + 修正 GT
  + 增强 verifier
  + 作为后续 RL 数据
```

#### 4.3 数据划分

必须按视频或原始样本实体划分：

```text
train / dev / test
```

不能只按抽帧或相邻窗口随机切分，否则同一个视频的相似片段可能同时出现在训练和评测中，导致 reward 和 ACC 虚高。

### 5. Reward 设计

推荐总 reward：

```text
R_total =
  w_time * R_time
  + w_format * R_format
  + w_boundary * R_boundary
  + w_evidence * R_evidence
  + w_task * R_task
  - w_length * P_length
  - w_hallucination * P_hallucination
```

每个分项都应单独记录，不能只保存一个总分。

#### 5.1 `R_time`：时间答案奖励

时间 reward 应同时提供平滑信号和业务命中信号：

```text
e = abs(pred_time - gt_time)

R_dense = exp(-e / τ)

R_hit =
  1.0, e <= business_tolerance
  0.0, otherwise
```

可以组合为：

```text
R_time = a * R_dense + b * R_hit
```

这样：

- 差 `0.1s` 和差 `2s` 不会被视为同样错误。
- 接近正确边界的回答仍有学习信号。
- 最终业务容忍阈值可以直接体现在 reward 中。

如果不同任务的 early 和 late 代价不一样，可以拆成：

```text
P_early
P_late
```

而不是只使用绝对误差。

无完成态样本要单独处理：

```text
GT = -1 且 pred = -1 -> 高 reward
GT = -1 但 pred >= 0 -> 惩罚
GT >= 0 但 pred = -1 -> 惩罚
```

#### 5.2 `R_format`：结构和解析奖励

检查：

- `<answer>` 是否完整闭合。
- JSON 是否可解析。
- `time` 是否是有限数字。
- 是否只出现一个最终 answer。
- `<time>/<caption>/<think>` 或 State/Event 标签是否符合 schema。
- 是否出现 Markdown 代码块或非法多余内容。

格式 reward 应该是软 reward，不能完全压过时间 reward：

```text
格式正确但时间错误
  < 
时间正确且格式正确
```

否则模型会学会只输出格式漂亮的错误答案。

#### 5.3 `R_boundary`：首次完成边界奖励

这是关键帧任务最重要的过程 reward：

```text
before：
  之前仍缺少必要证据，不能判完成。

current：
  当前首次满足全部必决条件。

after：
  后续没有核心内容二次刷新或替换。
```

可以拆成：

```text
R_before
  + R_current
  + R_after
  + R_monotonic
```

检查内容：

- 时间段是否单调递增。
- `answer.time` 是否和首次满足段起点对齐。
- before 是否真的发生在 answer 之前。
- current 是否包含完成态证据。
- after 是否提供稳定性或二次刷新复核。

仅用关键词判断 before/current/after 适合冷启动，但长期应升级为结构化字段和 verifier：

```json
{
  "status": "not_satisfied | satisfied | stable_after",
  "time": 6.97,
  "region": "product_area",
  "evidence": "..."
}
```

#### 5.4 `R_evidence`：视觉证据奖励

证据 reward 要回答：

```text
CoT 中说的 UI 元素是否在对应时间出现？
描述的状态是否真实？
证据是否支持最终时间？
是否把后面的变化错误写到了前面？
```

可以使用三层验证：

```text
时间层：
  evidence time 是否落在对应候选窗口。

区域层：
  region/box/OCR 是否对应目标 UI 区域。

语义层：
  caption/state 是否与帧内容一致。
```

参考 CoT anchor 可以作为辅助 reward，但不能成为唯一 reward。因为参考 CoT 也可能存在错误，过度奖励文本相似度会让模型复制教师噪声。

#### 5.5 `R_task`：任务规则奖励

不同 task_type 的完成态不同，应该支持任务级 verifier：

```text
商城加载：
  商品图、金刚区、页面主干必须满足规则。

小说内容加载：
  正文或书评等指定区域清晰稳定。

加购耗时：
  购物车角标或加购成功状态满足规则。
```

任务 reward 不应只检查文本中是否出现“加载完成”，而要检查任务要求的实际区域和状态。

#### 5.6 `P_length`：长度和重复惩罚

长度惩罚需要有上下限：

```text
太短：
  没有提供必要的边界证据。

合理长度：
  包含任务相关的 before/current/after。

太长：
  重复 caption、循环推理、无关视频描述。
```

不能简单地“越短越好”，否则模型会直接跳过证据，只输出答案。

#### 5.7 `P_hallucination`：幻觉惩罚

惩罚：

- 视频中不存在的 UI 元素。
- 不存在的用户点击或系统事件。
- 证据时间和文字状态不一致。
- 把正常交互解释为异常。
- 为了骗 reward 伪造与自身答案一致的证据。

### 6. 当前项目的 Reward 起始设计

当前已有结构化 reward 可以抽象为：

```text
R_time
  + R_format
  + R_boundary
  + R_cot_anchor
  + R_group_consistency
  + R_diversity
  - P_length
```

当前第一版可以使用以下起始配置作为实验基线：

| 配置项 | 起始值 | 含义 |
| --- | ---: | --- |
| `TIME_WEIGHT` | 1.00 | 时间答案 reward 权重 |
| `FORMAT_WEIGHT` | 0.25 | 结构和解析 reward 权重 |
| `BOUNDARY_WEIGHT` | 0.35 | before/current/after 边界 reward 权重 |
| `COT_ANCHOR_WEIGHT` | 0.35 | 与 clean reference CoT 的证据对齐权重 |
| `GROUP_CVK_WEIGHT` | 0.08 | 组内语义集中度调整 |
| `DVR_WEIGHT` | 0.05 | 在未解决 group 中保留合理多样性 |
| `LENGTH_WEIGHT` | 0.12 | 长度和重复惩罚权重 |
| `TIME_TOLERANCE` | 0.10s | 时间命中和边界对齐容忍度 |
| `MIN_THINK_CHARS` | 120 | 防止回答没有必要证据 |
| `MAX_THINK_CHARS` | 2600 | 限制思考文本长度 |
| `MAX_COMPLETION_CHARS` | 7000 | 限制完整回答长度 |
| `MAX_SEGMENTS` | 12 | 限制时间片段数量 |

这些参数只用于建立第一版可比较的 baseline。实际调参时应优先观察 reward 分项和困难集 ACC，不要仅凭总 reward 调整权重。

适合作为第一版起点，但需要注意：

1. `R_cot_anchor` 依赖 reference CoT，必须保证 reference CoT clean。
2. keyword-based boundary reward 可能误判，需要逐步替换为结构化 verifier。
3. CVK 类一致性奖励不能鼓励所有回答使用同一个模板。
4. DVR 类多样性奖励不能鼓励无关或错误的探索。
5. 总 reward 必须记录分项，否则无法判断模型到底利用了哪一项。

建议第一轮优先保证：

```text
R_time + R_format + R_boundary
```

确认模型能学到真实边界后，再逐步加入：

```text
R_evidence
  + R_cot_anchor
  + P_length
  + P_hallucination
```

### 7. 当前训练顺序

```text
M0：Direct SFT checkpoint
  -> M1：Structured CoT SFT
  -> 固定评测和 verifier 离线验证
  -> 生成 rollout，筛选 Normal-Level 数据
  -> M2：小规模 GRPO warmup
  -> 检查 reward 分布、KL 和 hard ACC
  -> M3：正式 GRPO
  -> 选择最佳 checkpoint
  -> 可选 [OPD 在线策略蒸馏](<../../../03_训练优化与对齐/后训练与对齐/On-Policy Distillation 在线策略蒸馏.md>) 或 answer-only distillation
```

#### 7.1 进入 GRPO 的准入条件

- CoT 输出格式解析率稳定。
- `answer.time` 可稳定提取。
- M1 在困难集上优于 M0 或至少没有退化。
- verifier 在人工抽样上与真实判断有较高一致性。
- reward 分项对真实 ACC 有相关性。
- group 内存在正确和错误回答，不是所有 reward 都相同。

#### 7.2 什么时候停止

满足以下条件时可以停止 GRPO：

- 困难指标达到目标。
- reward 继续上升但真实 ACC 不再上升。
- KL 快速上升或出现明显通用能力退化。
- 输出长度持续增长但边界准确率不变。

GRPO 不是必须训练到固定 epoch，应该以业务指标和护栏指标决定停止。

### 8. 初始参数建议

当前可作为第一轮起点的配置：

| 参数 | 起始值 | 作用和风险 |
| --- | ---: | --- |
| `num_generations` | 8 | 每个 prompt 生成 8 个回答，成本和组内方差折中 |
| `learning_rate` | `1e-6` | 全参数 GRPO 的保守起点 |
| `temperature` | 0.9 | 保留探索，过高会导致格式不稳定 |
| `beta` | 0.001 | KL 约束起点，需根据 KL 曲线调整 |
| `warmup_ratio` | 0.03 | 缓解 RL 初期更新过快 |
| `max_completion_length` | 4096 | 覆盖结构化 CoT，同时控制成本 |
| `global_batch_size` | 64 | 提升 reward 统计稳定性 |
| `precision` | BF16 | 多模态训练的稳定性和显存折中 |
| 并行策略 | ZeRO-3 | 降低全参数训练的显存压力 |

这些值只是起始点，不是固定最优配置。

#### 8.1 视觉编码器是否训练

第一轮建议优先冻结 Vision Encoder 和 aligner：

```text
先优化：
  语言侧结构化输出
  + 边界判断
  + reward 对齐
```

只有当 bad case 证明问题主要来自：

- 小 UI 元素看不清。
- 关键区域识别错误。
- 视觉证据无法进入语言侧。

才考虑解冻视觉侧，并使用更小学习率或分组学习率。

如果当前实验配置默认让 Vision Encoder、aligner 和 LLM 全部参与更新，建议把它作为单独的 full-tuning 对照组；第一版主实验优先冻结视觉侧。全量解冻视觉侧可能导致：

- reward 短期上升但视觉基础能力退化。
- 训练成本和显存显著增加。
- policy 更容易偏离 CoT SFT checkpoint。

### 9. 监控指标

训练中至少记录：

```text
reward_total
reward_time
reward_format
reward_boundary
reward_evidence
penalty_length
penalty_hallucination
reward_mean / reward_std
group_zero_variance_ratio
KL
clip_fraction
entropy
response_length
answer_parse_rate
format_parse_rate
```

同时固定评测：

```text
全量回归集
低 ACC 指标集
困难边界集
early/late 集
二次刷新集
GT 疑似错误集
```

### 10. 常见失败现象与处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| reward 几乎不变 | verifier 全部打相同分或 group 无差异 | 检查 parser、reward 分布和 `num_generations` |
| group std 接近 0 | 同题回答全部正确、全部错误或采样过于确定 | 使用 Normal-Level 数据、提高 temperature 或增加采样数 |
| format reward 上升，ACC 不升 | 格式 reward 权重过大 | 降低 format 权重，提高 time/boundary 权重 |
| CoT 持续变长 | 长输出被错误奖励 | 增加重复和长度惩罚，限制 completion length |
| KL 快速上升 | 学习率过大或 KL 约束太弱 | 降低 LR、增大 beta、减少 rollout 更新幅度 |
| KL 几乎不变且 reward 不升 | 更新过于保守 | 检查 LR、advantage 方差和 reward 是否有效 |
| 证据文本越来越像模板 | CoT anchor 或相似度奖励过强 | 降低文本相似度权重，增加视觉和边界 verifier |
| hard ACC 上升但全量 ACC 下降 | 只训练困难集导致通用能力退化 | 加入少量 direct replay 或定期做回归 |
| 高 reward 样本明显幻觉 | verifier 不完整或被 reward hacking | 人工抽查高 reward 样本，增加反例和幻觉惩罚 |

### 11. 关键结论

```text
GRPO 的核心不是“让模型多生成几次”，
而是让同一个任务下不同候选回答产生可比较的 reward 差异，
再用相对优势强化更好的回答。
```

对关键帧任务，最重要的不是让 CoT 更长，而是让 reward 能区分：

```text
6.53s 仍未完成
6.97s 首次满足
10.00s 只是后续稳定
```

如果 verifier 无法区分这些情况，GRPO 只能放大噪声；如果 verifier 可靠，GRPO 才有机会把 CoT SFT 学到的证据链进一步转化为更稳定的业务边界判断。

## 面试应对

### 为什么关键帧任务选择 GRPO，而不是 DPO、PPO、OPD 或 RFT？

回答思路：先看监督数据和目标，再比较探索能力、reward 形式和工程成本。

回答模板：

关键帧任务的目标是优化可验证的时间边界和证据判断，而不只是让模型偏好某种回答风格。DPO 需要高质量 chosen/rejected 偏好对，适合离线偏好优化，但本身没有在线探索能力；PPO 也能使用时间和证据 reward，但需要额外训练 Critic，工程和显存成本更高；OPD 需要强教师在学生 rollout 前缀上的 token-level logits，我当前主要拥有结构化 CoT 和业务 verifier，暂时不具备严格 OPD 的条件；RFT 可以作为低风险 baseline，通过筛选高 reward 回答继续 SFT，但它只能放大当前 policy 已经生成出来的行为，无法充分探索新的边界判断。GRPO 对同一个视频采样多条回答，用时间、格式、before/current/after 和证据 reward 做组内相对优化，不需要 Critic，和关键帧的可验证目标最匹配。因此我会先做 RFT 对照实验，再在 reward 可靠且 SFT 仍有边界缺口时使用 GRPO。

### GRPO 在关键帧任务中怎么设计？

回答思路：按 policy、rollout、verifier、reward、组内更新和评测回答。

回答模板：

我会从 Structured CoT SFT checkpoint 开始做 GRPO，而不是从 base model 直接训练。对同一个视频和 task_type 采样多条结构化回答，每条回答包含 before/current/after 证据和最终时间。然后用 verifier 计算分项 reward：时间误差、格式合法、边界证据、视觉区域一致性、长度和幻觉惩罚，再用组内均值和标准差构造相对 advantage，强化高于组内平均的回答，并用 KL 约束 policy 不要偏离 SFT 模型太远。数据上优先使用当前模型部分正确、部分错误的 Normal-Level 困难样本，因为全对样本没有优化空间，全错样本也缺少正向比较信号。最终同时看 reward 分项、KL、格式解析率、early/late、困难集 ACC 和全量回归集，不能只看 reward 上升。

### 关键帧 reward 为什么不能只看时间误差？

回答模板：

只看时间误差会让模型学会猜一个接近 GT 的时间，但不一定真正理解完成态边界，也可能通过错误的 CoT 获得较高分。关键帧任务至少需要加入格式、before/current/after 边界和视觉证据 reward。时间 reward 保证最终答案接近业务 GT，边界 reward 检查前面为什么未完成、当前为什么首次满足，证据 reward 检查描述的 UI 区域和状态是否真实。这样才能把“答案接近”与“判断过程可靠”区分开。

### GRPO 什么时候不应该做？

回答模板：

当 CoT SFT 还不能稳定生成合法结构，或者 verifier 不能可靠区分正确和错误边界时，不应该直接做 GRPO。因为此时 rollout 大量无法解析，组内 reward 可能全部相同，模型得到的学习信号很弱，甚至会利用格式或长度漏洞。只有当 policy 能稳定输出、reward 与真实业务指标有相关性、并且困难样本中存在部分成功和部分失败的回答时，GRPO 才值得开始。
