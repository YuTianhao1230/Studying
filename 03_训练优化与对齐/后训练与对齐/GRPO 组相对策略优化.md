# GRPO 组相对策略优化

## 知识点解析

### 概述

GRPO，全称 **Group Relative Policy Optimization（组相对策略优化）**，是一种用于大语言模型后训练的强化学习方法。它对同一 prompt 采样多条回答，用组内奖励均值和标准差构造相对优势，配合裁剪策略目标及可选的 reference KL 正则更新策略，无需单独训练 Critic。数学、代码等可验证任务常使用这类方法；有效策略信号取决于组内奖励差异，而不只取决于奖励绝对值。

### 背景

传统 [RLHF](<RLHF 基于人类反馈的强化学习.md>) 中常见的 PPO 链路通常会维护 `Policy Model`、`Reference Model`、`Reward Model` 和 `Value Model / Critic`。其中 `Policy Model` 是正在训练的模型，`Reference Model` 用来约束模型不要偏离原始模型太远，`Reward Model` 或规则打分器负责评价回答质量，`Value Model / Critic` 负责估计状态价值，用来计算 advantage。

这个链路的问题在于 Critic 很重。在大模型训练里，Critic 往往和 policy 同规模，意味着额外的显存、额外的前向计算、额外的训练稳定性问题。PPO 本身也比较难调，学习率、KL 系数、clip range、reward scale、rollout 配置都会影响训练稳定性。GRPO 的提出就是为了降低这部分复杂度：**既保留强化学习的在线探索能力，又尽量去掉 Critic 带来的系统成本。**

### 方法原理

高于同题组内平均奖励的回答得到正优势，低于均值的回答得到负优势，分别提供提高与降低生成概率的激励。优势不是“正确标签”：奖励为 0 也可能产生负优势，所有回答奖励都高也可能没有组内相对信号。

GRPO 沿用 [PPO](<PPO 近端策略优化.md>) 类的 clipped surrogate，但用组内奖励统计替换常见 Actor-Critic 实现中的优势估计。Clip 裁剪的是 surrogate 激励，不是实际概率比或 KL 的硬约束。组均值包含当前样本、标准差也来自随机采样，因此不能直接套用状态 baseline 无偏证明；相关推导见 [RL 强化学习基础](<RL 强化学习基础.md>)。

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
      / (std(r_group) + eps_norm)
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

- `ρ_i,t = πθ(y_i,t | q, y_i,<t) / π_old(y_i,t | q, y_i,<t)`：在相同前缀上，当前与采样旧策略的 token 概率比。
- `ε`：surrogate 的裁剪阈值；`eps_norm > 0` 是奖励归一化的数值稳定项，两者用途不同。
- `π_old`：生成本批回答的旧策略，本批多个 epoch 内旧 log-prob 和优势保持固定，下次采样时刷新。
- `π_reference`：KL 正则的锚点，通常是冻结的初始 SFT checkpoint，也可为 Base 或其他选定模型；它不随每批 rollout 刷新，不能替代 ratio 分母。
- `β`：KL 正则系数，可取 0；对 reference 的 KL 与对旧策略的更新幅度监控是不同量。

上式的期望包含 prompt、旧策略采样回答和有效 token。常见目标先对每条回答的有效 token 求平均，再对组内回答求平均；也有不同长度归一化变体，必须说明口径。序列级奖励得到的同一个 `A_i` 通常广播到整条回答的 token，不能据此认为每一步推理都被独立验证。KL 通常在采样前缀上估计，具体采样估计式与精确 KL 不应混写。

### 二元奖励与组内优势手算

取组大小 $G=4$，奖励为 $[1,0,0,1]$，采用总体标准差（分母为 $G$）：

1. 均值 $\bar r=0.5$。
2. 方差为 $(0.25+0.25+0.25+0.25)/4=0.25$，标准差 $\sigma=0.5$。
3. 忽略极小稳定项时，优势为 $[1,-1,-1,1]$；若保留稳定项，则各项幅度为 $0.5/(0.5+\mathrm{eps\_norm})$。
4. 两个奖励为 0 的回答有负优势，仍提供降低其生成概率的策略信号。

若奖励改为 $[0,0,0,0]$ 或 $[1,1,1,1]$，均值分别为 0 或 1，标准差为 0，每个中心化分子都为 0；加稳定项后优势严格为 0。仅剩 KL 等其他损失项可能更新参数：当 $\beta>0$ 且当前策略偏离 reference 时，KL 通常能提供梯度；若两者相同且无其他项，也可能完全没有更新。混合 batch 中其他非零优势组也仍可更新共享参数。

因此二元奖励完全可用，“错误全给 0”也可用；问题是组内是否同时出现不同奖励。独立采样且单回答正确率为 $p$ 时，二元奖励同分组的概率为 $p^G+(1-p)^G$；例如 $p=0.5,G=4$ 时为 $1/8$。回答高度相关时不能直接套独立公式。

### GRPO 为什么必须关注 reward 方差

GRPO 学习的是相对差异。如果同一个 group 中所有回答 reward 都相同：

```text
std(r_group) ≈ 0
```

完全同分时，clipped policy 项的组内信号为 0，不代表总损失必无梯度；仅接近同分时还需检查稳定项、浮点精度和噪声放大。常见原因：

- 所有回答都正确。
- 所有回答都错误。
- temperature 太低，回答几乎完全相同。
- parser 失败，所有回答被打成同一个分数。
- 对当前难度，二元奖励几乎总是全对或全错；不是 0/1 奖励本身不可用。

训练时应监控：

```text
reward_mean
reward_std
group_zero_variance_ratio
answer_parse_rate
format_parse_rate
```

需明确标准差采用总体还是样本口径；$G=1$ 没有相对信号，样本标准差还可能产生 NaN。很小的方差也可能放大评分噪声，不能只追求“方差非零”。

### GRPO 的 reward 设计原则

业务需要时，reward 可以由多个分项构成；可靠的单一二元结果奖励也可以使用：

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
2. 奖励粒度要匹配业务。二元正确性奖励可将错误统一记 0；只有确实能可靠评价部分进展时，才引入过程或分项奖励，避免为制造方差而奖励错误行为。
3. 必须抽检高 reward 样本，防止 reward hacking。
4. reward 分项要单独记录，不能只看总 reward。
5. reward 要与独立业务评测相关，否则 reward 上升不代表能力提升。

### GRPO 的工程训练循环

```text
1. 选择 policy 初始 checkpoint，并在使用 KL 时指定 reference。
2. 从 RL-friendly 数据集中取一个 prompt。
3. 对同一个 prompt 采样 G 个回答。
4. 解析回答并运行 verifier。
5. 计算每个回答的分项 reward。
6. 计算组内均值、标准差和 advantage。
7. 固定本批优势与旧 log-prob，使用 clipped surrogate 和配置的 KL 正则更新 policy。
8. 记录 reward、KL、长度、解析率和业务指标。
9. 定期在固定回归集和困难集上评测。
```

GRPO 可以直接从 Base Model 开始，算法本身不要求 SFT 冷启动。是否先做 SFT 取决于初始模型能否探索到有效答案、输出是否可解析以及 verifier 是否可靠。SFT 或 Reasoning SFT 常用于降低探索难度、改善格式和可读性，但不是必要前置条件。

### 和 PPO、DPO 的区别

GRPO、PPO、[DPO](<DPO 直接偏好优化.md>) 都服务于模型对齐或能力提升，但它们的训练范式不同。

常见 PPO 使用 Actor-Critic 并在线采样；典型 DPO 直接使用 `(prompt, chosen, rejected)` 离线偏好对训练；GRPO 则在线采样多条候选并用组内奖励构造优势。GRPO 省去 Critic 成本，但总成本还取决于组大小、回答长度和 verifier 开销，不能仅凭算法名排序。

| 维度 | DPO | PPO | GRPO |
| --- | --- | --- | --- |
| 数据来源 | 离线 chosen/rejected | 在线采样 + reward | 在线采样 + group reward |
| 是否需要 Critic | 不需要 | 常见实现需要 | 不需要 |
| 是否需要 reward | 隐含在偏好对里 | 需要 | 需要 |
| 训练中在线探索 | 典型离线版本不采样 | 通常采样 | 通常同题多次采样 |
| 主要额外成本 | 偏好数据 | Rollout、价值估计、评分 | 组内多次 rollout、评分 |
| 典型场景 | 风格偏好、通用对齐 | 通用 RLHF | 数学、代码、RLVR |

### 适用场景

GRPO 特别适合 **RLVR**，也就是 Reinforcement Learning with Verifiable Rewards。典型场景包括数学推理、代码生成、结构化输出、工具调用和部分 [Agent](<../../08_Agent/基础概念/Agent.md>) 任务。

数学题可以检查最终答案是否正确，代码题可以跑单测，格式任务可以做 JSON schema 校验，工具调用任务可以检查执行结果是否达成目标。这些任务的共同点是 reward 相对明确，不完全依赖人类主观偏好，因此更适合用 GRPO 这类在线 RL 方法强化模型的推理路径。

不太适合 GRPO 的场景，是那些 reward 很难定义或高度主观的任务，例如开放闲聊、创意写作、复杂价值判断。如果 reward 只能依赖一个不稳定的 LLM Judge，那么 GRPO 的收益会强依赖 Judge 质量。

### 优势与局限

GRPO 省去了 Critic 的参数、优化器状态及训练计算，并能直接使用规则、单测、执行结果作为 reward。在线探索可以产生离线偏好数据中没有的新候选，但能否改善能力仍取决于采样覆盖和奖励质量。

它的局限也很明确。GRPO 去掉了 Critic，但没有解决 reward 质量问题。如果 reward 设计不完整，模型仍然会 reward hacking。例如数学题只看最终答案，模型可能学会猜答案；代码题单测太弱，模型可能过拟合测试；格式 reward 太重，模型可能牺牲内容质量。另一个问题是组内比较本身有方差，group size 太小会导致 advantage 估计不稳，group size 太大又会增加采样成本。此外，GRPO 仍然是 RL 训练，KL 系数、clipping、采样温度、reward scale 都会影响稳定性。

### 什么时候不应该做 GRPO

出现以下问题时，应优先修数据、探索设置或 verifier，必要时做 SFT：

- 模型还不能稳定生成可解析回答。
- reward 无法区分正确和错误回答。
- group 内回答几乎没有差异。
- 高 reward 样本经人工抽检仍然大量错误。
- 真实业务指标与 reward 没有相关性。
- 训练成本无法支持 rollout 和验证。

如果 SFT 已经达到目标，也不需要为了完整训练流程强行加入 GRPO。GRPO 的价值在于利用在线探索继续优化 SFT 尚未解决的能力缺口。

### 复杂度

每批 $B$ 个 prompt、每题 $G$ 条回答、平均长度 $L$ 时，组内奖励统计为 $O(BG)$，token 级损失聚合为 $O(BGL)$，不含模型前反向。做 $K$ 个 epoch 的模型更新约为 $O(KBGLC_{\mathrm{model}})$；还需单独计入自回归生成和 verifier 成本。去掉 Critic 不等于 rollout 免费，也不保证总显存或总时延一定低于另一套 PPO 配置。

### 相关概念

[PPO](<PPO 近端策略优化.md>) 是经典 policy optimization，GRPO 保留了它的策略更新和 KL 约束思想。[DPO](<DPO 直接偏好优化.md>) 是离线偏好优化，适合已有高质量偏好对的场景。[RLHF](<RLHF 基于人类反馈的强化学习.md>) 是更大的后训练框架，GRPO 可以作为其中的 RL 算法选择。[RLVR](<RLVR 可验证奖励强化学习.md>) 是 GRPO 常见的 reward 来源，尤其适合数学、代码和工具调用任务。[Agentic RL](<Agentic RL 智能体强化学习.md>) 则把 RL 目标扩展到多步工具调用和任务轨迹。[Reward Model 与 Grader](<Reward Model 与 Grader 奖励模型与评分器.md>) 决定了 GRPO 的 reward 是否可靠，也是项目落地时最需要警惕的部分。

## 面试应对

### 常考点及考法

| 考法 | 解法/回答思路 |
| --- | --- |
| 给一组 reward 算 advantage | 先声明标准差口径，算均值、中心化、标准差，再除以带稳定项的分母 |
| 全 0 是否不能训练，奖励 0 是否无信号 | 分开讨论组内同分、个体低于均值、KL 和其他 batch 样本 |
| 旧策略是否就是 reference | 先写 ratio 分母，再说明 reference 的 KL 锚点职责 |
| 能否从 Base 开始 | 区分算法必要条件与探索成功率、格式、奖励可靠性的工程条件 |
| 如何验证业务收益 | 对照初始模型，做难度分桶、独立评测、奖励投机抽检和成本核算 |

### 易错点

- 将二元奖励或所有错误统一为 0 直接判为无效；决定相对信号的是同组奖励差异。
- 将零方差组等同于整个模型停止更新，忽略 KL、其他损失与其他组。
- 混用归一化稳定项和 clip 阈值，或用无效的单样本标准差制造 NaN。
- 声称 SFT 是强制前置条件，或把省去 Critic 等同于总训练成本必然更低。
- 把同一序列优势广播到各 token 当成过程级正确性监督。

### GRPO 是什么，核心训练目标是什么？

回答思路：先给定义，再讲它为什么出现，最后讲核心机制和适用场景。

回答模板：

GRPO 是组相对策略优化。它对同一个 prompt 采样多条回答，用组内奖励均值和标准差构造相对优势，再用当前与旧策略的 token 概率比构造裁剪目标，并可加入 reference KL 正则。它用组内比较代替单独的 Critic，省去价值模型训练成本，但需要多次采样和可靠评分。数学、代码这类可验证任务是常见应用，clip 本身不是概率比或 KL 的硬约束。

### GRPO 为什么可以去掉 Critic？

回答思路：先说明 PPO 里 Critic 的作用，再说明 GRPO 用组内均值替代 baseline。

回答模板：

常见 PPO 的 Critic 估计状态价值，用来构造优势。GRPO 不学习这个价值函数，而是对同题回答的奖励做中心化和标准化，得到组相对策略信号。因此它省去了 Value Model，但不意味着得到了无偏的真实优势估计；组大小、样本相关性和奖励方差仍然影响训练。

### GRPO 和 PPO 有什么区别？

回答思路：重点抓住 Critic、advantage 来源、训练成本三个维度。

回答模板：

常见 PPO 与 GRPO 都可使用 clipped surrogate，但优势来源不同：PPO 通常由 Critic 和回报构造，GRPO 由同题多条回答的相对奖励构造。GRPO 省去 Critic，代价是组内多次 rollout 与评分。二者都需要关注策略漂移；在 LLM 训练中可以额外加 reference KL 正则，而 reference 不是本批采样旧策略。

### GRPO 和 DPO 有什么区别？

回答思路：DPO 是离线偏好优化，GRPO 是在线采样加 reward 优化。

回答模板：

DPO 和 GRPO 都可以用于对齐，但典型训练数据不同。DPO 用离线偏好对 `(prompt, chosen, rejected)` 优化相对 reference 的偏好概率比，不保证每个 chosen 的绝对概率都上升；GRPO 在线生成同题多条回答，再根据 reward 构造相对优势。已有高质量偏好数据时 DPO 链路通常更简单；可持续采样且能可靠验证结果时，可考虑 GRPO。

### 为什么 GRPO 适合推理模型训练？

回答思路：强调可验证奖励、同题多解、在线探索和推理路径强化。

回答模板：

GRPO 适合推理模型训练，核心原因是数学、代码这类任务有比较可靠的 verifiable reward。同一个题目可以采样多个解法，然后用最终答案、单测、格式规则或执行结果给每个回答打分。GRPO 再用组内相对 reward 强化更好的解法。这样模型不只是模仿离线答案，而是在采样和验证中逐渐提高正确推理路径的概率，这也是它常和 RLVR、数学推理、代码推理一起讨论的原因。

### 如果一个 group 中所有回答的 reward 都一样，会发生什么？

回答思路：解释组内相对 advantage 和 reward 方差。

回答模板：

所有奖励完全相同时，中心化分子为零，加数值稳定项后优势也是零，该组 clipped policy 项没有相对信号。但 KL 项在当前策略偏离 reference 时仍可能产生梯度，其他组也可更新共享参数。二元奖励并不天然有问题，例如奖励为一半 1、一半 0 时，0 分回答具有负优势。排查要看同分组比例、难度、解析率和采样多样性，不能只为增加方差而制造不可靠奖励。

### 能否从 Base Model 开始？

回答思路：先回答算法允许，再说明什么时候值得先做 SFT。

回答模板：

可以，GRPO 本身不要求初始模型一定经过 SFT。关键是 Base 是否能采到可验证的有效答案，奖励是否可靠以及输出是否容易解析。若成功样本过少或格式混乱，先做 SFT 往往降低探索成本；如果已具备可用探索信号，也可以直接做 RL。Reference 是另行选择的 KL 锚点，不一定是 SFT 模型。

### GRPO 有哪些风险？

回答思路：围绕 reward hacking、group size、KL 稳定性和开放任务不适配回答。

回答模板：

GRPO 最大风险仍然是 reward 质量。如果 reward 只看最终答案，模型可能学会猜答案；如果单测覆盖不足，模型可能过拟合测试；如果格式 reward 太强，模型可能牺牲内容质量。另一个风险是组内比较的方差，group size 太小会导致 advantage 不稳定，太大又增加采样成本。虽然 GRPO 去掉了 Critic，但它仍然是 RL 训练，KL 系数、clipping、采样温度和 reward scale 都会影响稳定性。

### 如果项目中使用 GRPO，怎么验证有效？

回答思路：回答要覆盖 baseline、分桶评测、reward hacking 检查、训练稳定性和成本。

回答模板：

我会先确认任务是否有可靠 reward，比如数学答案、代码单测或工具执行结果。然后设置 baseline，比如 [SFT](<SFT 监督微调.md>)、DPO 或不做 RL 的模型，对比 GRPO 是否提升目标能力。评测时不能只看平均分，还要看题型、难度、长度、领域的分桶结果，并抽查 bad case。同时要监控 KL、reward 分布、response 长度、pass rate、训练吞吐和显存成本，防止 reward hacking 或能力退化。只有目标指标提升、护栏指标稳定、成本可接受，才说明 GRPO 真的有效。
