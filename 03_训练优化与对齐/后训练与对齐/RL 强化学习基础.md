# RL 强化学习基础

## 知识点解析

### 概述

强化学习（Reinforcement Learning，RL）研究智能体如何通过与环境交互，学习最大化期望累计奖励的策略。MDP 用状态、动作、转移和奖励描述决策过程；价值函数描述长期收益，Bellman 方程把长期问题递归成一步奖励与后继价值。策略梯度直接优化动作分布，Actor-Critic 用价值估计降低梯度方差，TD、MC 和 GAE 则提供不同偏差与方差的学习信号。采样策略是否与目标策略一致，决定了 on-policy、off-policy 及分布校正的使用边界。

### MDP、轨迹与 Return

MDP 可写成 $(\mathcal S,\mathcal A,P,R,\gamma,\rho_0)$：

| 符号 | 含义 |
| --- | --- |
| $s_t,a_t$ | 第 $t$ 步的状态和动作 |
| $P(s'\mid s,a)$ | 下一状态的条件分布 |
| $R(s,a,s')$ | 转移奖励的条件期望；实际观测记为 $r_t$ |
| $\pi_\theta(a\mid s)$ | 参数化策略 |
| $\rho_0,\gamma$ | 初始状态分布、折扣因子 |

这里 $r_t$ 指执行 $a_t$ 后得到的奖励，轨迹为 $(s_0,a_0,r_0,s_1,\ldots,s_T)$。Markov 性要求给定当前状态和动作后，下一状态及奖励不再依赖更早历史；观测不足时是 POMDP，可用历史、记忆或 belief state 建模，不能直接把单帧观测当作充分状态。

从 $t$ 开始的折扣回报及初始状态目标为：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,\qquad
J(\theta)=\mathbb E_{\tau\sim\pi_\theta}[G_0].
$$

有限回合通常允许 $\gamma=1$；无限时域在奖励有界、$0\leq\gamma<1$ 时回报有界。无限时域 $\gamma=1$ 需要额外终止条件或改用平均奖励目标。若真实任务有固定截止时间，剩余时间通常应纳入状态。

**手算回报**：奖励依次为 $[1,2,3]$，$\gamma=0.9$，最后是真实终止。

1. $G_2=3$。
2. $G_1=2+0.9\times3=4.7$。
3. $G_0=1+0.9\times4.7=5.23$。

即时奖励为正不代表动作值得强化：动作的长期回报可能低于同状态下其他动作的期望收益。

### V、Q 与 Advantage

$$
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],\qquad
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

$$
V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a),\qquad
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s).
$$

$V$ 评价一个状态；$Q$ 固定第一步动作后评价长期收益；$A$ 衡量该动作相对当前策略平均水平的增益。同一状态下有 $\sum_a\pi(a\mid s)A^\pi(s,a)=0$，并非任意有限采样组的优势和都必须为零。

**单步任务**：动作 L 奖励 2，动作 R 奖励 0，执行后终止。若 $\pi(L)=0.75$，则 $Q(L)=2,Q(R)=0$，$V=1.5$，$A(L)=0.5,A(R)=-1.5$。奖励 0 的 R 仍有负优势；即使两个奖励都为正，低于均值的动作也有负优势。

### Bellman Expectation 与 Optimality

固定策略的 Bellman expectation 方程：

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^\pi(s')\right],
$$

$$
Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right].
$$

最优价值满足 Bellman optimality 方程：

$$
V^*(s)=\max_a\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma V^*(s')\right],
$$

$$
Q^*(s,a)=\sum_{s'}P(s'\mid s,a)
\left[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')\right].
$$

真实终止状态的后继价值按 0 处理。有限时域价值还依赖时间，可写 $V_t$，或把时间并入状态。有限状态动作、奖励有界、$\gamma<1$ 时，Bellman 算子在最大范数下是收缩映射；该结论不能直接外推成“神经网络训练必收敛”。

**两状态手算**：$\gamma=0.9$。在 $s_1$ 只能拿奖励 2 后终止；在 $s_0$，动作 A 拿奖励 1 并进入 $s_1$，动作 B 拿奖励 2 后终止；$\pi(A\mid s_0)=0.5$。

1. $V^\pi(s_1)=2$。
2. $Q^\pi(s_0,A)=1+0.9\times2=2.8$，$Q^\pi(s_0,B)=2$。
3. $V^\pi(s_0)=0.5\times2.8+0.5\times2=2.4$，两个优势为 $0.4,-0.4$。
4. 最优动作取 A，故 $V^*(s_0)=\max(2.8,2)=2.8$。

Expectation 是按给定策略求平均；optimality 是对可选动作取最大。已知模型可以做策略评估、策略迭代或价值迭代；未知模型则从样本估计这些关系。

### 策略梯度：从轨迹概率到 REINFORCE

假设环境转移、奖励机制和初始状态分布不直接依赖 $\theta$，策略可微且满足交换求导与积分所需条件。轨迹概率为：

$$
p_\theta(\tau)=\rho_0(s_0)
\prod_{t=0}^{T-1}\pi_\theta(a_t\mid s_t)P(s_{t+1}\mid s_t,a_t).
$$

按四步推导：

1. **对期望求导**：$\nabla_\theta J=\int \nabla_\theta p_\theta(\tau)G_0\,d\tau$。
2. **使用 log-derivative trick**：$\nabla p=p\nabla\log p$，得到 $\mathbb E[G_0\nabla_\theta\log p_\theta(\tau)]$。
3. **展开轨迹概率**：环境项不含参数，故 $\nabla_\theta\log p_\theta(\tau)=\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)$。
4. **去掉动作之前的奖励**：给定历史，score 的条件期望为 0；过去奖励对当前 score 的期望贡献消失。因此

$$
\nabla_\theta J
=\mathbb E_\pi\left[
\sum_{t=0}^{T-1}\gamma^t G_t
\nabla_\theta\log\pi_\theta(a_t\mid s_t)
\right].
$$

再把 $G_t$ 条件期望替换为 $Q^\pi(s_t,a_t)$，得到策略梯度定理的轨迹形式。对无限折扣时域，定义归一化折扣状态分布
$d^\pi_\gamma(s)=(1-\gamma)\sum_{t\geq0}\gamma^t\Pr_\pi(s_t=s)$，可写为：

$$
\nabla_\theta J=
\frac{1}{1-\gamma}\mathbb E_{s\sim d^\pi_\gamma,a\sim\pi}
\left[\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)\right].
$$

**折扣口径**：对这里定义的 $J=\mathbb E[G_0]$，轨迹和式外的 $\gamma^t$ 不能无说明地省略。工程中常用均匀时间步采样与折扣优势构造 surrogate，或使用不折扣的有限回合目标；需要说明目标与采样分布。

REINFORCE 用完整采样回报构造 Monte Carlo 梯度，执行梯度上升；若用最小化器，则使用
$L=-\sum_t\gamma^t\log\pi_\theta(a_t\mid s_t)\,\operatorname{stopgrad}(G_t-b(s_t))$。
它不要求奖励可微，也不需要对环境反向传播，但通常方差较大且要等回合结束。

**手算一次梯度**：单步任务中，$\pi(L)=p=\sigma(\theta)=0.5$，L/R 奖励为 2/0。

1. $\partial_\theta\log\pi(L)=1-p=0.5$，$\partial_\theta\log\pi(R)=-p=-0.5$。
2. 不加 baseline，单样本梯度分别为 $2\times0.5=1$ 和 $0\times(-0.5)=0$。
3. 期望梯度为 $0.5\times1+0.5\times0=0.5$，与 $J=2\sigma(\theta)$ 的导数 $2p(1-p)=0.5$ 一致。
4. 初始 $\theta=0$，若采到 L，学习率 $0.1$，该次 REINFORCE 更新为 $\theta'=0.1$，$p'\approx0.52498$。这是一次随机更新，不是期望更新。

### 状态 Baseline 为何不引入偏差

对只依赖状态、不依赖当前采样动作的 $b(s)$：

$$
\mathbb E_{a\sim\pi(\cdot\mid s)}
[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]
=b(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)
=b(s)\nabla_\theta1=0.
$$

所以把 $G_t$ 替换为 $G_t-b(s_t)$ 不改变策略梯度的期望。取 $b=V^\pi$ 就得到优势形式；实际 baseline 可以近似、不必完全准确，但它影响方差。在 actor loss 中必须把 baseline 当作常量，避免额外的 $\nabla b_\theta$ 项混进策略梯度。

前述 $p=0.5$ 的单步例子，取 $b=V=1$ 后，L/R 的样本梯度分别为 $(2-1)\times0.5=0.5$ 和 $(0-1)\times(-0.5)=0.5$，期望仍为 0.5，且该例的方差从 0.25 降为 0。一般任务中 $V^\pi$ 不保证是梯度方差最小的 baseline。

**边界**：动作相关 baseline 一般不能直接相减；用同一批样本拟合 baseline 也需留意统计依赖。组内均值含当前回答自身奖励，并不满足上面的动作独立证明：对同状态下 $G$ 个独立样本，未除标准差时，自包含均值会把期望梯度缩放为 $(1-1/G)$；leave-one-out 均值可去掉该自包含因素。再除随机标准差后通常不再只是常数缩放。因此不能宣称 GRPO 的组内标准化就是无偏的 $A^\pi$ 估计。

### MC、TD 与 Actor-Critic

| 方法 | 价值学习目标 | 主要特点 |
| --- | --- | --- |
| MC | 完整 $G_t$ | 在完整 on-policy 回合等条件下，对 $V^\pi$ 无偏；方差可能大 |
| TD(0) | $r_t+\gamma V_\phi(s_{t+1})$ | 用后继估计 bootstrap，可逐步更新；近似价值会带来偏差 |
| n-step TD | 前 $n$ 步折扣奖励 $+\gamma^n V_\phi(s_{t+n})$ | 在完整回报与一步 bootstrap 之间折中 |

TD residual 为 $\delta_t=r_t+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$；真实终止时后继价值为 0。价值网络通常最小化 $(V_\phi(s_t)-\operatorname{stopgrad}(y_t))^2$，即半梯度更新，不对目标中的后继价值求导。

Actor-Critic 中 actor 表示策略，critic 估计 $V$ 或 $Q$，actor 用估计优势更新。它可以是 on-policy，也可以结合校正或其他目标构造 off-policy 方法；“有 Critic”不等于“PPO”。

**手算偏差**：沿前述 $s_0\xrightarrow{A}s_1$ 路径得到奖励 $[1,2]$。若当前 $V_\phi(s_0)=1,V_\phi(s_1)=1.5$：

1. MC 目标是 $2.8$，对应样本优势 $2.8-1=1.8$。
2. TD 目标是 $1+0.9\times1.5=2.35$，$\delta_0=1.35$。
3. $\delta_1=2-1.5=0.5$。TD 与 MC 不同，是因为后继价值估计尚不准确。

### GAE 与终止边界

GAE（Generalized Advantage Estimation）对 TD residual 做指数加权：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

反向递推是 $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$。$\lambda=0$ 是一步 TD；完整终止回合中 $\lambda=1$ 时望远镜消去后得到 $G_t-V_\phi(s_t)$。截断轨迹下仍含末端 bootstrap，不能称为完整 MC。较小 $\lambda$ 通常减少采样方差，但更依赖 critic；不是所有任务上严格单调的保证。

前例取 $\lambda=0.8$：

1. $\hat A_1=\delta_1=0.5$。
2. $\hat A_0=1.35+0.9\times0.8\times0.5=1.71$。
3. 对应 value target 为 $\hat A_0+V_\phi(s_0)=2.71$。
4. 若 $\lambda=1$，$\hat A_0=1.35+0.9\times0.5=1.8$，回到该轨迹的 MC 优势。

实际采样必须分开两个 mask：

$$
\delta_t=r_t+\gamma(1-\mathrm{terminated}_t)
V_{\mathrm{old}}(s_{t+1}^{\mathrm{final}})-V_{\mathrm{old}}(s_t),
$$

$$
\hat A_t=\delta_t+\gamma\lambda c_t\hat A_{t+1}.
$$

$c_t=1$ 仅当下一条采样记录仍是同一环境、同一回合的连续转移；真实终止、时间截断后 reset 或 rollout buffer 边界都停止递推。时间限制只是采样截断时，仍 bootstrap 截断前的最后观测；如果截止时间本来就是任务终点，则按终止处理。自动 reset 环境不能拿新回合初始观测代替 final observation。向量化数据应先按各环境时间轴递推，再展平，不能把不同环境或不同回合接成一条链。

### On-policy、Off-policy 与重要性采样

On-policy 使用目标策略自身产生的数据学习；off-policy 的行为策略 $\mu$ 与被评估或优化的目标策略 $\pi$ 可以不同，例如使用历史 replay。在线/离线描述数据是否继续采集，并不等同于 on/off-policy：在线交互也可以用 off-policy 算法。

在固定状态上，重要性采样恒等式是：

$$
\mathbb E_{a\sim\pi}[f(s,a)]
=\mathbb E_{a\sim\mu}
\left[\frac{\pi(a\mid s)}{\mu(a\mid s)}f(s,a)\right].
$$

**支持集条件**：目标分布有正概率且对目标有贡献的动作，行为分布必须有正概率；标准充分条件是 $\pi(\cdot\mid s)\ll\mu(\cdot\mid s)$。权重能修正采样频率，不能补出从来采不到的动作。

**手算**：固定状态，$\mu(L)=\mu(R)=0.5$，$\pi(L)=0.8,\pi(R)=0.2$，$f(L)=2,f(R)=0$。权重是 $1.6,0.4$，校正期望为 $0.5\times1.6\times2+0.5\times0.4\times0=1.6$，等于目标期望。若 $\mu(L)=0$，无法通过除法恢复 L 的贡献。

**状态分布不能漏掉**：只乘动作比率，将 $d^\mu(s)\mu(a\mid s)$ 变成 $d^\mu(s)\pi(a\mid s)$，并未变成 $d^\pi(s)\pi(a\mid s)$。若 $f$ 只依赖状态，两个状态下 $f=(1,0)$，行为状态占比 $(0.9,0.1)$、目标占比 $(0.1,0.9)$，动作校正后的期望仍为 0.9，而目标是 0.1。

同一初始分布和转移机制下，完整轨迹权重为 $\prod_t\pi(a_t\mid s_t)/\mu(a_t\mid s_t)$；per-decision IS 可对 $r_t$ 只乘截至 $t$ 的前缀比率。这能校正轨迹分布，但长序列比率乘积可能产生巨大方差。自归一化或截断权重一般引入偏差。严格的 off-policy 策略梯度还要匹配目标状态分布和价值估计，不能仅把一个动作比率乘上去就宣称无偏。

[PPO](<PPO 近端策略优化.md>) 用近期 $\pi_{\mathrm{old}}$ rollout 构造局部 surrogate，可在一批数据上做多个 epoch，但并不因此成为可任意重放陈旧数据的通用 off-policy 方法。[GRPO](<GRPO 组相对策略优化.md>) 则用同题组内奖励差异提供策略信号。

### 复杂度与适用边界

- 长度 $T$ 的回报、TD residual 和 GAE 均可 $O(T)$ 计算；反向递推额外状态为 $O(1)$，若存储每步结果则为 $O(T)$。
- 有限 MDP、稠密转移表下，一轮 Bellman backup 为 $O(|S|^2|A|)$；存 $V$ 为 $O(|S|)$，存 $Q$ 为 $O(|S||A|)$，转移表本身为 $O(|S|^2|A|)$。稀疏转移可按非零边数计算。
- 神经网络训练还需计入模型前反向、采样和环境成本，不能只用递推复杂度代表总训练成本。
- 稀疏奖励、长时信用分配、奖励投机和分布偏移不会被某个优化器自动解决；off-policy、函数逼近与 bootstrap 的组合还可能产生不稳定。
- LLM 中状态常为 prompt 加已生成前缀，动作为下一 token；EOS 可视为回合结束，但长度上限究竟是任务终点还是采样截断，必须与奖励和价值定义一致。

## 面试应对

### 常考点及考法

| 考法 | 解法/回答思路 |
| --- | --- |
| 给短轨迹算 return、TD、GAE | 先写奖励下标、$\gamma,\lambda$ 和终止类型，再从后向前递推 |
| 区分 V/Q/A，写 Bellman 方程 | 先说条件期望固定了什么，再区分按策略求平均与对动作取最大 |
| 白板推导策略梯度 | 轨迹概率、log trick、环境项消失、因果性去掉过去奖励 |
| 证明 baseline 不偏 | 固定状态，将 $\pi\nabla\log\pi$ 化成 $\nabla\pi$，用概率和为 1 |
| 比较 REINFORCE 与 Actor-Critic | 回报来源、是否 bootstrap、方差、critic 误差和采样成本 |
| 历史数据能否训练当前策略 | 检查动作支持集、行为概率、状态分布和权重方差 |

### 易错点

- 把 reward、return、value、advantage 混为一谈；优势符号取决于相对基准。
- 把任意近似 critic 或随机组均值等同于真实 $V^\pi$，或对 actor 的优势目标求导。
- 将 MC“无偏”外推到截断回合、off-policy 样本或不匹配的优化目标。
- 把 $\gamma$ 与 GAE 的 $\lambda$ 混淆；前者定义折扣，后者控制 residual 加权。
- 将时间截断当作终止，或在 bootstrap 中使用 reset 后观测。
- 只看动作 ratio 就认为已经修正状态分布；忽略零支持集和长轨迹高方差。

### 可直接复述的回答模板

**“如何把 RL 基础串起来？”**

强化学习在 MDP 中优化期望累计奖励。Return 是一条轨迹上的折扣奖励和，V 是给定状态的期望 return，Q 额外固定第一步动作，A 等于 Q 减 V。Bellman expectation 按当前策略对后续收益求期望，optimality 则选择价值最大的动作。策略梯度通过轨迹概率的 log 导数直接优化策略；REINFORCE 用完整回报，Actor-Critic 用价值估计构造优势以降低方差。TD 用后继价值 bootstrap，GAE 对多步 TD residual 加权，需要正确处理终止与截断。

**“为什么减 baseline 没有偏差？”**

对固定状态，只要 baseline 不依赖当前采样动作，baseline 乘策略 score 的条件期望就是 baseline 乘所有动作概率导数之和，等于零。因此相减不改变策略梯度期望，但可以降低方差。实现中 actor 不能对 baseline 反向传播；动作相关或包含自身奖励的随机 baseline 需要单独分析，不能直接套这个证明。

**“PPO 有概率比，为什么仍称 on-policy？”**

PPO 用近期旧策略采样的一批 rollout，再用当前策略与该旧策略的动作概率比构造局部 surrogate，并在固定旧概率和优势下做若干轮更新。这个比率没有完整修正目标策略的状态访问分布，clip 也会改变目标，因此它不支持无限复用任意陈旧数据。通常每轮重新采样，配合学习率、epoch 数和 KL 监控控制策略漂移。
