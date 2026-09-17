# PPO 近端策略优化

## 知识点解析

### 概述

**PPO (Proximal Policy Optimization，近端策略优化)** 是一种使用近期策略采样、通过 surrogate 目标优化策略的强化学习方法。常见的 PPO-Clip 配合 Actor-Critic，以新旧策略动作概率比和优势构造裁剪目标，降低沿有利方向过度更新的激励。裁剪不保证真实概率比或 KL 被限制在固定范围内，训练效果仍依赖奖励、价值估计与超参数。

### PPO 的核心思想

策略梯度步长过大可能使策略迅速偏离采样分布，导致性能下降。PPO-Clip 使用以下机制：

1. **Clipped surrogate**：对超过阈值且沿有利方向变化的样本，不再增加该样本的 surrogate 收益，而不是把策略参数投影到一个硬约束集合。
2. **常见 Actor-Critic 实现**：actor 输出策略，critic 估计状态价值，用来构造优势与价值回归目标。Return、V/Q/A、策略梯度和 GAE 推导见 [RL 强化学习基础](<RL 强化学习基础.md>)。
3. **On-policy 数据循环**：冻结旧策略收集一批 rollout，固定旧动作 log-prob、旧价值及由它们计算的优势和 return target；同批数据可做多个 minibatch、多个 epoch 更新，然后重新采样。不能无限复用陈旧 rollout。

### 裁剪目标与四种情况

令 $\rho_t(\theta)=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$，最大化：

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb E_{\mathrm{old}}
\left[\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)\right].
$$

这里 $\rho_t$ 是概率比，不是奖励；$\hat A_t$ 在本批更新中固定。取 $\epsilon=0.2$：

| 优势与概率比 | 未裁剪项 | 裁剪项 | min 结果 | 对该样本的作用 |
| --- | --- | --- | --- | --- |
| $\hat A=2,\rho=1.3$ | 2.6 | 2.4 | 2.4 | 增加好动作概率已越上界，停止额外激励 |
| $\hat A=2,\rho=0.7$ | 1.4 | 1.6 | 1.4 | 好动作概率下降，保留提高概率的梯度 |
| $\hat A=-2,\rho=0.7$ | -1.4 | -1.6 | -1.6 | 降低坏动作概率已越下界，停止额外激励 |
| $\hat A=-2,\rho=1.3$ | -2.6 | -2.4 | -2.6 | 坏动作概率上升，保留降低概率的梯度 |

区间内两项相同，例如 $\rho=1.1$ 时分别为 $2.2$ 和 $-2.2$。这是最大化目标；代码中的 actor loss 要取负号。

**边界**：裁剪饱和仅表示该样本的该项局部梯度为零。共享参数、其他样本、价值损失、熵或 KL 项仍可改变动作概率；单次优化步也可能越界。因此 clip 不是概率比硬限制，更不是 KL 上界。可配合较小学习率、有限 epoch、KL 监控和 target-KL 提前停止。

### 旧策略与 Reference

- **旧策略 $\pi_{\mathrm{old}}$**：产生本批 rollout 的行为策略，概率比的分母；每次新采样周期刷新。存下采样动作的旧 log-prob 后，不一定要额外保留完整旧模型。
- **Reference $\pi_{\mathrm{ref}}$**：RLHF 中用于 KL 正则的能力锚点，通常是冻结的初始 SFT 模型，也可由其他 checkpoint 初始化。一般不会每批刷新。
- PPO-Clip 本身不要求 reference 或 reward model；控制任务可直接用环境奖励。RLHF 中的 reference KL 与当前/旧策略 KL 监控含义不同，不能混用。

### 最简 Python 实现 (使用 PyTorch)

下面是 `Gymnasium CartPole` 的单环境教学示例，运行需要 PyTorch、NumPy 和 Gymnasium。采用单回合折扣回报，时间限制截断时补末端价值；不实现 GAE、并行采样和 KL 提前停止，不是生产训练器。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

def discounted_returns(rewards, gamma, last_value, terminated):
    # Time-limit truncation bootstraps; true termination does not.
    running = 0.0 if terminated else float(last_value)
    returns = [0.0] * len(rewards)
    for t in reversed(range(len(rewards))):
        running = float(rewards[t]) + gamma * running
        returns[t] = running
    return returns

### 定义 Actor-Critic 网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # 公共特征层
        self.fc = nn.Sequential(nn.Linear(state_dim, 64), nn.Tanh())
        # Actor层：输出动作概率分布
        self.actor = nn.Linear(64, action_dim)
        # Critic层：输出当前状态的价值
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        phi = self.fc(x)
        # 使用Softmax得到概率
        action_prob = F.softmax(self.actor(phi), dim=-1)
        state_value = self.critic(phi)
        return action_prob, state_value

### PPO 核心逻辑
class PPO:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
        self.gamma = 0.99    # 折扣因子
        self.eps_clip = 0.2  # PPO 截断系数
        self.epochs = 4      # 每次收集完数据后训练多少轮

    def train(self, memory):
        # 转换 memory 数据为 Tensor
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.LongTensor(np.array(memory['actions'])).view(-1, 1)
        old_log_probs = torch.FloatTensor(np.array(memory['log_probs'])).view(-1, 1)
        old_values = torch.FloatTensor(np.array(memory['values'])).view(-1, 1)
        returns = torch.FloatTensor(np.array(memory['returns'])).view(-1, 1)

        # Freeze rollout targets across all update epochs.
        advantages = (returns - old_values).detach()
        for _ in range(self.epochs):
            # 获取当前模型的概率和价值
            probs, values = self.model(states)
            dist = torch.distributions.Categorical(probs=probs)
            curr_log_probs = dist.log_prob(actions.squeeze(-1)).view(-1, 1)
            ratio = torch.exp(curr_log_probs - old_log_probs)

            # PPO 核心损失函数：Clipped Surrogate Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # 策略损失 + 价值损失 (均方误差)
            loss = -torch.min(surr1, surr2).mean() + F.mse_loss(values, returns)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

### 训练循环
env = gym.make('CartPole-v1')
ppo = PPO(4, 2)

for episode in range(500):
    state, _ = env.reset()
    memory = {
        'states': [], 'actions': [], 'log_probs': [],
        'values': [], 'rewards': [],
    }
    done = False

    # --- 阶段 1: 收集数据 ---
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            probs, value = ppo.model(state_tensor)
            dist = torch.distributions.Categorical(probs=probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            old_log_prob = dist.log_prob(action_tensor).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory['states'].append(state)
        memory['actions'].append(action)
        memory['log_probs'].append(old_log_prob)
        memory['values'].append(value.item())
        memory['rewards'].append(reward)
        state = next_state

    # --- 阶段 2: 用更新前的价值计算固定回报目标 ---
    # This non-autoreset environment returns the final observation.
    with torch.no_grad():
        last_value = 0.0 if terminated else ppo.model(
            torch.FloatTensor(state).unsqueeze(0)
        )[1].item()
    memory['returns'] = discounted_returns(
        memory['rewards'], ppo.gamma, last_value, terminated
    )

    # --- 阶段 3: 训练更新 ---
    ppo.train(memory)

    if episode % 20 == 0:
        print(f"Episode {episode}, Total Reward: {sum(memory['rewards'])}")

env.close()
```

### 代码关键点说明

1. `Categorical.sample()` 按策略分布探索；采样和更新用同一种分布计算 log-prob。
2. `ratio = exp(new_log_prob - old_log_prob)`；分母始终是产生 rollout 的策略，不随 epoch 更新。
3. `advantages = returns - old_values` 在循环外计算。Critic 可以继续拟合 returns，但不能用更新后的 critic 反复改写本批优势。
4. `terminated` 表示真实任务终止，不 bootstrap；`truncated` 若只是外部时间限制，则要 bootstrap。二者同时为真时按真实终止处理。
5. 本例一批只有单环境单回合，不跨 reset 回传。向量化版本需沿各环境时间轴计算回报/GAE 后再展平，并在自动 reset 时使用 final observation；不能把向量化环境与模型量化混为一谈。

**递推核验**：奖励 $[1,2]$、$\gamma=0.9$。真实终止时回报为 $[2.8,2]$；若只是时间截断且末端旧价值为 10，则末步目标为 $2+0.9\times10=11$，首步为 $1+0.9\times11=10.9$。若旧价值为 $[1,2]$，固定优势为 $[9.9,9]$，不能因 critic 更新而改变。

### 复杂度与教学边界

长度 $T$ 的回报递推时间和输出空间均为 $O(T)$；每轮 full-batch 更新需处理 $T$ 个样本，$K$ 个 epoch 的模型计算量约为 $O(KT C_{\mathrm{model}})$，其中 $C_{\mathrm{model}}$ 表示单样本前反向成本。网络为一层共享隐藏层加策略/价值双头；工程中常增加 minibatch、GAE、梯度裁剪、熵项和 KL 监控。

本例对时间步均匀求平均，是常见 PPO surrogate 实现，不声称精确等于折扣初始状态目标的无偏梯度。优势未标准化，超参数仅供演示，实际训练需根据环境评测和调整。

## 面试应对

### 常考点及考法

| 考法 | 解法/回答思路 |
| --- | --- |
| 写 PPO-Clip 公式并手算 | 先说明最大化目标，分别算两项，再取 min；负优势会反转乘法不等号 |
| clip 是否保证 KL 小 | 区分单样本 surrogate 饱和、共享参数更新和真实分布约束 |
| 为什么一批数据能训多个 epoch | 说明固定 rollout 目标、概率比校正与新一轮重新采样 |
| 审查代码的回报/优势计算 | 先看终止与截断，再看旧价值、旧 log-prob 是否固定及环境轴边界 |
| 旧策略和 reference 是否相同 | 分别回答采样分母与 KL 能力锚点的职责、刷新时机 |

### 易错点

- 把 `clamp(ratio)` 分支当成实际 ratio 的硬限制，忽略 `min` 与优势符号。
- 每个 epoch 重算优势，或者把旧 log-prob 更新成当前 log-prob，使分母失去采样含义。
- 将时间截断当真实终止，或对 reset 后的状态 bootstrap。
- 把 PPO 当成不需要新 rollout 的离线 replay 算法，或声称它必然最稳定。
- 将 RLHF 常见的多模型链路误认为 PPO 本身强制要求。

### 可直接复述的回答模板

**“PPO 是什么，训练信号是什么？”**

PPO 是一种基于近期 rollout 的策略优化方法。常见 PPO-Clip 用当前与旧策略的动作概率比乘固定优势，再与裁剪后的结果取较小值作为最大化目标。优势通常由奖励和 critic 构造，critic 则回归固定价值目标。裁剪减少沿有利方向继续过度更新的激励，但不是实际概率比或 KL 的硬约束。

**“PPO 一批数据更新几次，哪些量不能变？”**

同一批 rollout 可以做多个 minibatch 和多个 epoch；旧策略的动作 log-prob、旧价值及计算好的优势和回报目标保持固定，当前策略和 critic 才参与更新。更新过多会偏离采样分布，因此需要控制学习率、epoch 数并监控 KL，随后用新策略重新采样。

**“PPO 适合什么场景，有哪些风险？”**

PPO 适合能够继续采样、获得奖励反馈并承担交互成本的任务，例如控制和部分 RLHF 场景。代价是 rollout 与价值模型训练成本，风险包括 critic 误差、超参数敏感和奖励投机。clip 不能替代 KL 监控和独立评测；在 RLHF 中还要区分作为本批采样分母的旧策略，与作为长期能力锚点的 reference。
