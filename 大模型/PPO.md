**PPO (Proximal Policy Optimization，近端策略优化)** 是目前强化学习（RL）领域最流行、算法效果最稳健的算法之一。它由 OpenAI 在 2017 年提出，现在已成为许多 RL 项目（如 ChatGPT 的强化学习阶段）的默认基准算法。

### 1. PPO 的核心思想

在传统的策略梯度（Policy Gradient）算法中，如果步长（学习率）太大，策略更新就会过猛，导致模型坍塌且难以恢复。

PPO 解决了这个问题，它的核心是 **“限制更新幅度”**：
1.  **Clipped Objective (截断目标函数)**：它计算“新策略”和“旧策略”的比率。如果新策略偏离旧策略太多（超过了一个比例 $\epsilon$，通常是 0.2），它就会把这个比率“截断”，防止步子迈得太大。
2.  **Actor-Critic 架构**：
    *   **Actor (演员)**：负责选择动作（策略 $\pi$）。
    *   **Critic (评论家)**：负责预测当前状态的分数（价值 $V$），用来辅助 Actor 更新。
3.  **On-policy**：PPO 是一种在线学习算法，意味着它收集一段数据，更新一次，然后就把这些数据丢掉。

---

### 2. 最简 Python 实现 (使用 PyTorch)

为了保持代码足够简单，我们使用 `Gymnasium` 的经典环境 `CartPole`（平衡杆）。这个实现去掉了复杂的并行环境和 GAE（广义优势估计），只保留 PPO 的精髓。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

# 1. 定义 Actor-Critic 网络
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

# 2. PPO 核心逻辑
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
        old_probs = torch.FloatTensor(np.array(memory['probs'])).view(-1, 1)
        returns = torch.FloatTensor(np.array(memory['returns'])).view(-1, 1)

        for _ in range(self.epochs):
            # 获取当前模型的概率和价值
            probs, values = self.model(states)
            curr_probs = probs.gather(1, actions)
            
            # 计算优势 (Advantage): 实际回报 - 预测价值
            advantages = returns - values.detach()

            # 计算比率 ratio = curr_prob / old_prob
            ratio = curr_probs / old_probs

            # PPO 核心损失函数：Clipped Surrogate Objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 策略损失 + 价值损失 (均方误差)
            loss = -torch.min(surr1, surr2).mean() + F.mse_loss(values, returns)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# 3. 训练循环
env = gym.make('CartPole-v1')
ppo = PPO(4, 2)

for episode in range(500):
    state, _ = env.reset()
    memory = {'states': [], 'actions': [], 'probs': [], 'rewards': []}
    done = False
    
    # --- 阶段 1: 收集数据 ---
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = ppo.model(state_tensor)
        
        # 按概率采样动作
        action = torch.multinomial(probs, 1).item()
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory['states'].append(state)
        memory['actions'].append(action)
        memory['probs'].append(probs[0, action].item())
        memory['rewards'].append(reward)
        state = next_state

    # --- 阶段 2: 计算每一步的回报 (Returns) ---
    discounted_rewards = []
    running_add = 0
    for r in reversed(memory['rewards']):
        running_add = r + ppo.gamma * running_add
        discounted_rewards.insert(0, running_add)
    memory['returns'] = discounted_rewards

    # --- 阶段 3: 训练更新 ---
    ppo.train(memory)

    if episode % 20 == 0:
        print(f"Episode {episode}, Total Reward: {sum(memory['rewards'])}")
```

### 3. 代码关键点说明

1.  **`torch.multinomial`**: 这让智能体探索环境。它不是选择概率最大的动作，而是根据概率分布进行随机采样。
2.  **`ratio` (新旧策略比)**: 
    *   如果 `ratio > 1`，说明这个动作在新策略中更可能发生。
    *   如果 `ratio < 1`，说明在新策略中发生的可能性降低了。
3.  **`torch.clamp`**: 这是 PPO 的灵魂。它把 `ratio` 限制在 `[0.8, 1.2]` 之间。如果优势（Advantage）很大，但 `ratio` 已经超过了 1.2，损失函数就不会再增加，从而阻止了参数的剧烈波动。
4.  **`advantages` (优势)**: 告诉我们这个动作比平均水平好多少。如果返回的回报（returns）比 Critic 预测的价值（values）高，说明这个动作值得以后更多地被选中。

### 为什么这个代码是最简单的？
*   它没有使用复杂的向量化环境（如 SubprocVecEnv）。
*   它使用最基础的 Discounted Reward 计算，而不是复杂的 GAE。
*   网络结构极其简单（只有两层）。
*   所有逻辑都在一个文件内，适合理解原理。
