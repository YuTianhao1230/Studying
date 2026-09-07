# PGD

## 知识点解析

### 概述

PGD 是带随机初始化的多步投影梯度攻击，常被视为一阶白盒攻击下的强基线，也是对抗训练中最常用的内层攻击方法之一。

### 解决的问题

I-FGSM 从原始样本出发，可能受单一起点限制。PGD 在扰动球内随机初始化，再做多步投影梯度更新，更充分地搜索局部最坏扰动。

### 攻击公式

随机初始化：

```text
x_adv_0 = x + Uniform(-epsilon, epsilon)
```

迭代更新：

```text
x_adv_{t+1} = Proj_{B_inf(x, epsilon)}(
    x_adv_t + alpha * sign(grad_x L(f(x_adv_t), y))
)
```

最后还要裁剪到合法像素范围：

```text
x_adv = clip(x_adv, 0, 1)
```

### PyTorch 骨架

```python
import torch
import torch.nn.functional as F


def pgd_linf_attack(model, images, labels, epsilon, alpha, steps):
    model.eval()
    original = images.detach()
    adversarial = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
    adversarial = adversarial.clamp(0, 1)

    for _ in range(steps):
        adversarial.requires_grad_(True)
        loss = F.cross_entropy(model(adversarial), labels)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = (adversarial - original).clamp(-epsilon, epsilon)
        adversarial = (original + delta).clamp(0, 1).detach()

    return adversarial
```

### 和 FGSM/I-FGSM 的区别

| 方法 | 起点 | 更新方式 | 特点 |
| --- | --- | --- | --- |
| FGSM | 原图 | 单步 | 快，但攻击较弱 |
| I-FGSM | 原图 | 多步投影 | 白盒更强 |
| PGD | 扰动球内随机点 | 多步投影 | 更强白盒基线 |

### 在对抗训练中的作用

对抗训练可以写成 min-max：

```text
min_theta E[ max_{||delta||<=epsilon} L(f_theta(x + delta), y) ]
```

PGD 近似求内层最大化，模型参数优化外层最小化。因此 PGD adversarial training 是经典鲁棒训练方法。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | PGD 为什么是强攻击 | 随机初始化 + 多步更新 + 投影 |
| 对抗训练题 | PGD 在 min-max 中做什么 | 近似求内层最坏扰动 |
| 参数题 | `epsilon/alpha/steps` 含义 | 总预算、单步步长、搜索次数 |
| 评估题 | PGD 没打穿是否说明鲁棒 | 不一定，要排除 gradient masking 和 adaptive attack |

### 易错点

- 忘记随机初始化，把 PGD 写成 I-FGSM。
- 不投影回 `epsilon` 球，导致攻击预算不公平。
- 使用太少 steps 得出防御有效结论。
- 没有重复随机初始化，可能低估攻击强度。

## 面试应对

### PGD 为什么常作为强攻击基线？

回答思路：解释随机初始化、多步投影和鲁棒评估意义。

回答模板：

PGD 是 Projected Gradient Descent attack，可以理解为带随机初始化的多步 FGSM。它先在扰动预算球内随机选一个起点，然后每一步沿增大损失的梯度符号方向更新，并投影回 `epsilon` 球内。相比 FGSM，PGD 能更充分搜索样本邻域中的高损失区域，所以常被认为是一阶白盒攻击下的强基线，也常用在对抗训练的内层最大化中。
