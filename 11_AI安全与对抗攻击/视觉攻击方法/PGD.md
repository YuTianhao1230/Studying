# PGD

## 知识点解析

### 概述

PGD 是带随机初始化的多步投影梯度攻击，常被视为一阶白盒攻击下的强基线，也是[对抗训练](<../防御与鲁棒性/对抗训练.md>)中最常用的内层攻击方法之一。

### 解决的问题

I-FGSM 从原始样本出发，可能受单一起点限制。PGD 在扰动球内随机初始化，再做多步投影梯度更新，更充分地搜索局部最坏扰动。

### 攻击公式

随机初始化：

$$
x_{\mathrm{adv}}^{(0)}=x+u,
\qquad
u\sim\operatorname{Uniform}([-\epsilon,\epsilon]^d)
$$

迭代更新：

$$
x_{\mathrm{adv}}^{(t+1)}
=\Pi_{B_\infty(x,\epsilon)}
\left(
x_{\mathrm{adv}}^{(t)}
+\alpha\,\operatorname{sign}
\left(\nabla_x L(f(x_{\mathrm{adv}}^{(t)}),y)\right)
\right)
$$

最后还要裁剪到合法像素范围：

$$
x_{\mathrm{adv}}=\operatorname{clip}(x_{\mathrm{adv}},0,1)
$$

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
| [FGSM](<FGSM.md>) | 原图 | 单步 | 快，但攻击较弱 |
| I-FGSM | 原图 | 多步投影 | 白盒更强 |
| PGD | 扰动球内随机点 | 多步投影 | 更强白盒基线 |

### 在对抗训练中的作用

对抗训练可以写成 min-max：

$$
\min_{\theta}\;
\mathbb{E}_{(x,y)}
\left[
\max_{\lVert\delta\rVert\le\epsilon}
L(f_\theta(x+\delta),y)
\right]
$$

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

### 1. PGD 为什么常作为一阶白盒强攻击基线？

回答思路：写出随机初始化和投影梯度更新，解释多起点、多步搜索比单步或固定起点更充分，但不要把“强基线”表述成全局最优保证。

回答模板：

> PGD 先在 $B_\infty(x,\epsilon)$ 内随机初始化，再反复执行 $x_{\mathrm{adv}}\leftarrow\Pi(x_{\mathrm{adv}}+\alpha\,\operatorname{sign}(\nabla_x L))$。多步更新能沿非线性损失面持续搜索，随机起点和随机重启则减少固定起点陷入较差局部区域的风险，因此它通常比 FGSM 和固定起点的 I-FGSM 更适合作为一阶白盒基线。不过 PGD 只是在给定梯度、步数和重启次数下近似求解内层最大化，不能宣称一定找到全局最坏扰动。

### 2. PGD 在攻击评估和对抗训练中分别扮演什么角色？

回答思路：用同一个 min-max 目标解释两种用途，区分评估时固定模型找反例与训练时交替更新模型参数。

回答模板：

> 在鲁棒评估中，模型参数固定，PGD 用来近似寻找预算内使损失最大的扰动，以检验模型是否存在局部脆弱点。在对抗训练中，目标是 $\min_\theta\mathbb{E}[\max_{\lVert\delta\rVert\le\epsilon}L(f_\theta(x+\delta),y)]$：PGD 近似求内层最大化，优化器再对生成的对抗样本更新 `theta`，完成外层最小化。两者使用同一类攻击，但训练关注模型参数学习，评估关注在充分攻击下测得可信的 robust accuracy。

### 3. 防御没有被 PGD 打穿，为什么仍不能直接认定鲁棒？

回答思路：从攻击配置不足、随机性和 Gradient Masking 三类失败来源回答，并给出增强评估的操作。

回答模板：

> 单个 PGD 配置失败可能只是步长、步数或损失函数不合适，也可能被非光滑、随机化预处理造成的梯度遮蔽误导。我会增加迭代数和随机重启，扫描步长，比较不同损失，并检查攻击强度是否随预算和步数合理变化；对不可微操作使用 BPDA，对随机防御使用 EOT，还要补充无梯度或迁移攻击。只有多种自适应攻击都无法显著降低鲁棒指标，才能更有把握地支持防御有效。
