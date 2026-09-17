# FGSM

## 知识点解析

### 概述

FGSM 是单步梯度符号攻击方法，通过一次反向传播获得输入梯度，并在 `L_inf` 扰动约束下沿最能增大损失的方向生成对抗样本。

### 解决的问题

FGSM 解决的是“如何用最低成本生成对抗样本”的问题。它只需要一次 forward 和一次 backward，适合解释对抗样本原理、做快速鲁棒性测试和推导攻击公式。

### 攻击目标

非定向攻击：

```text
maximize L(f(x_adv), y)
subject to ||x_adv - x||_inf <= epsilon
```

定向攻击：

```text
minimize L(f(x_adv), y_target)
subject to ||x_adv - x||_inf <= epsilon
```

### 公式推导

对损失函数做一阶泰勒展开：

```text
L(x + delta, y) ~= L(x, y) + delta^T grad_x L(x, y)
```

在 `||delta||_inf <= epsilon` 下，为了最大化内积，每个维度都取梯度符号方向：

```text
delta = epsilon * sign(grad_x L)
x_adv = clip(x + delta)
```

定向攻击符号相反：

```text
x_adv = clip(x - epsilon * sign(grad_x L(f(x), y_target)))
```

### PyTorch 骨架

```python
import torch
import torch.nn.functional as F


def fgsm_attack(model, images, labels, epsilon):
    model.eval()
    images = images.detach().clone().requires_grad_(True)
    logits = model(images)
    loss = F.cross_entropy(logits, labels)
    gradient = torch.autograd.grad(loss, images)[0]
    adversarial = images + epsilon * gradient.sign()
    return adversarial.clamp(0, 1).detach()
```

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 公式推导 | 为什么是 `sign(gradient)` | 从一阶泰勒和 `L_inf` 约束推导 |
| 定向攻击 | 定向和非定向符号区别 | 非定向增大真实标签 loss，定向减小目标标签 loss |
| 预算理解 | `8/255` 是什么 | 图像归一化到 `[0,1]` 后每个像素最大变化 |
| 方法局限 | FGSM 为什么弱 | 单步线性近似粗糙，不能充分探索局部高损失区域 |

### 易错点

- 图像已归一化到 mean/std 后，直接加 `8/255` 会导致预算口径错误。
- 攻击时模型处于 train mode，BatchNorm/Dropout 造成不稳定。
- 只裁剪像素范围，不说明扰动预算。
- 定向攻击符号写反。

## 面试应对

### FGSM 是什么？

回答思路：定义、公式、推导、优缺点。

回答模板：

FGSM 是 Fast Gradient Sign Method，是一种单步白盒对抗攻击。它对输入求损失梯度，在 `L_inf` 约束下沿梯度符号方向加一个大小为 `epsilon` 的扰动，即 `x_adv = x + epsilon * sign(grad_x L)`。这个公式可以从损失的一阶泰勒展开推出来。它优点是非常快，只需要一次反向传播；缺点是单步近似比较粗，攻击强度通常不如 [PGD](<PGD.md>) 这类多步攻击。
