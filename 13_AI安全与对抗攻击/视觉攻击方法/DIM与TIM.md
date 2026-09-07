# DIM 与 TIM

## 知识点解析

### 概述

DIM 和 TIM 是提升视觉对抗样本迁移性的常见方法。DIM 通过随机输入变换降低对固定输入形态的过拟合，TIM 通过平移不变梯度平滑降低对具体空间位置的过拟合。

### 解决的问题

迭代攻击容易在源模型、固定分辨率和固定预处理上过拟合。黑盒迁移时，目标模型可能有不同输入尺寸、裁剪方式、架构和感受野。DIM/TIM 的目标是让扰动更稳定、更通用。

### DIM

DIM 全称 Diverse Input Method。核心是在每次求梯度前，对输入做随机变换：

```text
x_input = random_transform(x_adv)
gradient = grad_x L(f(x_input), y)
x_adv = update(x_adv, gradient)
```

常见变换：

- random resize
- random padding
- random crop
- scale jitter

直觉：如果一个扰动在多种输入变换后都能攻击源模型，它更不容易只适配固定像素位置和固定预处理，因此更可能迁移。

### TIM

TIM 全称 Translation-Invariant Method。它把梯度和一个卷积核做卷积，得到平滑后的梯度：

```text
g_smooth = W * grad_x L
x_adv = x_adv + alpha * sign(g_smooth)
```

其中 `W` 可以是均值核或高斯核。直觉是让扰动在空间平移后仍有效，减少对某个具体位置的依赖。

### 常见组合

DIM、TIM 常和 MI-FGSM 组合：

```text
M-DI-TI-FGSM
  = Momentum
  + Diverse Input
  + Translation-Invariant gradient
```

组合后通常更适合黑盒迁移攻击：

- Momentum 稳定优化方向。
- Diverse Input 提升尺度和预处理鲁棒性。
- Translation-Invariant 让梯度更空间稳定。

### 适用场景

- 跨 CNN/ViT 架构迁移。
- 源模型和目标模型预处理不同。
- 图像分类迁移攻击。
- 多模态攻击中的图像分支扰动增强。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 概念题 | DIM/TIM 分别做什么 | DIM 随机输入变换，TIM 平滑梯度 |
| 原因题 | 为什么能提升迁移 | 减少源模型和固定预处理过拟合 |
| 组合题 | M-DI-TI-FGSM 怎么理解 | 动量 + 输入多样性 + 平移不变 |
| 项目题 | 为什么多模态攻击使用 SIA/TI/DI | 提升图像扰动跨模型迁移能力 |

### 易错点

- 把 DIM 当成数据增强训练；它是在攻击过程中对输入变换求梯度。
- 把 TIM 当成对图像模糊；TIM 平滑的是梯度，不一定直接模糊图像。
- 忽略随机变换概率，导致每一步都变换或从不变换，和论文设定不一致。
- 只看源模型白盒 ASR，不看目标模型 transfer ASR。

## 面试应对

### DIM 和 TIM 为什么能提升迁移攻击？

回答思路：从“减少过拟合”解释。

回答模板：

DIM 和 TIM 都是在解决迭代攻击过拟合源模型的问题。DIM 在每次求梯度前对输入做随机 resize、padding 等变换，让扰动不能只适配固定输入尺度和预处理；TIM 则对梯度做卷积平滑，让攻击方向对空间平移更稳定。它们通常和动量攻击结合，形成 M-DI-TI-FGSM，从方向稳定性、输入多样性和空间平移不变性三个角度提升黑盒迁移性。
