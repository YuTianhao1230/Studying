# 深度学习基础

本目录整理神经网络入门到训练基础组件，适合在学习 Transformer、大模型训练和多模态模型之前先复习。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [深度学习基础.md](<深度学习基础.md>) | 神经网络、反向传播、优化和训练流程的总览。 |
| [Multi-Layer Perceptron.md](<Multi-Layer Perceptron.md>) | 多层感知机的结构、非线性表达能力和基础训练方式。 |
| [Feature Map.md](<Feature Map.md>) | 特征图的含义，以及在 CNN/视觉模型中的空间表示作用。 |
| [高阶特征.md](<高阶特征.md>) | 从低层纹理到高层语义特征的表示层级。 |
| [Normalization.md](<Normalization.md>) | BatchNorm、LayerNorm 等归一化方法的目的和差异。 |
| [正则化.md](<正则化.md>) | Dropout、权重衰减、数据增强等缓解过拟合的方法。 |
| [常见激活函数.md](<常见激活函数.md>) | Sigmoid、Tanh、ReLU、GELU、SiLU 等非线性函数的性质与选择。 |
| [GeLU.md](<GeLU.md>) | GELU 的数学形式、近似计算和 Transformer 应用。 |
| [常见分类损失函数.md](<常见分类损失函数.md>) | 交叉熵、Focal Loss、KL 散度等分类目标。 |
| [常见回归损失函数.md](<常见回归损失函数.md>) | MSE、MAE、Huber、分位数损失等回归目标。 |
| [LSTM.md](<LSTM.md>) | 循环神经网络的门控、记忆机制及与 Transformer 的差异。 |
| [参数初始化与数值稳定性.md](<参数初始化与数值稳定性.md>) | Xavier/He 初始化、稳定 Softmax、LogSumExp 和融合交叉熵。 |

## 学习路线

1. 先看 [深度学习基础.md](<深度学习基础.md>) 和 [Multi-Layer Perceptron.md](<Multi-Layer Perceptron.md>)。
2. 再看 [Feature Map.md](<Feature Map.md>) 和 [高阶特征.md](<高阶特征.md>)，理解表示学习。
3. 结合 [常见激活函数.md](<常见激活函数.md>)、[GeLU.md](<GeLU.md>)、[常见分类损失函数.md](<常见分类损失函数.md>) 和 [常见回归损失函数.md](<常见回归损失函数.md>)，掌握网络输出与训练目标。
4. 再看 [参数初始化与数值稳定性.md](<参数初始化与数值稳定性.md>)、[Normalization.md](<Normalization.md>) 和 [正则化.md](<正则化.md>)，连接训练稳定性和泛化能力。
5. 用 [LSTM.md](<LSTM.md>) 理解循环序列建模，再进入02的 Transformer 架构。
