# 参数

本目录整理训练中最常见的参数与优化组件，包括 learning rate、batch、warmup、optimizer、loss、activation、LoRA 参数等。它们共同决定模型每一步如何更新、训练是否稳定、是否过拟合，以及最终效果和训练成本。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [训练超参数调参指南.md](<训练超参数调参指南.md>) | 学习率、batch、warmup、weight decay、LoRA 等超参数的系统调参指南。 |
| [SFT 超参数怎么设与怎么调.md](<SFT 超参数怎么设与怎么调.md>) | 从一份真实 SFT 脚本出发，讲清 lr/warmup/epoch/梯度累积每个值为什么这么设、怎么按 loss 调。 |
| [Optimizer 优化器.md](<Optimizer 优化器.md>) | 解释 SGD、Momentum、Adam、AdamW、Adafactor 的更新逻辑和大模型微调中的选择。 |
| [常见分类损失函数.md](<常见分类损失函数.md>) | 交叉熵、二分类、多分类和分类任务常见 loss。 |
| [常见回归损失函数.md](<常见回归损失函数.md>) | MSE、MAE、Huber 等回归损失及适用场景。 |
| [常见激活函数.md](<常见激活函数.md>) | Sigmoid、Tanh、ReLU、GELU 等激活函数对比。 |
| [GeLU.md](<GeLU.md>) | GeLU 的平滑非线性特征及在 Transformer 中的应用。 |

## 学习路线

1. 先看 [训练超参数调参指南.md](<训练超参数调参指南.md>)，建立 learning rate、batch、epoch、warmup、weight decay、gradient clipping 等参数的总框架。
2. 再看 [SFT 超参数怎么设与怎么调.md](<SFT 超参数怎么设与怎么调.md>)，把通用参数落到真实 SFT 脚本和 loss 曲线调整上。
3. 然后看 [Optimizer 优化器.md](<Optimizer 优化器.md>)，理解 SGD、Adam、AdamW 等优化器如何真正更新参数。
4. 接着看 [常见分类损失函数.md](<常见分类损失函数.md>) 和 [常见回归损失函数.md](<常见回归损失函数.md>)，理解训练目标和 loss 设计。
5. 最后看 [常见激活函数.md](<常见激活函数.md>) 与 [GeLU.md](<GeLU.md>)，理解非线性表达和 Transformer FFN 中的常用激活。
