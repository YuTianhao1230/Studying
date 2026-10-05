# 超参数与优化器

本目录整理训练中最常见的超参数与优化器，包括 learning rate、batch、warmup、optimizer、weight decay、gradient clipping 和 LoRA 参数。通用损失函数与激活函数由01的[深度学习基础](<../../01_机器学习基础/深度学习基础/README.md>)统一维护。

索引按“通用调参框架与 SFT 实例 -> 优化器更新原理”排列。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [训练超参数调参指南.md](<训练超参数调参指南.md>) | 学习率、batch、warmup、weight decay、LoRA 等通用调参方法，以及 SFT 的具体默认值、调整顺序和配置示例。 |
| [Optimizer 优化器.md](<Optimizer 优化器.md>) | 解释 SGD、Momentum、Adam、AdamW、Adafactor 的更新逻辑和大模型微调中的选择。 |
