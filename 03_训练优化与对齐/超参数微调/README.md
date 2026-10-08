# 超参数微调

本目录整理训练中最常见的超参数、优化器与参数高效微调方法，包括 learning rate、batch、warmup、optimizer、weight decay、gradient clipping、PEFT、Adapter 和 LoRA。通用损失函数与激活函数由01的[深度学习基础](<../../01_机器学习基础/深度学习基础/README.md#深度学习基础>)统一维护。

索引按“通用调参框架与 SFT 实例 -> 优化器更新原理 -> 参数高效微调”排列。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [训练超参数调参指南.md](<训练超参数调参指南.md#训练超参数调参指南>) | 学习率、batch、warmup、weight decay、LoRA 等通用调参方法，以及 SFT 的具体默认值、调整顺序和配置示例。 |
| [Optimizer 优化器.md](<Optimizer 优化器.md#optimizer-优化器>) | 解释 SGD、Momentum、Adam、AdamW、Adafactor 的更新逻辑和大模型微调中的选择。 |
| [PEFT 参数高效微调.md](<PEFT 参数高效微调.md#peft-参数高效微调>) | 介绍冻结基座、adapter、prefix 和 soft prompt 等参数高效微调方法。 |
| [Adapter 参数高效微调.md](<Adapter 参数高效微调.md#adapter-参数高效微调>) | 介绍 Bottleneck Adapter 的结构、插入位置、训练方式及与 LoRA、Prefix Tuning 的区别。 |
| [LoRA 低秩适配.md](<LoRA 低秩适配.md#lora-低秩适配>) | 介绍 LoRA 的低秩旁路、训练配置、权重合并和常见适用边界。 |
