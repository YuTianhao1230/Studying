# 训练优化与对齐

本目录存放模型训练路线、训练工程、后训练对齐、参数体系和训练稳定性内容。分类依据是先建立训练总纲，再按训练阶段、训练系统、参数/优化目标和故障排查拆分。

| 子目录 | 内容说明 |
| --- | --- |
| [后训练与对齐](<后训练与对齐/README.md>) | SFT、RLHF、DPO、PPO、GRPO、RLVR、PEFT、Model Merging、Agentic RL、Reward Model 等后训练方法。 |
| [训练框架与并行](<训练框架与并行/README.md>) | DeepSpeed、ZeRO、FSDP、Megatron-LM、JAX/XLA、混合精度、Checkpoint、集合通信和分布式故障排查。 |
| [训练稳定性](<训练稳定性/README.md>) | Loss 异常、收敛排查、梯度爆炸和梯度消失等训练故障。 |
| [参数](<参数/README.md>) | 学习率、batch、warmup、optimizer、loss、activation、LoRA 参数等训练超参数和优化组件。 |

## 当前层文件

| 文件 | 内容说明 |
| --- | --- |
| [模型训练学习手册_预训练到后训练.md](<模型训练学习手册_预训练到后训练.md>) | 从预训练、SFT、DPO、GRPO/RLVR 到评测部署的训练路线总纲。 |
| [训练优化与大模型训练.md](<训练优化与大模型训练.md>) | 大模型训练中的优化目标、训练阶段和工程问题。 |
