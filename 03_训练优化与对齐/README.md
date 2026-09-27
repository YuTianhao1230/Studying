# 训练优化与对齐

本目录存放模型训练路线、训练工程、后训练对齐、参数体系和训练稳定性内容。分类依据是先建立训练总纲，再按训练阶段、训练系统、参数/优化目标和故障排查拆分。

| 子目录 | 内容说明 |
| --- | --- |
| [后训练与对齐](<后训练与对齐/README.md>) | SFT、RFT、RLHF、DPO、PPO、GRPO、RLVR、Response/On-Policy Distillation、PEFT、Adapter、Model Merging、Agentic RL、Reward Model 等后训练方法。 |
| [训练框架与并行](<训练框架与并行/README.md>) | DeepSpeed、ZeRO、FSDP、Megatron-LM、JAX/XLA、混合精度、Checkpoint、多 GPU 通信、吞吐和分布式故障排查。 |
| [训练稳定性](<训练稳定性/README.md>) | Loss 异常、收敛排查、梯度爆炸和梯度消失等训练故障。 |
| [超参数与优化器](<超参数与优化器/README.md>) | 学习率、batch、warmup、optimizer、weight decay、gradient clipping 和 LoRA 参数。 |
| [笔试训练](<笔试训练/README.md>) | AdamW、Warmup、梯度裁剪、混合精度和训练稳定性专项题。 |

强化学习建议按 [RL 基础](<../01_机器学习基础/概率与序列决策/强化学习基础.md>) → [PPO](<后训练与对齐/PPO 近端策略优化.md>) → [GRPO](<后训练与对齐/GRPO 组相对策略优化.md>) 阅读：先推导回报和优势，再理解策略更新与奖励设计。

训练工程建议按 [超参数与优化器](<超参数与优化器/README.md>) → [训练框架与并行](<训练框架与并行/README.md>) → [训练稳定性](<训练稳定性/README.md>) 阅读；从新任务到上线的阶段选择见[模型训练路线](<模型训练学习手册_预训练到后训练.md>)。

## 当前层文件

| 文件 | 内容说明 |
| --- | --- |
| [模型训练学习手册_预训练到后训练.md](<模型训练学习手册_预训练到后训练.md>) | 按任务定义、底座、数据、SFT、偏好、RLVR、评测和部署组织的训练路线导航。 |
