# 后训练与对齐

本目录覆盖预训练后的指令微调、偏好对齐、强化学习、能力迁移与参数高效适配，索引顺序从全局框架和冷启动逐步进入偏好优化、可验证强化、蒸馏与能力融合。

通用强化学习原理由[强化学习基础](<../../01_机器学习基础/概率与序列决策/强化学习基础.md#强化学习基础>)统一维护；各方法的定义、关系和边界由对应知识卡及[后训练发展史与方法对比](<后训练发展史与方法对比.md#后训练发展史与方法对比>)承载。

## 内容索引

| 文件 | 核心含义 |
| --- | --- |
| [后训练发展史与方法对比.md](<后训练发展史与方法对比.md#后训练发展史与方法对比>) | 后训练的定义、发展脉络、方法关系、资源与交付选型条件及建议学习顺序 |
| [SFT 监督微调.md](<SFT 监督微调.md#sft-监督微调>) | 用高质量指令-回答数据做监督训练，让模型学会任务格式和基础回答方式 |
| [强化学习基础.md](<../../01_机器学习基础/概率与序列决策/强化学习基础.md#强化学习基础>) | 通用 RL 权威卡：MDP、值方法、策略梯度、TD/GAE 和采样分布校正 |
| [RLHF 基于人类反馈的强化学习.md](<RLHF 基于人类反馈的强化学习.md#rlhf-基于人类反馈的强化学习>) | 用人类偏好训练奖励模型，再通过强化学习优化策略模型 |
| [DPO 直接偏好优化.md](<DPO 直接偏好优化.md#dpo-直接偏好优化>) | 直接用 chosen/rejected 偏好对优化模型，降低 RLHF 工程复杂度 |
| [PPO 近端策略优化.md](<PPO 近端策略优化.md#ppo-近端策略优化>) | 用裁剪 surrogate 减少过度更新激励，含优势符号算例与 rollout 边界处理 |
| [RFT 拒绝采样微调.md](<RFT 拒绝采样微调.md#rft-拒绝采样微调>) | 先生成候选并用 reward/verifier 筛选，再将高质量回答用于继续 SFT |
| [GRPO 组相对策略优化.md](<GRPO 组相对策略优化.md#grpo-组相对策略优化>) | 用同一 prompt 下多条回答的组内相对 reward 更新模型，常用于推理 RL |
| [RLVR 可验证奖励强化学习.md](<RLVR 可验证奖励强化学习.md#rlvr-可验证奖励强化学习>) | 用数学判题、单测、schema、工具结果等可验证信号作为 reward |
| [Reward Model 与 Grader 奖励模型与评分器.md](<Reward Model 与 Grader 奖励模型与评分器.md#reward-model-与-grader-奖励模型与评分器>) | 负责给模型输出、候选答案或轨迹打分的偏好模型或规则评分器 |
| [Reward Collapse 奖励坍缩.md](<Reward Collapse 奖励坍缩.md#reward-collapse-奖励坍缩>) | 模型通过钻 reward 漏洞获得高分，但真实质量下降的现象 |
| [Knowledge Distillation 知识蒸馏.md](<Knowledge Distillation 知识蒸馏.md#knowledge-distillation-知识蒸馏>) | 让小模型学习强模型输出、推理轨迹或分布的能力迁移方法 |
| [On-Policy Distillation 在线策略蒸馏.md](<On-Policy Distillation 在线策略蒸馏.md#on-policy-distillation-在线策略蒸馏>) | 让学生在自己的 rollout 状态上接受教师 token-level 软监督的蒸馏方法 |
| [PEFT 参数高效微调.md](<PEFT 参数高效微调.md#peft-参数高效微调>) | 冻结大模型主体，只训练少量 adapter、prefix 或 soft prompt 的高效微调方法体系 |
| [Adapter 参数高效微调.md](<Adapter 参数高效微调.md#adapter-参数高效微调>) | Bottleneck Adapter 的结构、插入位置、训练方式以及与 LoRA/Prefix Tuning 的区别 |
| [LoRA 低秩适配.md](<LoRA 低秩适配.md#lora-低秩适配>) | 参数高效微调方法，通过低秩矩阵适配大模型 |
| [Model Merging 模型合并.md](<Model Merging 模型合并.md#model-merging-模型合并>) | 在权重空间合并多个模型或 adapter，在不增加推理成本的情况下融合能力 |
| [Agentic RL 智能体强化学习.md](<Agentic RL 智能体强化学习.md#agentic-rl-智能体强化学习>) | 针对 Agent 多步工具调用、计划、观察和执行轨迹进行强化学习 |
