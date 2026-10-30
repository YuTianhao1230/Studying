# 方案与训练流程

本目录整理关键帧项目的完整训练方案及各阶段设计，覆盖 SFT、CoT-SFT、GRPO、Temporal-OPSD 和视觉原语。索引从完整主线逐步深入到各阶段原理与实现。

## 内容索引

| 文件 | 内容 |
| --- | --- |
| [关键帧检测完整训练方案.md](<关键帧检测完整训练方案.md>) | 只描述从任务、数据、baseline、SFT、CoT-SFT、GRPO、Temporal-OPSD 到评测上线的完整流程。 |
| [SFT 训练方案设计.md](<SFT 训练方案设计.md>) | Direct SFT 和 Structured CoT SFT 的冷启动、参数、收敛及问题排查。 |
| [CoT-SFT 蒸馏方案设计.md](<CoT-SFT 蒸馏方案设计.md>) | CoT 数据构造、Video Primitive、证据链、质量治理和训练使用。 |
| [GRPO 训练方案设计.md](<GRPO 训练方案设计.md>) | rollout、verifier、reward、参数、监控、失败排查和方法选型。 |
| [OPD 训练方案设计.md](<OPD 训练方案设计.md>) | Temporal-OPSD 的原理、数据构造、双视图训练、loss、训练器、评估和面试应对。 |
| [Thinking with Visual Primitives 视觉原语推理.md](<Thinking with Visual Primitives 视觉原语推理.md>) | 视觉原语论文总结及其对关键帧 CoT 的迁移。 |
