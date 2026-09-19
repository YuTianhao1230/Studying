# 方案与训练流程

本目录只保留关键帧项目的训练方案设计，主线是：

```text
Direct SFT
  -> Structured CoT SFT
  -> RFT baseline
  -> Verifier-based GRPO
  -> Temporal-OPSD On-Policy 蒸馏
  -> 评测、数据回流与持续迭代
```

## 内容索引

| 文件 | 内容 |
| --- | --- |
| [关键帧检测完整训练方案.md](<关键帧检测完整训练方案.md>) | 只描述从任务、数据、baseline、SFT、CoT-SFT、GRPO、Temporal-OPSD 到评测上线的完整流程。 |
| [SFT 训练方案设计.md](<SFT 训练方案设计.md>) | Direct SFT 和 Structured CoT SFT 的冷启动、参数、收敛及问题排查。 |
| [CoT-SFT 蒸馏方案设计.md](<CoT-SFT 蒸馏方案设计.md>) | CoT 数据构造、Video Primitive、证据链、质量治理和训练使用。 |
| [GRPO 训练方案设计.md](<GRPO 训练方案设计.md>) | rollout、verifier、reward、参数、监控、失败排查和方法选型。 |
| [OPD 训练方案设计.md](<OPD 训练方案设计.md>) | Temporal-OPSD 的原理、数据构造、双视图训练、loss、训练器、评估和面试应对。 |
| [Thinking with Visual Primitives 视觉原语推理.md](<Thinking with Visual Primitives 视觉原语推理.md>) | 视觉原语论文总结及其对关键帧 CoT 的迁移。 |

## 阅读顺序

1. 先看完整训练方案，建立主线。
2. 再看 SFT 和 CoT-SFT，理解前两阶段为什么这样设计。
3. 然后看 GRPO，理解 reward、rollout 和方法选型。
4. 回到完整训练方案确认 Temporal-OPSD 的阶段位置、准入准出和与评测回流的关系。
5. 阅读 [OPD 训练方案设计.md](<OPD 训练方案设计.md>)，深入学习双视图、on-policy logits、JSD/KL + CE、训练器和失败排查。
6. 最后看 Visual Primitives，补充 CoT 与 Video Primitive 的视觉证据表达。
