# 方案与训练流程

本目录只保留关键帧项目的训练方案设计，主线是：

```text
Direct SFT
  -> Structured CoT SFT
  -> RFT baseline
  -> Verifier-based GRPO
  -> 可选 OPD / answer-only distillation
```

## 内容索引

| 文件 | 内容 |
| --- | --- |
| [关键帧检测完整训练方案.md](<关键帧检测完整训练方案.md>) | 从任务、数据、baseline、SFT、CoT-SFT、GRPO、评测到上线的唯一主方案。 |
| [SFT 训练方案设计.md](<SFT 训练方案设计.md>) | Direct SFT 和 Structured CoT SFT 的冷启动、参数、收敛及问题排查。 |
| [CoT-SFT 蒸馏方案设计.md](<CoT-SFT 蒸馏方案设计.md>) | CoT 数据构造、Video Primitive、证据链、质量治理和训练使用。 |
| [GRPO 训练方案设计.md](<GRPO 训练方案设计.md>) | rollout、verifier、reward、参数、监控、失败排查和方法选型。 |
| [Thinking with Visual Primitives 视觉原语推理.md](<Thinking with Visual Primitives 视觉原语推理.md>) | 视觉原语论文总结及其对关键帧 CoT 的迁移。 |

## 阅读顺序

1. 先看完整训练方案，建立主线。
2. 再看 SFT 和 CoT-SFT，理解前两阶段为什么这样设计。
3. 然后看 GRPO，理解 reward、rollout 和方法选型。
4. 最后看 Visual Primitives，补充 CoT 设计的研究来源。
