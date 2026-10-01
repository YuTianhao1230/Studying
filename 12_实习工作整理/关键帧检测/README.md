# 关键帧检测

本目录整理关键帧检测项目，覆盖完整训练方案、任务与数据治理、分阶段训练设计、工程实现和研究参考。索引按从项目全貌、业务规则、训练方法到上线落地的顺序排列。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [关键帧检测完整训练方案.md](<方案与训练流程/关键帧检测完整训练方案.md>) | 任务、数据、baseline、SFT、CoT-SFT、GRPO、Temporal-OPSD、评测回流和上线的完整主线。 |
| [任务定义与标注标准.md](<任务与数据治理/任务定义与标注标准.md>) | 完成态、排除与豁免条件、标注边界和证据优先级。 |
| [BadCase归因与UI元素治理.md](<任务与数据治理/BadCase归因与UI元素治理.md>) | bad case 分类、小 UI、角标、Banner、GT 误差和 UI primitive。 |
| [SFT 训练方案设计.md](<方案与训练流程/SFT 训练方案设计.md>) | Direct SFT 与 Structured CoT SFT 的冷启动、参数、收敛和训练排查。 |
| [CoT-SFT 蒸馏方案设计.md](<方案与训练流程/CoT-SFT 蒸馏方案设计.md>) | CoT 数据构造、Video Primitive、证据链、质量治理和结构化监督。 |
| [GRPO 训练方案设计.md](<方案与训练流程/GRPO 训练方案设计.md>) | rollout、verifier、reward、参数、监控和方法选择。 |
| [OPD 训练方案设计.md](<方案与训练流程/OPD 训练方案设计.md>) | Temporal-OPSD 的双视图输入、on-policy logits、联合损失、训练器和评估协议。 |
| [数据飞轮与主动学习.md](<任务与数据治理/数据飞轮与主动学习.md>) | bad case 筛选、人工重标、分桶评测和主动采样。 |
| [并行DE与PE.md](<任务与数据治理/并行DE与PE.md>) | 数据工程和 Prompt/规则工程的并行治理闭环。 |
| [Agentic Model Optimization 模型自训练优化.md](<任务与数据治理/Agentic Model Optimization 模型自训练优化.md>) | Agent 接管数据、训练、评测和持续迭代的优化闭环。 |
| [Qwen3-VL关键帧数据格式.md](<工程实现/Qwen3-VL关键帧数据格式.md>) | JSONL、多模态字段、视频映射和标签格式。 |
| [ms-swift关键帧训练与推理.md](<工程实现/ms-swift关键帧训练与推理.md>) | ms-swift、Qwen3.5、full SFT、DeepSpeed、推理分片和参数。 |
| [部署上线与容量评估.md](<工程实现/部署上线与容量评估.md>) | 接口、下载瓶颈、并发压测、QPM 和线上回测。 |
| [Thinking with Visual Primitives 视觉原语推理.md](<方案与训练流程/Thinking with Visual Primitives 视觉原语推理.md>) | 视觉原语论文及其向关键帧时空证据链的迁移。 |
