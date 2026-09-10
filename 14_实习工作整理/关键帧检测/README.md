# 关键帧检测

本目录整理关键帧检测模型训练主项目，覆盖任务定义、标注标准、bad case 归因、数据飞轮、CoT/RL、部署上线和容量评估。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [关键帧检测全流程.md](<关键帧检测全流程.md>) | 从数据来源、任务定义、Qwen 输入、冷启动、SFT、调参、评测、bad case 到 CoT/RL 和上线的完整闭环。 |
| [模型训练总览.md](<模型训练总览.md>) | 项目整体背景、核心结果、训练数据、飞轮、基座优化和上线收益。 |
| [任务定义与标注标准.md](<任务定义与标注标准.md>) | task_type 完成态、排除条件、豁免条件和证据优先级。 |
| [BadCase归因与UI元素治理.md](<BadCase归因与UI元素治理.md>) | 小面积 UI、购物车角标、营销 Banner、边缘小图和 GT 标注误差治理。 |
| [数据飞轮与主动学习.md](<数据飞轮与主动学习.md>) | 多模型筛 bad case、人工重标、分桶评测、采样配比和高优指标治理。 |
| [并行DE与PE.md](<并行DE与PE.md>) | 数据侧治理和 prompt/规则侧治理的并行闭环，以及相关面试问答。 |
| [CoT蒸馏与RL方案.md](<CoT蒸馏与RL方案.md>) | 关键帧 CoT 数据蒸馏与质量提升方案：Video Primitive、Evidence Chain、人工抽检、质量分层、结构化 SFT 和 GRPO/RL。 |
| [Thinking with Visual Primitives 视觉原语推理.md](<Thinking with Visual Primitives 视觉原语推理.md>) | 视觉原语推理论文总结，以及对关键帧 Region/State/Event Primitive、CoT 和 GRPO 的迁移。 |
| [Qwen3-VL关键帧数据格式.md](<Qwen3-VL关键帧数据格式.md>) | 当前关键帧任务从标准 JSONL 到 qwen3 JSONL 的数据结构、字段含义、模态绑定和 Qwen 实际接收信息。 |
| [ms-swift关键帧训练与推理.md](<ms-swift关键帧训练与推理.md>) | 当前 train_video.sh 的 Qwen3.5 full SFT、分布式配置、推理分片、参数含义和结果合并逻辑。 |
| [部署上线与容量评估.md](<部署上线与容量评估.md>) | 接口字段、链路耗时、图片下载瓶颈、并发压测、QPM 和线上回测。 |

## 复习顺序

1. 先看 [关键帧检测全流程.md](<关键帧检测全流程.md>)，建立从数据到上线的完整主线。
2. 再看 [模型训练总览.md](<模型训练总览.md>)，补充项目背景、结果和业务收益。
3. 再看 [任务定义与标注标准.md](<任务定义与标注标准.md>) 和 [BadCase归因与UI元素治理.md](<BadCase归因与UI元素治理.md>)，深入任务难点和标注边界。
4. 接着看 [Qwen3-VL关键帧数据格式.md](<Qwen3-VL关键帧数据格式.md>) 和 [ms-swift关键帧训练与推理.md](<ms-swift关键帧训练与推理.md>)，掌握数据输入、训练参数和推理链路。
5. 然后看 [数据飞轮与主动学习.md](<数据飞轮与主动学习.md>)、[并行DE与PE.md](<并行DE与PE.md>)、[Thinking with Visual Primitives 视觉原语推理.md](<Thinking with Visual Primitives 视觉原语推理.md>) 和 [CoT蒸馏与RL方案.md](<CoT蒸馏与RL方案.md>)，深入数据治理、视觉原语、CoT 蒸馏和 RL。
6. 最后看 [部署上线与容量评估.md](<部署上线与容量评估.md>)，准备工程落地和线上排查追问。
