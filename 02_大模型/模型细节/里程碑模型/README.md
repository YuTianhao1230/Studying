# 里程碑模型

本目录按代表性模型整理大模型与多模态架构的关键演进。每张卡片重点回答模型解决了什么问题、核心结构如何工作、训练目标是什么，以及它对后续模型产生了什么影响。

建议按索引顺序从语言模型主干进入视觉与图文预训练，再学习视觉能力接入大语言模型的代表方案；跨模型关系由上一级 [阶段性里程碑模型详解](<../阶段性里程碑模型详解.md#阶段性里程碑模型详解>) 统一归纳，当前主流系列与选型见 [模型细节](<../README.md#模型细节>)。

## 内容索引

| 模型 | 核心内容 |
| --- | --- |
| [BERT](<BERT.md#bert>) | Encoder-only、双向掩码语言建模和预训练后微调范式。 |
| [GPT](<GPT.md#gpt>) | Decoder-only、自回归预训练、规模化生成与上下文学习。 |
| [T5](<T5.md#t5>) | Encoder-Decoder 和统一的 text-to-text 任务形式。 |
| [Llama](<Llama.md#llama-架构>) | 开放权重语言模型、现代 Decoder-only 组件与高质量数据训练。 |
| [DeepSeek](<DeepSeek.md#deepseek-架构>) | MoE、MLA、推理能力训练和效率优化路线。 |
| [ViT](<ViT.md#vit>) | 图像 Patch 序列化及 Transformer 视觉建模。 |
| [CLIP](<CLIP.md#clip>) | 图文对比学习、开放词表识别和跨模态表示对齐。 |
| [ALBEF](<ALBEF.md#albef>) | 图文对齐后融合、动量蒸馏和多阶段多模态预训练。 |
| [TCL](<TCL.md#tcl>) | 细粒度图文对比与跨模态一致性学习。 |
| [BLIP 与 BLIP-2](<BLIP与BLIP2.md#blip-与-blip-2>) | 图文理解与生成统一、Q-Former 连接视觉编码器和 LLM。 |
| [Flamingo](<Flamingo.md#flamingo>) | 视觉语言少样本学习、Perceiver Resampler 和交叉注意力。 |
| [LLaVA](<LLaVA.md#llava>) | 视觉指令微调、视觉投影器和多模态对话。 |
