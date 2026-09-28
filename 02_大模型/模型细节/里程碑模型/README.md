# 里程碑模型

本目录按代表性模型整理大模型与多模态架构的关键演进。每张卡片重点回答模型解决了什么问题、核心结构如何工作、训练目标是什么，以及它对后续模型产生了什么影响。

## 内容索引

| 模型 | 核心内容 |
| --- | --- |
| [BERT](<BERT.md>) | Encoder-only、双向掩码语言建模和预训练后微调范式。 |
| [GPT](<GPT.md>) | Decoder-only、自回归预训练、规模化生成与上下文学习。 |
| [T5](<T5.md>) | Encoder-Decoder 和统一的 text-to-text 任务形式。 |
| [Llama](<Llama.md>) | 开放权重语言模型、现代 Decoder-only 组件与高质量数据训练。 |
| [DeepSeek](<DeepSeek.md>) | MoE、MLA、推理能力训练和效率优化路线。 |
| [ViT](<ViT.md>) | 图像 Patch 序列化及 Transformer 视觉建模。 |
| [CLIP](<CLIP.md>) | 图文对比学习、开放词表识别和跨模态表示对齐。 |
| [ALBEF](<ALBEF.md>) | 图文对齐后融合、动量蒸馏和多阶段多模态预训练。 |
| [TCL](<TCL.md>) | 细粒度图文对比与跨模态一致性学习。 |
| [BLIP 与 BLIP-2](<BLIP与BLIP2.md>) | 图文理解与生成统一、Q-Former 连接视觉编码器和 LLM。 |
| [Flamingo](<Flamingo.md>) | 视觉语言少样本学习、Perceiver Resampler 和交叉注意力。 |
| [LLaVA](<LLaVA.md>) | 视觉指令微调、视觉投影器和多模态对话。 |

## 推荐学习顺序

1. 先按 [BERT](<BERT.md>) → [GPT](<GPT.md>) → [T5](<T5.md>) 对比三类主干架构和预训练目标。
2. 再看 [Llama](<Llama.md>) 与 [DeepSeek](<DeepSeek.md>)，理解现代开放模型在架构、数据和效率上的演进。
3. 从 [ViT](<ViT.md>)、[CLIP](<CLIP.md>) 进入视觉与图文对齐，再学习 [ALBEF](<ALBEF.md>)、[TCL](<TCL.md>) 和 [BLIP 与 BLIP-2](<BLIP与BLIP2.md>)。
4. 最后用 [Flamingo](<Flamingo.md>) 与 [LLaVA](<LLaVA.md>) 理解视觉能力如何接入大语言模型。

跨模型的历史关系见 [阶段性里程碑模型详解](<../阶段性里程碑模型详解.md>)；当前主流模型系列和选型见上一级 [模型细节](<../README.md>)。
