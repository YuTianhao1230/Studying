# 大模型

本目录存放大语言模型、视觉多模态模型和生成模型相关知识。分类依据是先理解通用架构与模型细节，再按应用问题、视觉多模态和生成任务展开；大模型发展史作为跨目录总览放在当前层。

| 子目录 | 内容说明 |
| --- | --- |
| [基础架构](<基础架构/README.md>) | Transformer、Self-Attention、RoPE、GQA、MoE、Dense Model 等结构概念。 |
| [应用与问题](<应用与问题/README.md>) | RAG、CoT、幻觉、Prompt 调优、NLP 与大模型关系等应用问题。 |
| [模型细节](<模型细节/README.md>) | Transformer、CLIP、ViT、ALBEF、TCL、Qwen、DeepSeek、GPT 等里程碑与 SOTA 模型详解。 |
| [视觉多模态与生成模型](<视觉多模态与生成模型/README.md>) | 视觉基础、VLM/视频/grounding、多模态任务和扩散生成模型。 |
| [笔试训练](<笔试训练/README.md>) | 重点覆盖基础架构、高效推理、预训练生成、RAG、模型选型和多模态专项题集。 |

## 当前层文件

| 文件 | 内容说明 |
| --- | --- |
| [大模型预训练与推理基础.md](<大模型预训练与推理基础.md>) | Tokenizer、预训练目标、数据配比、Scaling Law、长上下文、采样和推理时扩展。 |
| [大模型发展历史与SOTA迭代框架.md](<大模型发展历史与SOTA迭代框架.md>) | 从 GPT、Llama、Claude、Gemini、DeepSeek 等模型演进理解行业 SOTA 和训练范式变化。 |

## 模型细节重点入口

| 文件 | 内容说明 |
| --- | --- |
| [模型细节/阶段性里程碑模型详解.md](<模型细节/阶段性里程碑模型详解.md>) | Transformer、BERT、GPT、T5、ViT、CLIP、ALBEF、TCL、BLIP、Flamingo、LLaVA 等模型的架构、问题和影响。 |
| [模型细节/当前SOTA模型详解.md](<模型细节/当前SOTA模型详解.md>) | GPT/o、Claude、Gemini、Qwen、DeepSeek、Llama、Mistral 等模型系列的能力定位和架构特点。 |
| [模型细节/模型架构对比与选型.md](<模型细节/模型架构对比与选型.md>) | Encoder-only、Decoder-only、MoE、长上下文、多模态连接器、推理模型和业务选型。 |
| [模型细节/Qwen千问架构.md](<模型细节/Qwen千问架构.md>) | Qwen3 文本模型与 Qwen3-VL 多模态模型的整体架构、视频输入链路、预/后训练和面试回答。 |
| [模型细节/Qwen3-VL输入处理逻辑.md](<模型细节/Qwen3-VL输入处理逻辑.md>) | 详细拆解 Qwen3-VL 对文本、图像、视频的 processor、视觉 token、时间戳、MRoPE 和 DeepStack 处理。 |
