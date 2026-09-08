# 模型细节

本目录整理典型大模型、多模态模型和当前主流 SOTA 模型的架构细节、解决的问题、能力边界和面试回答方式。它和上一级的“发展历史”互补：发展历史回答“模型如何演进”，本目录回答“每个模型为什么出现、架构怎么设计、面试怎么讲”。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [阶段性里程碑模型详解.md](<阶段性里程碑模型详解.md>) | Transformer、BERT、GPT、T5、ViT、CLIP、ALBEF、TCL、BLIP、Flamingo、LLaVA 等里程碑模型的横向总览。 |
| [当前SOTA模型详解.md](<当前SOTA模型详解.md>) | GPT/o 系列、Claude、Gemini、Qwen、DeepSeek、Llama、Mistral 等主流模型系列的架构与能力。 |
| [模型架构对比与选型.md](<模型架构对比与选型.md>) | Decoder-only、Encoder-Decoder、Dense、MoE、长上下文、多模态连接器、推理模型和 Agent 模型的对比。 |
| [Qwen千问架构.md](<Qwen千问架构.md>) | Qwen3 文本模型、Qwen3-VL 三模块架构、视频输入链路、Interleaved-MRoPE、DeepStack 和训练流程。 |

## 阶段性里程碑模型卡片

| 模型 | 卡片 |
| --- | --- |
| Transformer | [里程碑模型/Transformer.md](<里程碑模型/Transformer.md>) |
| BERT | [里程碑模型/BERT.md](<里程碑模型/BERT.md>) |
| GPT | [里程碑模型/GPT.md](<里程碑模型/GPT.md>) |
| Qwen3 / Qwen3-VL | [Qwen千问架构.md](<Qwen千问架构.md>) |
| T5 | [里程碑模型/T5.md](<里程碑模型/T5.md>) |
| ViT | [里程碑模型/ViT.md](<里程碑模型/ViT.md>) |
| CLIP | [里程碑模型/CLIP.md](<里程碑模型/CLIP.md>) |
| ALBEF | [里程碑模型/ALBEF.md](<里程碑模型/ALBEF.md>) |
| TCL | [里程碑模型/TCL.md](<里程碑模型/TCL.md>) |
| BLIP / BLIP-2 | [里程碑模型/BLIP与BLIP2.md](<里程碑模型/BLIP与BLIP2.md>) |
| Flamingo | [里程碑模型/Flamingo.md](<里程碑模型/Flamingo.md>) |
| LLaVA | [里程碑模型/LLaVA.md](<里程碑模型/LLaVA.md>) |

## 学习路线

1. 先看 [阶段性里程碑模型详解.md](<阶段性里程碑模型详解.md>)，理解每个关键模型解决了哪个历史瓶颈。
2. 再按需进入单模型卡片，重点背每个模型的组件、层数、训练目标、改动点和适用边界。
3. 接着看 [当前SOTA模型详解.md](<当前SOTA模型详解.md>)，把 GPT、Qwen、DeepSeek、Gemini 等模型放到架构和能力坐标系里。
4. 最后看 [模型架构对比与选型.md](<模型架构对比与选型.md>)，练习按任务、成本、上下文、多模态和部署约束做模型选择。
