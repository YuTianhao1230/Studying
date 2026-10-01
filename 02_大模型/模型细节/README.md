# 模型细节

本目录整理典型大模型、多模态模型和当前主流 SOTA 模型的架构细节、训练目标、能力边界和面试回答方式。

上一级的 [大模型发展历史与 SOTA 迭代框架](<../大模型发展历史与SOTA迭代框架.md>) 负责解释模型如何演进，本目录侧重单个模型为什么出现、如何设计以及怎样在具体约束下选型。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [阶段性里程碑模型详解.md](<阶段性里程碑模型详解.md>) | Transformer、BERT、GPT、T5、ViT、CLIP、ALBEF、TCL、BLIP、Flamingo、LLaVA 等里程碑模型的横向总览。 |
| [里程碑模型](<里程碑模型/README.md>) | BERT、GPT、T5、Llama、DeepSeek、ViT、CLIP、ALBEF、TCL、BLIP、Flamingo、LLaVA 单模型卡片索引。 |
| [当前SOTA模型详解.md](<当前SOTA模型详解.md>) | GPT/o 系列、Claude、Gemini、Qwen、DeepSeek、Llama、Mistral 等主流模型系列的架构与能力。 |
| [模型架构对比与选型.md](<模型架构对比与选型.md>) | Decoder-only、Encoder-Decoder、Dense、MoE、长上下文、多模态连接器、推理模型和 Agent 模型的对比。 |
| [Qwen千问架构.md](<Qwen千问架构.md>) | Qwen3 文本模型、Qwen3-VL 三模块架构、视频输入链路、Interleaved-MRoPE、DeepStack 和训练流程。 |
| [Qwen3-VL输入处理逻辑.md](<Qwen3-VL输入处理逻辑.md>) | 按文本、图像、视频拆解 Qwen3-VL 的 processor、视觉 token、时间戳、MRoPE 和 LLM 融合链路。 |
