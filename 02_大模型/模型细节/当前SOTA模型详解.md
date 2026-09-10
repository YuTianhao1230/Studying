# 当前SOTA模型详解

## 知识点解析

### 概述

本文按技术路线和代表性阶段整理 [GPT](<里程碑模型/GPT.md>)/o 系列、Claude、Gemini、[Qwen](<Qwen千问架构.md>)、[DeepSeek](<里程碑模型/DeepSeek.md>)、[Llama](<里程碑模型/Llama.md>)、Mistral 等模型系列。这里的“SOTA”表示某个时间点、任务或能力维度上的代表性路线，不等于永久的排行榜第一；实际选型必须以具体模型版本、评测集、成本和部署约束为准。

### 先建立 SOTA 模型坐标系

当前 SOTA 模型不能只按“参数更大”理解，更应该按能力和系统形态分层：

| 维度 | 典型路线 | 解决的问题 |
| --- | --- | --- |
| 基座架构 | [Decoder-only](<../基础架构/Decoder-only vs Encoder-Decoder.md>)、[MoE](<../基础架构/MoE.md>)、GQA/MLA、长上下文 | 语言建模、成本、上下文和吞吐 |
| 后训练 | [SFT](<../../03_训练优化与对齐/后训练与对齐/SFT 监督微调.md>)、RLHF/RLAIF、[DPO](<../../03_训练优化与对齐/后训练与对齐/DPO 直接偏好优化.md>)、RLVR/GRPO | 指令跟随、偏好、安全、数学代码推理 |
| 多模态 | 视觉编码器 + projector/Q-Former/cross-attention/原生多模态 | 图像、视频、音频、OCR、GUI |
| 推理时计算 | thinking mode、test-time compute、多采样、自检 | 复杂数学、代码、规划和科学推理 |
| [Agent](<../../10_Agent/基础概念/Agent.md>) 能力 | tool use、function calling、computer use、IDE/CLI agent | 从回答问题到执行任务 |
| 部署生态 | 开源权重、API、端侧小模型、私有化部署 | 成本、可控性、数据安全和工程落地 |

### OpenAI GPT / o 系列

#### 解决的问题

OpenAI 系列的主线是从通用语言生成走向“统一助手 + 推理模型 + 多模态 Agent”。GPT-3 解决少样本泛化，InstructGPT/ChatGPT 解决指令对齐，GPT-4 解决复杂推理和可靠性，GPT-4o 解决实时多模态，o 系列解决需要长思考的数学、代码和规划问题。

#### 架构与训练特点

公开细节有限，但从产品形态和技术报告可以归纳为：

```text
large decoder-only / multimodal foundation model
  -> large-scale pretraining
  -> instruction tuning
  -> preference alignment / safety alignment
  -> reasoning-oriented post-training for o-series
  -> tool use / multimodal API / agent runtime
```

关键特点：

- [GPT](<里程碑模型/GPT.md>) 主线偏通用助手与多模态统一。
- o 系列偏推理时计算，适合复杂数学、代码、科学和规划。
- GPT-4o 强调文本、图像、音频的低延迟统一交互。
- API 生态强调 function calling、structured output、tools 和多模态输入。

#### 能力边界

- 闭源模型无法完全确认训练数据、结构参数和后训练细节。
- 推理模型通常更强但更慢、更贵，需要控制 reasoning budget。
- 多模态强但仍会有视觉幻觉、细粒度定位和长视频理解问题。

### Anthropic Claude

#### 解决的问题

Claude 系列强调安全对齐、长上下文、写作与代码能力。Claude 3.5/4 之后，重点转向 coding agent、computer use、长周期任务和更可靠的工具调用。

#### 架构与训练特点

```text
large language model
  -> constitutional / preference-based alignment
  -> long-context training
  -> tool/computer-use post-training
  -> coding and agent workflow optimization
```

核心特点：

- Constitutional AI：用原则和 AI feedback 强化 helpful、harmless、honest。
- 长上下文：适合长文档、代码库和研究材料阅读。
- Computer use：模型能观察屏幕并执行点击、输入等操作。
- Claude Code：面向真实软件工程任务的 Agent 产品形态。

#### 能力边界

Claude 在长文档、写作和代码工作流中表现强，但闭源细节有限。电脑使用和 Agent 执行仍需要权限隔离、审计、回滚和人类确认。

### Google Gemini

#### 解决的问题

Gemini 系列强调原生多模态、长上下文和 Agentic 工作流。它从设计上覆盖文本、图像、音频、视频，并把长上下文、多模态理解和工具使用作为核心能力。

#### 架构与训练特点

```text
native multimodal model family
  -> text / image / audio / video inputs
  -> long-context attention / memory mechanisms
  -> tool use and agentic capabilities
  -> Pro / Flash / Lite capability-cost tiers
```

核心特点：

- Pro：偏旗舰能力和复杂推理。
- Flash：偏低延迟、高吞吐和性价比。
- 长上下文：适合长文档、长视频、长音频和代码库。
- Thinking 模式：把推理时计算纳入主线能力。

#### 能力边界

长上下文不等于长文档一定可靠。实际使用仍要关注检索定位、信息遗忘、引用准确性、上下文污染和推理成本。

### Alibaba Qwen

#### 解决的问题

Qwen 系列解决中文/多语言强开源模型、多模态、代码、数学、长上下文和 Agent 能力的综合需求。它的特点是模型谱系完整，覆盖 dense、MoE、VL、Audio、Omni、Embedding、Reranker、Coder、Math 和 Guard。

#### 架构与训练特点

```text
Qwen LLM base/instruct
  -> decoder-only transformer
  -> GQA / long context / multilingual data
  -> SFT and preference alignment

Qwen-VL / Qwen2.5-VL / Qwen3-VL
  -> vision encoder
  -> visual token compression / dynamic resolution
  -> LLM backbone
  -> vision instruction tuning

Qwen3 / QwQ / Qwen Coder
  -> thinking / non-thinking modes
  -> RL-oriented reasoning or coding post-training
  -> tool use and agentic coding
```

核心特点：

- 中文、多语言、代码和数学能力均衡。
- 开源生态强，适合私有化部署、微调和实验复现。
- VL 系列覆盖 OCR、文档、视频、GUI 和视觉 Agent。
- Qwen3 之后强调混合思考模式：简单任务快速答，复杂任务深度推理。

#### 能力边界

开源权重不等于完整可复现，训练数据 recipe、清洗策略、后训练数据和 RL 细节仍是关键壁垒。部署时还要关注上下文长度、显存、吞吐和[量化](<../../05_推理部署与系统/推理工程/量化.md>)精度损失。

### DeepSeek

#### 解决的问题

DeepSeek 的主线是以更低成本训练和推理获得接近闭源 SOTA 的能力，尤其在 MoE、[MLA](<../基础架构/MLA.md>)、代码、数学推理和 RLVR/GRPO 方向影响很大。

#### 架构与训练特点

```text
DeepSeek base model
  -> decoder-only / MoE architecture
  -> efficient attention such as MLA
  -> large-scale pretraining
  -> SFT
  -> reasoning-oriented RL such as GRPO/RLVR
  -> distillation to smaller dense models
```

关键概念：

- MoE：每个 token 只激活部分专家，在较低推理成本下扩大总参数容量。
- MLA：通过压缩 KV 表示降低长上下文 [KV Cache](<../../05_推理部署与系统/推理工程/KV_Cache与Prefill_Decode.md>) 成本。
- GRPO/RLVR：用可验证奖励强化数学、代码等任务中的推理能力。
- Distillation：把强推理模型能力蒸馏到更小模型中，降低部署成本。

#### 能力边界

DeepSeek 代表“高性价比推理模型”路线，但 MoE 的训练稳定性、专家负载均衡、通信开销、服务部署和长链路推理成本仍是工程重点。推理模型也可能出现长度偏置、过度思考和 reward hacking。

### Meta Llama

#### 解决的问题

Llama 系列的核心价值是开放权重生态。它降低了研究、微调、私有部署和推理优化门槛，使 [LoRA](<../../03_训练优化与对齐/后训练与对齐/LoRA 低秩适配.md>)、[QLoRA](<../../03_训练优化与对齐/后训练与对齐/PEFT 参数高效微调.md>)、[vLLM](<../../05_推理部署与系统/推理工程/vLLM.md>)、量化、[RAG](<../应用与问题/RAG.md>) 和企业私有化应用快速发展。

#### 架构与训练特点

```text
decoder-only transformer
  -> large-scale curated pretraining
  -> instruct tuning and safety tuning
  -> long context and tool-use variants
  -> vision / multimodal variants in later versions
```

核心特点：

- 多尺寸覆盖，便于不同资源下部署。
- 社区生态强，适合微调和应用实验。
- Llama 3 之后在 tokenizer、数据规模、[GQA](<../基础架构/GQA.md>)、长上下文和 instruct 能力上明显增强。
- Llama 4 路线引入多模态和 MoE，代表开放模型继续追赶闭源前沿。

#### 能力边界

开放模型的优势是可控和可部署，但效果高度依赖微调数据、推理框架、量化方式和安全策略。企业落地时要补足内容安全、权限、监控和评测闭环。

### Mistral

#### 解决的问题

Mistral 系列强调小而强、开源友好和高效推理。Mixtral 代表稀疏 MoE 在开放模型中的重要应用。

#### 架构与训练特点

```text
dense small language models
  -> efficient decoder-only architecture
  -> sliding window / grouped-query attention variants

Mixtral
  -> sparse mixture-of-experts
  -> router selects experts per token
  -> larger capacity with limited active parameters
```

核心特点：

- 7B/8x7B/8x22B 等模型强调性价比。
- MoE 扩大总容量但保持较低激活参数。
- 适合部署、微调、欧洲生态和私有化场景。

#### 能力边界

MoE 推理部署比 dense 模型复杂，专家并行、路由负载、KV Cache、批处理和显存布局都需要工程优化。

### 当前 SOTA 的共同趋势

| 趋势 | 代表模型 | 解决的问题 | 面试表达 |
| --- | --- | --- | --- |
| 推理模型 | o 系列、DeepSeek-R1、Qwen3、Gemini Thinking | 数学、代码、规划能力不足 | 从生成式回答走向推理时计算 |
| MoE | DeepSeek-V3/R1、Mixtral、Qwen MoE、Llama 4 | dense 成本过高 | 用稀疏激活扩大容量、控制成本 |
| 长上下文 | Gemini、GPT-4.1、Qwen、Claude | 长文档、视频、代码库处理 | 长上下文要配合检索和引用验证 |
| 原生多模态 | GPT-4o、Gemini、Qwen-VL/Omni、Llama Vision | 图像、音频、视频和 GUI | 从外挂视觉模块转向统一交互 |
| Agent 化 | Claude Code、Codex、Qwen Coder、Gemini Agent | 模型只回答不执行 | 工具、环境、权限、审计成为能力一部分 |
| 小模型/蒸馏 | GPT mini/nano、Qwen 小模型、DeepSeek Distill | 成本和延迟 | 强模型教小模型，服务高频场景 |

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 系列对比 | Qwen、DeepSeek、GPT 区别 | 从开源/API、MoE、推理、多模态、生态回答 |
| 架构题 | 为什么 MoE 能降成本 | 解释总参数大但每 token 只激活少数专家 |
| 推理题 | o 系列/DeepSeek-R1 为什么强 | 讲推理时计算和可验证奖励，而不是只说参数大 |
| 选型题 | 业务中选哪个模型 | 按效果、成本、延迟、私有化、多模态、工具调用和安全 |
| 局限题 | SOTA 模型还有什么问题 | 幻觉、长上下文可靠性、评测污染、安全、成本、Agent 失败恢复 |

### 易错点

- 把 SOTA 简化成排行榜第一，不看任务和成本。
- 把开源权重等同于训练过程完全公开。
- 讲 DeepSeek 只说便宜，不讲 MoE、MLA、RLVR/GRPO 和蒸馏。
- 讲 Qwen 只说中文好，不讲 VL、Coder、Embedding、Reranker、Agent 生态。
- 讲 GPT/o 系列只说强，不区分通用助手模型和推理模型。
- 讲长上下文只说 token 多，不讲检索定位、引用准确性和成本。

## 面试应对

### 现在主流 SOTA 模型的发展方向是什么？

回答思路：不要只报模型名，按能力轴总结。

回答模板：

现在 SOTA 模型的发展方向已经不只是扩大参数，而是几条线并行。第一是推理模型，用推理时计算和可验证奖励提升数学、代码和复杂规划，比如 o 系列、DeepSeek-R1、Qwen3。第二是架构效率，用 MoE、GQA、MLA、FlashAttention 降低训练和推理成本。第三是原生多模态，把文本、图像、音频、视频和 GUI 接到统一模型里。第四是 Agent 化，让模型能调用工具、操作代码库和浏览器。最后是模型分层和蒸馏，用小模型服务高频低成本场景。

### Qwen、DeepSeek、GPT 怎么对比？

回答思路：按定位、架构和落地方式回答。

回答模板：

GPT 系列更像闭源通用旗舰，优势在通用能力、多模态、工具生态和产品化体验，o 系列则突出推理时计算。Qwen 的特点是开源生态和模型谱系完整，覆盖 LLM、VL、Coder、Math、Embedding、Reranker、Omni 和 Agent 场景，中文和私有化部署优势明显。DeepSeek 更突出低成本高性能路线，核心看点是 MoE、MLA、GRPO/RLVR 和推理模型蒸馏。实际选型不能只看榜单，要看任务、成本、延迟、是否私有化、是否需要多模态和工具调用。

### 为什么 DeepSeek-R1 这类推理模型重要？

回答思路：把它放到“从模仿回答到解题探索”的变化里。

回答模板：

DeepSeek-R1 这类推理模型重要，是因为它把能力提升从单纯预训练和 SFT 推到推理时计算和可验证奖励上。普通指令模型更多是在模仿高质量回答，而推理模型会在数学、代码这类可验证任务中通过 RL 强化探索过程，学习更长链条的解题策略。它带来的变化是模型在复杂题上更会分解、验证和修正，但代价是推理延迟、输出长度和成本上升，因此需要根据任务设置 reasoning budget。

### 业务里怎么选 SOTA 模型？

回答思路：从任务和约束出发，而不是从模型名出发。

回答模板：

我会先明确任务类型：是文本生成、代码、数学推理、RAG、图像理解、视频理解还是 Agent 工具执行。然后看约束：效果要求、延迟、成本、并发、上下文长度、是否需要私有化、数据安全和可微调性。如果是高风险复杂推理，可以选 GPT/o、Claude、Gemini 或 DeepSeek-R1 这类强推理模型；如果要私有部署和中文生态，Qwen、DeepSeek、Llama 更合适；如果是高频简单任务，则应优先考虑小模型、蒸馏模型或路由方案。
