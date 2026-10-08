# 大模型发展历史与 SOTA 迭代时间线（2017-2026）

> 来源：https://bytedance.larkoffice.com/docx/FUjKdPEdioj73ox8KzMcTWn6ntg

> **阅读口径：**本文只维护版本时间线，回答“行业和同一模型系列如何逐代演进”。当前模型的横向能力、边界与业务选型见 [当前 SOTA 模型详解](<模型细节/当前SOTA模型详解.md#当前sota模型详解>)；架构、后训练、多模态与 Agent 原理分别进入对应专题。

## 主线一：按业内 SOTA 时间线看技术迭代
这条线回答：行业是如何一步步从 Transformer 走到 ChatGPT、GPT-4、Claude、Gemini、Qwen、DeepSeek-R1 和 Agent 模型的。

| 年份 | 代表模型 / 工作 | 主要机构 | 解决的问题 | 关键优化 | 留下的问题 | 参考入口 |
|-|-|-|-|-|-|-|
| 2017 | Transformer | Google | 自注意力替代 RNN/CNN，解决长依赖与并行训练问题 | 并行训练 + attention 成为统一底座 | 序列长度二次复杂度，为后来的 FlashAttention/长上下文优化埋下问题 | [NeurIPS](https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need) |
| 2018 | GPT-1 / BERT | OpenAI / Google | 从任务专用模型转向“预训练 + 微调” | GPT 用自回归预训练；BERT 用双向 masked LM | 生成与理解路线分化；下游任务仍需标注微调 | [BERT](https://arxiv.org/abs/1810.04805) |
| 2019 | GPT-2 / T5 / Megatron-LM | OpenAI / Google / NVIDIA | 验证扩大参数和数据能显著提升生成能力 | 更大语料、更大模型、统一 text-to-text、模型并行 | 对齐和事实性仍弱；训练工程成为壁垒 | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| 2020 | GPT-3 / [RAG](<应用与问题/RAG.md#rag>) / Switch Transformer | OpenAI / Meta / Google | 让模型不用每个任务都微调，探索少样本泛化 | 175B dense LM、in-context learning、检索增强、MoE 稀疏激活 | 成本极高；会续写但不一定会遵循指令 | [GPT-3](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) |
| 2021 | [CLIP](<模型细节/里程碑模型/CLIP.md#clip>) / Codex / FLAN | OpenAI / Google | 把语言模型扩展到图文、代码和指令泛化 | 图文对比学习、代码预训练、多任务 instruction tuning | 能力开始泛化，但对话体验、安全和复杂推理仍不足 | [FLAN](https://openreview.net/forum?id=gEZrGCozdqR) |
| 2022 | InstructGPT / PaLM / Chinchilla / ChatGPT | OpenAI / Google / DeepMind | 让 base model 变成可用助手，并修正规模化训练配比 | SFT + RM + [PPO](<../03_训练优化与对齐/后训练与对齐/PPO 近端策略优化.md#ppo-近端策略优化>)；540B PaLM；Chinchilla 数据-参数最优配比 | RLHF 贵且不稳；ChatGPT 证明产品形态，但幻觉仍存在 | [InstructGPT](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html) |
| 2023 | GPT-4 / Claude / LLaMA / Gemini 1.0 / Qwen / Mistral | OpenAI / Anthropic / Meta / Google / Alibaba / Mistral | 闭源旗舰与开放权重生态同时爆发 | 多模态、RLAIF、开放权重、MoE、函数调用、长上下文 | SOTA 不再单一；开源追赶但数据和后训练细节差距仍大 | [GPT-4 Report](https://arxiv.org/abs/2303.08774) |
| 2024 | GPT-4o / Claude 3.5 / Gemini 1.5 / Llama 3.1 / Qwen2.5 / DeepSeek-V3 / o1 | 多家机构 | 从聊天走向实时多模态、长上下文、代码和推理模型 | omni 多模态、1M 上下文、405B 开放模型、MoE/MLA、test-time compute | 推理成本、长上下文可靠性、工具调用安全成为新问题 | [GPT-4o](https://openai.com/index/hello-gpt-4o/) |
| 2025 | DeepSeek-R1 / Gemini 2.5 / o3-o4-mini / Claude 4 / Qwen3 / Llama 4 | 多家机构 | 推理能力、可验证奖励和 Agent 成为竞争中心 | RLVR/GRPO、thinking budget、MoE、长上下文、agentic coding | 需以官方发布为准；推理增强带来成本、长度偏置和安全挑战 | [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) |
| 2026 | 趋势：多[模型路由](<../05_推理部署与系统/生产系统设计/模型路由.md#模型路由>)、Agent 工程化、可靠评测、私有部署 | 产业界 | 从单模型能力转向端到端工作流可靠性 | 模型路由、推理预算控制、RAG/工具/环境反馈闭环、持续评测 | 2026 新旗舰细节变化快；报告只把公开可信方向作为框架 | [Qwen3](https://qwenlm.github.io/blog/qwen3/) |

---

## SOTA 迭代的阶段性总结
| 阶段 | 时间 | 核心问题 | 主流解法 | 你应该形成的判断 |
|-|-|-|-|-|
| 预训练范式确立 | 2017-2019 | 如何获得通用语言表示 | Transformer + GPT/BERT + 大语料 | 这是大模型能力的地基，但还不是好用助手 |
| 规模化涌现 | 2020-2021 | 模型能否少样本泛化 | 扩大参数、数据、算力；prompt / in-context learning | 规模带来能力，但不自动带来可靠性 |
| 助手化对齐 | 2022-2023 | 如何让模型听指令、少胡说、更安全 | SFT + RLHF/RLAIF + 高质量指令数据 | ChatGPT 的关键不只是 GPT-3.5，而是后训练和产品反馈 |
| 开放生态追赶 | 2023-2024 | 如何让强模型可复现、可部署、低成本 | LLaMA/Qwen/Mistral/DeepSeek，LoRA/QLoRA/vLLM | 开源路线的核心是 recipe、数据和系统工程 |
| 推理模型兴起 | 2024-2025 | 如何提升数学、代码、复杂规划能力 | test-time compute、[RLVR](<../03_训练优化与对齐/后训练与对齐/RLVR 可验证奖励强化学习.md#rlvr-可验证奖励强化学习>)、[GRPO](<../03_训练优化与对齐/后训练与对齐/GRPO 组相对策略优化.md#grpo-组相对策略优化>)、推理蒸馏 | 能力竞争从“答得像”转向“能不能真正解题” |
| Agent 与多模态工作流 | 2025-2026 | 如何把模型放进真实环境完成任务 | 工具调用、浏览器/IDE/文件系统、长上下文、多模型路由 | 下一阶段 SOTA 是模型 + 工具 + 评测 + 安全闭环 |

---

## 主线二：按同系列模型看迭代历史（更新至 2026-06-14）

本节按截至 2026-06-14 可核验的公开资料，整理 OpenAI、Google Gemini/PaLM、Meta LLaMA、Anthropic Claude、Alibaba Qwen、DeepSeek、Mistral 等主线系列。每行关注上一阶段的问题、本代的主要变化和可核验入口；模型排名和能力结论仍需结合具体时间、任务和评测集判断。

| 口径 | 说明 | 为什么这样处理 |
|-|-|-|
| 完整范围 | 覆盖本文已有主线系列，不把每个 API 小版本、Embedding、安全分类器、图像/视频生成模型都展开成主线 | 避免把产品 SKU 当作基础模型代际；重要专项会在对应系列中保留 |
| 可信来源 | 官方博客、官方模型卡、官方 API 更新日志、官方 GitHub/HF/ModelScope、正式论文页面优先 | 减少传闻和第三方榜单造成的误导 |
| 性质标注 | 区分开放权重、API/产品模型、技术报告、研究模型、访问受限模型 | 同样叫模型，但训练可复现性、部署方式和研究价值不同 |

### OpenAI GPT / o 系列：从语言建模到统一推理 Agent
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| GPT-1 | 2018 | 论文/研究模型 | NLP 依赖任务专用模型，迁移能力弱 | 生成式预训练 + 下游微调，验证 decoder-only 预训练可迁移 | [OpenAI PDF](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) |
| GPT-2 | 2019 | 论文/研究模型 | GPT-1 规模小，zero-shot 生成能力不明显 | 扩大模型和 WebText，展示无监督多任务生成 | [OpenAI PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) |
| GPT-3 | 2020 | 论文/API 基座 | 每个任务都微调成本高，少样本泛化未被系统证明 | 175B + in-context learning，让 prompt 成为任务接口 | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) |
| InstructGPT / ChatGPT | 2022 | 后训练/API 产品 | Base 模型像续写器，不会稳定当助手 | SFT + [Reward Model](<../03_训练优化与对齐/后训练与对齐/Reward Model 与 Grader 奖励模型与评分器.md#reward-model-与-grader-奖励模型与评分器>) + PPO/RLHF，把模型对齐到人类指令和偏好 | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html) |
| GPT-4 | 2023 | 技术报告/API | ChatGPT 复杂推理、代码、多模态、可靠性不足 | 更强预训练和后训练，加入图像输入和系统化安全评测 | [Technical Report](https://arxiv.org/abs/2303.08774) |
| GPT-4o | 2024-05 | API/产品模型 | GPT-4 多模态链路割裂，语音视觉实时交互慢 | 原生 omni 多模态，统一文本、图像、音频，降低交互延迟 | [OpenAI](https://openai.com/index/hello-gpt-4o/) |
| GPT-4o mini | 2024-07 | API/产品模型 | 强模型成本高，不适合高频批量任务 | 小型高性价比 omni 模型，降低延迟和调用成本 | [OpenAI](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) |
| o1-preview / o1-mini / o1 | 2024-09/12 | 推理模型 | 普通 chat 模型复杂数学/代码容易一步到位出错 | 显式推理时计算，先思考再回答，o1-mini 降低 STEM 推理成本 | [OpenAI](https://openai.com/index/introducing-openai-o1-preview/) |
| o3-mini | 2025-01 | 推理模型 | o1 推理成本和延迟偏高，生产高频 STEM 场景不经济 | 把推理模型做小做快，面向代码、数学、逻辑高频调用 | [OpenAI](https://openai.com/index/openai-o3-mini/) |
| GPT-4.5 | 2025-02 | 研究预览/API | 非推理模型的世界知识、写作和自然对话仍需提升 | 继续扩大预训练和对齐，提升低幻觉、自然对话和创意写作 | [OpenAI](https://openai.com/index/introducing-gpt-4-5/) |
| GPT-4.1 / mini / nano | 2025-04 | API 模型族 | 代码、指令遵循和长上下文 API 场景需要更稳 | 1M 上下文，强化代码、指令遵循，并提供 mini/nano 成本梯度 | [OpenAI](https://openai.com/index/gpt-4-1/) |
| o3 / o4-mini | 2025-04 | 推理+工具模型 | 推理模型需要进入真实多步骤工具任务 | 推理模型深度结合工具、视觉、Python、搜索和图像能力 | [OpenAI](https://openai.com/index/introducing-o3-and-o4-mini/) |
| o3-pro | 2025-06 | 高可靠推理模型 | o3 仍需要更高可靠性和长思考 | 为科学、教育、编程、商业等复杂任务提供更可靠长思考版本 | [Release notes](https://help.openai.com/en/articles/9624314-model-release-notes) |
| GPT-5 | 2025-08 | 统一模型/API/产品 | 用户需要在 GPT 和推理模型间手动选型 | 统一快速回答与深度思考路由，降低模型选择复杂度 | [Release notes](https://help.openai.com/en/articles/9624314-model-release-notes) |
| GPT-5-Codex | 2025-09 | 代码 Agent 模型 | 通用模型做长期代码任务、CLI/IDE 工作流仍不稳 | 面向 Codex 的 agentic coding，强化代码审查、长期任务和工具执行 | [Release notes](https://help.openai.com/en/articles/9624314-model-release-notes) |
| GPT-5.1 Instant / Thinking | 2025-11 | 统一模型升级 | 默认模型对话自然度、指令遵循和推理分配需更好 | Instant 更会判断何时思考，Thinking 更清晰高效 | [OpenAI](https://openai.com/index/gpt-5-1/) |
| GPT-5.3-Codex | 2026-02 | 代码/电脑 Agent | 代码模型需要从生成代码走向操作计算机完成工程任务 | 长期运行、工具使用、前端生成、部署调试和安全研究能力增强 | [OpenAI](https://openai.com/index/introducing-gpt-5-3-codex/) |
| GPT-5.4 Thinking / mini / nano | 2026-03 | 推理+小模型族 | 深度任务需要更好上下文管理，小任务需要更低成本 | Thinking 强化复杂工作流；mini/nano 服务高吞吐编码、分类、子 Agent | [OpenAI](https://openai.com/index/introducing-gpt-5-4-mini-and-nano/) |
| GPT-5.5 / GPT-5.5 Pro | 2026-04 | 前沿 Agent 模型 | 模型需要更长周期地理解意图、跨工具执行和自检 | 强化 agentic coding、电脑使用、知识工作、科研和网络安全防护 | [OpenAI](https://openai.com/index/introducing-gpt-5-5/) |

### Google PaLM / Gemini：从 Pathways 规模化到 Agentic Multimodal
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| PaLM | 2022-04 | 论文/研究模型 | 需要验证超大 dense LM 的少样本、多语言和推理能力 | Pathways 扩展到 540B，提升 few-shot、推理、代码和语言理解 | [Google Research](https://blog.research.google/2022/04/pathways-language-model-palm-scaling-to.html) |
| PaLM 2 | 2023-05 | 产品/模型族 | PaLM 多语言、代码、数学和移动端适配不足 | 改进数据和训练，提供 Gecko/Otter/Bison/Unicorn 多尺寸 | [Google](https://blog.google/technology/ai/google-palm-2-ai-large-language-model/) |
| Gemini 1.0 Ultra/Pro/Nano | 2023-12 | 原生多模态模型族 | 外挂式多模态难统一处理文本、图像、音频、视频 | 从底层构建原生多模态，覆盖云端复杂任务到端侧模型 | [Google](https://blog.google/technology/ai/google-gemini-ai/) |
| Gemini 1.5 Pro | 2024-02 | 长上下文/MoE | 长文档、长视频、长音频无法完整放进上下文 | MoE 提升效率，主打百万 token 长上下文 | [Google](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/) |
| Gemini 1.5 Flash | 2024-05 | 低成本长上下文 | 1.5 Pro 成本和延迟不适合高频应用 | 从 Pro 蒸馏出更快更便宜的 Flash，服务总结、抽取、字幕等 | [Google](https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/) |
| Gemini 2.0 Flash | 2024-12 | Agentic 模型 | 模型需要工具使用、原生输出和实时多模态进入 Agent 场景 | 面向 agentic era，增强工具、图像/音频输出和多步任务 | [Google DeepMind](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/) |
| Gemini 2.0 Flash Thinking | 2024-12 | 实验推理模型 | Flash 快但复杂数学、代码、视觉推理不足 | 在 Flash 上加入显式 thinking，平衡推理与速度 | [Gemini API docs](https://ai.google.dev/gemini-api/docs/models) |
| Gemini 2.5 Pro / Flash / Flash-Lite | 2025-03/06 | Thinking 模型族 | 复杂推理需要成为主线能力，而非单独实验 | 把 thinking 内建到主线模型；Pro/Flash/Lite 覆盖能力到成本梯度 | [Google DeepMind](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) |
| Gemini 3 Pro / Deep Think | 2025-11 | 前沿推理/Agent | 复杂学习、构建、规划和交互式生成仍需更强 | 提升推理、多模态、Agentic coding 和交互式生成 | [Google](https://blog.google/products/gemini/gemini-3/) |
| Gemini 3.1 Pro | 2026-02 | 增强版旗舰 | Gemini 3 Pro 在代码库级理解和科研工程推理仍需增强 | 面向百万 token 多模态、复杂任务、仓库级代码和科研工程推理 | [Model card](https://deepmind.google/models/model-cards/gemini-3-1-pro/) |
| Gemini 3.1 Flash Live / Audio | 2026-03 | 实时音频/语音模型 | 实时语音 Agent 需要低延迟、自然轮次和可靠任务执行 | Flash Live/Audio 强化实时音频对话、长轮次和任务执行 | [Google](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-flash-live/) |
| Gemini 3.5 Flash | 2026-05 | Agent/编码旗舰速度档 | 前沿智能和速度通常难兼得，长程 Agent 成本高 | 面向 complex agentic workflows，强化编码、子 Agent、多模态理解和低延迟 | [Google](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/) |

### Meta LLaMA：开放权重生态从文本到多模态 MoE
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| LLaMA 1 | 2023-02 | 开放权重 base | 研究者缺少高质量可访问 base model | 高质量数据和较优算力配比训练 7B-65B | [Meta](https://ai.meta.com/blog/large-language-model-llama-meta-ai/) |
| Llama 2 | 2023-07 | 开放商用 base/chat | LLaMA 1 许可和官方 chat 对齐不足 | 开放商用许可，发布 base + chat，加入 RLHF 和安全评测 | [Meta](https://ai.meta.com/llama/) |
| Code Llama | 2023-08/2024-01 | 代码专用开放模型 | 通用模型代码补全、FIM、长代码上下文不足 | 在代码语料继续训练，提供 Python/Instruct 和 70B 更新 | [Meta](https://ai.meta.com/blog/code-llama-large-language-model-coding/) |
| Llama 3 8B/70B | 2024-04 | 开放权重 base/instruct | Llama 2 数据规模、tokenizer、代码/推理能力不足 | 15T token、改进 tokenizer/GQA 和后训练，提升开放模型上限 | [Meta](https://ai.meta.com/blog/meta-llama-3/) |
| Llama 3.1 8B/70B/405B | 2024-07 | 开放旗舰/长上下文 | 开放模型缺 405B 级旗舰、长上下文和工具调用 | 405B、128K、多语言、工具调用，缩小与闭源差距 | [Llama docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/) |
| Llama 3.2 1B/3B/11B-V/90B-V | 2024-09 | 端侧+视觉 | 开放模型需要端侧小模型和官方视觉模型 | 1B/3B 端侧；11B/90B Vision 支持图像、图表、文档 VQA | [Llama docs](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/) |
| Llama 3.3 70B Instruct | 2024-12 | 后训练增强 | 70B 成本更低但质量需接近 405B | 通过后训练提升 70B instruct，降低部署成本 | [Model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/MODEL_CARD.md) |
| Llama 4 Scout / Maverick | 2025-04 | 原生多模态 MoE | 开放 Llama 需要原生多模态、MoE 和超长上下文 | Scout 面向单 H100/超长上下文；Maverick 面向更强图文、代码、推理 | [Llama 4](https://www.llama.com/models/llama-4/) |
| Llama 4 Behemoth | 2025-04 宣布 | 教师模型/未公开权重 | Scout/Maverick 需要更强教师蒸馏 | 官方定位为更大教师模型；截至 2026-06-14 未公开权重 | [Llama 4](https://www.llama.com/models/llama-4/) |

### Anthropic Claude：安全对齐、长上下文、电脑使用和 Mythos-class
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| Claude 1 / Constitutional AI | 2022-2023 | 对齐研究/产品 | RLHF 助手容易有害、讨好或不符合原则 | Constitutional AI，用原则和 AI feedback 做安全对齐 | [Anthropic](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) |
| Claude 2 / 2.1 | 2023 | 长上下文助手 | 长文档处理、可靠性和安全性不足 | 100K/200K 上下文，强化长文档问答和摘要 | [Anthropic](https://www.anthropic.com/news/claude-2-1) |
| Claude 3 Opus/Sonnet/Haiku | 2024-03 | 三档模型族 | 需要覆盖快/便宜/强，并补视觉能力 | Opus/Sonnet/Haiku 分层，加入视觉输入和更低误拒 | [Anthropic](https://www.anthropic.com/news/claude-3-family) |
| Claude 3.5 Sonnet / Haiku + computer use | 2024-06/10 | 代码/电脑使用 | 前代旗舰贵，模型无法直接操作电脑 | 3.5 Sonnet 中档反超前代旗舰；computer use 能看屏幕、点击、输入 | [Anthropic](https://www.anthropic.com/news/3-5-models-and-computer-use) |
| Claude 3.7 Sonnet | 2025-02 | 混合推理 | 模型需要快速回答和长思考可切换 | 首个混合推理 Claude，配合 Claude Code 强化编码 Agent | [Anthropic](https://www.anthropic.com/news/claude-3-7-sonnet) |
| Claude Opus 4 / Sonnet 4 | 2025-05 | Agent/代码旗舰 | 代码 Agent 和长时间工具任务需要更强可靠性 | 扩展思考调用工具、并行工具、记忆和 Claude Code GA | [Anthropic](https://www.anthropic.com/news/claude-4) |
| Claude Opus 4.1 | 2025-08 | 旗舰增强 | 真实代码、搜索 Agent、研究和数据分析细节跟踪不足 | 升级 Opus 4 的智能体任务、编码、推理、深度研究精度 | [Anthropic](https://www.anthropic.com/news/claude-opus-4-1) |
| Claude Sonnet 4.5 / Haiku 4.5 | 2025-09/10 | 主力+小模型 | 复杂 Agent 和实时低成本子 Agent 需求上升 | Sonnet 4.5 强化复杂 Agent；Haiku 4.5 下放编码/电脑使用能力 | [Anthropic](https://www.anthropic.com/news/claude-sonnet-4-5) |
| Claude Opus 4.5 | 2025-11 | 旗舰升级 | 旗舰在代码、agents、电脑使用和知识工作上还需更稳且更便宜 | 提升代码/agents/研究/表格/幻灯片，并降低 Opus 价格 | [Anthropic](https://www.anthropic.com/news/claude-opus-4-5) |
| Claude Opus 4.8 | 2026-05 | 旗舰增强 | 长时协作、诚实性、专业工作一致性仍需提升 | 加入 effort control、dynamic workflows，强化编码和专业任务 | [Anthropic](https://www.anthropic.com/news/claude-opus-4-8) |
| Claude Fable 5 / Mythos 5 | 2026-06 | Mythos-class/访问受限 | 更长周期工程、科研、视觉和网络防御任务仍是瓶颈 | Fable 5 面向通用超强模型；Mythos 5 面向可信网络防御；6/12 官方暂停访问 | [Anthropic](https://www.anthropic.com/news/claude-fable-5-mythos-5) |

### Alibaba Qwen：从中文开源基座到多模态 Agent 与 Qwen3.7
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| Qwen 初代 7B/14B/72B | 2023 | 开放 LLM | 中文和中英双语强开源模型不足 | 发布 base/chat，强化双语、代码、数学和工具调用基础 | [GitHub](https://github.com/QwenLM/Qwen) |
| Qwen-VL / Qwen-Audio | 2023 | 多模态分支 | 文本模型无法处理图像/音频 | 扩展到图文/OCR/文档/视觉定位和音频理解 | [Qwen-VL](https://github.com/QwenLM/Qwen-VL) |
| Qwen1.5 / MoE / CodeQwen / 110B | 2024-02\~04 | 模型族扩展 | 初代尺寸覆盖、部署体验和成本效率不足 | 补齐尺寸、32K、Transformers 生态，探索 MoE、代码专项和 110B 大模型 | [Qwen1.5](https://qwenlm.github.io/blog/qwen1.5/) |
| Qwen2 / Qwen2-Audio / Qwen2-VL | 2024-06\~08 | 多语言+多模态升级 | 多语言、代码、数学、长上下文和视频理解不足 | [GQA](<基础架构/GQA.md#mha-mqa-gqa>)、多语言增强、最高 128K；VL 支持动态分辨率和长视频 | [Qwen2](https://qwenlm.github.io/blog/qwen2/) |
| Qwen2.5 LLM/Coder/Math | 2024-09 | 专项能力升级 | 结构化输出、代码、数学、长文本和指令跟随需加强 | 18T token，强化 JSON、代码、数学、长文本、指令跟随 | [Qwen2.5](https://qwenlm.github.io/blog/qwen2.5/) |
| Qwen2.5-VL / 1M / Max / Omni | 2025-01\~03 | 视觉/长上下文/全模态 | 视觉 Agent、长上下文、全模态实时交互不足 | 文档解析、GUI/视频、1M 上下文、MoE Max、端到端全模态 | [Qwen2.5-VL](https://qwenlm.github.io/blog/qwen2.5-vl/) |
| QwQ / QVQ | 2025-03 | 推理模型 | 通用 instruct 在复杂数学/视觉推理不足 | 强化学习驱动文本推理和视觉推理模型 | [Qwen blog index](https://qwenlm.github.io/page/3/) |
| Qwen3 | 2025-04 | 开源混合思考模型 | 模型需要快速回答和深度推理可切换 | thinking/non-thinking 混合模式，dense+MoE，119 语言 | [Qwen3](https://qwenlm.github.io/zh/blog/qwen3/) |
| Qwen3-Embedding/Reranker | 2025-06 | 检索/RAG | 生成模型之外，RAG 需要更强[召回](<../09_搜索推荐广告/召回粗排精排重排.md#召回粗排精排与重排>)和排序 | 文本向量和重排模型，服务检索、聚类、分类和 RAG | [Qwen](https://qwen.ai/blog?id=qwen3-vl-embedding) |
| Qwen3-Coder | 2025-07 | Agentic coding | 代码模型需要仓库级理解和多步工具调用 | 面向 coding agent，支持长上下文、工具调用和软件工程任务 | [Qwen](https://qwen.ai/blog?id=qwen3-coder) |
| Qwen3-VL / Qwen3Guard | 2025-09 | 视觉语言/安全 | 视觉 Agent、长视频、GUI、空间理解和安全审核不足 | Qwen3-VL 强化视觉感知/推理/GUI/长视频；Guard 做安全分类 | [Qwen3-VL](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef) |
| Qwen3-VL-Embedding/Reranker | 2026-01 | 多模态检索 | 图文视频统一检索和跨模态重排不足 | 统一多模态向量和 reranker，服务视频/图文 RAG | [Qwen](https://qwen.ai/blog?id=qwen3-vl-embedding) |
| Qwen3.5 / Qwen3.5-Omni | 2026-02/03 | 原生 VL Agent/全模态 | 多模态 Agent、长音视频、实时语音和工具调用需统一 | 线性注意力+稀疏 MoE；Omni 支持长音视频、实时打断、WebSearch、Function Call | [Qwen3.5](https://qwen.ai/blog?id=qwen3.5) |
| Qwen3.6-Plus | 2026-04 | 闭源/API Agent | 真实世界 Agent、代码、工具、长上下文记忆稳定性不足 | 强化 coding agent、general agent、tool usage、1M context 和多模态推理 | [Qwen3.6](https://qwen.ai/blog?id=qwen3.6) |
| Qwen3.5-LiveTranslate-Flash | 2026-05 | 实时翻译 | 低延迟语音翻译需要结合视觉上下文和术语控制 | 实时多模态同传、语音克隆、热词术语和视觉辅助翻译 | [Qwen](https://qwen.ai/blog?id=qwen3.5-livetranslate) |
| Qwen3.7-Max / Qwen-VLA | 2026-05 | Agent/具身智能 | 长程自治、多工具、办公自动化和行动闭环不足 | Max 面向 Agent 时代；VLA 从视觉语言理解走向动作决策 | [Qwen Research](https://qwen.ai/research/) |
| Qwen3.7-Plus | 2026-06 | 多模态 Agent API | Agent 需要统一 GUI/CLI、视觉编码、移动端导航和跨框架泛化 | 多模态 interactive hybrid agent，统一视觉语言、代码、工具和 GUI 操作 | [Qwen3.7-Plus](https://qwen.ai/blog?id=qwen3.7-plus) |

### DeepSeek：低成本 MoE、GRPO/RLVR 与 1M Agent
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| DeepSeek-Coder | 2023-11 | 代码开源模型 | 开源代码模型仓库级理解和 FIM 不足 | 代码生成、补全、FIM、多语言和仓库级代码理解 | [GitHub](https://github.com/deepseek-ai/DeepSeek-Coder) |
| DeepSeek-LLM / MoE | 2024-01 | 通用 LLM/MoE | 通用开源底座和 MoE 参数效率需验证 | 发布通用 LLM；MoE 探索细粒度专家和共享专家 | [GitHub](https://github.com/deepseek-ai/DeepSeek-LLM) |
| DeepSeekMath | 2024-02 | 数学推理/RL | SFT 数学模型继续提升困难，PPO 工程复杂 | 引入 GRPO 和可验证数学奖励，强化推理能力 | [GitHub](https://github.com/deepseek-ai/DeepSeek-Math) |
| DeepSeek-VL | 2024-03 | 视觉语言 | 通用 LLM 无法处理真实图文/OCR/网页截图 | 视觉语言模型支持真实场景图文理解、OCR、文档和网页截图 | [GitHub](https://github.com/deepseek-ai/DeepSeek-VL) |
| DeepSeek-V2 / Coder-V2 | 2024-05/06 | 低成本 MoE/代码 | 大模型 [KV cache](<../05_推理部署与系统/推理工程/KV_Cache与Prefill_Decode.md#kvcache与prefilldecode>) 和推理成本高，代码长上下文不足 | MLA + DeepSeekMoE，128K，提升吞吐和代码工程能力 | [GitHub](https://github.com/deepseek-ai/DeepSeek-V2) |
| DeepSeek-Prover / V1.5 | 2024-05/08 | 形式化证明 | 数学形式化证明长程搜索和奖励稀疏 | Lean 4 证明模型，RLPAF/RMaxTS 提升证明成功率 | [GitHub](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5) |
| DeepSeek-V2.5 / 1210 | 2024-09/12 | Chat+Coder 合并 | 通用对话和代码能力分线，体验割裂 | 合并 Chat 与 Coder，并增强数学、代码、写作、搜索 | [DeepSeek](https://api-docs.deepseek.com/news/news0905) |
| DeepSeek-V3 | 2024-12 | 开源 MoE 基座 | 需要低成本训练超大 MoE 且保持前沿性能 | 671B/37B MoE，FP8、负载均衡、多 token prediction | [GitHub](https://github.com/deepseek-ai/DeepSeek-V3) |
| DeepSeek-R1 / R1-Zero / Distill | 2025-01 | RLVR 推理模型 | 强推理不应只靠人工 [CoT](<应用与问题/CoT.md#cot>) SFT | 基于 V3 做大规模 RLVR/GRPO，发布推理模型和蒸馏模型 | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |
| DeepSeek-V3-0324 / R1-0528 | 2025-03/05 | 能力增强 | 前端代码、中文写作、函数调用和幻觉仍需优化 | 更新版本提升推理、Web 前端、搜索报告、JSON/Function Calling | [Updates](https://api-docs.deepseek.com/zh-cn/updates/) |
| DeepSeek-V3.1 / Terminus | 2025-08/09 | 混合推理/Agent | 需要一个模型支持思考/非思考，并提升工具 Agent | 混合推理架构，优化 Code/Search Agent 和反馈问题 | [DeepSeek](https://api-docs.deepseek.com/news/news250821/) |
| DeepSeek-V3.2 / Speciale | 2025-12 | Agent+思考 | 长上下文效率和 Agent 能力继续成为瓶颈 | 统一 chat/reasoner，融入思考推理；Speciale 提供高输出长度深度推理 | [DeepSeek](https://api-docs.deepseek.com/zh-cn/news/news251201) |
| DeepSeek-V4-Pro / V4-Flash | 2026-04 | 开源+API/1M Agent | 长上下文、Agent、世界知识和推理需要同时提升且普惠 | 1M 上下文标配，DSA 稀疏注意力，Pro 高性能，Flash 低成本 | [DeepSeek](https://api-docs.deepseek.com/zh-cn/news/news260424) |

### Mistral：开放小模型、MoE、专项模型与企业自托管
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| Mistral 7B | 2023-09 | 开放小模型 | 开源小模型效率和质量仍有空间 | 7B 高质量 recipe，降低本地/私有部署门槛 | [Mistral](https://mistral.ai/news/announcing-mistral-7b/) |
| Mixtral 8x7B / 8x22B | 2023-12/2024-04 | 开放 MoE | dense 扩大成本高，开源复杂任务能力不足 | 稀疏 MoE 提升参数效率和复杂任务性能 | [Mistral](https://mistral.ai/news/mixtral-of-experts/) |
| Mistral Large / Large 2 | 2024-02/07 | 商业旗舰 | Mistral 需要闭源 API 旗舰覆盖企业复杂任务 | 提升复杂推理、代码、数学、多语言和企业 API 能力 | [Mistral](https://mistral.ai/news/mistral-large-2407/) |
| Codestral / Mathstral / Codestral Mamba | 2024-05/07 | 专项模型 | 通用模型代码、数学、长代码延迟不足 | 代码、数学和 Mamba 长序列代码专项模型 | [Mistral](https://mistral.ai/news/codestral/) |
| Mistral NeMo / Ministral | 2024-07/10 | 高性价比/端侧 | 企业需要 12B 级和端侧小模型 | NeMo 12B 多语言；Ministral 3B/8B 低延迟端侧 | [Mistral](https://mistral.ai/news/mistral-nemo/) |
| Pixtral 12B / Pixtral Large | 2024-09/11 | 视觉语言 | Mistral 缺开放多模态能力 | 图像理解、文档/图表/截图和多图推理 | [Mistral](https://mistral.ai/news/pixtral-12b/) |
| Mistral Small 3.1 / OCR | 2025-03 | 开放多模态/OCR | 企业文档 RAG 需要结构化抽取和单机可跑 [VLM](<../02_大模型/视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md#vlm-与-vision-instruction-tuning>) | 24B 开放多模态 128K；OCR 处理 PDF/图表/公式 | [Mistral](https://mistral.ai/news/mistral-small-3-1/) |
| Mistral Medium 3 / Devstral | 2025-05 | 企业中型/代码 Agent | 企业需要低成本强模型和真实代码库 Agent | Medium 3 覆盖代码/STEM/视觉；Devstral 面向 SWE-bench 类任务 | [Mistral](https://mistral.ai/news/devstral) |
| Magistral | 2025-06 | 推理模型 | Mistral 缺长思考推理模型 | 首批推理模型，补齐数学、逻辑和多步问题求解 | [Mistral](https://mistral.ai/news/magistral) |
| Voxtral | 2025-07 | 语音理解 | 开放语音理解和音频函数调用不足 | 转写、音频问答、总结、多语言语音与函数调用 | [Mistral](https://mistral.ai/news/voxtral) |
| Mistral 3 / Large 3 / Ministral 3 | 2025-12 | 统一开放权重家族 | 模型线分散，企业需要统一自托管矩阵 | 统一旗舰开放权重与小模型家族，覆盖视觉、长上下文、边缘部署 | [Mistral](https://mistral.ai/news/mistral-3/) |

### 其他主流 SOTA 补充：本文未展开但需要知道的路线
| 版本/系列 | 时间 | 性质 | 前一阶段解决不了什么 | 本代怎么解决/主要优化 | 官方入口 |
|-|-|-|-|-|-|
| xAI Grok 4 / Grok 4 Heavy | 2025-07 | 推理+实时搜索 | 需要强推理结合实时信息和工具 | 强化推理、原生工具、实时搜索和 X 数据 | [xAI](https://x.ai/news/grok-4) |
| Z.AI GLM-4.5 / Air | 2025-08 | 开源 MoE Agent | Agent、Reasoning、Coding 需要统一开源模型 | 面向工具调用、代码和复杂推理的开源 MoE | [Z.AI docs](https://docs.z.ai/guides/llm/glm-4.5) |
| Google Gemma 3 | 2025-03 | 开放小模型/多模态 | 端侧和单 GPU 需要强开放小模型 | 单 GPU/TPU 可跑，支持多语言、视觉和 128K | [Google](https://blog.google/technology/developers/gemma-3/) |
| Microsoft Phi-4 | 2024-12 | 小模型推理 | 受限硬件上需要高质量 STEM/数学推理 | 14B 小模型专注复杂推理和合成数据训练 | [Microsoft](https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introducing-phi-4-microsoft’s-newest-small-language-model-specializing-in-comple/4357090) |

## 专题导航

时间线只记录模型代际与能力演进。横向对比和选型见 [当前 SOTA 模型详解](<模型细节/当前SOTA模型详解.md#当前sota模型详解>)，方法原理继续由以下专题维护：

- 后训练演进：[后训练发展史与方法对比](<../03_训练优化与对齐/后训练与对齐/后训练发展史与方法对比.md#后训练发展史与方法对比>)
- 架构与模型形态：[模型架构对比与选型](<模型细节/模型架构对比与选型.md#模型架构对比与选型>)
- 多模态架构：[VLM 与 Vision Instruction Tuning](<视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md#vlm-与-vision-instruction-tuning>)
- Agent 系统：[Agent](<../08_Agent/基础概念/Agent.md#agent>)

## 面试应对

### 如何概括大模型从 Transformer 到 Agent 的发展主线？

回答思路：不要按年份堆模型名称，而要围绕架构与规模化、指令与偏好对齐、多模态与推理、工具使用与系统化四次能力跃迁展开，并指出每次跃迁解决的核心瓶颈。

回答模板：

大模型的发展不是简单增加参数，而是能力来源不断扩展。Transformer 先统一了可并行扩展的序列建模底座，随后 Scaling Law、数据工程和训练系统推动预训练规模化；Instruction Tuning 与 RLHF 让 Base Model 变成可交互的助手；多模态、长上下文和 MoE 扩展了输入范围与计算效率；RLVR、推理时计算和工具调用又把模型从生成答案推进到规划、执行和反馈闭环。因此今天的 SOTA 更接近由模型、数据、后训练、推理和工具共同构成的系统，而不只是参数最多的单体模型。

### 如何看待 SOTA 榜单与模型迭代结论的边界？

回答思路：指出榜单受数据、提示词、推理预算和污染影响，版本能力也会变化；结论必须限定时间、任务和成本口径。

回答模板：

SOTA 是特定时间和评测协议下的相对结果，不等于所有场景中的最优模型。榜单分数会受到数据污染、提示词模板、采样次数、工具权限和推理 Token 预算影响，闭源模型还可能静默更新。因此我会记录模型版本、评测日期、数据集、推理配置和成本，在同一口径下比较，并补充真实业务失败案例。最终结论只能表述为某模型在给定约束下更合适，不能外推成普遍能力排序。
