# 大厂算法工程师 JD 能力矩阵

## 调研结论

近两年大厂算法工程师 JD 的重心已经从“会训练一个模型”升级为“能把模型、数据、评测、系统和业务闭环串起来”。尤其是大模型和 Agent 方向，岗位要求明显强调：

- 扎实机器学习、深度学习、NLP/CV/RL 基础。
- PyTorch/TensorFlow/JAX 等框架实践能力。
- LLM、VLM、Transformer、RAG、Tool Use、Agent、Post-training、RLHF/RLVR。
- 数据构造、数据清洗、数据质量、合成数据、评测体系。
- 大规模训练、分布式训练、推理优化、低延迟 Serving。
- 搜索、推荐、广告等业务算法在部分岗位高频出现，但对模型训练方向不是主线，可作为补充知识。
- 工程能力：Python/C++/Go/Java、数据结构算法、Linux、系统设计、线上监控。
- 端到端 ownership：从模糊问题到实验、上线、监控和复盘。

## 参考来源

- OpenAI Research Engineer, Codex：强调 agentic models、coding、tool use、computer use、multi-agent coordination、long-horizon execution、post-training、RL、evals、graders、training data、diagnostics、production harness。
- OpenAI Machine Learning Engineer, Integrity：强调 PyTorch/TensorFlow、数据结构算法、搜索相关性、广告排序、LLM、distillation、SFT、policy optimization、生产部署。
- 字节 Seed 大语言模型 Agent 算法工程师：强调 Generalized Agent、Search、Coding、Interpreter、Tool Use、GUI、CodeAgent、Long-horizon Tasks、NLP/RL、PyTorch/TensorFlow/JAX。
- 字节 / TikTok 多模态与 Code AI：强调 VLM、视频理解、多模态 encoder、自适应帧率、音频和用户行为融合、代码理解与推理。
- 腾讯混元多模态算法研究：强调多模态数据构造、基础模型算法、pre-training/SFT/RL、模型评测、Diffusion、Autoregressive、CPU/GPU 加速、分布式训练与推理优化。
- 阿里大语言模型算法工程师：强调 NLP、大模型、知识表示、机器翻译、长思维链推理、对话系统、文本生成、业务落地、PyTorch/TensorFlow、Transformer/BERT/GPT/RNN/LSTM。
- 美团 Search Agent / AI 搜索方向：强调联网搜索、边想边搜、Deep Research、Mid-Train、SFT、Generative Reward Model、RLVR、Agentic RL、搜索链路 Query 理解、语义召回、排序、任务拆解、文本改写、多轮对话、数据挖掘和评估迭代。
- Meta / Google DeepMind MLE 面试趋势：强调算法编码、Applied ML Coding、ML System Design、训练到 Serving、推荐/搜索/广告、评测框架、MLOps、JAX/PyTorch、分布式训练、线上监控。

## 能力矩阵

| 能力域 | JD 高频关键词 | 你需要掌握到什么程度 | 当前知识库位置 |
| --- | --- | --- | --- |
| 数学与 ML 基础 | 概率统计、优化、线代、传统 ML | 能解释核心公式、适用条件、评价指标 | [数学基础](<../../01_机器学习基础/数学与机器学习/数学基础.md>)、[机器学习基础](<../../01_机器学习基础/数学与机器学习/机器学习基础.md>)、[统计推断与因果推断基础](<../../01_机器学习基础/数学与机器学习/统计推断与因果推断基础.md>) |
| 深度学习基础 | MLP、Normalization、激活函数、梯度问题 | 能解释训练稳定性、梯度流、正则化 | [深度学习基础](<../../01_机器学习基础/深度学习基础/深度学习基础.md>)、[MLP](<../../01_机器学习基础/深度学习基础/Multi-Layer Perceptron.md>)、[Normalization](<../../01_机器学习基础/深度学习基础/Normalization.md>)、[梯度问题](<../../03_训练优化与对齐/训练稳定性/怎么解决梯度爆炸和梯度消失？.md>) |
| Transformer / LLM | Transformer、BERT、GPT、Decoder-only、RoPE、GQA、MoE | 能讲结构、复杂度、训练/推理影响 | [Transformer](<../../02_大模型/基础架构/Transformer.md>)、[Self-Attention](<../../02_大模型/基础架构/Self-Attention.md>)、[Decoder-only](<../../02_大模型/基础架构/在生成式大模型中，为何通常采用 Decoder-only 架构而非 Encoder-Decoder 结构？.md>)、[RoPE](<../../02_大模型/基础架构/RoPE.md>)、[GQA](<../../02_大模型/基础架构/GQA.md>)、[MoE](<../../02_大模型/基础架构/MoE.md>) |
| Post-training | SFT、RLHF、DPO、PPO、GRPO、Reward Model | 能讲数据、目标函数、训练流程、风险 | [Post-training](<../../03_训练优化与对齐/后训练与对齐/Post-training.md>)、[SFT](<../../03_训练优化与对齐/后训练与对齐/SFT.md>)、[RLHF](<../../03_训练优化与对齐/后训练与对齐/RLHF.md>)、[DPO](<../../03_训练优化与对齐/后训练与对齐/DPO.md>)、[PPO](<../../03_训练优化与对齐/后训练与对齐/PPO.md>)、[GRPO](<../../03_训练优化与对齐/后训练与对齐/GRPO.md>)、[Reward Model 与 Grader](<../../03_训练优化与对齐/后训练与对齐/Reward_Model与Grader.md>)、[RLVR 与 Agentic RL](<../../03_训练优化与对齐/后训练与对齐/RLVR与Agentic_RL.md>) |
| Agent | Tool Use、GUI、CodeAgent、Long-horizon、Workflow、MCP | 能设计 Agent Loop、工具、记忆、评测和安全边界 | [Agent 开发完整流程](<../../10_Agent/基础概念/Agent开发完整流程.md>)、[Agent](<../../10_Agent/基础概念/Agent.md>)、[Workflow](<../../10_Agent/基础概念/Workflow.md>)、[Tool Call](<../../10_Agent/基础概念/Tool_Call与Function_Calling.md>)、[MCP](<../../10_Agent/基础概念/MCP.md>)、[生产级 Agent 案例](<../../10_Agent/基础概念/生产级Agent案例.md>) |
| 评测体系 | Evals、graders、benchmark、diagnostics、failure analysis | 能设计自动化评测、bad case、trajectory 归因 | [模型评测与实验设计](<../../04_评测实验与数据质量/模型评测与实验设计.md>)、[LLM Judge](<../../04_评测实验与数据质量/LLM_Judge.md>)、[数据泄漏与 Benchmark 污染](<../../04_评测实验与数据质量/数据泄漏与Benchmark污染.md>)、[Agent Eval](<../../10_Agent/基础概念/Agent_Eval.md>)、[Harness](<../../10_Agent/基础概念/Harness.md>)、[Trajectory](<../../10_Agent/基础概念/Trajectory与Observability.md>) |
| 多模态 | VLM、CLIP、BLIP、Diffusion、Video Understanding、OCR | 能讲视觉编码、多模态融合、数据构造、评测 | [VLM 与 Vision Instruction Tuning](<../../06_视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>)、[CLIP](<../../06_视觉多模态与生成模型/多模态模型/CLIP.md>)、[BLIP](<../../06_视觉多模态与生成模型/多模态模型/BLIP.md>)、[Video Understanding](<../../06_视觉多模态与生成模型/多模态模型/Video_Understanding.md>)、[OCR 与文档理解](<../../06_视觉多模态与生成模型/视觉基础/OCR与文档理解.md>)、[Multimodal Grounding](<../../06_视觉多模态与生成模型/多模态模型/Multimodal_Grounding.md>) |
| 数据工程 | 数据清洗、合成数据、Hive、Spark、Feature Store、Data Quality | 能搭数据管线、做数据版本和质量控制 | [数据工程与数据质量](<../../04_评测实验与数据质量/数据工程与数据质量.md>)、[训练数据构造与合成数据](<../../04_评测实验与数据质量/训练数据构造与合成数据.md>)、[Hive、Spark 与 Feature Store](<../../04_评测实验与数据质量/Hive_Spark与Feature_Store.md>) |
| 训练系统 | Distributed Training、FSDP、DeepSpeed、Megatron、JAX | 能解释并行策略、显存优化、吞吐瓶颈 | [训练优化与大模型训练](<../../03_训练优化与对齐/训练优化与大模型训练.md>)、[DeepSpeed](<../../03_训练优化与对齐/训练框架与并行/DeepSpeed.md>)、[ZeRO](<../../03_训练优化与对齐/训练框架与并行/ZeRO.md>)、[FSDP](<../../03_训练优化与对齐/训练框架与并行/FSDP.md>)、[Megatron-LM](<../../03_训练优化与对齐/训练框架与并行/Megatron_LM.md>)、[JAX 与 XLA](<../../03_训练优化与对齐/训练框架与并行/JAX与XLA.md>)、[Loss 异常与收敛排查](<../../03_训练优化与对齐/训练稳定性/Loss异常与收敛排查.md>) |
| 推理部署 | vLLM、TensorRT-LLM、Quantization、Dynamic Batching、p99 | 能设计低延迟高吞吐推理服务 | [模型部署与推理工程](<../../05_推理部署与系统/推理工程/模型部署与推理工程.md>)、[vLLM](<../../05_推理部署与系统/推理工程/vLLM.md>)、[TensorRT-LLM](<../../05_推理部署与系统/推理工程/TensorRT_LLM.md>)、[Batching](<../../05_推理部署与系统/推理工程/Batching.md>)、[KV Cache](<../../05_推理部署与系统/推理工程/KV_Cache与Prefill_Decode.md>)、[量化](<../../05_推理部署与系统/推理工程/量化.md>)、[CUDA Graph](<../../05_推理部署与系统/推理工程/CUDA_Graph.md>) |
| 系统工程 | Linux、C++、Python、服务化、监控、回滚、CI/CD | 能把模型稳定上线并排障 | [MLOps 与模型生产化](<../../05_推理部署与系统/系统设计/MLOps与模型生产化.md>)、[Serving](<../../05_推理部署与系统/推理工程/Serving.md>)、[CUDA 与 Triton 基础](<../../05_推理部署与系统/推理工程/CUDA与Triton基础.md>)、[编程与算法工程能力](<../../07_Python与工程/编程与算法工程能力.md>) |
| 编码能力 | 数据结构算法、Applied ML Coding、Attention from scratch | 能写可运行、可测试、复杂度清楚的代码 | [Applied ML Coding](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)、[Beam Search](<../../07_Python与工程/PyTorch/Beam_Search.md>)、[算法刷题](<../../07_Python与工程/算法刷题/刷题技巧.md>) |

## 模型训练方向最该掌握的 7 个方向

### LLM 评测与 Grader

原因：OpenAI、字节、美团 Agent 岗位都强调 evals、graders、diagnostics、模型行为分析。

需要知道：

- Rule-based eval。
- [LLM-as-a-Judge](<../../04_评测实验与数据质量/LLM_Judge.md>)。
- Pairwise preference。
- [Reward Model / Generative Reward Model](<../../03_训练优化与对齐/后训练与对齐/Reward_Model与Grader.md>)。
- [Agent trajectory 评测](<../../10_Agent/基础概念/Trajectory与Observability.md>)。
- [数据泄漏、benchmark contamination](<../../04_评测实验与数据质量/数据泄漏与Benchmark污染.md>)。
- [Bad case 聚类和错误归因](<../../04_评测实验与数据质量/实验分析与问题排查.md>)。

### SFT / RLHF / RLVR / Agentic RL

原因：大模型算法 JD 里 Post-training 已经是核心关键词，美团 Search Agent 明确提到 Generative Reward Model、RLVR、Agentic RL。

需要知道：

- [SFT 数据格式和训练目标](<../../03_训练优化与对齐/后训练与对齐/SFT.md>)。
- [RLHF 三阶段：SFT、Reward Model、PPO](<../../03_训练优化与对齐/后训练与对齐/RLHF.md>)。
- [DPO](<../../03_训练优化与对齐/后训练与对齐/DPO.md>) / [GRPO](<../../03_训练优化与对齐/后训练与对齐/GRPO.md>) 和 [PPO](<../../03_训练优化与对齐/后训练与对齐/PPO.md>) 的差异。
- [Verifiable Reward：为什么代码、数学、搜索任务适合 RLVR](<../../03_训练优化与对齐/后训练与对齐/RLVR与Agentic_RL.md>)。
- [Agentic RL：针对多步工具调用和长任务轨迹做强化学习](<../../03_训练优化与对齐/后训练与对齐/RLVR与Agentic_RL.md>)。

### 大规模训练系统

原因：大厂不只招会调模型的人，也招能把训练跑起来、跑得快、跑得稳的人。

需要知道：

- DDP、[FSDP](<../../03_训练优化与对齐/训练框架与并行/FSDP.md>)、[ZeRO](<../../03_训练优化与对齐/训练框架与并行/ZeRO.md>)。
- [Tensor Parallel、Pipeline Parallel、Sequence Parallel](<../../05_推理部署与系统/推理工程/推理优化方法_并行策略.md>)。
- [Megatron-LM](<../../03_训练优化与对齐/训练框架与并行/Megatron_LM.md>)。
- [Checkpoint 保存、恢复、切分](<../../03_训练优化与对齐/训练框架与并行/Checkpoint.md>)。
- [训练吞吐、显存、通信瓶颈](<../../03_训练优化与对齐/训练稳定性/Loss异常与收敛排查.md>)。
- [JAX/XLA 的基本思想](<../../03_训练优化与对齐/训练框架与并行/JAX与XLA.md>)。

### 多模态与视频理解

原因：字节、腾讯、阿里多模态岗位都强调 VLM、Omni、视频理解、Diffusion、AR。

需要知道：

- [CLIP](<../../06_视觉多模态与生成模型/多模态模型/CLIP.md>) / [BLIP](<../../06_视觉多模态与生成模型/多模态模型/BLIP.md>) / [VLM 基础](<../../06_视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>)。
- [Vision Instruction Tuning](<../../06_视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>)。
- [Multimodal Grounding](<../../06_视觉多模态与生成模型/多模态模型/Multimodal_Grounding.md>)。
- [OCR / Document Understanding](<../../06_视觉多模态与生成模型/视觉基础/OCR与文档理解.md>)。
- [Video Understanding：帧采样、时序建模、音频融合](<../../06_视觉多模态与生成模型/多模态模型/Video_Understanding.md>)。
- [Diffusion](<../../06_视觉多模态与生成模型/生成模型/Latent Diffusion Models.md>) 与 [Autoregressive](<../../02_大模型/基础架构/Autoregressive Model.md>) 生成范式差异。

### 推理系统与性能优化

原因：JD 高频出现 inference optimization、low latency、throughput、GPU acceleration。

需要知道：

- [Prefill / Decode](<../../05_推理部署与系统/推理工程/KV_Cache与Prefill_Decode.md>)。
- [KV Cache](<../../05_推理部署与系统/推理工程/KV_Cache与Prefill_Decode.md>)。
- [Continuous Batching / Dynamic Batching](<../../05_推理部署与系统/推理工程/Batching.md>)。
- [vLLM / PagedAttention](<../../05_推理部署与系统/推理工程/vLLM.md>)。
- [TensorRT-LLM](<../../05_推理部署与系统/推理工程/TensorRT_LLM.md>)。
- [Speculative Decoding](<../../05_推理部署与系统/推理工程/Speculative_Decoding.md>)。
- [量化：INT8、FP8、AWQ、GPTQ](<../../05_推理部署与系统/推理工程/量化.md>)。
- p50/p95/p99 latency 和吞吐权衡。

### MLOps 与生产监控

原因：OpenAI/Meta/Google 都强调 production readiness、observability、reproducibility、monitoring。

需要知道：

- [数据版本、模型版本、实验追踪](<../../05_推理部署与系统/系统设计/MLOps与模型生产化.md>)。
- [模型注册和发布](<../../05_推理部署与系统/系统设计/MLOps与模型生产化.md>)。
- [灰度、回滚、A/B Test](<../../05_推理部署与系统/系统设计/MLOps与模型生产化.md>)。
- [Feature Drift / Data Drift / Concept Drift](<../../05_推理部署与系统/系统设计/MLOps与模型生产化.md>)。
- [训练-Serving Skew](<../../04_评测实验与数据质量/Hive_Spark与Feature_Store.md>)。
- [线上监控和告警](<../../05_推理部署与系统/系统设计/MLOps与模型生产化.md>)。

### Applied ML Coding

原因：很多面试不再只考 LeetCode，还会考小型 ML 实现。

需要能手写：

- [Scaled Dot-Product Attention](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)。
- [Top-k / Top-p Sampling](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)。
- [Beam Search](<../../07_Python与工程/PyTorch/Beam_Search.md>)。
- [AUC / PR-AUC / NDCG](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)。
- [小型训练循环](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)。
- [Gradient Accumulation](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)。
- [RAG chunking](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)。
- [简单 Eval Harness](<../../10_Agent/基础概念/Harness.md>)。

## 建议学习优先级

### P0：面试和工作都最常用

1. LLM 评测与 Grader。
2. SFT / RLHF / DPO / GRPO / RLVR。
3. 分布式训练：FSDP、ZeRO、Megatron。
4. 推理优化：KV Cache、Batching、量化、vLLM。
5. 数据质量、数据版本和训练可复现。

### P1：区分中高级候选人

1. MLOps：数据版本、模型注册、灰度、监控。
2. 多模态：VLM、Video Understanding、OCR、Grounding。
3. JAX/XLA、CUDA/Triton 基础。
4. Agent Workflow / Tool Use / MCP / Eval。

### P2：加分项

1. 顶会论文阅读和复现。
2. 开源贡献。
3. 训练数据、评测数据、后训练数据的构造和清洗经验。
4. 端到端项目作品集：数据、训练、评测、部署、监控闭环。

## 面试准备路线

### 第一阶段：补齐基础

- 复习 `01_机器学习基础/`。
- 复习 `02_大模型/基础架构/`。
- 手写 Attention、AUC、Top-k Sampling、训练循环。

### 第二阶段：补齐模型训练工程

- 系统学习分布式训练、checkpoint、混合精度和训练稳定性。
- 能解释 FSDP、ZeRO、Tensor Parallel、Pipeline Parallel 的适用场景。
- 能从显存、吞吐、通信、数据加载和 checkpoint 恢复角度排查训练问题。

### 第三阶段：补齐大模型工程

- 系统学习 Post-training。
- 系统学习推理优化。
- 能从吞吐、延迟、显存、成本四个角度分析大模型服务。

### 第四阶段：补齐 Agent

- 学习 `10_Agent/` 下的基础概念。
- 能设计 Agent Workflow、Tool Schema、Memory、Guardrails 和 Eval。
- 能解释 Vibe Coding 与 Spec Coding 的工程价值。

### 第五阶段：项目表达

把自己的项目统一整理成这个结构：

```text
业务问题
  -> 数据来源与质量
  -> 模型/算法方案
  -> 训练与调参
  -> 评测指标
  -> 线上部署
  -> A/B 或离线回归
  -> Bad case 分析
  -> 迭代收益
```

## 面试官真正想听到什么

不是只背模型名，而是证明你能闭环：

- 为什么选这个模型？
- 数据怎么构造？
- 指标怎么定义？
- 评测是否可信？
- 线上怎么部署？
- 延迟和成本如何控制？
- Bad case 怎么归因？
- 下一轮实验怎么设计？

## 补充知识：搜索 / 推荐 / 广告算法

这部分不是你的主攻方向，但可以作为补充知识保留。原因是很多大厂算法岗 JD 会混合出现搜索、推荐、广告、排序、召回和 A/B 测试关键词；即使做模型训练，理解这些业务算法也有助于看懂数据来源、评测指标和线上反馈。

需要知道到“能交流”的程度即可：

- 召回、粗排、精排、重排。
- Two-Tower、DSSM、DIN、DeepFM、DCN。
- CTR/CVR/GMV/留存/完播率等业务指标。
- AUC、GAUC、NDCG、Recall@K、MRR。
- 冷启动、曝光偏差、位置偏差、样本选择偏差。
- 在线 A/B 测试与离线指标不一致问题。

对应知识库位置：

- [11_搜索推荐广告/](<../../11_搜索推荐广告/>)

## 已补充到知识库的专题

本轮已根据 JD 缺口补充：

- [统计推断与因果推断基础](<../../01_机器学习基础/数学与机器学习/统计推断与因果推断基础.md>)
- [Hive、Spark 与 Feature Store](<../../04_评测实验与数据质量/Hive_Spark与Feature_Store.md>)
- [训练数据构造与合成数据](<../../04_评测实验与数据质量/训练数据构造与合成数据.md>)
- [FSDP](<../../03_训练优化与对齐/训练框架与并行/FSDP.md>)
- [Megatron-LM](<../../03_训练优化与对齐/训练框架与并行/Megatron_LM.md>)
- [JAX 与 XLA](<../../03_训练优化与对齐/训练框架与并行/JAX与XLA.md>)
- [Loss 异常与收敛排查](<../../03_训练优化与对齐/训练稳定性/Loss异常与收敛排查.md>)
- [SFT](<../../03_训练优化与对齐/后训练与对齐/SFT.md>)
- [RLHF](<../../03_训练优化与对齐/后训练与对齐/RLHF.md>)
- [Reward Model 与 Grader](<../../03_训练优化与对齐/后训练与对齐/Reward_Model与Grader.md>)
- [RLVR 与 Agentic RL](<../../03_训练优化与对齐/后训练与对齐/RLVR与Agentic_RL.md>)
- [Speculative Decoding](<../../05_推理部署与系统/推理工程/Speculative_Decoding.md>)
- [TensorRT-LLM](<../../05_推理部署与系统/推理工程/TensorRT_LLM.md>)
- [CUDA Graph](<../../05_推理部署与系统/推理工程/CUDA_Graph.md>)
- [CUDA 与 Triton 基础](<../../05_推理部署与系统/推理工程/CUDA与Triton基础.md>)
- [MLOps 与模型生产化](<../../05_推理部署与系统/系统设计/MLOps与模型生产化.md>)
- [LLM Judge](<../../04_评测实验与数据质量/LLM_Judge.md>)
- [数据泄漏与 Benchmark 污染](<../../04_评测实验与数据质量/数据泄漏与Benchmark污染.md>)
- [VLM 与 Vision Instruction Tuning](<../../06_视觉多模态与生成模型/多模态模型/VLM与Vision_Instruction_Tuning.md>)
- [OCR 与文档理解](<../../06_视觉多模态与生成模型/视觉基础/OCR与文档理解.md>)
- [Video Understanding](<../../06_视觉多模态与生成模型/多模态模型/Video_Understanding.md>)
- [Multimodal Grounding](<../../06_视觉多模态与生成模型/多模态模型/Multimodal_Grounding.md>)
- [生产级 Agent 案例](<../../10_Agent/基础概念/生产级Agent案例.md>)
- [Applied ML Coding](<../../07_Python与工程/PyTorch/Applied_ML_Coding.md>)
- [Beam Search](<../../07_Python与工程/PyTorch/Beam_Search.md>)

补充知识已保留：

- [推荐系统基础](<../../11_搜索推荐广告/推荐系统基础.md>)
- [搜索系统基础](<../../11_搜索推荐广告/搜索系统基础.md>)
- [广告排序与 CTR 预估](<../../11_搜索推荐广告/广告排序与CTR预估.md>)
- [排序指标与 A/B 测试](<../../11_搜索推荐广告/排序指标与A_B测试.md>)

## 仍可继续扩展的方向

当前 JD 高频基础缺口已补齐。后续如果继续扩展，建议优先围绕真实项目和论文精读补：

- 大模型训练稳定性、loss 异常和收敛问题案例。
- 后训练数据构造、Reward / Grader 设计和 RLVR 实验案例。
- 大模型训练和推理性能分析案例。
- 多模态视频理解和 Grounding 的 benchmark 与数据集。
- 搜索/推荐/广告的工业论文复现可以作为可选补充，不作为当前主线。
