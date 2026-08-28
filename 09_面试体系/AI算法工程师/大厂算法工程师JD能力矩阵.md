# 大厂算法工程师 JD 能力矩阵

## 调研结论

近两年大厂算法工程师 JD 的重心已经从“会训练一个模型”升级为“能把模型、数据、评测、系统和业务闭环串起来”。尤其是大模型和 Agent 方向，岗位要求明显强调：

- 扎实机器学习、深度学习、NLP/CV/RL 基础。
- PyTorch/TensorFlow/JAX 等框架实践能力。
- LLM、VLM、Transformer、RAG、Tool Use、Agent、Post-training、RLHF/RLVR。
- 数据构造、数据清洗、数据质量、合成数据、评测体系。
- 大规模训练、分布式训练、推理优化、低延迟 Serving。
- 搜索、推荐、广告、排序、召回、重排、A/B 测试等业务算法。
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
| 数学与 ML 基础 | 概率统计、优化、线代、传统 ML | 能解释核心公式、适用条件、评价指标 | `01_机器学习基础/` |
| 深度学习基础 | MLP、Normalization、激活函数、梯度问题 | 能解释训练稳定性、梯度流、正则化 | `01_机器学习基础/`、`03_训练优化与对齐/` |
| Transformer / LLM | Transformer、BERT、GPT、Decoder-only、RoPE、GQA、MoE | 能讲结构、复杂度、训练/推理影响 | `02_大模型/`、`07_Python与工程/PyTorch/` |
| Post-training | SFT、RLHF、DPO、PPO、GRPO、Reward Model | 能讲数据、目标函数、训练流程、风险 | `03_训练优化与对齐/` |
| Agent | Tool Use、GUI、CodeAgent、Long-horizon、Workflow、MCP | 能设计 Agent Loop、工具、记忆、评测和安全边界 | `10_Agent/` |
| 评测体系 | Evals、graders、benchmark、diagnostics、failure analysis | 能设计自动化评测、bad case、trajectory 归因 | `04_评测实验与数据质量/`、`10_Agent/` |
| 多模态 | VLM、CLIP、BLIP、Diffusion、Video Understanding、OCR | 能讲视觉编码、多模态融合、数据构造、评测 | `06_视觉多模态与生成模型/` |
| 搜索推荐广告 | Retrieval、Ranking、Re-ranking、CTR、AUC、Query Understanding | 能设计召回/排序/重排系统，理解业务指标 | `11_搜索推荐广告/` |
| 数据工程 | 数据清洗、合成数据、Hive、Spark、Feature Store、Data Quality | 能搭数据管线、做数据版本和质量控制 | `04_评测实验与数据质量/` |
| 训练系统 | Distributed Training、FSDP、DeepSpeed、Megatron、JAX | 能解释并行策略、显存优化、吞吐瓶颈 | `03_训练优化与对齐/` |
| 推理部署 | vLLM、TensorRT-LLM、Quantization、Dynamic Batching、p99 | 能设计低延迟高吞吐推理服务 | `05_推理部署与系统/` |
| 系统工程 | Linux、C++、Python、服务化、监控、回滚、CI/CD | 能把模型稳定上线并排障 | `05_推理部署与系统/`、`07_Python与工程/` |
| 编码能力 | 数据结构算法、Applied ML Coding、Attention from scratch | 能写可运行、可测试、复杂度清楚的代码 | `07_Python与工程/` |

## 你当前最该补的 8 个方向

### 1. 搜索 / 推荐 / 广告算法

原因：字节、美团、Meta、Google、阿里都大量围绕搜索、推荐、广告和排序招算法工程师。大模型也在进入搜索推荐链路，例如 Query 理解、语义召回、重排、生成式推荐、Agentic Search。

需要知道：

- 召回、粗排、精排、重排。
- Two-Tower、DSSM、DIN、DeepFM、DCN。
- CTR/CVR/GMV/留存/完播率等业务指标。
- AUC、GAUC、NDCG、Recall@K、MRR。
- 冷启动、曝光偏差、位置偏差、样本选择偏差。
- 在线 A/B 测试与离线指标不一致问题。

### 2. LLM 评测与 Grader

原因：OpenAI、字节、美团 Agent 岗位都强调 evals、graders、diagnostics、模型行为分析。

需要知道：

- Rule-based eval。
- LLM-as-a-Judge。
- Pairwise preference。
- Reward Model / Generative Reward Model。
- Agent trajectory 评测。
- 数据泄漏、benchmark contamination。
- Bad case 聚类和错误归因。

### 3. SFT / RLHF / RLVR / Agentic RL

原因：大模型算法 JD 里 Post-training 已经是核心关键词，美团 Search Agent 明确提到 Generative Reward Model、RLVR、Agentic RL。

需要知道：

- SFT 数据格式和训练目标。
- RLHF 三阶段：SFT、Reward Model、PPO。
- DPO/GRPO 和 PPO 的差异。
- Verifiable Reward：为什么代码、数学、搜索任务适合 RLVR。
- Agentic RL：针对多步工具调用和长任务轨迹做强化学习。

### 4. 多模态与视频理解

原因：字节、腾讯、阿里多模态岗位都强调 VLM、Omni、视频理解、Diffusion、AR。

需要知道：

- CLIP/BLIP/VLM 基础。
- Vision Instruction Tuning。
- Multimodal Grounding。
- OCR / Document Understanding。
- Video Understanding：帧采样、时序建模、音频融合。
- Diffusion 与 Autoregressive 生成范式差异。

### 5. 大规模训练系统

原因：大厂不只招会调模型的人，也招能把训练跑起来、跑得快、跑得稳的人。

需要知道：

- DDP、FSDP、ZeRO。
- Tensor Parallel、Pipeline Parallel、Sequence Parallel。
- Megatron-LM。
- Checkpoint 保存、恢复、切分。
- 训练吞吐、显存、通信瓶颈。
- JAX/XLA 的基本思想。

### 6. 推理系统与性能优化

原因：JD 高频出现 inference optimization、low latency、throughput、GPU acceleration。

需要知道：

- Prefill / Decode。
- KV Cache。
- Continuous Batching / Dynamic Batching。
- vLLM / PagedAttention。
- TensorRT-LLM。
- Speculative Decoding。
- 量化：INT8、FP8、AWQ、GPTQ。
- p50/p95/p99 latency 和吞吐权衡。

### 7. MLOps 与生产监控

原因：OpenAI/Meta/Google 都强调 production readiness、observability、reproducibility、monitoring。

需要知道：

- 数据版本、模型版本、实验追踪。
- 模型注册和发布。
- 灰度、回滚、A/B Test。
- Feature Drift / Data Drift / Concept Drift。
- 训练-Serving Skew。
- 线上监控和告警。

### 8. Applied ML Coding

原因：很多面试不再只考 LeetCode，还会考小型 ML 实现。

需要能手写：

- Scaled Dot-Product Attention。
- Top-k / Top-p Sampling。
- Beam Search。
- AUC / PR-AUC / NDCG。
- 小型训练循环。
- Gradient Accumulation。
- RAG chunking。
- 简单 Eval Harness。

## 建议学习优先级

### P0：面试和工作都最常用

1. 搜索推荐广告基础。
2. LLM 评测与 Grader。
3. SFT / RLHF / DPO / GRPO / RLVR。
4. Agent Workflow / Tool Use / MCP / Eval。
5. 推理优化：KV Cache、Batching、量化、vLLM。

### P1：区分中高级候选人

1. 分布式训练：FSDP、ZeRO、Megatron。
2. MLOps：数据版本、模型注册、灰度、监控。
3. 多模态：VLM、Video Understanding、OCR、Grounding。
4. JAX/XLA、CUDA/Triton 基础。

### P2：加分项

1. 顶会论文阅读和复现。
2. 开源贡献。
3. 业务场景沉淀：电商、搜索、推荐、广告、内容安全。
4. 端到端项目作品集：数据、训练、评测、部署、监控闭环。

## 面试准备路线

### 第一阶段：补齐基础

- 复习 `01_机器学习基础/`。
- 复习 `02_大模型/基础架构/`。
- 手写 Attention、AUC、Top-k Sampling、训练循环。

### 第二阶段：补齐业务算法

- 系统学习搜索、推荐、广告三件套。
- 能设计一个完整的召回-排序-重排系统。
- 能解释离线指标和线上 A/B 指标不一致的原因。

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

## 已补充到知识库的专题

本轮已根据 JD 缺口补充：

- `01_机器学习基础/数学与机器学习/统计推断与因果推断基础.md`
- `11_搜索推荐广告/推荐系统基础.md`
- `11_搜索推荐广告/搜索系统基础.md`
- `11_搜索推荐广告/广告排序与CTR预估.md`
- `11_搜索推荐广告/排序指标与A_B测试.md`
- `04_评测实验与数据质量/Hive_Spark与Feature_Store.md`
- `03_训练优化与对齐/训练框架与并行/FSDP.md`
- `03_训练优化与对齐/训练框架与并行/Megatron_LM.md`
- `03_训练优化与对齐/训练框架与并行/JAX与XLA.md`
- `03_训练优化与对齐/后训练与对齐/SFT.md`
- `03_训练优化与对齐/后训练与对齐/RLHF.md`
- `03_训练优化与对齐/后训练与对齐/Reward_Model与Grader.md`
- `03_训练优化与对齐/后训练与对齐/RLVR与Agentic_RL.md`
- `05_推理部署与系统/推理工程/Speculative_Decoding.md`
- `05_推理部署与系统/推理工程/TensorRT_LLM.md`
- `05_推理部署与系统/推理工程/CUDA_Graph.md`
- `05_推理部署与系统/推理工程/CUDA与Triton基础.md`
- `05_推理部署与系统/系统设计/MLOps与模型生产化.md`
- `04_评测实验与数据质量/LLM_Judge.md`
- `04_评测实验与数据质量/数据泄漏与Benchmark污染.md`
- `06_视觉多模态与生成模型/视觉基础/OCR与文档理解.md`
- `06_视觉多模态与生成模型/多模态模型/Video_Understanding.md`
- `06_视觉多模态与生成模型/多模态模型/Multimodal_Grounding.md`
- `10_Agent/基础概念/生产级Agent案例.md`
- `07_Python与工程/PyTorch/Applied_ML_Coding.md`

## 仍可继续扩展的方向

当前 JD 高频基础缺口已补齐。后续如果继续扩展，建议优先围绕真实项目和论文精读补：

- 搜索/推荐/广告的工业论文复现。
- Agentic Search / Deep Research 的系统案例。
- 大模型训练和推理性能分析案例。
- 多模态视频理解和 Grounding 的 benchmark 与数据集。
