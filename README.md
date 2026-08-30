# Studying 知识库索引

这个目录已经按“知识领域 + 查找场景”重排。根目录只保留本索引，具体笔记放到编号目录中，便于后续按主题定位。

## 目录地图

```text
Studying/
├── 01_机器学习基础/
│   ├── 数学与机器学习/
│   └── 深度学习基础/
├── 02_大模型/
│   ├── 基础架构/
│   └── 应用与问题/
├── 03_训练优化与对齐/
│   ├── 训练框架与并行/
│   ├── 后训练与对齐/
│   ├── 损失函数与激活函数/
│   └── 训练稳定性/
├── 04_评测实验与数据质量/
├── 05_推理部署与系统/
│   ├── 推理工程/
│   ├── 系统设计/
│   └── 工程工具/
├── 06_视觉多模态与生成模型/
│   ├── 视觉基础/
│   ├── 多模态模型/
│   └── 生成模型/
├── 07_Python与工程/
│   ├── Python语法/
│   ├── 算法刷题/
│   ├── 常用库/
│   ├── PyTorch/
│   ├── 深度学习框架/
│   └── 常用函数/
├── 08_程序分析与代码智能/
│   ├── 静态分析基础/
│   ├── 安全漏洞分析/
│   ├── 代码智能模型/
│   └── 分析工具/
├── 09_面试体系/
│   ├── AI算法工程师/
│   ├── 项目闭环与训练排查/
│   └── 综合面试题/
├── 10_Agent/
│   ├── 基础概念/
│   ├── Skills/
│   ├── AI_Coding/
│   └── 资料索引/
├── 11_搜索推荐广告/
└── README.md
```

## 快速查找

| 想查什么 | 入口 |
| --- | --- |
| 数学、统计推断、因果推断、机器学习、深度学习基础 | [01_机器学习基础/](<01_机器学习基础/>) |
| Transformer、注意力、MoE、RAG、CoT、Prompt、大模型发展历史与 SOTA 演进 | [02_大模型/](<02_大模型/>) |
| DeepSpeed、ZeRO、FSDP、Megatron-LM、JAX/XLA、LoRA、后训练发展史、DPO、PPO、GRPO、Checkpoint、Loss 异常、损失函数、模型训练路线 | [03_训练优化与对齐/](<03_训练优化与对齐/>) |
| 评测、LLM Judge、Benchmark 污染、训练数据构造、合成数据、Hive/Spark、Feature Store、数据质量 | [04_评测实验与数据质量/](<04_评测实验与数据质量/>) |
| 推理框架、Serving、KV Cache、Batching、量化、TensorRT-LLM、CUDA Graph、CUDA/Triton、系统设计、Docker | [05_推理部署与系统/](<05_推理部署与系统/>) |
| VLM、CLIP、BLIP、UNet、Latent Diffusion、OCR、Video Understanding、Grounding | [06_视觉多模态与生成模型/](<06_视觉多模态与生成模型/>) |
| Python 语法、刷题、常用库、PyTorch/TensorFlow/JAX 框架选型、Hugging Face、Applied ML Coding、Beam Search | [07_Python与工程/](<07_Python与工程/>) |
| 静态分析、数据流、污点分析、CodeQL、代码大模型 | [08_程序分析与代码智能/](<08_程序分析与代码智能/>) |
| 面试复习体系、项目闭环复盘、训练排查和综合面试题 | [09_面试体系/](<09_面试体系/>) |
| Agent、Workflow、Skills、MCP、Tool Call、Memory、Context Engineering、AI Coding | [10_Agent/](<10_Agent/>) |
| 搜索、推荐、广告、召回、排序、CTR、A/B 测试 | [11_搜索推荐广告/](<11_搜索推荐广告/>) |

## 重点入口

- [学习路线总览.md](<学习路线总览.md>)：跨目录复习路线，按模型训练、后训练、项目闭环、推理部署、多模态、Agent 和 Coding 组织。
- [09_面试体系/AI算法工程师/README.md](<09_面试体系/AI算法工程师/README.md>)：系统复习主入口。
- [09_面试体系/项目闭环与训练排查/README.md](<09_面试体系/项目闭环与训练排查/README.md>)：真实项目复盘、训练工程、数据质量和训练问题排查入口。
- [09_面试体系/项目闭环与训练排查/项目面试追问总表.md](<09_面试体系/项目闭环与训练排查/项目面试追问总表.md>)：HR 和用人领导视角的项目深挖问题总表。
- [09_面试体系/项目闭环与训练排查/用人领导视角能力地图.md](<09_面试体系/项目闭环与训练排查/用人领导视角能力地图.md>)：从招聘视角反推项目经历需要证明的能力。
- [09_面试体系/项目闭环与训练排查/简历项目表达与STAR故事库.md](<09_面试体系/项目闭环与训练排查/简历项目表达与STAR故事库.md>)：项目简历 bullet、30 秒介绍、2 分钟介绍和 STAR 故事库。
- [02_大模型/大模型发展历史与SOTA迭代框架.md](<02_大模型/大模型发展历史与SOTA迭代框架.md>)：大模型从 Transformer 到 Agent 的历史演进和 SOTA 迭代框架。
- [02_大模型/应用与问题/NLP与大语言模型.md](<02_大模型/应用与问题/NLP与大语言模型.md>)：NLP 与 LLM 复习入口。
- [03_训练优化与对齐/模型训练学习手册_预训练到后训练.md](<03_训练优化与对齐/模型训练学习手册_预训练到后训练.md>)：从预训练、SFT、DPO、GRPO/RLVR 到评测部署的模型训练路线总纲。
- [03_训练优化与对齐/后训练与对齐/README.md](<03_训练优化与对齐/后训练与对齐/README.md>)：后训练与对齐目录索引，按“英文缩写或术语 + 中文翻译”组织核心名词。
- [03_训练优化与对齐/后训练与对齐/后训练发展史与方法对比.md](<03_训练优化与对齐/后训练与对齐/后训练发展史与方法对比.md>)：后训练从 SFT、RLHF、DPO 到 RLVR/GRPO 的发展史和横向对比。
- [03_训练优化与对齐/训练优化与大模型训练.md](<03_训练优化与对齐/训练优化与大模型训练.md>)：训练优化复习入口。
- [03_训练优化与对齐/训练稳定性/Loss异常与收敛排查.md](<03_训练优化与对齐/训练稳定性/Loss异常与收敛排查.md>)：训练 loss 异常、NaN、发散和不收敛排查入口。
- [04_评测实验与数据质量/训练数据构造与合成数据.md](<04_评测实验与数据质量/训练数据构造与合成数据.md>)：训练数据、后训练数据和合成数据构造入口。
- [07_Python与工程/深度学习框架/深度学习框架选型.md](<07_Python与工程/深度学习框架/深度学习框架选型.md>)：PyTorch、TensorFlow/Keras、JAX、Hugging Face 等框架怎么选。
- [07_Python与工程/深度学习框架/PyTorch训练工程基础.md](<07_Python与工程/深度学习框架/PyTorch训练工程基础.md>)：PyTorch 训练循环、autograd、DataLoader、checkpoint 和 OOM 排查。
- [05_推理部署与系统/推理工程/模型部署与推理工程.md](<05_推理部署与系统/推理工程/模型部署与推理工程.md>)：部署与推理工程入口。
- [05_推理部署与系统/推理工程/推理框架总览.md](<05_推理部署与系统/推理工程/推理框架总览.md>)：推理框架和工程术语入口。
- [10_Agent/README.md](<10_Agent/README.md>)：Agent、Workflow、Skills、MCP、生产级 Agent、AI Coding 专题入口。
- [09_面试体系/AI算法工程师/大厂算法工程师JD能力矩阵.md](<09_面试体系/AI算法工程师/大厂算法工程师JD能力矩阵.md>)：从大厂 JD 反推的能力地图和补齐路线。
- [11_搜索推荐广告/README.md](<11_搜索推荐广告/README.md>)：搜索推荐广告专题入口。
- [05_推理部署与系统/系统设计/MLOps与模型生产化.md](<05_推理部署与系统/系统设计/MLOps与模型生产化.md>)：模型上线、监控、灰度和回滚入口。

## 分类说明

### 机器学习基础

放数学、传统机器学习、深度学习基础概念。

- `数学与机器学习/`：数学基础、机器学习基础、统计推断、因果推断、余弦相似度等。
- `深度学习基础/`：MLP、Normalization、正则化、feature map、高阶特征等。

### 大模型

放模型结构、生成范式、注意力机制、大模型发展历史以及大模型应用层问题。

- `基础架构/`：Transformer、Self-Attention、Autoregressive Model、Dense Model、MoE、GQA、MLA、RMSNorm、RoPE 等。
- `应用与问题/`：RAG、CoT、Prompt 调优、模型幻觉、NLP 与大语言模型综述等。

### 训练优化与对齐

放模型训练路线、训练工程、分布式优化、后训练、对齐算法和训练稳定性。

- `训练框架与并行/`：DeepSpeed、ZeRO、FSDP、Megatron-LM、JAX/XLA、Mixed Precision Training、Checkpoint 等训练工程概念。
- `后训练与对齐/`：后训练发展史、Post-training 后训练、SFT 监督微调、RLHF 基于人类反馈的强化学习、DPO 直接偏好优化、PPO 近端策略优化、GRPO 组相对策略优化、RLVR 可验证奖励强化学习、Agentic RL 智能体强化学习、Reward Model 与 Grader、LoRA、Knowledge Distillation、Reward Collapse。
- `损失函数与激活函数/`：分类损失、回归损失、激活函数、GeLU。
- `训练稳定性/`：梯度爆炸、梯度消失、Loss 异常、NaN、发散和收敛排查等。

### 评测实验与数据质量

放模型评测、LLM Judge、Benchmark 污染、训练数据构造、合成数据、Hive/Spark、Feature Store、实验设计、数据治理、问题排查和 AI 产品判断。Agent 评测脚手架相关内容统一放入 `10_Agent/`。

### 推理部署与系统

放推理工程、Serving、系统设计和工程工具。

- `推理工程/`：模型部署、推理框架、Serving、KV Cache、Batching、量化、Speculative Decoding、TensorRT-LLM、CUDA Graph、CUDA/Triton、算子、vLLM、ms-swift 等工程概念。
- `系统设计/`：AI 系统设计、MLOps 与模型生产化。Agent 调度与编排相关概念统一放入 `10_Agent/`。
- `工程工具/`：Docker 使用手册。

### 视觉多模态与生成模型

放计算机视觉、多模态模型和生成模型。

- `视觉基础/`：Bicubic Interpolation、OCR 与文档理解。
- `多模态模型/`：VLM、Vision Instruction Tuning、CLIP、BLIP、Video Understanding、Multimodal Grounding。
- `生成模型/`：UNet、Latent Diffusion Models。

### Python 与工程

放 Python 语言、刷题模板、工程库、深度学习框架基础和 PyTorch API。

- `Python语法/`：字典、内置函数、异常、列表推导式、lambda、collections、Counter、pairwise 等。
- `算法刷题/`：回溯、并查集、刷题技巧、下一个排列。
- `常用库/`：argparse、Dataloader、re、tqdm、可视化。
- `深度学习框架/`：PyTorch、TensorFlow/Keras、JAX、Hugging Face、Trainer、Accelerate、Lightning 等框架选型和训练脚手架基础。
- `PyTorch/`：torch、torchvision、torch.unsqueeze、Applied ML Coding、Beam Search。
- `常用函数/`：detach、enumerate、torch.inference_mode、折叠注释。

### 程序分析与代码智能

放程序分析理论、安全漏洞分析、代码大模型和工具。

- `静态分析基础/`：程序分析、静态分析、控制流、数据流、污点分析、AST/CFG/DFG。
- `安全漏洞分析/`：CWE、CVE、CWE-Bench-Java。
- `代码智能模型/`：CodeBERT、GraphCodeBERT、CodeT5、CodeT5+、UniXcoder、大模型和静态分析。
- `分析工具/`：CodeQL、Joern。

### 面试体系

放面试复习框架、项目闭环复盘、训练排查和综合面试题。

### Agent

放 Agent、Workflow、AI Coding、Skills、MCP、Tool Call、SubAgent、Memory、Context Engineering、Computer Use、Guardrails、Trajectory、Harness、Hermes、生产级 Agent 案例和 Agent Eval 等相关内容。

### 搜索推荐广告

放搜索、推荐、广告、召回、排序、重排、CTR/CVR 预估、排序指标和 A/B 测试等业务算法内容。

## 后续维护规则

1. 新笔记优先放入最具体的二级目录，不直接放根目录。
2. 如果一篇笔记横跨多个领域，按主要用途归类，必要时在 README 的重点入口处加链接。
3. 系统性复习材料放 `09_面试体系/`；Agent 相关概念、工具链和实践方法放 `10_Agent/`。
4. 工程实践类内容优先放到实际使用场景对应目录，例如推理部署放 `05_推理部署与系统/`，Python API 放 `07_Python与工程/`。
