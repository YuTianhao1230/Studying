# DeepSeek 架构

## 知识点解析

### 概述

DeepSeek 是以高性价比大模型、MoE、注意力显存优化和推理后训练为主线的模型系列。

面试里讲 DeepSeek，建议按版本区分：

```text
DeepSeek-V2/V3：
DeepSeekMoE + MLA + MTP
  -> 在较低激活计算和 KV Cache 成本下扩大模型容量

DeepSeek-R1：
V3 基座
  + SFT 冷启动
  + GRPO / RLVR 推理强化
  + 蒸馏到更小模型
```

不能把 DeepSeek 的所有版本都简单说成“MoE 模型”，因为 DeepSeek 还有 dense、Coder、Math、VL 和不同后训练形态。

### DeepSeek-V2/V3 的总体架构

```text
token ids
  -> tokenizer
  -> embedding
  -> decoder blocks
       -> RMSNorm
       -> MLA
       -> residual
       -> RMSNorm
       -> DeepSeekMoE FFN
       -> residual
  -> LM head
  -> next-token logits
```

每层主要由两部分组成：

- **MLA（Multi-head Latent Attention）**：压缩 K/V 表示，降低 KV Cache。
- **DeepSeekMoE**：每个 token 只路由到少数专家，扩大总参数容量但控制激活计算。

### MLA 解决什么问题

传统 MHA/GQA 直接缓存多头 K/V，长上下文和高并发时 KV Cache 很容易成为显存瓶颈。MLA 的思路是：

```text
K/V hidden states
  -> 低秩压缩到 latent vector
  -> 缓存 latent representation
  -> 注意力计算时恢复需要的表示
```

MLA 还需要处理 RoPE 和内容表示的兼容问题，常见做法是把内容部分和位置部分解耦：

```text
content latent：负责压缩内容信息
decoupled RoPE branch：保留位置相关信息
```

所以 MLA 和 GQA 的差别是：

- GQA：减少 K/V head 数量。
- MLA：压缩 K/V 表示维度，并缓存 latent。

两者都减少 KV Cache，但 MLA 的结构和实现更复杂。

### DeepSeekMoE 解决什么问题

稠密模型每个 token 都经过全部 FFN 参数，模型变大后计算成本同步上升。MoE 把 FFN 替换为多个专家：

```text
token hidden state
  -> router
  -> select top-k experts
  -> selected experts compute
  -> weighted sum
```

DeepSeekMoE 的重点：

- 细粒度专家：把专家拆得更细，让路由组合更灵活。
- 共享专家：保留一部分 shared experts 处理所有 token 的通用知识。
- 稀疏激活：每个 token 只激活少数 routed experts。
- 负载均衡：避免少数专家过载、其他专家闲置。

因此要区分：

```text
总参数量：所有专家参数加起来
激活参数量：一个 token 实际使用的参数
```

服务成本更接近激活参数量，但显存、通信和权重加载仍受总参数量影响。

### MTP：Multi-Token Prediction

DeepSeek-V3 还使用 Multi-Token Prediction（MTP）作为训练和推理相关的增强方向。它不只训练模型预测下一个 token，还让模型学习预测后续多个 token：

```text
当前 hidden state
  -> 预测 token t+1
  -> 预测 token t+2
  -> ...
```

MTP 可以提供更丰富的训练信号，也可以和 speculative decoding 等推理加速思路结合。面试时应把它和 MoE、MLA 分开讲：

- MoE 主要解决激活计算和模型容量。
- MLA 主要解决 KV Cache。
- MTP 主要增强多 token 预测和训练/推理效率。

### DeepSeek-R1 的后训练路线

DeepSeek-R1 的核心贡献更多在 reasoning post-training，而不是重新发明一套 decoder block：

```text
DeepSeek-V3 base
  -> reasoning SFT / cold start
  -> GRPO / RLVR
  -> rejection sampling / additional SFT
  -> general preference and capability alignment
  -> distillation to smaller dense models
```

关键点：

- **GRPO**：同一个 prompt 采样多个答案，用组内相对奖励估计优势，省掉 PPO 的独立 Critic。
- **RLVR**：数学、代码等任务用可验证结果直接给 reward。
- **蒸馏**：把大推理模型的答案和推理轨迹蒸馏给更小模型，降低部署成本。
- **长度控制**：推理模型可能通过无限延长输出获得表面 reward，需要处理长度偏置、重复和 reward hacking。

### DeepSeek、Qwen、Llama 怎么区分

| 系列 | 主要优势 | 架构/训练重点 |
| --- | --- | --- |
| DeepSeek | 高性价比、推理和代码 | DeepSeekMoE、MLA、MTP、GRPO/RLVR |
| Qwen | 中文/多语言、多模态和 Agent 生态 | Qwen3、Qwen3-VL、Thinking、Embedding/Reranker |
| Llama | 开放权重生态和通用基座 | Dense decoder、GQA、长上下文、工具调用 |
| GPT | 闭源通用能力和产品化体验 | 具体架构不公开，强调多模态、推理和工具生态 |

### 工程使用注意点

- MoE 的总参数量不等于单 token 激活参数量，容量规划要同时看两者。
- MLA 需要推理框架原生支持，不能把 DeepSeek 当作普通 GQA 模型加载。
- MoE 多卡服务要关注 expert parallel、通信和负载均衡。
- R1 类 reasoning 模型要控制 thinking budget、最大输出和重复。
- Distill 模型不是简单裁剪，需单独检查 tokenizer、chat template、推理格式和能力回归。

## 面试应对

### DeepSeek-V2/V3 的架构重点是什么？

回答思路：用“MoE 控制激活计算，MLA 压缩 KV Cache，MTP 增强多 token 预测”三点回答。

回答模板：

DeepSeek-V2/V3 的核心不是单个 decoder block，而是三条效率路线。第一是 DeepSeekMoE，通过 router 让每个 token 只激活少数 routed experts，并保留 shared experts，在控制计算量的同时扩大总参数容量；第二是 MLA，把 K/V 表示压缩成 latent 并缓存，降低长上下文推理的 KV Cache 和显存带宽压力；第三是 MTP，让模型学习预测多个后续 token，为训练信号和推理加速提供支持。三者分别对应模型容量、KV Cache 和多 token 效率。

### MLA 和 GQA 有什么区别？

回答思路：区分“减少 K/V head 数量”和“低秩压缩 K/V 表示”。

回答模板：

GQA 是让多个 Query head 共享较少的 Key/Value head，通过减少 K/V head 数量来降低 KV Cache；MLA 则进一步把 K/V 信息压缩成低维 latent representation，推理时主要缓存 latent，需要计算时再恢复使用。GQA 的结构更直观、兼容性更好；MLA 的 KV Cache 压缩潜力更大，但位置编码解耦、权重结构和推理实现更复杂。两者的共同目标都是降低长上下文 decode 阶段的显存和带宽成本。

### DeepSeekMoE 为什么能降低计算成本？

回答思路：区分总参数和激活参数，并补充 router、top-k、shared experts 和负载均衡。

回答模板：

DeepSeekMoE 把 FFN 拆成多个专家，每个 token 经过 router 后只选择 top-k 个 routed experts，再加上 shared experts，而不是让所有专家都计算。因此总参数量可以很大，但单个 token 的激活参数量和计算量相对可控。工程上还要解决专家负载均衡和多卡通信问题，否则少数专家过载会拖慢整个 step，稀疏计算带来的收益也可能被通信开销抵消。

### DeepSeek-R1 的推理能力是怎么训练出来的？

回答思路：按 base、冷启动 SFT、GRPO/RLVR、筛选再 SFT、蒸馏串起完整链路。

回答模板：

DeepSeek-R1 的推理能力不是只靠 prompt 产生的，而是在 base model 上做后训练得到的。先用高质量推理数据做冷启动 SFT，让模型学会基本的推理格式和解题过程；再用 GRPO 或其他 RLVR 方法，在数学、代码等有可验证结果的任务上直接优化奖励；训练后通过 rejection sampling 和额外 SFT 清洗错误、重复和格式不稳定的答案，最后还可以把大推理模型蒸馏到更小模型。核心是可验证奖励、相对策略优化和数据筛选共同作用。
