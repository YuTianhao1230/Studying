# Llama 架构

## 知识点解析

### 概述

Llama 是 Meta 推出的开放权重大语言模型系列，核心价值是提供高质量、可研究、可微调和可私有化部署的基础模型。

面试里讲 Llama，建议抓住三条主线：

```text
Llama 1/2/3：
Decoder-only Transformer
  + RoPE
  + RMSNorm / Pre-Norm
  + SwiGLU
  + GQA
  + 大规模预训练和指令对齐

Llama 3.1/3.2/3.3：
长上下文、工具调用、端侧小模型、视觉模型

Llama 4：
原生多模态和 MoE，具体结构不能直接套用 Llama 3
```

### Llama 的基本架构

Llama 2/3 主线是 Decoder-only Transformer：

```text
token ids
  -> tokenizer
  -> token embedding
  -> N 层 decoder blocks
       -> RMSNorm
       -> causal self-attention + RoPE
       -> residual
       -> RMSNorm
       -> SwiGLU FFN
       -> residual
  -> final RMSNorm
  -> LM head
  -> next-token logits
```

和 GPT 一样，Llama 使用 causal mask，每个位置只能关注自己和之前的 token。

### 关键组件

| 组件 | Llama 里的作用 |
| --- | --- |
| Decoder-only | 统一对话、代码、推理和工具调用的生成目标 |
| RoPE | 注入相对位置信息，支持长上下文扩展 |
| RMSNorm / Pre-Norm | 稳定深层 Transformer 的训练 |
| SwiGLU | 提供带门控的非线性 FFN |
| GQA | 让多个 Query heads 共享较少的 KV heads，降低 KV Cache |
| BPE tokenizer | 处理多语言、代码和符号，影响 token 数和上下文成本 |

Llama 3 开始，GQA 不再只是大模型的特殊配置，常见尺寸也采用了 GQA。面试时不要说“所有 Llama 版本都完全相同”，应先说明具体版本。

### 以 Llama 3 8B 为例

Llama 3 8B 常见配置可以这样记：

| 配置 | 典型值 |
| --- | --- |
| Decoder layers | 32 |
| Hidden size | 4096 |
| Attention heads | 32 Query heads |
| KV heads | 8，使用 GQA |
| Head dimension | 128 |
| Position encoding | RoPE |
| FFN | SwiGLU |

Llama 3.1 的上下文长度、tokenizer 和后训练配置又有调整，所以这些数字只用于解释典型结构，不能代表整个 Llama 家族。

### Llama 的训练与模型形态

```text
高质量预训练数据
  -> base model
  -> instruction tuning
  -> preference / safety alignment
  -> chat / instruct model
```

常见模型形态：

- **Base**：只完成预训练，适合继续训练和研究。
- **Instruct/Chat**：经过指令微调和安全对齐，适合直接对话和工具调用。
- **Code Llama**：在代码、填空和代码指令上继续训练。
- **Vision**：在语言模型之外接入视觉编码器，支持图像理解。

### Llama 为什么影响大

Llama 的影响不只是模型本身效果，而是降低了整个开源生态的门槛：

- 研究者可以下载权重做消融和微调。
- 工程团队可以用 LoRA/QLoRA 做业务适配。
- vLLM、TensorRT-LLM、llama.cpp、量化工具广泛支持。
- 大量模型在 Llama 数据格式、权重格式和指令模板上形成生态。

### Llama、Qwen、DeepSeek 怎么区分

| 系列 | 主要特点 | 典型架构重点 |
| --- | --- | --- |
| Llama | 开放权重生态和通用基座 | Dense decoder、GQA、长上下文、工具调用 |
| Qwen | 中文/多语言、多模态、代码和 Agent 生态完整 | Dense/MoE、Qwen-VL、Thinking、Embedding/Reranker |
| DeepSeek | 性价比、MoE、长上下文和推理模型 | DeepSeekMoE、MLA、MTP、GRPO/RLVR |
| GPT | 闭源通用旗舰和产品生态 | 具体结构不公开，强调通用、多模态和工具体验 |

### 工程使用注意点

- 明确区分 Llama 2、Llama 3、Llama 3.1、Llama 3.2、Llama 4。
- Base、Instruct、Code、Vision 不是同一个 checkpoint。
- tokenizer、chat template、special tokens 必须和模型版本匹配。
- GQA 会影响 KV Cache 显存，不能只看参数量估算服务容量。
- 量化、LoRA adapter、上下文长度和推理框架要做回归验证。

## 面试应对

### Llama 的架构是什么？

回答思路：按 Decoder-only、RoPE、RMSNorm、SwiGLU、GQA 和预训练/指令对齐回答。

回答模板：

Llama 2/3 主线是 decoder-only Transformer，使用 causal self-attention 做自回归 next-token prediction。每个 block 通常是 Pre-RMSNorm、带 RoPE 的 self-attention、残差，再接 RMSNorm 和 SwiGLU FFN。Llama 3 系列常用 GQA，让多个 Query head 共享较少的 Key/Value head，从而降低 KV Cache 和推理带宽。训练上先得到 base model，再通过 instruction tuning 和安全/偏好对齐得到 instruct 或 chat 模型。Llama 4 已经引入原生多模态和 MoE，不能把 Llama 3 的 dense 结构直接套过去。

### Llama 为什么采用 GQA？

回答思路：从 KV Cache、显存带宽和长上下文服务解释。

回答模板：

GQA 让多个 Query head 共享较少的 Key/Value head。在自回归 decode 阶段，历史 K/V 会缓存下来并反复读取，所以 KV head 数越多，KV Cache 和显存带宽压力越大。Llama 采用 GQA 后，在尽量保持多头注意力表达能力的同时减少 KV Cache，长上下文和高并发服务更容易达到较好的吞吐。

### Llama Base 和 Instruct 有什么区别？

回答思路：区分预训练目标相同但后训练目标不同，落到使用场景。

回答模板：

Llama Base 主要通过 next-token prediction 学习语言、代码和世界知识，它更适合作为继续预训练或 SFT 的基础。Llama Instruct/Chat 在 base model 上继续做指令微调、偏好对齐和安全训练，学会理解用户意图、遵循输出格式和拒绝危险请求。实际使用中，如果需要直接对话或工具调用，我会选 Instruct；如果要做领域继续训练或自定义对齐，通常从 Base 开始。
