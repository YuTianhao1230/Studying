# Transformer

## 知识点解析

### 概述

Transformer 是以自注意力为核心的序列建模架构，原始版本采用 Encoder-Decoder 结构，奠定了 BERT、GPT、T5、ViT、CLIP 和现代 LLM/VLM 的基础。

### 解决的问题

Transformer 主要解决 RNN/LSTM 的三个瓶颈：

- 串行计算：RNN 必须按时间步递推，训练难并行。
- 长距离依赖：长序列中信息要跨很多步传播，容易衰减。
- 表示容量：固定隐状态难同时表达全局依赖和局部关系。

Transformer 用 self-attention 让任意两个 token 一跳交互，并把序列建模变成矩阵计算，更适合 GPU/TPU 并行。

### 原始模型规格

论文中的 Base 配置：

| 组件 | 配置 |
| --- | --- |
| Encoder 层数 | 6 |
| Decoder 层数 | 6 |
| hidden size / `d_model` | 512 |
| attention heads | 8 |
| FFN hidden size / `d_ff` | 2048 |
| dropout | 0.1 |
| position encoding | sinusoidal positional encoding |

Big 配置通常为 `d_model=1024`、`d_ff=4096`、`heads=16`，层数仍为 6 encoder + 6 decoder。

### 完整架构

```text
source tokens
  -> token embedding + positional encoding
  -> Encoder block x 6
       -> multi-head self-attention
       -> add & norm
       -> position-wise FFN
       -> add & norm
  -> encoder memory

target tokens
  -> shifted token embedding + positional encoding
  -> Decoder block x 6
       -> masked multi-head self-attention
       -> add & norm
       -> encoder-decoder cross-attention
       -> add & norm
       -> position-wise FFN
       -> add & norm
  -> linear projection
  -> softmax over vocabulary
```

### 核心组件

Self-Attention：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

- `Q`：当前位置想查询什么信息。
- `K`：每个位置提供什么索引。
- `V`：每个位置真正传递的内容。
- `sqrt(d_k)`：防止点积过大导致 softmax 饱和。

Multi-Head Attention 把 hidden states 投影到多个子空间，每个 head 学不同关系，最后 concat 再线性映射。

FFN 是逐 token 的两层 MLP：

```text
FFN(x) = max(0, xW1 + b1)W2 + b2
```

### 做了什么改变

相比 RNN/CNN 序列模型，Transformer 的核心变化是：

1. 不再按时间递归，用 attention 直接建立全局依赖。
2. 不依赖卷积窗口，通过位置编码注入顺序信息。
3. Encoder 和 Decoder 都由统一 block 堆叠，扩展性强。
4. 训练阶段可并行处理整个序列。

### 后续影响

- BERT：只使用 Encoder，做双向理解。
- GPT：只使用 Decoder 的 masked self-attention，做自回归生成。
- T5：保留 Encoder-Decoder，统一 text-to-text。
- ViT：把图像 patch 当 token 输入 Transformer Encoder。
- LLM：在 decoder block 上加入 RoPE、RMSNorm、SwiGLU、GQA/MoE 等改造。

### 常见考法与解题方法

| 考法 | 怎么考 | 怎么解 |
| --- | --- | --- |
| 架构题 | Transformer 每层包含什么 | 分 Encoder block 和 Decoder block 回答 |
| 公式题 | Attention 公式为什么除以 `sqrt(d_k)` | 点积方差随维度增大，缩放防止 softmax 饱和 |
| 对比题 | Transformer 相比 RNN 优势 | 并行、长依赖、全局交互 |
| 变体题 | BERT/GPT/T5 与 Transformer 关系 | Encoder-only、Decoder-only、Encoder-Decoder |

### 易错点

- 把 Transformer 等同于 GPT；GPT 只是 Transformer decoder-only 路线。
- 只说 self-attention，不讲 positional encoding，否则模型不知道顺序。
- 忽略 decoder 里有 masked self-attention 和 cross-attention 两类 attention。
- 把多头注意力理解成多个模型投票；它本质是多个表示子空间的并行注意力。

## 面试应对

### Transformer 的完整架构是什么？

回答思路：先讲 Encoder-Decoder，再讲每个 block 的子层。

回答模板：

原始 Transformer 是 Encoder-Decoder 架构。Encoder 输入源序列，每层包含 multi-head self-attention 和 position-wise FFN，并配合残差连接和 LayerNorm；Decoder 输入右移后的目标序列，每层先做 masked self-attention，再对 encoder memory 做 cross-attention，最后经过 FFN。Base 版本是 6 层 encoder、6 层 decoder，`d_model=512`，8 个 attention heads，FFN hidden size 是 2048。它的核心贡献是用自注意力替代 RNN 递归，使序列全局交互和并行训练成为可能。
