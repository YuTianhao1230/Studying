# KV_Cache与Prefill_Decode

## 知识点解析

### 概述

KV Cache 是大模型自回归生成时缓存历史 token 的 Key 和 Value；Prefill 是处理输入上下文，Decode 是逐 token 生成输出。

### 自回归生成是什么

[GPT](<../../02_大模型/模型细节/里程碑模型/GPT.md#gpt>) 类模型一次生成一个 token：

```text
输入: 今天天气
生成第 1 个 token: 很
生成第 2 个 token: 好
生成第 3 个 token: 。
```

每生成一个新 token，都要基于之前所有 token。

### Prefill 阶段

Prefill 处理用户输入 prompt。

例如输入有 1000 个 token，模型一次性计算这 1000 个 token 的隐藏状态，并生成它们对应的 KV cache。

特点：

- 输入越长，prefill 越慢。
- 并行度较高。
- 主要影响首 token 延迟 TTFT。

### Decode 阶段

Decode 从第一个输出 token 开始，每次生成一个 token。

特点：

- 每步只生成一个 token。
- 依赖历史 KV cache。
- 输出越长，decode 总耗时越高。
- 主要影响 TPOT 和整体延迟。

### KV Cache 是什么

[Transformer](<../../02_大模型/基础架构/Transformer.md#transformer>) attention 中每层都会计算 Q、K、V。

生成第 `t` 个 token 时，新 token 需要关注之前所有 token。如果每一步都重新计算所有历史 token 的 K/V，会非常浪费。

KV Cache 的做法是：

```text
历史 token 的 K/V 计算过后缓存起来。
下一步生成时，只计算新 token 的 Q/K/V，并复用历史 K/V。
```

### KV Cache 的收益

- 避免重复计算历史 token。
- 大幅提升 decode 速度。
- 是 LLM 高效推理的核心机制。

### KV Cache 的代价

KV cache 会占显存，并且随以下因素增长：

- batch size。
- 序列长度。
- 模型层数。
- `num_key_value_heads`。
- `head_dim`。
- 数据类型，如 FP16/BF16。

忽略分页和对齐开销时，容量近似为 `2 * batch_size * sequence_length * num_layers * num_key_value_heads * head_dim * bytes_per_element`，其中 2 代表 K 和 V。

直观理解：

```text
请求越多、上下文越长、模型越大，KV cache 越占显存。
```

### 常见问题

#### 为什么长上下文推理容易 OOM？

因为 KV cache 随上下文长度增长。输入很长或输出很长，都会让缓存变大。

#### 为什么限制 max_tokens 能降低风险？

`max_tokens` 限制最大输出长度，能限制 decode 步数和 KV cache 继续增长。

#### 为什么 PagedAttention 有用？

它把 KV cache 分块管理，减少显存碎片和浪费。

## 面试应对

### Prefill 和 Decode 的瓶颈为什么不同？

回答思路：从每步处理的 Token 数、并行度和主要资源约束比较。

回答模板：

Prefill 一次处理整段输入，可以在 Token 维度并行，通常计算量大，更容易表现为计算瓶颈，并主要影响 TTFT。Decode 每轮只生成一个新 Token，步骤之间串行，还要反复读取权重和历史 KV Cache，通常更受显存带宽、调度和并发影响，并主要决定 TPOT。两阶段的负载特征不同，所以优化和调度不能只看一个总延迟。

### KV Cache 为什么能加速自回归生成？

回答思路：解释历史 Token 的 K/V 不随新 Token 改变，因此可以缓存并复用。

回答模板：

在自回归 Decode 中，第 t 步仍需关注前 t-1 个 Token，但这些历史 Token 在各层算出的 Key 和 Value 已经固定。KV Cache 把它们保存下来，下一步只计算新 Token 的 Q、K、V，再让新 Query 与缓存的 Key、Value 做注意力。这样避免每一步重新计算整个历史序列，是用显存换计算；代价是缓存会随并发数和序列长度增长。

### 为什么长上下文和大 Batch 容易导致 KV Cache OOM？

回答思路：指出缓存规模受层数、KV Head、Head Dimension、精度、并发和总序列长度共同影响，再给出治理手段。

回答模板：

每个活跃请求都要在每一层保存历史 Token 的 K 和 V，因此缓存占用会随层数、KV Head 数、Head Dimension、元素字节数、并发请求数和上下文加输出长度近似线性增长。长上下文和大 Batch 会同时放大这些维度。工程上可以限制最大上下文与输出、使用 GQA/MQA 或低精度 KV、实施请求准入，并通过 PagedAttention 分页分配缓存以减少碎片。

### PagedAttention 解决了什么问题？

回答思路：区分“避免重复计算”和“改善缓存分配”：前者是 KV Cache，后者是分页管理。

回答模板：

普通连续分配需要为不确定长度的请求预留较大连续空间，容易产生内部浪费和外部碎片。PagedAttention 把 KV Cache 划分为固定大小的块，通过逻辑块到物理块的映射按需分配，使不同请求的缓存不必物理连续。它提高的是显存利用率和调度灵活性，并不会消除 KV Cache 随有效 Token 数增长这一事实。
