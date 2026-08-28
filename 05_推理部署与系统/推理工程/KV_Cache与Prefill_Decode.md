# KV Cache 与 Prefill / Decode

## 一句话解释

KV Cache 是大模型自回归生成时缓存历史 token 的 Key 和 Value；Prefill 是处理输入上下文，Decode 是逐 token 生成输出。

## 自回归生成是什么

GPT 类模型一次生成一个 token：

```text
输入: 今天天气
生成第 1 个 token: 很
生成第 2 个 token: 好
生成第 3 个 token: 。
```

每生成一个新 token，都要基于之前所有 token。

## Prefill 阶段

Prefill 处理用户输入 prompt。

例如输入有 1000 个 token，模型一次性计算这 1000 个 token 的隐藏状态，并生成它们对应的 KV cache。

特点：

- 输入越长，prefill 越慢。
- 并行度较高。
- 主要影响首 token 延迟 TTFT。

## Decode 阶段

Decode 从第一个输出 token 开始，每次生成一个 token。

特点：

- 每步只生成一个 token。
- 依赖历史 KV cache。
- 输出越长，decode 总耗时越高。
- 主要影响 TPOT 和整体延迟。

## KV Cache 是什么

Transformer attention 中每层都会计算 Q、K、V。

生成第 `t` 个 token 时，新 token 需要关注之前所有 token。如果每一步都重新计算所有历史 token 的 K/V，会非常浪费。

KV Cache 的做法是：

```text
历史 token 的 K/V 计算过后缓存起来。
下一步生成时，只计算新 token 的 Q/K/V，并复用历史 K/V。
```

## KV Cache 的收益

- 避免重复计算历史 token。
- 大幅提升 decode 速度。
- 是 LLM 高效推理的核心机制。

## KV Cache 的代价

KV cache 会占显存，并且随以下因素增长：

- batch size。
- 序列长度。
- 模型层数。
- hidden size。
- attention head 数。
- 数据类型，如 FP16/BF16。

直观理解：

```text
请求越多、上下文越长、模型越大，KV cache 越占显存。
```

## 常见问题

### 为什么长上下文推理容易 OOM？

因为 KV cache 随上下文长度增长。输入很长或输出很长，都会让缓存变大。

### 为什么限制 max_tokens 能降低风险？

`max_tokens` 限制最大输出长度，能限制 decode 步数和 KV cache 继续增长。

### 为什么 PagedAttention 有用？

它把 KV cache 分块管理，减少显存碎片和浪费。

## 面试可能怎么问

1. KV cache 是什么？
2. Prefill 和 decode 有什么区别？
3. 为什么首 token 延迟和后续 token 延迟要分开看？
4. 长上下文为什么显存压力大？
5. 如何优化 KV cache 显存占用？

