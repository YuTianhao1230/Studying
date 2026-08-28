# KV_Cache与Prefill_Decode

## 知识点解析

### 概述

KV Cache 是大模型自回归生成时缓存历史 token 的 Key 和 Value；Prefill 是处理输入上下文，Decode 是逐 token 生成输出。

### 自回归生成是什么

GPT 类模型一次生成一个 token：

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

Transformer attention 中每层都会计算 Q、K、V。

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
- hidden size。
- attention head 数。
- 数据类型，如 FP16/BF16。

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

### KV cache 是什么？

回答思路：先给一句话定义，再说明它解决什么问题，最后补一个工程例子或常见风险。

回答模板：

KV cache 是 KV Cache与Prefill Decode 里的核心概念，主要解决系统在能力、效率、稳定性或可控性上的问题。面试中要说明它的定义、适用场景、限制，以及在真实工程中如何验证它有效。

### Prefill 和 decode 有什么区别？

回答思路：先分别定义两个概念，再从目标、机制、适用场景和工程取舍四个角度对比。

回答模板：

我会先分别定义相关概念，再从目标、输入数据、核心机制、工程成本和适用场景对比。围绕 KV_Cache与Prefill_Decode，关键是说明它解决的是哪类问题，以及相比相近方案的取舍：有的方案更简单稳定，有的方案探索能力更强但成本更高。最后要落到工程选择标准，而不是只列名词。

### 为什么首 token 延迟和后续 token 延迟要分开看？

回答思路：先指出背后的核心约束，再解释收益，最后补充如果不这样做会带来的风险。

回答模板：

因为 首 token 延迟和后续 token 延迟要分开看 背后有明确工程约束：如果不处理，容易带来质量不稳定、成本上升、错误难排查或安全风险。在 KV Cache与Prefill Decode 场景里，这个问题的关键是说明它如何提升系统可控性、可验证性和线上稳定性。

### 长上下文为什么显存压力大？

回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。

回答模板：

KV_Cache与Prefill_Decode 的核心含义是：KV Cache 是大模型自回归生成时缓存历史 token 的 Key 和 Value；Prefill 是处理输入上下文，Decode 是逐 token 生成输出。 面试回答时我会先给定义，再说明它解决的背景问题，接着讲核心机制和适用场景，最后补充限制与工程风险。这样回答比只背一句概念更完整。

### 如何优化 KV cache 显存占用？

回答思路：按“目标定义 -> 关键步骤 -> 风险控制 -> 指标验证”的顺序回答，避免只讲抽象原则。

回答模板：

我会把 优化 KV cache 显存占用 拆成几个步骤：先明确目标和输入输出，再设计关键模块或策略，然后加入失败处理和权限/质量约束，最后用离线指标、线上指标或回归测试验证效果。在 KV Cache与Prefill Decode 场景里，重点是能落到真实工程流程。
