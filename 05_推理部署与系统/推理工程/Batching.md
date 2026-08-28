# Batching

## 知识点解析

### 概述

Batching 是把多个请求合并在一起执行，从而提升 GPU 利用率和整体吞吐。

### 为什么需要 Batching

GPU 擅长并行计算。如果一次只处理一个很小的请求，GPU 可能吃不满。

Batching 把多个请求合并：

```text
request 1
request 2
request 3
request 4
  -> 一个 batch
  -> GPU 一起计算
```

这样可以提升吞吐。

### Static Batching

固定 batch size，凑够一批再执行。

优点：

- 实现简单。
- 适合离线批量任务。

缺点：

- 在线请求可能需要等待凑 batch。
- 请求长度不同会 padding 浪费。

### Dynamic Batching

在一个很短的时间窗口内收集请求，动态组成 batch。

例如等待 5ms，把这 5ms 内到达的请求组成一个 batch。

优点：

- 在线服务中更灵活。
- 能在延迟和吞吐之间折中。

缺点：

- 等待窗口过大会增加延迟。
- 过小则 batch 不够大，吞吐不足。

### Continuous Batching

大模型生成请求长度差异很大。Continuous batching 允许请求动态进入和退出正在执行的 batch。

```text
batch 中 request A 结束
  -> 立即加入 request D
  -> 不必等待整个 batch 全部结束
```

优点：

- 减少 GPU 空转。
- 提高 decode 阶段吞吐。
- 更适合 LLM 在线服务。

### Padding 浪费

同一个 batch 中，序列长度通常要对齐到最长样本。

例如：

```text
样本 A: 100 token
样本 B: 1000 token
```

如果放在同一 batch，短样本可能要 padding 到 1000，造成计算浪费。

优化方法：

- 按长度分桶。
- 动态 batch。
- packing。
- 使用更高效的 attention mask 和调度策略。

## 面试应对

### batching 为什么能提升吞吐？

回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。

回答模板：

batching 把多个请求合并成一次 GPU 计算，提高矩阵计算规模和 GPU 利用率，减少单请求执行造成的资源空转。

### dynamic batching 和 continuous batching 有什么区别？

回答思路：先分别定义两个概念，再从目标、机制、适用场景和工程取舍四个角度对比。

回答模板：

我会先分别定义相关概念，再从目标、输入数据、核心机制、工程成本和适用场景对比。围绕 Batching，关键是说明它解决的是哪类问题，以及相比相近方案的取舍：有的方案更简单稳定，有的方案探索能力更强但成本更高。最后要落到工程选择标准，而不是只列名词。

### batching 会不会增加延迟？

回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。

回答模板：

会。为了凑 batch 可能引入排队等待，所以 batching 是吞吐和延迟的权衡。线上通常用最大等待时间和 batch token 上限控制。

### LLM 推理中为什么 batching 比普通模型更复杂？

回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。

回答模板：

LLM 请求输入长度、输出长度和结束时间都不同，还要维护各自 KV Cache。调度器需要动态加入、退出和重排请求。

### 如何在延迟和吞吐之间做平衡？

回答思路：按“目标定义 -> 关键步骤 -> 风险控制 -> 指标验证”的顺序回答，避免只讲抽象原则。

回答模板：

根据 SLA 设置最大 batch size、max waiting time、token budget 和优先级队列。低延迟业务牺牲部分吞吐，高吞吐离线任务可增大 batch。
