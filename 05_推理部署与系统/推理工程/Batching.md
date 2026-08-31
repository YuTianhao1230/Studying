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

回答思路：从 GPU 擅长并行、单请求吃不满算力这一点切入。

回答模板：

GPU 擅长大规模并行计算，如果一次只处理一个很小的请求，矩阵计算规模太小，GPU 算力吃不满、大量在空转。Batching 把多个请求合并成一个 batch 一起算，把小矩阵拼成大矩阵，显著提高 GPU 利用率，所以在吞吐上收益很大。代价是需要凑批，可能带来一点排队延迟。

### dynamic batching 和 continuous batching 有什么区别？

回答思路：讲清一个是批级别调度、一个是请求级别动态进出，落到 LLM 生成场景。

回答模板：

Dynamic batching 是在一个很短的时间窗口（比如 5ms）内收集到达的请求，动态凑成一个 batch 一起执行，粒度是"整批同进同出"，适合请求长度差不多的场景。Continuous batching 是针对 LLM 生成长度差异大的问题：它允许请求在 decode 过程中动态进出正在执行的 batch，某个请求生成完就立刻退出、把新请求补进来，不用等整批都结束。所以 continuous batching 能大幅减少 decode 阶段的 GPU 空转，是 vLLM 这类 LLM 推理引擎提升吞吐的关键。

### batching 会不会增加延迟？

回答思路：点明这是吞吐和延迟的权衡，并给出线上控制手段。

回答模板：

会。为了凑 batch 可能引入排队等待，所以 batching 本质是吞吐和延迟的权衡：等待窗口太大延迟高，太小又凑不够 batch、吞吐不足。线上通常用最大等待时间、最大 batch size 和 batch token 上限来控制，低延迟业务把窗口调小、宁可牺牲一点吞吐。

### LLM 推理中为什么 batching 比普通模型更复杂？

回答思路：从变长、KV Cache 和动态调度三点说。

回答模板：

因为 LLM 请求的输入长度、输出长度和结束时间都不一样，还要为每个请求各自维护 KV Cache。普通模型一次前向就出结果，batch 同进同出很简单；LLM 是自回归逐 token 生成，一个 batch 里有的请求早早生成完、有的还很长，如果同进同出会造成大量 padding 和空转。所以需要 continuous batching 这类调度器动态地加入、退出、重排请求，并管理好各自的 KV Cache 显存。

### 如何在延迟和吞吐之间做平衡？

回答思路：落到按 SLA 配置调度参数、区分在线离线业务。

回答模板：

我会根据业务 SLA 来配置调度参数：设置最大 batch size、最大等待时间、token budget 和优先级队列。对延迟敏感的在线业务，把等待窗口和 batch 调小、宁可牺牲部分吞吐来保 P99；对离线批量任务，则把 batch 尽量调大、追求吞吐。此外还可以按长度分桶、用 packing 减少 padding 浪费，让同一 batch 内长度更接近，进一步兼顾两端。
