# Batching

## 一句话解释

Batching 是把多个请求合并在一起执行，从而提升 GPU 利用率和整体吞吐。

## 为什么需要 Batching

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

## Static Batching

固定 batch size，凑够一批再执行。

优点：

- 实现简单。
- 适合离线批量任务。

缺点：

- 在线请求可能需要等待凑 batch。
- 请求长度不同会 padding 浪费。

## Dynamic Batching

在一个很短的时间窗口内收集请求，动态组成 batch。

例如等待 5ms，把这 5ms 内到达的请求组成一个 batch。

优点：

- 在线服务中更灵活。
- 能在延迟和吞吐之间折中。

缺点：

- 等待窗口过大会增加延迟。
- 过小则 batch 不够大，吞吐不足。

## Continuous Batching

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

## Padding 浪费

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

## 面试可能怎么问

1. batching 为什么能提升吞吐？
2. dynamic batching 和 continuous batching 有什么区别？
3. batching 会不会增加延迟？
4. LLM 推理中为什么 batching 比普通模型更复杂？
5. 如何在延迟和吞吐之间做平衡？

