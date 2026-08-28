# Speculative Decoding

## 一句话解释

Speculative Decoding 是一种大模型推理加速方法：用小模型先快速草拟多个 token，再由大模型并行验证，从而减少大模型逐 token 解码次数。

## 为什么需要

自回归生成的瓶颈在 decode 阶段：

```text
生成第 1 个 token -> 生成第 2 个 token -> 生成第 3 个 token -> ...
```

每个 token 都要依赖前一个 token，难以完全并行。

Speculative Decoding 的思路是：

```text
小模型 draft 多个 token
  -> 大模型一次性验证
  -> 接受正确 token
  -> 拒绝后回退
```

## 基本流程

1. Draft Model 根据当前上下文生成若干候选 token。
2. Target Model 对这些候选 token 做验证。
3. 如果候选 token 与 Target Model 分布一致或满足采样规则，则接受。
4. 如果某个位置不通过，则从该位置回退，用 Target Model 生成。
5. 重复直到完成。

## 为什么能加速

小模型生成便宜，大模型验证多个 token 可以并行。

当小模型预测和大模型足够接近时，一次验证能接受多个 token，减少大模型调用次数。

## 适合场景

- 解码阶段瓶颈明显。
- 小模型和大模型输出分布接近。
- 长文本生成。
- 对吞吐和延迟敏感的在线服务。

不适合：

- 小模型质量太差，接受率低。
- 生成很短，调度开销抵消收益。
- 系统已经被其他瓶颈限制，例如网络或队列。

## 关键指标

- Acceptance Rate：草稿 token 接受率。
- Tokens per Second。
- p50/p95/p99 latency。
- Draft Model 成本。
- Target Model 验证开销。

## 和其他推理优化的关系

- KV Cache：减少重复 Attention 计算。
- Continuous Batching：提高吞吐。
- Quantization：降低显存和计算成本。
- Speculative Decoding：减少大模型解码步数。

这些方法可以组合使用，但系统复杂度会增加。

## 常见误区

- 以为 Speculative Decoding 一定提速。
- 忽略小模型加载和调度成本。
- 只看平均延迟，不看 p99。
- 没有监控接受率。
- 小模型和大模型 tokenizer 或输出分布不匹配。

## 面试可能怎么问

1. Speculative Decoding 的基本原理是什么？
2. 为什么小模型草稿能加速大模型推理？
3. 接受率低时会发生什么？
4. 它和 KV Cache、Batching、量化分别解决什么问题？
5. 如何在线上判断这个优化是否值得开启？
