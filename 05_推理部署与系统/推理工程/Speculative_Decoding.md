# Speculative_Decoding

## 知识点解析

### 概述

Speculative Decoding 是一种大模型推理加速方法：用小模型先快速草拟多个 token，再由大模型并行验证，从而减少大模型逐 token 解码次数。

### 为什么需要

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

### 基本流程

1. Draft Model 根据当前上下文生成若干候选 token。
2. Target Model 对这些候选 token 做验证。
3. 如果候选 token 与 Target Model 分布一致或满足采样规则，则接受。
4. 如果某个位置不通过，则从该位置回退，用 Target Model 生成。
5. 重复直到完成。

### 为什么能加速

小模型生成便宜，大模型验证多个 token 可以并行。

当小模型预测和大模型足够接近时，一次验证能接受多个 token，减少大模型调用次数。

### 适合场景

- 解码阶段瓶颈明显。
- 小模型和大模型输出分布接近。
- 长文本生成。
- 对吞吐和延迟敏感的在线服务。

不适合：

- 小模型质量太差，接受率低。
- 生成很短，调度开销抵消收益。
- 系统已经被其他瓶颈限制，例如网络或队列。

### 关键指标

- Acceptance Rate：草稿 token 接受率。
- Tokens per Second。
- p50/p95/p99 latency。
- Draft Model 成本。
- Target Model 验证开销。

### 和其他推理优化的关系

- KV Cache：减少重复 Attention 计算。
- Continuous Batching：提高吞吐。
- Quantization：降低显存和计算成本。
- Speculative Decoding：减少大模型解码步数。

这些方法可以组合使用，但系统复杂度会增加。

### 常见误区

- 以为 Speculative Decoding 一定提速。
- 忽略小模型加载和调度成本。
- 只看平均延迟，不看 p99。
- 没有监控接受率。
- 小模型和大模型 tokenizer 或输出分布不匹配。

## 面试应对

### Speculative_Decoding 是什么？

回答思路：先定位它属于推理框架、推理优化还是底层执行机制。

回答模板：

Speculative Decoding 是一种大模型推理加速方法：用小模型先快速草拟多个 token，再由大模型并行验证，从而减少大模型逐 token 解码次数。 它关注的核心不是提升模型本身能力，而是让已有模型在服务中更高吞吐、更低延迟、更稳定地运行。

### Speculative_Decoding 解决什么问题？

回答思路：从 KV cache、batching、显存、并发、p99 延迟等推理瓶颈回答。

回答模板：

自回归生成的瓶颈在 decode 阶段： 每个 token 都要依赖前一个 token，难以完全并行。Speculative Decoding 的思路是：小模型生成便宜，大模型验证多个 token 可以并行。 在工程上通常要结合 p50/p95/p99 延迟、tokens/s、显存峰值、并发数和失败率来判断它是否有效。

### Speculative_Decoding 的核心机制是什么？

回答思路：讲清楚它改变了哪部分推理流程，以及为什么能改善吞吐或显存。

回答模板：

Draft Model 根据当前上下文生成若干候选 token。Target Model 对这些候选 token 做验证。如果候选 token 与 Target Model 分布一致或满足采样规则，则接受。如果某个位置不通过，则从该位置回退，用 Target Model 生成。 这类机制的价值在于减少无效计算、降低显存碎片、提升 GPU 利用率或稳定服务调度。

### Speculative_Decoding 有哪些限制？

回答思路：说明适用边界、参数配置风险和线上排查重点。

回答模板：

Speculative Decoding 是一种大模型推理加速方法：用小模型先快速草拟多个 token，再由大模型并行验证，从而减少大模型逐 token 解码次数。 如果线上效果异常，需要检查请求长度分布、batch 配置、KV cache、显存利用率、并发策略和模型并行配置。
