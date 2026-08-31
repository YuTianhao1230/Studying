# CUDA_Graph

## 知识点解析

### 概述

CUDA Graph 是 NVIDIA CUDA 提供的执行图机制，可以把一串 GPU 操作提前捕获成图，后续重复执行时减少 CPU 调度和 kernel launch 开销。

### 为什么大模型推理会用到

大模型推理包含大量重复的 GPU kernel 调用。

普通执行方式：

```text
CPU 发起 kernel 1
CPU 发起 kernel 2
CPU 发起 kernel 3
...
```

每次 kernel launch 都有 CPU 调度开销。对于 decode 阶段，单步计算可能很碎，这个开销会变得明显。

CUDA Graph 的思路是：

```text
先捕获一段固定形状的 GPU 执行流程
后续直接 replay 整张图
```

这样可以减少 launch overhead，提高推理稳定性。

### 适合场景

- 计算图结构稳定。
- 输入 shape 相对固定。
- 同一批次配置反复执行。
- decode 阶段大量重复操作。
- 对 p99 latency 敏感。

### 不适合场景

- shape 频繁变化。
- 控制流高度动态。
- batch size 和 sequence length 变化太大。
- 每次请求的执行路径都不同。

因此线上推理系统常常需要配合 padding、bucket、固定 batch shape 等策略使用 CUDA Graph。

### 和其他推理优化的关系

- KV Cache：减少重复 Attention 计算。
- Continuous Batching：提升吞吐。
- Speculative Decoding：减少大模型 decode 步数。
- CUDA Graph：减少 CPU launch 调度开销。
- TensorRT-LLM：常结合底层 kernel 优化和 graph 机制做高性能推理。

### 关键收益

- 降低 CPU overhead。
- 降低延迟抖动。
- 提升小 batch / decode 场景效率。
- 改善 p99 latency。

### 常见风险

- shape 不稳定导致 graph 难复用。
- 捕获阶段复杂。
- 内存地址和执行路径需要稳定。
- 动态控制流支持有限。
- 和动态 batching 组合时需要额外调度设计。

## 面试应对

### CUDA_Graph 是什么？

回答思路：一句话定位它是底层执行机制，核心是把一串 GPU 操作捕获成图后 replay，省掉每次 kernel launch 的 CPU 调度开销，而不改变模型能力。

回答模板：

CUDA Graph 是 NVIDIA CUDA 提供的执行图机制，可以把一串 GPU 操作提前捕获成图，后续重复执行时减少 CPU 调度和 kernel launch 开销。 它关注的核心不是提升模型本身能力，而是让已有模型在服务中更高吞吐、更低延迟、更稳定地运行。

### CUDA_Graph 解决什么问题？

回答思路：聚焦 decode 阶段单步计算碎、kernel launch 频繁导致的 CPU 调度开销和延迟抖动，说明它针对的是 launch overhead 而非显存或 batching。

回答模板：

大模型推理包含大量重复的 GPU kernel 调用。普通执行方式： 每次 kernel launch 都有 CPU 调度开销。对于 decode 阶段，单步计算可能很碎，这个开销会变得明显。 在工程上通常要结合 p50/p95/p99 延迟、tokens/s、显存峰值、并发数和失败率来判断它是否有效。

### CUDA_Graph 的核心机制是什么？

回答思路：讲清“先捕获固定形状执行流、后续整图 replay”这一机制，说明它靠减少 CPU 参与来降低 launch 开销和延迟抖动，尤其利于小 batch/decode。

回答模板：

CUDA Graph 是 NVIDIA CUDA 提供的执行图机制，可以把一串 GPU 操作提前捕获成图，后续重复执行时减少 CPU 调度和 kernel launch 开销。 这类机制的价值在于减少无效计算、降低显存碎片、提升 GPU 利用率或稳定服务调度。

### CUDA_Graph 有哪些限制？

回答思路：强调它要求 shape、内存地址和执行路径稳定，动态控制流支持有限，因此线上要配合 padding/分桶固定 batch shape，并和动态 batching 做调度协调。

回答模板：

shape 不稳定导致 graph 难复用。内存地址和执行路径需要稳定。动态控制流支持有限。和动态 batching 组合时需要额外调度设计。 如果线上效果异常，需要检查请求长度分布、batch 配置、KV cache、显存利用率、并发策略和模型并行配置。
