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

### CUDA Graph 解决什么问题？

回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。

回答模板：

我会先分别定义相关概念，再从目标、输入数据、核心机制、工程成本和适用场景对比。围绕 CUDA_Graph，关键是说明它解决的是哪类问题，以及相比相近方案的取舍：有的方案更简单稳定，有的方案探索能力更强但成本更高。最后要落到工程选择标准，而不是只列名词。

### 为什么大模型 decode 阶段容易受 kernel launch overhead 影响？

回答思路：先指出背后的核心约束，再解释收益，最后补充如果不这样做会带来的风险。

回答模板：

因为 大模型 decode 阶段容易受 kernel launch overhead 影响 背后有明确工程约束：如果不处理，容易带来质量不稳定、成本上升、错误难排查或安全风险。在 CUDA Graph 场景里，这个问题的关键是说明它如何提升系统可控性、可验证性和线上稳定性。

### CUDA Graph 对输入 shape 有什么要求？

回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。

回答模板：

CUDA_Graph 的核心含义是：CUDA Graph 是 NVIDIA CUDA 提供的执行图机制，可以把一串 GPU 操作提前捕获成图，后续重复执行时减少 CPU 调度和 kernel launch 开销。 面试回答时我会先给定义，再说明它解决的背景问题，接着讲核心机制和适用场景，最后补充限制与工程风险。这样回答比只背一句概念更完整。

### CUDA Graph 和 continuous batching 是否冲突？

回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。

回答模板：

CUDA_Graph 的核心含义是：CUDA Graph 是 NVIDIA CUDA 提供的执行图机制，可以把一串 GPU 操作提前捕获成图，后续重复执行时减少 CPU 调度和 kernel launch 开销。 面试回答时我会先给定义，再说明它解决的背景问题，接着讲核心机制和适用场景，最后补充限制与工程风险。这样回答比只背一句概念更完整。

### 如何判断线上推理服务是否适合开启 CUDA Graph？

回答思路：按“目标定义 -> 关键步骤 -> 风险控制 -> 指标验证”的顺序回答，避免只讲抽象原则。

回答模板：

我会把 判断线上推理服务是否适合开启 CUDA Graph 拆成几个步骤：先明确目标和输入输出，再设计关键模块或策略，然后加入失败处理和权限/质量约束，最后用离线指标、线上指标或回归测试验证效果。在 CUDA Graph 场景里，重点是能落到真实工程流程。
