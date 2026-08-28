# CUDA Graph

## 一句话解释

CUDA Graph 是 NVIDIA CUDA 提供的执行图机制，可以把一串 GPU 操作提前捕获成图，后续重复执行时减少 CPU 调度和 kernel launch 开销。

## 为什么大模型推理会用到

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

## 适合场景

- 计算图结构稳定。
- 输入 shape 相对固定。
- 同一批次配置反复执行。
- decode 阶段大量重复操作。
- 对 p99 latency 敏感。

## 不适合场景

- shape 频繁变化。
- 控制流高度动态。
- batch size 和 sequence length 变化太大。
- 每次请求的执行路径都不同。

因此线上推理系统常常需要配合 padding、bucket、固定 batch shape 等策略使用 CUDA Graph。

## 和其他推理优化的关系

- KV Cache：减少重复 Attention 计算。
- Continuous Batching：提升吞吐。
- Speculative Decoding：减少大模型 decode 步数。
- CUDA Graph：减少 CPU launch 调度开销。
- TensorRT-LLM：常结合底层 kernel 优化和 graph 机制做高性能推理。

## 关键收益

- 降低 CPU overhead。
- 降低延迟抖动。
- 提升小 batch / decode 场景效率。
- 改善 p99 latency。

## 常见风险

- shape 不稳定导致 graph 难复用。
- 捕获阶段复杂。
- 内存地址和执行路径需要稳定。
- 动态控制流支持有限。
- 和动态 batching 组合时需要额外调度设计。

## 面试可能怎么问

1. CUDA Graph 解决什么问题？
2. 为什么大模型 decode 阶段容易受 kernel launch overhead 影响？
3. CUDA Graph 对输入 shape 有什么要求？
4. CUDA Graph 和 continuous batching 是否冲突？
5. 如何判断线上推理服务是否适合开启 CUDA Graph？
