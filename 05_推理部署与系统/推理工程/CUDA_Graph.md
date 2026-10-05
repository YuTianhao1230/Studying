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

- [KV Cache](<KV_Cache与Prefill_Decode.md#kvcache与prefilldecode>)：减少重复 [Attention](<../../02_大模型/基础架构/Self-Attention.md#self-attention>) 计算。
- [Continuous Batching](<Batching.md#batching>)：提升吞吐。
- [Speculative Decoding](<Speculative_Decoding.md#speculativedecoding>)：减少大模型 decode 步数。
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

### CUDA Graph 为什么对 Decode 阶段尤其有用？

回答思路：先说明 Decode 单步计算小、重复次数多，再解释 CPU launch overhead 为什么会占据更高比例。

回答模板：

Decode 每一步通常只处理一个新 Token，却会重复发射大量小 Kernel；当单步 GPU 计算较短时，CPU 逐个提交 Kernel 的开销和抖动就会变得明显。CUDA Graph 把一段稳定的 GPU 操作捕获为图，之后通过一次 Replay 重放整段执行流，减少 CPU 参与和 Kernel Launch 次数。因此它常能改善小 Batch Decode 的 TPOT 和尾延迟，但不会减少 KV Cache 占用，也不能替代 Batching。

### CUDA Graph 的 Capture 和 Replay 如何工作？

回答思路：按预热、捕获、固定缓冲区和重放四步回答，强调重放时复用执行拓扑与内存地址。

回答模板：

使用 CUDA Graph 时，通常先完成必要的预热和内存分配，再用一组固定形状、固定地址的输入捕获 Kernel、依赖关系和执行顺序。后续请求把新数据写入同一组静态缓冲区，然后 Replay 已捕获的图。这样不必让 CPU 每轮重新构造并提交整串操作，但代价是输入形状、内存地址和控制流必须足够稳定。

### 动态请求如何与 CUDA Graph 共存？

回答思路：说明不能为任意形状直接复用同一张图，再给出分桶、Padding、多图缓存及 Eager Fallback。

回答模板：

线上请求的 Batch Size 和序列长度是动态的，而一张 CUDA Graph 通常要求形状和地址稳定。常见做法是按 Batch Size 或长度分桶，为若干常用形状分别捕获图，请求进入后 Padding 到最近桶；不常见或含动态控制流的请求回退到 Eager 执行。这样可以提高图复用率，但会引入 Padding 浪费、图缓存显存和更复杂的调度，所以要和 Continuous Batching 联合设计。

### 什么情况下不应使用 CUDA Graph？

回答思路：从图复用率、动态控制流、计算粒度和额外显存判断，并说明如何用数据验证。

回答模板：

如果请求形状高度离散、执行路径频繁变化、图几乎无法复用，或者单次计算已经很大、Launch Overhead 占比很低，CUDA Graph 的收益可能不足以覆盖 Padding、图缓存和调度成本。评估时应在相同请求分布下对比开启前后的 TPOT、吞吐、p95/p99、显存峰值和图命中率，并保留非图执行的回退路径。
