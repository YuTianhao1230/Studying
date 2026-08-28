# CUDA 与 Triton 基础

## 一句话解释

CUDA 是 NVIDIA GPU 编程平台；Triton 是更高层的 GPU kernel 编写语言，常用于为深度学习模型实现高性能自定义算子。

## 为什么算法工程师要知道

大模型训练和推理的性能瓶颈常常不只在算法，也在底层算子和 GPU 利用率。

JD 中提到的这些词都和底层性能相关：

- CUDA。
- Triton。
- Kernel fusion。
- GPU acceleration。
- TensorRT-LLM。
- FlashAttention。
- Quantization。
- p99 latency。

不一定要每个算法工程师都手写 CUDA，但需要理解性能瓶颈从哪里来。

## CUDA 核心概念

### 1. Kernel

Kernel 是在 GPU 上并行执行的函数。

例如矩阵乘、LayerNorm、Softmax 都可以由 GPU kernel 执行。

### 2. Thread / Block / Grid

CUDA 并行层级：

```text
Grid
  -> Block
  -> Thread
```

一个 kernel 会启动大量线程并行处理数据。

### 3. Memory Hierarchy

GPU 内存层级会影响性能：

- Global Memory：容量大，访问慢。
- Shared Memory：block 内共享，速度快。
- Register：线程私有，最快。
- Cache：缓存常用数据。

### 4. Memory Coalescing

相邻线程访问连续内存更高效。

如果访存不连续，GPU 带宽利用率会下降。

## Triton 是什么

Triton 让你用 Python 风格写 GPU kernel，比 CUDA C++ 更易上手。

常用于：

- 自定义 MatMul。
- LayerNorm。
- Softmax。
- Attention。
- Quantization kernel。
- 算子融合。

## 性能优化关注点

- 算子是否被融合。
- 是否重复读写显存。
- Tensor Core 是否被充分利用。
- batch 和 sequence length 是否适合当前 kernel。
- 显存带宽还是计算算力是瓶颈。
- p99 latency 是否受长尾请求影响。

## 和大模型推理的关系

大模型推理中常见瓶颈：

- Prefill 阶段 attention 计算重。
- Decode 阶段 batch 小、访存重。
- KV Cache 读写占显存带宽。
- 小算子太多导致 kernel launch overhead。

优化方向：

- FlashAttention。
- PagedAttention。
- Kernel fusion。
- Quantization。
- CUDA Graph。
- TensorRT-LLM。

## 面试可能怎么问

1. CUDA kernel 是什么？
   - 回答思路：先给一句话定义，再说明它解决什么问题，最后补一个工程例子或常见风险。
   - 回答模板：CUDA kernel 是 CUDA与Triton基础 里的核心概念，主要解决系统在能力、效率、稳定性或可控性上的问题。面试中要说明它的定义、适用场景、限制，以及在真实工程中如何验证它有效。
2. 为什么 GPU 适合矩阵计算？
   - 回答思路：先指出背后的核心约束，再解释收益，最后补充如果不这样做会带来的风险。
   - 回答模板：因为 GPU 适合矩阵计算 背后有明确工程约束：如果不处理，容易带来质量不稳定、成本上升、错误难排查或安全风险。在 CUDA与Triton基础 场景里，这个问题的关键是说明它如何提升系统可控性、可验证性和线上稳定性。
3. Triton 和 CUDA 有什么关系？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 CUDA与Triton基础 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
4. Kernel fusion 为什么能提速？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 CUDA与Triton基础 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
5. 大模型推理中哪些地方容易成为 GPU 瓶颈？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 CUDA与Triton基础 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
