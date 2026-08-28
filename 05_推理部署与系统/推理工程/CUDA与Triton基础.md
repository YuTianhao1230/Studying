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
2. 为什么 GPU 适合矩阵计算？
3. Triton 和 CUDA 有什么关系？
4. Kernel fusion 为什么能提速？
5. 大模型推理中哪些地方容易成为 GPU 瓶颈？
