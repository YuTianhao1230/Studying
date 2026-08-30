# CUDA与Triton基础

## 知识点解析

### 概述

CUDA 是 NVIDIA GPU 编程平台；Triton 是更高层的 GPU kernel 编写语言，常用于为深度学习模型实现高性能自定义算子。

### 为什么算法工程师要知道

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

### CUDA 核心概念

#### Kernel

Kernel 是在 GPU 上并行执行的函数。

例如矩阵乘、LayerNorm、Softmax 都可以由 GPU kernel 执行。

#### Thread / Block / Grid

CUDA 并行层级：

```text
Grid
  -> Block
  -> Thread
```

一个 kernel 会启动大量线程并行处理数据。

#### Memory Hierarchy

GPU 内存层级会影响性能：

- Global Memory：容量大，访问慢。
- Shared Memory：block 内共享，速度快。
- Register：线程私有，最快。
- Cache：缓存常用数据。

#### Memory Coalescing

相邻线程访问连续内存更高效。

如果访存不连续，GPU 带宽利用率会下降。

### Triton 是什么

Triton 让你用 Python 风格写 GPU kernel，比 CUDA C++ 更易上手。

常用于：

- 自定义 MatMul。
- LayerNorm。
- Softmax。
- Attention。
- Quantization kernel。
- 算子融合。

### 性能优化关注点

- 算子是否被融合。
- 是否重复读写显存。
- Tensor Core 是否被充分利用。
- batch 和 sequence length 是否适合当前 kernel。
- 显存带宽还是计算算力是瓶颈。
- p99 latency 是否受长尾请求影响。

### 和大模型推理的关系

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

## 面试应对

### CUDA与Triton基础 是什么？

回答思路：先定位它属于推理框架、推理优化还是底层执行机制。

回答模板：

CUDA 是 NVIDIA GPU 编程平台；Triton 是更高层的 GPU kernel 编写语言，常用于为深度学习模型实现高性能自定义算子。Triton 让你用 Python 风格写 GPU kernel，比 CUDA C++ 更易上手。 它关注的核心不是提升模型本身能力，而是让已有模型在服务中更高吞吐、更低延迟、更稳定地运行。

### CUDA与Triton基础 解决什么问题？

回答思路：从 KV cache、batching、显存、并发、p99 延迟等推理瓶颈回答。

回答模板：

大模型训练和推理的性能瓶颈常常不只在算法，也在底层算子和 GPU 利用率。JD 中提到的这些词都和底层性能相关： CUDA。Kernel fusion。 在工程上通常要结合 p50/p95/p99 延迟、tokens/s、显存峰值、并发数和失败率来判断它是否有效。

### CUDA与Triton基础 的核心机制是什么？

回答思路：讲清楚它改变了哪部分推理流程，以及为什么能改善吞吐或显存。

回答模板：

CUDA 是 NVIDIA GPU 编程平台；Triton 是更高层的 GPU kernel 编写语言，常用于为深度学习模型实现高性能自定义算子。 这类机制的价值在于减少无效计算、降低显存碎片、提升 GPU 利用率或稳定服务调度。

### CUDA与Triton基础 有哪些限制？

回答思路：说明适用边界、参数配置风险和线上排查重点。

回答模板：

CUDA 是 NVIDIA GPU 编程平台；Triton 是更高层的 GPU kernel 编写语言，常用于为深度学习模型实现高性能自定义算子。 如果线上效果异常，需要检查请求长度分布、batch 配置、KV cache、显存利用率、并发策略和模型并行配置。
