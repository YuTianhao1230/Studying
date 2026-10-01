# CUDA与Triton基础

## 知识点解析

### 概述

CUDA 是 NVIDIA GPU 编程平台；Triton 是更高层的 GPU kernel 编写语言，常用于为深度学习模型实现高性能自定义[算子](<算子.md>)。

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
- [Attention](<../../02_大模型/基础架构/Self-Attention.md>)。
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
- [KV Cache](<KV_Cache与Prefill_Decode.md>) 读写占显存带宽。
- 小算子太多导致 kernel launch overhead。

优化方向：

- FlashAttention。
- [PagedAttention](<vLLM.md>)。
- Kernel fusion。
- Quantization。
- CUDA Graph。
- TensorRT-LLM。

## 面试应对

### CUDA 和 Triton 的定位有什么区别？

回答思路：从抽象层级、开发方式、控制能力和适用场景比较。

回答模板：

CUDA 是 NVIDIA 的 GPU 编程平台，CUDA C++ 能直接控制线程层级、共享内存、同步和底层硬件特性，能力最完整，但开发和调优成本高。Triton 是面向张量计算的高层 Kernel 语言，使用 Python 风格按数据块描述程序，由编译器完成线程映射等工作。常规深度学习自定义算子可优先尝试 Triton；需要极细粒度硬件控制、特殊同步或 Triton 不支持的能力时，再使用 CUDA。

### 如何判断 Kernel 是计算瓶颈还是访存瓶颈？

回答思路：比较计算量与数据搬运量，结合 Profiler 的计算单元、带宽和访存指标定位，再选择优化方向。

回答模板：

我会先用 Profiler 看 Kernel 时间、显存带宽利用率、计算单元利用率和数据搬运量。如果带宽接近上限而计算单元利用率低，通常是 Memory-bound，应优先做连续访存、算子融合和减少中间 Tensor；如果计算单元接近饱和，则更像 Compute-bound，应关注 Tensor Core、数据类型、Tile 大小和算法计算量。优化前必须先定位瓶颈，否则手写 Kernel 可能只是增加维护成本。

### 连续访存和 Shared Memory 为什么能提升性能？

回答思路：从 Global Memory 访问代价、Warp 合并事务和片上数据复用解释。

回答模板：

GPU 的 Global Memory 延迟高、带宽宝贵。相邻线程访问连续地址时，硬件可以把多个访问合并成更少的内存事务；把会被重复使用的数据先加载到 Shared Memory 或寄存器，还能减少反复访问 Global Memory。优化时也要控制 Shared Memory 和寄存器占用，因为资源用得过多会降低 Occupancy，最终需要用 Profiler 验证而不是只凭规则判断。

### 什么时候值得写 Triton 自定义算子？

回答思路：先检查现有库和编译器，再依据热点占比、可融合性和维护成本决策。

回答模板：

只有当 Profiler 证明某段算子是主要热点，且现有 PyTorch、cuBLAS、cuDNN、FlashAttention 或编译器生成结果不能满足需求时，我才会考虑 Triton。它特别适合融合多个逐元素操作、减少中间读写或实现规则的块级张量计算。完成后要对不同 Shape 和数据类型做正确性、数值误差、性能与回退测试；如果收益只覆盖极少输入，就不值得承担额外维护成本。
