# TensorRT_LLM

## 知识点解析

### 概述

TensorRT-LLM 是 NVIDIA 面向大语言模型推理优化的高性能推理框架，重点利用 GPU kernel 优化、量化、并行和 batching 来降低延迟、提升吞吐。

### 为什么大厂 JD 会提到

大模型上线后，成本和延迟往往比模型本身更影响业务可用性。

推理工程需要回答：

- 单请求延迟能不能降下来？
- 多请求吞吐能不能撑住？
- GPU 显存够不够？
- p99 是否稳定？
- 单 token 成本能不能接受？

TensorRT-LLM 属于解决这些问题的工程工具之一。

### 核心优化方向

#### Kernel 优化

把 Transformer 中常见计算做高性能实现，例如：

- GEMM。
- Attention。
- LayerNorm / RMSNorm。
- MLP。
- Softmax。

#### 量化

降低权重和激活精度，减少显存和计算成本。

常见：

- FP8。
- INT8。
- INT4。
- Weight-only quantization。

#### Batching

把多个请求合并执行，提高 GPU 利用率。

关注：

- batch size。
- token 数差异。
- prefill / decode 阶段调度。
- p99 latency。

#### 并行策略

大模型可能需要多 GPU：

- Tensor Parallel。
- Pipeline Parallel。
- Expert Parallel。

#### KV Cache 管理

推理系统需要高效管理 KV Cache，避免显存碎片和重复计算。

### 和 vLLM 的关系

- vLLM 更常被用于易用、高吞吐 serving，核心代表是 PagedAttention 和 continuous batching。
- TensorRT-LLM 更强调 NVIDIA GPU 上的底层推理性能优化。

实际系统中可能按场景选择，也可能组合使用不同组件。

### 常见指标

- Time To First Token。
- Tokens per Second。
- p50/p95/p99 latency。
- GPU utilization。
- Memory bandwidth。
- Throughput。
- Cost per 1M tokens。

### 常见误区

- 只看平均延迟，不看 p99。
- 只看 tokens/s，不看首 token 延迟。
- 忽略输入输出长度分布。
- 量化后不评估质量损失。
- 没有区分 prefill 和 decode 瓶颈。
- 只优化模型，不优化调度和服务链路。

## 面试应对

### TensorRT_LLM 是什么？

回答思路：定位为 NVIDIA GPU 上的高性能推理框架，点出它靠 kernel 优化、量化、并行和 batching 组合来降延迟提吞吐，属于让已有模型跑得更省更快的工程工具。

回答模板：

TensorRT-LLM 是 NVIDIA 面向大语言模型推理优化的高性能推理框架，重点利用 GPU kernel 优化、量化、并行和 batching 来降低延迟、提升吞吐。 它关注的核心不是提升模型本身能力，而是让已有模型在服务中更高吞吐、更低延迟、更稳定地运行。

### TensorRT_LLM 解决什么问题？

回答思路：落在模型上线后的成本与延迟压力上——单请求延迟、多请求吞吐、显存够不够、p99 稳不稳、单 token 成本，说明它是系统性解决这些指标的工具。

回答模板：

大模型上线后，成本和延迟往往比模型本身更影响业务可用性。推理工程需要回答： 单请求延迟能不能降下来？多请求吞吐能不能撑住？ 在工程上通常要结合 p50/p95/p99 延迟、tokens/s、显存峰值、并发数和失败率来判断它是否有效。

### TensorRT_LLM 的核心机制是什么？

回答思路：拆解四条优化线——GEMM/Attention 等 kernel 优化、FP8/INT8/INT4 量化、TP/PP/EP 并行、batching 与 KV Cache 管理，说明它们如何合起来提利用率降成本。

回答模板：

TensorRT-LLM 是 NVIDIA 面向大语言模型推理优化的高性能推理框架，重点利用 GPU kernel 优化、量化、并行和 batching 来降低延迟、提升吞吐。 这类机制的价值在于减少无效计算、降低显存碎片、提升 GPU 利用率或稳定服务调度。

### TensorRT_LLM 有哪些限制？

回答思路：点出它绑定 NVIDIA 生态、构建/编译成本高，且常见误区是量化后不评估质量损失、不区分 prefill/decode 瓶颈、只看均值不看 p99，可对比 vLLM 说明选型。

回答模板：

TensorRT-LLM 是 NVIDIA 面向大语言模型推理优化的高性能推理框架，重点利用 GPU kernel 优化、量化、并行和 batching 来降低延迟、提升吞吐。 如果线上效果异常，需要检查请求长度分布、batch 配置、KV cache、显存利用率、并发策略和模型并行配置。
