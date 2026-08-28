# TensorRT-LLM

## 一句话解释

TensorRT-LLM 是 NVIDIA 面向大语言模型推理优化的高性能推理框架，重点利用 GPU kernel 优化、量化、并行和 batching 来降低延迟、提升吞吐。

## 为什么大厂 JD 会提到

大模型上线后，成本和延迟往往比模型本身更影响业务可用性。

推理工程需要回答：

- 单请求延迟能不能降下来？
- 多请求吞吐能不能撑住？
- GPU 显存够不够？
- p99 是否稳定？
- 单 token 成本能不能接受？

TensorRT-LLM 属于解决这些问题的工程工具之一。

## 核心优化方向

### 1. Kernel 优化

把 Transformer 中常见计算做高性能实现，例如：

- GEMM。
- Attention。
- LayerNorm / RMSNorm。
- MLP。
- Softmax。

### 2. 量化

降低权重和激活精度，减少显存和计算成本。

常见：

- FP8。
- INT8。
- INT4。
- Weight-only quantization。

### 3. Batching

把多个请求合并执行，提高 GPU 利用率。

关注：

- batch size。
- token 数差异。
- prefill / decode 阶段调度。
- p99 latency。

### 4. 并行策略

大模型可能需要多 GPU：

- Tensor Parallel。
- Pipeline Parallel。
- Expert Parallel。

### 5. KV Cache 管理

推理系统需要高效管理 KV Cache，避免显存碎片和重复计算。

## 和 vLLM 的关系

- vLLM 更常被用于易用、高吞吐 serving，核心代表是 PagedAttention 和 continuous batching。
- TensorRT-LLM 更强调 NVIDIA GPU 上的底层推理性能优化。

实际系统中可能按场景选择，也可能组合使用不同组件。

## 常见指标

- Time To First Token。
- Tokens per Second。
- p50/p95/p99 latency。
- GPU utilization。
- Memory bandwidth。
- Throughput。
- Cost per 1M tokens。

## 常见误区

- 只看平均延迟，不看 p99。
- 只看 tokens/s，不看首 token 延迟。
- 忽略输入输出长度分布。
- 量化后不评估质量损失。
- 没有区分 prefill 和 decode 瓶颈。
- 只优化模型，不优化调度和服务链路。

## 面试可能怎么问

1. TensorRT-LLM 主要解决什么问题？
2. vLLM 和 TensorRT-LLM 的关注点有什么不同？
3. 如何评估大模型推理服务性能？
4. 量化为什么能降低推理成本？
5. 为什么 p99 latency 对线上服务很重要？
