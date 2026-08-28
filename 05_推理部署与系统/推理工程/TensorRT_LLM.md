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
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 TensorRT LLM 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
2. vLLM 和 TensorRT-LLM 的关注点有什么不同？
   - 回答思路：先分别定义两个概念，再从目标、机制、适用场景和工程取舍四个角度对比。
   - 回答模板：这个问题在 TensorRT LLM 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
3. 如何评估大模型推理服务性能？
   - 回答思路：按“目标定义 -> 关键步骤 -> 风险控制 -> 指标验证”的顺序回答，避免只讲抽象原则。
   - 回答模板：同时看 TTFT、TPOT、吞吐、p50/p95/p99、GPU 利用率、显存、错误率、成本和输出质量。不同业务对首 token 和完成时延的权重不同。
4. 量化为什么能降低推理成本？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 TensorRT LLM 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
5. 为什么 p99 latency 对线上服务很重要？
   - 回答思路：先指出背后的核心约束，再解释收益，最后补充如果不这样做会带来的风险。
   - 回答模板：因为 p99 latency 对线上服务很重要 背后有明确工程约束：如果不处理，容易带来质量不稳定、成本上升、错误难排查或安全风险。在 TensorRT LLM 场景里，这个问题的关键是说明它如何提升系统可控性、可验证性和线上稳定性。
