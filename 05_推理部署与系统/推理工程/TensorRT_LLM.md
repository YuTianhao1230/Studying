# TensorRT_LLM

## 知识点解析

### 概述

TensorRT-LLM 是 NVIDIA 面向大语言模型推理优化的高性能推理框架，重点利用 GPU kernel 优化、[量化](<量化.md#量化>)、并行和 batching 来降低延迟、提升吞吐。

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

把 [Transformer](<../../02_大模型/基础架构/Transformer.md#transformer>) 中常见计算做高性能实现，例如：

- GEMM。
- [Attention](<../../02_大模型/基础架构/Self-Attention.md#self-attention>)。
- LayerNorm / [RMSNorm](<../../02_大模型/基础架构/RMSNorm.md#rmsnorm>)。
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

推理系统需要高效管理 [KV Cache](<KV_Cache与Prefill_Decode.md#kvcache与prefilldecode>)，避免显存碎片和重复计算。

### 和 vLLM 的关系

- vLLM 更常被用于易用、高吞吐 serving，核心代表是 [PagedAttention](<vLLM.md#pagedattention-和-continuous-batching-分别做什么>) 和 continuous batching。
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

### TensorRT-LLM 通过哪些层次优化推理？

回答思路：按算子、精度、并行、调度和缓存五个层次组织，不把它简化成单一编译器。

回答模板：

TensorRT-LLM 是 NVIDIA GPU 上的大语言模型高性能推理框架。它在算子层使用优化后的 GEMM、Attention 和融合 Kernel，在数值层支持 FP8、INT8、INT4 等量化，在多卡层提供 TP、PP、EP 等并行，并结合 In-flight Batching 与 KV Cache 管理提高资源利用率。它的价值来自整条执行链的联合优化，而不是某一个单独技巧。

### TensorRT-LLM 和 vLLM 应该怎么选？

回答思路：从硬件生态、性能上限、模型接入、构建成本和迭代速度比较，避免绝对化结论。

回答模板：

如果生产硬件以 NVIDIA GPU 为主、模型结构稳定，并且愿意承担构建和调优成本来追求低延迟或高性能上限，我会重点评估 TensorRT-LLM。若更看重模型接入速度、OpenAI 兼容服务、动态调度和快速迭代，vLLM 往往更方便。最终不能只按框架名选择，而要用目标模型、量化方案、并发和长度分布做同口径压测。

### 如何公平评测 TensorRT-LLM 的收益？

回答思路：固定模型、硬件、精度和流量，分开报告延迟、吞吐、显存、质量与稳定性。

回答模板：

我会固定模型版本、权重精度、GPU 型号与数量、输入输出长度分布、并发和采样参数，分别测 TTFT、TPOT、端到端 p50/p95/p99、Token 吞吐、显存峰值和错误率。若启用量化，还要在代表性评测集上验证质量变化。只有同时给出延迟受限和吞吐受限场景，才能判断收益来自 Kernel、量化、Batching 还是并行配置。

### TensorRT-LLM 的主要工程代价是什么？

回答思路：说明 NVIDIA 生态绑定、构建与版本兼容、模型支持和量化验证成本。

回答模板：

TensorRT-LLM 主要面向 NVIDIA 生态，部署时要处理驱动、CUDA、框架版本和插件兼容，部分模型或自定义结构还需要额外适配。构建 Engine、选择 Profile 和量化校准也会增加发布成本。出现收益不及预期时，我会先区分 Prefill 与 Decode 瓶颈，再检查 Shape、Batch、KV Cache 和并行配置，而不是默认框架会自动给出最优结果。
