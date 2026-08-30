# vLLM

## 知识点解析

### 概述

vLLM 是一个高吞吐的大语言模型推理和服务框架，核心特点是 PagedAttention、continuous batching 和 OpenAI-compatible API。

### vLLM 主要解决什么问题

大模型推理的痛点是：

- 多用户并发请求长度不同。
- 每个请求生成长度不同。
- KV cache 占用大量显存。
- 静态 batch 容易浪费计算。
- 长上下文容易造成显存碎片。

vLLM 通过更高效的 KV cache 管理和请求调度提升吞吐。

### 核心概念：PagedAttention

传统 KV cache 通常为每个请求分配连续显存。问题是：

- 不同请求长度差异大。
- 输出长度事先不确定。
- 容易产生显存浪费和碎片。

PagedAttention 借鉴操作系统分页思想，把 KV cache 切成 block 管理。

直观理解：

```text
传统方式：
每个请求占一大段连续显存。

PagedAttention：
每个请求由多个小 block 组成，按需分配和回收。
```

好处：

- 降低显存浪费。
- 更容易支持长上下文。
- 更适合动态并发请求。

### 核心概念：Continuous Batching

传统 batching 可能是：

```text
等一批请求全部生成结束，再处理下一批。
```

问题是有些请求很短，有些很长，短请求完成后 GPU 位置空出来但不能马上补新请求。

Continuous Batching 的思想是：

```text
某个请求生成结束后，马上把新请求插入 batch。
```

这样可以提升 GPU 利用率和整体吞吐。

### vLLM 适合什么场景

- LLM 在线服务。
- 高并发文本生成。
- 离线批量推理。
- 多模型评测。
- OpenAI API 风格服务替代。

### vLLM 不等于什么

- 不等于训练框架。
- 不主要负责模型微调。
- 不负责数据清洗和评测指标。
- 不保证模型效果提升，它主要提升推理效率和服务能力。

### 常见参数

- `max_model_len`：模型最大上下文长度。
- `tensor_parallel_size`：张量并行卡数。
- `gpu_memory_utilization`：允许使用的 GPU 显存比例。
- `max_num_seqs`：同时处理的序列数量上限。
- `temperature`：采样随机性。
- `top_p`：nucleus sampling 参数。
- `max_tokens`：最大生成 token 数。

## 面试应对

### vLLM 是什么？

回答思路：先定位它属于推理框架、推理优化还是底层执行机制。

回答模板：

vLLM 是一个高吞吐的大语言模型推理和服务框架，核心特点是 PagedAttention、continuous batching 和 OpenAI-compatible API。 它关注的核心不是提升模型本身能力，而是让已有模型在服务中更高吞吐、更低延迟、更稳定地运行。

### vLLM 解决什么问题？

回答思路：从 KV cache、batching、显存、并发、p99 延迟等推理瓶颈回答。

回答模板：

大模型推理的痛点是： 多用户并发请求长度不同。每个请求生成长度不同。KV cache 占用大量显存。 在工程上通常要结合 p50/p95/p99 延迟、tokens/s、显存峰值、并发数和失败率来判断它是否有效。

### vLLM 的核心机制是什么？

回答思路：讲清楚它改变了哪部分推理流程，以及为什么能改善吞吐或显存。

回答模板：

传统 KV cache 通常为每个请求分配连续显存。问题是： 不同请求长度差异大。输出长度事先不确定。容易产生显存浪费和碎片。 这类机制的价值在于减少无效计算、降低显存碎片、提升 GPU 利用率或稳定服务调度。

### vLLM 有哪些限制？

回答思路：说明适用边界、参数配置风险和线上排查重点。

回答模板：

vLLM 是一个高吞吐的大语言模型推理和服务框架，核心特点是 PagedAttention、continuous batching 和 OpenAI-compatible API。 如果线上效果异常，需要检查请求长度分布、batch 配置、KV cache、显存利用率、并发策略和模型并行配置。
