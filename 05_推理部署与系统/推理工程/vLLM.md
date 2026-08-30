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

回答思路：先定位为 LLM 推理服务框架，再说明它主要优化吞吐、显存和并发调度。

回答模板：

vLLM 是一个面向大语言模型推理和服务的高吞吐框架。它不改变模型参数，也不提升模型本身能力，核心价值是在在线服务或离线批量推理中更高效地管理 KV cache 和并发请求。它的代表机制是 PagedAttention 和 Continuous Batching，前者降低 KV cache 显存浪费，后者提升动态请求下的 GPU 利用率。

### vLLM 解决什么推理瓶颈？

回答思路：从请求长度不一致、生成长度不可预知、KV cache 占用大和 batch 利用率低展开。

回答模板：

LLM 推理的难点在于请求是动态的：不同用户 prompt 长度不同，生成长度也不确定，而每个请求都要维护 KV cache。传统静态 batching 容易出现短请求等长请求、显存连续分配浪费、GPU 空转等问题。vLLM 通过更细粒度的 KV cache 分页管理和持续批处理，把完成的请求及时移出 batch，把新请求补进来，从而提升吞吐并降低显存碎片。

### PagedAttention 和 Continuous Batching 分别做什么？

回答思路：分别解释显存管理和调度策略，不要混成一个概念。

回答模板：

PagedAttention 主要解决 KV cache 的显存管理问题。它把 KV cache 切成固定大小的 block，按需分配和回收，类似操作系统分页，因此不要求每个请求占用连续的大块显存，能减少碎片和预留浪费。Continuous Batching 解决的是请求调度问题：当 batch 中某些序列生成结束后，系统可以立即加入新的请求，而不是等整批全部结束。前者提升显存利用率，后者提升 GPU 计算利用率，两者合起来让 vLLM 更适合高并发 LLM 服务。

### 使用 vLLM 需要关注哪些参数和风险？

回答思路：结合线上指标回答，包括上下文长度、并发、显存、延迟和并行配置。

回答模板：

使用 vLLM 时，我会重点看 `max_model_len`、`max_num_seqs`、`gpu_memory_utilization`、`tensor_parallel_size` 这些参数，因为它们会直接影响显存峰值、并发量和延迟。线上评估不能只看 tokens/s，还要看首 token 延迟、p95/p99 延迟、请求失败率、OOM、不同长度请求下的吞吐变化。如果长上下文或高并发场景下频繁 OOM，就需要调整上下文上限、并发数、KV cache 策略或模型并行配置。
