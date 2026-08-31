# Applied_ML_Coding

## 知识点解析

### 概述

Applied ML Coding 是算法工程师面试中偏工程实现的 ML 编程题，要求你用 Python / NumPy / PyTorch 写出小型可运行组件，而不是只讲概念。

### 为什么重要

大厂 MLE 面试越来越常见两类编码题：

- 普通数据结构算法题。
- 贴近 ML 日常工作的实现题。

后者考察你是否真的理解模型、训练、评测和数据处理，而不是只会调库。

### 高频题型

#### Attention from Scratch

需要能写：

- Q/K/V 矩阵计算。
- scaled dot-product。
- causal mask。
- softmax。
- attention output。

关注点：

- 张量 shape。
- mask 广播。
- 数值稳定性。
- 时间和空间复杂度。

#### Sampling

常见：

- Greedy decoding。
- Temperature sampling。
- Top-k sampling。
- Top-p sampling。
- Beam search。

关注点：

- logits 到概率。
- 排序和截断。
- 随机采样。
- 终止条件。

#### Training Loop

需要能写：

- forward。
- loss。
- backward。
- optimizer step。
- gradient accumulation。
- eval mode。
- checkpoint。

关注点：

- `model.train()` / `model.eval()`。
- `torch.no_grad()` / `torch.inference_mode()`。
- 梯度清零。
- mixed precision。

#### Metrics

常见手写：

- Accuracy。
- Precision / Recall / F1。
- AUC。
- PR-AUC。
- NDCG。
- MRR。

#### RAG Utility

常见：

- 文档 chunking。
- overlap。
- top-k retrieval。
- rerank。
- citation mapping。

#### Eval Harness

常见：

- 读取 JSONL。
- 调用模型或 mock 模型。
- 解析输出。
- 计算指标。
- 保存 bad case。

### 面试写代码的原则

- 先写清楚输入输出。
- 明确 shape。
- 先实现正确版本，再优化。
- 对边界条件写测试。
- 能解释复杂度。
- 不要过度封装。

### 常见误区

- 只背公式，写不出 shape 正确的代码。
- softmax 没做数值稳定。
- mask 方向写反。
- eval 时忘记关闭梯度。
- 指标实现没有处理极端样本。
- 写了函数但没有最小测试。

### 准备清单

- 用 NumPy 手写 softmax、cross entropy、AUC。
- 用 PyTorch 手写 scaled dot-product attention。
- 写一个最小 training loop。
- 写一个 top-k/top-p 采样函数。
- 写一个 JSONL eval harness。
- 写一个 RAG chunking + retrieval demo。
## 面试应对

### Applied_ML_Coding 是什么？

回答思路：点明它考的是能手写 attention、采样、training loop、指标这类可运行 ML 组件，而不是背概念。

回答模板：

Applied ML Coding 指把机器学习方法落到可运行、可复现、可维护的代码中，包括数据处理、训练循环、评测和实验管理。

### Applied_ML_Coding 适合什么场景？

回答思路：强调它落在数据边界、指标口径、随机性、日志、checkpoint 这些真实工程细节上，而非调库。

回答模板：

重点不是只会调库，而是能处理数据边界、指标口径、随机性、日志、checkpoint 和异常恢复。

### Applied_ML_Coding 常见坑是什么？

回答思路：按 shape 对不上、softmax 没减 max、mask 方向反、eval 忘关梯度、指标不处理极端样本这几类实战坑来讲。

回答模板：

写 Applied ML 编码题我最常踩的坑按几类记：一是 shape 对不上，写 attention 或广播时没先把每一步张量的形状标清楚；二是数值稳定性，softmax 不减 max 会溢出、log 里没加 eps；三是 mask 方向写反，causal mask 本该屏蔽未来 token 却屏蔽了历史；四是评估时忘了 `model.eval()` 和 `torch.no_grad()`，或者反向前没清零梯度导致累积；五是指标实现没处理极端样本，比如分母为 0、只有单一类别时的 AUC。我的习惯是先写正确版本、明确输入输出和 shape，再对边界条件补一个最小测试，最后才谈优化和复杂度。
