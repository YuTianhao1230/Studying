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

回答思路：先说明它解决的具体编程问题，再给一个典型使用场景。

回答模板：

Applied ML Coding 指把机器学习方法落到可运行、可复现、可维护的代码中，包括数据处理、训练循环、评测和实验管理。

### Applied_ML_Coding 适合什么场景？

回答思路：结合训练脚本、数据处理或算法题说明使用边界。

回答模板：

重点不是只会调库，而是能处理数据边界、指标口径、随机性、日志、checkpoint 和异常恢复。

### Applied_ML_Coding 常见坑是什么？

回答思路：从类型、返回值、副作用、性能和可读性检查。

回答模板：

使用 Applied_ML_Coding 时，我会重点确认输入类型、返回值语义、是否修改原对象，以及在循环或大数据场景下是否有额外开销。对于工程代码，还要保证命名和结构清晰，必要时用小例子或单元测试固定边界行为。
