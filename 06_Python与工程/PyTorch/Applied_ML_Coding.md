# Applied_ML_Coding

## 知识点解析

### 概述

本文整理使用 Python、NumPy 和 PyTorch 实现常见机器学习组件的方法，包括张量运算、模型模块、训练过程和评测指标。

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
- 写一个 [RAG](<../../02_大模型/应用与问题/RAG.md>) chunking + retrieval demo。

## 题面输入输出约定

- 稳定 Softmax 接受任意 batch 维，需处理大 logits、NaN 和全屏蔽行。
- 交叉熵题通常要求给出 logits 形状、均值损失和梯度；标签可以是类别下标或 one-hot/soft label。
- Attention 题需明确 `Q/K/V` 的最后两维、mask 广播方向和 causal mask 的可见范围；全遮行应有明确输出约定。
- AUC、AP、NDCG 题需先声明同分、积分规则、`IDCG=0` 和单类标签处理。
- KMeans 题需说明返回标签和中心、随机初始化、空簇和 `k>n` 边界。
- Top-k/Top-p 题需说明边界 token 是否保留、过滤后是否重新归一化，以及是否负责实际随机采样。

### 复杂度与边界

| 组件 | 主要复杂度 | 必查边界 |
| --- | --- | --- |
| Softmax / LayerNorm | `O(... × classes)` | 大 logits、NaN、空最后维度 |
| Attention | `O(B × Tq × Tk × d)` | causal mask、广播、全遮行 |
| AUC / AP | 排序 `O(n log n)` | 同分、单类标签 |
| KMeans | 每轮 `O(n × k × d)` | `k>n`、空簇、随机种子 |
| 训练循环 | 每步由模型前反向决定 | 尾窗口、`eval/no_grad`、恢复后状态 |

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
