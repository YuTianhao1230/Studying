# Applied ML Coding

## 一句话解释

Applied ML Coding 是算法工程师面试中偏工程实现的 ML 编程题，要求你用 Python / NumPy / PyTorch 写出小型可运行组件，而不是只讲概念。

## 为什么重要

大厂 MLE 面试越来越常见两类编码题：

- 普通数据结构算法题。
- 贴近 ML 日常工作的实现题。

后者考察你是否真的理解模型、训练、评测和数据处理，而不是只会调库。

## 高频题型

### 1. Attention from Scratch

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

### 2. Sampling

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

### 3. Training Loop

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

### 4. Metrics

常见手写：

- Accuracy。
- Precision / Recall / F1。
- AUC。
- PR-AUC。
- NDCG。
- MRR。

### 5. RAG Utility

常见：

- 文档 chunking。
- overlap。
- top-k retrieval。
- rerank。
- citation mapping。

### 6. Eval Harness

常见：

- 读取 JSONL。
- 调用模型或 mock 模型。
- 解析输出。
- 计算指标。
- 保存 bad case。

## 面试写代码的原则

- 先写清楚输入输出。
- 明确 shape。
- 先实现正确版本，再优化。
- 对边界条件写测试。
- 能解释复杂度。
- 不要过度封装。

## 常见误区

- 只背公式，写不出 shape 正确的代码。
- softmax 没做数值稳定。
- mask 方向写反。
- eval 时忘记关闭梯度。
- 指标实现没有处理极端样本。
- 写了函数但没有最小测试。

## 准备清单

- 用 NumPy 手写 softmax、cross entropy、AUC。
- 用 PyTorch 手写 scaled dot-product attention。
- 写一个最小 training loop。
- 写一个 top-k/top-p 采样函数。
- 写一个 JSONL eval harness。
- 写一个 RAG chunking + retrieval demo。

## 面试可能怎么问

1. 手写 scaled dot-product attention。
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 Applied ML Coding 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
2. 实现 top-k sampling。
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 Applied ML Coding 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
3. 写一个支持 gradient accumulation 的训练循环。
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 Applied ML Coding 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
4. 实现 AUC 或 NDCG。
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 Applied ML Coding 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
5. 写一个简单的模型评测脚手架。
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：这个问题在 Applied ML Coding 场景下，核心是说明它解决什么实际工程问题，以及如何落地。完整回答需要覆盖目标、执行方式、失败风险和验证指标。
