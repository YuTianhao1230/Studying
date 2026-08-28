# SFT

## 一句话解释

SFT，Supervised Fine-Tuning，是用高质量指令-回答数据对预训练模型进行监督微调，让模型学会按人类期望的格式和任务方式回答。

## 为什么需要 SFT

预训练模型主要学习“预测下一个 token”，不天然等于好用的助手。

SFT 解决的是：

- 学会遵循指令。
- 学会对话格式。
- 学会特定领域任务。
- 学会输出结构化结果。
- 为后续 RLHF / DPO / GRPO 打基础。

## 数据格式

常见样本：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个专业助手"},
    {"role": "user", "content": "解释 KV Cache"},
    {"role": "assistant", "content": "KV Cache 是..."}
  ]
}
```

关键不是数量越多越好，而是质量、覆盖面和格式一致性。

## 训练目标

SFT 通常仍然是 next-token prediction，只是训练数据变成了人类整理好的指令数据。

常见做法：

- 只对 assistant 部分计算 loss。
- user/system 部分只作为上下文。
- 多轮对话要保留角色边界。

## 常见数据来源

- 人工标注。
- 专家问答。
- 业务日志清洗。
- 高质量模型蒸馏。
- 合成数据。
- 失败 case 修正。

## 常见问题

- 数据格式不统一，模型学坏输出格式。
- 低质量合成数据太多，导致能力退化。
- 领域数据覆盖不足，模型只学到表面模式。
- 多轮对话 mask 错误，导致训练目标污染。
- 过拟合小数据，通用能力下降。

## 和后续对齐的关系

```text
Pretrain -> SFT -> Reward Model / Preference Data -> RLHF / DPO / GRPO
```

SFT 让模型“会做任务”；偏好优化让模型“更符合人类偏好或业务目标”。

## 面试可能怎么问

1. SFT 和预训练有什么区别？
2. 为什么 SFT 通常只对 assistant 部分算 loss？
3. SFT 数据质量怎么控制？
4. SFT 后模型能力退化怎么办？
5. SFT、RLHF、DPO 的关系是什么？
