# Megatron-LM

## 一句话解释

Megatron-LM 是 NVIDIA 开源的大模型训练框架，核心价值是提供张量并行、流水并行、序列并行等能力，用于训练超大规模 Transformer 模型。

## 为什么重要

当模型规模超过单机单卡能力时，需要把模型和训练过程拆到多张 GPU 上。

Megatron-LM 主要解决：

- 模型太大，单卡放不下。
- batch 太大，单机跑不动。
- 训练吞吐不够。
- 多 GPU 通信和计算需要高效调度。

## 核心并行方式

### 1. Tensor Parallel

把单层矩阵计算切到多张 GPU 上。

例如 Transformer 中的 MLP 或 Attention 权重矩阵，可以按列或行切分。

适合：

- 单层参数很大。
- 模型宽度很大。

代价：

- 每层内部通信频繁。
- 对高速互联依赖强。

### 2. Pipeline Parallel

把模型不同层切到不同 GPU 上。

```text
GPU0: Layer 1-8
GPU1: Layer 9-16
GPU2: Layer 17-24
GPU3: Layer 25-32
```

适合：

- 模型层数很多。
- 单卡放不下完整模型。

问题：

- Pipeline bubble。
- micro-batch 调度复杂。

### 3. Data Parallel

每组模型副本处理不同数据，再同步梯度。

常和 tensor/pipeline parallel 组合。

### 4. Sequence Parallel

进一步切分序列维度，降低 activation 显存压力。

常用于长上下文训练。

## 3D Parallelism

大模型训练通常组合三种并行：

```text
Data Parallel
  x Tensor Parallel
  x Pipeline Parallel
```

这就是常说的 3D 并行。

## 和 DeepSpeed 的关系

- Megatron-LM 更强调模型并行，尤其是 Tensor Parallel 和 Pipeline Parallel。
- DeepSpeed 更强调 ZeRO、优化器状态切分、训练加速和系统工程。
- 实际训练中常见 Megatron-DeepSpeed 组合。

## 常见瓶颈

- GPU 显存不足。
- GPU 利用率低。
- 通信占比过高。
- Pipeline bubble。
- 数据加载跟不上。
- checkpoint 保存和恢复慢。

## 面试可能怎么问

1. Megatron-LM 主要解决什么问题？
2. Tensor Parallel 和 Pipeline Parallel 区别是什么？
3. 什么是 3D 并行？
4. Pipeline bubble 是什么？
5. Megatron 和 DeepSpeed 如何互补？
