# FSDP

## 知识点解析

### 概述

FSDP，Fully Sharded Data Parallel，是 PyTorch 提供的全参数切分数据并行方案，把模型参数、梯度和优化器状态切分到不同 GPU 上，降低单卡显存压力。

### 为什么需要 FSDP

普通 DDP 会在每张卡上保存一份完整模型参数：

```text
每张 GPU：
  参数 full copy
  梯度 full copy
  优化器状态 full copy
```

模型越大，单卡显存越容易爆。

FSDP 的思路是：

```text
每张 GPU 只保存一部分参数、梯度和优化器状态
需要计算某层时，再临时 all-gather 对应参数
计算完后释放或重新切分
```

### 和 DDP 的区别

| 维度 | DDP | FSDP |
| --- | --- | --- |
| 参数 | 每卡完整一份 | 每卡保存切片 |
| 梯度 | 每卡完整一份 | 可切片 |
| 优化器状态 | 每卡完整一份 | 可切片 |
| 显存 | 高 | 低 |
| 通信 | 主要梯度 all-reduce | 参数 all-gather + reduce-scatter |
| 适用 | 中小模型 | 大模型 |

### 和 ZeRO 的关系

FSDP 可以理解为 PyTorch 原生实现的 ZeRO-3 类方案。

- ZeRO-1：切优化器状态。
- ZeRO-2：切优化器状态和梯度。
- ZeRO-3：切优化器状态、梯度和参数。
- FSDP：核心目标同样是 full shard。

### 核心机制

#### Sharding

把参数按 rank 切分，每张卡只保存一部分。

#### All-Gather

某一层 forward/backward 需要完整参数时，临时聚合参数。

#### Reduce-Scatter

反向传播后，把梯度规约并重新切回各卡。

#### Activation Checkpointing

为了进一步省显存，可以不保存全部激活，反向时重新计算。

### 优点

- 显著降低显存占用。
- PyTorch 原生支持，生态兼容较好。
- 适合大模型训练和微调。

### 代价

- 通信更复杂。
- 对网络带宽更敏感。
- wrapping 策略影响性能。
- checkpoint 保存和加载更复杂。

## 面试应对

### FSDP 是什么？

回答思路：定位成 PyTorch 原生的全分片数据并行，对比 DDP 每卡存整份，强调它把参数/梯度/优化器状态切到各卡、用时再 all-gather，本质是 ZeRO-3 的官方实现。

回答模板：

FSDP，Fully Sharded Data Parallel，是 PyTorch 提供的全参数切分数据并行方案，把模型参数、梯度和优化器状态切分到不同 GPU 上，降低单卡显存压力。 它的作用通常体现在让更大模型、更长序列或更高吞吐的训练成为可能。

### FSDP 的核心机制是什么？

回答思路：抓 sharding、forward/backward 前 all-gather 聚合参数、算完释放、反向用 reduce-scatter 规约梯度这条闭环，再补上可叠加 activation checkpointing 进一步省显存。

回答模板：

FSDP，Fully Sharded Data Parallel，是 PyTorch 提供的全参数切分数据并行方案，把模型参数、梯度和优化器状态切分到不同 GPU 上，降低单卡显存压力。 核心判断是它改变了哪些训练状态的存储、计算或通信方式，以及由此带来的显存和吞吐变化。

### FSDP 的工程取舍是什么？

回答思路：讲省显存与原生生态兼容的收益，同时点出 all-gather 通信更重、对带宽敏感、wrapping 策略影响性能、分片 checkpoint 保存加载更复杂这些代价。

回答模板：

显著降低显存占用。PyTorch 原生支持，生态兼容较好。适合大模型训练和微调。 训练系统选型时要结合模型规模、GPU 数量、网络带宽、batch size、序列长度、checkpoint 策略和排障成本。
