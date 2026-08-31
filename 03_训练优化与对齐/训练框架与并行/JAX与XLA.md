# JAX与XLA

## 知识点解析

### 概述

JAX 是面向高性能数值计算和机器学习研究的 Python 框架；XLA 是加速线性代数计算的编译器，可以把计算图优化后运行在 GPU/TPU 上。

### 为什么要知道

很多大模型研究和 Google DeepMind 相关岗位会提到 JAX。

JAX 的特点是：

- 函数式编程风格。
- 自动求导。
- `jit` 编译加速。
- `vmap` 自动向量化。
- `pmap` / sharding 做并行。
- 和 XLA 深度结合，适合 TPU/GPU 高性能训练。

### 核心概念

#### `grad`

自动求导。

```python
from jax import grad

def loss_fn(x):
    return x ** 2

grad_loss = grad(loss_fn)
```

#### `jit`

把 Python 函数编译成高性能计算图。

适合：

- 计算密集函数。
- 训练 step。
- 推理 step。

注意：

- 首次编译有开销。
- 输入 shape 变化可能触发重新编译。

#### `vmap`

把单样本函数自动扩展成批处理函数。

直觉：

```text
不用手写 for loop，让框架自动对 batch 维向量化
```

#### `pmap` / Sharding

把计算分布到多个设备上。

适合：

- 多 GPU。
- TPU pod。
- 大模型并行。

### 函数式思想

JAX 倾向于纯函数：

- 输入明确。
- 输出明确。
- 不依赖隐式状态。
- 随机数需要显式传入 key。

这让编译、并行和复现更容易，但对 PyTorch 用户来说需要适应。

### JAX 和 PyTorch 的区别

| 维度 | PyTorch | JAX |
| --- | --- | --- |
| 风格 | 命令式、动态图 | 函数式、编译友好 |
| 易用性 | 上手容易，生态广 | 研究和高性能并行友好 |
| 编译 | `torch.compile` 逐步增强 | `jit` 是核心范式 |
| 随机数 | 隐式全局状态较常见 | 显式 PRNGKey |
| 大规模训练 | PyTorch/FSDP/DeepSpeed 常见 | TPU/DeepMind 生态常见 |

### XLA 是什么

XLA，Accelerated Linear Algebra，是编译器。

它会做：

- 算子融合。
- 内存优化。
- 常量折叠。
- 计算图优化。
- 针对 GPU/TPU 后端生成高效代码。

## 面试应对

### JAX与XLA 是什么？

回答思路：分别定位——JAX 是函数式的高性能数值计算/研究框架，XLA 是把计算图做算子融合、内存优化后编译到 GPU/TPU 的编译器，二者深度绑定支撑 TPU 高性能训练。

回答模板：

JAX 是面向高性能数值计算和机器学习研究的 Python 框架；XLA 是加速线性代数计算的编译器，可以把计算图优化后运行在 GPU/TPU 上。XLA，Accelerated Linear Algebra，是编译器。 它的作用通常体现在让更大模型、更长序列或更高吞吐的训练成为可能。

### JAX与XLA 的核心机制是什么？

回答思路：抓四个函数变换 grad/jit/vmap/pmap，以及纯函数、随机数需显式传 PRNGKey 的范式，说明正是无隐式状态才让 jit 编译、并行和复现更容易。

回答模板：

JAX 倾向于纯函数： 输入明确。不依赖隐式状态。随机数需要显式传入 key。这让编译、并行和复现更容易，但对 PyTorch 用户来说需要适应。 核心判断是它改变了哪些训练状态的存储、计算或通信方式，以及由此带来的显存和吞吐变化。

### JAX与XLA 的工程取舍是什么？

回答思路：对比 PyTorch，讲 JAX 在 TPU/DeepMind 生态和高性能并行上的优势，代价是函数式范式上手成本、jit 首次编译开销与 shape 变化触发重编译，以及生态相对更窄。

回答模板：

JAX 是面向高性能数值计算和机器学习研究的 Python 框架；XLA 是加速线性代数计算的编译器，可以把计算图优化后运行在 GPU/TPU 上。 训练系统选型时要结合模型规模、GPU 数量、网络带宽、batch size、序列长度、checkpoint 策略和排障成本。
