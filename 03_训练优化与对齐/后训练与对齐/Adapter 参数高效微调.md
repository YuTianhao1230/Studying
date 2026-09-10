# Adapter 参数高效微调

## 知识点解析

### Adapter 是什么

在参数高效微调语境中，Adapter 通常指插入预训练模型内部的小型可训练模块。

基本思路是：

```text
冻结 base model
  -> 在 Transformer 层旁边插入小模块
  -> 只训练 Adapter 参数
  -> 推理时把 Adapter 和 base model 一起使用
```

它解决的问题是：不重新训练整个大模型，也能让模型适配新任务、领域、语言或输出风格。

### Bottleneck Adapter 的结构

经典 Adapter 是一个 bottleneck 结构：

```text
hidden state h
  -> down projection
  -> low-dimensional adapter space
  -> activation
  -> up projection
  -> residual add
```

可以写成：

```text
Adapter(h) = W_up * f(W_down * h)
h' = h + Adapter(h)
```

其中：

- `W_down` 把 hidden size 降到较小的 bottleneck dimension。
- `f` 通常是 ReLU、GELU 或类似非线性激活。
- `W_up` 再把维度映射回 hidden size。
- residual connection 把 Adapter 输出加回原 hidden state。

因为 bottleneck dimension 远小于 hidden size，所以 Adapter 参数量比全参数微调少很多。

### Adapter 插在哪里

不同 Adapter 方法的插入位置不同，常见位置有：

#### Transformer block 内

```text
Self-Attention
  -> Adapter
  -> residual

FFN
  -> Adapter
  -> residual
```

有些方法放在 attention/FFN 子层之后，有些放在 LayerNorm 前后。插入位置会影响训练稳定性、表达能力和推理开销。

#### Embedding/输入层

给不同任务增加任务特定的 embedding 或 soft prompt，适合任务格式差异明显但不需要大幅改动模型能力的场景。

#### 输出头

分类、序列标注等任务可以只训练任务头。对生成式大模型，单独训练输出头通常不足以完成复杂领域适配。

### Sequential Adapter 和 Parallel Adapter

#### Sequential Adapter

Adapter 串在原始 Transformer 子层后面：

```text
h -> Transformer block -> Adapter -> residual output
```

优点是结构直观、容易插入已有模型；缺点是推理路径变长。

#### Parallel Adapter

Adapter 与原始子层并行：

```text
h -> Transformer block ----\
  -> Adapter ------------- + -> merge
```

优点是更容易保留原始模型路径，某些场景下更适合多任务组合；缺点是融合方式和初始化更复杂。

### Adapter、LoRA 和 Prefix Tuning 的区别

| 方法 | 可训练对象 | 是否增加 hidden-state 计算 | 参数形式 | 主要特点 |
| --- | --- | --- | --- | --- |
| Adapter | bottleneck 模块 | 会增加一小段前向计算 | 小型神经网络 | 模块化、可插拔 |
| LoRA | 低秩矩阵 `A/B` | 训练时增加旁路，merge 后可消除 | `Delta W = BA` | 参数少、工程最成熟 |
| Prefix Tuning | 每层 prefix 的 K/V 或连续向量 | 会增加 prefix 序列计算 | 可学习 prefix | 适合控制生成条件 |
| Prompt Tuning | 输入侧 soft prompt | 增加输入 token/向量 | 可学习 embedding | 参数极少，容量有限 |
| BitFit | bias 参数 | 基本不增加结构 | 原模型 bias | 极轻量但表达能力有限 |

一句话：

```text
Adapter 是小网络模块；
LoRA 是低秩权重更新；
Prefix/Prompt Tuning 是可学习的连续提示。
```

它们都可以归入 PEFT，但不是同一种实现。

### Adapter 和 LoRA 的关键差异

LoRA 直接改变某个线性层的权重更新：

```text
W' = W + B A
```

经典 Adapter 不改变原线性层权重，而是在 hidden state 路径上增加一个 bottleneck：

```text
h' = h + W_up f(W_down h)
```

因此：

- LoRA 更容易在训练后 merge 到 base model。
- Adapter 更像一个可独立开关的网络模块。
- LoRA merge 后通常不增加推理层数。
- Adapter 即使只加载一个模块，也可能增加推理延迟。
- Adapter 更适合模块化、多任务切换和任务隔离。

### Adapter 的参数量

假设 Transformer hidden size 是 `d`，Adapter bottleneck 是 `m`，其中 `m << d`：

```text
Adapter 参数量约为：
  d * m + m * d
  = 2 d m
```

如果每层 attention 和 FFN 后都插入 Adapter，再乘以层数和插入位置数量。

当 `d=4096`、`m=64` 时，一个 Adapter 模块的参数量大约是：

```text
2 * 4096 * 64 ≈ 0.52M
```

相比数十亿参数的 base model，仍然很小。

### Adapter 的训练流程

```text
加载 base model
  -> 冻结 base model
  -> 插入 Adapter
  -> 只把 Adapter 参数设为 trainable
  -> 训练任务数据
  -> 保存 Adapter 权重和配置
  -> 推理时加载 base model + Adapter
```

训练前必须确认：

- base model 参数确实被冻结。
- Adapter 参数数量和比例符合预期。
- optimizer 只接收 Adapter 参数。
- label mask 和数据格式正确。
- Adapter 的 hidden size、层数和插入位置与 base model 匹配。

### 多 Adapter 使用

同一个 base model 可以配多个任务 Adapter：

```text
同一个 base model
  + Adapter_A：代码
  + Adapter_B：客服
  + Adapter_C：关键帧
```

推理时可以按请求切换：

```text
request type
  -> 选择对应 Adapter
  -> base model + adapter
```

优势：

- 不需要为每个任务保存完整模型。
- 多任务切换成本低。
- 任务参数相互隔离。

风险：

- 不同 Adapter 需要基于兼容的 base model。
- 直接叠加多个 Adapter 可能产生能力冲突。
- Adapter 切换可能带来加载、缓存和并发管理开销。

### Adapter Merge

经典 Adapter 通常不能像 LoRA 一样简单地把所有影响完全吸收到原始线性权重中，因为它本身是一个额外的非线性网络模块。

部署方式通常是：

```text
方式一：
base model + Adapter 一起加载

方式二：
使用框架提供的融合/编译能力优化 Adapter 推理
```

不能默认认为“Adapter 一定可以 merge 后零开销部署”。是否能融合、融合到什么程度，要看 Adapter 结构和推理框架。

### 什么时候选 Adapter

适合：

- 需要多个任务模块独立切换。
- 希望任务能力和 base model 解耦。
- 任务需要比 soft prompt 更强的非线性适配能力。
- 不能或不希望直接修改 base model 权重。
- 需要保留多个领域模块分别管理。

不一定适合：

- 追求极致推理吞吐，且可以使用 LoRA merge。
- 任务只需要非常轻量的输入控制。
- 推理框架对额外模块支持不好。
- 任务需要大幅改变视觉或语言底层能力。

### 如何选择 Adapter、LoRA 和全参数微调

```text
先问：问题是什么？

只是格式/风格适配
  -> Prompt Tuning / LoRA

需要多个任务独立切换
  -> Adapter / LoRA 多 adapter

需要低显存快速实验
  -> LoRA / QLoRA

需要较大幅度改变视觉、语言或跨模态能力
  -> Full Fine-tuning
```

最终选择还要看：

- base model 和任务差距。
- 数据量。
- 显存和训练预算。
- 是否需要多任务切换。
- 推理延迟要求。
- 是否需要合并成单一模型。

## 面试应对

### Adapter 是什么？

回答思路：先定义为插入 Transformer 的小型可训练模块，再讲冻结 base model 和 bottleneck。

回答模板：

Adapter 是参数高效微调的一种方法。它在预训练模型的 Transformer 层中插入一个小型 bottleneck 模块，训练时冻结 base model，只更新 Adapter 参数。Adapter 通常先把 hidden state 降到低维空间，经过非线性变换后再映射回原维度，并通过残差连接加回主路径。这样可以用很少的参数让模型适配新任务，同时保留 base model 的通用能力，并且方便保存和切换不同任务模块。

### Adapter 和 LoRA 有什么区别？

回答思路：区分 hidden-state 小网络和权重低秩更新。

回答模板：

LoRA 是在原始线性层旁边增加低秩矩阵，学习权重更新 `Delta W = BA`；Adapter 是在 Transformer hidden state 路径上插入一个 bottleneck 小网络，学习 `h' = h + W_up f(W_down h)`。LoRA 训练和部署生态更成熟，训练后通常可以 merge 到 base model，推理额外开销较小；Adapter 模块化和任务隔离更强，但推理时通常仍保留额外计算路径。两者都属于 PEFT，但适合的工程约束不同。

### Adapter 为什么能节省显存？

回答思路：从冻结 base model、减少梯度和 optimizer state 解释。

回答模板：

Adapter 节省显存的主要原因是冻结 base model，只对新增的少量 Adapter 参数计算梯度并维护 optimizer state。全参数微调需要为所有参数保存梯度、优化器状态和激活；Adapter 只需要为 bottleneck 模块保存这些训练状态，所以显存和 checkpoint 都小很多。需要注意的是，Adapter 不会自动消除激活显存和输入显存，长序列或多模态输入仍然可能成为主要开销。

### 多个 Adapter 可以一起加载吗？

回答思路：先说可以，但要讲兼容性、冲突和推理管理。

回答模板：

多个 Adapter 可以在同一个 base model 上组合或按请求切换，但前提是它们的 base model、模型结构、层位置和 hidden size 兼容。组合时不同任务的更新可能互相干扰，不能简单认为能力会线性叠加；还要考虑 Adapter 顺序、缩放系数、显存缓存和推理延迟。工程上我会先分别评测单 Adapter，再评测组合 Adapter，确认目标任务提升的同时没有明显能力退化。

### 什么时候不用 Adapter 而选择全参数微调？

回答思路：从任务差距、可训练数据、资源预算和能力改变幅度回答。

回答模板：

如果任务只需要格式、风格或小范围领域适配，我会优先用 Adapter 或 LoRA。只有当任务和 base model 差距较大，需要同时改变视觉、语言或跨模态底层能力，或者数据量和算力足够支撑时，才考虑全参数微调。全参数微调的表达能力更强，但显存、训练和存储成本更高，也更容易出现灾难性遗忘，所以必须用 dev 指标和通用能力回归来验证。
