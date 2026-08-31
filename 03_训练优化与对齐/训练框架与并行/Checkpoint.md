# Checkpoint

## 知识点解析

### 概述

Checkpoint 是模型训练或运行过程中保存下来的状态快照，最常见的是模型权重，也可能包括优化器状态、学习率调度器、随机种子和训练进度。

### 为什么需要 Checkpoint

训练大模型成本很高，可能跑几小时、几天甚至几周。

Checkpoint 用来解决：

- 训练中断后恢复。
- 保存不同训练阶段的模型。
- 选择验证集效果最好的版本。
- 对比不同 step 的模型能力。
- 支持模型发布、回滚和复现。

### Checkpoint 里通常有什么

常见内容：

- model weights：模型参数。
- optimizer state：优化器状态，例如 Adam 的一阶/二阶矩。
- lr scheduler state：学习率调度器状态。
- global step：当前训练步数。
- epoch：当前训练轮数。
- random state：随机数状态。
- tokenizer / config：模型配置和 tokenizer 信息。
- adapter weights：LoRA 等微调参数。

### 只保存权重够不够

看目标。

如果只是推理：

```text
模型权重 + config + tokenizer 通常够用。
```

如果要断点恢复训练：

```text
还需要 optimizer state、scheduler state、global step、随机状态等。
```

否则恢复后训练曲线可能不连续，甚至效果不同。

### Best Checkpoint 和 Last Checkpoint

#### Last Checkpoint

最后一次保存的 checkpoint。

适合：

- 断点续训。
- 保存训练最新状态。

#### Best Checkpoint

验证集指标最好的 checkpoint。

适合：

- 模型发布。
- 效果对比。

注意：如果验证集被反复选择，可能产生对验证集过拟合。

### Checkpoint 和模型版本的关系

Checkpoint 是文件级产物，模型版本是管理层概念。

一个模型版本通常应该记录：

- base model。
- checkpoint 路径。
- 训练数据版本。
- 训练代码 commit。
- 训练参数。
- tokenizer 版本。
- 评测结果。
- 发布时间。

### 常见问题

#### Checkpoint 太多怎么办？

可以保留：

- latest。
- best。
- 固定间隔 checkpoint。
- 关键实验 checkpoint。

同时清理无价值中间版本。

#### LoRA checkpoint 为什么不能单独推理？

LoRA checkpoint 通常只保存 adapter 参数，需要和 base model 一起加载。

#### 加载 checkpoint 报 shape mismatch 怎么办？

可能原因：

- base model 不一致。
- tokenizer 词表变了。
- 模型结构配置不一致。
- LoRA target module 不一致。

## 面试应对

### Checkpoint 是什么？

回答思路：抓“存了什么、为什么存”，先列 checkpoint 里的权重/优化器状态/scheduler/global step/随机态，再落到断点续训和选优发布这两大刚需。

回答模板：

Checkpoint 是模型训练或运行过程中保存下来的状态快照，最常见的是模型权重，也可能包括优化器状态、学习率调度器、随机种子和训练进度。 它的作用通常体现在让更大模型、更长序列或更高吞吐的训练成为可能。

### Checkpoint 的核心机制是什么？

回答思路：抓“只存权重够不够”这个分水岭——推理只要权重加 config、tokenizer，断点续训必须带 optimizer、scheduler、global step 和随机态，否则恢复后训练曲线不连续。

回答模板：

Checkpoint 是模型训练或运行过程中保存下来的状态快照，最常见的是模型权重，也可能包括优化器状态、学习率调度器、随机种子和训练进度。 核心判断是它改变了哪些训练状态的存储、计算或通信方式，以及由此带来的显存和吞吐变化。

### Checkpoint 的工程取舍是什么？

回答思路：抓 best 与 last 的用途取舍、只保留 latest/best/间隔版本控制数量与存储成本，再点出 LoRA checkpoint 依赖 base、加载 shape mismatch 这些常见坑。

回答模板：

Checkpoint 是模型训练或运行过程中保存下来的状态快照，最常见的是模型权重，也可能包括优化器状态、学习率调度器、随机种子和训练进度。 训练系统选型时要结合模型规模、GPU 数量、网络带宽、batch size、序列长度、checkpoint 策略和排障成本。
