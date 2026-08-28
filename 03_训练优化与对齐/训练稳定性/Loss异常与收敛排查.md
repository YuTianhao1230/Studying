# 训练 loss 异常怎么排查？

## 直接回答

训练 loss 异常时，不要一上来就调学习率。应该先确认现象，再按“数据 -> loss 计算 -> 优化配置 -> 数值精度 -> 分布式训练 -> 评测口径”的顺序排查。

面试可以这样回答：

```text
我会先确认 loss 异常的类型，是 NaN、突然 spike、不下降、震荡，还是 train loss 下降但 eval 不提升。
然后定位异常发生的 step、数据 shard、模型版本和配置变更。
排查顺序上，我会先查数据和 label，再查 loss mask、label shift、padding mask，然后看学习率、warmup、grad norm、混合精度和分布式同步。
最后用小数据单卡复现，逐步关闭复杂配置，判断是数据问题、代码问题还是分布式配置问题。
```

## 常见 loss 异常类型

### loss 变成 NaN / Inf

常见原因：

- 学习率过大。
- 梯度爆炸。
- FP16 overflow。
- softmax / exp 数值溢出。
- loss 里出现 `log(0)` 或除零。
- label 越界。
- 数据里有异常值。

### loss 不下降

常见原因：

- 学习率太小或 optimizer 没有正常 step。
- 参数被冻结了。
- label / mask 错误。
- 训练目标和数据格式不匹配。
- 模型加载错了。
- 数据过难或噪声太大。

### loss 震荡很大

常见原因：

- 学习率过大。
- batch size 太小，梯度方差大。
- 数据混合比例不稳定。
- 某些 batch 特别长或特别难。
- 多任务 loss 权重不合理。

### train loss 下降但 eval 不提升

常见原因：

- 过拟合。
- train/eval 分布不一致。
- eval 指标实现有问题。
- 推理参数和训练目标不一致。
- 数据泄漏导致训练表现虚高。
- SFT 学到了格式，但没有学到真实能力。

## 排查流程

### 先确认问题是否真实

先看：

- 异常是单次 spike 还是持续异常。
- 是否可复现。
- 是否从某个 step 或 checkpoint 后开始。
- 是否只影响某个任务、数据源、rank 或模型版本。
- 是否和最近代码、数据、配置变更有关。

不要只看一条 loss 曲线，要同时看：

- train loss。
- eval loss。
- grad norm。
- learning rate。
- token throughput。
- batch token length。
- GPU memory。
- 各 rank 状态。

### 检查数据

数据是训练异常的高频来源。

重点查：

- 空样本。
- 超长样本。
- 文本编码异常。
- 特殊 token 异常。
- label 为空或错位。
- 输入输出格式不一致。
- 数据分布突然变化。
- 某个数据源质量明显低。

如果怀疑是数据问题，可以按数据源、任务类型、长度区间做 loss 分桶。

### 检查 loss 和 mask

SFT / 大模型训练里，mask 错非常常见。

重点查：

- 是否把 user prompt 也算进 loss。
- assistant 部分是否被正确监督。
- padding token 是否 mask 掉。
- label shift 是否正确。
- ignore index 是否一致。
- packing 多条样本时，不同样本之间是否错误 attention。
- 多任务 loss 权重是否过大。

### 检查学习率和优化器

重点查：

- learning rate 是否过大或过小。
- warmup 是否太短。
- scheduler 是否按预期变化。
- optimizer 是否真的执行了 `step()`。
- gradient accumulation 是否正确。
- weight decay 是否过大。
- checkpoint 恢复后 optimizer state 是否完整。

### 检查混合精度

如果用 FP16 / BF16，要查：

- 是否出现 overflow。
- loss scaling 是否异常。
- attention logits 是否过大。
- 某些算子是否不稳定。
- 是否有 rank 先 NaN，随后扩散到全局。

大模型训练中，BF16 通常比 FP16 更稳。

### 检查分布式训练

如果单卡正常，多卡异常，优先查分布式配置：

- global batch size 是否算错。
- gradient accumulation step 是否一致。
- DDP / FSDP 参数同步是否正常。
- sampler 是否重复或漏样本。
- 每个 rank 的数据分布是否一致。
- checkpoint 是否完整恢复参数、optimizer、scheduler 和随机种子。

## 最小复现方法

复杂训练问题要收敛变量。

建议顺序：

1. 固定随机种子。
2. 取小数据集。
3. 单卡小 batch 训练。
4. 关闭混合精度。
5. 关闭复杂并行。
6. 打印关键 tensor 的 min / max / mean。
7. 单独复现异常 batch。

判断逻辑：

- 小数据单卡都不收敛：优先查代码、loss、mask、label。
- 单卡正常，多卡异常：优先查分布式、sampler、同步、checkpoint。
- 只有某类数据异常：优先查数据质量和任务格式。
- 训练指标正常但 eval 异常：优先查评测集、指标实现、推理参数。

## 常见追问

### 追问一：loss spike 一定要停训吗？

回答：

```text
不一定。大模型训练中偶发 loss spike 可能来自长样本或难 batch，只要 grad norm 没失控、loss 能恢复、eval 不退化，可以继续观察。
但如果 spike 后持续变差，或者出现 NaN、grad norm 爆炸、eval 大幅下降，就要暂停并定位数据、学习率、混合精度和异常 batch。
```

### 追问二：train loss 下降但 eval 变差怎么办？

回答：

```text
这通常说明模型在训练集上拟合了，但没有泛化。
我会先检查 train/eval 分布是否一致，是否存在数据泄漏或标签噪声，再看 eval 指标实现和推理参数是否正确。
如果确认评测没问题，就要考虑过拟合、数据比例不合理、训练步数过多或模型学到了格式捷径。
```

### 追问三：如何判断是不是 mask 错了？

回答：

```text
我会抽样打印 token、label 和 loss mask，确认 user prompt、padding token、system prompt 是否被 mask 掉，只在 assistant 目标部分算 loss。
如果用了 packing，还要检查样本之间是否错误互相 attention。
mask 错通常会表现为 loss 异常、输出格式奇怪，或者模型学会复述用户输入。
```

### 追问四：为什么要做小数据 overfit test？

回答：

```text
小数据 overfit test 是判断训练链路是否正常的有效办法。
如果模型连几十条干净样本都拟合不了，说明大概率是代码、loss、mask、optimizer 或模型加载有问题。
如果小数据能很快拟合，但全量数据不行，问题更可能在数据分布、数据质量、超参数或分布式配置。
```

### 追问五：线上效果变差但训练 loss 正常，怎么排查？

回答：

```text
训练 loss 正常只能说明训练目标在下降，不代表线上效果一定好。
我会检查评测集是否覆盖线上分布，推理参数是否和评测一致，线上输入分布是否漂移，服务版本和数据处理链路是否变更。
同时要做 bad case 分桶，看问题集中在任务类型、长度、语言、数据源还是某类边界场景。
```
