# PEFT 参数高效微调

## 知识点解析

### 概述

PEFT（Parameter-Efficient Fine-Tuning，参数高效微调）是一类**只训练少量新增参数或少量子模块，而冻结大部分预训练模型权重**的微调方法。它的目标是在尽量保留基座模型通用能力的前提下，用更低显存、更低存储和更低训练成本完成任务适配。

一句话：

```text
PEFT = 冻结大模型主体 + 只训练少量可插拔参数，让模型适配新任务
```

LoRA 是目前最常用的 PEFT 方法，但 PEFT 不等于 LoRA。PEFT 是方法体系，LoRA 是其中一种具体实现。

### PEFT 要解决什么问题

全参数微调会更新模型所有权重，问题很明显：

1. **显存成本高**：不仅要存模型权重，还要存梯度和 optimizer state。
2. **训练成本高**：大模型全参回传非常贵。
3. **存储成本高**：每个任务都保存一份完整模型，空间开销很大。
4. **灾难性遗忘风险更高**：新任务数据质量或规模不够时，容易破坏原有通用能力。
5. **多任务切换不灵活**：不同业务线、不同风格模型需要维护多份完整 checkpoint。

PEFT 的思路是：**不要直接大幅改动基座模型，而是在它旁边训练一个小的任务适配层**。

### 常见 PEFT 方法

#### LoRA

LoRA（Low-Rank Adaptation）冻结原始权重 `W0`，只训练低秩更新量：

```text
W = W0 + ΔW
ΔW = B * A
```

其中 `A` 和 `B` 是低秩矩阵，参数量远小于原始权重矩阵。LoRA 常插在 Transformer 的线性层上，比如 `q_proj`、`v_proj`、`o_proj`、`up_proj`、`down_proj` 等。

特点：

- 工程成熟，效果稳定。
- 训练参数少，显存友好。
- adapter 可以单独保存，也可以 merge 回基座模型。
- 是当前 LLM SFT 里最常用的 PEFT 方法。

#### QLoRA

QLoRA 是 LoRA 的进一步省显存版本：**基座模型用低比特量化加载，LoRA adapter 仍用较高精度训练**。

常见组合：

```text
4-bit base model + LoRA adapter + paged optimizer
```

特点：

- 显存占用更低。
- 可以在更小显卡上微调较大模型。
- 需要注意量化误差和训练稳定性。

#### Prefix Tuning

Prefix Tuning 不改模型主体，而是在每层 attention 前面加一段可训练的 prefix 向量，让模型在生成时受到这段连续提示的影响。

特点：

- 参数量很小。
- 更像“可训练的软提示”。
- 对生成任务有用，但表达能力通常不如 LoRA 稳定。

#### Prompt Tuning / P-Tuning

Prompt Tuning 是训练一组连续向量作为 soft prompt，把它拼到输入前面，让模型通过这组可学习提示适配任务。

特点：

- 参数量极小。
- 对超大模型更有效，小模型上可能效果不稳定。
- 适合任务格式比较固定的场景。

#### Adapter Tuning

Adapter Tuning 在 Transformer 层中插入小型 bottleneck 模块，只训练这些 adapter。

特点：

- 模块化强。
- 多任务切换方便。
- 推理时会多一小段计算路径，可能带来额外延迟。

### PEFT 和全参数微调的区别

| 维度 | 全参数微调 | PEFT |
| --- | --- | --- |
| 更新对象 | 全部模型参数 | 少量 adapter / soft prompt / prefix |
| 显存成本 | 高 | 低 |
| 存储成本 | 每任务一份完整模型 | 每任务只保存 adapter |
| 能力改动幅度 | 大 | 中等，受 adapter 容量限制 |
| 灾难性遗忘风险 | 更高 | 更低 |
| 多任务切换 | 成本高 | 灵活，可加载不同 adapter |
| 工程成熟度 | 直接但贵 | LoRA/QLoRA 很成熟 |

### PEFT 的关键超参数

以 LoRA 为例，常见关键参数包括：

- `r`：低秩矩阵的 rank，决定 adapter 容量。
- `lora_alpha`：缩放系数，实际缩放通常是 `alpha / r`。
- `lora_dropout`：adapter 分支上的 dropout，用于缓解过拟合。
- `target_modules`：把 adapter 插到哪些层，例如 `q_proj`、`v_proj`、`o_proj`、FFN 层等。
- `learning_rate`：LoRA 通常比全参数微调用更大的学习率。

核心判断：

```text
欠拟合 -> 增大 r、扩大 target_modules、增加训练步数
过拟合 -> 降低 r、增加 dropout、减少 epoch、清洗数据
显存不够 -> QLoRA、减 batch、梯度累积、gradient checkpointing
```

### PEFT 的风险

1. **容量不足**：adapter 太小，复杂任务学不动。
2. **target_modules 选错**：LoRA 没插到关键层，loss 可能下降很慢。
3. **过拟合小数据**：虽然参数少，但小数据仍然会过拟合。
4. **多 adapter 冲突**：多个任务 adapter 混用或合并时可能相互干扰。
5. **合并后效果不一致**：adapter merge 到 base model 后，需要重新验证推理输出。
6. **底座能力限制**：如果基座模型本身不会某类能力，PEFT 不一定能凭少量参数补出来。

### 什么时候选 PEFT

适合：

- 算力/显存有限，但要微调大模型。
- 同一个 base model 要服务多个任务或业务线。
- 数据量不大，希望降低灾难性遗忘风险。
- 想快速迭代多个实验版本。
- 需要只发布 adapter，不发布完整模型。

不适合：

- 任务需要大幅改变模型底层能力。
- 数据量很大且算力充足，追求极限效果。
- 需要深度修改模型结构。
- base model 和任务差距太大。

## 面试应对

### PEFT 是什么？

回答思路：先给出“冻结大模型主体、只训练少量适配参数”的定义，再强调它解决显存、存储、多任务适配和灾难性遗忘问题。

回答模板：

PEFT 是 Parameter-Efficient Fine-Tuning，也就是参数高效微调。它不是更新大模型的全部参数，而是冻结基座模型主体，只训练少量新增参数或少量模块，比如 LoRA adapter、prefix 或 soft prompt。这样可以用更低显存和更低训练成本完成任务适配，每个任务也只需要保存很小的 adapter，不用保存完整模型。它适合资源有限、多任务快速适配、数据量不大的场景，LoRA 是目前最常用的 PEFT 方法。

### PEFT 和全参数微调有什么区别？

回答思路：从更新对象、显存/存储成本、能力改动幅度和灾难性遗忘风险四个维度对比。

回答模板：

全参数微调会更新模型所有权重，能力改动幅度大，但显存、训练和存储成本都很高，而且更容易破坏预训练能力。PEFT 冻结大部分原始权重，只训练少量 adapter 或 soft prompt，显存更省、训练更快、每个任务只保存一份小 adapter，也更容易在多个任务之间切换。代价是 adapter 容量有限，如果任务需要大幅改变模型能力，PEFT 可能不如全参数微调。

### LoRA 和 PEFT 是什么关系？

回答思路：讲清 PEFT 是方法体系，LoRA 是其中最主流的一种实现，不要把二者混为同义词。

回答模板：

PEFT 是参数高效微调的一类方法，LoRA 是 PEFT 里最常用的一种。除了 LoRA，PEFT 还包括 Prefix Tuning、Prompt Tuning、Adapter Tuning、QLoRA 等。LoRA 的具体做法是冻结原始权重，只训练低秩矩阵产生的更新量 ΔW，所以它是 PEFT 思想的一种落地方式，而不是 PEFT 的全部。

### 什么时候 PEFT 可能效果不好？

回答思路：抓住 adapter 容量、target_modules、基座能力和数据质量四个限制。

回答模板：

PEFT 效果不好通常有几类原因。第一是 adapter 容量不够，比如 LoRA rank 太小，复杂任务学不动；第二是 target_modules 没选好，adapter 没插到关键层；第三是 base model 本身能力不足，少量参数很难补出新能力；第四是数据质量差或分布太窄，导致过拟合。遇到这种情况，我会先确认数据和 label 没问题，再看 loss 曲线，如果欠拟合就增大 rank、扩大 target_modules 或增加训练步数，如果仍然不行再考虑全参数微调或换更强 base model。

### QLoRA 和 LoRA 的区别是什么？

回答思路：点明 LoRA 只是低秩训练 adapter，QLoRA 进一步把冻结的基座模型低比特量化加载，从而更省显存。

回答模板：

LoRA 是冻结基座模型，只训练低秩 adapter；QLoRA 是在 LoRA 的基础上进一步把基座模型用低比特量化加载，比如 4-bit base model，然后仍然训练 LoRA adapter。这样显存占用更低，可以在更小显卡上微调更大的模型。区别在于 QLoRA 多了量化这一步，所以更省显存，但也要注意量化误差和训练稳定性。
