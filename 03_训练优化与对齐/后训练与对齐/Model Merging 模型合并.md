# Model Merging 模型合并

## 知识点解析

### 概述

Model Merging（模型合并）是在**不重新训练或只用很少额外计算**的情况下，把多个模型或多个 adapter 的能力融合到一个模型里的技术。它常用于把不同任务微调得到的模型、不同 LoRA adapter、不同 checkpoint 或不同能力模型合并成一个统一模型。

一句话：

```text
Model Merging = 不再跑完整训练，而是在权重空间里组合多个模型的能力
```

它不是 ensemble。Ensemble 是推理时同时跑多个模型再融合输出；Model Merging 是把权重先合成一个模型，推理时仍然只跑一个模型。

### 为什么需要 Model Merging

在大模型工程里，经常会出现多个专项模型：

- 一个模型数学能力强。
- 一个模型代码能力强。
- 一个模型中文风格好。
- 一个模型遵循业务格式更稳。
- 一个 LoRA adapter 专门学工具调用。
- 一个 LoRA adapter 专门学某个业务数据。

如果每个能力都单独部署一份模型，成本高、路由复杂、维护困难。Model Merging 希望把这些能力合到一个模型里，在不增加推理成本的情况下获得更均衡的能力。

### Model Merging 和继续训练的区别

| 维度 | 继续训练 | Model Merging |
| --- | --- | --- |
| 核心方式 | 继续跑梯度更新 | 直接组合已有权重或 adapter |
| 是否需要训练数据 | 需要 | 通常不需要，或只需少量校验数据 |
| 计算成本 | 高 | 低 |
| 能力融合方式 | 通过训练学习 | 通过权重空间组合 |
| 风险 | 过拟合、遗忘、训练不稳定 | 权重冲突、能力互相抵消、合并后退化 |
| 推理成本 | 一个模型 | 一个模型 |

### Model Merging 和 Ensemble 的区别

Model Merging 和 Ensemble 都能融合多个模型能力，但位置不同：

```text
Ensemble：推理阶段融合多个模型输出
Model Merging：训练后/部署前融合多个模型权重
```

Ensemble 的优点是效果通常更稳，但推理成本是多个模型叠加；Model Merging 的优点是推理成本仍然是一个模型，但合并效果依赖模型是否兼容、权重是否冲突。

### 常见合并对象

#### 全模型权重合并

多个模型必须通常来自**同一个 base model**，结构完全一致，才能直接做权重合并。

例如：

```text
base model
  -> math SFT model
  -> code SFT model
  -> chat SFT model

merge(math, code, chat) -> merged model
```

#### LoRA / Adapter 合并

多个 LoRA adapter 可以先合并 adapter，也可以分别 merge 到 base 后再合并模型。

常见场景：

```text
base model + LoRA_A + LoRA_B + LoRA_C -> merged adapter / merged model
```

这比全模型合并更常见，因为 adapter 小、版本多、方便实验。

#### Checkpoint Averaging

把同一次训练中不同 step 或不同 epoch 的 checkpoint 做平均，常用于提升稳定性。

它更像训练过程中的平滑：

```text
checkpoint_1000
checkpoint_1500
checkpoint_2000
  -> averaged checkpoint
```

### 常见方法

#### Weight Averaging

最简单的方法是权重平均：

```text
W_merge = α * W1 + (1 - α) * W2
```

如果多个模型：

```text
W_merge = α1 * W1 + α2 * W2 + α3 * W3
```

适合模型来源接近、训练目标不冲突的情况。优点是简单，缺点是容易把不同任务的有效更新互相冲掉。

#### Task Arithmetic

Task Arithmetic 先计算每个任务模型相对 base model 的“任务向量”：

```text
task_vector = W_task - W_base
```

然后把多个任务向量加到 base 上：

```text
W_merge = W_base + α1 * ΔW_task1 + α2 * ΔW_task2
```

直观理解：每个微调模型相对 base 的变化量，表示它为某个任务学到的方向；合并时就是把这些能力方向组合起来。

#### TIES-Merging

TIES 关注的是多个 task vector 之间的冲突。它的大致思路是：

1. 只保留重要的权重变化。
2. 对更新方向做符号投票，减少正负方向相互抵消。
3. 再把保留下来的更新合并。

适合多个任务更新方向存在冲突的场景。

#### DARE

DARE 的思路是对 task vector 做随机稀疏化：丢掉一部分权重变化，再对保留部分做缩放，目的是减少不同任务之间的干扰。

直观理解：不是把所有更新都硬塞进 merged model，而是减少冗余和冲突更新，让关键变化更突出。

#### SLERP

SLERP（Spherical Linear Interpolation）是在球面上插值两个模型权重，而不是简单线性平均。

它常用于两个模型之间的平滑合并，尤其当希望保留权重向量方向和范数结构时，比普通线性插值更稳一些。

### 合并前提

Model Merging 不是任意模型都能合。

通常要求：

1. **结构一致**：层数、hidden size、attention heads、tokenizer、vocab 等要一致。
2. **最好同源 base**：多个模型来自同一个基座，合并成功率最高。
3. **训练目标不要强冲突**：一个强格式模型和一个强开放生成模型可能互相拉扯。
4. **权重尺度要接近**：不同训练强度、不同 LR、不同数据规模可能导致 task vector 尺度差异很大。
5. **必须做合并后评测**：不能只看平均分，要看分任务指标和 bad case。

### 常见风险

1. **能力互相抵消**：数学提升了，代码掉了，或者格式能力变差。
2. **风格冲突**：一个模型回答简洁，另一个模型回答啰嗦，合并后风格不稳定。
3. **安全/拒答策略被冲淡**：安全对齐模型和能力模型合并后，安全边界可能变弱。
4. **tokenizer / config 不一致**：结构不一致会直接不能合，或合完不可用。
5. **平均分掩盖子集退化**：总分提升但关键业务子集下降。
6. **合并比例不可解释**：alpha 随便调可能碰巧有效，但缺少可复现规律。

### 实操建议

一个稳妥流程：

1. 先确认所有模型来自同一个 base，tokenizer 和 config 完全一致。
2. 准备分能力评测集，例如数学、代码、格式遵循、安全、业务关键集。
3. 从简单权重平均或 LoRA adapter 合并开始。
4. 扫几个合并比例，例如 `0.2/0.8`、`0.5/0.5`、`0.8/0.2`。
5. 如果任务冲突明显，再尝试 TIES、DARE、SLERP。
6. 每次合并后做分桶评测和 bad case diff。

判断是否值得上线，不看“平均分是否涨一点”，而看：

```text
目标能力是否提升
关键能力是否不掉
安全/格式/业务底线是否稳定
推理输出是否和预期一致
```

## 面试应对

### Model Merging 是什么？

回答思路：先和 ensemble 区分，强调它是在权重空间合并多个模型/adapter，推理时仍然只跑一个模型。

回答模板：

Model Merging 是把多个模型或多个 adapter 的权重融合成一个模型，用来组合不同微调模型的能力。它和 ensemble 不一样：ensemble 是推理时同时跑多个模型再融合输出，推理成本更高；Model Merging 是部署前把权重合成一个模型，推理时仍然只跑一个模型。它常用于把数学、代码、对话、业务格式等不同专项能力融合到一个统一模型里。

### Model Merging 为什么最好要求同一个 base model？

回答思路：抓住权重空间可对齐这个前提，同源 base 的参数位置和语义更一致，task vector 才有可加性。

回答模板：

Model Merging 最好要求模型来自同一个 base，因为合并本质是在权重空间做加权或组合。如果两个模型结构、tokenizer 或初始化来源不同，同一个位置的参数不一定表示同一种语义，直接平均就没有意义。同源 base 下，微调模型相对 base 的变化量更像“任务向量”，这些变化方向才更可能被相加和组合。

### Task Arithmetic 是什么？

回答思路：讲清“先减 base 得到任务向量，再把多个任务向量加回 base”的核心公式。

回答模板：

Task Arithmetic 是一种模型合并方法。它先把任务模型减去 base model，得到这个任务带来的权重变化量，也就是 task vector：`ΔW = W_task - W_base`。合并时再把多个任务向量加回 base：`W_merge = W_base + α1ΔW1 + α2ΔW2`。直观上，每个 task vector 表示模型为了某个任务学到的能力方向，合并就是把这些方向组合起来。

### Model Merging 有哪些风险？

回答思路：从权重冲突、能力抵消、安全边界变弱和平均分掩盖子集退化四点讲。

回答模板：

Model Merging 的主要风险是不同任务的权重更新方向可能冲突，合并后能力会互相抵消，比如数学涨了但代码掉了，或者格式遵循变差。另一个风险是安全和拒答策略被冲淡，尤其是把能力模型和安全对齐模型合并时。还有一个常见问题是平均分掩盖关键子集退化，所以合并后一定要做分任务评测、关键业务集评测和 bad case diff，不能只看总分。

### Model Merging 和继续训练怎么选？

回答思路：对比成本和可控性：合并便宜快速但不保证学到新能力，继续训练成本高但能用数据明确优化目标。

回答模板：

如果我已经有多个同源微调模型或多个 LoRA adapter，想低成本融合能力，我会先尝试 Model Merging，因为它不需要重新跑完整训练，推理时也还是一个模型。但如果目标能力缺口很大、多个模型冲突明显，或者需要严格优化某个指标，继续训练更可控，因为它可以用明确数据和 loss 去学习。简单说，Model Merging 适合低成本能力组合，继续训练适合明确目标下的能力学习。

### 合并 LoRA adapter 时要注意什么？

回答思路：强调 base/tokenizer 一致、target_modules 兼容、合并比例要扫、merge 后必须重新评测。

回答模板：

合并 LoRA adapter 时，首先要确认它们基于同一个 base model，tokenizer 和模型结构一致，target_modules 也兼容。然后要扫合并比例，因为不同 adapter 的更新幅度可能不一样，不能简单默认等权。合并完成后必须重新跑评测，尤其看各任务子集、格式遵循、安全边界和 bad case。adapter 合并不是把能力无损相加，冲突时可能出现某个能力提升但另一个能力退化。
