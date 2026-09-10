# 关键帧检测：SFT 训练方案设计

## 知识点解析

### 概述

关键帧检测的 SFT 分为两层：Direct SFT 负责建立视频输入、任务遵循、完成态识别和短答案格式；Structured CoT SFT 负责在已有基础上补充目标区域、时间证据和首次完成边界。SFT 的目标不是单纯降低 loss，而是让训练目标、输出协议和下游 ACC 保持一致。训练时应遵循“单样本链路验证 -> 小数据过拟合 -> 小规模实验 -> 完整训练 -> 分桶评测”的顺序。

### 1. Direct SFT 解决什么问题

Direct SFT 的训练目标：

```text
视频 + task_type + 完成态规则
  -> <answer>{"time": ...}</answer>
```

主要建立：

- 视频输入和 processor 链路。
- task_type 和业务规则遵循。
- 明显完成态识别。
- 时间点预测。
- 稳定的结构化输出。

它不能充分表达：

```text
前一候选为什么不满足
当前候选为什么第一次满足
后续为什么只是稳定延续
```

这些能力属于后续 Structured CoT SFT 的重点。

### 2. 冷启动流程

#### 2.1 环境检查

- 模型、tokenizer、processor 可以加载。
- 视频可以解码。
- `<video>` 与 `videos[0]` 对应。
- BF16、FlashAttention、ZeRO/FSDP 等依赖正常。
- 训练和推理的 chat template 一致。

#### 2.2 单样本 forward

确认：

```text
视频读取
  -> prompt 构造
  -> visual inputs 生成
  -> forward 正常
  -> loss 为有限值
```

单样本失败时先检查输入、labels、设备和 dtype，不要先调学习率。

#### 2.3 小数据过拟合

使用 10~100 条样本验证：

- train loss 是否明显下降。
- 模型是否记住样本。
- assistant labels 是否参与 loss。
- 输出是否符合 schema。

如果小数据都无法过拟合，常见问题是：

- labels 全被 mask。
- `<video>` 与视频字段不匹配。
- 数据被截断。
- 参数没有真正参与训练。
- 视觉输入没有进入 forward。

#### 2.4 小规模正式实验

固定一小份 train/dev，训练少量 steps 或短周期，检查：

```text
loss
格式解析率
time/frame error
PASS
显存和吞吐
```

确认链路可用后，再训练完整数据。

### 3. SFT 参数如何设置

#### 3.1 Learning Rate

全参数微调建议从较小范围开始：

```text
5e-6 / 1e-5 / 2e-5
```

观察：

- 初期 loss 是否 spike。
- train/dev loss 是否同步。
- 下游 ACC 是否提升。
- 输出格式是否退化。

LoRA 可以使用更大学习率，但不能把 LoRA 经验直接套到 full fine-tuning。

#### 3.2 Effective Batch Size

```text
effective_batch_size =
  per_device_batch_size
  * global_gpu_count
  * gradient_accumulation_steps
```

显存不足时：

```text
降低 per-device batch
  -> 增加 gradient accumulation
  -> 尽量维持 effective batch
```

#### 3.3 Warmup

常见起点：

```text
warmup_ratio = 0.03~0.1
```

如果训练初期 loss spike，可以增加 warmup；如果总 steps 很少，warmup 不能占比过高。

#### 3.4 Epoch 和 checkpoint

不能默认最后一个 checkpoint 最好：

| 现象 | 判断 |
| --- | --- |
| train loss 降、dev 指标升 | 可以继续 |
| train loss 降、dev loss 升 | 过拟合 |
| train/dev 都不降 | 输入、标签、学习率或模型能力问题 |
| loss 降、ACC 不升 | 训练目标和业务目标不一致 |

每隔固定 steps 保存 checkpoint，用 dev、困难集和分桶 ACC 选择最佳版本。

### 4. 视频输入预算

FPS、最大帧数和视觉 token 预算共同决定：

```text
时间覆盖能力
  vs
显存、吞吐和上下文成本
```

低 FPS 可能错过边界，高 FPS 可能导致显存和推理成本上升。关键帧任务适合：

```text
低 FPS 全局粗定位
  -> 候选窗口高 FPS 精筛
```

训练和评测必须使用一致的：

- FPS。
- 视频裁剪方式。
- 最大帧数。
- 分辨率和视觉 token 预算。
- 时间坐标定义。

### 5. 根据训练结果定位问题

#### Loss 不下降

```text
检查数据格式
  -> 检查 labels
  -> 检查 <video> 和 videos
  -> 检查 trainable 参数
  -> 检查 learning rate
```

#### Loss spike 或 NaN

优先检查：

- 学习率过大。
- warmup 太短。
- 异常 batch。
- BF16/FP16 溢出。
- 梯度爆炸。
- 空标签或损坏视频。

#### Train loss 降，Dev loss 升

通常是：

- 过拟合。
- train/dev 分布不一致。
- 训练集重复或标签冲突。
- dev 标注噪声更大。

#### Loss 变好，ACC 不变

重点检查：

- loss 主要落在格式 token，而不是答案 token。
- GT 时间噪声。
- Prompt 规则与评测规则不一致。
- FPS 和时间坐标不一致。
- 模型学会格式但没有学会边界。

#### 输出格式不稳定

优先检查：

- train/infer chat template。
- assistant target 是否统一。
- 是否混入自然语言、Markdown 或多种 XML schema。
- `max_new_tokens` 和采样参数。
- checkpoint 是否加载正确。

### 6. Direct SFT 与 Structured CoT SFT

| 阶段 | 输入 | 输出 | 主要目标 |
| --- | --- | --- | --- |
| Direct SFT | 视频 + 任务规则 | 短时间 JSON | 基础能力和稳定格式 |
| Structured CoT SFT | 视频 + 任务规则 | 状态/事件/边界 + 时间 | 过程证据和边界判断 |

不建议让两种 target 在相同 prompt 下裸混。若必须联合训练，应显式增加：

```text
output_mode=direct
output_mode=structured_cot
```

并分别评测两种模式。

### 7. 相关工程卡片

- [Qwen3-VL关键帧数据格式.md](<../工程实现/Qwen3-VL关键帧数据格式.md>)
- [ms-swift关键帧训练与推理.md](<../工程实现/ms-swift关键帧训练与推理.md>)
- [关键帧检测完整训练方案.md](<关键帧检测完整训练方案.md>)

## 面试应对

### 关键帧任务的 SFT 怎么冷启动？

回答模板：

我会先做环境和单样本 forward 检查，确认视频能解码、`<video>` 与 `videos[0]` 对齐、assistant labels 参与 loss。然后用 10 到 100 条样本做过拟合测试，验证 loss、输出格式和多模态输入链路。小样本通过后，再用固定 train/dev 做短周期实验，观察 loss、格式解析率、时间误差和 ACC，最后才训练完整数据并按 dev 和困难集选择 checkpoint。Direct SFT 先学习短答案和基础完成态，Structured CoT SFT 再补充边界证据。

### 为什么不能只看 train loss？

回答模板：

train loss 只能说明模型是否在拟合训练 target，不能直接说明关键帧时间是否准确。如果标签存在噪声、loss 主要落在格式 token，或者模型学会了输出模板，train loss 都可能下降但 ACC 不提升。因此我会同时看 dev loss、格式解析率、frame/time error、PASS、分 task_type ACC、early/late 分布和 bad case。
