# ms-swift 视频多模态训练与推理

## 知识点解析

### 这套方案做什么

当前项目使用 ms-swift 完成 Qwen3.5 视频多模态模型的训练和离线推理，整体链路是：

```text
关键帧 JSONL 数据
  -> 视频多模态 SFT
  -> 保存 checkpoint
  -> 多卡切分测试数据
  -> 每张 GPU 独立推理
  -> 合并结果
  -> 关键帧指标评测
```

当前主流程是：

```text
Direct SFT -> Infer
```

Structured CoT SFT 和 GRPO/RL 属于后续扩展方向，不是当前主流程的必经步骤。

### ms-swift 在这里负责什么

ms-swift 在这套方案中主要负责四件事：

1. 加载 Qwen3.5 多模态 base model。
2. 根据 JSONL 数据构造视频-文本训练样本。
3. 调用 PyTorch、[DeepSpeed](<../../../03_训练优化与对齐/训练框架与并行/DeepSpeed.md>) 和 FlashAttention 完成分布式训练。
4. 加载 checkpoint，对测试集批量推理并写出结果。

它本身不是关键帧算法规则的来源。完成态、排除条件、任务描述和时间标签已经在数据构造阶段写进样本，ms-swift 负责把这些数据送进模型进行训练和推理。

### SFT 训练链路

训练阶段可以抽象成：

```text
Qwen3.5 base model
  + video keyframe SFT dataset
  -> full-parameter multimodal fine-tuning
  -> checkpoint
```

当前配置的典型特点：

```text
模型：Qwen3.5-9B
训练方式：full fine-tuning
精度：BF16
资源：默认两机八卡
单卡 batch：1
梯度累积：2
epoch：1
learning rate：1e-5
warmup ratio：0.05
```

### 为什么使用 full fine-tuning

当前训练不只是调整语言输出格式，还希望模型适应：

- UI 小元素和局部视觉细节。
- 视频帧之间的状态变化。
- 不同 task_type 的完成态规则。
- “完成态第一次成立”的时间边界。
- 结构化时间输出。

因此训练时不仅更新语言模型，也允许视觉 encoder 和视觉语言对齐模块参与更新。

这和 [LoRA](<../../../03_训练优化与对齐/后训练与对齐/LoRA 低秩适配.md>) 的区别是：

```text
LoRA：
  base model 大部分冻结，只训练 adapter，成本低。

Full fine-tuning：
  语言、视觉和对齐模块都可以更新，适配能力更强，但成本和遗忘风险更高。
```

### 训练参数

#### 模型和数据

| 参数 | 当前用法 | 作用 |
| --- | --- | --- |
| `model` | Qwen3.5-9B base model | 指定初始化模型 |
| `dataset` | 关键帧 qwen3 JSONL | 提供视频、任务 prompt 和时间标签 |
| `tuner_type` | `full` | 使用全参数训练 |
| `torch_dtype` | `bfloat16` | 降低显存并保持较好的数值稳定性 |
| `num_train_epochs` | `1` | 先快速验证训练链路和数据质量 |

#### Batch 和学习率

```text
effective_batch_size =
per_device_train_batch_size
* GPU 总数
* gradient_accumulation_steps
```

当前默认是两台机器、每台八张 GPU：

```text
effective batch size = 1 * 16 * 2 = 32
```

参数含义：

- `per_device_train_batch_size=1`：每张 GPU 每次只处理一条视频样本。视频视觉 token 多，单条样本显存开销较大。
- `gradient_accumulation_steps=2`：两个 micro-batch 累积后再更新一次参数，用时间换显存。
- `learning_rate=1e-5`：全参数微调常用较小学习率，避免破坏 base model 原有能力。
- `warmup_ratio=0.05`：训练前 5% 的 optimizer steps 逐渐升高学习率，降低初期发散风险。
- `num_train_epochs=1`：关键帧数据已经经过筛选，先用一轮观察泛化和下游指标，避免过拟合。

#### 显存和速度

| 参数 | 当前配置 | 解决的问题 |
| --- | --- | --- |
| `deepspeed` | [ZeRO-3](<../../../03_训练优化与对齐/训练框架与并行/ZeRO.md>) | 分片保存参数、梯度和 optimizer state |
| `attn_impl` | FlashAttention | 降低 attention 中间张量和显存读写 |
| `gradient_checkpointing` | 开启 | 用反向重算换激活显存 |
| `vit_gradient_checkpointing` | 开启 | 同样降低视觉 encoder 的激活显存 |
| `freeze_vit` | `false` | 视觉 encoder 参与训练 |
| `freeze_aligner` | `false` | 视觉语言对齐模块参与训练 |

这几项是配套关系：

```text
视频 token 多
  + vision encoder 也训练
  + full fine-tuning
  -> 显存压力大
  -> BF16 + [ZeRO-3](<../../../03_训练优化与对齐/训练框架与并行/ZeRO.md>) + FlashAttention + 两类 checkpointing
```

#### 数据字段保留

训练时需要保留视频和多模态字段，因此使用：

```text
remove_unused_columns = false
```

否则通用 Trainer 可能把它认为“不在文本模型 forward 参数里”的视频字段或自定义字段提前删掉，导致模型拿不到视觉输入。

#### 保存和日志

当前保存和日志配置的含义：

- 按 step 定期保存 checkpoint。
- 定期记录训练 loss、学习率等指标。
- 限制 checkpoint 保留数量，避免训练结果占满磁盘。
- 使用 W&B 记录实验曲线和配置。
- 只保存模型相关产物时可以节省空间，但完整断点恢复能力会弱一些。

### 训练时模型真正看到什么

对一条关键帧样本，模型训练输入可以概括为：

```text
视频
  + human prompt
  -> Qwen3.5-VL 编码和理解
  -> 生成 assistant answer
```

其中 human prompt 包含：

- 关键帧检测任务定义。
- 完成态标准。
- 排除条件和豁免条件。
- 当前 task_type 的具体任务描述。
- 固定输出格式。

assistant 标签通常是：

```text
<answer>{"time": 12.3}</answer>
```

模型学习的是：

```text
视频视觉内容 + 业务规则
  -> 完成态第一次成立的时间
```

### 推理链路

推理阶段采用的是**数据并行式离线推理**：

```text
测试集
  -> 切成与 GPU 数量相同的多个 shard
  -> 每张 GPU 独立启动一个 swift infer
  -> 每个进程加载一份模型
  -> 分别处理自己的 shard
  -> 合并所有结果
```

它和训练阶段的分布式并行不同：

```text
训练：
  多卡共同训练一份模型。

推理：
  多卡各自加载一份模型，处理不同数据。
```

优点是实现简单、适合离线评测；缺点是每张 GPU 都需要保存一份模型副本。

### 推理参数

| 参数 | 当前用法 | 作用 |
| --- | --- | --- |
| `model` | 最新训练 checkpoint | 指定推理模型 |
| `val_dataset` | 当前数据 shard | 指定本进程处理的数据 |
| `max_new_tokens` | `2048` | 限制最大生成长度 |
| `max_batch_size` | `8` | 限制单进程推理 batch 上限 |
| `attn_impl` | FlashAttention | 降低推理 attention 开销 |
| `remove_unused_columns` | `true` | 推理时清理不参与模型输入的字段 |
| `result_path` | 当前 shard 的结果文件 | 保存原始推理结果 |

`max_batch_size=8` 是调度上限，不代表每次一定能凑够 8 条。视频长度、视觉 token 数和显存会决定实际 batch。

`max_new_tokens=2048` 对当前关键帧任务通常偏宽松，因为正常答案只需要一个短 JSON。如果模型出现长输出或重复，优先检查：

- prompt 是否明确限制输出格式。
- checkpoint 是否加载正确。
- `max_new_tokens` 是否过大。
- 推理采样参数是否和评测要求一致。

### 推理结果为什么要回填源数据

每个 shard 推理后，系统会把结果和源样本重新对齐，检查：

- 源数据和预测结果行数一致。
- 视频路径一致。
- 源标签和推理结果中的 labels 一致。
- 结果保留原始 `infos`。

最后合并为一个结果文件，方便后续计算：

```text
frame_err
time_err
PASS
按 task_type / 业务线 / 难度分桶的 ACC
```

这一步很重要，因为离线推理不只是“生成答案”，还必须保证预测结果和原始样本、视频、GT 一一对应。

### 视觉输入预算

项目中还会通过环境配置控制视频输入：

| 配置 | 典型值 | 作用 |
| --- | --- | --- |
| FPS | `10` | 视频采样帧率 |
| 最大帧数 | `300` | 限制单条视频最多输入多少帧 |
| 视频视觉 token 上限 | `128` | 控制视频视觉 token 预算 |
| 图像视觉 token 上限 | `1024` | 控制图像视觉 token 预算 |

这几个量会共同影响：

```text
FPS 越高 / 帧数越多
  -> 时间覆盖更密
  -> 视觉 token 和显存更高
  -> 训练/推理更慢
```

关键帧任务不能只追求增加 FPS，还要保证训练和评测的采样策略一致，否则模型输出时间可能和 GT 不在同一个时间坐标系。

### Structured CoT SFT 和 GRPO

当前主流程是 Direct SFT，但可以继续扩展：

```text
Direct SFT
  -> Structured CoT SFT
  -> GRPO/RL
```

#### [Structured CoT SFT](<../方案与训练流程/CoT-SFT 蒸馏方案设计.md>)

让模型先输出结构化的视频状态、事件和边界证据，再输出最终时间：

```text
视频
  -> 状态/事件证据
  -> 问题驱动判断
  -> <answer>{"time": ...}</answer>
```

它适合解决边界难例，但不建议一开始让所有样本都输出很长 CoT，否则会增加 token 成本、训练难度和格式风险。

#### [GRPO/RL](<../方案与训练流程/GRPO 训练方案设计.md>)

可以用结构化 reward 约束模型：

```text
R = R_time
  + R_format
  + R_boundary_evidence
  - P_length
```

- `R_time`：时间误差是否小。
- `R_format`：输出是否可解析。
- `R_boundary_evidence`：是否引用了正确的完成态证据。
- `P_length`：是否存在无效长输出、复读或绕路推理。

GRPO 阶段通常还需要配置采样数量、生成长度、温度、KL 约束和 reward plugin。

## 参数之间的整体关系

当前配置可以概括成：

```text
Qwen3.5 视频模型
  + full fine-tuning
  + BF16
  + 多机多卡
  + per-device batch 1
  + gradient accumulation 2
  + learning rate 1e-5
  + 视觉 encoder 和 aligner 都训练
  + ZeRO-3 / FlashAttention / checkpointing
  + 固定 FPS、最大帧数和视觉 token 预算
```

它们不是独立设置的：

- 视频样本显存大，所以单卡 batch 小。
- batch 小但希望梯度稳定，所以使用梯度累积。
- full fine-tuning 直接更新 base model，所以学习率不能过大。
- 视觉侧也训练，所以需要更强的显存优化。
- 视频帧数和 token 预算决定了训练速度、显存、时间定位精度。
- 推理阶段按数据分片并行，换取离线吞吐，但每张卡都要加载模型。

## 常见排查点

| 现象 | 优先检查 |
| --- | --- |
| 多机训练无法启动 | 主节点地址、端口、节点编号、全局进程数 |
| 训练 OOM | 视频帧数、视觉 token、batch、ZeRO-3、两类 checkpointing |
| loss 不下降 | 数据格式、`<video>` 与视频字段是否匹配、标签是否正确、full tuning 是否生效 |
| 训练初期 loss spike | 学习率、warmup、异常样本、BF16 和梯度范数 |
| 推理输出很长 | 输出 prompt、`max_new_tokens`、checkpoint 和采样参数 |
| 推理结果行数不一致 | shard 是否完整、推理进程是否失败、结果文件是否截断 |
| 结果和源数据错位 | 视频路径、样本顺序、labels 和 infos 是否保持一致 |
| GPU 利用率低 | 视频解码 IO、DataLoader、视觉 token、推理 batch |
| 训练和推理效果不一致 | tokenizer、chat template、视频采样、FPS、max frames 和模型 checkpoint |

## 面试应对

### 你的 ms-swift 视频训练和推理链路是什么？

回答思路：先讲 Direct SFT，再讲 checkpoint、分片推理和结果合并。

回答模板：

我使用 ms-swift 对 Qwen3.5 视频模型做全参数多模态 SFT。训练时输入是视频加关键帧任务 prompt，监督目标是完成态第一次成立的时间 JSON。配置上使用 BF16、ZeRO-3、FlashAttention 和 gradient checkpointing 来控制显存；单卡 batch 较小，再通过梯度累积得到有效 batch。训练完成后找到 checkpoint，把测试集按 GPU 数量切成多个 shard，每张 GPU 独立启动 `swift infer` 处理一份数据，最后合并结果，并校验视频、labels 和原始样本是否一一对应。

### 为什么使用 full fine-tuning，而不是 LoRA？

回答思路：结合视觉细节、时间边界和多模态对齐回答。

回答模板：

这个任务不只是让模型学习一种输出格式，还要适应 UI 小元素、视频状态变化、task_type 完成态和时间边界，所以希望语言模型、视觉 encoder 和对齐模块都能更新，使用 full fine-tuning。它的代价是显存和训练成本更高，也有通用能力退化风险，因此需要 BF16、ZeRO-3、FlashAttention 和 checkpointing。LoRA 更省资源，适合先做快速验证或只调整任务格式；如果视觉和时间能力需要较大幅度适配，再考虑全参数训练。

### 有效 batch size 怎么计算？

回答思路：给出公式，并强调多机多卡要使用全局 GPU 数。

回答模板：

有效 batch size 等于单卡 batch 乘以全局 GPU 数，再乘以梯度累积步数。当前典型配置是两台机器、每台八张卡，单卡 batch 为 1，梯度累积为 2，所以有效 batch size 是 `1×16×2=32`。如果机器数、GPU 数或梯度累积发生变化，需要重新计算，学习率和训练稳定性也要一起观察。

### 为什么要使用 ZeRO-3、FlashAttention 和 gradient checkpointing？

回答思路：分别对应训练状态、attention 中间张量和激活显存。

回答模板：

ZeRO-3 把参数、梯度和 optimizer state 分片到多张卡，降低单卡保存完整训练状态的压力；FlashAttention 减少 attention 中间结果和显存访问；gradient checkpointing 不保存所有中间激活，反向时重算来换显存。视频多模态 full fine-tuning 同时有视觉 token、语言 token 和视觉 encoder 参数，这些优化需要组合使用。代价是通信和重算会增加训练时间。

### 为什么推理阶段要把数据切 shard？

回答思路：区分训练分布式并行和推理数据并行。

回答模板：

离线推理时，我把测试集切成多个 shard，每张 GPU 独立加载一份模型，分别处理不同样本。这是数据并行推理，不是多卡共同执行一个模型。它实现简单、容易横向扩展，适合批量评测；代价是每张卡都需要加载模型副本。所有 shard 完成后还要做行数、视频路径和 labels 校验，最后再合并，避免结果和 GT 错位。
