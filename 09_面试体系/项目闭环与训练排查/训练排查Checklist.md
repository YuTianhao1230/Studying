# 训练排查Checklist

## 使用方式

这个文件用于面试前快速背排查框架，也可以用于真实训练任务排障。回答训练问题时，不要一上来就猜原因，先按 checklist 缩小范围。

## 总体排查顺序

### 第一步：确认现象

- 问题是什么：loss spike、NaN、OOM、hang、指标下降、吞吐下降、线上离线不一致。
- 什么时候发生：第几个 step、哪个 epoch、哪个 checkpoint、哪个版本之后。
- 是否可复现：固定 seed、小样本、单卡是否能复现。
- 影响范围：全量任务、某个数据集、某个 task_type、某个 rank、某个线上流量段。

### 第二步：确认最近变更

- 代码 commit 是否变了。
- 数据版本是否变了。
- 模型基座或 tokenizer 是否变了。
- 超参数是否变了。
- 依赖环境或镜像是否变了。
- 资源、GPU 型号、并行策略是否变了。
- 评测脚本、parser、阈值是否变了。

### 第三步：分层定位

| 层级 | 检查项 |
| --- | --- |
| 数据 | 样本量、空样本、坏文件、长度分布、label、mask、重复、泄漏、分布漂移 |
| 模型 | config、tokenizer、adapter、特殊 token、checkpoint 加载 |
| 优化 | learning rate、warmup、scheduler、weight decay、batch size、gradient accumulation |
| 精度 | fp16/bf16、loss scale、NaN/Inf、softmax/log/exp 稳定性 |
| 分布式 | rank 日志、sampler、drop_last、barrier、NCCL、all-reduce/all-to-all |
| 评测 | parser、指标阈值、分桶逻辑、无效样本过滤、线上离线一致性 |
| 服务 | 下载、预处理、推理、后处理、队列、超时、fallback、监控 |

## 具体问题 Checklist

### Loss spike

优先检查：

- spike step 对应 batch 的样本 ID。
- batch 中是否有异常长序列、坏视频、空 label。
- learning rate 当前值是否异常。
- gradient norm 是否突然变大。
- loss scale 是否频繁 overflow。
- 是否刚好发生 checkpoint save/eval/数据 shard 切换。

回答模板：

如果 loss spike，我会先定位 spike 的 step 和 batch，回放这批数据，确认是否存在坏样本、异常长度或 label 问题。然后检查 learning rate、warmup、gradient norm 和混合精度 loss scale。如果 spike 只和特定样本相关，优先隔离数据；如果全局持续震荡，更可能是优化参数或精度设置问题。

### Loss NaN

优先检查：

- 输入 tensor 是否已有 NaN/Inf。
- label 是否越界。
- mask 是否全 0。
- 自定义 loss 是否有除零、log(0)、exp overflow。
- fp16 是否溢出，bf16/fp32 是否正常。
- learning rate 是否过大。

回答模板：

NaN 排查要找第一次出现 NaN 的位置。我会在数据输入、模型输出、loss 前后和梯度上加检查，判断 NaN 是数据带来的、前向计算产生的，还是 backward 后出现的。然后临时切 bf16/fp32、降低 learning rate、加 gradient clipping，验证是否是数值稳定性问题。

### OOM

优先检查：

- OOM 发生在 forward、backward、optimizer step 还是 eval。
- batch size、max length、max frames、max pixels。
- 是否保存了带计算图的 tensor。
- 是否开启 gradient checkpointing。
- 是否使用 bf16。
- optimizer state 是否过大。
- ZeRO/FSDP 是否正确启用。

回答模板：

OOM 不能只说减 batch。我会先定位 OOM 阶段。如果 forward OOM，多半和输入规模有关；backward OOM 多半和 activation 有关；optimizer step OOM 则可能是 optimizer state。处理手段包括减 batch、梯度累积、降低视频帧数或分辨率、开启 activation checkpointing、bf16、ZeRO/FSDP，以及检查日志里是否保存了未 detach 的 tensor。

### 训练很慢

优先检查：

- GPU utilization 是否低。
- DataLoader 是否成为瓶颈。
- 网络盘 IO 是否慢。
- 图片/视频解码是否慢。
- batch 长度是否过长。
- checkpoint/eval 频率是否太高。
- 多机通信是否耗时。

回答模板：

训练慢要先拆耗时。如果 GPU 利用率低，优先看数据加载、网络盘 IO、视频解码和 CPU 预处理；如果 GPU 利用率高但 step time 长，再看模型计算和序列长度；分布式场景还要看通信。优化手段包括预处理缓存、增加 DataLoader worker、数据本地化、动态 batching、减少无效 eval/checkpoint 和优化并行策略。

### 指标不涨

优先检查：

- 训练目标和评测指标是否一致。
- 输出格式是否被 parser 正确解析。
- 验证集 GT 是否可信。
- 指标是否被少数大类主导。
- 分桶后哪些 task_type 没提升。
- 是否出现过拟合或能力遗忘。

回答模板：

指标不涨时，我会先分桶看，而不是只看平均值。比如按业务线、task_type、样本长度、难度和数据来源拆开，找到真正没提升的部分。然后抽 bad case 判断是模型不会、数据噪声、标注标准冲突还是 parser 错。如果训练 loss 降但指标不涨，通常说明训练目标和评测目标不完全一致，或者模型学到了格式但没学到关键能力。

### 线上离线不一致

优先检查：

- 同一线上请求能否离线复放。
- 线上和离线的 prompt 是否一致。
- 抽帧 fps、resize、max_frames 是否一致。
- checkpoint、tokenizer、adapter 是否一致。
- parser 和后处理是否一致。
- 线上流量分布是否不同。
- 延迟、超时、fallback 是否影响结果。

回答模板：

线上离线不一致时，我会先拿同一批线上请求离线复放。如果同一输入输出不同，说明是模型版本、预处理、prompt 或后处理链路不一致；如果同一输入输出一致，但整体指标不同，说明线上流量分布或业务反馈口径和离线评测集不同。关键是先对齐同一请求，再谈分布差异。

### 分布式 hang

优先检查：

- 哪个 rank 最先异常。
- 各 rank step 是否一致。
- 是否有 rank 数据读取失败。
- DistributedSampler 是否配置正确。
- drop_last 是否导致 batch 数不一致。
- checkpoint save 是否 barrier 不当。
- NCCL 日志是否有通信错误。

回答模板：

分布式 hang 的核心是找第一个异常 rank。我会先看所有 rank 的日志和 step，判断是否有某个 rank 数据加载失败或提前退出。如果 rank 数和 batch 数不一致，就可能卡在 collective。然后开启 NCCL debug，看是通信卡住还是数据卡住。必要时缩小到单机多卡或单卡复现，把数据、通信和 checkpoint 保存逐层排除。

## 项目场景化回答

### 关键帧项目里怎么用这个 checklist？

回答模板：

关键帧项目里，如果某轮训练指标异常，我会先看数据版本和评测版本是否一致，再看分业务线和分 task_type 指标。如果只是某些指标下降，我会抽 bad case 看是否是标注标准变化或 GT 错误。如果所有指标都下降，再查训练配置、基座模型、fps、max_frames 和 checkpoint。上线问题则先复放同一请求，对齐图片下载、抽帧、prompt、模型版本和 parser。

### Syner-Attack 项目里怎么用这个 checklist？

回答模板：

Syner-Attack 里如果 ASR 异常，我会先确认攻击样本生成协议是否一致，包括 epsilon、迭代步数、源模型、文本替换比例和随机种子。然后看评测协议是否一致，例如 prompt 模板、CLIP-proxy 阈值、API 重试和样本过滤。若某个 target model 上效果异常，再分 image-only、text-only、image-and-text 看是哪条攻击分支失效，并检查语义保持和防御设置是否改变。

