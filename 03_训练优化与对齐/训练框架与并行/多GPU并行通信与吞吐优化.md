# 多 GPU 并行通信与吞吐优化

## 知识点解析

### 多 GPU 训练到底在优化什么

多 GPU 训练的目标不是简单“卡越多越快”，而是在显存、计算、通信和数据加载之间找到平衡：

```text
模型放得下
  + 每张卡算得饱
  + 卡间通信少
  + 数据供得上
  + 各 Rank 负载均衡
  -> 有效吞吐提升
```

常见吞吐指标：

| 指标 | 含义 |
| --- | --- |
| samples/s | 每秒处理多少样本 |
| tokens/s | 每秒处理多少 token |
| step time | 一个训练 step 的耗时 |
| communication ratio | 通信时间占 step 的比例 |
| GPU utilization | GPU 计算单元利用率 |
| MFU | 实际 FLOPs 与理论 FLOPs 的比例 |
| scaling efficiency | 扩卡后的实际加速 / 理想加速 |

如果 8 卡吞吐是 6 卡的 1.2 倍，问题通常不是“GPU 不够”，而是通信、数据、负载或同步开销已经成为瓶颈。

### 四类并行策略

| 策略 | 切分对象 | 主要通信 | 适合场景 |
| --- | --- | --- | --- |
| 数据并行 DP/DDP | 每卡完整模型，不同数据 | 梯度 All-Reduce | 模型和状态能放进单卡 |
| FSDP/ZeRO | 参数、梯度、优化器状态 | All-Gather、Reduce-Scatter | 单卡放不下训练状态 |
| 张量并行 TP | 单层矩阵和 attention head | All-Reduce、All-Gather | 单层计算或权重过大 |
| 流水线并行 PP | 不同 Transformer 层 | 激活点对点通信 | 层数多、模型深 |
| 序列并行 SP | 序列或激活维度 | 集合通信 | 长序列激活占用大 |
| 专家并行 EP | MoE 专家 | All-to-All | MoE 模型专家分布式执行 |

实际大模型训练通常是混合并行：

```text
DP/ZeRO：
  扩大数据规模、切训练状态

TP：
  把单层矩阵切到同机多卡

PP：
  把不同层放到不同设备

EP：
  把 MoE 专家分布到不同设备
```

### 集合通信是什么

#### All-Reduce

每个 Rank 都有一份数据，通信后所有 Rank 都得到归约结果：

```text
各卡梯度
  -> 求和/平均
  -> 每卡拿到相同梯度
```

DDP 的梯度同步主要使用 All-Reduce。

#### Reduce-Scatter

先完成归约，再把结果分片给不同 Rank：

```text
各卡数据
  -> 归约
  -> 每卡只保留一片
```

FSDP/ZeRO 常用它来减少每卡保存的梯度或参数状态。

#### All-Gather

每个 Rank 贡献一片，通信后每个 Rank 得到完整张量。ZeRO-3/FSDP 在计算某层前经常需要临时 All-Gather 参数。

#### All-to-All

每个 Rank 向所有其他 Rank 发送不同数据分片。MoE 专家路由常用，但通信量和不规则性都比较高。

### 通信为什么会成为瓶颈

单个 step 的时间可以粗略拆成：

```text
step time =
data loading
  + forward
  + backward
  + collective communication
  + optimizer
  + checkpoint/eval
```

多卡后，理想情况是计算量被分摊；但通信和同步会增加：

- DP 需要同步梯度。
- TP 几乎每层都可能通信。
- PP 会产生阶段间激活传输和 bubble。
- EP 需要 All-to-All 分发 token。
- ZeRO/FSDP 需要参数 All-Gather 和梯度 Reduce-Scatter。

同步训练还有一个特点：

```text
一个 Rank 变慢
  -> 其他 Rank 等待
  -> 全局 step 被最慢 Rank 决定
```

### GPU 拓扑如何影响并行

通信路径不是等价的：

```text
同卡显存
  > 同机 NVLink/NVSwitch
  > 同机 PCIe
  > 跨机 RDMA/高速网络
  > 普通网络
```

经验：

- 高频 TP 尽量放在同机高速互联内。
- 跨机优先使用 DP 或合理的 PP，减少每层高频同步。
- 多机训练前检查 GPU 拓扑、NIC、RDMA 和 NCCL 识别情况。
- 并行组不要只按 GPU 数量划分，要按物理拓扑划分。

### 通信与计算重叠

理想训练不是：

```text
计算 -> 等计算结束 -> 通信 -> 再计算
```

而是：

```text
计算后层梯度
  -> 立即启动通信
  -> 同时继续计算前层梯度
```

常见方法：

- 梯度 Bucket 化。
- 异步 All-Reduce。
- 调整 Bucket 大小。
- 参数预取和梯度预取。
- 使用通信 stream 与计算 stream 重叠。
- 避免频繁 `.item()`、CPU/GPU 同步和隐式 barrier。

Bucket 过大，通信启动晚；Bucket 过小，通信调用太碎，启动开销高。需要通过 profile 找平衡点。

### 如何减少通信损耗

#### 1. 选对并行策略

不要用 TP 解决所有问题：

```text
模型能放下：
  优先 DP/DDP，通信模式简单。

模型状态放不下：
  用 FSDP/ZeRO，牺牲一部分通信换显存。

单层矩阵放不下：
  用 TP，尽量限制在高速互联域。

层数太深：
  用 PP，但要控制 bubble。

MoE：
  用 EP，同时优化 All-to-All 和负载均衡。
```

#### 2. 增大单次有效计算量

通信启动有固定延迟。如果每次只同步很小的 tensor，通信效率很差。可以：

- 增大 micro-batch，但不能超过显存。
- 用 gradient accumulation 减少同步频率。
- 合理设置 global batch。
- 减少过度切碎的 tensor。
- 让矩阵计算规模更大。

梯度累积的作用不是减少总数据量，而是减少 optimizer update 频率、提高单次计算/通信的比例。

#### 3. 使用通信友好的布局

- 将参数和梯度按连续大块组织。
- 使用 fused gradient buffer。
- 减少小 tensor 的单独通信。
- 避免不同 Rank 进入不同通信分支。
- 保持 collective 调用顺序一致。

#### 4. 降低需要通信的数据量

- ZeRO/FSDP 切分状态。
- 混合精度降低通信字节数。
- 梯度压缩或低精度通信需验证数值稳定性。
- 减少不必要的同步统计。
- 只在必要阶段同步指标和日志。

#### 5. 做数据和负载均衡

通信慢有时是计算慢造成的假象：

- 某个 Rank 拿到更多长样本。
- 视频帧数、token 数分布不均。
- MoE token 路由到少数专家。
- DataLoader 某个 Rank 读盘慢。

因此应使用按长度分桶、动态 batch、合理 sampler 和 MoE load balancing。

### 如何提升训练吞吐

训练吞吐低时按下面顺序排查：

```text
1. 数据是否及时到达 GPU
2. GPU 是否真正吃满
3. forward/backward 是否高效
4. 通信是否占比过高
5. 是否存在慢 Rank
6. checkpoint/eval 是否太频繁
```

#### 数据侧

- 预处理和视频解码离线化。
- 使用本地 SSD/cache，减少网络盘读取。
- 增加合理的 DataLoader worker。
- 按长度或帧数分桶。
- 减少 padding。
- 预取下一批数据。

#### 计算侧

- 使用 BF16/FP16。
- FlashAttention。
- 算子融合。
- Activation checkpointing 只在显存需要时开启。
- 增大有效矩阵计算规模。
- 检查 GPU 是否降频或出现热 throttling。

#### 通信侧

- TP 放高速互联。
- 调整 DP/TP/PP/EP 组合。
- 做通信计算重叠。
- 增大 bucket，避免小消息泛滥。
- 减少不必要的同步。
- 检查 NCCL、RDMA、NIC 和拓扑。

#### 训练流程侧

- 减少频繁 eval 和 checkpoint。
- 避免保存时聚合完整模型造成峰值。
- 只记录必要日志。
- 使用高效 checkpoint 格式。

### 如何判断瓶颈在哪里

| 现象 | 更可能的瓶颈 | 下一步 |
| --- | --- | --- |
| GPU 利用率低，DataLoader 等待高 | 数据/IO/预处理 | 缓存、worker、预取、数据本地化 |
| GPU 利用率高，但通信时间占比高 | 通信/并行策略 | 检查拓扑、TP、bucket、overlap |
| 多卡扩展后吞吐几乎不增 | 通信或同步 | 计算 scaling efficiency、Collective 占比 |
| 某一个 Rank 总是更慢 | 负载/数据/GPU 状态 | 比较各 Rank 样本长度、IO、频率、网络 |
| step 周期性变慢 | checkpoint/eval/共享存储 | 单独统计保存和评测时间 |
| MoE 吞吐不稳定 | expert load imbalance/All-to-All | 检查 token 分布和专家容量 |

必须看分阶段 profile，不能只看总 step time。

### 多卡通信故障排查

#### Hang

常见原因：

- 某个 Rank 先异常退出。
- 不同 Rank 数据量不一致。
- 不同 Rank 进入不同控制分支。
- collective 顺序不一致。
- DataLoader worker 卡住。
- NCCL/RDMA/网络异常。

排查：

```text
对齐所有 Rank 最后一条日志和 step
  -> 判断卡在数据、计算、通信还是保存
  -> 检查是否所有 Rank 都调用同一 collective
  -> 查看 NCCL 和网络日志
  -> 单卡 -> 单机多卡 -> 多机逐级复现
```

#### 慢节点

同步训练由最慢 Rank 决定。比较：

- DataLoader 等待。
- 输入 token/帧数。
- forward/backward。
- collective。
- GPU 利用率、频率、温度。
- 网络吞吐和重传。

### 多 GPU 优化的完整方法

```text
先测 baseline
  -> 拆数据/计算/通信/保存时间
  -> 确认 GPU 拓扑
  -> 选择 DP/TP/PP/EP/FSDP/ZeRO
  -> 做通信计算重叠
  -> 平衡 batch、bucket 和 micro-batch
  -> 做长度分桶和数据本地化
  -> 重新测 scaling efficiency
  -> 验证 loss、精度和稳定性不退化
```

不能只追求 samples/s。还要同时看：

```text
吞吐
  + 单步延迟
  + 通信占比
  + GPU 利用率
  + 显存
  + loss/指标
  + 故障率
```

## 面试应对

### 多 GPU 训练为什么不一定线性加速？

回答思路：计算可以分摊，但通信、同步、负载不均和固定开销会随卡数放大。

回答模板：

多 GPU 训练不一定线性加速，因为除了计算分摊，还要承担梯度同步、参数聚合、激活传输和跨卡调度。卡数增加后，如果通信时间、慢节点、数据加载或流水线 bubble 占比上升，新增 GPU 就会更多地等待而不是计算。我的做法是把 step 拆成数据、前向、反向、collective、优化器和保存分别计时，再计算 scaling efficiency，确认瓶颈后选择并行策略和通信优化。

### 如何减少多 GPU 训练的通信损耗？

回答思路：从并行策略、拓扑、通信计算重叠、消息粒度和同步频率回答。

回答模板：

我会先根据模型和 GPU 拓扑选择并行策略：模型能放下时优先数据并行，单层放不下才用张量并行，并尽量把高频 TP 通信限制在同机高速互联内；模型状态放不下时使用 FSDP 或 ZeRO；MoE 再考虑专家并行。然后通过梯度 bucket、异步 collective、参数预取让通信和计算重叠，使用连续大 buffer 减少小消息调用，并通过梯度累积降低同步频率。最后检查数据长度和专家路由是否均衡，避免某个 Rank 或专家成为慢点。

### All-Reduce、All-Gather、Reduce-Scatter、All-to-All 分别做什么？

回答思路：用“聚合、收集、归约分片、全互发”四个动作记忆。

回答模板：

All-Reduce 是各卡先把数据聚合，再让每张卡都拿到完整结果，DDP 梯度同步常用；All-Gather 是各卡贡献分片，最后每张卡都得到完整张量；Reduce-Scatter 是先聚合再把结果切片分给不同卡，FSDP 和 ZeRO 常用；All-to-All 是每张卡向所有其他卡发送不同分片，MoE 专家路由常用。理解这些 collective 的数据流，就能判断通信量、显存占用和适用并行方式。

### 训练吞吐低，你怎么定位？

回答思路：按数据、计算、通信、同步和保存拆分，不直接改 batch。

回答模板：

我会先把一个 step 拆成数据等待、前向、反向、通信、优化器和 checkpoint/eval，分别记录耗时。如果 GPU 利用率低且 DataLoader 等待高，优先查数据读取、网络盘、视频解码和预处理；如果 GPU 利用率高但 collective 占比高，检查并行策略、GPU 拓扑、NCCL 和通信计算重叠；如果只有个别 Rank 慢，比较各 Rank 的样本长度、IO、GPU 状态和网络。优化后还要重新看 scaling efficiency、loss 和业务指标，不能只报告峰值吞吐。
