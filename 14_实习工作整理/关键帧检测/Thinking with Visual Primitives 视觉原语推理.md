# Thinking with Visual Primitives：视觉原语推理

## 知识点解析

### 概述

《Thinking with Visual Primitives》提出了一种面向多模态推理的视觉原语方法：把 point、bounding box 等空间标记从最终答案或事后校验信号，提升为推理过程中的“最小思维单元”。论文认为，多模态模型除了存在“看不清”的 Perception Gap，还存在“指不准”的 Reference Gap：模型可能已经看到了目标，但自然语言中的“这个物体”“左边那个”“它”无法稳定指向复杂场景中的同一个视觉实体，导致后续推理发生引用漂移和逻辑崩溃。论文通过视觉原语预训练、专项 SFT、基于格式/质量/答案的 GRPO、统一 RFT 和 OPD，训练模型在思考过程中直接引用空间坐标。对关键帧检测项目的核心启发是：把视觉区域、时间点和状态变化绑定成可验证证据，让 CoT 从“描述视频”升级为“引用证据并完成边界判断”。

论文中的方法主要面向图像空间推理，但其思想可以迁移到关键帧视频任务：

```text
视觉原语：
  point / bounding box

关键帧项目的迁移：
  Region Primitive + State Primitive + Event Primitive
  + frame/time reference
  + before/current/after boundary evidence
```

### 资料与相关方案

- 本地论文：[Lu 等 - Thinking with Visual Primitives.pdf](<Lu 等 - Thinking with Visual Primitives.pdf>)
- 主方案：[CoT蒸馏与RL方案.md](<CoT蒸馏与RL方案.md>)
- 相关知识：[Qwen千问架构.md](<../../02_大模型/模型细节/Qwen千问架构.md>)

### 1. 论文要解决的问题

#### 1.1 Perception Gap：看不清

Perception Gap 指模型没有获得足够的视觉细节，例如：

- 输入分辨率过低。
- 小目标、文字或局部区域被压缩。
- 高分辨率图像产生大量视觉 token，推理成本过高。
- 视觉编码器没有保留足够的局部特征。

解决这类问题通常会使用：

```text
更高分辨率
  + 动态分辨率
  + 局部 crop
  + 多尺度视觉特征
  + 更高的视觉 token 预算
```

#### 1.2 Reference Gap：指不准

Reference Gap 是论文的核心问题。它不是模型完全看不到目标，而是模型无法在语言推理过程中稳定、无歧义地指向目标。

例如，复杂场景中模型可能需要判断：

```text
哪个灰色物体？
左边第三个目标是否满足条件？
前面提到的那个物体是否和另一个物体大小相同？
这条路径在交叉点之后连接到哪个终点？
```

如果只用自然语言描述，推理链容易出现：

- 同一个词在不同步骤指向不同对象。
- “这个”“那个”“左边目标”等指代漂移。
- 漏掉某个对象或重复统计同一对象。
- 已经找到证据，但后续推理无法继续引用。
- 空间关系复杂时出现 cascading hallucination。

论文的判断是：

```text
视觉推理不应只是“先看图，再用语言思考”；
视觉对象的空间标记也应该直接参与思考。
```

### 2. Visual Primitive 是什么

论文把视觉原语定义为空间参考的最小单位，主要包括 bounding box 和 point。

#### 2.1 Bounding Box

Bounding box 用矩形框表示对象的空间范围：

```text
<|ref|>object<|/ref|>
<|box|>[[x1, y1, x2, y2]]<|/box|>
```

论文中坐标通常被归一化为 `0` 到 `999` 的离散整数。多个对象可以输出多个框，并按约定顺序排列。

Bounding box 的优点：

- 空间范围明确。
- 比单点包含更多几何信息。
- 适合对象检测、计数、属性筛选和区域关系判断。
- 标注相对确定，便于自动校验。
- 一个 box 可以由左上角和右下角两个 point 表示，因此具有较好的表示覆盖能力。

#### 2.2 Point

Point 用坐标表示一个位置或轨迹：

```text
<|point|>[[x1, y1], [x2, y2], ...]<|/point|>
```

Point 更适合：

- 目标中心位置。
- 路径和轨迹。
- 迷宫搜索过程。
- 抽象拓扑关系。
- 需要连续记录运动过程的任务。

Point 的缺点是标注歧义更大：一个对象内部许多点都可能是合法指代位置，遮挡时还可能出现点落在前景遮挡物上的问题。

#### 2.3 为什么 Box 是更好的预训练起点

论文更偏向先大规模训练 bounding box，原因包括：

1. **确定性更好**：矩形框的边界相对明确，点标注可能有多个合理位置。
2. **信息更丰富**：box 同时表达位置、宽度和高度。
3. **任务泛化更广**：box 可以退化为中心点或两个角点。
4. **更容易自动校验**：可以检查是否覆盖目标、是否严重截断、是否覆盖整张图。

### 3. Thinking with Visual Primitives 的核心范式

#### 3.1 从“事后 grounding”变成“过程 grounding”

普通视觉 grounding 通常是：

```text
模型先用自然语言思考
  -> 最后输出 box
  -> 用 box 做答案验证
```

论文的方法是：

```text
模型边思考边引用 box/point
  -> 每个判断步骤都有空间锚点
  -> 最终答案建立在可追踪的视觉证据上
```

核心变化：

| 方式 | 视觉原语的角色 |
| --- | --- |
| 传统 CoT | 主要是最终输出或事后验证 |
| Visual Primitive CoT | 推理过程中的引用、比较、筛选和路径记录 |

#### 3.2 典型推理结构

以细粒度计数为例：

```text
1. Intent Analysis：
   明确要数什么，哪些属性是筛选条件。

2. Grounding：
   用 box 找出所有候选对象。

3. Filtering：
   排除类别相同但属性不符合的 hard negative。

4. Aggregation：
   对剩余视觉原语进行统计。

5. Answer：
   输出最终数量和简短结论。
```

以空间关系判断为例：

```text
1. 定位参照物。
2. 用 box/point 绑定参照物。
3. 定位候选对象。
4. 比较属性、位置、大小或关系。
5. 输出判断。
```

以迷宫为例：

```text
1. 定位起点和终点。
2. 用 point 表示当前位置。
3. 沿可行路径探索。
4. 遇到死路后回溯。
5. 记录完整轨迹。
6. 输出可达性和最终路径。
```

### 4. 论文模型与视觉 token 压缩

论文模型采用类似 LLaVA 的视觉语言结构：

```text
Image
  -> DeepSeek-ViT
  -> 视觉 token
  -> DeepSeek-V4-Flash MoE LLM
  -> 语言和视觉原语交错的推理输出
```

论文中使用的语言骨干是 DeepSeek-V4-Flash：

- 总参数量约 `284B`。
- 推理时激活参数约 `13B`。
- 采用 MoE 结构。

视觉侧的主要压缩路径：

```text
原始图像
  -> 14 x 14 patch embedding
  -> 3 x 3 空间 token 压缩
  -> Compressed Sparse Attention 压缩 KV cache
```

论文给出的示例中，`756 x 756` 图像经过处理后大致经历：

```text
2916 个 ViT patch token
  -> 324 个输入 LLM 的视觉 token
  -> 81 个 KV cache entries
```

整体压缩比例约为 `7056x`。这说明论文的目标不是用无限视觉 token 解决问题，而是同时优化：

```text
视觉信息是否足够
  + 推理过程是否能准确引用
  + 视觉 token 和 KV cache 是否高效
```

对关键帧项目的启发是：视觉原语不必意味着无限增加图片分辨率和输出长度。更合理的方向是让模型只保留任务相关区域、候选时间窗口和边界附近的高价值证据。

### 5. 视觉原语预训练

#### 5.1 数据来源与质量治理

论文从互联网中收集 box grounding 数据，并使用自动化质量过滤：

```text
约 97,984 个 box grounding 数据源
  -> 语义质量审核后保留约 43,141 个
  -> 几何质量审核后保留约 31,701 个
  -> 每个类别最多采样约 1,000 张
  -> 最终得到 4,000 万级高质量样本
```

**语义审核**主要过滤：

- 无语义的数字编码或乱码标签。
- 不能泛化的私有实体名称。
- 只有“OK”“NG”等缺少具体视觉语义的模糊标签。

**几何审核**主要过滤：

- 严重漏标：图中有多个目标但只标出少量目标。
- 严重截断或偏移：box 没有合理包住目标。
- Mega Box：box 无意义地覆盖超过约 90% 的图像。

这与关键帧项目的 CoT 质量治理很接近：

```text
先做语义质量审核
  -> 再做时空/几何证据审核
  -> 最后做训练样本采样
```

#### 5.2 统一输出格式

Box grounding：

```text
<|ref|>TARGET<|/ref|>
<|box|>[[x1,y1,x2,y2], ...]<|/box|>
```

Point grounding：

```text
<|point|>[[x1,y1], [x2,y2], ...]<|/point|>
```

统一格式的作用：

- 让模型知道视觉原语的边界。
- 让程序能够解析并检查坐标。
- 让不同来源数据可以混合训练。
- 让后续 reward 能判断格式、引用和答案是否一致。

### 6. Cold-Start 数据设计

预训练只让模型具备一般视觉原语能力，后训练还需要少量高质量、可验证的冷启动数据。论文设计了四类任务。

#### 6.1 Counting

分为：

- Coarse-grained Counting：统计一般类别。
- Fine-grained Counting：结合颜色、属性、空间位置等条件统计。

Coarse-grained Counting 的推理格式：

```text
Intent Analysis
  -> Batch Grounding
  -> Statistical Summation
```

Fine-grained Counting 强调：

- 顺序扫描所有候选。
- 记录符合条件和不符合条件的对象。
- 加入 hard negative。
- 加入答案为零的 negative sample。

训练前检查：

- 所有 box 是否与 metadata 坐标一致。
- box 格式是否合法。
- box 数量是否与最终计数一致。
- 最终答案是否与视觉原语统计一致。

#### 6.2 Spatial Reasoning 与 General VQA

自然场景使用图像和 scene graph 构造空间推理问题，合成场景使用 CLEVR 的对象、关系和程序执行轨迹构造多跳样本。

典型推理流程：

```text
Intent Analysis
  -> Object Grounding
  -> Attribute Filtering
  -> Relational Inference
  -> Answer
```

论文还加入不存在目标或不存在关系的 negative sample，让模型学会：

```text
没有证据时拒绝强行指向目标
```

这对关键帧任务很重要：如果目标区域仍是灰块、白膜或根本不存在，模型应该输出未完成或证据不足，而不是编造一个完成态。

#### 6.3 Maze Navigation

迷宫任务用 point 表示：

- 起点。
- 终点。
- 当前探索位置。
- 分叉点。
- 死路和回溯路径。
- 最终路线。

训练数据通过 DFS、Prim、Kruskal 等算法生成可解和不可解迷宫，并随机化：

- 网格大小。
- 迷宫拓扑。
- 背景和墙体风格。
- 标记类型。
- 图像分辨率和宽高比。

难度由推理步数控制。更大的迷宫意味着更多分叉、死路和回溯，需要模型保持更长的视觉轨迹。

#### 6.4 Path Tracing

Path Tracing 要求模型沿着交错曲线追踪到终点。模型输出一系列 point：

```text
起点
  -> 中间 waypoint
  -> 交叉点后的连续分支
  -> 终点
```

waypoint 密度根据局部几何复杂度自适应：

- 直线段使用较少点。
- 弯曲和交叉密集区域使用更多点。

这对关键帧任务的迁移启发是：证据采样也不应平均分布。平稳区间可以稀疏采样，候选边界、局部刷新和状态切换区域需要密集采样。

### 7. 后训练路线

论文采用“先训练专家，再统一合并”的路线：

```text
视觉原语预训练
  -> Box 专项 SFT
  -> Point 专项 SFT
  -> Box 专项 RL
  -> Point 专项 RL
  -> Unified RFT
  -> OPD
```

#### 7.1 Specialized SFT

SFT 数据配比：

```text
70% 通用多模态和纯文本数据
30% Thinking with Visual Primitives 专项数据
```

Box 和 Point 数据分开训练，得到两个专家模型：

```text
FTwG：Thinking with Grounding
FTwP：Thinking with Pointing
```

分开训练的原因是：当专项数据比例较小时，box 和 point 两种输出模式可能互相干扰，模型容易出现格式混用或推理模式冲突。

#### 7.2 Specialized RL

论文使用 GRPO，并且在 RL 阶段不再显式监督每个视觉原语。原因是：

1. 冷启动数据中的视觉原语已经经过严格校验。
2. SFT 已经让模型学会基本的 box/point 输出方式。
3. RL 数据可以只保留图片、问题和最终答案，数据获取成本更低。
4. reward model 负责检查格式、质量和任务答案。

RL 仍然需要三类 reward：

```text
Format RM
  -> 格式是否合法，是否重复生成 box。

Quality RM
  -> 是否冗余、矛盾、引用无意义目标或 reward hacking。

Accuracy RM
  -> 根据任务判断最终答案和过程质量。
```

#### 7.3 Reward 设计

**Counting reward**

论文使用平滑的相对误差奖励：

```text
R = alpha * exp(
  - beta * abs(pred - gt) / (abs(gt) + 1)
)
```

示例设置：

```text
alpha = 0.7
beta = 3
```

相较于完全正确得 1、错误得 0 的二值奖励，平滑 reward 能区分“差一个”和“差很多”的回答。

**Spatial Reasoning reward**

使用 LLM-based GRM 分别评估：

- thinking 是否正确。
- final answer 是否正确。

最终取两部分分数的平均。

**Maze reward**

拆成：

- 合法探索进度。
- 不可解迷宫的探索完整度。
- 穿墙惩罚。
- 最终路径合法性。
- 最终可解性判断。

这种设计让模型即使暂时没有得到最终答案，也能从合法探索和正确回溯中获得部分奖励。

**Path Tracing reward**

拆成：

- 预测轨迹与真实曲线的双向距离。
- 起点和终点准确率。
- 轨迹连续性惩罚。
- 终点标签正确性。

双向距离很重要：

- 只计算预测点到真实曲线的距离，模型可能只输出起点附近的少量安全点。
- 只计算真实曲线到预测轨迹的距离，模型可能生成错误绕路。
- 双向评估才能同时约束精度和覆盖率。

#### 7.4 RL 数据难度分桶

对每个样本生成 `N` 个 rollout，按照正确数量分桶：

```text
Easy：
  N 个 rollout 全部正确。

Normal：
  1 <= 正确 rollout 数量 < N。

Hard：
  N 个 rollout 全部错误。
```

论文主要选择 Normal-Level 样本进行 RL，因为：

- Easy 样本几乎没有优化空间。
- Hard 样本可能缺少有效正向学习信号。
- Normal 样本同时包含正确和错误轨迹，最适合形成相对 reward。

#### 7.5 Unified RFT

两个专家模型分别 rollout 后，构造统一 RFT 数据：

```text
保留全部 Normal-Level 样本
  + 随机采样少量 Easy-Level 样本
  -> 使用统一数据重新训练模型
```

Easy 样本只保留少量，是为了防止模型在统一训练时遗忘基础能力。

#### 7.6 On-Policy Distillation

Unified RFT 后，如果统一模型仍然落后于两个专家模型，可以使用 OPD：

```text
student 根据自己的 policy 生成轨迹
  -> 专家模型对这些状态提供完整 token 分布
  -> student 学习专家分布
```

论文使用多个专家模型进行加权 KL 蒸馏。OPD 的优势是：学生模型暴露的是自己的真实错误状态，教师提供的是这些状态下的软分布，而不是只模仿教师预先生成的固定答案。

### 8. 实验结论与能力边界

#### 8.1 实验设置

论文使用 HAI-LLM 训练：

- 预训练序列长度约 `64K`。
- 后训练序列长度扩展到 `256K`。
- Specialized SFT/RL 使用 FP8。
- Unified RFT/OPD 阶段使用 FP4/MXFP4。

评测包含：

- Counting。
- Spatial Reasoning。
- General VQA。
- Maze Navigation。
- Path Tracing。

#### 8.2 代表性结果

论文模型在选定的困难视觉任务上取得了有竞争力的结果，例如：

- DS Fine-grained Counting：`88.7`。
- DS Spatial Reasoning：`98.7`。
- DS Maze Navigation：`66.9`。
- DS Path Tracing：`56.7`。

论文同时强调，这些结果只覆盖与视觉原语推理直接相关的评测维度，不能代表模型的全部能力。

#### 8.3 主要限制

论文明确指出：

1. 视觉输入分辨率仍会限制细粒度视觉原语的精度。
2. 当前机制依赖显式 trigger words，模型还不能完全自主决定何时启用视觉原语思考。
3. Point-based 拓扑推理的跨场景泛化仍然有限。

### 9. 对关键帧 CoT 蒸馏的直接启发

#### 9.1 从 Visual Primitive 到 Evidence Primitive

论文中的空间原语可以迁移为关键帧任务的三类证据原语：

```text
Region Primitive：
  region_id、UI 名称、OCR、box 或粗区域。

State Primitive：
  <state t="6.53" region="product_area" status="loading">

Event Primitive：
  <event start="6.53" end="6.97"
         region="product_area"
         transition="placeholder_to_loaded">
```

关键变化是：

```text
自然语言：“页面好像加载完成了”
  -> 可验证证据：“6.97s 的商品区域从占位状态变为清晰图文状态”
```

#### 9.2 将二维引用扩展到时空引用

论文主要解决二维图像中的 `where`，关键帧任务还需要 `when`：

```text
论文：
  <ref>object</ref> + <box>...</box>

关键帧：
  <time>6.97</time>
  + <region>product_area</region>
  + <state>loaded_and_stable</state>
  + <evidence>...</evidence>
```

完整的关键帧证据应包含：

```text
when：哪一帧或哪个时间点
where：哪个 UI 区域
what：发生了什么状态变化
why：为什么满足或不满足完成态
after：后续是否推翻判断
```

#### 9.3 对现有训练路线的影响

论文方法可以映射到当前项目：

| 论文路线 | 关键帧项目迁移 |
| --- | --- |
| Visual Primitive Pretraining | 利用 Qwen3-VL 基础视觉能力，离线构造 Region/State/Event Primitive |
| Box/Point Specialized SFT | Structured CoT SFT，学习时间、区域和状态证据格式 |
| Specialized RL | 对低 ACC 指标使用 verifier-based GRPO |
| Format/Quality/Accuracy RM | 格式、视觉一致性、时间边界和答案 reward |
| Normal-Level RL data | 选择部分正确、部分错误的 rollout，避免全对或全错样本 |
| Unified RFT | 将不同业务线或不同错误类型的专家数据统一混训 |
| OPD | 用更强教师指导学生自己的边界错误状态 |

#### 9.4 当前项目不应照搬的部分

- 不需要从零训练一个视觉原语基础模型，Qwen3-VL 已经提供视觉编码和视觉语言建模能力。
- 不需要直接把所有 UI 都转成精确 box，先用区域 ID、OCR 和粗框做离线 verifier 更稳妥。
- 不应把关键帧 CoT 变成完整视频 caption，必须保持问题驱动和边界相关。
- 不应在 RL 中只奖励 primitive 格式，最终时间和完成态判断仍是业务主目标。
- 不应让模型只学习输出 box/point，而忽略局部异步、豁免条件和二次刷新。

#### 9.5 最终迁移方案

```text
视频低 FPS 粗筛
  -> 生成候选时间窗口
  -> Region Primitive 标记关键 UI 区域
  -> State/Event Primitive 描述状态变化
  -> 高 FPS 精筛边界
  -> Structured CoT SFT 学习证据链
  -> GRPO 优化时间、格式、证据和边界 reward
  -> 评测集和 bad case 回流
```

这就是论文思想在关键帧任务中的真正落点：不是简单加入框，而是让模型在推理过程中持续引用“哪个时间点的哪个区域处于什么状态”，从而减少自然语言指代漂移和边界判断错误。

## 面试应对

### 常考点及考法

| 常考问题 | 考察重点 |
| --- | --- |
| 论文解决了什么问题？ | 是否理解 Perception Gap 与 Reference Gap |
| 什么是 Visual Primitive？ | 是否能说明 point/box 如何参与推理 |
| 为什么不能只在最后输出 box？ | 是否理解过程 grounding 和引用稳定性 |
| 论文如何训练？ | 是否能讲清预训练、专项 SFT、GRPO、RFT 和 OPD |
| 对关键帧任务有什么启发？ | 是否能把二维空间原语迁移成时空证据原语 |

### 解法/回答思路

```text
先定义 Reference Gap
  -> 说明 point/box 是推理中的最小思维单元
  -> 解释“边看、边指、边推理”
  -> 讲清视觉原语预训练和冷启动任务
  -> 说明格式/质量/准确性 reward
  -> 最后映射到关键帧的 time + region + state
```

### 易错点

- 不要把 Visual Primitive 只理解成目标检测输出。
- 不要把 Reference Gap 和分辨率不足混为一谈。
- 不要只讲 box，不讲 point 适合轨迹和拓扑推理。
- 不要忽略论文的质量过滤、负样本和自动 verifier。
- 不要把论文的图像方法直接等同于关键帧方法，关键帧还需要时间维度和前后边界。
- 不要说 RL 阶段完全不需要任何质量约束，论文仍然使用 Format、Quality 和 Accuracy RM。

### 回答模板

《Thinking with Visual Primitives》解决的核心问题不是模型看不到图像，而是模型在复杂视觉推理中无法用自然语言稳定指向同一个视觉对象，这被称为 Reference Gap。论文把 bounding box 和 point 从最终输出或事后校验信号提升为推理过程中的最小思维单元，让模型在思考时直接引用空间坐标。训练上先通过大规模 grounding 数据学习视觉原语，再用专项 SFT 学习计数、空间推理、迷宫和路径追踪，之后用 GRPO 结合格式、质量和任务准确性 reward 优化，并通过统一 RFT 和 OPD 合并不同专家能力。对关键帧任务，我不会直接照搬二维 box，而是把它扩展成 Region、State 和 Event Primitive，并绑定时间戳或帧号，形成“哪个时间点的哪个 UI 区域处于什么状态”的时空证据链，再用 Structured CoT SFT 和 verifier-based GRPO 优化首次完成边界。
