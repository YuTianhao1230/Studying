# 关键帧检测：OPD / Temporal-OPSD 训练方案

## 知识点解析

### 概述

本文是关键帧检测主训练流程中 On-Policy 阶段的详细设计卡片。主方案只描述训练阶段之间的关系和准入准出；本文负责展开 Temporal-OPSD 的原理、数据、训练目标、工程实现、实验协议和面试应对。

方案把 Vision-OPD 的“局部 crop/zoom 教师”改造成“局部时间窗口教师”：

```text
学生：
  看完整视频，按照当前 policy 自己生成边界判断。

教师：
  看关键时间附近的高密度局部视频窗口，提供更清晰的状态转移判断。

训练：
  教师在学生自己的 rollout 前缀上提供 token-level soft distribution。

推理：
  仍然只输入完整视频，不需要真实调用局部 crop 工具。
```

Temporal-OPSD 位于 Structured CoT SFT 和评测回流之后，使用 Student 的完整视频 rollout 与 Teacher 的高密度局部时间窗口进行 token-level 对齐。它不能替代 verifier；只有具备 Teacher logits/log-prob 时，才能称为严格的 token-level OPD。

### 1. 先给方法选择结论

#### 1.1 OPD 与 GRPO 的分工

可以替代**训练阶段的优化方法**，但不能简单认为两者功能完全相同。

| 维度 | Temporal-OPSD/OPD | Verifier-based GRPO |
| --- | --- | --- |
| 核心监督 | 教师 token 分布 | 时间、格式、边界和证据 reward |
| 主要目标 | 迁移教师的局部边界判断能力 | 直接优化业务目标 |
| 训练状态 | 学生自己的 rollout | 学生自己的 rollout |
| 主要优点 | token-level 监督密集、通常比 RL 平滑 | 直接对齐 GT 和业务规则 |
| 主要风险 | 继承教师错误，受教师输入优势影响 | reward hacking、reward 稀疏或不稳定 |
| 必备条件 | 教师 logit/log-prob，或至少可复现教师前向 | 可用 verifier 和有区分度的 reward |
| 当前项目适配 | 需要定制双视频训练链路 | 已有 reward plugin 和 verifier 基础 |

在关键帧主流程中按以下条件进入 OPD：

```text
有可靠的局部时间教师 + 可获得 logits：
  进入 Temporal-OPSD On-Policy 阶段。

只有教师 API 文本答案：
  做 Response Distillation/RFT，不是严格 OPD。

已有可靠 verifier，但没有教师 logits：
  GRPO 仍然更自然。

教师窗口质量不稳定：
  不要直接做 OPD，先做教师质量评估和 CoT-SFT。
```

#### 1.2 主流程中的训练顺序

```text
Base Qwen3-VL
  -> Direct SFT
  -> Structured CoT SFT
  -> Temporal-OPSD
  -> answer-only distillation
  -> 部署
```

若 Temporal-OPSD 的离线结果不足，回到数据治理和 verifier 分析：

```text
Structured CoT SFT
  -> Verifier-based GRPO
```

现有 GRPO reward 代码继续用于样本筛选、rollout 分桶和对照评估。

### 2. 从 Vision-OPD 到关键帧检测

#### 2.1 Vision-OPD 原始思路

Vision-OPD 的基本设定是：

```text
同一个模型看局部 crop：
  关键区域清楚，答案更容易正确。

同一个模型看完整图像：
  关键区域被全局视觉 token 淹没，可能回答错误。
```

训练时将两种视觉条件绑定：

| 策略 | 视觉输入 | 作用 |
| --- | --- | --- |
| Teacher | 证据中心 crop、放大图或清晰局部视图 | 提供特权感知 |
| Student | 完整图像 | 学习在普通输入中恢复局部能力 |

学生先生成自己的回答，教师再在学生的 prefix 上提供下一 token 分布。

#### 2.2 关键帧任务中的对应关系

关键帧检测的决定性证据不是单纯的空间区域，而是某个时间窗口中的状态转移：

```text
页面还未完成
  -> 核心区域首次满足完成态
  -> 后续保持稳定或发生二次刷新
```

因此应将特权信息从空间 crop 改成：

```text
Temporal Zoom：
  在 GT 附近使用更密集的时间采样。

Optional Spatial Zoom：
  对关键 UI 区域使用更高分辨率或局部 crop。

Temporal Evidence：
  同时保留 before/current/after 证据。
```

这条路线可以命名为：

```text
Temporal-OPSD
```

如果教师是外部强模型，而不是同一个模型的特权输入版本，则称为：

```text
Temporal-OPD
```

### 3. 模型角色和输入定义

#### 3.1 Student

学生模型使用线上真实的完整视频输入：

```text
x_global =
  完整视频
  + task_type
  + 完成态规则
  + 排除条件
  + 豁免条件
  + 输出 schema
```

它必须使用和线上一致的：

- FPS。
- `max_frames`。
- `max_pixels`。
- 视频时长和时间坐标。
- Qwen3-VL chat template。
- 输出格式。

学生的输出是部署时真正需要的策略，因此学生必须自己 rollout。

#### 3.2 Teacher

教师模型使用训练阶段的特权时间窗口：

```text
x_privileged =
  GT 附近的局部视频窗口
  + 更高的时间采样密度
  + 可选 UI 区域放大
  + 原始视频绝对时间信息
```

教师窗口不能只包含单张 GT 帧。因为单帧无法判断：

- 之前是否已经满足完成态。
- 当前是否是第一次满足。
- 后续是否有核心内容二次刷新。

因此教师窗口至少应包含：

```text
before：
  GT 之前仍未完成的证据。

current：
  GT 附近第一次满足完成态的证据。

after：
  稳定延续或二次刷新证据。
```

#### 3.3 Teacher 的参数版本

建议分三版实现：

```text
V1 Frozen Temporal-OPSD：
  Teacher 和 Student 初始参数相同。
  Teacher 使用冻结的 CoT-SFT checkpoint。

V2 EMA Temporal-OPSD：
  Teacher 初始为 CoT-SFT checkpoint。
  后续使用 Student 参数的 EMA 更新。

V3 External Temporal-OPD：
  Teacher 是更强的外部 VLM 或关键帧专家模型。
```

第一版优先使用 V1。它最容易判断收益来自“时间特权输入”，也能避免 Dynamic Teacher 造成训练坍塌。

### 4. 数据构造方案

#### 4.1 基础样本字段

当前 `MllmData` 只有一套 `videos[0]`，Temporal-OPSD 需要增加教师视图字段。推荐的离线中间格式如下：

```json
{
  "id": "sample_000001",
  "videos": [
    "/data/videos/sample_000001.mp4"
  ],
  "teacher_videos": [
    "/data/teacher_windows/sample_000001_t0550_0850.mp4"
  ],
  "conversations": [
    {
      "from": "human",
      "value": "<video>请判断任务完成态第一次成立的绝对时间。"
    },
    {
      "from": "gpt",
      "value": "<event>...</event><answer>{\"time\": 6.97}</answer>"
    }
  ],
  "infos": {
    "task_name": "video_keyframe",
    "task_type": "商品详情页加载",
    "data_type": "video",
    "duration": 12.0,
    "gt_time": 6.97,
    "answer_obj": "{\"time\": 6.97}",
    "teacher_window": [5.5, 8.5],
    "teacher_fps": 12,
    "student_fps": 10,
    "teacher_view_type": "temporal_dense",
    "problem_type": "temporal_keyframe"
  },
  "image": [],
  "images": [],
  "meta": {
    "id": "sample_000001"
  }
}
```

实际训练时可以把教师视频字段改成自定义字段，例如：

```text
teacher_video
teacher_frames
teacher_frame_timestamps
teacher_window_start
teacher_window_end
```

字段名称不重要，但必须同时保留：

```text
教师视频资源
教师窗口在原始视频中的起止时间
教师帧的绝对时间戳
Student 和 Teacher 的输出坐标口径
```

#### 4.2 教师窗口生成

对于有完成态的样本，设 GT 时间为 `t*`，推荐先使用：

```text
window_start = max(0, t* - 1.5s)
window_end   = min(duration, t* + 1.5s)
```

后续根据任务类型调整：

| 任务类型 | 建议窗口 | 原因 |
| --- | --- | --- |
| 页面加载 | `[-2.0s, +2.0s]` | 需要看局部异步和稳定性 |
| 购物车角标 | `[-1.0s, +1.0s]` | 重点是数字首次变化 |
| 头图滑动 | `[-1.5s, +1.5s]` | 需要判断停稳和居中 |
| SKU 面板加载 | `[-2.0s, +2.0s]` | 需要确认核心区域完整出现 |
| 二次刷新任务 | `[-2.0s, +3.0s]` | 需要看后续替换是否推翻当前状态 |

教师窗口使用更高的采样密度，例如：

```text
Student：10 FPS，最多 300 帧
Teacher：窗口内 12-20 FPS，保留全部边界附近帧
```

注意：教师多看了帧，收益包含“输入信息更多”的因素。实验中必须同时报告：

```text
Full-video Student baseline
Dense-window Teacher baseline
Temporal-OPSD
```

否则无法判断收益来自教师输入还是蒸馏机制。

#### 4.3 无完成态样本

如果 `gt_time=-1`，不能伪造一个正向 GT 窗口。推荐三种处理：

```text
方案 A：
  Teacher 和 Student 都看完整视频，只做 hard-label/CE。

方案 B：
  从模型候选时间中选择一个疑似完成窗口给 Teacher，
  Teacher 学习判断“窗口内没有完成态”。

方案 C：
  生成多个局部负窗口，要求 Teacher 输出 -1。
```

第一版建议使用方案 A，避免负样本的特权窗口构造引入额外噪声。

#### 4.4 GT 附近 hard negative

Temporal-OPSD 的核心不是只让教师看 GT，而是让模型学会边界。

对每个正样本构造：

```text
before window：
  [t* - 0.8s, t* + 0.2s]

target window：
  [t* - 0.2s, t* + 0.8s]

after window：
  [t*, t* + 1.5s]
```

还应保留以下困难样本：

- 学生预测早于 GT 的样本。
- 学生预测晚于 GT 的样本。
- 第一次完成和第二次完成同时存在的样本。
- 视觉状态几乎不变但业务状态改变的样本。
- 页面主体已出现但核心小元素未完成的样本。
- 局部异步加载可以豁免的样本。

### 5. Prompt 和输出协议

#### 5.1 Student prompt

Student prompt 应与线上 prompt 保持一致，不暴露教师窗口：

```text
你是 UI 性能关键帧检测模型。

给定完整录屏、task_type、完成态定义、排除条件和豁免条件，
请判断该指标的完成态第一次成立的绝对时间。

必须区分：
1. 之前仍未完成的状态；
2. 第一次满足全部必决条件的状态；
3. 后续稳定延续或二次刷新。

所有时间使用原始视频绝对秒数。
输出结构化证据和最终答案。
```

#### 5.2 Teacher prompt

Teacher prompt 可以说明局部窗口的时间范围，但不能给出 GT：

```text
你是 UI 性能关键帧检测教师模型。

当前输入是原始视频中的局部高密度时间窗口，
窗口范围是 [5.5s, 8.5s]，所有输出时间仍使用原始视频绝对秒数。

请根据视频和 task_type 判断：
before：什么时候仍未完成；
current：什么时候第一次满足完成态；
after：后续是否稳定或发生二次刷新。

不要假设窗口中心就是正确答案，不要输出窗口中心作为答案。
```

Teacher 和 Student 的任务语义、输出 schema 和时间坐标必须一致。二者只应该在视觉输入条件上有差异。

#### 5.3 推荐输出格式

为了获得更多 token-level 监督，建议先使用结构化输出：

```text
<event>
<time>5.50-6.53</time>
<caption>页面主体出现，但商品图仍为占位状态。</caption>
<think>当前不满足核心商品区域完整展示条件。</think>
</event>

<event>
<time>6.97</time>
<caption>核心商品图、价格和权益栏清晰显示，页面首次停止位移。</caption>
<think>这是第一次满足完成态的时间。</think>
</event>

<event>
<time>7.50-8.50</time>
<caption>后续没有核心区域二次刷新。</caption>
<think>该段用于稳定性复核。</think>
</event>

<answer>{"time": 6.97}</answer>
```

现有 `video_keyframe_structured_reward.py` 已经能够解析类似的时间、caption、thinking、状态和事件信息，可以继续用于离线质量检查。

### 6. 训练目标

#### 6.1 学生 on-policy rollout

对每个输入样本：

```text
y ~ p_S(. | x_global, q)
```

训练中必须保存：

- `input_ids`。
- 学生生成的 token。
- 每个生成 token 的 log-prob。
- 每个 prefix 的位置。
- `answer.time` 解析结果。
- 格式解析结果。
- 是否属于 early/late/normal/hard。

只保存最后的预测时间，无法计算严格的 OPD loss。

#### 6.2 Teacher 和 Student 前向

对每个 student prefix `y_<t`：

```text
teacher_logits =
  Teacher(teacher_video, teacher_prompt, y_<t)

student_logits =
  Student(full_video, student_prompt, y_<t)
```

Teacher 必须停止梯度：

```python
with torch.no_grad():
    teacher_logits = teacher_forward(...)

student_logits = student_forward(...)
loss = divergence(teacher_logits, student_logits)
```

#### 6.3 推荐联合损失

第一版推荐：

\[
\mathcal{L}_{total}
=
\lambda \mathcal{L}_{OPD}
+(1-\lambda)\mathcal{L}_{CE}
+\mu\mathcal{L}_{ref}
\]

其中：

- `L_OPD`：JSD 或 reverse KL。
- `L_CE`：GT 时间和 clean CoT 的硬标签监督。
- `L_ref`：可选的 Student 与起始 CoT-SFT checkpoint 的 KL 约束。

推荐初始值：

```text
JSD beta = 0.5
lambda = 0.5
mu = 0.01-0.05
temperature = 1.0-2.0
```

如果训练早期出现格式坍塌：

```text
降低 lambda
提高 CE 权重
冻结 Teacher
减少 rollout 长度
```

如果模型只学会复制 CoT 文字但时间指标不提升：

```text
增加 time token 的 OPD mask 权重
增加 GT 附近 hard negative
增加 boundary evidence 的 clean target
```

#### 6.4 Loss mask

不要对所有 token 无差别蒸馏。建议分别记录：

```text
L_structure：
  <event>、<time>、<caption>、<think>、<answer> 结构 token。

L_boundary：
  before/current/after、未完成、首次满足、二次刷新等 token。

L_time：
  answer.time 数值 token。

L_content：
  UI 区域、状态和事件描述 token。
```

第一轮可以使用：

```text
L_total =
  0.2 * L_structure
  + 0.4 * L_boundary
  + 0.3 * L_time
  + 0.1 * L_content
```

这不是固定结论，最终以困难集指标和格式稳定性调节。

### 7. 两种 OPD 实现路线

#### 7.1 路线 A：Sampled-token reverse KL

这是最容易写代码的原型版本。

Student 生成 token `y_t` 后，只查询 Teacher 和 Student 对该 token 的 log-prob：

\[
r_t^{KD}
=
\log p_T(y_t|s_t)-\log p_S(y_t|s_t)
\]

或者使用：

\[
\mathcal{L}_{sample}
=
\log p_S(y_t|s_t)-\log p_T(y_t|s_t)
\]

优点：

- 只需要学生实际采样 token 的概率。
- 显存和通信开销低。
- 可以先验证 Temporal-OPSD 是否有收益。

缺点：

- 不是完整的 logit-level distribution matching。
- 丢失教师对其他候选 token 的相对概率信息。
- 对低熵的数字和格式 token 可能不够稳定。

建议先用路线 A 做 smoke test，再实现路线 B。

#### 7.2 路线 B：Top-K JSD/KL

完整版本保留 Student Top-K token，并查询 Teacher 对应 token 的 logits：

```text
student_topk_ids = topk(student_logits, K)
teacher_selected_logits =
    gather(teacher_logits, student_topk_ids)
```

再加上剩余词表的 tail probability 近似，计算截断分布上的 JSD/KL。

推荐起点：

```text
K = 100
JSD beta = 0.5
temperature = 1.0
```

路线 B 更接近 Vision-OPD 的完整做法，但需要更高的显存、通信和工程复杂度。

### 8. 训练器实现结构

当前标准 ms-swift SFT 和 `ORM reward` 不能直接完成双视图 OPD，需要增加一个自定义训练器或独立训练入口。

推荐模块划分：

```text
temporal_opd_dataset.py
  读取 full video、teacher video、窗口时间和 GT。

temporal_opd_collator.py
  同时构造 Student 和 Teacher 两套多模态输入。

temporal_opd_rollout.py
  用 Student 生成 on-policy response。

temporal_opd_logits.py
  在相同 prefix 上计算 Teacher/Student logits。

temporal_opd_loss.py
  实现 sampled-token KL、Top-K KL、JSD 和 CE 混合。

temporal_opd_trainer.py
  负责 rollout、前向、反向、优化器和 EMA。

temporal_opd_eval.py
  计算关键帧指标、teacher gap 和回归集结果。
```

核心训练循环可以写成：

```python
for batch in dataloader:
    student_rollout = rollout_student(
        model=student,
        videos=batch["student_videos"],
        prompts=batch["student_prompts"],
        max_new_tokens=rollout_max_tokens,
        temperature=rollout_temperature,
    )

    prefix_batch = build_prefix_batch(
        batch=batch,
        rollout=student_rollout,
    )

    with torch.no_grad():
        teacher_logits = forward_teacher(
            model=teacher,
            videos=batch["teacher_videos"],
            prompts=batch["teacher_prompts"],
            prefix_batch=prefix_batch,
        )

    student_logits = forward_student(
        model=student,
        videos=batch["student_videos"],
        prompts=batch["student_prompts"],
        prefix_batch=prefix_batch,
    )

    loss_opd = compute_masked_jsd(
        teacher_logits=teacher_logits,
        student_logits=student_logits,
        token_mask=prefix_batch["opd_mask"],
        beta=0.5,
        temperature=1.0,
    )

    loss_ce = compute_supervised_ce(
        model=student,
        batch=batch,
        token_mask=batch["answer_and_cot_mask"],
    )

    loss = lambda_opd * loss_opd + (1.0 - lambda_opd) * loss_ce

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(student.parameters(), max_norm=1.0)
    optimizer.step()

    update_ema_teacher(
        teacher=teacher,
        student=student,
        update_rate=ema_update_rate,
    )
```

实际实现时还必须处理：

- 视频视觉特征的 batch 对齐。
- Student/Teacher 不同视频长度。
- prompt token、视觉 token 和 assistant token 的位置。
- padding、EOS 和截断 mask。
- teacher forward 的显存释放。
- rollout 和训练 forward 的 cache 是否复用。
- 分布式训练下 Teacher 参数的同步。

### 9. 训练阶段和参数建议

#### 9.1 阶段零：先测 privileged gap

使用同一个 CoT-SFT checkpoint，分别测试：

```text
Full-video input
Dense temporal-window input
```

记录：

- `time_error`。
- `frame_error`。
- 完成态 ACC。
- early/late 比例。
- before/current/after 准确率。
- 无完成态准确率。
- 各 `task_type` 指标。

如果教师窗口输入本身没有明显优于完整视频，先不要做 OPD。

#### 9.2 阶段一：教师质量验证

从候选数据中抽取：

```text
正常样本
early bad case
late bad case
二次刷新样本
局部异步样本
无完成态样本
```

分别检查 Teacher：

- 最终时间是否正确。
- 时间是否使用绝对坐标。
- before/current/after 是否递增。
- UI 证据是否真实。
- 是否把窗口中心当答案。
- 是否产生重复、幻觉或格式错误。

教师质量不稳定时，先回流 CoT 数据治理，不直接训练 OPD。

#### 9.3 阶段二：小规模 smoke test

推荐先使用：

```text
1,000-2,000 条高质量样本
以 early/late/边界难例为主
先使用 Frozen Teacher
先使用 sampled-token reverse KL
rollout_max_tokens = 256-512
num_train_epochs = 0.1-0.3
```

smoke test 必须验证：

- loss 能下降。
- Teacher logits 不参与梯度。
- Student/Teacher 的时间坐标一致。
- OPD mask 只覆盖目标 token。
- 输出格式不会迅速退化。
- 困难集指标有改善趋势。

#### 9.4 阶段三：正式 Temporal-OPSD

建议起始配置：

| 参数 | 建议起点 |
| --- | --- |
| 初始化模型 | Structured CoT SFT checkpoint |
| Teacher | Frozen checkpoint |
| divergence | JSD |
| `beta` | `0.5` |
| `lambda_opd` | `0.3-0.5` |
| Top-K | `100`，或先用 sampled-token |
| rollout 长度 | `256-512` |
| rollout temperature | `0.7-1.0` |
| gradient clipping | `1.0` |
| EMA | 第二版再打开 |
| hard negative | GT 前后窗口、early/late、二次刷新 |

如果正式训练稳定，再切换：

```text
Frozen Teacher
  -> EMA Teacher
  -> Top-K JSD
```

不要同时修改 Teacher 类型、divergence、Top-K 和数据分布，否则无法定位收益或退化来源。

### 10. 评估协议

#### 10.1 必须保留的对照

至少比较：

```text
A：Direct/Structured CoT SFT
B：完整视频上的 Student
C：局部密集窗口上的 Teacher
D：Response Distillation
E：Frozen Temporal-OPSD
F：Temporal-OPSD + CE
G：Temporal-OPSD + EMA
H：原有 Verifier-based GRPO
```

其中 `C` 很重要，它衡量“教师输入本身有多强”；没有这个对照，不能证明 OPD 机制有效。

#### 10.2 指标

业务指标：

- 全量 ACC。
- 困难集 ACC。
- `time_error`。
- `frame_error`。
- early/late error。
- 首次完成边界准确率。
- 二次刷新样本准确率。
- 无完成态准确率。
- 各业务线和 `task_type` 指标。

训练指标：

- `loss_opd`。
- `loss_ce`。
- Teacher-Student KL/JSD。
- Teacher entropy。
- Student entropy。
- rollout 解析率。
- 输出长度。
- 重复率。
- 格式合法率。

#### 10.3 建议的准入和停止条件

可以使用以下工程准入标准：

```text
困难集 ACC 相比 CoT-SFT 有明确提升；
全量 ACC 不出现明显退化；
格式解析率不下降；
early/late 至少有一个主要方向明显改善；
无完成态样本没有被系统性误报；
Teacher-Student gap 在训练中缩小；
```

建议先以：

```text
困难集提升 1-2 个百分点
全量指标下降不超过 0.5 个百分点
```

作为小规模实验的继续条件。正式阈值仍应根据业务容忍度确定。

### 11. 与现有 GRPO 代码的关系

当前的：

```text
video_keyframe_reward.py
video_keyframe_structured_reward.py
```

提供的是标量 reward，不是 Teacher token distribution。因此：

```text
不能只替换 ORM reward 就得到 OPD。
```

现有 reward 代码可以继续用于：

- Teacher 样本过滤。
- Student rollout 质量分析。
- early/late 分桶。
- 训练后评估。
- 与 GRPO 的结果对照。

但 OPD 需要额外实现：

```text
双视图数据读取
Teacher forward
Student 同 prefix forward
logits/log-prob 对齐
OPD loss
rollout 刷新
EMA Teacher
```

### 12. 常见失败模式

#### 12.1 教师窗口中心泄漏答案

表现：

```text
Teacher 总是输出窗口中心时间。
Student 学到窗口中心偏置。
```

处理：

- 窗口前后不对称。
- 随机扰动窗口中心。
- 加入 GT 前后 hard negative。
- Prompt 中明确禁止使用窗口中心。
- 检查 Teacher 在不同窗口长度下的稳定性。

#### 12.2 教师看到更多帧，Student 无法复现

表现：

```text
Teacher 很准，OPD 后 Student 不升反降。
```

原因可能是教师使用了 Student 根本没有看到的帧。

处理：

- 增加 Student 的边界附近采样预算。
- 使用更接近部署配置的教师窗口。
- 报告“输入增益”和“蒸馏增益”两个结果。
- 不把教师输入优势全部归因于 OPD。

#### 12.3 纯 OPD 发生模式坍塌

表现：

```text
Recall 很高但 Precision 很低。
模型大量输出同一种完成态。
格式或时间数字漂移。
```

处理：

- 降低 `lambda_opd`。
- 增加 CE anchor。
- 冻结 Teacher。
- 加入负样本和无完成态样本。
- 使用 JSD 替代单向 Reverse KL。
- 对时间 token 和结构 token 单独监控。

#### 12.4 只学会 CoT 风格

表现：

```text
输出越来越像 Teacher，
但 time_error、early/late 没有改善。
```

处理：

- 增大 `L_time` 和 `L_boundary` 权重。
- 减少无关 caption token 的 mask 权重。
- 增加 GT 附近 hard negative。
- 使用时间和证据一致性作为离线筛选条件。

#### 12.5 只有教师文本，没有 logits

这时不能实现严格的 logit-level OPD。可改成：

```text
教师生成结构化答案
  -> verifier 过滤
  -> 保留高质量回答
  -> 对 Student 做 SFT/RFT
```

它仍然有价值，但名称应写成 Response Distillation 或 RFT，而不是严格 OPD。

## 面试应对

### 常考点及考法

| 常考点 | 常见问法 | 回答重点 |
| --- | --- | --- |
| 方法迁移 | 如何把 Vision-OPD 用到视频关键帧 | 将空间 crop 改为关键时间附近的高密度窗口 |
| 教师输入 | 为什么不能只给教师一张关键帧 | 单帧无法判断首次完成和后续二次刷新 |
| 方法选择 | OPD 能否替代 GRPO | 可以替代优化阶段，但需要教师 logits；不能替代 verifier |
| 工程实现 | 现有 ms-swift 能否直接支持 | 默认单视频 SFT 不够，需要双视图 collator 和自定义 trainer |
| 训练稳定性 | 如何避免模式坍塌 | Frozen/EMA Teacher、JSD、CE anchor 和 hard negative |
| 效果归因 | 教师看更多帧怎么办 | 加 Teacher-only 对照，区分输入增益和蒸馏增益 |

### 解法/回答思路

回答这个方案时按以下顺序展开：

```text
1. 先定义关键帧任务是首次完成边界判断。
2. 说明 Vision-OPD 的空间 crop 要改造成时间窗口 zoom。
3. 说明 Student 看完整视频，Teacher 看 GT 附近高密度窗口。
4. 强调 Student 必须先自己 rollout，Teacher 在相同 prefix 上给 logits。
5. 用 JSD/KL + CE anchor 训练。
6. 用 Teacher-only、SFT、OPD 和 GRPO 做对照。
7. 根据是否有 Teacher logits 决定能否称为严格 OPD。
```

### 易错点

- 把教师单独看 GT 帧等同于完整 Temporal-OPD。
- 让教师看到 GT 时间文本，造成标签泄漏。
- 让 Teacher 输出局部相对时间，却让 Student 输出原视频绝对时间。
- 只保留最终 `answer.time`，没有保存 Student rollout prefix。
- 只看 OPD loss 下降，不看 early/late 和业务 ACC。
- 将 `video_keyframe_reward.py` 误认为可以直接提供 OPD loss。
- 忽略教师输入帧数更多导致的输入增益。
- 用 Dynamic Teacher 作为第一版，导致师生共同漂移。

### 回答模板

#### 关键帧检测如何使用 OPD？

我会把 Vision-OPD 的空间局部 crop 改成时间局部 zoom。Student 输入完整视频和任务规则，按照当前 policy 自己生成结构化的 before/current/after 判断；Teacher 输入 GT 附近的高密度短视频窗口，但不看到 GT 时间文本，并且仍然使用原视频绝对时间。Teacher 在 Student 自己生成的每个 prefix 上提供 token-level logits，Student 用 JSD 或 KL 对齐 Teacher，同时保留 GT 时间的 CE anchor。这样训练时利用了局部窗口的清晰边界证据，部署时仍然只需要完整视频。

#### OPD 可以完全替代 GRPO 吗？

Temporal-OPSD 可以作为 GRPO 之后的 On-Policy 优化阶段，但前提是有可靠的 Teacher logits 或 log-prob，以及质量稳定的局部时间教师。它不能替代 verifier，因为 OPD 主要迁移教师行为，可能继承教师错误；GRPO 则直接优化时间、格式和业务证据 reward。两者在主流程中分别承担业务 reward 优化和局部时间证据蒸馏。

#### 为什么教师不能只看 GT 关键帧？

因为关键帧标签表示“完成态第一次成立”，不是“某一帧看起来完成”。只看单帧无法判断前一帧是否已经满足，也无法判断当前完成态是否会被后续核心内容刷新推翻。因此教师至少需要看到 GT 前后的局部时间窗口，并保留 before/current/after 三类证据。

#### 当前代码怎么改？

当前 reward plugin 只能返回标量 reward，不能直接实现 OPD。工程上需要增加双视图数据字段、Teacher/Student collator、Student rollout、相同 prefix 上的双模型 forward、masked KL/JSD loss 和可选 EMA Teacher。现有关键帧 reward 代码可以继续作为 Teacher 样本过滤、rollout 分桶和最终评测工具。
