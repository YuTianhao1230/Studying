# On-Policy Distillation 在线策略蒸馏

## 知识点解析

### 概述

On-Policy Distillation（OPD，在线策略蒸馏）是一种让学生模型在自己的策略分布上接受教师模型监督的能力迁移方法。学生先根据当前 policy 生成真实 rollout，教师再对这些学生生成的前缀计算 token-level 概率分布或提供等价的软监督，学生通过蒸馏损失学习在自己容易出错的状态下采取更好的下一步动作。与教师预先生成答案再做普通 SFT 相比，OPD 的关键差异在于训练状态来自学生自身，因此更贴近学生实际部署时的错误分布。

OPD 的核心链路是：

```text
冻结教师模型
  -> 学生按当前 policy 生成 rollout
  -> 教师在学生前缀上计算 soft distribution
  -> 学生在相同前缀上对齐教师分布
  -> 用验证集、困难集和能力回归评估
```

它解决的主要问题不是“如何让学生背下教师答案”，而是：

```text
学生在哪些状态上会犯错？
教师在这些状态下认为哪些 token 更合理？
如何用密集的 token-level 信号修正学生的策略？
```

### 一、OPD 在蒸馏方法中的位置

根据学生接触到的训练状态和教师监督形式，可以区分几种路线：

| 方法 | 训练状态来源 | 教师信号 | 主要特点 |
| --- | --- | --- | --- |
| Label SFT | 数据集中的人工或程序标签 | 硬标签 | 稳定、便宜，但监督信息较少 |
| Response Distillation | 教师预先生成的输入-回答对 | 教师文本回答 | 不需要教师 logits，工程简单 |
| Logit Distillation | 固定数据或教师轨迹上的相同前缀 | 教师 token 分布 | 监督更细，但要求可获得 logits |
| OPD | 学生自己生成的前缀 | 教师在学生前缀上的 token 分布 | 覆盖学生真实错误，训练成本更高 |
| GRPO/RLVR | 学生自己生成的 rollout | verifier 或 reward | 直接优化任务结果，不依赖教师逐 token 模仿 |

“在线”主要指 rollout 状态由当前学生 policy 产生，并不等于教师参数必须在线更新。典型 OPD 中教师保持冻结，学生周期性生成新的 rollout，教师只负责提供监督分布。

### 二、为什么教师轨迹蒸馏不一定能修复学生错误

普通 Response Distillation 的数据形态是：

```text
教师：
  x -> y_teacher

学生：
  学习在 x 上生成 y_teacher
```

如果学生在部署时通常会生成 `y_student`，而教师数据中没有出现学生自己的错误前缀，那么学生可能只学到教师的表达方式，却没有学习如何从自己的错误状态回到正确轨迹。

OPD 的训练形态是：

```text
学生：
  x -> y_student,1 -> y_student,2 -> ...

教师：
  在 x、x+y_student,1、x+y_student,1+y_student,2 等前缀上给出分布

学生：
  在这些真实前缀上对齐教师的下一 token 分布
```

因此，OPD 更接近一种错误状态覆盖机制：

```text
学生自己的分布
  -> 学生暴露真实错误
  -> 教师提供局部纠偏
  -> 学生逐步减少这些错误
```

### 三、标准训练流程

#### 3.1 准备教师与学生

- 教师模型应在目标任务上明显强于学生。
- 教师参数冻结，只进行推理。
- 学生通常从稳定的 SFT 或 CoT SFT checkpoint 开始。
- 学生和教师应使用兼容的 tokenizer、输出模板和任务输入。
- 多模态任务还要对齐图片/视频帧、视觉分辨率、帧采样和系统提示。

直接从 Base Model 开始做 OPD，难以区分学生不会任务、不会格式，还是不会模仿教师。通常先用 SFT 让学生具备基本任务能力，再用 OPD 解决学生特有的边界错误。

#### 3.2 学生生成 rollout

对输入 `x`，学生按当前 policy 生成回答：

```text
y ~ π_student(y | x)
```

训练中可以保留：

- 学生生成的完整回答。
- 每个 token 的前缀。
- 生成 token 的 log probability。
- 是否解析成功。
- 最终答案和任务 verifier 结果。

rollout 不应只保存最终文本，因为 OPD 的监督位置是学生实际经过的前缀。

#### 3.3 教师计算软分布

在学生生成的每个前缀 `y_<t` 上，教师输出下一 token 的 logits：

```text
p_teacher(. | x, y_<t)
```

学生也在相同前缀上计算：

```text
p_student(. | x, y_<t)
```

教师不需要认可学生已经生成的前缀。即使前缀包含错误，教师仍然可以判断在这个错误状态之后更合理的 token 分布。

#### 3.4 计算蒸馏损失

常见做法是对齐教师分布和学生分布。引入温度 `T` 后：

```text
p_T^T = softmax(z_teacher / T)
p_S^T = softmax(z_student / T)

L_OPD =
  T^2 * KL(p_T^T || p_S^T)
```

对多个 token 和多个样本取平均：

```text
L_OPD =
  E_{x, y~π_student}
  [ sum_t m_t * KL(p_teacher,t || p_student,t) ]
```

其中 `m_t` 是 mask，用于只在回答 token、有效 token 或指定结构化字段上计算损失。

实际训练中常把 OPD 与硬标签或任务损失混合：

```text
L_total =
  λ_opd * L_OPD
  + λ_sft * L_SFT
  + λ_task * L_task
  + λ_reg * L_reg
```

各项含义：

- `L_OPD`：学习教师在学生状态上的软分布。
- `L_SFT`：保持人工标签、结构化输出和基本格式。
- `L_task`：时间、分类、区域或其他任务级监督。
- `L_reg`：控制模型偏移、长度、重复或其他约束。

#### 3.5 更新学生并刷新 rollout

更新学生参数后，旧 rollout 只能在有限范围内复用。随着学生 policy 改变，学生错误分布也会改变，因此需要周期性重新生成 rollout：

```text
student rollout
  -> teacher logits
  -> student update
  -> new student rollout
  -> repeat
```

这也是 OPD 比离线 Response Distillation 成本更高的原因。

### 四、关键帧和视频任务中的 OPD

关键帧检测可以把 OPD 放在以下路线中：

```text
Base VLM
  -> Direct SFT
  -> Structured CoT SFT
  -> OPD 或 Verifier-based GRPO
  -> answer-only distillation / 部署
```

#### 4.1 学生的 on-policy 状态

学生可能出现：

- 将主体出现误判为页面完全加载。
- 错过购物车角标或边缘小元素。
- 把后续稳定帧当成首次完成帧。
- 跳过第一次滑动，直接选择第二次完成。
- 在正确时间附近输出不稳定格式。
- 描述了视频中不存在的 UI 状态。

这些错误前缀正是 OPD 应重点覆盖的状态。

#### 4.2 教师监督内容

教师可以在学生前缀上提供：

- 结构化状态 token 的分布。
- before/current/after 证据表达。
- 候选时间和边界判断的 token 分布。
- `<answer>`、JSON 或 XML 结构的下一 token 分布。

但纯 token-level KL 可能只让学生模仿教师的语言风格，不一定真正修正时间边界。因此关键帧任务应将 OPD 与任务监督结合：

```text
token-level OPD
  + 时间误差或边界 verifier
  + 格式校验
  + UI 证据一致性检查
```

#### 4.3 关键帧场景的样本选择

优先选择：

1. 学生 rollout 有正确和错误分支的 Normal 难度样本。
2. 学生稳定偏早或偏晚的样本。
3. 小 UI、二次刷新、局部异步和遮挡样本。
4. Structured CoT 已经过视觉一致性校验的样本。
5. 教师置信度较高且能解释边界的样本。

不建议把所有简单样本都投入 OPD。简单样本的教师和学生分布可能已经接近，训练成本高但新增监督少。

#### 4.4 多模态输入必须严格对齐

对于视频模型，教师和学生至少要对齐：

```text
视频文件或帧列表
  + FPS / max_frames
  + 视频裁剪和时间坐标
  + 分辨率和视觉 token budget
  + task_type prompt
  + chat template
  + 输出 schema
```

如果教师看的是高 FPS 局部窗口，而学生只看低 FPS 全局视频，教师分布中可能包含学生无法观察到的证据。此时 OPD 的收益不能直接归因于蒸馏，应该明确区分“教师更强”与“教师输入更多”。

### 五、OPD 与 SFT、GRPO 的关系

#### 5.1 OPD 与 SFT

SFT 使用固定的目标答案或教师 response：

```text
x -> y_target
```

OPD 使用学生自己的生成前缀：

```text
x + y_student,<t -> teacher distribution
```

SFT 更适合：

- 冷启动。
- 学习输出协议。
- 学习基本任务能力。
- 训练学生远离明显错误。

OPD 更适合：

- 修复学生已暴露的错误状态。
- 获得比 one-hot 标签更密集的分布监督。
- 迁移教师在边界决策上的局部偏好。

#### 5.2 OPD 与 GRPO/RLVR

| 维度 | OPD | GRPO/RLVR |
| --- | --- | --- |
| 监督来源 | 教师 token 分布 | verifier 或 reward |
| 优化对象 | 学生在教师指导下的局部行为 | 学生 rollout 的任务结果和过程质量 |
| 是否需要教师 logits | 严格 logit-level OPD 通常需要 | 不需要 |
| 训练信号 | 密集、低方差、受教师影响 | 任务对齐强，但 reward 方差和设计风险更高 |
| 主要风险 | 继承教师错误、过度模仿 | reward hacking、训练不稳定、探索不足 |
| 适用定位 | 能力迁移和错误纠偏 | 目标优化和策略搜索 |

两者可以组合：

```text
OPD 提供教师的局部行为监督
  -> GRPO 用任务 verifier 修正教师与业务目标的差异
```

也可以反过来先用 GRPO 得到更强学生，再将其 rollout 蒸馏给更小的部署模型。

### 六、教师信号与学生状态的质量控制

OPD 不是教师越强、数据越多就一定越好。至少要做以下过滤：

#### 教师质量

- 教师最终答案是否正确。
- 教师证据是否与视觉内容一致。
- 教师是否出现重复、幻觉或格式错误。
- 教师在当前前缀上的分布熵是否过高。
- 教师是否只是使用了学生看不到的额外信息。

#### 学生状态

- rollout 是否来自当前学生 policy。
- 前缀是否完整、没有被错误截断。
- 学生是否已经进入异常循环。
- 是否存在大量完全相同的 rollout。
- 是否保留了正确和错误状态的平衡。

#### 训练信号

- 教师和学生 tokenizer 是否兼容。
- prompt token 是否被错误计入损失。
- padding、EOS 和截断 mask 是否正确。
- 蒸馏温度和损失权重是否导致过度平滑。
- 学生是否只学会教师的格式而没有提升任务指标。

### 七、主要优点

1. **贴近部署分布**：学生自己生成的错误状态更接近上线时真实暴露的问题。
2. **监督更密集**：软分布比单个 one-hot 标签包含更多 token 选择信息。
3. **比纯 RL 更稳定**：教师分布提供直接的局部监督，不完全依赖高方差 reward。
4. **适合困难样本纠偏**：可以针对学生的 early、late、格式和证据错误做定向蒸馏。
5. **可与其他后训练组合**：可以和 SFT、任务损失、Verifier 或 GRPO 组合使用。

### 八、局限与风险

1. **教师推理成本高**：每轮学生 rollout 都需要教师前向，视频任务还会重复消耗视觉编码和 KV cache。
2. **教师错误会被迁移**：教师在学生错误前缀上也可能判断错误，不能把教师分布视为绝对真值。
3. **模型兼容性要求高**：tokenizer、词表、模板、视觉输入或输出头不兼容时，不能直接做 token-level KL。
4. **容易过度模仿**：学生可能复制教师风格、长度和格式，却没有提升真实任务能力。
5. **分布仍可能偏窄**：学生 rollout 只覆盖当前 policy 已经能到达的状态，无法自动覆盖所有困难区域。
6. **多模态证据不对称**：教师看到更多帧或更高分辨率时，蒸馏信号可能混入输入优势。
7. **训练和部署有偏差**：如果只在训练 rollout 上蒸馏，不做固定测试集和线上回归，可能出现局部收益、整体退化。

### 九、评估与监控

OPD 不应只看蒸馏 loss。建议同时监控：

```text
student task accuracy
student-specific bad case accuracy
teacher-student KL
teacher entropy
answer format rate
early / late error
evidence consistency
output length and repetition
general capability regression
```

关键对照至少包括：

1. 原始学生模型。
2. 只做 Response Distillation 的学生模型。
3. 做 OPD 的学生模型。
4. OPD 加任务 verifier 或 hard label 的学生模型。

对视频关键帧任务，还要增加：

- 业务线和 task_type 分桶。
- 帧误差和时间误差。
- 关键 UI 元素类别。
- 首次完成、二次刷新和无完成态样本。
- 线上输入分布与离线训练分布。

### 十、推荐的工程决策

如果只有教师 API 输出，没有教师 logits：

```text
优先使用 Response Distillation
  + rejection sampling
  + verifier 过滤
  + 学生 on-policy 重新打标
```

如果可以获得教师 logits，且教师与学生输入、词表和模板兼容：

```text
先做小规模 token-level OPD
  -> 验证学生困难集是否改善
  -> 再扩大 rollout 和教师调用规模
```

如果业务目标有可靠 verifier：

```text
SFT
  -> OPD 做能力迁移
  -> GRPO/RLVR 做业务目标修正
```

如果教师和学生结构差异很大，或者教师输入明显更丰富，不建议强行做 token-level OPD，可以改用：

- 最终答案蒸馏。
- 结构化证据蒸馏。
- 中间状态或区域级蒸馏。
- 教师生成候选，verifier 过滤后做 SFT。

## 面试应对

### 常考点及考法

| 常考点 | 常见问法 | 回答重点 |
| --- | --- | --- |
| 基本概念 | 什么是 OPD | 学生自己 rollout，教师在学生前缀上提供软监督 |
| 方法区别 | OPD 和普通蒸馏有什么不同 | 训练状态来自学生，而不是教师预先生成的固定轨迹 |
| 训练信号 | OPD 的 loss 怎么写 | 学生前缀上的 token-level KL，可混合 SFT/任务损失 |
| 方法选型 | OPD 和 GRPO 怎么选 | OPD 做能力迁移，GRPO 做 verifier 驱动的目标优化 |
| 工程代价 | OPD 为什么贵 | 学生 rollout 加教师逐 token 前向，视频还要重复视觉计算 |
| 失败风险 | 教师不可靠怎么办 | 教师过滤、置信度、任务 verifier 和固定回归集 |
| 多模态 | 视频任务要注意什么 | 对齐帧采样、分辨率、prompt、tokenizer 和时间坐标 |

### 解法/回答思路

回答 OPD 时按“学生状态、教师信号、优化目标、与其他方法的边界、代价与验证”展开。不要只说“教师指导学生”，要明确教师是在学生自己生成的前缀上提供 token-level 分布，这才是 OPD 与普通 Response Distillation 的关键差异。

### 易错点

- 把教师生成答案、学生做 SFT 也称为严格 OPD。
- 忽略学生 rollout，直接拿固定教师数据做 KL。
- 认为教师 logits 是绝对正确标签。
- 只看 KL 降低，不看真实任务指标和困难集。
- 忽略 tokenizer、模板和多模态视觉输入不兼容。
- 把 OPD 当成 RL，混淆教师监督和 verifier reward。
- 视频教师使用更多帧，却把收益全部归因于蒸馏方法。

### 回答模板

#### 什么是 On-Policy Distillation？

On-Policy Distillation 是让学生模型在自己的策略分布上接受教师监督。学生先根据当前 policy 生成 rollout，教师再在这些学生实际经过的前缀上计算下一 token 的软分布，学生通过 token-level KL 等蒸馏损失去对齐教师。它和普通 Response Distillation 的关键区别是训练状态来自学生自己，因此能够覆盖学生部署时真实会犯的错误，而不是只模仿教师预先生成的标准答案。

#### OPD 和普通知识蒸馏有什么区别？

普通 Response Distillation 通常是教师先生成输入-回答数据，学生对教师回答做 SFT，成本低但不一定覆盖学生自己的错误状态。OPD 则让学生先生成回答，再让教师在学生生成的每个前缀上提供软分布，因此训练信号更贴近学生实际分布。严格的 logit-level OPD 通常需要教师 logits、兼容的 tokenizer 和额外教师推理成本。

#### OPD 和 GRPO 应该怎么区分？

OPD 的监督来源是教师模型的 token 分布，主要作用是能力迁移和局部错误纠偏；GRPO 的监督来源是 verifier 或 reward，主要作用是直接优化任务目标和策略行为。对于关键帧任务，我会先用 SFT 建立基本格式和视频理解，再根据资源选择 OPD 迁移强教师的边界判断，最后用时间、格式和证据 verifier 做 GRPO 修正教师与业务目标之间的差异。

#### 关键帧任务中如何使用 OPD？

我会从 Structured CoT SFT checkpoint 开始，让学生在包含边界难例、局部 UI 和二次刷新的视频上生成自己的 rollout，再让更强教师在相同视频、相同帧采样和相同 task prompt 下提供 token-level 监督。训练时不能只优化语言分布，还要结合时间误差、格式合法性和 UI 证据一致性验证，避免学生只学会教师的表达风格，却没有真正改善首次完成帧判断。
