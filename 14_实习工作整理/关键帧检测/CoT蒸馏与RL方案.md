# 关键帧检测：CoT 蒸馏与 GRPO/RL 方案

## 知识点解析

### 方案要解决什么问题

Direct SFT 适合解决：

```text
模型会不会看视频？
模型会不会遵循任务说明？
模型会不会按固定格式输出？
模型能不能大致预测完成时间？
```

但关键帧任务的主要难点不只是“输出一个时间”，而是判断**第一次满足完成态的边界**。只用最终时间标签做 SFT，模型看不到以下监督：

- 为什么前一段不能选。
- 为什么当前时间是第一次满足。
- 为什么后面的时间只是稳定延续。
- 哪个 UI 区域是真正的目标证据。
- 哪些变化属于排除项。
- 哪些局部变化可以豁免。
- 当前样本的 GT 是否本身存在标注误差。

所以这套方案的目标不是让模型输出更长的解释，而是把隐藏在人工判断中的过程拆成可验证的训练信号：

```text
任务规则理解
  -> 视频时间轴理解
  -> 目标区域定位
  -> 候选边界比较
  -> 首次完成态判断
  -> 结构化答案
```

### 为什么 SFT 不够

#### 1. 只有终点监督，没有边界监督

普通关键帧 SFT 的形式是：

```text
视频 + task_type -> {"time": 6.97}
```

模型只知道目标时间，不知道：

```text
6.53 为什么不行？
6.97 为什么第一次满足？
10.00 为什么只是后续稳定？
```

当相邻帧差异很小、页面存在二次刷新或局部异步加载时，单点标签不足以教会模型边界判断。

#### 2. 模型容易学到“答案模式”，而不是视觉证据

如果训练数据只重复“某类任务通常在某个时间范围完成”，模型可能利用任务先验猜时间，而不是观察视频。这会导致：

- 换视频后泛化差。
- task_type 相似时互相干扰。
- 预测时间集中在常见区间。
- 低 FPS 或新 UI 出现时错误增多。

#### 3. SFT 没有在线探索

SFT 只能模仿已有答案。它不会主动尝试：

```text
先看全视频
  -> 找候选窗口
  -> 比较候选前后帧
  -> 发现证据不足再扩大窗口
```

GRPO/RL 的价值是让当前策略生成多个候选判断，再用可验证 reward 强化更好的路径。

#### 4. CoT 本身也可能是脏的

教师模型生成的解释不一定是真实视觉证据，可能出现：

- 看到 GT 后事后合理化。
- 描述了画面中不存在的 UI 元素。
- 时间顺序与视频相反。
- 前后证据互相矛盾。
- 用很多重复句子掩盖没有判断依据。

因此不能“生成 CoT 后直接训练”，必须先做质量过滤和隐藏校验。

### 目标能力拆解

这套方案要训练的不是一个模糊的“推理能力”，而是四项可观察能力：

| 能力 | 具体问题 | 训练信号 |
| --- | --- | --- |
| Temporal Understanding | 视频中什么时候发生了状态转移 | State/Event Primitive |
| Task Grounding | 当前 task_type 要看哪个区域和条件 | task understanding、target region |
| Boundary Decision | 哪个时间第一次满足，前后为什么不同 | before/current/after evidence |
| Structured Output | 能否稳定输出可解析答案 | schema、format reward |

### Video Primitive：先学会看时间轴

Video Primitive 是视频理解的最小时间单元，不直接等于最终答案。

#### State Primitive

描述一段相对稳定的状态：

```text
时间：0.00-2.00
状态：页面稳定停留在登录选择界面，显示两个登录入口。
```

它回答：

```text
这一段时间页面处于什么状态？
```

#### Event Primitive

描述一次状态转移：

```text
时间：2.00-2.70
事件：用户选择手机号登录。
before：页面停留在登录选择页。
during：页面切换，输入框获得焦点，键盘弹出。
after：页面稳定停留在手机号登录页。
```

它回答：

```text
什么事件发生了？
变化前是什么？
变化过程中发生了什么？
变化后进入什么状态？
```

### 为什么 Primitive 和任务判断要解耦

第一阶段只描述客观视频内容，不判断：

- 是否完成。
- 哪一帧是 GT。
- 哪个 task_type 应该选哪里。
- 哪个时间是最终答案。

这样做有三个好处：

1. 把“看懂视频”和“按业务规则做判断”拆开。
2. 同一份视频时间轴可以被多个 task_type 复用。
3. 降低教师模型围绕答案编造解释的风险。

注意，Primitive 是**理解单元**，不是要求模型在每一步推理中复述完整视频。问题驱动的推理只应引用与当前任务相关的少量状态和事件。

### Evidence Chain：再学会用证据回答

关键帧任务的 evidence chain 可以组织为：

```text
任务理解
  -> 目标区域
  -> 必须满足的证据
  -> 边界前观察
  -> 边界处观察
  -> 边界后稳定性
  -> 排除项检查
  -> 最终时间
```

推荐的结构化样本：

```json
{
  "task_understanding": "定位首屏完成加载并稳定的最早时间",
  "target_region": "首屏商品区域、权益栏和底部导航",
  "required_evidence": [
    "核心商品图文完成渲染",
    "页面主干停止位移",
    "后续没有核心内容二次替换"
  ],
  "candidate_observation": [
    {
      "time": 6.53,
      "status": "not_satisfied",
      "evidence": "商品区域仍在刷新，部分图片为空白占位。"
    },
    {
      "time": 6.97,
      "status": "satisfied",
      "evidence": "核心商品图文和权益栏清晰显示，页面停止位移。"
    },
    {
      "time": 10.00,
      "status": "stable_after",
      "evidence": "后续没有核心区域二次刷新或内容替换。"
    }
  ],
  "boundary_check": "6.53 尚未完成，6.97 首次满足，10.00 只是后续稳定。",
  "answer": {
    "time": 6.97
  }
}
```

这种结构比长篇自由文本更适合训练，因为每个字段都有明确作用，也能被程序检查。

## CoT 数据怎么得到

### 数据来源

CoT 数据不应该从单一来源直接批量生成，建议组合：

```text
高质量人工标注样本
  + 已有 Direct SFT 数据
  + 当前模型 bad case
  + 边界窗口 hard negative
  + Video Primitive 数据
  + 强教师模型生成结果
```

各类数据作用不同：

| 数据 | 作用 |
| --- | --- |
| 人工高质量样本 | 提供可靠完成态和业务标准 |
| Direct SFT 样本 | 保证基础任务覆盖 |
| bad case | 针对性补模型现有短板 |
| hard negative | 训练模型区分“接近完成”和“首次完成” |
| Primitive | 提供可复用时间轴理解 |
| 强教师结果 | 补充边界解释和复杂样本推理 |

### 数据质量调研带来的结论

关于多模态 CoT 数据质量，调研中的公开方案大致分成三类：

| 路线 | 人工介入方式 | 对当前项目的启发 |
| --- | --- | --- |
| 自动生成 + 测试集人工校正 | 训练数据主要自动生成，人工集中修正评测集 | 适合快速扩充训练数据，但必须建立高质量评测集 |
| 自动 CoT + 失败样本抽查 | 训练 CoT 自动生成，人工抽查困难或失败样本 | 适合把人工成本集中到边界和长尾问题 |
| 逐步骤专家审核 | 对每一步推理、错误原因和证据进行人工审核 | 质量高但成本大，适合冷启动金标和小规模验证集 |

调研中有方案报告：自动生成的时间标注里，约 `93.82%` 与人工修正结果的时间区间 IoU 大于 `0.5`。这个结果不能直接当成关键帧项目的效果承诺，但说明一个重要原则：

```text
人工不应该逐条重写所有自动 CoT。
应先测自动数据的可用率，再把人工集中到高价值位置。
```

推荐把人工投入分成三个优先级：

```text
P0：冷启动金标 + 高质量 CoT 评测集
  -> 必须人工确认，保证训练和评测标准可靠。

P1：困难集和失败样本
  -> 重点审核 early/late、二次刷新、局部异步、小 UI 元素和证据矛盾。

P2：普通自动生成训练集
  -> 以程序校验、教师复核和抽样审核为主，不逐条人工重写。
```

人工的主要价值不在于代替模型生成文字，而在于：

- 定义正确的任务边界。
- 建立可复用的质量标准。
- 发现自动生成器的系统性错误。
- 构造可信的困难评测集。
- 校正 reward 和 verifier。

### 第一步：定义教师输入

教师模型至少需要看到：

```text
视频
task_type
任务完成态标准
排除条件
豁免条件
输出 schema
```

教师不应在可见 prompt 中看到：

```text
“GT 是 6.97 秒”
“答案必须选 6.97 秒”
“人工标注认为这里正确”
```

可以在数据生成系统内部保存参考边界，用于后验校验和候选窗口定位，但不能让教师直接围绕答案编造解释。

更稳的实现是：

```text
教师先独立观察视频和任务规则
  -> 输出时间轴、候选边界和最终判断
  -> 再用 GT/规则做隐藏校验
  -> 通过后才形成训练样本
```

### 第二步：生成任务无关 Primitive

让教师先生成完整或局部的视频时间轴：

```text
稳定状态：
  时间段 + 客观视觉描述

状态事件：
  时间段 + before/during/after
```

约束：

- 覆盖视频关键时间段。
- 时间段不能大面积重叠。
- 描述必须来自画面。
- 不输出 GT、关键帧、完成态结论。
- 不为了凑长度重复描述静止画面。
- 状态和事件边界要有视觉意义。

### 第三步：生成问题驱动 CoT

将 task_type 规则和 Primitive 交给教师，让教师只抽取与当前问题有关的证据：

```text
当前任务要看什么？
  -> 哪些 Primitive 相关？
  -> 哪些候选时间值得比较？
  -> 前一候选为什么不满足？
  -> 当前候选满足了哪些条件？
  -> 后续是否保持稳定？
  -> 最终输出什么时间？
```

不要把所有 Primitive 原文全部塞给学生，也不要让 CoT 变成完整视频 caption。真正有价值的是：

```text
少量相关证据 + 清楚的边界判断
```

### 第四步：生成不同难度数据

建议分层：

#### Easy

- 完成态明显。
- 前后帧差异大。
- 没有二次刷新。
- 单一目标区域。

作用：让模型学会基本格式和任务定义。

#### Medium

- 过渡态持续时间较长。
- 局部组件异步加载。
- 需要区分豁免和必决条件。
- 任务需要看多个区域。

作用：学习业务规则组合。

#### Hard

- GT 前后视觉差异很小。
- 存在 loading/白膜/骨架屏。
- 有二次刷新或内容替换。
- 早期假停稳、后期真实完成。
- 多个相似候选事件。

作用：学习边界判定和抗干扰能力。

训练时不建议只堆 Hard 样本。更合理的是：

```text
Easy 建立格式和基础能力
  -> Medium 学规则组合
  -> Hard 学边界和纠错
```

## CoT 数据怎么过滤

### 过滤总原则

CoT 样本必须同时满足：

```text
结构合法
  + 时间可信
  + 证据可见
  + 前后逻辑闭环
  + 没有答案来源泄漏
  + 长度适中
```

### 第一层：视频和 Primitive 质量

过滤：

- 视频无法解码。
- 时间轴不完整。
- 状态/事件边界混乱。
- 描述与画面不一致。
- 出现大量重复 caption。
- 时间信息与视频时序矛盾。

Primitive 质量不合格，后面 CoT 写得再好也不能进入训练。

### 第二层：结构校验

检查：

- 必需字段是否存在。
- 时间是否是合法数字。
- `candidate_observation` 是否至少包含 before/current/after 三类证据。
- `answer` 是否能解析。
- 标签是否闭合。
- JSON、XML 或结构化文本是否符合 schema。

### 第三层：时间校验

检查：

```text
最终 answer 时间
  是否落在候选窗口？
是否满足允许误差？
是否早于视频结束？
是否和 before/current/after 的顺序一致？
```

如果参考 GT 可能有噪声，不能简单把所有偏差都判成模型错误。应分成：

```text
gt_correct
  -> 进入训练或评测

ambiguous
  -> 降低权重、人工抽检或单独训练

reject
  -> 进入审计，不进入训练
```

### 第四层：证据闭环

必须能回答：

```text
前面为什么不满足？
当前看到了什么？
为什么是第一次满足？
后续有没有推翻当前判断？
```

以下情况拒收：

- 只说“页面完成了”，没有视觉证据。
- 证据描述的时间和 answer 不一致。
- 当前证据其实发生在 answer 之后。
- before 已经满足，但答案仍选更晚时间。
- after 发生二次刷新，却声称一直稳定。

### 第五层：来源泄漏

CoT 中禁止出现：

```text
GT
人工标注
参考答案
内部候选
给定时间
标签说
```

也要检查隐式泄漏：

- 教师直接复述输入中的参考时间。
- 时间点精确到不合理程度但没有对应视觉证据。
- 解释先写结论，再补一句泛化理由。
- 所有样本都套完全相同的证据句式。

### 第六层：长度和重复

CoT 不是越长越好。过滤：

- 重复描述同一 UI 状态。
- 与任务无关的全视频 caption。
- 大量“因此”“可以判断”但没有新证据。
- 解释过长导致 answer 被截断。
- 为了展示思考而引入不必要的猜测。

建议目标：

```text
短任务：
  结论 + 1~2 条关键证据

边界任务：
  before + current + after + boundary_check
```

### 数据质量闭环

CoT 数据质量不能只在生成后检查一次，应形成闭环：

```text
自动生成 CoT
  -> 规则和结构校验
  -> 教师/模型质量评分
  -> 抽样人工审核
  -> 建立困难评测集
  -> 训练 Direct/CoT SFT
  -> 跑固定评测和 bad case
  -> 反查 CoT 错误类型
  -> 更新生成提示、过滤器和 reward
```

需要分别维护三类数据：

| 数据集 | 目的 | 是否进入训练 |
| --- | --- | --- |
| Cold-start gold set | 建立高可信任务标准和初始行为 | 是 |
| CoT training set | 扩大证据链和边界样本覆盖 | 通过过滤后进入 |
| CoT evaluation set | 衡量证据真实、边界正确和格式稳定 | 不直接进入训练 |

评测集必须和训练集隔离。尤其不能把人工修正后的困难 CoT 全部回流训练，否则会失去评测集对泛化能力的检验作用。

### 如何判断自动 CoT 是否值得进入训练

可以对自动样本计算质量分，而不是只用一个“通过/拒绝”：

```text
Q =
  w_structure * structure_score
  + w_visual * visual_grounding_score
  + w_time * temporal_alignment_score
  + w_boundary * boundary_score
  - w_leakage * leakage_penalty
  - w_redundancy * redundancy_penalty
```

其中：

- `structure_score`：字段和格式是否完整。
- `visual_grounding_score`：描述是否能在对应时间看到。
- `temporal_alignment_score`：时间段和事件是否一致。
- `boundary_score`：是否解释 before/current/after。
- `leakage_penalty`：是否暴露 GT 或答案来源。
- `redundancy_penalty`：是否重复、冗长、无新增信息。

质量分可以用于：

```text
高分 clean：
  直接进入 CoT SFT。

中间分 ambiguous：
  降采样比例、降低 loss 权重或人工抽检。

低分 reject：
  进入审计集，不进入训练。
```

### 人工应该审核什么

人工审核不应只看“这段话写得像不像”，而应按证据和边界审核：

1. **任务理解**：是否真的理解 task_type 要找的完成态。
2. **目标区域**：是否关注正确 UI 区域。
3. **时间边界**：当前时间是否是第一次满足，而不是更晚稳定帧。
4. **视觉证据**：描述的文字、按钮、图片、loading 是否在画面中存在。
5. **前后关系**：before 未完成、current 满足、after 没有推翻。
6. **排除/豁免**：是否错误等待了非关键变化，或忽略了必须等待的小元素。
7. **答案格式**：最终答案是否和证据、时间一致。

人工审核结果还要沉淀为错误 taxonomy，后续用于：

```text
bad case 分桶
  + 采样困难数据
  + 更新教师 prompt
  + 更新 verifier
  + 更新 reward 权重
```

## 完整训练方案

### 目标

训练目标不是简单提高 CoT 文本质量，而是：

```text
时间更准
  + 边界更稳
  + 证据更真实
  + 格式可解析
  + 训练和推理成本可控
```

### 阶段一：Base/Prompt Baseline

先用 base model 和固定 prompt 在固定测试集上跑 baseline。

记录：

- `frame_err`。
- `time_err`。
- PASS 率。
- 格式解析率。
- early/late 偏差。
- 各 task_type 指标。
- bad case 分类。

没有 baseline，就无法证明 CoT 或 RL 带来了真实收益。

### 阶段二：Direct SFT 冷启动

先用短答案训练：

```text
视频 + 任务规则
  -> <answer>{"time": ...}</answer>
```

目的：

- 学会基础视频输入。
- 学会 task_type 遵循。
- 学会时间输出格式。
- 建立一个稳定的 policy。

Direct SFT 的准出条件：

- 小数据可以过拟合。
- 格式解析率稳定。
- dev 时间误差优于 base/prompt baseline。
- 没有明显业务线灾难性退化。

### 阶段三：Structured CoT SFT

在 Direct SFT 基础上，加入少量高质量结构化 CoT：

```text
视频 + 任务规则
  -> task understanding
  -> target region
  -> required evidence
  -> before/current/after
  -> boundary check
  -> answer
```

为什么放在 Direct SFT 后：

- 模型先具备基本视频和格式能力。
- CoT 样本只负责补边界证据，不承担全部基础能力。
- 训练更容易判断收益来自 CoT，而不是输入链路首次跑通。

数据配比不要固定照搬。可以从：

```text
Direct SFT 数据为主
  + 少量 clean CoT
  + 中等比例 hard boundary CoT
```

开始做消融，再根据格式率、长度和业务指标调整。

### 阶段四：GRPO/RL with Verifiers

当 Structured CoT SFT 已经稳定后，再进行 GRPO/RL。

#### GRPO 的训练循环

对每个 prompt：

```text
当前 policy 生成 G 个回答
  -> 解析每个回答
  -> 计算多项 reward
  -> 组内归一化 reward
  -> 优化高于组平均的回答
  -> KL 约束 policy 不要偏离 reference 太远
```

组内优势可以简化为：

```text
A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
```

GRPO 不需要额外训练 Critic，但需要：

- 可靠 verifier。
- 足够有区分度的 group reward。
- 合理的采样温度和生成长度。
- KL/更新幅度控制。

#### 为什么 GRPO 放在 CoT SFT 后

如果模型还不会稳定输出合法 CoT，RL 阶段的大量 rollout 会变成：

- 无法解析。
- 没有视觉证据。
- 只输出猜测时间。
- 生成重复长文本。
- reward 大量相同，组内没有学习信号。

CoT SFT 先建立“会观察、会引用、会输出”的冷启动能力，GRPO 再优化“哪种观察和边界判断更有效”。

### Reward 设计

不要只用一个总 reward，应该拆成可诊断的分项：

```text
R_total =
  w_answer * R_answer
  + w_time * R_time
  + w_format * R_format
  + w_evidence * R_evidence
  + w_boundary * R_boundary
  - w_length * P_length
  - w_hallucination * P_hallucination
```

#### `R_answer`

判断最终任务结论是否正确：

- 关键帧时间是否在允许误差内。
- 断言是否正确。
- 无完成态时是否正确输出负结果。

#### `R_time`

针对关键帧时间：

```text
误差越小，reward 越高
超过阈值，reward 快速下降
过早和过晚可以分别设置惩罚
```

如果业务标准是 `frame_err <= 3` 或 `time_err <= 100ms`，reward 应该与这个准出标准一致，而不是优化一个和业务无关的平均距离。

#### `R_format`

检查：

- 是否有合法 answer 标签。
- JSON 是否可解析。
- `time` 类型是否正确。
- 是否输出多余字段或 Markdown。

格式 reward 不能压过答案 reward，否则模型可能学会“格式正确但时间错误”。

#### `R_evidence`

检查 CoT 中引用的证据是否真的存在：

- 目标区域是否正确。
- UI 元素是否在对应时间可见。
- 事件时间是否和视频一致。
- 证据是否支持最终结论。

#### `R_boundary`

重点奖励：

```text
before 未完成
current 首次满足
after 稳定或不推翻
```

这项 reward 是关键帧任务区别于普通时间预测的核心。

#### `P_length`

惩罚：

- 无效重复。
- 不必要的长解释。
- 生成超出预算。
- 只写模板化理由。

长度惩罚不能设置过强，否则模型可能跳过必要证据。

### 可选阶段：OPD

OPD（On-Policy Distillation）可以作为强 teacher 和小 student 之间的可选实验路线：

```text
student policy 生成自己的 rollout
  -> teacher 对这些 rollout 计算 token-level 分布
  -> 用 teacher soft distribution 监督 student
```

它和普通离线蒸馏的区别是：

- 离线蒸馏：student 学 teacher 预先生成的答案。
- OPD：student 先暴露自己真实会犯的错误，再让 teacher 对这些状态给分布监督。

OPD 可能有价值的原因：

- student 的错误状态更贴近部署真实分布。
- teacher 分布比硬标签更平滑。
- 可以补充边界判断和证据表达。

但它不是默认必做：

- 需要 teacher rollout 或 token-level logits，资源成本高。
- teacher 和 student 任务分布、词表、模板要兼容。
- 训练稳定性和收益需要实验验证。
- 不能在 Direct SFT/CoT SFT 尚未稳定时引入。

建议顺序：

```text
Direct SFT 稳定
  -> Structured CoT SFT 有明确收益
  -> reward verifier 可靠
  -> 仍有能力差距且资源允许
  -> 再评估 OPD
```

## 冷启动和准入准出标准

### 冷启动检查

进入训练前确认：

- 视频能解码。
- 模态占位符和视频输入能对应。
- 单条样本 forward/loss 正常。
- assistant 标签确实参与 loss。
- 小样本能过拟合。
- 输出格式能解析。
- 评测脚本能读取输出。

### Direct SFT 准出

- 格式正确率稳定。
- dev 时间误差优于 base baseline。
- 关键业务线没有明显退化。
- 训练 loss 和 dev 指标趋势合理。
- bad case 已经能按类型归因。

### Structured CoT SFT 准出

- CoT 不是简单变长。
- before/current/after 证据完整。
- 证据和时间一致。
- 无明显 GT 泄漏。
- 目标业务边界错误下降。
- 输出长度仍在可接受范围。

### GRPO 准入

只有满足以下条件才进入：

- SFT policy 已经能稳定生成合法答案。
- verifier 对正确/错误结果有较高可靠性。
- reward 分项和最终业务指标相关。
- 训练资源能支持 rollout。
- 已经有 SFT checkpoint、固定评测集和回滚版本。

## 训练参数怎么设

参数必须从 baseline 和资源约束出发，不存在一套固定最优值。

### CoT SFT 参数

优先关注：

| 参数 | 调整逻辑 |
| --- | --- |
| learning rate | 全参数通常小，LoRA 通常大；先扫 2~3 个量级相近的值 |
| effective batch | 影响梯度稳定和样本覆盖，显存不够用梯度累积 |
| warmup | 初期 spike 增大，短训练不要占比过高 |
| epoch | dev 指标上升可继续，train 降而 dev 降说明过拟合 |
| max length | 覆盖完整 CoT 和 answer，但不要无限放大无效文本 |
| CoT 数据比例 | 先少量加入，观察边界能力和输出长度 |

### GRPO 参数

优先关注：

| 参数 | 调整逻辑 |
| --- | --- |
| policy learning rate | 通常比 SFT 更保守，更新过大容易 KL 飙升 |
| group size | 太小方差大，太大 rollout 成本高 |
| temperature | 太低探索不足，太高格式和答案不稳定 |
| max completion length | 过小截断证据，过大增加成本和重复 |
| KL 系数 | 太强学不动，太弱容易偏离 reference |
| reward 权重 | 先保证 answer/format，再逐步加入 evidence/length |
| rollout batch | 受显存、视频 token 和推理吞吐约束 |

调参顺序：

```text
先固定 reward，验证 rollout 能正确解析
  -> 扫 policy learning rate
  -> 调 group size 和 temperature
  -> 调 KL
  -> 调 reward 权重
  -> 最后调长度和采样预算
```

## 如何判断训练是否有效

不能只看 reward 或 loss：

| 结果 | 可能含义 |
| --- | --- |
| reward 升，真实时间指标不升 | reward hacking 或 reward 与业务不一致 |
| CoT 变长，时间误差不降 | 学到解释模板，没有学到边界 |
| format reward 升，answer reward 不升 | 格式奖励过强 |
| KL 快速升高 | 更新过猛或 KL 约束太弱 |
| KL 很低且 reward 不变 | 更新太保守、采样缺乏区分度或 reward 全相同 |
| response length 持续增长 | reward 鼓励长输出，缺少长度惩罚 |
| 训练指标升，难例变差 | 过拟合简单样本或 hard case 被稀释 |

最终必须看：

```text
固定回归集
  + 困难边界集
  + 分 task_type 指标
  + frame/time error
  + 格式正确率
  + evidence accuracy
  + response length
  + reward 分项相关性
```

## 失败样本如何回流

### 过早预测

补：

- loading/骨架屏 hard negative。
- “主体出现但核心区域未完成”样本。
- before/current 相邻边界样本。
- 明确排除条件的 CoT。

### 过晚预测

补：

- current 已满足、after 只是稳定延续的样本。
- 第一完成帧和更晚稳定帧的对比样本。
- early/late 对称 reward。

### 证据幻觉

处理：

- 降低该教师样本权重或 reject。
- 加强 Primitive 质量过滤。
- 引入证据可见性校验。
- 把“无法确认”作为合法结果，而不是强行编证据。

### 输出过长

处理：

- 缩短 schema。
- 只保留关键证据。
- 加长度惩罚。
- 限制 completion length。
- 训练时混入 Direct SFT 短答案。

## 最终推荐路线

```text
阶段 0：Base / Prompt baseline
  -> 明确当前问题和指标

阶段 1：Direct SFT
  -> 学会看视频、遵循 task_type、输出时间

阶段 2：Primitive 表征数据
  -> 学会 State/Event 和时间锚点

阶段 3：Structured CoT SFT
  -> 学会 target region、before/current/after、boundary check

阶段 4：Verifier-based GRPO/RL
  -> 用时间、格式、证据和长度 reward 优化

阶段 5：可选 OPD
  -> 资源允许且 student 真实错误仍多时，用强 teacher 分布蒸馏

阶段 6：评测和数据飞轮
  -> 困难集、bad case、重标、reward 更新、回归
```

不要一开始就做：

- 全量长 CoT。
- 直接 GRPO。
- 没有 verifier 的 RL。
- 把审计字段全部混进训练答案。
- 用一个总 reward 掩盖各分项问题。

## 面试应对

### 为什么关键帧任务要从 Direct SFT 升级到 CoT 和 GRPO？

回答思路：分别说清 SFT、CoT SFT、GRPO 的监督边界。

回答模板：

Direct SFT 只能告诉模型最终时间，适合建立基础视频理解、任务遵循和输出格式，但不能充分监督“前一段为什么不满足、当前为什么第一次满足、后续为什么只是稳定延续”。结构化 CoT SFT 把这些边界证据显式化，让模型学会 target region、before/current/after 和 boundary check。GRPO 则在模型已经会基本判断的前提下，对同一个 prompt 采样多个回答，用时间、格式、证据和长度等可验证 reward 强化更好的判断路径。所以三者是冷启动、证据学习和在线优化的递进关系。

### CoT 数据是怎么得到的？

回答思路：按人工数据、Primitive、教师生成、过滤和训练集隔离回答。

回答模板：

CoT 数据不是直接让教师围绕 GT 编一段解释。我会先准备视频、task_type、完成态规则和原始标注，再让教师模型抽取任务无关的 Video Primitive，包括稳定的 State 和状态转移的 Event。之后针对具体 task_type，只引用相关 Primitive，生成 task understanding、target region、required evidence、before/current/after 观察、boundary check 和最终答案。生成后还要做视频质量、时间对齐、证据一致性、GT 泄漏、格式和长度过滤，最后把 clean 样本和 reject/审计样本严格分开，只有通过过滤的训练记录进入 CoT SFT。

### 为什么不能直接把完整视频 caption 当 CoT 训练？

回答思路：区分理解单元和推理单元，强调问题驱动引用。

回答模板：

完整视频 caption 主要是理解单元，回答“视频里发生了什么”；关键帧 CoT 是推理单元，回答“针对当前 task_type，哪些证据决定第一次完成边界”。如果把完整 caption 原样塞进每条 CoT，里面会有大量与当前问题无关的信息，增加 token 和噪声，也容易让模型学会复述而不是判断。因此更合理的是先用 State/Event Primitive 建立时间轴，再让问题驱动的 CoT 只引用相关证据。

### GRPO 阶段具体怎么训练？

回答思路：讲清 rollout、verifier、group advantage、policy update 和 KL。

回答模板：

GRPO 阶段对同一个关键帧 prompt 让当前 policy 采样一组回答，每个回答都包含最终时间和结构化证据。然后用 verifier 计算多个 reward：时间误差、格式合法、before/current/after 证据、任务规则遵循和长度惩罚。再用组内 reward 的均值和标准差构造相对 advantage，强化高于组平均的回答，同时用 KL 约束 policy 不要偏离 reference model 太远。关键是 reward 必须和最终业务指标一致，不能只奖励“解释写得长”或“格式看起来像”。

### 为什么 GRPO 不能一开始就做？

回答思路：从探索成本、无效 rollout、reward 区分度和训练稳定性解释。

回答模板：

GRPO 需要模型先能生成基本合法、可验证的回答。如果模型还不会处理视频、不会输出格式，RL 采样出来的结果大多是无法解析或没有证据的无效 rollout，组内 reward 也没有区分度，训练成本很高但信号很弱。因此我会先用 Direct SFT 建立基础能力，再用少量高质量结构化 CoT SFT 教模型表达证据，确认 verifier 可靠后再做小规模 GRPO。这样 RL 优化的是边界判断，而不是从零学习数据格式。

### 如何判断 CoT/RL 带来了真实收益？

回答思路：比较 baseline、分项指标、困难集和成本，防止只看 reward。

回答模板：

我会固定模型、数据、评测集和推理配置，做 Direct SFT、Structured CoT SFT、CoT+GRPO 的逐阶段对照。指标上同时看 frame/time error、PASS、格式正确率、evidence accuracy、early/late 分布、输出长度和训练/推理成本，并单独看困难边界集和各 task_type。如果 reward 上升但真实时间指标不升，说明可能是 reward hacking 或 reward 与业务目标不一致；如果 CoT 变长但边界准确率不升，说明学到的是解释模板，不是真正的视觉判断。

### OPD 在这条路线里应该放在哪里？

回答思路：定位为可选实验，不与主线 SFT/GRPO 混为必经阶段。

回答模板：

OPD 可以放在 Structured CoT SFT 之后、GRPO 之外作为可选实验。它让 student 先按自己的 policy 生成 rollout，再由更强的同源 teacher 对这些 rollout 提供 token-level 分布或软标签，适合弥补 student 当前真实错误状态。它的优点是比单纯模仿 teacher 固定答案更贴近 student 的问题，缺点是需要 teacher 推理和 logits 资源，训练稳定性也需要验证。因此主线应先做 Direct SFT、CoT SFT 和 verifier-based GRPO，只有当 student 仍有明显能力差距且资源允许时再评估 OPD。
