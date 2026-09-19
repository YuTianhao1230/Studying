# Agentic Model Optimization：模型自训练与 Loop-Engineering

## 知识点解析

### 概述

Agentic Model Optimization 是一种由 AI Agent 驱动的模型自迭代优化范式。它把过去依赖人工完成的：

```text
发现 bad case
  -> 分析错误原因
  -> 构造数据 / 修改 Prompt / 调整超参数
  -> 训练
  -> 推理
  -> 评测
  -> 决定下一轮动作
```

变成一个可恢复、可评估、可持续迭代的 Loop-Engineering 系统。

它的核心不是“让 Agent 自己随便训练模型”，而是把模型优化过程拆成：

```text
目标定义
  + 可调用的 Skill
  + 受约束的实验空间
  + 可复现的训练流水线
  + 可靠的评测门禁
  + 可追踪的状态和产物
  -> Agent 自主提出并执行下一轮优化动作
```

在关键帧检测中，Agent 可以围绕某个 `task_type` 自动完成：

```text
筛选 early/late、局部异步、小 UI 元素等 bad case
  -> 判断是 GT、规则、输入还是模型问题
  -> 选择重标、增广、课程学习、Prompt 或超参数策略
  -> 启动 SFT / CoT-SFT / GRPO / OPD 实验
  -> 推理并计算 ACC、时间误差、边界准确率
  -> 通过准出门禁决定继续、回滚或结束
```

### 1. 为什么需要 Agentic Model Optimization

#### 1.1 人工优化的瓶颈

传统模型优化通常需要工程师、算法工程师、数据工程师和标注团队共同协作：

| 环节 | 传统方式 | 主要问题 |
| --- | --- | --- |
| bad case 发现 | 人工抽样或手工看结果 | 覆盖有限，难以发现长尾错误 |
| 错误分析 | 人工阅读预测和视频 | 分析成本高，归因口径不稳定 |
| 数据构造 | 手工重标、手工写样本 | 速度慢，样本策略难以系统探索 |
| Prompt 调整 | 凭经验改描述和规则 | 难以知道收益来自哪处修改 |
| 超参数搜索 | 人工试几个配置 | 搜索空间大，实验记录容易丢失 |
| 训练与推理 | 人工提交任务、找 checkpoint | 等待长，容易重复执行 |
| 评测与决策 | 人工看一张总表 | 容易只看平均指标，忽略回归和长尾 |

真正耗时的不是单次训练，而是大量“分析—修改—等待—对比”的循环。

#### 1.2 Agent 化带来的变化

Agentic Model Optimization 的价值体现在两个层面：

1. **扩大探索范围**：Agent 可以按固定策略批量尝试数据配比、Prompt 版本、学习率、batch size、训练轮数和损失组合。
2. **缩短反馈周期**：Agent 自动复用已有 checkpoint、推理结果和评测产物，把时间集中在真正需要的新实验上。

但 Agent 不能替代业务标准的定义者。完成态、排除条件、豁免条件、指标阈值和高风险变更仍需要由人定义并审核。

### 2. 总体架构

一个可落地的 Agentic Model Optimization 系统可以拆成八层：

```text
用户目标与约束
  -> Loop Orchestrator
  -> Optimization Agent
  -> Skill 路由
  -> 本地训练/推理/评测 Pipeline
  -> Artifact 与 State Store
  -> Evaluator / Gate
  -> DONE、NEXT_ITERATION 或 ROLLBACK
```

#### 2.1 目标与约束层

用户不能只给一句“把 F1 提高”，还需要明确：

```text
优化对象：
  task_type、业务线、模型版本、数据版本

主指标：
  F1、Precision、Recall、ACC、time_error 等

护栏指标：
  全量回归、格式解析率、无完成态误报、推理延迟

目标阈值：
  主指标至少提升多少，护栏指标最多下降多少

预算约束：
  最大迭代轮数、GPU 时长、并发数和单轮实验数量
```

目标定义越明确，Agent 的搜索空间越小，实验结论越可信。

#### 2.2 Loop Orchestrator

Orchestrator 负责：

- 创建和关闭一次优化运行。
- 调度 Agent 产生下一步计划。
- 控制训练、推理、评测任务的先后关系。
- 处理重试、超时、失败和断点恢复。
- 检查锁，避免同一份资源被并发修改。
- 根据 Gate 结果决定 `DONE`、`NEXT_ITERATION` 或 `ROLLBACK`。

它不负责理解所有模型细节，复杂能力应通过 Skill 提供。

#### 2.3 Optimization Agent

Agent 负责把当前状态转换成下一轮可执行计划：

```text
读取当前实验状态
  -> 分析 bad case 和指标变化
  -> 提出一个或多个候选策略
  -> 估计收益、成本和风险
  -> 选择下一轮实验
  -> 调用 Skill 执行
  -> 读取产物并更新判断
```

Agent 的计划必须是结构化的，至少包含：

```text
hypothesis：
  认为当前问题的原因是什么。

action：
  本轮具体修改什么。

expected_effect：
  哪些指标应该如何变化。

control_group：
  与哪个 baseline 对照。

budget：
  预计消耗多少数据、GPU 和时间。

stop_condition：
  什么情况下停止或回滚。
```

#### 2.4 Skill 路由层

Skill 是 Agent 能力的稳定接口，不应把所有操作都暴露为任意 shell 命令。关键 Skill 可以包括：

| Skill | 职责 |
| --- | --- |
| `pipeline` | 准备数据、启动训练、推理、评测和汇总产物 |
| `data-strategy` | 选择样本、构造 hard negative、课程学习和数据配比 |
| `badcase-diagnosis` | 聚类错误、分析模型分歧和判定可能根因 |
| `prompt-strategy` | 修改任务规则、证据优先级和输出 schema |
| `hparam-optimizer` | 在允许的搜索空间内提出超参数实验 |
| `evaluator` | 计算主指标、分桶指标和护栏指标 |
| `gatekeeper` | 根据准入准出规则决定是否继续 |
| `artifact-manager` | 注册 checkpoint、数据版本、推理结果和评测报告 |

Skill 需要显式声明输入、输出、失败状态和副作用，避免 Agent 只知道“能做什么”，却不知道“做完以后留下什么”。

#### 2.5 执行层

执行层可以使用现有训练框架和脚本，例如：

```text
训练：
  swift sft 或自定义 SFT/RL/OPD 入口

推理：
  swift infer 或分片推理入口

评测：
  default_eval.py、结构化 verifier 和分桶统计

产物：
  checkpoint、infer_result、metrics、日志、配置和摘要
```

Agent 不应该绕过统一 Pipeline 直接拼命令，因为这样会造成：

- 训练参数无法复现。
- 输入输出路径不一致。
- 产物没有注册。
- 失败状态无法被 Loop 识别。

#### 2.6 状态与产物层

状态文件需要记录一次运行的最小可恢复信息：

```json
{
  "run_id": "keyframe_agentic_iter_006",
  "status": "EVALUATING",
  "parent_run_id": "keyframe_agentic_iter_005",
  "model_version": "qwen3-vl-sft-v3",
  "dataset_version": "keyframe-data-v12",
  "prompt_version": "keyframe-prompt-v7",
  "strategy": {
    "type": "curriculum_badcase_mix",
    "parameters": {
      "hard_ratio": 0.35,
      "replay_ratio": 0.20
    }
  },
  "checkpoint_path": "...",
  "infer_result_path": "...",
  "eval_result_path": "...",
  "lock_owner": "...",
  "attempt": 1
}
```

状态与产物必须分离：

- 状态描述“现在进行到哪一步”。
- 产物描述“已经生成了什么结果”。
- 日志描述“为什么得到这个结果”。

### 3. Loop 状态机

一个完整 Loop 可以抽象成以下状态：

```text
INIT
  -> BASELINE_READY
  -> DIAGNOSING
  -> PLAN_READY
  -> DATA_READY / STRATEGY_READY
  -> TRAINING
  -> INFERENCING
  -> EVALUATING
  -> ANALYZING
  -> DONE
  -> NEXT_ITERATION
  -> ROLLBACK
  -> FAILED
```

#### 3.1 每个状态的职责

| 状态 | 必须完成的事情 |
| --- | --- |
| `INIT` | 读取目标、预算、模型和数据版本 |
| `BASELINE_READY` | 固定 baseline、主指标和护栏指标 |
| `DIAGNOSING` | 读取 bad case、分歧、日志和历史实验 |
| `PLAN_READY` | 形成有假设、对照和停止条件的实验计划 |
| `DATA_READY` | 生成数据版本并完成质量门禁 |
| `STRATEGY_READY` | 固定 Prompt、超参数和训练策略版本 |
| `TRAINING` | 训练并检查日志、loss、资源和 checkpoint |
| `INFERENCING` | 在固定数据和线上一致配置上推理 |
| `EVALUATING` | 计算主指标、分桶指标和护栏指标 |
| `ANALYZING` | 判断收益来源、回归原因和下一步策略 |
| `DONE` | 达到目标并保存完整实验摘要 |
| `NEXT_ITERATION` | 生成下一轮计划，保留当前最优版本 |
| `ROLLBACK` | 恢复到上一个可用 checkpoint 或数据策略 |
| `FAILED` | 记录可重试、需人工处理或不可恢复原因 |

#### 3.2 断点续跑

断点续跑不能只根据“目录里有文件”判断。至少需要验证：

```text
文件存在
  + 文件归属 run_id 一致
  + 配置 hash 一致
  + 产物未截断
  + 任务状态不是 FAILED
  + 结果可以被下游解析
```

例如只有 checkpoint 没有评测结果时，可以跳过训练继续推理或评测；只有推理结果但数据版本不一致时，不能直接复用。

#### 3.3 并发互斥

同一模型、同一数据版本和同一输出目录不能被多个 Agent 同时写入。需要：

- run lock。
- 过期锁检测。
- PID 或任务 ID 记录。
- 失败后可人工接管。
- 产物写入临时目录，完成后原子提交。

### 4. Agent 如何接管模型优化

#### 4.1 Bad case 发现

Agent 可以从四类信号发现候选样本：

```text
模型与 GT 不一致
  + 多模型预测分歧
  + 同一模型多次采样分歧
  + 高优业务线低分
  + 预测置信度低或结构解析失败
```

关键帧任务中还要细分：

- early：过早判断完成。
- late：等待过久才判断完成。
- missed：漏掉短暂状态或小 UI 元素。
- refresh：忽略了核心内容二次刷新。
- rule-following：没有遵循 task_type 规则。
- evidence-hallucination：证据描述与视频不一致。
- format：输出无法解析。

Agent 不应只输出“这批样本错了”，而应输出：

```text
样本分桶
  -> 错误模式
  -> 可能根因
  -> 建议动作
  -> 是否需要人工确认
```

#### 4.2 数据层策略

Agent 可搜索的数据策略包括：

| 策略 | 作用 |
| --- | --- |
| bad case oversampling | 提高高价值错误样本的训练权重 |
| hard negative | 强化相邻帧、伪完成和二次刷新边界 |
| curriculum learning | 从明显样本逐步过渡到复杂长尾样本 |
| replay buffer | 保留旧任务和高质量基础样本，避免遗忘 |
| relabel | 修正确认错误的 GT |
| augment | 对视频、时间窗口和结构化证据做受控增广 |
| task balancing | 防止高数据量 task_type 主导训练 |
| business weighting | 提高高优业务线的训练和评测权重 |

数据策略必须同时维护：

```text
新增样本比例
  + bad case 比例
  + clean replay 比例
  + hard negative 比例
  + 各业务线 / task_type 配比
```

不能只提高 bad case 比例，否则模型可能过度适应错误分布，导致全量能力下降。

#### 4.3 Prompt 与规则策略

Prompt 可以被 Agent 优化，但必须版本化。重点搜索空间包括：

- 任务目标是否明确。
- 完成态条件是否拆解。
- 排除条件和豁免条件是否冲突。
- 证据优先级是否明确。
- 是否要求 before/current/after。
- 输出 JSON/XML schema 是否稳定。
- 是否加入典型反例和边界例。

Prompt 改动应遵循：

```text
提出规则假设
  -> 只改一个主要因素
  -> 在固定评测集验证
  -> 分析是遵循性提升还是数据分布变化
```

不要把大量规则、few-shot、输出格式和采样参数同时修改，否则无法归因。

#### 4.4 超参数策略

超参数搜索应被视为受约束实验，而不是随机试错。搜索空间可以包括：

```text
learning_rate
batch_size
gradient_accumulation_steps
lr_scheduler
warmup_ratio
num_train_epochs
max_frames
max_pixels
hard_sample_ratio
replay_ratio
loss_weight
```

Agent 每轮应记录：

```text
变更参数
  + 旧值 / 新值
  + 预期影响
  + 实际影响
  + 成本
  + 是否保留
```

优先采用：

1. 低成本 smoke test。
2. 单变量或少变量消融。
3. 失败后收缩搜索空间。
4. 只在主指标和护栏指标都可接受时晋级。

#### 4.5 训练范式选择

Agent 不应把所有问题都转成更复杂的训练方法。可以按能力缺口选择：

| 现象 | 优先策略 |
| --- | --- |
| 不会遵循任务或输出格式不稳定 | Direct SFT |
| 会答但边界证据不足 | Structured CoT SFT |
| 有多个候选且 reward 可验证 | RFT / GRPO |
| 完整视频漏掉局部时间证据，且有 Teacher logits | Temporal-OPSD |
| 数据和规则都不稳定 | 先修数据治理和 Prompt |

训练范式升级必须有准入条件，否则 Agent 容易为了“继续迭代”盲目增加 RL、蒸馏或复杂损失。

### 5. 评测与准出门禁

#### 5.1 评测对象

每次迭代至少包含：

```text
固定回归集
  + 高优业务集
  + 低 ACC 指标集
  + bad case 困难集
  + early/late 集
  + 无完成态集
  + 标准冲突和 GT 疑似错误集
```

训练数据可以变化，但固定回归集不能随意替换，否则无法判断模型是否真正提升。

#### 5.2 指标分层

主指标和护栏指标必须分开：

| 层级 | 指标示例 | 作用 |
| --- | --- | --- |
| 业务主指标 | Precision、Recall、F1、ACC | 判断目标能力是否提升 |
| 边界指标 | early、late、time_error、frame_error | 判断关键帧定位是否改善 |
| 证据指标 | before/current/after、evidence accuracy | 判断过程是否可信 |
| 稳定性指标 | 格式解析率、重复率、空答案率 | 防止输出退化 |
| 回归指标 | 全量、各业务线、各 task_type | 防止局部提升换来全局退化 |
| 工程指标 | GPU 时长、吞吐、延迟、失败率 | 判断是否值得上线 |

#### 5.3 Gate 规则

一个可执行的 Gate 至少要回答：

```text
主指标是否达到最小提升？
高优业务是否提升？
全量回归是否在容忍范围？
输出格式是否稳定？
是否出现严重误报或幻觉？
成本是否超过预算？
```

示例：

```text
LOCAL_DONE：
  困难集 F1 提升 >= 2pp
  且全量 F1 下降 <= 0.5pp
  且格式解析率不下降
  且无完成态误报率不超过阈值

LOCAL_NOTDONE：
  主指标未达到目标，但仍有可验证的下一步策略

ROLLBACK：
  主指标提升但护栏严重退化
  或模型利用评测漏洞获得虚假收益
```

阈值应配置化，不能写死在 Agent 的自然语言判断中。

#### 5.4 防止指标投机

Agent 可能通过以下方式获得虚假收益：

- 过度输出正例，提高 Recall 但破坏 Precision。
- 只优化某个困难集，牺牲全量能力。
- 利用数据泄漏或重复样本。
- 修改评测口径而不是提升模型。
- 过度依赖 reference answer 或文本相似度。
- 通过更长输出骗取格式或证据分数。

防护方法：

```text
固定评测集和版本
  + 训练/评测实体隔离
  + 多粒度指标
  + 评测脚本版本化
  + 人工抽检高分样本
  + 主指标与护栏指标同时过门
```

### 6. 实验治理与可靠性

#### 6.1 数据血缘

每个训练结果必须能反查：

```text
模型 checkpoint
  <- 训练配置
  <- Prompt 版本
  <- 数据集版本
  <- 样本筛选规则
  <- bad case 来源
  <- 评测脚本版本
```

没有数据血缘，Agent 只能“找到一个看起来更好的 checkpoint”，无法解释为什么更好，也无法安全复现。

#### 6.2 实验记录

实验记录至少包含：

```text
run_id / parent_run_id
模型、Tokenizer、Processor 版本
训练代码和评测代码版本
数据和 Prompt 版本
完整超参数
随机种子
资源规格和实际成本
主指标、护栏指标和样本数量
Agent 假设、动作和结论
```

#### 6.3 失败分类

失败不能只记为“训练失败”，应分类：

| 类型 | 例子 | 后续动作 |
| --- | --- | --- |
| 数据失败 | 视频损坏、字段缺失、标签越界 | 修复数据链路 |
| 规则失败 | 完成态冲突、评测口径变更 | 人工确认标准 |
| 训练失败 | loss NaN、显存不足、任务超时 | 调整训练配置或资源 |
| 推理失败 | 解析失败、输出截断、视觉输入异常 | 检查推理配置 |
| 评测失败 | 脚本异常、数据版本不匹配 | 阻止结果进入比较 |
| 策略失败 | 指标无提升、护栏退化 | 回到诊断阶段 |
| 环境失败 | 依赖、权限、路径或服务异常 | 修复运行环境 |

只有可重试的失败才允许 Agent 自动重试；标准冲突和高风险变更必须升级给人。

#### 6.4 成本与停止

Agent 需要有明确的预算：

```text
单轮最大 GPU 时长
单轮最大训练次数
最大并发数
最大总迭代轮数
失败重试次数
```

停止条件包括：

- 达到主指标目标。
- 连续若干轮没有有效提升。
- 最优 checkpoint 已稳定。
- 预算耗尽。
- 发现评测或数据存在系统性问题。
- 后续候选策略的预期收益低于成本。

### 7. 人在闭环中的位置

Agent 自主优化不等于完全无人值守。建议把人工参与点分成三类：

#### 自动执行

- 读取日志和评测结果。
- 聚类 bad case。
- 生成候选数据策略。
- 运行低风险 smoke test。
- 复用已有产物。

#### 人工审核后执行

- 修改完成态、排除条件和豁免条件。
- 大规模 relabel。
- 解冻视觉编码器或改变模型结构。
- 修改 reward、评测脚本和准出阈值。
- 发布新的线上模型。

#### 禁止 Agent 单独决定

- 删除原始数据或固定评测集。
- 覆盖线上 checkpoint。
- 放宽评测阈值以制造达标。
- 将未审核的伪标签直接作为真值。
- 绕过权限、锁和审计记录。

### 8. 在关键帧检测中的落地

#### 8.1 一轮 Agentic 优化流程

```text
输入：
  task_type、当前 checkpoint、数据版本、Prompt 版本、目标指标

1. 固定 baseline：
  在全量、业务线、task_type 和困难集上评测。

2. 发现 bad case：
  聚类 early、late、missed、refresh、rule-following 和 format 错误。

3. 归因：
  先检查视频/时间链路，再检查 GT/规则，最后归因模型。

4. 选择策略：
  数据重标、hard negative、课程学习、Prompt、超参数或训练范式。

5. 生成实验：
  固定对照，只改变一个主要因素，注册 run_id。

6. 执行训练和推理：
  保存 checkpoint、日志和 infer_result。

7. 分桶评测：
  计算 ACC、F1、time_error、early/late、证据和格式指标。

8. Gate：
  达标则保存最优版本；未达标则分析下一轮；退化则回滚。
```

#### 8.2 与现有卡片的关系

```text
任务定义与标注标准：
  提供完成态、排除条件、豁免条件和证据优先级。

BadCase 归因与 UI 元素治理：
  提供错误类型、UI 元素和小区域治理方法。

数据飞轮与主动学习：
  提供样本筛选、人工重标、回流训练和分桶评测。

并行 DE 与 PE：
  让数据标准和 Prompt/规则同步演进。

Agentic Model Optimization：
  在上述能力之上增加 Agent 规划、策略搜索、执行、状态恢复和自动 Gate。

方案与训练流程：
  提供 SFT、CoT-SFT、GRPO、OPD 等训练阶段的具体实现。
```

#### 8.3 推荐的最小落地版本

第一版不需要一开始实现多 Agent 协作，可以采用：

```text
一个 Orchestrator
  + 一个 Optimization Agent
  + 四个 Skill：
      pipeline
      badcase-diagnosis
      data-strategy
      evaluator
  + 一个 run_state.json
  + 一个固定评测集
  + 一个人工确认点
```

最小闭环先支持：

```text
固定模型预测
  -> bad case 分桶
  -> 课程学习 / bad case 配比候选
  -> 小规模 SFT
  -> 推理和评测
  -> 自动判断继续或停止
```

在最小闭环稳定后，再加入：

```text
超参数搜索
  -> Prompt 版本搜索
  -> CoT / GRPO / OPD 训练选择
  -> 多 Agent 协同
  -> 更复杂的资源调度
```

### 9. 能力边界与风险

Agentic Model Optimization 适合解决：

- bad case 数量大、类型重复但人工分析成本高。
- 数据策略和超参数存在明确可搜索空间。
- 训练、推理和评测已经有稳定脚本。
- 主指标和护栏指标可以自动计算。
- 需要持续覆盖长尾任务和业务线。

它不适合直接解决：

- 完成态标准本身尚未达成共识。
- 评测结果不可复现或指标定义频繁变化。
- 训练链路无法稳定执行。
- 没有固定回归集。
- 只有少量样本，无法区分随机波动和真实收益。
- 模型错误需要大量领域常识，而 Agent 没有可靠工具和知识。

最大的风险不是 Agent 不会调参，而是 Agent 在错误目标上高效迭代。因此应优先保证：

```text
标准可信
  + 数据可追踪
  + 评测可复现
  + 实验可回滚
  + 资源有预算
  + 高风险动作有人审核
```

## 面试应对

### 常考点及考法

| 常考点 | 常见问法 | 回答重点 |
| --- | --- | --- |
| 概念定义 | 什么是 Agentic Model Optimization？ | Agent 接管模型优化闭环，但运行在受约束的实验空间内 |
| 架构设计 | Agent 如何完成自迭代？ | Orchestrator、Agent、Skill、Pipeline、State、Evaluator、Gate |
| 与数据飞轮区别 | 和主动学习有什么不同？ | 数据飞轮偏样本闭环，Agentic Loop 还负责规划、策略搜索和执行 |
| bad case | 如何自动找 bad case？ | GT 不一致、模型分歧、低置信度、高优业务低分和结构失败 |
| 防止乱调参 | 如何避免 Agent 随机试验？ | 假设、对照、预算、单变量、指标门禁和实验记录 |
| 可靠性 | 如何支持断点续跑？ | run state、产物复用、配置 hash、锁和失败分类 |
| 防投机 | Agent 可能刷指标怎么办？ | 固定评测集、多粒度指标、护栏、人工抽检和版本隔离 |
| 关键帧落地 | 如何用到关键帧检测？ | 围绕 early/late、边界、UI 元素和 task_type 做闭环 |

### 解法/回答思路

回答时按以下顺序：

```text
1. 先定义问题：人工模型优化的循环成本高，尤其是 bad case、数据、训练和评测之间反复往返。
2. 再讲核心思想：让 Agent 在固定目标、数据、工具和评测门禁下自主执行下一轮实验。
3. 再讲架构：Orchestrator 负责状态和调度，Agent 负责规划，Skill 负责能力，Pipeline 负责执行，Evaluator/Gate 负责判断。
4. 再讲关键帧落地：模型筛 early/late 和分歧样本，Agent 选择重标、增广、Prompt 或超参数策略，然后训练、推理、分桶评测。
5. 最后讲治理：数据血缘、固定回归集、版本、预算、锁、回滚和人工审核保证结果可信。
```

### 易错点

- 把 Agentic Model Optimization 说成“Agent 自动写训练代码”。
- 只讲 Agent 调参，不讲 bad case、数据策略和评测闭环。
- 把数据飞轮和 Agentic Loop 当成完全相同的概念。
- 没有固定 baseline 和回归集，就让 Agent 自己判断收益。
- 只看一个总 F1，不看业务线、task_type、early/late 和格式稳定性。
- 把伪标签或模型解释直接当作真值。
- 允许 Agent 任意修改评测脚本或准出阈值。
- 忽略 checkpoint、数据、Prompt 和评测脚本的版本关系。
- 没有预算和停止条件，导致 Agent 无限迭代。
- 认为 Agent 能替代业务专家定义完成态标准。

### 回答模板：什么是 Agentic Model Optimization？

Agentic Model Optimization 是让 AI Agent 在明确目标、受约束的实验空间和可靠评测门禁下，接管模型优化闭环的方法。它首先分析模型预测和 bad case，判断问题来自数据、任务规则、Prompt、超参数还是训练范式；然后通过 Skill 选择数据策略或训练策略，自动完成训练、推理和评测；最后根据主指标、护栏指标和预算决定保存当前模型、回滚还是进入下一轮。它不是让 Agent 无限制地修改训练代码，而是把模型优化工程化为可追踪、可恢复、可复现的 Loop-Engineering。

### 回答模板：它和数据飞轮、主动学习有什么区别？

数据飞轮和主动学习主要解决“哪些样本值得被发现、复查和回流”。Agentic Model Optimization 在此基础上进一步接管策略规划和执行：不仅筛选 bad case，还会决定是重标数据、构造 hard negative、调整课程学习配比、修改 Prompt、搜索超参数，还是切换训练范式，然后自动完成训练、推理、评测和下一轮决策。因此数据飞轮是数据闭环，Agentic Model Optimization 是覆盖数据、策略、训练和评测的更大闭环。

### 回答模板：如何把它用于关键帧检测？

我会先固定关键帧任务的完成态规则、数据版本、Prompt 和回归集，然后让 Agent 在同一批视频上收集模型与 GT 不一致、多模型分歧、early/late 和高优指标低分样本。Agent 先区分数据链路、GT 标准和模型能力问题，再选择人工重标、hard negative、课程学习、Prompt 或超参数策略。之后通过统一 Pipeline 完成 SFT、CoT-SFT、GRPO 或 OPD 训练、分片推理和分桶评测。只有在困难集提升、全量回归和格式稳定性都满足门禁时才保留新 checkpoint，否则回滚并进入下一轮诊断。

### 回答模板：如何保证 Agent 优化结果可信？

我会从五方面保证可信：第一，固定 baseline、回归集和评测脚本版本；第二，记录数据、Prompt、训练配置、checkpoint 和评测结果的完整血缘；第三，每轮实验都记录假设、变更、对照、预算和停止条件；第四，同时检查主指标、业务分桶指标和护栏指标，防止模型通过刷 Recall、数据泄漏或修改评测口径获得虚假收益；第五，对标准修改、伪标签大规模回流、模型结构变化和线上发布设置人工审核，并支持锁、断点续跑和回滚。

