# Jev 与 System One 决策模型

## 知识点解析

### 概述

Jev 是 TypeSafe AI 发布的首个 System One 模型。它不生成开放文本，而是接收 `state + typed questions`，返回可被程序直接消费的类型化答案、概率分布，以及部分问题类型的置信度。

它适合 Agent 中高频、低延迟、输出空间可预先定义的判断，例如模型路由、工具风险门控、分类、评分、验证和人工升级；开放式推理、规划与内容生成仍应交给 LLM。

### 输入与输出

一次请求包含：

- `state`：模型判断所依据的文本、结构化数据或消息状态。
- `questions`：一个或多个预先定义类型和输出空间的原子问题。
- `answers`：每个问题对应的类型化结果和概率信息。

同一请求中的多个问题针对同一份 `state` **彼此独立、并行评估**。复杂判断应拆成多个原子问题，再由确定性代码组合权重、阈值和业务规则，而不是要求某个问题同时完成长链推理。

### 三种原语

| 原语 | 回答的问题 | 返回结果 | 典型用途 |
| --- | --- | --- | --- |
| `Choice` | 应从预定义选项中选哪一个？ | 选项、各选项概率、confidence | 意图分类、模型路由、处理链路选择 |
| `Score` | 按有序量表应评为哪一级？ | 分数、概率分布、confidence | 风险等级、质量等级、优先级 |
| `Noul` | 某个陈述成立的概率是多少？ | 0 到 1 的真值概率 | 是否敏感、是否紧急、是否需要升级 |

`Noul` 的名称和普通布尔值不同：它保留不确定性，而不是只返回 `true/false`。业务代码仍需根据误判成本选择阈值。

### System One 与 RLCD

TypeSafe 将这类模型称为 System One 模型，借用了“快速、直觉式判断”的概念。这里是产品和模型类别名称，不代表模型具有人类认知机制，也不代表判断天然正确。

官方将训练方法称为 RLCD（Reinforcement Learning for Calibrated Decisions），目标是让模型输出结构化决策及与正确率相匹配的概率。公开材料没有给出足以独立复现全部训练过程的细节，因此应把 RLCD 理解为官方方法定位，而不是擅自推断具体损失函数或训练配方。

### 与 LLM、规则引擎的分工

| 能力 | 更适合的组件 | 原因 |
| --- | --- | --- |
| 硬权限、金额上限、状态机约束 | 规则与普通代码 | 必须确定、可审计，不能交给概率模型 |
| 分类、评分、路由、风险判断 | Jev 一类决策模型 | 输出空间固定，需要低延迟和概率 |
| 开放式分析、规划、解释和生成 | LLM | 需要组合知识、长链推理或自由文本 |
| 高风险且不确定的动作 | 规则 + Jev/LLM + 人工 | 用确定性约束和升级机制共同兜底 |

Jev 不是 LLM 的直接替代品。更合理的结构是让 LLM 负责“想和写”，让 Jev 负责高频的“分、选、评、判”，再由代码负责约束、执行和审计。

### 在 Agent 中的位置

```text
用户请求 / Agent state
  -> 硬规则预检
  -> Jev：意图、难度、风险、质量、是否升级
  -> 确定性路由与权限代码
       -> 小模型 / 强模型 / RAG / 工具 / 拒绝
       -> 低置信或高风险：LLM 复核或 Human-in-the-Loop
  -> 执行结果验证与审计
```

常见用法：

1. **模型路由**：区分普通问答、RAG、代码、多模态和高风险任务。
2. **工具风险门控**：在 shell、删除、支付或权限类工具执行前给出风险概率。
3. **分类与抽取**：把邮件、工单、内容或事件归入固定类别。
4. **评分与验证**：评估答案质量、证据支持度、任务完成度或安全风险。
5. **HITL 升级**：低 confidence、高风险或分布外输入进入更强模型或人工。

LangChain 的公开集成示例覆盖了模型路由和工具风险门控，这两类场景都位于 Agent 的决策层，而不是文本生成层。

### 伪代码示例

下面展示接口形状和决策组合方式，不绑定具体 SDK 版本：

```python
request = {
    "model": "jev-latest",
    "state": {
        "message": user_message,
        "has_image": has_image,
        "requested_tool": requested_tool,
    },
    "questions": {
        "route": {
            "type": "choice",
            "criteria": {
                "fast_llm": "直接查询、抽取或局部修改",
                "reasoning_llm": "复杂推理和架构决策",
                "vlm": "需要理解图像或视频",
                "human": "高风险或信息不足",
            },
            "instructions": "选择能可靠完成请求的最低成本处理方。",
        },
        "risk": {
            "type": "score",
            "criteria": ["低风险且可逆", "影响有限但需审计", "高风险或不可逆"],
            "instructions": "根据动作可逆性、权限和影响范围评估风险。",
        },
        "needs_private_data": {
            "type": "noul",
            "instructions": "完成请求是否需要访问私有数据？",
        },
    },
}

result = evaluate_with_jev(request)
answers = result["answers"]

# Deterministic and fail-closed: reject unless policy explicitly allows every scope.
authorize_or_reject(
    actor=user,
    tool=requested_tool,
    resource=requested_resource,
    requested_scopes=requested_scopes,
)

if (
    answers["risk"].score >= 1.50
    or answers["risk"].confidence < 0.60
    or answers["route"].confidence < 0.80
):
    route = "human"
else:
    route = answers["route"].choice

if answers["needs_private_data"].noul >= 0.20:
    require_data_handling_review()

dispatch(route)
```

这里的关键不是示例阈值本身，而是根据误放行、误拦截和人工成本，用验证集确定阈值。`authorize_or_reject`必须依据 actor、工具、资源和请求 scope 做确定性鉴权，任何未显式允许的访问都应拒绝；Jev 只能触发额外复核，不能授予权限或绕过鉴权。

### 能力边界与风险

#### 类型安全不等于判断正确

Jev 的输出被限制在预定义 schema 内，因此不会返回 schema 外的选项或错误类型。TypeSafe 将这一点表述为不会产生类型层面的 hallucination。

但模型仍可能：

- 在合法选项中选错。
- 给错误答案较高概率。
- 因 state 缺失、歧义或分布漂移而误判。
- 在业务代价不对称时使用了不合适的阈值。

`Choice`和`Score`的 confidence 由返回概率的集中程度计算；分布集中不等于答案正确。因此，“不能 hallucinate”只能解释为**不会生成结构外结果**，不能解释为“不会做错判断”。

#### 不适合的任务

- 需要自由文本、代码或长篇解释。
- 需要多步探索、工具反馈和长链推理。
- 输出空间无法提前定义或类别频繁变化。
- 零容错的权限、安全和合规规则。
- 缺少可代表线上分布的标注集，无法验证概率与阈值。

#### 工程风险

- 问题把多个判断混在一起，导致概率含义不清。
- 多个独立问题被误当成有因果顺序的推理链。
- 只看 accuracy，不检查概率校准和高风险错误。
- 直接沿用通用阈值，没有按业务成本与数据分布调参。
- 模型、问题定义或线上输入变化后没有重新校准。

### 如何评测

#### 离线评测

| 维度 | 推荐指标 |
| --- | --- |
| `Choice` 分类 | Accuracy、Macro-F1、Top-k Accuracy、混淆矩阵 |
| `Score` 有序评分 | MAE、等级准确率、Spearman、严重误差率 |
| `Noul` 概率判断 | AUROC、PR-AUC、Brier Score、Log Loss |
| 概率校准 | ECE、Reliability Diagram、分桶准确率 |
| 升级策略 | Risk-Coverage Curve、不同阈值下的自动化率与错误率 |
| 系统收益 | 端到端任务成功率、p95/p99、单请求成本、人工升级率 |

评测集应覆盖常规、边界、对抗和分布外输入，并单独统计高风险误放行与低风险误拦截。概率模型上线前必须先做 shadow test，不能只用厂商 benchmark 代替业务验证。

#### 在线监控

- 路由分布、类别占比和 confidence 分布是否漂移。
- 人工复核与最终结果是否持续支持模型判断。
- 高风险 false negative、误拦截和 fallback 成功率。
- Jev、LLM、规则及人工各自承担的流量、延迟和成本。
- 模型版本、问题定义、阈值和每次路由原因是否可追溯。

### 官方性能口径

TypeSafe 在 2026 年 9 月的发布材料中报告：服务端到端响应约为 `70 ms-500 ms`，System One 形状的任务可能比对比 LLM 快 `40x-200x`；其 workflow eval 给出 `193.6x` 加速和 `444.6x` 降本。官方同时说明这些属于其自建工作流评测的较高端结果，且工作流由模型能力团队成员制作，可能存在偏差。

这些数字不是独立第三方结论。实际选型应使用相同输入、相同决策任务、相同质量门槛和真实网络环境，对 Jev、规则、小分类器和结构化输出 LLM 做对照实验。

### 常见误区

1. **“Jev 不会 hallucinate，所以不会出错。”**
   它保证输出落在 schema 内，不保证语义判断正确。
2. **“有概率就已经校准。”**
   校准必须在目标数据分布上用 Brier、ECE 等指标验证。
3. **“并行问题可以替代推理链。”**
   问题彼此独立；依赖关系和组合逻辑应由代码或后续模型处理。
4. **“速度快，所以所有 Agent 节点都应换成 Jev。”**
   只有固定输出空间的判断适合；规划、解释和生成仍需要 LLM。

## 面试应对

### Jev 是什么，适合放在 Agent 的哪里？

回答思路：先说明它是面向类型化概率决策的 System One 模型，再讲 Choice/Score/Noul、与 LLM 和规则的分工，最后补上类型安全边界与评测方法。

回答模板：

Jev 是 TypeSafe 的 System One 决策模型。它不生成文本，而是接收 state 和 Choice、Score、Noul 等类型化问题，并行返回结构化答案和概率。它适合放在 Agent 的决策层，处理模型路由、工具风险门控、分类、评分和人工升级；LLM 继续负责开放式推理与生成，规则代码负责权限和硬约束。它的类型安全只保证输出不会越过 schema，不保证判断正确，因此仍要评估准确率、概率校准、风险覆盖曲线和业务误判成本，并为低置信或高风险请求保留 LLM 或人工兜底。

## 参考资料

- [TypeSafe: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe Docs: Introduction](https://docs.typesafe.ai/introduction)
- [LangChain: Building a Harness with Jev](https://www.langchain.com/blog/building-a-harness-with-jev)
- [ByteTech：Jev 相关文章（内部访问）](https://bytetech.info/articles/7688557832006254632?from=message_bot&sender_type=101&message_id=7689299772059729958&column=like#AfCrd5CgNoMh2JxlDD9ccyKgn6d)
