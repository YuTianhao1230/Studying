# Agent

本目录按三条主线组织：`基础概念`负责搭建 Agent 知识框架和面试表达，`实战落地`负责把 Agent 做成真实业务系统，`常用Skill`负责沉淀日常使用过的 Agent Skill。学习与实践主线通过“概念 -> 设计 -> 实现 -> 评测 -> 生产”连接。

## 三部分入口

| 部分 | 内容边界 |
| --- | --- |
| [基础概念](<基础概念/README.md>) | Agent、Workflow、Tool Call、Context、Skill、SubAgent、安全、Computer Use、评测、RAG、调度和生产案例等基础知识与面试应对。 |
| [实战落地](<实战落地/README.md>) | 从问题定义到生产 Agent 的完整工程链路，重点以 D2C 视觉一致性评估场景为案例，覆盖 Skill、Workflow、MCP、RAG、数据、评测、可靠性、安全和排障。 |
| [常用Skill](<常用Skill/README.md>) | 已安装和实际使用过的 Skill，记录 GitHub 来源、解决的问题、核心机制、安装位置、使用方式和适用场景。 |

## 推荐入口

1. 先读 [基础概念](<基础概念/README.md>) 中的 Agent、Workflow、Tool Call 与 Context，建立系统模型。
2. 再读 [实战总览](<实战落地/实战总览：从问题到生产Agent.md>)，理解如何把概念转成任务契约和工程链路。
3. 以 [D2C 视觉一致性评估 Agent 案例](<实战落地/D2C视觉一致性评估Agent案例.md>) 为主线，按实战目录的推荐顺序深入。
4. 需要查已安装能力时，进入 [常用Skill](<常用Skill/README.md>) 查看来源和使用说明。

## 学习顺序

### 基础概念阶段

1. 用 [Agent](<基础概念/Agent.md>)、[Workflow](<基础概念/Workflow.md>)、[Tool Call 与 Function Calling](<基础概念/Tool_Call与Function_Calling.md>) 建立 Agent 系统模型。
2. 用 [Context Engineering](<基础概念/Context_Engineering.md>)、[Skill](<基础概念/Skill.md>)、[SubAgent 与 Multi-Agent](<基础概念/SubAgent与Multi_Agent.md>) 理解上下文、能力复用和协作。
3. 用 [Guardrails 与 Human-in-the-Loop](<基础概念/Guardrails与Human_in_the_Loop.md>)、[Agent Eval、Trajectory 与 Harness](<基础概念/Agent_Eval.md>) 建立安全和评测框架。

### 实战落地阶段

1. 阅读 [实战总览](<实战落地/实战总览：从问题到生产Agent.md>)，把概念变成任务契约和工程分层。
2. 以 [D2C 视觉一致性评估 Agent 案例](<实战落地/D2C视觉一致性评估Agent案例.md>) 为主线，依次学习数据、Workflow、Skill、Tool/MCP、RAG 和评测。
3. 最后补齐可靠性、安全、排障和 [AI Coding](<实战落地/AI_Coding与Spec_First工程实践.md>)，形成生产交付能力。

### 常用 Skill 阶段

1. 先看 [常用Skill总览](<常用Skill/README.md>)，了解每个 Skill 解决的问题和使用边界。
2. 做创造性开发或行为变更前，参考 [Superpowers](<常用Skill/Superpowers.md>) 的 brainstorming -> writing-plans -> executing-plans 流程。
3. 做跨多轮、需要恢复或持续记录的研究任务时，参考 [Planning with Files](<常用Skill/Planning_with_Files.md>) 的文件化计划和进度管理。
4. 准备交付文档、索引、标题或交接信息时，参考 [No Negative Echo](<常用Skill/No_Negative_Echo.md>) 做最终表面检查。

## 主题归属

| 主题 | 主入口 | 内容范围 |
| --- | --- | --- |
| Agent 开发流程 | [基础概念/Agent](<基础概念/Agent.md>) | 定义、组成、循环、开发步骤和系统模型。 |
| Planning 与 ReAct | [基础概念/Workflow](<基础概念/Workflow.md>) | 任务编排、执行模式、质量回路和终止条件。 |
| MCP 与工具执行 | [基础概念/Tool Call 与 Function Calling](<基础概念/Tool_Call与Function_Calling.md>) | 外部能力接入、调用协议、结果处理和错误模型。 |
| Memory 与上下文 | [基础概念/Context Engineering](<基础概念/Context_Engineering.md>) | 上下文组织、长期记忆、按需注入和污染控制。 |
| Agent Skill 设计 | [基础概念/Skill](<基础概念/Skill.md>) | Skill 原理、路由、目录结构、评测和维护。 |
| Agent 评测与轨迹 | [基础概念/Agent Eval](<基础概念/Agent_Eval.md>) | 执行记录、Harness、指标体系和失败诊断。 |
| D2C 视觉评估 | [实战落地/D2C 案例](<实战落地/D2C视觉一致性评估Agent案例.md>) | 采集、匹配、确定性检测、模型复核和结果回流。 |
| Skill 工程化 | [实战落地/Skill 工程化](<实战落地/Skill工程化：设计、路由、实现与迭代.md>) | 业务能力的路由、目录、工具、评测和版本迭代。 |
| 生产可靠性 | [实战落地/可靠性](<实战落地/可靠性性能成本与可观测性.md>) | 重试、幂等、并发、缓存、Trace、容量和降级。 |
| 常用 Skill | [常用Skill](<常用Skill/README.md>) | 外部 Skill 的来源、机制、安装、调用和适用边界。 |

## 主题边界

| 主题 | 主要回答的问题 | 关联边界 |
| --- | --- | --- |
| 基础概念 | 原理、模式、边界和面试表达 | 业务接口和操作步骤进入实战卡片。 |
| D2C 视觉一致性评估 | 一个真实 D2C Agent 如何工作 | 通用 Agent 定义见基础概念。 |
| Skill 实战 | 如何把能力写成可触发、可执行、可评测的包 | Prompt 基础见 Prompt 调优专题。 |
| Tool/MCP 实战 | 如何接入外部系统并完成授权治理 | MCP 基础定义见 Tool Call 卡片。 |
| 评测实战 | 如何冻结数据、跑轨迹、回流 GT | 模型评测理论进入评测实验专题。 |
| 常用 Skill | 如何选择和使用已安装的 Agent 能力 | 每个 Skill 的完整规则以本地 `SKILL.md` 为准。 |

## 目录边界

- `基础概念/`：承载通用原理、设计模式、边界和面试答案。
- `实战落地/`：承载业务场景、抽象职责映射、接口契约、授权、评测、排障和生产治理。
- `常用Skill/`：沉淀外部 Skill 的来源、用途、机制和使用方法。

## 维护规则

- 新内容先判断属于通用原理、业务实施还是可复用 Skill，再放入对应分支。
- 同一知识设置一张主卡片；相关页面提供边界说明和导航。
- 新增 Skill、工具或 Workflow 时，记录输入输出、失败语义和验证方式。
- 外部 Skill 资料至少记录 GitHub 仓库、版本或分支、安装路径、解决的问题、核心机制和使用方式。
- 写回外部系统的流程要说明身份、权限、确认、审计和回滚。
- 卡片的“概述”只总结正文知识，不写整理过程或会话元信息。
