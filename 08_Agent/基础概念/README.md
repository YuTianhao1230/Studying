# Agent 基础概念

本目录专门搭建 Agent 的通用知识框架。每张卡片都回答定义、组成、运行机制、设计边界、常见失败和面试应对；业务场景的操作步骤、实现原则映射和生产排障统一放在 [实战落地](<../实战落地/README.md>)。

## 知识卡片

| 卡片 | 核心内容 |
| --- | --- |
| [Agent](<Agent.md>) | Agent 的定义、组成、Agent Loop、从 0 到 1 的开发流程和系统概念地图。 |
| [Workflow](<Workflow.md>) | Workflow 与 Agent 的边界、Prompt Chaining、Routing、Parallelization、Planning、ReAct 和质量回路。 |
| [Tool Call 与 Function Calling](<Tool_Call与Function_Calling.md>) | 工具调用机制、Schema、MCP 的基础关系、Tools/Resources/Prompts 和 Code Execution。 |
| [Context Engineering](<Context_Engineering.md>) | 上下文组织、信息筛选、渐进加载，以及短期、工作、长期、情节和语义记忆。 |
| [Skill](<Skill.md>) | Skill 的定位、触发路由、目录结构、渐进式加载、评测、发布和维护。 |
| [SubAgent 与 Multi-Agent](<SubAgent与Multi_Agent.md>) | Orchestrator-Workers、Specialist Agents、Debate/Voting、分工、协作和冲突控制。 |
| [Guardrails 与 Human-in-the-Loop](<Guardrails与Human_in_the_Loop.md>) | 输入输出校验、权限、预算、人工确认、敏感信息保护和高风险动作控制。 |
| [Computer Use](<Computer_Use.md>) | GUI 感知与操作、grounding、状态验证、浏览器自动化和人工确认。 |
| [Agent Eval、Trajectory 与 Harness](<Agent_Eval.md>) | Agent 评测维度、执行轨迹、可观测性、评测脚手架、回归和 bad case 诊断。 |
| [Agentic RAG](<Agentic_RAG.md>) | 检索规划、多轮检索、查询改写、证据验证、结果融合和噪声控制。 |
| [Hermes](<Hermes.md>) | 任务调度、消息分发、模型服务编排、Agent 编排、重试、幂等和扩缩容。 |
| [生产级 Agent 案例](<生产级Agent案例.md>) | Code Agent、Search/Deep Research Agent、数据分析 Agent、GUI Agent 和生产架构。 |

## 基础与实战的边界

| 基础概念回答 | 实战落地回答 |
| --- | --- |
| Agent 和 Workflow 分别是什么 | 如何用 LangGraph 编排一条可恢复链路 |
| Tool Call、MCP、Skill 的职责如何区分 | 如何接入 Figma、Lynx DevTool 和评测系统并做授权 |
| Context、Memory、RAG 的基本原理 | 如何按 `任务类型` 加载 few-shot、过滤版本并回流 bad case |
| Agent Eval、Trajectory 和 Harness 是什么 | 如何冻结输入、跑评测、归因并生成 GT vNext |
| Guardrails 和 HITL 为什么需要 | 如何做 dry-run、审批、审计、灰度和回滚 |

## 推荐学习顺序

1. 阅读 [Agent](<Agent.md>)，建立整体系统模型。
2. 阅读 [Workflow](<Workflow.md>)、[Tool Call 与 Function Calling](<Tool_Call与Function_Calling.md>) 和 [Context Engineering](<Context_Engineering.md>)，理解规划、执行和上下文。
3. 阅读 [Skill](<Skill.md>)、[SubAgent 与 Multi-Agent](<SubAgent与Multi_Agent.md>)、[Guardrails 与 Human-in-the-Loop](<Guardrails与Human_in_the_Loop.md>)，理解能力复用、协作和安全边界。
4. 阅读 [Agent Eval、Trajectory 与 Harness](<Agent_Eval.md>)，建立评测和诊断闭环。
5. 最后阅读 [Computer Use](<Computer_Use.md>)、[Agentic RAG](<Agentic_RAG.md>)、[Hermes](<Hermes.md>) 和 [生产级 Agent 案例](<生产级Agent案例.md>)。
6. 转入 [实战落地](<../实战落地/README.md>)，把概念映射到 D2C 真实业务。

## 主题归属

- Planning 与 ReAct：见 [Workflow](<Workflow.md>)。
- MCP 与 Code Execution：见 [Tool Call 与 Function Calling](<Tool_Call与Function_Calling.md>)。
- Memory：见 [Context Engineering](<Context_Engineering.md>)。
- Agent Skills 设计与维护：见 [Skill](<Skill.md>)。
- Trajectory、Observability 与 Harness：见 [Agent Eval、Trajectory 与 Harness](<Agent_Eval.md>)。
- Agent 开发流程：见 [Agent](<Agent.md>)。

基础卡片按主题聚合，便于从一个入口完成概念、边界、失败模式和面试表达的复习。
