# Agent 基础概念

本目录专门搭建 Agent 的通用知识框架。每张卡片都回答定义、组成、运行机制、设计边界、常见失败和面试应对；业务场景的操作步骤、实现原则映射和生产排障统一放在 [实战落地](<../实战落地/README.md#agent-实战落地>)。

## 内容索引

| 卡片 | 核心内容 |
| --- | --- |
| [Agent](<Agent.md#agent>) | Agent 的定义、组成、Agent Loop、从 0 到 1 的开发流程和系统概念地图。 |
| [Workflow](<Workflow.md#workflow>) | Workflow 与 Agent 的边界、Prompt Chaining、Routing、Parallelization、Planning、ReAct 和质量回路。 |
| [Tool Call 与 Function Calling](<Tool_Call与Function_Calling.md#tool-call-与-function-calling>) | 工具调用机制、Schema、MCP 的基础关系、Tools/Resources/Prompts 和 Code Execution。 |
| [Context Engineering](<Context_Engineering.md#contextengineering>) | 上下文组织、信息筛选、渐进加载，以及短期、工作、长期、情节和语义记忆。 |
| [Skill](<Skill.md#skill>) | Skill 的定位、触发路由、目录结构、渐进式加载、评测、发布和维护。 |
| [SubAgent 与 Multi-Agent](<SubAgent与Multi_Agent.md#subagent与multiagent>) | Orchestrator-Workers、Specialist Agents、Debate/Voting、分工、协作和冲突控制。 |
| [Guardrails 与 Human-in-the-Loop](<Guardrails与Human_in_the_Loop.md#guardrails与humanintheloop>) | 输入输出校验、权限、预算、人工确认、敏感信息保护和高风险动作控制。 |
| [Computer Use](<Computer_Use.md#computeruse>) | GUI 感知与操作、grounding、状态验证、浏览器自动化和人工确认。 |
| [Agent Eval、Trajectory 与 Harness](<Agent_Eval.md#agent-evaltrajectory-与-harness>) | Agent 评测维度、执行轨迹、可观测性、评测脚手架、回归和 bad case 诊断。 |
| [Jev 与 System One 决策模型](<Jev与System_One决策模型.md#jev-与-system-one-决策模型>) | `state + typed questions` 的结构化决策模型、三种原语、Agent 路由与风险门控、校准和能力边界。 |
| [Agentic RAG](<Agentic_RAG.md#agentic-rag>) | 检索规划、多轮检索、查询改写、证据验证、结果融合和噪声控制。 |
| [Hermes](<Hermes.md#hermes>) | 任务调度、消息分发、模型服务编排、Agent 编排、重试、幂等和扩缩容。 |
| [生产级 Agent 案例](<生产级Agent案例.md#生产级agent案例>) | Code Agent、Search/Deep Research Agent、数据分析 Agent、GUI Agent 和生产架构。 |
