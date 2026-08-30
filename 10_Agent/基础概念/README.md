# 基础概念

本目录整理 Agent 体系的核心名词和架构概念，适合系统学习 Agent 从任务规划、工具调用到评测观测的完整链路。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [Agent.md](<Agent.md>) | Agent 的基本定义、能力边界和系统组成。 |
| [Workflow.md](<Workflow.md>) | 多节点工作流，以及 Agent 在工作流中的位置。 |
| [Skill.md](<Skill.md>) | 可复用能力单元和 Agent 能力封装方式。 |
| [MCP.md](<MCP.md>) | Model Context Protocol 与工具/资源暴露标准。 |
| [Tool_Call与Function_Calling.md](<Tool_Call与Function_Calling.md>) | 模型如何结构化请求外部工具或函数。 |
| [Planning与ReAct.md](<Planning与ReAct.md>) | 计划、思考-行动-观察循环和 ReAct 模式。 |
| [Memory.md](<Memory.md>) | 短期/长期记忆和上下文持久化。 |
| [Context_Engineering.md](<Context_Engineering.md>) | 上下文组织、压缩、检索和注入策略。 |
| [SubAgent与Multi_Agent.md](<SubAgent与Multi_Agent.md>) | 子 Agent 和多 Agent 协作。 |
| [Guardrails与Human_in_the_Loop.md](<Guardrails与Human_in_the_Loop.md>) | 安全护栏、人类确认和高风险操作控制。 |
| [Trajectory与Observability.md](<Trajectory与Observability.md>) | Agent 轨迹记录、可观测性和错误归因。 |
| [Agent_Eval.md](<Agent_Eval.md>) | Agent 评测任务、指标和回归集。 |
| [Agentic_RAG.md](<Agentic_RAG.md>) | Agent 与 RAG 结合的动态检索和工具化问答。 |
| [Code_Execution_with_Tools.md](<Code_Execution_with_Tools.md>) | Agent 调用代码执行环境完成任务。 |
| [Computer_Use.md](<Computer_Use.md>) | GUI/浏览器/桌面操作型 Agent 能力。 |
| [Harness.md](<Harness.md>) | Agent 执行环境和评测 harness 概念。 |
| [Hermes.md](<Hermes.md>) | Hermes 相关 Agent 知识点。 |
| [Agent开发完整流程.md](<Agent开发完整流程.md>) | 从需求到工具、记忆、评测和部署的完整 Agent 开发流程。 |
| [生产级Agent案例.md](<生产级Agent案例.md>) | 生产级 Agent 的架构、案例和工程边界。 |

## 学习路线

1. 先看 [Agent.md](<Agent.md>)、[Workflow.md](<Workflow.md>)、[Tool_Call与Function_Calling.md](<Tool_Call与Function_Calling.md>) 和 [MCP.md](<MCP.md>)。
2. 再看 [Planning与ReAct.md](<Planning与ReAct.md>)、[Memory.md](<Memory.md>)、[Context_Engineering.md](<Context_Engineering.md>)。
3. 接着看 Guardrails、Trajectory、Agent Eval，理解生产化和评测。
4. 最后看 [Agent开发完整流程.md](<Agent开发完整流程.md>) 和 [生产级Agent案例.md](<生产级Agent案例.md>)。
