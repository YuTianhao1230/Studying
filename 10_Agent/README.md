# Agent 专题索引

这个目录用于沉淀 Agent、AI Coding、Skills、工具调用、任务编排、评测 Harness 等相关知识。

## 推荐阅读路径

1. [基础概念/Agent.md](<基础概念/Agent.md>)
2. [基础概念/Agent开发完整流程.md](<基础概念/Agent开发完整流程.md>)
3. [基础概念/Workflow.md](<基础概念/Workflow.md>)
4. [基础概念/Planning与ReAct.md](<基础概念/Planning与ReAct.md>)
5. [基础概念/Tool_Call与Function_Calling.md](<基础概念/Tool_Call与Function_Calling.md>)
6. [基础概念/MCP.md](<基础概念/MCP.md>)
7. [基础概念/Skill.md](<基础概念/Skill.md>)
8. [基础概念/SubAgent与Multi_Agent.md](<基础概念/SubAgent与Multi_Agent.md>)
9. [基础概念/Memory.md](<基础概念/Memory.md>)
10. [基础概念/Context_Engineering.md](<基础概念/Context_Engineering.md>)
11. [基础概念/Computer_Use.md](<基础概念/Computer_Use.md>)
12. [基础概念/Guardrails与Human_in_the_Loop.md](<基础概念/Guardrails与Human_in_the_Loop.md>)
13. [基础概念/Trajectory与Observability.md](<基础概念/Trajectory与Observability.md>)
14. [基础概念/Agent_Eval.md](<基础概念/Agent_Eval.md>)
15. [基础概念/Agentic_RAG.md](<基础概念/Agentic_RAG.md>)
16. [基础概念/Code_Execution_with_Tools.md](<基础概念/Code_Execution_with_Tools.md>)
17. [基础概念/Harness.md](<基础概念/Harness.md>)
18. [基础概念/Hermes.md](<基础概念/Hermes.md>)
19. [基础概念/生产级Agent案例.md](<基础概念/生产级Agent案例.md>)
20. [Skills/Agent_Skills设计与维护.md](<Skills/Agent_Skills设计与维护.md>)
21. [AI_Coding/Vibe_Coding面试与实践.md](<AI_Coding/Vibe_Coding面试与实践.md>)
22. [AI_Coding/全栈_AI_Coding最佳实践工作流.md](<AI_Coding/全栈_AI_Coding最佳实践工作流.md>)
23. [资料索引/Agent学习资料索引.md](<资料索引/Agent学习资料索引.md>)

## 分类说明

### 基础概念

放 Agent 工程链路里容易反复出现的底层概念。

- Agent：以 LLM 为决策核心、能观察环境并调用工具完成目标的系统。
- Agent 开发完整流程：从任务边界、Workflow、Agent Loop、Tool Call、MCP、Context、Memory、Guardrails、Eval 到上线迭代的完整链路。
- Workflow：预定义的多步骤 LLM/工具编排流程。
- Planning 与 ReAct：Agent 拆解任务、行动、观察、反思的执行模式。
- Tool Call / Function Calling：模型通过结构化参数请求外部工具执行动作。
- MCP：连接 Agent 与外部工具/资源/Prompt 的标准协议。
- Skill：可动态加载的领域能力包。
- SubAgent / Multi-Agent：多智能体分工、委派和协作模式。
- Memory：跨步骤和跨会话保存信息的机制。
- Context Engineering：设计 Agent 当前应该看见什么上下文的工程方法。
- Computer Use：让 Agent 操作 GUI、浏览器和桌面环境。
- Guardrails / Human-in-the-Loop：安全边界、权限控制和人工确认机制。
- Trajectory / Observability：记录和分析 Agent 执行轨迹。
- Agent Eval：评估 Agent 任务成功率、过程质量、工具调用和安全性。
- Agentic RAG：可规划、多轮检索、可校验证据的 RAG 工作流。
- Code Execution with Tools：让 Agent 在沙箱中用代码组合工具，减少上下文搬运。
- Harness：评测脚手架，负责数据读取、模型调用、指标统计、错误归因和结果落盘。
- Hermes：任务调度、消息分发、服务编排或 Agent 执行层的常见内部命名。
- 生产级 Agent 案例：Code Agent、Search Agent、数据分析 Agent、GUI Agent 的工程约束和风险控制。

### Skills

放 Agent Skill 的设计、维护、评测和信息架构方法。

核心问题：

- 什么时候需要 Skill？
- Skill 的 description 应该怎么写？
- 如何做 progressive disclosure？
- 如何维护 gotchas 和 eval？
- 如何避免 Skill 污染上下文或误触发？

### AI_Coding

放 Vibe Coding、AI Coding、Code Agent、AI 协同开发与面试考察方法。

核心问题：

- AI Coding 面试到底考察什么？
- 如何用 Spec-First 方式约束 AI Coding？
- 多仓库 Workspace 如何帮助前后端联动开发？
- 为什么自动化测试是 AI Review 的核心抓手？
- 如何观察候选人的 AI 协作过程？
- 如何设计可验证的端到端题目？
- 如何评价 Prompt、需求澄清、工程验证和调试能力？

### 资料索引

放外部文章、内部文档、飞书 Wiki、ByteTech 资料的来源、阅读状态和后续精读方向。

## 和其他目录的边界

- 推理框架、Serving、KV Cache、Batching、量化等仍放在 [../05_推理部署与系统/推理工程](<../05_推理部署与系统/推理工程/>)。
- RAG、CoT、Prompt 调优等模型应用能力仍放在 [../02_大模型/应用与问题](<../02_大模型/应用与问题/>)。
- Agent 目录重点放“如何让模型作为执行主体完成任务”的工程方法，包括工具调用、上下文组织、任务分解、调度、验证和评测。
