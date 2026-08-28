# Agent 学习资料索引

这个文件记录 Agent、AI Coding、Skills、MCP、OpenClaw/Clawdbot 等相关资料来源和阅读状态。

## 已整理资料

### Vibe Coding 面试方式-实践分享

- 来源：飞书 Wiki
- 链接：https://bytedance.larkoffice.com/wiki/GoMhwEJ3RixEKCkfvYUcosbtnwf
- 状态：已读取并整理
- 整理笔记：[../AI_Coding/Vibe_Coding面试与实践.md](<../AI_Coding/Vibe_Coding面试与实践.md>)

核心价值：

- 梳理 Vibe Coding 面试的背景、考察目标和题型设计。
- 强调 AI 协作过程可观察、可验证。
- 将候选人能力拆成需求澄清、Prompt 设计、工程实现、调试验证、交付责任。

### AI Coding 文档学习合集

- 来源：飞书 Wiki
- 链接：https://bytedance.larkoffice.com/wiki/ZPPBwxf5yitGBhk6o4tcXvjpnud
- 状态：已读取，作为资料地图沉淀

主题分类：

- Skills：Skill 概念、实操、Claude Code 插件实践、Trae Skill 实践。
- MCP：MCP 工具提效。
- Spec Coding：Spec Coding 最佳实践、Spec Coding Agent。
- Trae：Trae 使用经验。
- Claude Code：Claude Code 使用、代理、工作流实践。
- OpenClaw / Clawdbot：内部开发机、Hermes 灰度、Fornax Trace、手机接力编码。
- AGENTS.md：AI 辅助编程规范与仓库级上下文实践。
- DeepResearch 类 Agent：调研型 Agent 能力与工作流。

后续精读建议：

1. Skills 与 SubAgent 相关资料。
2. Spec Coding Agent 实践。
3. OpenClaw 团队 Agent 的实践之路。
4. AGENTS.md 最佳实践。
5. Agent 不只是 Tool Call Loop。

### Designing, Refining and Maintaining Agent Skills at Perplexity

- 来源：Perplexity Research
- 链接：https://research.perplexity.ai/articles/designing-refining-and-maintaining-agent-skills-at-perplexity
- 状态：已读取并整理
- 整理笔记：[../Skills/Agent_Skills设计与维护.md](<../Skills/Agent_Skills设计与维护.md>)

核心价值：

- 明确 Skill 是目录、格式、可调用上下文和渐进式知识组织方式。
- 强调 description 是路由触发器，不是功能说明。
- 强调 eval 先行、负例重要、gotchas 持续沉淀。
- 提醒 Skill 是上下文成本，不能把普通 README 当 Skill。

### 全栈 AI Coding 最佳实践工作流探索

- 来源：ByteTech
- 链接：https://bytetech.info/articles/7628067999768903723#Wf8ldg90eocFEwxrdmycoVn1n2b
- 状态：已读取并整理
- 整理笔记：[../AI_Coding/全栈_AI_Coding最佳实践工作流.md](<../AI_Coding/全栈_AI_Coding最佳实践工作流.md>)

核心价值：

- 从真实项目总结 Trae SOLO + 多仓库 Spec + 测试验证 Skill 的工作流。
- 明确 Spec-First 比纯 Vibe Coding 更适合复杂需求。
- 强调用自动化测试替代“AI 猜测式 Review”。
- 说明从 0 到 1 项目和存量业务系统的适用性差异。
- 提出通过 Skill、AGENTS.md、测试用例沉淀长期 AI 协作能力。

## Agent 学习路线建议

### 第一阶段：基础概念

- [../基础概念/Harness.md](<../基础概念/Harness.md>)
- [../基础概念/Hermes.md](<../基础概念/Hermes.md>)

目标：理解 Agent 工程中“评测脚手架”和“任务调度/编排层”的边界。

### 第二阶段：AI Coding 实践

- [../AI_Coding/Vibe_Coding面试与实践.md](<../AI_Coding/Vibe_Coding面试与实践.md>)
- [../AI_Coding/全栈_AI_Coding最佳实践工作流.md](<../AI_Coding/全栈_AI_Coding最佳实践工作流.md>)

目标：理解 AI Coding 的真实考察点，建立需求澄清、方案设计、AI 协作编码、验证闭环的完整工作流。

### 第三阶段：Skills 工程化

- [../Skills/Agent_Skills设计与维护.md](<../Skills/Agent_Skills设计与维护.md>)

目标：理解如何把领域经验沉淀成 Agent 可稳定调用的上下文能力。

### 第四阶段：扩展主题

建议后续继续补充：

- Tool Call Loop 与生产级 Agent 架构。
- MCP 协议与工具生态。
- SubAgent 分工与多 Agent 协作。
- AGENTS.md 与仓库级上下文管理。
- Computer Use 智能体。
- Agent 评测与 trajectory 分析。
