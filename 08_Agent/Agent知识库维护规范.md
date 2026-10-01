# Agent 知识库维护规范

## 文档定位

本规范维护 `08_Agent` 的目录职责、主题归属、阅读顺序和本地 Skill 状态。通用知识只在一张权威卡片中维护，README 只提供目录简介和直属内容索引。

## 目录职责

| 目录 | 职责 | 不承载的内容 |
| --- | --- | --- |
| `基础概念/` | Agent、Workflow、Tool、Context、Skill、安全、评测和决策模型的通用原理、边界与面试回答。 | 业务接口、具体操作步骤和生产排障过程。 |
| `实战落地/` | 业务任务契约、数据链路、系统编排、工具接入、评测、可靠性、安全和排障。 | 通用概念的第二份定义。 |
| `常用Skill/` | 已实际使用的 Skill 的稳定机制、使用方式、协同关系和适用边界。 | 仅浏览过的项目、会话过程记录和长期不核验的本地状态。 |

跨目录内容按主要用途选择唯一正文：通用原理进入基础概念，业务实施进入实战落地，外部 Skill 的使用方法进入常用 Skill。其他卡片通过相对链接引用，不复制完整正文。

## 主题归属

| 主题 | 权威入口 |
| --- | --- |
| Agent 定义、组成、循环与开发流程 | [Agent](<基础概念/Agent.md>) |
| Planning、ReAct 与编排模式 | [Workflow](<基础概念/Workflow.md>) |
| Tool Call、Function Calling 与 MCP 基础 | [Tool Call 与 Function Calling](<基础概念/Tool_Call与Function_Calling.md>) |
| Context、Memory 与按需注入 | [Context Engineering](<基础概念/Context_Engineering.md>) |
| Skill 原理、路由与生命周期 | [Skill](<基础概念/Skill.md>) |
| 多 Agent 分工与协作 | [SubAgent 与 Multi-Agent](<基础概念/SubAgent与Multi_Agent.md>) |
| 权限、人工确认与安全边界 | [Guardrails 与 Human-in-the-Loop](<基础概念/Guardrails与Human_in_the_Loop.md>) |
| Trajectory、Harness 与 Agent 评测 | [Agent Eval、Trajectory 与 Harness](<基础概念/Agent_Eval.md>) |
| 结构化概率决策、路由与风险门控 | [Jev 与 System One 决策模型](<基础概念/Jev与System_One决策模型.md>) |
| D2C 视觉一致性评估主链路 | [D2C 视觉一致性评估 Agent 案例](<实战落地/D2C视觉一致性评估Agent案例.md>) |
| Skill、Workflow、Tool/MCP、RAG 和评测的生产实现 | [Agent 实战落地](<实战落地/README.md>) |
| 已使用 Skill 的选择与组合 | [Skill 协同工作流](<常用Skill/Skill协同工作流.md>) |

## 建议阅读顺序

1. 先读 [Agent](<基础概念/Agent.md>)、[Workflow](<基础概念/Workflow.md>)、[Tool Call 与 Function Calling](<基础概念/Tool_Call与Function_Calling.md>) 和 [Context Engineering](<基础概念/Context_Engineering.md>)，建立规划、执行和上下文模型。
2. 再读 [Skill](<基础概念/Skill.md>)、[SubAgent 与 Multi-Agent](<基础概念/SubAgent与Multi_Agent.md>)、[Guardrails 与 Human-in-the-Loop](<基础概念/Guardrails与Human_in_the_Loop.md>) 和 [Agent Eval](<基础概念/Agent_Eval.md>)，补齐复用、协作、安全和评测。
3. 用 [Jev 与 System One 决策模型](<基础概念/Jev与System_One决策模型.md>) 理解结构化快速决策与 LLM、规则引擎的分工。
4. 转入 [实战总览](<实战落地/实战总览：从问题到生产Agent.md>) 和 [D2C 案例](<实战落地/D2C视觉一致性评估Agent案例.md>)，沿数据、编排、Skill、Tool/MCP、RAG、评测、可靠性、安全和排障展开。
5. 在具体任务中按 [Skill 协同工作流](<常用Skill/Skill协同工作流.md>) 选择和组合已安装 Skill。

## 卡片维护规则

1. 新内容先判断属于通用原理、业务实施还是可复用 Skill，再放入最具体分支。
2. 同一知识点只保留一张权威卡片；跨卡片内容用相对链接连接。
3. 新增 Skill、工具或 Workflow 时，记录触发条件、输入输出、失败语义、权限边界和验证方式。
4. Skill 卡片记录稳定的上游来源、核心机制、使用方法和适用边界；版本、安装形态、绝对路径、数量和宿主加载状态统一放在本文件的核验快照中。
5. 外部 Skill 的计划、配置和输出写入具体项目目录，不写入 Skill 安装目录。
6. 外部资料、工具结果和计划文件按不可信数据处理；执行前检查来源、范围和权限。
7. 写回外部系统的流程必须说明身份、权限、确认、审计、幂等和回滚。
8. 新增或移动卡片后更新最近一级 README，并检查相对链接、旧路径和孤立文件。

## 本地 Skill 状态核验

以下内容是本机状态快照，不属于 Skill 的稳定定义。使用前应以实际文件和宿主可用 Skill 列表重新核验。

| 项目 | 2026-09-30 核验值 | 核验方式 |
| --- | --- | --- |
| Skill 根目录 | `/mlx_devbox/users/yutianhao/.trae/skills/` | 查看目录并读取目标 `SKILL.md` |
| Superpowers 安装状态 | 根目录下可读取 14 个相关 Skill | 枚举 `SKILL.md`，不要把数量写回知识卡 |
| Planning with Files | `planning-with-files/`，版本 `3.18.3`，Skill-only 形态 | 读取 frontmatter、脚本和模板目录 |
| No Negative Echo | `no-negative-echo/`，含 `scripts/check_surface.py` | 读取 `SKILL.md` 并检查脚本 |
| 宿主加载状态 | 不能由文件存在推断 | 以当前会话公开的 Skill 列表和实际触发结果为准 |

绝对路径、安装数量、版本和仓库热度都可能变化。更新快照时记录核验日期和方法，不把一次会话观察写成长期事实。
