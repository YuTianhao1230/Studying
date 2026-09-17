# 常用 Skill

本分支记录已经安装或实际使用过的 Agent Skill。每张卡片固定说明 GitHub 来源、主要用途、解决的问题、核心机制、安装位置、使用方式和适用边界。

## 当前 Skill

| Skill | GitHub 仓库 | 主要用途 | 本地入口 |
| --- | --- | --- | --- |
| [Superpowers](<Superpowers.md>) | [obra/superpowers](https://github.com/obra/superpowers) | 软件开发流程与协作方法，包括 brainstorming、计划、TDD、调试、代码审查和完成前验证。 | `/mlx_devbox/users/yutianhao/.trae/skills/` 下的 14 个 Skill 目录 |
| [Planning with Files](<Planning_with_Files.md>) | [OthmanAdi/planning-with-files](https://github.com/OthmanAdi/planning-with-files) | 用持久化 Markdown 文件保存任务计划、发现和进度，支持长任务恢复与上下文压缩后的续跑。 | `/mlx_devbox/users/yutianhao/.trae/skills/planning-with-files/` |
| [No Negative Echo](<No_Negative_Echo.md>) | [LB623/no-negative-echo](https://github.com/LB623/no-negative-echo) | 让最终标题、说明、元数据和交接文本从已接受结果出发，减少工作过程中的无关信息残留。 | `/mlx_devbox/users/yutianhao/.trae/skills/no-negative-echo/` |

## 推荐使用顺序

1. 开始创造性开发、功能设计或行为变更时，先参考 [Superpowers](<Superpowers.md>) 的 `brainstorming`，明确目标、约束、方案和验收标准。
2. 任务包含多个阶段、需要多轮工具调用或可能经历上下文压缩时，启用 [Planning with Files](<Planning_with_Files.md>)，把计划和发现写入项目目录。
3. 设计确认后，用 Superpowers 的 `writing-plans`、`executing-plans`、`test-driven-development` 和 `verification-before-completion` 完成实现与验证。
4. 遇到异常时使用 `systematic-debugging`；涉及多人或多 Agent 协作时使用 `using-git-worktrees`、`subagent-driven-development` 或 `dispatching-parallel-agents`。
5. 交付文档、索引、标题或交接信息前，用 [No Negative Echo](<No_Negative_Echo.md>) 检查最终表面是否准确表达已接受结果。

## 记录规则

- GitHub 仓库链接指向上游来源，Skill 名称和版本以本地 `SKILL.md` 为准。
- 安装位置与宿主加载状态分开记录：文件落地可以验证，当前会话是否热加载需要单独验证。
- 外部 Skill 的计划文件、配置和输出属于项目工作产物，应写在项目目录，不写进安装目录。
- 计划、网页资料和外部输入都按不可信数据处理；执行前检查来源、范围和权限。

## 后续扩展

后续使用新的 Skill 时，新增一张卡片并在本页登记。优先记录真正解决过问题的 Skill，避免把仅浏览过的仓库当成已使用能力。
