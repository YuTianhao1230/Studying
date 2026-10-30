# Superpowers

## 基本信息

| 项目 | 内容 |
| --- | --- |
| GitHub 仓库 | [obra/superpowers](https://github.com/obra/superpowers) |
| 仓库定位 | Agentic skills framework & software development methodology |
| 许可证 | MIT |
| 安装方式 | 以仓库中的 `skills/<skill-name>` 作为 Skill 目录安装 |

## 主要做什么

Superpowers 为软件开发任务提供一套可组合的工作流程。它把“需求澄清、设计、计划、实现、测试、调试、代码审查和交付验证”拆成不同 Skill，让 Agent 在不同阶段采用明确的方法和检查点。

## 解决什么问题

它主要解决 Agent 在软件工程任务中的流程失控问题：

- 需求还没有澄清就开始写代码，导致实现方向偏移。
- 多阶段任务缺少计划，长会话中容易丢失上下文和验收标准。
- 遇到报错时反复猜测和打补丁，没有先定位根因。
- 只生成测试文件但没有真正运行测试，或在没有证据时宣称完成。
- 多人或多 Agent 并行修改同一工作区，造成改动互相覆盖。

## 怎么解决

它通过阶段化流程和硬性检查点解决上述问题：

1. `brainstorming` 先判断任务是 Spike、Bounded 还是 Architectural，再澄清目标、约束、方案和验收标准。创造性开发在设计获得确认前不进入实现。
2. `writing-plans` 把已确认的需求拆成文件级、步骤级、可验证的实现计划，明确接口、测试和执行顺序。
3. `executing-plans` 按计划分批执行，在阶段之间保留检查点；`subagent-driven-development` 和 `dispatching-parallel-agents` 支持拆分独立任务。
4. `test-driven-development` 强调先写失败测试，再用最小实现使测试通过，最后重构。
5. `systematic-debugging` 要求先读取错误、稳定复现、追踪数据流、形成单一假设，再做最小修复。
6. `verification-before-completion` 要求在声称完成前运行新鲜的验证命令并读取结果，以证据支持结论。
7. `using-git-worktrees`、代码审查和 `finishing-a-development-branch` 负责隔离改动、复核结果和收尾交付。

## 怎么用

### 常用组合

```text
新功能：brainstorming -> writing-plans -> executing-plans -> verification-before-completion
Bug：systematic-debugging -> test-driven-development -> verification-before-completion
复杂开发：using-git-worktrees -> brainstorming -> writing-plans -> subagent-driven-development
交付前：requesting-code-review -> receiving-code-review -> finishing-a-development-branch
```

### 直接使用

在支持 Skill 名称调用的宿主中，可以通过 Skill 名称显式调用，例如：

```text
$brainstorming
$systematic-debugging
$verification-before-completion
```

如果宿主按自动触发规则工作，则根据任务类型使用对应 Skill：创造性变更进入 `brainstorming`，遇到故障进入 `systematic-debugging`，准备交付时进入 `verification-before-completion`。

### 本地检查

```bash
find <skill-root> -maxdepth 2 -name SKILL.md | sort
sed -n "1,80p" <skill-root>/brainstorming/SKILL.md
```

## 适用边界

- 适合软件开发、研究代码、复杂文档工程和需要明确验证的长任务。
- `brainstorming` 的设计确认门槛较强，简单的一次性查询不需要套完整开发流程。
- TDD 需要有可运行的测试环境；纯文档整理可以使用其验证思想，但不必虚构代码测试。
- Skill 文件可读不等于宿主当前会话已经加载；应通过宿主公开的能力列表或实际触发结果验证。

## 面试应对

### Superpowers 的核心价值是什么？

回答思路：不要把它说成单一编码工具，按开发阶段和检查点说明其作用。

回答模板：

Superpowers 是一组面向软件工程生命周期的流程 Skill。它把需求澄清、设计、计划、实现、测试、调试、代码审查和交付验证拆成明确阶段，并为每个阶段设置进入条件和证据要求。核心价值不是让模型写更多代码，而是减少需求未确认就实现、遇错盲改和没有验证就宣布完成等流程失控问题。

### 新功能和故障修复为什么要走不同流程？

回答思路：对比未知点：新功能需要先确定目标和设计，故障需要先基于证据定位根因。

回答模板：

新功能的主要风险是目标和方案不清，因此先用 brainstorming 澄清需求、比较方案并获得设计确认，再制定计划和实现。故障修复的主要风险是误判根因，因此先用 systematic-debugging 稳定复现、收集证据和验证单一假设，再用测试驱动最小修复。两条流程最后都进入 completion verification，用新鲜测试结果支撑结论。

### 如何避免流程 Skill 变成形式主义？

回答思路：说明流程深度要与风险匹配，并要求每个产物服务于下一步决策或验证。

回答模板：

我会让流程深度与任务复杂度和风险匹配。小范围修改只需要简短设计、聚焦实现和对应验证；跨模块或高风险任务才需要完整规格、计划和审查。每个产物都必须服务于下一步，例如设计明确接口，计划明确可验证步骤，测试证明行为。如果某个文档既不减少歧义，也不支持执行或验收，就不应为了形式而创建。

## 与当前工作区的结合

- `brainstorming` 适合在修改 `Studying` 的目录结构、索引和学习卡片前先明确目标和合并范围。
- `writing-plans` 和 `planning-with-files` 可以共同使用：前者描述实现步骤，后者把长任务状态持久化到项目文件。
- `verification-before-completion` 适合检查 Markdown 链接、词表扫描、`git diff --check` 和安装文件回读结果。
