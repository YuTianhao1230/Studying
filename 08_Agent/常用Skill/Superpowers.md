# Superpowers

## 基本信息

| 项目 | 内容 |
| --- | --- |
| GitHub 仓库 | [obra/superpowers](https://github.com/obra/superpowers) |
| 仓库定位 | Agentic skills framework & software development methodology |
| 许可证 | MIT |
| 安装方式 | 以仓库中的 `skills/<skill-name>` 作为 Skill 目录安装 |
| 本地安装位置 | `/mlx_devbox/users/yutianhao/.trae/skills/` |
| 当前安装数量 | 14 个 Skill |

## 主要做什么

Superpowers 为软件开发任务提供一套可组合的工作流程。它把“需求澄清、设计、计划、实现、测试、调试、代码审查和交付验证”拆成不同 Skill，让 Agent 在不同阶段采用明确的方法和检查点。

当前安装的 Skill 包括：

```text
brainstorming
writing-plans
executing-plans
test-driven-development
systematic-debugging
verification-before-completion
using-git-worktrees
using-superpowers
requesting-code-review
receiving-code-review
subagent-driven-development
dispatching-parallel-agents
finishing-a-development-branch
writing-skills
```

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
find /mlx_devbox/users/yutianhao/.trae/skills -maxdepth 2 -name SKILL.md | sort
sed -n "1,80p" /mlx_devbox/users/yutianhao/.trae/skills/brainstorming/SKILL.md
```

## 适用边界

- 适合软件开发、研究代码、复杂文档工程和需要明确验证的长任务。
- `brainstorming` 的设计确认门槛较强，简单的一次性查询不需要套完整开发流程。
- TDD 需要有可运行的测试环境；纯文档整理可以使用其验证思想，但不必虚构代码测试。
- Superpowers 的 Skill 文件已安装并可读，不等于宿主当前会话已经热加载；重启 TraeCode 或开启新会话后再验证可用 Skill 列表。

## 与当前工作区的结合

- `brainstorming` 适合在修改 `Studying` 的目录结构、索引和学习卡片前先明确目标和合并范围。
- `writing-plans` 和 `planning-with-files` 可以共同使用：前者描述实现步骤，后者把长任务状态持久化到项目文件。
- `verification-before-completion` 适合检查 Markdown 链接、词表扫描、`git diff --check` 和安装文件回读结果。
