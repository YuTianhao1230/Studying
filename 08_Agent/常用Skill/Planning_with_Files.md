# Planning with Files

## 基本信息

| 项目 | 内容 |
| --- | --- |
| GitHub 仓库 | [OthmanAdi/planning-with-files](https://github.com/OthmanAdi/planning-with-files) |
| 仓库定位 | Persistent file-based planning for AI coding agents and long-running tasks |
| 默认分支 | `master` |
| 当前版本 | `3.18.3`，以本地 `SKILL.md` 的 metadata 为准 |
| 许可证 | MIT |
| 本地安装位置 | `/mlx_devbox/users/yutianhao/.trae/skills/planning-with-files/` |
| 当前安装形态 | Skill-only 安装，已包含 `SKILL.md`、脚本和模板 |

## 主要做什么

Planning with Files 把长任务的工作记忆持久化到项目文件中。它围绕一个被选中的计划目录维护：

```text
task_plan.md   任务目标、阶段、决策和下一步
findings.md    调研发现、技术事实和外部资料摘要
progress.md    执行日志、阶段结果和验证记录
.planning/<id>/  多任务或并行工作时的隔离计划目录
```

它还提供计划解析、初始化、完成检查、账本记录、会话恢复和生命周期 hook 脚本。

## 解决什么问题

它主要解决长时间 Agent 任务中的状态丢失和计划漂移：

- 多轮工具调用后，任务目标、已完成阶段和下一步不再清晰。
- 会话压缩、清空或切换后，Agent 只能依赖不完整的聊天上下文恢复。
- 调研发现和外部资料没有固定落点，后续容易重复搜索或误用旧结论。
- 多个并行任务共用一个根计划，可能互相覆盖进度。
- Agent 在没有读回计划、检查命令和文件状态时提前结束。

## 怎么解决

1. 用 `init-session.sh` 初始化计划，生成任务目录和 `PLAN_ID`；有多个计划时用 `resolve-plan-dir.sh` 或 `.planning/.active_plan` 选择唯一计划。
2. 把目标、阶段、决策和错误写入 `task_plan.md`，把调研和外部内容写入 `findings.md`，把执行过程和测试结果写入 `progress.md`。
3. 每个阶段结束后更新状态和 `Next Step`，在重要决策前重新读取计划；通过文件系统而不是聊天记录保存关键状态。
4. 并行任务使用独立计划目录或独立 `PLAN_ID`，由一个计划所有者维护共享摘要，避免多个 Agent 同时覆盖同一计划文件。
5. `check-complete.sh` 根据计划阶段状态做完成检查；`verification-before-completion` 则负责在对外声称完成前读取真实验证输出，两者关注点不同。
6. Skill 的 hook 可以在用户输入、工具调用、写文件、停止和压缩等生命周期节点注入计划上下文；但 hook 是否被当前宿主加载，需要单独验证。

## 怎么用

### 初始化长任务

```bash
cd /path/to/project
sh /mlx_devbox/users/yutianhao/.trae/skills/planning-with-files/scripts/init-session.sh "任务名称"
export PLAN_ID=<脚本输出的计划 ID>
export PWF_PLAN_ROOT=/path/to/project
```

初始化后，先读取选中的 `task_plan.md`、`findings.md` 和 `progress.md`，再开展多步工作。

### 典型工作循环

```text
初始化/恢复计划
  -> 读取 task_plan.md、findings.md、progress.md
  -> 执行 1~2 个动作
  -> 把新发现写入 findings.md 或 progress.md
  -> 更新阶段状态和 Next Step
  -> 运行检查命令
  -> 完成前重新读取文件并验证
```

### 常用脚本

```bash
SKILL_DIR=/mlx_devbox/users/yutianhao/.trae/skills/planning-with-files
sh "$SKILL_DIR/scripts/resolve-plan-dir.sh"
sh "$SKILL_DIR/scripts/check-complete.sh"
sh "$SKILL_DIR/scripts/plan-doctor.sh"
python3 "$SKILL_DIR/scripts/session-catchup.py" --metadata "$(pwd)"
```

`session-catchup.py` 的 `--metadata` 和 `--replay` 是显式模式；普通调用不会读取宿主会话记录。外部网页、命令输出和写入 `findings.md` 的资料都应作为不可信数据处理。

### 可选模式

仓库还提供 `--autonomous` 和 `--gated` 计划模式。它们会改变计划注入和完成门行为，适合明确需要持续推进的长任务；默认 legacy 模式更适合普通交互式任务。使用前应确认宿主支持相应 hook 或停止事件。

## 适用边界

- 适合研究、资料整理、复杂开发和 5 次以上工具调用的长任务。
- 一次性问答、单文件小改动不需要额外建立计划文件。
- 它持久化的是计划和工作状态，不会自动替代测试、代码审查或业务验收。
- Skill-only 安装包含 `SKILL.md`、脚本和模板，但不包含仓库插件安装路径里的 slash commands；当前 TraeCode 是否支持这些 hook 不能仅凭文件落地确认。
- 计划文件在项目目录中产生，属于工作产物；不要把项目计划写入 Skill 安装目录。

## 与当前工作区的结合

- 适合整理 `Studying` 这类跨目录知识库、长时间论文返修、实验复盘和多阶段资料建设任务。
- `task_plan.md` 管阶段和决策，`findings.md` 管外部资料与调研事实，`progress.md` 管执行日志和验证结果。
- 本卡片记录的是 Skill 的使用方法；实际任务计划应创建在具体项目或工作区的计划目录中。
