# Agent_Skills设计与维护

## 知识点解析

### 概述

资料来源：[Designing, Refining and Maintaining Agent Skills at Perplexity](https://research.perplexity.ai/articles/designing-refining-and-maintaining-agent-skills-at-perplexity)

### 核心结论

Skill 不是普通代码文档，而是给 Agent 注入的“可调用上下文”。写 Skill 的目标不是把所有知识写全，而是在正确时机给模型提供它原本容易做错、漏做或不稳定执行的关键约束。

### Skill 是什么

### Skill 是目录，不只是一个文件

一个高质量 Skill 通常包含：

- `SKILL.md`：触发条件和核心指令。
- `scripts/`：确定性脚本，避免 Agent 每次重复造轮子。
- `references/`：大段文档，只在需要时读取。
- `assets/`：模板、schema、示例数据。
- `config.json`：首轮配置和用户偏好。

这种结构的价值在于“渐进加载”：常用规则放入口，重资料放到子文件，只有触发具体需求时才读。

### Skill 是格式

`SKILL.md` 的 frontmatter 至少要包含：

- `name`：通常与目录名一致。
- `description`：不是介绍文档，而是路由触发条件。

一个常见错误是把 description 写成“这个 Skill 做什么”。更好的写法是“用户在什么意图下应该加载这个 Skill”。

### Skill 是可调用上下文

Agent 不是一直加载所有 Skill，而是在运行中根据任务意图选择加载。通常存在三层上下文成本：

| 层级 | 加载内容 | 成本特点 |
| --- | --- | --- |
| Index | 所有 Skill 的 name + description | 每个会话都付费，必须极短 |
| Load | 被选中 Skill 的 `SKILL.md` 正文 | 一旦加载，会持续占用上下文 |
| Runtime | scripts、references、assets 等 | 只有 Agent 主动读取时才付费 |

### 什么时候需要 Skill

需要 Skill 的情况：

- 模型没有稳定掌握某个企业内部流程。
- 任务需要严格遵守团队约定。
- Agent 在无额外上下文时容易犯同一类错误。
- 某个流程需要可复现、一致、可审计。
- 需要注入领域品味、工程取舍或高价值反例。

不需要 Skill 的情况：

- 只是普通命令清单，模型本来就会。
- 内容与系统提示重复。
- 信息变化太快，维护成本大于收益。
- 只是人类文档，没有转化成 Agent 执行约束。

判断标准：

```text
如果没有这句话，Agent 是否会稳定做错？
如果不会，这句话就不该放进 Skill。
```

### 如何写 Skill

### Step 0：先写 Evals

先准备评测样例，再写 Skill。至少覆盖：

- 正例：什么请求必须触发这个 Skill。
- 反例：什么相邻请求不能触发这个 Skill。
- 已知失败：历史上 Agent 做错的 case。
- 端到端任务：加载 Skill 后是否真的完成任务。

负例很重要，因为 Skill 的最大风险不是“不加载”，而是“误加载污染上下文”。

### Step 1：写好 Description

description 是最难写的一行，它决定路由质量。

好的 description：

- 以“Load when...”或等价触发语义开头。
- 目标 50 词以内。
- 描述用户意图，不展开 workflow。
- 使用真实用户查询中的关键词。
- 包含少量边界词，避免误触发。

坏的 description：

- 介绍 Skill 有多有用。
- 把流程步骤塞进去。
- 覆盖范围过宽，导致和其他 Skill 互相抢路由。

### Step 2：正文只写高信号内容

Skill 正文不要写成普通 README。模型已知的通用步骤应删除，留下：

- 关键判断标准。
- 团队偏好的处理方式。
- 容易踩坑的特殊情况。
- 失败恢复策略。
- 输出格式约束。

示例：

```text
差：git log；git checkout main；git checkout -b；git cherry-pick...
好：Cherry-pick the commit onto a clean branch. Resolve conflicts preserving intent. If it cannot land cleanly, explain why.
```

### Step 3：利用目录层级

把大段、条件性、分支型内容拆出去：

- 确定性逻辑进 `scripts/`。
- 大文档进 `references/`。
- 模板和 schema 进 `assets/`。
- 特殊错误处理进 `SPECIAL_CASES.md` 或类似文件。

目标是让 `SKILL.md` 保持短而强，把 token 成本推迟到真正需要时。

### Step 4：迭代

description 的小改动可能造成大范围路由变化，因此需要在合并前反复跑正例、反例和相邻 Skill 的回归。

### Step 5：发布

发布前最好是单个完整 changeset，包含：

- Skill 本体。
- 相关 scripts/references/assets。
- eval 集合。
- 已知 gotchas。

### 如何维护 Skill

Skill 的维护重点是 gotchas 飞轮：

```text
Agent 犯错
  -> 抽象成 gotcha
  -> 加入 Skill 或 reference
  -> 补充 eval
  -> 防止同类错误复发
```

维护原则：

- description 不要频繁改，除非有 eval 支撑。
- gotchas 多数情况下 append-only。
- 新增 Skill 可能影响其他 Skill，要做边界回归。
- 系统提示变化后，要检查 Skill 是否重复或冲突。
- 针对不同模型做回归，因为不同模型对 Skill 路由和文件读取行为可能不同。

### 面试回答要点

如果被问“如何设计一个 Agent Skill”，可以按这个顺序回答：

1. 先定义目标：这个 Skill 要修复 Agent 哪类不稳定行为。
2. 设计触发边界：description 只写用户意图和触发条件。
3. 分层组织上下文：`SKILL.md` 放核心规则，复杂内容放 references/scripts/assets。
4. 用 eval 验证：正例、反例、相邻 Skill 误触发、端到端完成度。
5. 持续维护 gotchas：从真实失败中沉淀高价值反例。

### 易错点

- 把 Skill 写成人类教程。
- 把命令流水账塞进 `SKILL.md`。
- description 太宽，导致误加载。
- 没有负例，无法验证边界。
- 大量内容常驻上下文，降低其他能力。
- 用 LLM 一次性生成 Skill，不做迭代和评测。

## 面试应对

### Agent_Skills设计与维护 是什么？

回答思路：先给清晰定义，再说明它解决的问题和适用边界。

回答模板：

Agent_Skills设计与维护 是一个需要从定义、机制、场景和限制一起理解的知识点。资料来源：Designing, Refining and Maintaining Agent Skills at Perplexity

### Agent_Skills设计与维护 的核心机制是什么？

回答思路：拆关键步骤和影响因素回答。

回答模板：

Agent_Skills设计与维护 的核心机制是：Skill 不是普通代码文档，而是给 Agent 注入的“可调用上下文”。写 Skill 的目标不是把所有知识写全，而是在正确时机给模型提供它原本容易做错、漏做或不稳定执行的关键约束。

### Agent_Skills设计与维护 有哪些使用场景和注意事项？

回答思路：先讲场景，再讲风险和边界。

回答模板：

Agent_Skills设计与维护 常见使用场景包括：资料来源：Designing, Refining and Maintaining Agent Skills at Perplexity 使用时要注意：资料来源：Designing, Refining and Maintaining Agent Skills at Perplexity
