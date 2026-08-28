# Skill

## 一句话解释

Skill 是给 Agent 动态加载的领域能力包，通常由指令、脚本、参考文档、模板和配置组成，用来让通用 Agent 在特定任务上表现得更稳定。

## 为什么需要 Skill

大模型本身具备通用知识，但在这些场景下容易不稳定：

- 企业内部流程模型不知道。
- 某个工具调用有特殊参数和坑点。
- 团队有固定输出格式或工程规范。
- 相邻任务容易混淆，需要明确边界。
- 同类任务经常失败，需要沉淀 gotchas。

Skill 的价值是把这些经验变成 Agent 可复用的上下文。

## Skill 通常包含什么

```text
skill-name/
├── SKILL.md
├── scripts/
├── references/
├── assets/
└── config.json
```

- `SKILL.md`：触发条件和核心执行规则。
- `scripts/`：确定性逻辑，供 Agent 调用而不是重写。
- `references/`：大文档，只在需要时读取。
- `assets/`：模板、schema、样例。
- `config.json`：用户配置或环境配置。

## Skill 的核心原则

### 1. Description 是触发器

description 不是介绍“这个 Skill 很厉害”，而是告诉 Agent “什么时候应该加载它”。

好的 description 应该：

- 短。
- 具体。
- 覆盖真实用户意图。
- 避免和其他 Skill 抢路由。

### 2. 正文要高信号

不要把 Skill 写成人类教程。模型已经知道的通用知识不需要写进去。

应该写：

- 特殊规则。
- 工具坑点。
- 输出约束。
- 失败恢复。
- 禁止行为。

### 3. 渐进加载

常驻上下文很贵，所以要分层：

- Skill Index：只放名字和描述。
- `SKILL.md`：只放核心规则。
- references/scripts/assets：按需读取。

## Skill 和 Prompt 的区别

- Prompt：一次任务内的临时指令。
- Skill：可复用、可维护、可评测的长期能力包。

如果一条指令只服务于当前任务，用 Prompt 即可；如果它会被重复使用、需要稳定执行、涉及工具或特殊流程，就适合沉淀为 Skill。

## 常见误区

- 把 Skill 写成长篇知识库。
- 把命令清单原样塞进 Skill。
- description 写得太宽，导致误触发。
- 没有 eval，无法判断是否该加载。
- 没有 gotchas，无法从失败中进化。

## 面试可能怎么问

1. Skill 和普通 Prompt 有什么区别？
2. Skill 的 description 为什么重要？
3. 如何设计 Skill 的目录结构？
4. 什么内容应该放 `SKILL.md`，什么内容应该放 references？
5. 如何评测一个 Skill 是否有效？
