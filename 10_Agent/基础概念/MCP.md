# MCP

## 一句话解释

MCP，全称 Model Context Protocol，是一种把外部工具、数据源和 Prompt 标准化暴露给 Agent 的协议。

## 为什么需要 MCP

没有 MCP 时，每个 Agent 框架都要为每个工具单独写适配：

```text
Agent A -> 工具 1
Agent A -> 工具 2
Agent B -> 工具 1
Agent B -> 工具 2
```

工具和 Agent 越多，集成复杂度越高。

MCP 的目标是把这个关系变成：

```text
Agent / MCP Client <-> MCP Server <-> Tools / Resources / Prompts
```

只要工具以 MCP Server 的形式暴露，多个 Agent 就可以通过统一协议发现和调用。

## MCP 的三个核心对象

- Tools：可执行动作，例如查数据库、发请求、读文件、创建任务。
- Resources：可读取资源，例如文件、文档、数据库记录、配置。
- Prompts：可复用提示模板。

## MCP Client 和 MCP Server

- MCP Client：通常嵌在 Agent 或 IDE 里，负责连接 server、读取能力清单、发起调用。
- MCP Server：负责暴露工具、资源和 prompt，并执行真实逻辑。

## MCP 带来的价值

- 降低工具集成成本。
- 让工具可以跨 Agent 框架复用。
- 统一能力发现、参数描述和调用方式。
- 让企业内部系统更容易接入 Agent。

## 工程注意点

### 1. 工具数量不是越多越好

如果一次性把上百个工具定义塞进上下文，会增加 token 成本，也会让模型更容易选错工具。

### 2. 工具描述要面向模型

描述应该回答：

- 什么时候用？
- 输入是什么？
- 输出是什么？
- 不该什么时候用？

### 3. 大结果不要直接塞回上下文

对于大文档、大表格、大日志，最好让工具支持：

- 分页。
- 摘要。
- 过滤。
- 写临时文件。
- 返回引用路径。

### 4. 高风险操作需要人类确认

例如删除、支付、发消息、改权限、提交任务等，不能让 Agent 无门禁执行。

## MCP 和 Skill 的区别

- MCP：连接外部工具和数据的协议。
- Skill：告诉 Agent 什么时候、如何使用某类能力的上下文包。

二者可以配合：

```text
Skill 负责策略和流程
MCP 负责工具接入和执行
```

## 面试可能怎么问

1. MCP 解决了什么问题？
2. MCP 的 Tools、Resources、Prompts 分别是什么？
3. MCP Client 和 MCP Server 的职责是什么？
4. MCP 工具很多时会有什么问题？
5. MCP 和 Skill 如何配合？
