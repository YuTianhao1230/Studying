# Agentic RAG

## 一句话解释

Agentic RAG 是把传统 RAG 从“一次检索 + 一次生成”升级为可规划、可多轮检索、可调用工具、可自我校验的 Agent 工作流。

## 传统 RAG

传统 RAG 通常是：

```text
用户问题
  -> 向量检索
  -> 拼接上下文
  -> LLM 生成答案
```

优点是简单稳定；缺点是面对复杂问题时容易一次检索不够、证据不足或无法纠错。

## Agentic RAG

Agentic RAG 更像：

```text
用户问题
  -> 判断需要哪些信息
  -> 多轮检索 / 搜索 / 工具调用
  -> 比较证据
  -> 发现缺口后继续检索
  -> 生成答案
  -> 校验引用和结论
```

它把检索过程变成可迭代的决策过程。

## 常见能力

- Query Planning：把复杂问题拆成多个查询。
- Query Rewriting：改写检索词。
- Multi-hop Retrieval：多跳检索。
- Tool Routing：决定用搜索、数据库、文档还是代码工具。
- Evidence Verification：检查证据是否支持结论。
- Citation Grounding：给答案绑定引用来源。
- Self-correction：发现证据不足时继续检索。

## 适合场景

- 多文档问答。
- 企业知识库问答。
- 复杂技术调研。
- 需要引用来源的报告生成。
- 需要跨系统查询的业务分析。

不适合：

- 简单事实问答。
- 低延迟强约束场景。
- 检索源质量很差且无法校验的场景。

## 和普通 Agent 的关系

Agentic RAG 是 Agent 的一种具体应用。它的工具主要围绕信息获取和证据校验，而不是任意外部动作。

```text
Agentic RAG = RAG + Planning + Tool Use + Reflection + Verification
```

## 常见失败模式

- 检索到相似但不相关的内容。
- 多轮检索越跑越偏。
- 引用不支持结论。
- 上下文过长导致关键信息被忽略。
- 没有区分事实、推断和假设。
- 为简单问题引入过重流程。

## 面试可能怎么问

1. Agentic RAG 和传统 RAG 的区别是什么？
2. 什么场景需要多轮检索？
3. 如何判断 RAG 答案是否有证据支撑？
4. 如何减少检索噪声？
5. Agentic RAG 为什么更慢、更贵？
