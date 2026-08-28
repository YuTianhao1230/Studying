# SubAgent 与 Multi-Agent

## 一句话解释

SubAgent 是由主 Agent 调度的专门子任务执行者；Multi-Agent 是多个 Agent 通过分工、通信和协作共同完成复杂任务的系统。

## 为什么需要 SubAgent

单个 Agent 同时处理所有事情会遇到几个问题：

- 上下文太长。
- 工具太多，选择困难。
- 任务之间互相干扰。
- 专业领域差异大。
- 并行能力不足。

SubAgent 的思路是把任务拆给更小、更专注的执行单元。

## 常见结构

### 1. Orchestrator-Workers

```text
主 Agent
  -> 拆分任务
  -> 分配给多个 Worker
  -> 汇总结果
  -> 输出最终答案
```

适合：

- 多文件代码审查。
- 多资料调研。
- 多模块实现。
- 多候选方案比较。

### 2. Specialist Agents

每个 Agent 负责一个专业领域：

- Research Agent：查资料。
- Coding Agent：写代码。
- Review Agent：审查代码。
- Test Agent：运行测试。
- Data Agent：分析数据。

### 3. Debate / Voting

多个 Agent 独立生成答案，再做投票或仲裁。

适合：

- 开放问题。
- 方案比较。
- 高风险判断。

## 主 Agent 的职责

- 理解用户目标。
- 拆解任务。
- 选择合适的 SubAgent。
- 控制上下文边界。
- 合并结果。
- 处理冲突。
- 负责最终交付质量。

## SubAgent 的职责

- 在窄领域内深入执行。
- 返回结构化结果。
- 不越权修改全局计划。
- 明确列出证据、假设和风险。

## 常见失败模式

- 过度拆分，沟通成本超过收益。
- SubAgent 之间信息不一致。
- 主 Agent 不做最终审查，直接拼接结果。
- 没有共享状态管理。
- 多个 Agent 同时改同一文件导致冲突。
- 缺少终止条件，循环委派。

## 设计原则

- 能用单 Agent 稳定完成时，不要强行多 Agent。
- SubAgent 职责要窄，输入输出要清晰。
- 主 Agent 必须保留最终决策权。
- 对并行任务要提前定义合并规则。
- 重要结论要带证据，不只带观点。

## 面试可能怎么问

1. SubAgent 和普通工具调用有什么区别？
2. 什么场景适合 Multi-Agent？
3. Orchestrator-Workers 模式如何工作？
4. 多 Agent 系统如何避免互相冲突？
5. 多 Agent 的成本和风险是什么？
