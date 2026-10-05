# 任务与数据治理

本目录处理关键帧项目的业务规则、标注质量、bad case、数据飞轮和 Prompt/规则治理，不负责具体训练框架参数。

## 内容索引

| 文件 | 内容 |
| --- | --- |
| [任务定义与标注标准.md](<任务定义与标注标准.md#关键帧检测任务定义与标注标准>) | 完成态、排除条件、豁免条件和证据优先级。 |
| [BadCase归因与UI元素治理.md](<BadCase归因与UI元素治理.md#关键帧检测bad-case-归因与-ui-元素治理>) | bad case 分类、小 UI、角标、Banner、GT 误差和 UI primitive。 |
| [数据飞轮与主动学习.md](<数据飞轮与主动学习.md#关键帧检测数据飞轮与主动学习>) | bad case 筛选、人工重标、分桶评测和主动采样。 |
| [并行DE与PE.md](<并行DE与PE.md#并行-de-与-pe>) | 数据工程和 Prompt/规则工程的并行治理闭环。 |
| [Agentic Model Optimization 模型自训练优化.md](<Agentic Model Optimization 模型自训练优化.md#agentic-model-optimization模型自训练与-loop-engineering>) | Agent 接管 bad case、数据策略、训练、评测和持续迭代的 Loop-Engineering。 |
