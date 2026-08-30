# 训练框架与并行

本目录整理大模型训练系统和分布式训练框架，重点理解显存拆分、通信、混合精度、checkpoint 和不同框架的工程取舍。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [DeepSpeed.md](<DeepSpeed.md>) | DeepSpeed 训练框架、ZeRO、优化器和大模型训练工程能力。 |
| [ZeRO.md](<ZeRO.md>) | 参数、梯度、优化器状态切分的显存优化机制。 |
| [FSDP.md](<FSDP.md>) | PyTorch Fully Sharded Data Parallel 的 sharding 和通信流程。 |
| [Megatron_LM.md](<Megatron_LM.md>) | Megatron-LM 的张量并行、流水线并行和大模型训练范式。 |
| [JAX与XLA.md](<JAX与XLA.md>) | JAX/XLA 的编译式训练、函数式编程和性能优化特点。 |
| [Mixed Precision Training.md](<Mixed Precision Training.md>) | fp16/bf16、loss scaling 和混合精度训练稳定性。 |
| [Checkpoint.md](<Checkpoint.md>) | 模型、优化器、调度器和训练状态保存/恢复机制。 |

## 学习路线

1. 先看 [Mixed Precision Training.md](<Mixed Precision Training.md>) 和 [Checkpoint.md](<Checkpoint.md>)，理解单次训练任务的基础工程。
2. 再看 [ZeRO.md](<ZeRO.md>) 和 [FSDP.md](<FSDP.md>)，理解显存切分。
3. 接着看 [DeepSpeed.md](<DeepSpeed.md>) 和 [Megatron_LM.md](<Megatron_LM.md>)，理解大规模训练框架。
4. 最后看 [JAX与XLA.md](<JAX与XLA.md>)，补充编译式训练生态。
