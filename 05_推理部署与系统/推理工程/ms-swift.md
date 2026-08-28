# ms-swift

## 一句话解释

ms-swift 是 ModelScope 生态里的大模型训练、微调、推理和评测工具链，常用于 SFT、LoRA、QLoRA、推理测试和模型导出。

## 它的定位

ms-swift 不是像 vLLM 那样专注高吞吐在线 serving 的纯推理引擎。

它更像一个大模型研发工具箱：

- 支持加载多种开源模型。
- 支持 SFT、LoRA、QLoRA 等微调。
- 支持数据集处理。
- 支持训练配置管理。
- 支持简单推理和评测。
- 支持模型导出和部署衔接。

## 常见使用场景

### 1. 指令微调

用已有 base model 和指令数据训练一个更适合任务的模型。

```text
base model
  + SFT data
  -> fine-tuned model
```

### 2. LoRA / QLoRA 微调

冻结大部分原模型参数，只训练少量低秩适配参数，降低显存成本。

```text
base model frozen
  + trainable LoRA adapter
  -> adapter checkpoint
```

### 3. 推理验证

训练后用少量样本检查模型是否能正常回答。

### 4. 评测

在某些 benchmark 或自定义数据上跑模型效果。

## ms-swift 和 vLLM 的区别

| 维度 | ms-swift | vLLM |
|---|---|---|
| 主要定位 | 训练/微调/评测工具链 | 高吞吐推理服务框架 |
| 是否训练 | 支持 | 通常不负责训练 |
| 是否推理 | 支持 | 强项 |
| 是否 serving | 可衔接部署 | 原生面向服务 |
| 核心优势 | 训练流程和模型适配 | KV cache 管理和并发吞吐 |

## 为什么项目会从 ms-swift 迁移到 xLLM / vLLM 类服务

常见原因：

- ms-swift 更适合研发和微调，不一定适合大规模线上推理。
- 线上服务需要更稳定的并发、限流、监控和服务治理。
- 推理框架需要更强的 batching、KV cache 管理和多卡调度。
- 平台可能统一使用某种 serving 方案来降低运维成本。

## 常见误区

- 把 ms-swift 当成纯推理框架。
- 用训练脚本直接承担高并发线上服务。
- 推理参数没有和训练/评测时对齐，导致效果差异。
- LoRA adapter、base model、tokenizer 版本不一致。

## 面试可能怎么问

1. ms-swift 主要用于什么？
2. ms-swift 和 vLLM 的定位有什么区别？
3. LoRA 微调产物如何用于推理？
4. 为什么训练框架不一定适合线上 serving？
5. 模型从训练工具链迁移到推理服务时要注意什么？

