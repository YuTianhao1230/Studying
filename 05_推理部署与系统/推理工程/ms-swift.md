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
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：ms-swift 主要用于大模型训练、微调、对齐和实验管理，例如 SFT、LoRA、DPO 等。它更偏训练工具链，不是专门的线上 serving 框架。
2. ms-swift 和 vLLM 的定位有什么区别？
   - 回答思路：先分别定义两个概念，再从目标、机制、适用场景和工程取舍四个角度对比。
   - 回答模板：先分别定义两个对象，再比较目标、机制、适用场景和风险。围绕 ms-swift，关键是说明它们解决的问题不同。
3. LoRA 微调产物如何用于推理？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：推理时加载 base model 和 LoRA adapter，或先把 adapter merge 到 base model 后部署。必须保证 base model、tokenizer 和 adapter 配置匹配。
4. 为什么训练框架不一定适合线上 serving？
   - 回答思路：先指出背后的核心约束，再解释收益，最后补充如果不这样做会带来的风险。
   - 回答模板：训练框架关注梯度、优化器、数据加载和 checkpoint；线上 serving 关注并发、延迟、batching、资源隔离、灰度和监控。二者优化目标不同。
5. 模型从训练工具链迁移到推理服务时要注意什么？
   - 回答思路：先定义问题，再说明核心机制、适用边界、风险和验证方式。
   - 回答模板：要检查权重格式、base model、tokenizer、chat template、精度、量化、max length、采样参数和接口协议，并用回归集验证迁移前后输出。

