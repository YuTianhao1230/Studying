# TensorFlow 与 Keras 基础

## 知识点解析

### 概述

TensorFlow 是 Google 推出的深度学习框架，Keras 是其高层模型构建 API。TensorFlow 的特点是工程部署生态较完整，Keras 的特点是接口简洁、适合快速搭建常规模型。

在当前大模型训练生态里，PyTorch 更主流，但 TensorFlow/Keras 仍然是算法工程师需要了解的基础框架，尤其在传统工业部署、端侧推理、TensorFlow [Serving](<../../05_推理部署与系统/推理工程/Serving.md#serving>)、TFLite、TFX 等场景中仍会被问到。

### 基本使用方式

Keras 常见写法是先定义模型，再 `compile` 配置 loss、optimizer、metrics，最后用 `fit` 训练、`evaluate` 评估、`predict` 推理。

这套流程抽象程度高，适合标准监督学习任务。比如图像分类、文本分类、结构化数据建模，Keras 可以快速建立 baseline。相比 PyTorch 手写训练循环，Keras 默认帮你封装了训练、验证、日志和 callback。

如果需要更复杂的自定义训练逻辑，TensorFlow 也可以使用 `tf.GradientTape` 手写训练循环。`GradientTape` 类似 PyTorch autograd，会记录前向计算并用于反向求导。

### TensorFlow 的特点

TensorFlow 早期强调静态计算图，后来 TensorFlow 2 默认使用 eager execution，更接近动态图体验。但它仍然保留图编译和部署优势，例如 `tf.function` 可以把 Python 函数转换为图执行，提高性能并便于导出。

TensorFlow 生态里常见组件包括：

- Keras：高层模型构建和训练 API。
- SavedModel：模型保存和部署格式。
- TensorBoard：[可视化](<../常用库/可视化.md#可视化>)训练指标。
- TensorFlow Serving：服务化部署。
- TFLite：移动端和端侧推理。
- TFX：生产级 ML pipeline。

### 和 PyTorch 的区别

PyTorch 更偏研究和自定义训练，动态图调试直观；TensorFlow/Keras 更偏高层封装和生产部署生态。现在大模型开源生态中，PyTorch 占主流；但在一些历史系统、端侧部署和传统工业链路里，TensorFlow 仍然常见。

| 维度 | PyTorch | TensorFlow/Keras |
| --- | --- | --- |
| 编程体验 | Pythonic、动态图、调试方便 | Keras 高层封装简洁，TF 图模式更工程化 |
| 研究生态 | 大模型和开源研究更主流 | 相对弱一些 |
| 部署生态 | 依赖 TorchScript/ONNX/Serving 组件 | SavedModel、TF Serving、TFLite 较成熟 |
| 自定义训练 | 手写 loop 灵活 | `GradientTape` 可自定义，但 Keras 默认更封装 |
| 常见岗位要求 | 大模型训练、研究、微调 | 传统工业、端侧、老系统维护 |

### 使用边界

如果你主攻大模型训练，TensorFlow 不需要像 PyTorch 一样深入到每个 API，但至少要能回答它的基本训练流程、Keras 的抽象、`GradientTape` 的作用、`tf.function` 的意义，以及它和 PyTorch 的选型差异。
