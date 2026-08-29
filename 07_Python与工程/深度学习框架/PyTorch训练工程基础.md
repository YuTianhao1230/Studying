# PyTorch 训练工程基础

## 知识点解析

### 概述

PyTorch 是当前深度学习和大模型训练中最常用的框架之一。它的核心优势是动态图、Pythonic、调试方便、生态完整，适合研究实验、自定义模型、复杂训练逻辑和大模型微调。

理解 PyTorch 不应该停留在会写 `model(x)`，而要掌握一条完整训练链路：Tensor 表示数据，`nn.Module` 管理模型参数，Autograd 自动求导，Optimizer 更新参数，Dataset/DataLoader 提供数据，Checkpoint 保存训练状态。

### 核心组件

`torch.Tensor` 是 PyTorch 的基本数据结构，表示输入、标签、模型参数和中间激活。Tensor 可以放在 CPU 或 GPU 上，训练时最常见的问题之一就是 device 不一致，例如模型在 GPU、输入还在 CPU。

`torch.nn.Module` 是模型基类，负责注册参数和子模块。一般在 `__init__` 里定义层，在 `forward` 里定义数据流。只要层被挂到 `self.xxx` 上，PyTorch 就能通过 `model.parameters()` 找到可训练参数。

Autograd 是自动求导系统。前向传播时，PyTorch 会记录张量操作形成动态计算图；调用 `loss.backward()` 后，沿计算图反向计算梯度。训练中要注意梯度默认累积，所以每步更新前通常需要 `optimizer.zero_grad()`。

Optimizer 负责根据梯度更新参数。典型流程是 `zero_grad -> forward -> loss -> backward -> step`。大模型训练里常见 AdamW，并配合 learning rate scheduler、warmup、gradient clipping、mixed precision。

Dataset 和 DataLoader 负责数据管线。Dataset 定义如何取一个样本，DataLoader 负责 batching、shuffle、多进程加载和 collate。对于文本、多模态、变长输入，`collate_fn` 经常是关键点。

### 标准训练循环

一个最小训练循环通常包括：

1. 设置模型为训练模式：`model.train()`。
2. 从 DataLoader 取 batch。
3. 把 batch 移到正确 device。
4. 前向计算 logits 或 loss。
5. 反向传播：`loss.backward()`。
6. 梯度裁剪或混合精度处理。
7. 参数更新：`optimizer.step()`。
8. 清空梯度：`optimizer.zero_grad()`。
9. 记录 loss、lr、grad norm、吞吐等日志。

评估阶段要用 `model.eval()`，并配合 `torch.no_grad()` 或 `torch.inference_mode()`，避免构建计算图，节省显存和时间。

### 工程注意点

训练脚本要能复现，至少要保存模型权重、optimizer 状态、scheduler 状态、随机种子、训练 step、数据版本和代码版本。只保存模型权重，通常无法严格断点续训。

显存问题需要分来源看：参数、梯度、优化器状态、激活、中间 buffer、输入 batch。常见优化手段包括 mixed precision、gradient accumulation、activation checkpointing、减小 sequence length、FSDP/ZeRO、LoRA。

训练不稳定时不能只看 loss。还要看学习率曲线、grad norm、样本长度分布、异常 batch、数据质量、label mask、混合精度溢出和分布式通信错误。

### PyTorch 和大模型生态

PyTorch 本身提供基础张量、模型、优化器和分布式能力；Hugging Face Transformers 提供模型结构、tokenizer、预训练权重加载和 Trainer；PEFT 提供 LoRA/QLoRA；Accelerate、FSDP、DeepSpeed、Megatron-LM 负责大规模训练；vLLM、TensorRT-LLM、TGI 更偏推理部署。

大模型训练里，面试官通常不是问你会不会写 `nn.Linear`，而是问你是否能把 PyTorch 放进真实训练系统里：数据怎么喂、梯度怎么传、显存怎么省、checkpoint 怎么恢复、分布式怎么排障。

## 面试应对

### PyTorch 的训练流程是什么？

回答思路：按数据、模型、loss、反向传播、优化器、日志和 checkpoint 顺序回答。

回答模板：

PyTorch 训练流程通常是先用 Dataset/DataLoader 构造 batch，把输入和标签移动到对应 device，然后调用 `model.train()` 进入训练模式。每一步先清空梯度，前向计算 logits 和 loss，再调用 `loss.backward()` 反向传播，必要时做 gradient clipping 或 mixed precision，最后 `optimizer.step()` 更新参数。工程上还要记录 loss、learning rate、grad norm 和吞吐，并定期保存 model、optimizer、scheduler、step 和随机种子，保证可恢复。

### `model.train()` 和 `model.eval()` 有什么区别？

回答思路：指出它们不是控制是否求导，而是控制模块行为，重点讲 dropout 和 batch norm。

回答模板：

`model.train()` 和 `model.eval()` 控制的是模型中某些层的行为，不是直接控制梯度。比如 dropout 在 train 模式会随机丢弃神经元，在 eval 模式会关闭；batch norm 在 train 模式使用 batch 统计量并更新 running mean/var，在 eval 模式使用历史统计量。是否计算梯度主要由 `torch.no_grad()`、`torch.inference_mode()` 和张量的 `requires_grad` 决定。

### 为什么每步训练要 `optimizer.zero_grad()`？

回答思路：说明 PyTorch 梯度默认累积，再补充 gradient accumulation 的例外。

回答模板：

PyTorch 中参数的 `.grad` 默认是累积的，也就是说每次 `loss.backward()` 会把新梯度加到已有梯度上。如果每个 step 前不清空梯度，模型实际更新的就是多个 batch 梯度的累加，训练会不符合预期。所以标准训练循环会调用 `optimizer.zero_grad()`。不过在 gradient accumulation 场景下，会故意累积多个 micro batch 的梯度，再统一 step。

### `torch.no_grad()` 和 `torch.inference_mode()` 有什么区别？

回答思路：先讲共同点，再讲 inference_mode 更彻底但限制更多。

回答模板：

二者都用于不需要反向传播的场景，可以减少显存和计算图开销。`torch.no_grad()` 是关闭 autograd 记录；`torch.inference_mode()` 更进一步，还会关闭一些版本计数和 view tracking，因此推理更快、开销更低。但 inference_mode 更严格，适合纯推理；如果后面还要把结果接回需要梯度的计算，使用 no_grad 更稳。

### PyTorch 训练 OOM 你会怎么排查？

回答思路：按显存来源拆解，再给优化手段。

回答模板：

我会先确认 OOM 发生在 forward、backward、optimizer step 还是 evaluation。显存主要来自参数、梯度、优化器状态、激活、输入 batch 和临时 buffer。优化手段包括减小 batch size 或 sequence length、使用 bf16/fp16、gradient accumulation、activation checkpointing、清理无用 tensor、避免保存带计算图的 loss、使用 LoRA、FSDP 或 ZeRO。大模型场景还要关注 tokenizer 后的真实 token 长度和多模态 token 数。
