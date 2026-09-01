# ms-swift

## 知识点解析

### 概述

ms-swift 是 ModelScope 生态里的大模型训练、微调、推理和评测工具链，常用于 SFT、LoRA、QLoRA、推理测试和模型导出。

### 它的定位

ms-swift 不是像 vLLM 那样专注高吞吐在线 serving 的纯推理引擎。

它更像一个大模型研发工具箱：

- 支持加载多种开源模型。
- 支持 SFT、LoRA、QLoRA 等微调。
- 支持数据集处理。
- 支持训练配置管理。
- 支持简单推理和评测。
- 支持模型导出和部署衔接。

### 常见使用场景

#### 指令微调

用已有 base model 和指令数据训练一个更适合任务的模型。

```text
base model
  + SFT data
  -> fine-tuned model
```

#### LoRA / QLoRA 微调

冻结大部分原模型参数，只训练少量低秩适配参数，降低显存成本。

```text
base model frozen
  + trainable LoRA adapter
  -> adapter checkpoint
```

#### 推理验证

训练后用少量样本检查模型是否能正常回答。

#### 评测

在某些 benchmark 或自定义数据上跑模型效果。

### ms-swift 和 vLLM 的区别

| 维度 | ms-swift | vLLM |
|---|---|---|
| 主要定位 | 训练/微调/评测工具链 | 高吞吐推理服务框架 |
| 是否训练 | 支持 | 通常不负责训练 |
| 是否推理 | 支持 | 强项 |
| 是否 serving | 可衔接部署 | 原生面向服务 |
| 核心优势 | 训练流程和模型适配 | KV cache 管理和并发吞吐 |

### 为什么项目会从 ms-swift 迁移到 xLLM / vLLM 类服务

常见原因：

- ms-swift 更适合研发和微调，不一定适合大规模线上推理。
- 线上服务需要更稳定的并发、限流、监控和服务治理。
- 推理框架需要更强的 batching、KV cache 管理和多卡调度。
- 平台可能统一使用某种 serving 方案来降低运维成本。

### 常见误区

- 把 ms-swift 当成纯推理框架。
- 用训练脚本直接承担高并发线上服务。
- 推理参数没有和训练/评测时对齐，导致效果差异。
- LoRA adapter、base model、tokenizer 版本不一致。

### 安装与环境配置

ms-swift 依赖 Python、PyTorch、CUDA 和 ModelScope / Transformers 生态。安装前先确认 GPU 和 PyTorch 是否正常：

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

推荐单独建环境：

```bash
conda create -n swift python=3.10 -y
conda activate swift
pip install -U pip
pip install -U ms-swift
```

如果需要训练多模态模型、使用 flash attention、deepspeed 或特定量化后端，通常还要按项目脚本额外安装对应依赖。安装后可以检查命令是否可用：

```bash
swift --help
```

常见环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NPROC_PER_NODE=4
```

其中 `CUDA_VISIBLE_DEVICES` 控制用哪些 GPU，`NPROC_PER_NODE` 通常对应单机启动多少个训练进程。

### 最小 LoRA SFT 示例

一个最小的 LoRA SFT 命令大致长这样：

```bash
swift sft \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset ./train.jsonl \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --max_length 4096 \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --save_steps 500 \
  --eval_steps 500 \
  --output_dir ./output/qwen_lora_sft
```

参数怎么理解：

- `--model`：base model 路径或模型名。
- `--dataset`：训练数据路径。
- `--train_type lora`：使用 LoRA，而不是全参数微调。
- `--torch_dtype bfloat16`：使用 BF16 训练，通常比 FP16 稳定。
- `--gradient_accumulation_steps`：用小 micro-batch 模拟更大的 effective batch size。
- `--learning_rate`：LoRA 常用 `5e-5` 到 `2e-4`，全参通常更小。
- `--max_length`：输入加输出的最大 token 长度，太小会截断，太大会占显存。
- `--lora_rank` / `--lora_alpha`：控制 LoRA adapter 的容量和缩放。
- `--target_modules`：LoRA 插到哪些线性层。

有效 batch size 要按这个公式算：

```text
effective_batch_size =
per_device_train_batch_size * GPU 数 * gradient_accumulation_steps
```

### 训练数据格式示例

SFT 数据通常是 jsonl，每行一条样本。具体字段要以当前 ms-swift 版本和项目脚本为准，一个常见写法是：

```json
{"messages": [{"role": "user", "content": "解释一下 LoRA 是什么"}, {"role": "assistant", "content": "LoRA 是一种参数高效微调方法..."}]}
```

如果是多模态任务，还要额外提供图像或视频字段，并保证数据预处理、抽帧、token 上限和评测时一致。

### 推理验证示例

训练完成后，可以先用少量样本做推理 smoke test：

```bash
swift infer \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapters ./output/qwen_lora_sft/checkpoint-500 \
  --temperature 0 \
  --max_new_tokens 256
```

如果已经把 LoRA merge 成完整模型，则推理时直接加载 merge 后模型：

```bash
swift infer \
  --model ./output/qwen_lora_sft/merged \
  --temperature 0 \
  --max_new_tokens 256
```

### LoRA 合并与导出

LoRA 训练产物有两种部署方式：

```text
方式一：base model + adapter
方式二：adapter merge 回 base model，导出完整模型
```

如果后续要放到 vLLM 这类推理服务里，常见做法是先合并导出：

```bash
swift export \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapters ./output/qwen_lora_sft/checkpoint-500 \
  --merge_lora true \
  --output_dir ./output/qwen_lora_sft/merged
```

导出后要重点检查：

- base model 和 adapter 是否匹配。
- tokenizer 和 chat template 是否一致。
- 推理精度、量化方式、max length 是否和评测一致。
- 迁移到 vLLM 后输出是否和 swift infer 基本一致。

### 常见配置怎么调

| 现象 | 优先检查/调整 |
| --- | --- |
| 训练 OOM | 降低 `per_device_train_batch_size`、降低 `max_length`、开启 gradient checkpointing、增大梯度累积 |
| loss 一开始 spike / NaN | 降低 `learning_rate`、增加 `warmup_ratio`、使用 BF16、检查异常样本 |
| loss 基本不降 | 检查数据格式和 label mask、提高 LR、确认 LoRA target modules 生效 |
| train loss 降但 eval 不好 | 减少 epoch、降低 LoRA rank、增加 dropout、清洗数据 |
| 推理格式不稳定 | 统一训练数据模板、固定 temperature/top_p/max_new_tokens、检查 chat template |
| swift 推理和 vLLM 结果不一致 | 对齐 tokenizer、chat template、采样参数、max length、LoRA merge 方式 |

## 面试应对

### ms-swift 主要用于什么？

回答思路：定位为训练/微调/对齐工具链，点明它不是线上 serving 框架。

回答模板：

ms-swift 主要用于大模型训练、微调、对齐和实验管理，例如 SFT、LoRA、DPO 等。它更偏训练工具链，不是专门的线上 serving 框架。

### ms-swift 和 vLLM 的定位有什么区别？

回答思路：一句话点明"训练工具链 vs 推理服务框架"，再从是否训练、推理强项对比。

回答模板：

两者定位完全不同。ms-swift 是 ModelScope 生态的训练/微调/评测工具链，强项是 SFT、LoRA/QLoRA 这类训练流程和多模型适配，也能做简单推理和评测，但不是为高并发线上服务设计的。vLLM 是高吞吐推理服务框架，本身通常不负责训练，强项是 PagedAttention 做 KV cache 管理和 continuous batching 做并发调度。所以实际项目里常见的组合是：用 ms-swift 微调出模型，再导出到 vLLM 上做线上 serving。

### LoRA 微调产物如何用于推理？

回答思路：讲清两种部署方式，并强调版本对齐。

回答模板：

有两种方式：一是推理时同时加载 base model 和 LoRA adapter，运行时把低秩增量叠加上去，好处是可以按需切换不同 adapter；二是先把 adapter 权重 merge 回 base model 变成一个完整模型再部署，好处是推理时没有额外开销、方便对接 vLLM 这类引擎。不管哪种方式，都必须保证 base model、tokenizer 和 adapter 的配置版本严格匹配，否则会出现输出异常或效果掉点。

### 为什么训练框架不一定适合线上 serving？

回答思路：对比两者的优化目标——训练关注梯度和吞吐，serving 关注延迟和稳定。

回答模板：

训练框架关注梯度、优化器、数据加载和 checkpoint；线上 serving 关注并发、延迟、batching、资源隔离、灰度和监控。二者优化目标不同。

### 模型从训练工具链迁移到推理服务时要注意什么？

回答思路：列出权重格式、tokenizer、chat template、精度/量化、采样参数等易踩坑点，最后用回归集验证。

回答模板：

要检查权重格式、base model、tokenizer、chat template、精度、量化、max length、采样参数和接口协议，并用回归集验证迁移前后输出。
