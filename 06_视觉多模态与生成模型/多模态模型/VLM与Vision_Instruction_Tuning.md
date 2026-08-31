# VLM 与 Vision Instruction Tuning

## 知识点解析

### 概述

VLM（Vision-Language Model，视觉语言模型）是能同时处理图像/视频和文本的多模态模型；Vision Instruction Tuning（视觉指令微调）是用图文指令数据对 VLM 做监督微调，让它从“会对齐图文”升级到“会按用户指令完成视觉任务”，如视觉问答、图像描述、OCR、目标定位和多模态推理。

两者的关系类似 LLM 里的“预训练 → SFT”：VLM 的预训练解决模态对齐，Vision Instruction Tuning 解决指令跟随。

### 主流 VLM 的三段式结构

现在主流的 VLM（LLaVA、Qwen-VL、InternVL 等）几乎都遵循同一个范式：

1. **视觉编码器（Vision Encoder）**：通常是 CLIP ViT 或 SigLIP，把图像编码成一串视觉 token（patch embedding）。用预训练好的对比学习视觉塔，是因为它的视觉表示已经和语言对齐得比较好。
2. **连接器（Projector / Connector）**：把视觉特征投影到 LLM 的词嵌入空间。最简单的是一个 MLP（LLaVA），复杂一些的用 Q-Former（BLIP-2）或 cross-attention（Flamingo）压缩视觉 token 数量。
3. **大语言模型（LLM Backbone）**：把投影后的视觉 token 当作“软 prompt”拼在文本 token 前面，用自回归方式生成回答。

推理时输入是 `[视觉 token][文本 token]`，输出是文本。视觉 token 本质上被当成了一种“外语词”喂进 LLM。

### 为什么需要 Vision Instruction Tuning

VLM 预训练阶段（图文对比 / 图像描述）只让模型学会“图文对齐”和“看图说话”，但不会主动按指令回答问题——你问它“图里有几只猫”，它可能只输出一句泛泛的 caption。Vision Instruction Tuning 用大量 `(图像, 指令, 期望回答)` 三元组做 SFT，教会模型：

- 按不同指令切换任务（问答 / 描述 / OCR / 定位 / 推理）。
- 输出符合要求的格式（JSON、bounding box、步骤化推理）。
- 拒绝图中没有依据的问题，减少幻觉。

LLaVA 的关键贡献之一就是用 GPT-4 把图像的标注（caption + bbox）反向合成成多轮对话/推理指令数据，低成本造出了指令微调集。

### 典型训练流程（以 LLaVA 为例）

```text
阶段一：特征对齐预训练
  冻结 Vision Encoder 和 LLM，只训 Projector
  用图文对让视觉特征对齐到 LLM 词空间

阶段二：视觉指令微调
  冻结（或部分解冻）Vision Encoder，训练 Projector + LLM
  用图文指令数据教模型跟随指令
```

分两阶段的核心考量：先用便宜的对齐阶段把连接器训好，避免一开始就用指令数据同时动 LLM 导致训练不稳、破坏 LLM 已有能力。

### 优势与局限

优势：复用强大的预训练视觉塔和 LLM，训练成本低（尤其只训 Projector 的阶段）、指令泛化好、能快速迁移到新任务。

局限：

- 视觉分辨率受限，细粒度 OCR、密集小目标、复杂图表容易出错（后续 AnyRes / 动态分辨率切图就是为解决这个）。
- 视觉 token 占用大量上下文，多图或视频场景成本高。
- 幻觉：会说出图中不存在的对象或属性，尤其当指令数据分布偏了。
- 评测困难：生成式回答难用单一指标衡量，常需 LLM-as-judge 或专门 benchmark（MMBench、MMMU、MME 等）。

## 面试应对

### VLM 的典型结构是什么？各部分为什么这么设计？

回答思路：按“视觉编码器 → 连接器 → LLM”三段式讲，并解释每一段为什么用现成的预训练模块。

回答模板：

主流 VLM 是三段式结构：视觉编码器、连接器、LLM。视觉编码器一般用 CLIP 或 SigLIP 的 ViT，把图像切成 patch 编码成视觉 token，用它是因为对比学习出来的视觉表示已经和语言对齐得比较好。连接器把视觉特征投影到 LLM 的词嵌入空间，最简单是一个 MLP，也可以用 Q-Former 或 cross-attention 来压缩 token 数。LLM backbone 把投影后的视觉 token 当作前缀 prompt，用自回归生成回答。这样设计的好处是最大限度复用现成的强视觉塔和强 LLM，训练时往往只需要训中间的连接器，成本很低。

### 为什么已经有 VLM 预训练，还需要 Vision Instruction Tuning？

回答思路：类比 LLM 的“预训练 → SFT”，说清预训练学到的是什么、指令微调补的是什么。

回答模板：

VLM 预训练主要学图文对齐和看图说话，但模型不会主动按指令切换任务，你问“图里有几只猫”，它可能只回一句泛泛描述。Vision Instruction Tuning 相当于多模态的 SFT，用大量“图像 + 指令 + 期望回答”的数据，教模型按不同指令完成问答、OCR、定位、推理，并输出符合格式要求的结果。它解决的是指令跟随和格式对齐，而不是重新学视觉能力。LLaVA 的做法就是用 GPT-4 把图像的 caption 和 bbox 反向合成成对话和推理指令，低成本造出了指令数据。

### LLaVA 为什么要分两阶段训练？

回答思路：先说两阶段各训什么、冻结什么，再解释为什么不能一步到位。

回答模板：

第一阶段是特征对齐，冻结视觉编码器和 LLM，只训中间的 Projector，用图文对把视觉特征对齐到 LLM 的词空间。第二阶段是指令微调，解冻 LLM，用图文指令数据训 Projector 加 LLM。分两阶段是因为如果一上来就用指令数据同时动 LLM，连接器还没对齐好，训练不稳，还容易破坏 LLM 原有的语言能力。先用便宜的对齐阶段把桥搭好，再做指令微调，稳定性和效果都更好。

### VLM 常见的失败模式有哪些？怎么缓解？

回答思路：从分辨率、幻觉、上下文成本、评测四个角度讲，每个都给缓解手段。

回答模板：

最常见的是分辨率不足导致的细粒度识别错误，比如小字 OCR、密集小目标、复杂图表，缓解办法是用动态分辨率或 AnyRes 把大图切块分别编码。第二是幻觉，会说出图里没有的对象或属性，通常和指令数据分布、正负样本构造有关，可以引入 grounding 监督、要求模型给视觉证据，评测时专门看幻觉 benchmark。第三是视觉 token 占用上下文太多，多图和视频场景成本高，可以用 Q-Former 或 token 压缩。最后是评测难，生成式回答不好用单一指标，我会结合 MMBench、MMMU 这类选择题 benchmark 和 LLM-as-judge，同时重点看 bad case。
