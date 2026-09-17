# Qwen3-VL 输入处理逻辑：文本、图像与视频

## 知识点解析

### 概述

Qwen3-VL 不是把图像或视频先转换成一段人工生成的 caption，再交给文本大模型，而是把不同模态统一组织成一个可以被 Qwen3 decoder 处理的多模态序列。文本先经过 tokenizer；图像和视频先经过 processor 转换成动态数量的视觉 token；文本中的视觉占位符再按照真实视觉 token 数量展开，最后由 Vision Encoder、MLP Vision-Language Merger 和 Qwen3 LLM 共同完成理解与生成。[1, 3, 6, 7]

这里的“不是先 caption 再输入”是根据论文三模块架构和官方实现对处理链路的归纳，不是论文中的逐字表述。

三种输入的核心差异是：

```text
纯文本：
  文本 -> tokenizer -> text token embedding -> Qwen3 decoder

图像：
  图片 -> 动态 resize/归一化 -> 2D patch -> ViT -> 2x2 spatial merge
       -> visual embedding -> 替换 image placeholder -> Qwen3 decoder

视频：
  视频 -> 抽帧 -> 动态 resize/归一化 -> 3D temporal-spatial patch
       -> video visual embedding
       + 每个 temporal patch 前的文本时间戳
       -> 交错的 timestamp + visual block -> Qwen3 decoder
```

最容易记错的一点是：论文严格说的是**每个视频 temporal patch 前面加一个文本时间戳**，不是无条件地给每一个原始视频帧各加一个 timestamp。当前 Transformers 实现默认 `temporal_patch_size=2`，通常将连续两帧组成一个 temporal patch，并用这两帧时间的平均值构造一个时间戳。因此，时间戳数量通常等于时间 patch 数，而不是原始帧数。[1, 5, 6]

Qwen3-VL 的三项关键设计分别解决不同问题：

| 机制 | 处理位置 | 主要作用 |
| --- | --- | --- |
| Native Dynamic Resolution | processor / Vision Encoder 输入侧 | 按图片或视频的像素预算动态调整分辨率，保留文字和局部细节 |
| Interleaved-MRoPE | Qwen3 LLM 的位置编码 | 用交错方式编码文本位置以及视觉的时间、高度、宽度结构 |
| Text-Timestamp Alignment | 视频进入语言序列时 | 用模型熟悉的文本 token 直接表达采样片段的绝对时间 |
| DeepStack | ViT 到 LLM 的融合 | 把 ViT 多个中间层的视觉特征注入 LLM 早期层，保留低层细节和高层语义 |


上表是对论文和官方实现的功能归纳，不是四个模块都来自同一个代码文件。[1, 2, 5, 6, 7]

### 1. 先建立完整处理链路

Qwen3-VL 的输入处理可以拆成六个阶段。[1, 3, 6, 7]

```text
用户消息
  -> chat template：把多轮消息和模态类型转换成带特殊 token 的文本
  -> processor：分别处理文本、图片、视频
  -> tokenizer：把普通文本、时间戳和视觉占位符转成 input_ids
  -> Vision Encoder：把图片/视频像素转成视觉特征
  -> Merger：空间压缩并投影到 Qwen3 hidden size
  -> embedding 替换和 DeepStack 注入
  -> Qwen3 decoder：统一做 attention 和自回归生成
```

这里有两个输入必须保持一一对应。[3, 6]

```text
文本中的模态占位符：
  <|vision_start|><|image_pad|><|vision_end|>
  或
  <|vision_start|><|video_pad|><|vision_end|>

真正的媒体数据：
  images / videos
```

占位符告诉模型“视觉特征应该出现在序列的哪个位置”；`images` 或 `videos` 则告诉 processor “从哪里读取像素”。只有两者都存在，模型才能把视觉 embedding 放到正确的上下文位置。[6, 7]

在当前 Transformers 实现中，主要的多模态输入字段是：

| 字段 | 含义 |
| --- | --- |
| `input_ids` | tokenizer 产生的文本、特殊 token、视觉占位 token |
| `attention_mask` | padding 和有效 token 的掩码 |
| `mm_token_type_ids` | 文本、图像、视频的类型标识，通常分别为 `0/1/2` |
| `pixel_values` | 图像经过 resize、归一化并整理成视觉 patch 后的输入张量 |
| `pixel_values_videos` | 视频经过抽帧、resize、归一化和时空 patch 化后的输入张量 |
| `image_grid_thw` | 每张图像的视觉网格，逻辑上为 `[T, H, W]`，其中 `T=1` |
| `video_grid_thw` | 每个视频的时空网格 `[T, H, W]` |

字段名称和返回内容来自 Qwen3-VL processor 的官方文档与实现；其中 `mm_token_type_ids` 的取值约定来自模型源码的 `get_rope_index()` 文档。[4, 6, 7]

### 2. 文本输入是怎么处理的

#### 2.1 纯文本输入

当输入只有文本时，Qwen3-VL 的行为和 Qwen3 文本模型基本一致。[1, 4, 7]

```text
字符串
  -> Qwen tokenizer
  -> input_ids
  -> token embedding
  -> Qwen3 decoder blocks
  -> final RMSNorm
  -> LM head
  -> next-token logits
```

模型不会调用 Vision Encoder，也不会产生 `pixel_values`、`image_grid_thw` 或 `video_grid_thw`。每个文本 token 通过 embedding table 映射到 Qwen3 的 hidden space，然后经过 decoder-only Transformer 做 causal self-attention。[4, 7]

纯文本输入的训练或推理目标仍然是自回归语言建模。下面的概率表达式是 decoder-only 语言模型的通用形式，不是 Qwen3-VL 论文中的专有公式。

```text
P(x_1, x_2, ..., x_n) = product_i P(x_i | x_<i)
```

因此，Qwen3-VL 的“VL”并不意味着每次都必须有视觉输入。没有图像和视频时，它仍然可以作为 Qwen3 语言模型使用。[4, 7]

#### 2.2 文本和视觉输入混合时

多模态场景下，文本不再是孤立的 prompt，而是负责：[1, 3]

1. 指定图片或视频在对话中的位置。
2. 提出问题和任务约束。
3. 为视觉特征提供语言上下文。
4. 在视频中承载文本时间戳。
5. 约束最终输出格式，例如 JSON、坐标或时间区间。

例如：

```text
请阅读这张图片中的价格，并返回 JSON。
```

经过 chat template 后，图片位置可能变成：

```text
<|im_start|>user
请阅读这张图片中的价格：
<|vision_start|><|image_pad|><|vision_end|>
并返回 JSON。
<|im_end|>
```

`<|image_pad|>` 只是一个占位符。processor 读取图片、计算视觉 token 数后，会把它扩展成与真实视觉 embedding 数量相同的视觉占位 token。模型前向时，视觉 embedding 会通过 `masked_scatter` 写入这些位置；它们不再使用普通文本 token embedding。[6, 7]

#### 2.3 时间戳本身也是文本

Qwen3-VL 的时间戳不是额外设计的一套浮点数输入，也不是把秒数直接写进某个 RoPE position id。论文将它定义为 textual token-based time encoding。工程实现中，时间戳首先被格式化成字符串，例如：[1, 6]

```text
<3.0 seconds>
<00:01:12>
```

然后和普通文字一样经过 tokenizer，成为语言序列中的 text token。论文训练阶段使用 seconds 和 HMS 两种格式，帮助模型理解不同的时间码表达方式。需要区分实现版本：当前 Hugging Face processor 的默认构造逻辑通常使用 `f"<{curr_time:.1f} seconds>"`，论文中的 HMS 支持主要体现为训练数据和模型能力，并不意味着每个 processor 默认都会自动输出 HMS。[1, 6]

“模型可以直接处理数字、单位、顺序和语言描述”是对 textual token-based 设计的解释，不是论文给出的独立实验证明。

### 3. 图像输入是怎么处理的

#### 3.1 图像处理总览

一张图像的典型处理链路是：

```text
本地文件 / URL / base64 / PIL Image
  -> 读取并转 RGB
  -> 按像素预算保持宽高比 resize
  -> 将高宽对齐到 patch_size * spatial_merge_size 的倍数
  -> rescale + normalize
  -> 2D patch embedding
  -> Vision Transformer
  -> 2x2 spatial merge
  -> MLP 投影到 Qwen3 hidden size
  -> 写入 image placeholder 对应的位置
```

Qwen3-VL 使用 Native Dynamic Resolution，因此它不是把所有图片强行压成固定的 `224x224` 或 `448x448`。processor 会根据：[1, 2, 3, 4, 9]

- `min_pixels`：最低像素预算，避免小字和局部细节被压没。
- `max_pixels`：单张图片的最大像素预算，控制视觉 token 数。
- `resized_height` / `resized_width`：用户显式指定的尺寸。
- patch 和 merge 的对齐要求。

进行动态缩放。通常会保持原始宽高比，再把结果取整到模型需要的网格倍数。[3, 4, 9]

#### 3.2 图像的 patch 化

Qwen3-VL 的典型视觉配置如下。以下是当前 Transformers 配置示例，不应当无条件泛化到所有 Qwen3-VL checkpoint。[8]

| 配置 | 常见值 | 作用 |
| --- | --- | --- |
| `patch_size` | `16` | 每个视觉 patch 覆盖 `16x16` 像素 |
| `spatial_merge_size` | `2` | 相邻 `2x2` patch 合并成一个语言侧视觉 token |
| Vision hidden size | `1152` 左右 | ViT 内部视觉特征维度，具体随变体变化 |
| Vision Transformer depth | `27` 层左右 | 逐层提取视觉特征 |
| Vision attention heads | `16` 左右 | 视觉 patch 之间做多头注意力 |

假设 resize 后图像尺寸为 `H' x W'`，且 `H'` 和 `W'` 都是 32 的倍数，那么：

```text
patch 网格：
  H_patch = H' / 16
  W_patch = W' / 16

Merger 后视觉 token 网格：
  H_token = H' / 32
  W_token = W' / 32

图像视觉 token 数：
  N_image = (H' / 32) * (W' / 32)
```

例如，`672x448` 的图像经过尺寸对齐后，视觉 token 数大致为：

```text
(672 / 32) * (448 / 32) = 21 * 14 = 294
```

这里的 294 是由 patch 和 merge 配置推导出的视觉 token 数，不包括 `<|vision_start|>`、`<|vision_end|>` 和问题文本本身的 token。[3, 6, 8, 9]

#### 3.3 为什么图像也会出现 `image_grid_thw`

Qwen3-VL 统一用三维网格描述视觉输入：[4, 7, 9]

```text
image_grid_thw = [1, H_patch, W_patch]
```

图像没有真正的时间序列，所以逻辑时间维为 `1`。[4, 7] 在当前兼容的图像处理实现中，为了适配 `temporal_patch_size=2` 的 3D patch 接口，单张图像可能在内部复制最后一帧或补齐 temporal patch；这不代表图片被当成了视频，也不产生视频时间戳。[9]

`image_grid_thw` 的作用是告诉模型：[4, 7]

- 视觉 token 有多少个。
- token 原本对应多大的二维网格。
- LLM 侧的高度和宽度位置如何生成。
- 视觉 embedding 应该如何和文本中的 image placeholder 对齐。

#### 3.4 图像在语言序列中的形式

单张图像进入 chat template 后，逻辑上是：[3, 6]

```text
文本 token
  + <|vision_start|>
  + <|image_pad|> * N_image
  + <|vision_end|>
  + 文本 token
```

前向时：[7]

```text
image pixels
  -> Vision Encoder
  -> image visual features
  -> MLP Merger
  -> image embeddings: [N_image, hidden_size]

input_ids 中的 image_pad 位置
  -> 被 image embeddings 替换
```

`vision_start` 和 `vision_end` 主要是视觉片段边界标记；真正承载图像内容的是和 `N_image` 一一对应的视觉 embedding。

#### 3.5 多图输入

多张图片不会先被拼成一张大图。每一张图片独立完成：[3, 6]

```text
resize -> patchify -> Vision Encoder -> Merger
```

然后按照它们在消息中的位置插入同一个上下文序列：

```text
图片 1 -> image block 1
文本问题
图片 2 -> image block 2
文本问题
```

这样模型可以通过同一个 Qwen3 decoder 进行图片比较、差异分析、跨图引用和多轮追问。前半句是官方多图使用场景，后半句是 processor 分支逻辑的归纳。[3, 6, 7]

### 4. 视频输入是怎么处理的

#### 4.1 视频处理总览

Qwen3-VL 不会把一个 mp4 文件直接作为一个不可分解的 token 输入。典型视频链路是：[1, 3, 5, 6, 7]

```text
视频 URL / 本地视频 / 已采样帧列表
  -> 视频解码
  -> 按 fps 或 num_frames 采样
  -> 根据总像素预算调整空间分辨率
  -> RGB、rescale、normalize
  -> 按 temporal_patch_size 和 patch_size 做 3D patch
  -> 2x2 空间 merge
  -> 为每个 temporal patch 计算文本时间戳
  -> 组成 timestamp + video visual block 的交错序列
  -> Vision Encoder
  -> Merger
  -> 替换 video_pad
  -> Qwen3 decoder
```

视频内容本身最终仍然变成视觉 embedding；视频的“绝对时间”则以额外的文本 timestamp 进入语言序列。[1, 6, 7]

#### 4.2 第一步：抽帧

视频预处理通常支持两种互斥控制方式：[3, 5]

| 方式 | 含义 |
| --- | --- |
| `fps` | 按目标每秒帧数均匀抽帧 |
| `num_frames` | 在整个视频时间范围内均匀抽取固定数量的帧 |

当前 Transformers 的 Qwen3-VL video processor 中，默认配置为以下示例。它们是版本相关的工程默认值，不是论文固定协议。[5]

```text
fps = 2
min_frames = 4
max_frames = 768
do_sample_frames = True
```

不同 checkpoint、推理框架和业务代码可能覆盖这些值。`fps` 与 `num_frames` 在该实现中不能同时设置。[5]

用原视频 FPS 为 `F_video`、总帧数为 `N`、目标采样率为 `r` 时，抽帧数大致为。该式是对 `sample_frames()` 实现的简化表达：[5]

```text
N_sample = clamp(floor(N / F_video * r), min_frames, max_frames)
```

当前实现使用 `np.linspace()` 生成均匀采样的 frame indices。[5] 真正做时间对齐时，必须保留。后四项是关键帧业务的工程要求：[10, 11]

```text
原始视频 FPS
抽取的 frame indices
视频 duration
采样后的帧顺序
```

只保存“第几张抽样图片”而丢掉原始 FPS，会导致模型看到的 timestamp 和 GT 时间不在同一个时间坐标系。[5, 6, 10, 11]

#### 4.3 第二步：视频动态 resize

视频的动态 resize 和图片有一个重要差别：视频通常需要控制的是**所有采样帧的总像素预算**，而不是单帧像素预算。[3, 5]

如果采样后有 `T_frame` 帧，原始空间尺寸为 `H x W`，processor 会综合：

```text
T_frame * H' * W'
```

以及 `min_pixels`、`max_pixels` 或 `total_pixels` 来决定 `H'`、`W'`。当前实现会将空间尺寸对齐到：

```text
patch_size * spatial_merge_size = 16 * 2 = 32
```

因此视频可能在保持宽高比的基础上被缩小或放大，但所有帧通常共享同一个 resize 后的空间尺寸，才能组成规则的时空网格。[5]

#### 4.4 第三步：3D patch embedding

视频的视觉 patch 同时覆盖时间和空间：[1, 5, 8]

```text
temporal_patch_size = 2
patch_size = 16

一个 3D patch：
  连续 2 帧
  每帧一个 16x16 的空间区域
```

设采样后帧数为 `T_frame`，resize 后图像大小为 `H' x W'`，则。下面的网格关系是由 processor 和配置推导出的：[5, 8]

```text
时间 patch 网格：
  T_patch = ceil(T_frame / temporal_patch_size)

空间 patch 网格：
  H_patch = H' / 16
  W_patch = W' / 16

video_grid_thw：
  [T_patch, H_patch, W_patch]
```

如果采样帧数不是 `temporal_patch_size` 的倍数，当前 processor 会复制最后一帧补齐，再做 temporal patch 化。[5] 补齐帧只为满足张量形状，不应被误认为原视频中真的多了一帧。

随后每个空间 `2x2` patch 经过 Merger 合成一个语言侧视觉 token：[1, 5, 6, 8]

```text
视频视觉 token 数：
  N_video_visual
    = T_patch * (H_patch / 2) * (W_patch / 2)
    = ceil(T_frame / 2) * (H' / 32) * (W' / 32)
```

这个公式只计算视觉 `video_pad` 数量，不包含 timestamp 文本 token 和视觉边界 token。[5, 6, 8]

#### 4.5 第四步：构造视频时间戳

论文的描述是：**每个 video temporal patch 前面加一个 formatted text timestamp**。例如：[1]

```text
<3.0 seconds>
```

当前 Transformers 实现的具体逻辑是：[6]

1. 根据采样帧的原始 indices 和原视频 FPS 计算每个采样帧时间：

   ```text
   timestamp(frame_i) = frame_index_i / original_video_fps
   ```

2. 按 `temporal_patch_size` 对相邻采样帧分组。
3. 对一个 temporal patch 内的第一帧和最后一帧时间取平均。
4. 将平均时间格式化为例如 `<3.0 seconds>`。
5. 把这个字符串放在对应 visual block 之前。

因此，默认 `temporal_patch_size=2` 时：[5, 6]

```text
原始帧 0 + 原始帧 1 -> 一个 temporal patch -> 一个 timestamp
原始帧 2 + 原始帧 3 -> 一个 temporal patch -> 一个 timestamp
```

这就是“不是严格每个原始帧一个 timestamp”的原因。论文中的“每个 temporal patch”比“每一帧”更准确。[1, 5, 6]

#### 4.6 第五步：形成交错输入

假设视频被处理成 3 个 temporal patch，每个 temporal patch 经过空间 merge 后有 `M` 个 visual token，那么视频在语言序列中近似为。[2, 6]

```text
<t_1 seconds>
<|vision_start|>
<|video_pad|> * M
<|vision_end|>

<t_2 seconds>
<|vision_start|>
<|video_pad|> * M
<|vision_end|>

<t_3 seconds>
<|vision_start|>
<|video_pad|> * M
<|vision_end|>
```

完整问题则可能是：

```text
请找出用户第一次完成支付的时间。
<t_1 seconds><vision_start><video_pad>...<vision_end>
<t_2 seconds><vision_start><video_pad>...<vision_end>
<t_3 seconds><vision_start><video_pad>...<vision_end>
```

注意这不是把 timestamp 画到视频图片里，也不是把时间放在一个模型外部的 metadata 字段里不让 LLM 看到。它是 tokenizer 能读懂的文本 token，并且和对应的视觉 block 在序列中相邻排列。[1, 6]

#### 4.7 第六步：视觉编码和语言序列融合

视频视觉部分的前向可以简化为：[1, 5, 7]

```text
pixel_values_videos
  -> 3D patch embedding
  -> 27 层左右的 SigLIP-2-based ViT
  -> final visual hidden states
  -> 2x2 spatial merge + MLP
  -> video embeddings
```

语言侧则是：[6, 7]

```text
timestamp text tokens -> 普通 token embedding
video_pad positions   -> video embeddings
vision_start/end      -> 视觉片段边界 token
```

最终同一段序列里同时存在：

```text
普通文本 embedding
时间戳文本 embedding
图像/视频 visual embedding
```

Qwen3 decoder 对它们进行统一的 causal attention，最后用同一个 language model head 生成文本、时间、坐标、JSON 或工具调用。[1, 2, 7]

### 5. 时间戳、MRoPE 与视频时间的关系

这是理解 Qwen3-VL 视频输入的核心。下面三层是对采样实现、论文机制和 position id 实现的归纳。[1, 5, 6, 7]

| 层次 | 具体机制 | 表达什么 |
| --- | --- | --- |
| 媒体采样层 | frame indices / original FPS | 采样帧在原视频中的真实秒数 |
| 文本时间层 | `<3.0 seconds>`、`<00:01:12>` | 让 LLM 直接读到绝对时间 |
| 结构位置层 | Interleaved-MRoPE 的 T/H/W | token 在多模态序列和视觉网格中的结构位置 |

#### 5.1 Qwen2.5-VL 的问题

Qwen2.5-VL 主要依靠时间同步的 MRoPE，把绝对时间反映到 temporal position id 中。论文指出，这条路线存在两个问题。[1]

1. 长视频会产生很大且稀疏的 temporal position id，长程建模不稳定。
2. 为了让模型适应不同 FPS，需要构造大量且均匀覆盖多帧率的数据，训练数据成本高。

#### 5.2 Qwen3-VL 的改进

Qwen3-VL 不再把原始绝对秒数直接塞进 temporal position id，而是：[1, 7]

```text
绝对时间：
  用文本 timestamp token 表达

视觉/序列结构位置：
  用 Interleaved-MRoPE 表达
```

所以，timestamp 和 MRoPE 不是互相替代后只保留一个，而是分工协作。这是对论文设计和源码位置处理的综合解释：[1, 6, 7]

- timestamp 负责告诉模型“这是原视频的第几秒”。[1, 6]
- MRoPE 负责告诉模型“这个 token 位于哪一个视觉时间组、哪一行、哪一列以及多模态序列的什么结构位置”。[1, 7]

#### 5.3 Interleaved-MRoPE 怎么做

普通文本 RoPE 主要处理一个一维位置 `p`。视觉输入需要同时表达：[1]

```text
t：时间位置
h：图像高度方向位置
w：图像宽度方向位置
```

Qwen2-VL 系列将 RoPE 扩展成 MRoPE。旧式做法把 head dimension 分成连续区块：[1]

```text
[ t t t ... | h h h ... | w w w ... ]
```

这样不同轴可能只占据某一段频率范围，导致频率分布不均衡。Qwen3-VL 将三条轴交错分布：[1, 7]

```text
[ t h w | t h w | t h w | ... ]
```

这样 `t/h/w` 都能覆盖低频到高频：[1]

- 低频更适合表达长距离、长视频的变化。
- 高频更适合表达局部位置和细粒度变化。

在当前实现中，LLM position ids 的形状可以理解为：[7]

```text
[3, batch_size, sequence_length]
```

三行分别对应视觉位置的 `t/h/w` 轴。纯文本 token 通常在三个轴上使用相同的一维文本位置；视觉 token 根据 `grid_thw` 获得时间和二维空间位置。随后交错的 MRoPE 将三组频率写入 attention 的 query/key。[7]

#### 5.4 Vision Encoder 内部的二维位置和 LLM MRoPE 不要混为一谈

Qwen3-VL 至少有两类容易被混淆的位置机制：[1, 7]

```text
Vision Encoder：
  2D-RoPE + 按输入尺寸插值的绝对位置 embedding

Qwen3 LLM：
  Interleaved-MRoPE，处理多模态序列的 T/H/W
```

前者帮助 ViT 在不同分辨率下理解视觉 patch 的二维布局；后者帮助 Qwen3 decoder 在统一序列中理解视觉 token 的时间和空间结构。视频绝对秒数则另外由文本 timestamp 提供。[1, 6, 7]

### 6. DeepStack 如何参与输入处理

如果只把 ViT 最后一层输出一次性投影给 LLM，视觉信息在进入语言模型前可能已经损失一部分低层细节。OCR、小字号、按钮边缘、局部 UI 状态等任务尤其依赖中间层表示。这是对多层特征融合动机和细粒度视觉任务的解释。[1]

Qwen3-VL 的 DeepStack 流程是：[1, 7, 8]

```text
ViT layer 8  -> 专用 merger -> visual feature 1 -> LLM 第一个早期层残差注入
ViT layer 16 -> 专用 merger -> visual feature 2 -> LLM 第二个早期层残差注入
ViT layer 24 -> 专用 merger -> visual feature 3 -> LLM 第三个早期层残差注入
```

公开 Transformers 配置示例中的 `deepstack_visual_indexes` 为 `[8, 16, 24]`；具体 checkpoint 仍应以自身 config 为准。[8]

在 LLM 的视觉 token 位置，DeepStack 做的事情可以近似写成。下面是源码 residual addition 的数学化表达：[7]

```text
hidden_states[visual_positions]
  = hidden_states[visual_positions]
  + deepstack_visual_embedding
```

它不是额外追加一串上下文 token，因此不会按照 DeepStack 的层数线性增加 context length；它是在对应视觉位置做 residual addition。[1, 7]

对视频而言，DeepStack 的作用同时覆盖。下面是由多层 ViT feature 的层级含义推导出的解释。[1, 7]

- 低层的局部纹理和文字细节。
- 中层的控件、物体和局部状态。
- 高层的语义、动作和事件线索。

### 7. 三种输入的对比

下表是根据 processor、modeling 和位置编码实现整理出的对比，不是论文中的原始表格。[1, 5, 6, 7]

| 对比项 | 文本 | 图像 | 视频 |
| --- | --- | --- | --- |
| 原始数据 | 字符串 | 图片像素 | 视频文件或帧序列 |
| 是否抽帧 | 否 | 否 | 是 |
| 是否动态 resize | 不适用 | 是 | 是，通常按总像素预算 |
| 视觉 patch | 无 | 2D patch | temporal-spatial 3D patch |
| 视觉网格 | 无 | `image_grid_thw=[1,H,W]` | `video_grid_thw=[T,H,W]` |
| timestamp | 无 | 无 | 有，通常每个 temporal patch 一个 |
| 视觉 token 数 | 0 | `H/32 * W/32` | `T * H/32 * W/32` |
| 位置结构 | 1D 文本位置 | H/W | T/H/W |
| 进入 LLM 的方式 | token embedding | image embedding 替换 image pad | timestamp text + video embedding 替换 video pad |
| 输出 | 文本 | 文本/坐标/JSON/工具调用 | 文本/时间段/坐标/JSON/工具调用 |

### 8. 一个完整的视频例子

下面是为了说明 token 数量而构造的算例，不是论文或官方 benchmark 的实际样本。[5, 8]

假设：

```text
原视频：
  原始 FPS = 30
  总帧数 = 120
  时长约 4 秒

采样：
  fps = 2
  得到 8 个采样帧

视觉配置：
  temporal_patch_size = 2
  patch_size = 16
  spatial_merge_size = 2
  resize 后尺寸 = 448 x 320
```

则。以下结果均由前述 patch、merge 和 temporal patch 公式计算得到：[5, 8]

```text
T_patch = 8 / 2 = 4
H_patch = 448 / 16 = 28
W_patch = 320 / 16 = 20

每个 temporal patch 的语言侧视觉 token：
  M = (28 / 2) * (20 / 2) = 14 * 10 = 140

整个视频的视觉 token：
  N_video_visual = 4 * 140 = 560
```

语言序列中大致会有：

```text
<t_1 seconds> <vision_start> <video_pad> * 140 <vision_end>
<t_2 seconds> <vision_start> <video_pad> * 140 <vision_end>
<t_3 seconds> <vision_start> <video_pad> * 140 <vision_end>
<t_4 seconds> <vision_start> <video_pad> * 140 <vision_end>
```

其中：[5, 6, 8]

- 8 个采样帧被两两合成 4 个 temporal patch。
- 4 个 temporal patch 对应 4 个 timestamp。
- 560 个 `video_pad` 最终对应 560 个视频 visual embedding。
- timestamp 字符串还会被 tokenizer 切成普通文本 token，因此真实输入长度大于 560。
- 问题、系统 prompt、视觉边界 token 和模型输出也会占用上下文。

这个例子也说明了两个工程事实。它们是 token 预算的直接推导，不是 Qwen3-VL 论文的独立实验结论：[5, 6, 8]

1. 提高 `fps` 会同时提高 timestamp 数量、temporal patch 数量和视觉 token 数。
2. 提高空间分辨率会提高每个 temporal patch 的视觉 token 数，但不会改变 timestamp 数量。

### 9. 论文机制与工程实现的对应关系

下表明确区分论文机制和当前工程实现。[1, 3, 5, 6, 7, 8]

| 论文概念 | 工程中的典型体现 |
| --- | --- |
| Text-Timestamp Alignment | processor 将 `<0.0 seconds>` 等字符串插入每个视频 temporal patch 前。 |
| Temporal patch | `temporal_patch_size=2`，相邻两帧组成一个时间 patch。 |
| Dynamic resolution | `min_pixels`、`max_pixels`、`total_pixels` 和尺寸对齐。 |
| Spatial compression | `patch_size=16`，`spatial_merge_size=2`，空间上约 32 倍压缩。 |
| Temporal compression | `temporal_patch_size=2`，时间上约 2 帧合成一个 patch。 |
| `video_grid_thw` | 记录 temporal patch、空间 patch 的 T/H/W 网格。 |
| Interleaved-MRoPE | 对 Qwen3 LLM 的 T/H/W position ids 重新交错分配频率。 |
| DeepStack | ViT 多层特征经过专用 merger 后注入 LLM 前几个 decoder 层。 |
| 统一自回归生成 | 所有回答最终由 Qwen3 LLM 的 LM head 生成。 |

### 10. 常见误区和能力边界

本节中的“正确说法”主要是对论文和当前实现的澄清；其中涉及业务风险的部分属于推导，不是官方能力保证。[1, 5, 6, 7]

#### 10.1 误区一：每个原始帧都一定有一个 timestamp

更准确的说法是每个 temporal patch 有一个 timestamp。当前默认 `temporal_patch_size=2` 时，一般两帧共享一个 timestamp，时间取组内首尾帧时间的平均值。不同实现如果修改 temporal patch 配置，timestamp 数量也会改变。[1, 5, 6]

#### 10.2 误区二：timestamp 是画在视频帧上的水印

不是。Qwen3-VL 的 timestamp 是插入语言序列的文本 token，不需要把时间数字渲染到每帧图片上。[1, 6] 把 timestamp 烧录到图片上是业务侧可以采用的额外手段，但不等于论文中的 Text-Timestamp Alignment。

#### 10.3 误区三：MRoPE 直接保存了绝对秒数

Qwen3-VL 用文本 token 表达绝对时间；Interleaved-MRoPE 主要表达多模态 token 的结构位置和 T/H/W 关系。[1, 6, 7] 把两者都说成“把秒数编码进 RoPE”会混淆 Qwen2.5-VL 和 Qwen3-VL。

#### 10.4 误区四：模型看到了视频的每一帧

模型看到的是 processor 采样后的帧或 temporal patch，不是原视频的所有原始帧。[5] 事件短、采样稀疏或 resize 过小，都可能让关键证据丢失。[10, 11]

#### 10.5 误区五：视频文件里的音频会自动参与推理

当前官方 Qwen3-VL processor 的输入分支是 text、image 和 video，没有音频处理分支。[3, 4, 7] 因此，视频中的音频轨道不会自动变成语言模型可用的声学 token；需要音频理解时，应额外接入 ASR 或音频编码器，并把结果作为文本或其他支持的模态输入。

#### 10.6 误区六：视觉 token 越多越好

视觉 token 越多，通常越有利于保留细节，但显存、prefill 延迟和上下文消耗也会增加。[5, 6, 7] 视频任务需要在：

```text
采样密度
空间分辨率
总视觉 token 预算
关键事件漏检风险
```

之间做平衡。

### 11. 对关键帧检测任务的直接启发

你的关键帧任务不是普通的视频 caption，而是要找“第一个满足业务完成态的时间点”。因此输入侧需要重点保证。以下内容是基于 Qwen3-VL 输入机制和你现有业务定义给出的工程建议，不是论文结论。[10, 11]

| 输入环节 | 关键帧任务的要求 |
| --- | --- |
| FPS | 训练、推理和 GT 时间使用同一时间坐标系 |
| frame indices | 保留原视频帧索引，不能只保留抽样序号 |
| timestamp | 明确它对应 temporal patch 的中心时间，而不一定是某个原始帧时间 |
| `max_frames` | 不能为了省 token 让短暂完成态完全落在采样间隔之间 |
| `min_pixels` | UI 小字、价格、角标和按钮必须保持可读 |
| `max_pixels` / `total_pixels` | 控制显存和上下文长度，避免长视频溢出 |
| `video_grid_thw` | 训练和推理的 patch/grid 逻辑保持一致 |
| 答案格式 | 时间点、时间段和 JSON 字段固定，方便 verifier 计算误差 |

一个实用的理解方式是。这是业务侧的责任划分，不是模型论文中的模块边界：[10, 11]

```text
Qwen3-VL 负责：
  看懂采样到的视觉证据
  理解文本任务规则
  在时间结构上进行跨帧推理

业务数据链路负责：
  采样是否覆盖关键事件
  timestamp 是否和 GT 对齐
  UI 分辨率是否足够
  输出时间如何映射回原视频帧
```

模型再强，也不能从没有被采样到的帧中恢复确定的视觉证据。[5, 10, 11]

### 12. 参考文献与代码

本文正文只保留简短的数字引用，例如 `[1, 5, 6]`。完整的论文、官方文档、源码和业务依据统一列在本节。

1. **Qwen3-VL Technical Report**：[论文原文](https://arxiv.org/abs/2511.21631)。主要引用章节：
   - `Model Architecture`：三模块架构、SigLIP-2、2x2 Merger、动态分辨率。
   - `§2.1 Interleaved MRoPE`：T/H/W 轴和交错频率分配。
   - `§2.2 DeepStack`：ViT 中间层特征注入 LLM 前几个层。
   - `§2.3 Video Timestamp`：每个 temporal patch 前加文本 timestamp、seconds/HMS、Qwen2.5-VL 的问题。
   - `§3.1 Training Recipe` 和 Table 1：四阶段预训练。
2. **Qwen3-VL 官方发布说明**：[Qwen3-VL: Sharper Vision, Deeper Thought, Broader Action](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef)。主要用于 `timestamps-video frames` 交错输入、seconds/HMS 输出格式和官方能力描述。
3. **Qwen 官方仓库 README**：[QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)。主要用于 Quickstart、图片/视频消息格式、多图输入、视频处理参数和视觉 token 预算示例。
4. **Transformers 官方文档**：[Qwen3-VL model documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_vl)。主要用于 processor 返回字段、`Qwen3VLVisionConfig`、`image_grid_thw`、`video_grid_thw` 和模型接口。
5. **Qwen3-VL video processor 源码**：[video_processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py)。主要函数：
   - `smart_resize()`：视频总像素预算和尺寸对齐。
   - `Qwen3VLVideoProcessor` 类属性：`patch_size`、`temporal_patch_size`、`merge_size`、`fps`、帧数上下限。
   - `sample_frames()`：FPS/固定帧数采样、`np.linspace()` 和原始帧索引。
   - `_preprocess()`：归一化、补帧、时空 patch reshape 和 `video_grid_thw`。
6. **Qwen3-VL processor 源码**：[processing_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)。主要函数：
   - `Qwen3VLProcessor.__call__()`：image/video placeholder 展开、timestamp 字符串拼接和 tokenizer 调用。
   - `_calculate_timestamps()`：`frame_index / original_fps`、temporal patch 分组和组内时间平均。
7. **Qwen3-VL modeling 源码**：[modeling_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)。主要函数：
   - `Qwen3VLVisionModel.forward()`：视觉 patch、视觉位置和 DeepStack feature 提取。
   - `Qwen3VLModel.forward()`：视觉 embedding 替换到 placeholder。
   - `get_rope_index()`：`mm_token_type_ids`、T/H/W position ids 和 timestamp 分隔的视频 grid。
   - `apply_interleaved_mrope()`：交错频率布局。
   - `_deepstack_process()`：视觉位置上的 residual addition。
8. **Qwen3-VL 配置源码**：[configuration_qwen3_vl.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/configuration_qwen3_vl.py)。主要用于当前配置示例：Vision depth 27、hidden size 1152、16 heads、patch size 16、spatial merge 2、temporal patch 2、DeepStack indexes `[8, 16, 24]`。
9. **兼容图像处理源码**：[Qwen2VLImageProcessor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py)。只用于说明当前兼容图像处理路径中的单图 temporal 补齐、动态 resize 和 image grid 行为，不把这些版本相关实现细节误写成 Qwen3-VL 论文结论。
10. **关键帧任务定义**：[任务定义与标注标准.md](<../../12_实习工作整理/关键帧检测/任务与数据治理/任务定义与标注标准.md>)。用于本文中“首次完成态”“GT 时间”和业务时间边界的说明。
11. **关键帧 Qwen3-VL 数据格式**：[Qwen3-VL关键帧数据格式.md](<../../12_实习工作整理/关键帧检测/工程实现/Qwen3-VL关键帧数据格式.md>)。用于本文中视频字段、`<video>` 占位符、视频路径、FPS 和工程数据链路的说明。

公式、token 数量、序列展开示例和跨来源综合解释属于本文推导，主要根据 `[3]` 至 `[9]` 得出，不是外部文献的直接原文。关键帧检测部分的采样、GT、UI 分辨率和 verifier 建议属于业务建议，主要依据 `[10]` 和 `[11]`，不代表 Qwen3-VL 官方保证。

正文引用的阅读方式：

- `[1]`、`[2]`：论文和官方技术说明。
- `[3]`、`[4]`：官方仓库和 Transformers 文档。
- `[5]` 至 `[9]`：具体 processor、modeling、config 和图像处理源码。
- `[10]`、`[11]`：当前知识库中的关键帧业务定义和数据格式。
- 没有数字引用的解释性句子，是对相邻已引用材料的通俗化转述或本文推导。

## 面试应对

### 常考点及考法

下面的问题是根据论文和实现整理出的面试考法，不是官方题目清单。

1. 给出一段文本、图片或视频后，Qwen3-VL 的输入如何进入 LLM？
2. Qwen3-VL 的视频是不是每一帧前面都加一个时间戳？
3. 视频 timestamp 和 MRoPE 分别解决什么问题？
4. `image_grid_thw` / `video_grid_thw` 的含义是什么？
5. Qwen3-VL 如何控制视觉 token 数量？
6. DeepStack 为什么要从 ViT 中间层取特征？
7. 视频输入是否包含音频？如何处理长视频的成本？

### 解法 / 回答思路

回答这类题，建议固定按五步展开。下面是面试表达上的归纳，不是论文原文。

```text
第一步：先说统一框架
  文本 tokenizer，图像/视频 Vision Encoder，最后进入 Qwen3 decoder。

第二步：再说视觉 token 化
  动态 resize -> patch -> spatial merge -> MLP projector。

第三步：单独说视频
  抽帧 -> temporal patch -> 每个 temporal patch 前加文本 timestamp。

第四步：区分位置机制
  timestamp 表达绝对时间，Interleaved-MRoPE 表达 T/H/W 结构位置。

第五步：补工程边界
  采样帧不是全部原始帧，音频不自动处理，token 数受分辨率和帧数控制。
```

### 易错点

下表中的“正确说法”是对论文和当前实现的复述或澄清；涉及精度、音频和采样风险的部分还包含工程推导。

| 错误说法 | 正确说法 |
| --- | --- |
| 每个原始帧前必然有 timestamp | 每个 temporal patch 前有 timestamp；默认通常两帧一个 temporal patch |
| timestamp 是写到图片上的水印 | timestamp 是语言序列中的普通文本 token |
| Qwen3-VL 把绝对秒数直接写入 MRoPE | 绝对秒数用文本 timestamp，MRoPE 表达 T/H/W 结构位置 |
| 视频一次性输入所有帧 | processor 会按 `fps` 或 `num_frames` 采样 |
| 图像统一 resize 到固定正方形 | Native Dynamic Resolution 按像素预算和网格倍数动态调整 |
| DeepStack 是把更多 token 拼到上下文里 | DeepStack 通过 residual injection 把多层视觉特征注入 LLM 早期层 |
| 视频天然包含音频理解 | 这条 Qwen3-VL 图文视频链路主要处理帧和文本，音频需另接模块 |
| 只要模型能输出秒数，时间就一定精确 | 精度受 FPS、frame indices、timestamp、视觉 token 预算和关键事件采样共同限制 |

### 回答模板

Qwen3-VL 的核心做法是把文本、图像和视频统一成一个多模态 token 序列。纯文本先经过 tokenizer 和 Qwen3 decoder；图像和视频则先经过 Vision Encoder，再通过 MLP-based Vision-Language Merger 做空间压缩并投影到语言模型 hidden size，最后替换文本序列中的 image/video placeholder。[1, 3, 6, 7]

图像采用动态分辨率，按像素预算 resize 后切成二维 patch，经过 ViT 和 `2x2` spatial merge 得到视觉 token，并用 `image_grid_thw` 保留二维网格信息。视频会先按 `fps` 或 `num_frames` 抽帧，再按 `temporal_patch_size` 把相邻帧组成 temporal patch，经过 3D patch embedding 和空间 merge 得到视频视觉 token。[1, 3, 5, 8]

视频时间建模要区分两件事：第一，Qwen3-VL 在每个 temporal patch 前插入类似 `<3.0 seconds>` 的文本 timestamp，让模型直接读取原视频绝对时间；第二，Interleaved-MRoPE 用交错的方式编码视觉 token 的时间、高度和宽度结构位置。因此它不是简单把秒数塞进 RoPE，也不是严格给每个原始帧加时间戳。以默认 temporal patch size 为 2 的实现为例，通常两帧共享一个 timestamp，时间戳取这组帧时间的平均值。最后，DeepStack 把 ViT 多个中间层的视觉特征通过 residual 的方式注入 LLM 前几个层，保留 OCR、UI 小字和局部状态等细节，再由 Qwen3 decoder 自回归生成文本、时间段、坐标或结构化结果。[1, 2, 5, 6, 7]
