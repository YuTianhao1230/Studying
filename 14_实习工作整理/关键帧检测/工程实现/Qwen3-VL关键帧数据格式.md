# Qwen3-VL 关键帧数据格式

## 知识点解析

### 这套数据格式的结论

你当前 `video_eval_keyframe -> qwen3` 数据链路，最终不是 OpenAI API 那种直接的 `messages` 格式，而是项目内部的 `MllmData` JSONL 格式：

```text
一行 JSON = 一个训练样本
  -> id
  -> conversations：用户问题 + 助手答案
  -> videos：视频文件路径
  -> image：空列表
  -> infos：任务和运行元数据
  -> meta：样本 ID
```

对 Qwen3-VL 真正重要的是：

```text
视频内容：
  videos[0] 指向视频文件

视频在对话中的位置：
  human.value 以 <video> 开头

任务要求：
  human.value 中的关键帧完成态判断规则 + task_prompt

训练标签：
  gpt.value 中的 <answer>{"time": 秒数}</answer>
```

### 数据生成链路

这类关键帧数据通常经过三个阶段：

```text
原始关键帧标注数据
  -> 任务样本构造和质量检查
  -> train/dev/test JSONL
  -> 转成 Qwen3-VL 多模态格式
  -> 按运行环境适配视频路径
  -> 训练/评测数据
```

### 第一层：任务生成后的标准样本

任务数据构造阶段为每条样本构造一个 `MllmData`：

```json
{
  "id": "sample_000001",
  "conversations": [
    {
      "from": "human",
      "value": "完整的关键帧判断 prompt"
    },
    {
      "from": "gpt",
      "value": "<answer>{\"time\": 12.3}</answer>"
    }
  ],
  "infos": {
    "task_name": "video_keyframe",
    "image_source": "video_keyframe",
    "image_size": [["原始宽度", "原始高度"]],
    "video_info": {},
    "task_type": "中文业务指标名",
    "image_path": "/data/videos/sample.mp4",
    "data_type": "video",
    "answer_obj": "{\"time\": 12.3}",
    "problem_type": "temporal span grounding",
    "data_version": "keyframe_dataset",
    "fps": "由视频采样配置决定",
    "max_frames": "由输入预算决定",
    "max_pixels": "由视觉处理配置决定"
  },
  "image": [],
  "videos": [
    "/data/videos/sample.mp4"
  ]
}
```

其中时间标签的计算逻辑是：

```text
time = round(keyframe_index / fps, 2)
answer = {"time": time}
```

也就是说，你的训练标签最终是**秒级时间点**，不是直接把 `keyframe_index` 作为模型输出标签。

### 第二层：Qwen3-VL 格式转换后的样本

转换成 Qwen3-VL 格式后，关键变化有四个。

#### 1. 给用户轮添加 `<video>`

关键帧任务是一条视频，所以会给 `conversations` 的 human 文本添加视频占位符。

实际效果是：

```text
原始 human.value：
你是一个UI测试专家，负责在一段视频中定位用户操作完成的时间点。
...

转换后 human.value：
<video>你是一个UI测试专家，负责在一段视频中定位用户操作完成的时间点。
...
```

`<video>` 是文本中的模态占位符，告诉 Qwen3-VL：视频特征要插入到这里。

#### 2. 视频路径放到 `videos`

格式转换阶段会把视频资源路径转换成当前运行环境可访问的路径：

```python
videos = [
    "/data/videos/sample.mp4"
]
```

因此：

```text
human.value 的 <video>
  <-> videos[0] 的视频文件
```

两者必须一一对应。关键帧任务只有一个视频，所以只使用 `videos[0]`。

#### 3. 写入模型类型和样本元信息

格式转换阶段会增加：

```json
{
  "infos": {
    "mllm_type": "qwen3"
  },
  "meta": {
    "id": "sample_000001"
  }
}
```

#### 4. `image` 和 `images` 保持为空

这是视频任务，不是图片任务：

```json
{
  "image": [],
  "images": [],
  "videos": ["/data/videos/sample.mp4"]
}
```

不要因为同时看到 `image`、`images`、`videos` 就认为模型会收到三份视觉输入。对当前关键帧样本，真正使用的是 `videos[0]`。

### 最终运行环境版本

为了让同一份数据在不同训练机器上运行，通常会根据挂载环境生成多个路径版本。

这些版本的主要区别是视频路径前缀：

```text
同一条样本、同一个视频、同一个 prompt、同一个答案
  -> 根据训练机器所在磁盘替换 videos[0] 和 infos.image_path 的前缀
```

例如：

```text
/data/disk_a/...
/data/disk_b/...
/data/disk_c/...
```

这一步不改变 Qwen 数据格式、prompt 或答案，只是让当前运行环境能找到视频文件。

### 你当前最终样本的完整示例

下面是按当前代码逻辑整理的结构示例。`human.value` 中的长 prompt 省略了中间规则，但真实文件里会保留完整内容：

```json
{
  "id": "sample_000001",
  "conversations": [
    {
      "from": "human",
      "value": "<video>你是一个UI测试专家，负责在一段视频中定位用户操作完成的时间点。\n\n...完成态定义、排除条件、豁免条件、任务描述...\n\n请只输出：<answer>{\"time\": 秒数}</answer>"
    },
    {
      "from": "gpt",
      "value": "<answer>{\"time\": 12.3}</answer>"
    }
  ],
  "infos": {
    "task_name": "video_keyframe",
    "image_source": "video_keyframe",
    "image_size": [["原始宽度", "原始高度"]],
    "input_size": [["预处理宽度", "预处理高度"]],
    "video_info": {
      "duration": "视频时长",
      "frame_size": ["原始宽度", "原始高度"]
    },
    "task_type": "中文业务指标名",
    "task_type_en": "english_task_name",
    "image_path": "/data/videos/sample.mp4",
    "data_type": "video",
    "answer_obj": "{\"time\": 12.3}",
    "problem_type": "temporal span grounding",
    "data_version": "keyframe_dataset",
    "fps": "由视频采样配置决定",
    "max_frames": "由输入预算决定",
    "max_pixels": "由视觉处理配置决定",
    "mllm_type": "qwen3"
  },
  "image": [],
  "images": [],
  "original_images": [],
  "videos": [
    "/data/videos/sample.mp4"
  ],
  "fps": "由视频采样配置决定",
  "max_frames": "由输入预算决定",
  "max_pixels": "由视觉处理配置决定",
  "meta": {
    "id": "sample_000001"
  }
}
```

说明：

- `task_type_en` 可以由中文到英文的映射表补充，是元数据，不会重新生成 prompt。
- `original_images` 是为了兼容图片字段保留的空列表，对当前视频任务没有视觉内容。
- `fps`、`max_frames`、`max_pixels` 的顶层副本方便训练/推理框架读取，和 `infos` 中的配置保持一致。
- `input_size` 是预处理尺寸记录，不代表 Qwen3-VL 要求所有图片固定成这个尺寸。Qwen3-VL 的动态 resize、像素预算和视觉 token 计算见 [Qwen千问架构.md](<../../../02_大模型/模型细节/Qwen千问架构.md>)。

### Qwen3-VL 实际接收什么

需要区分“模型输入”和“数据集元数据”。

#### 训练时模型真正使用的信息

```text
1. videos[0]
   视频文件，processor 从路径读取视频并按 fps / max_frames / max_pixels 采样和预处理。

2. human.value
   <video> 占位符 + 关键帧判断规则 + task_type 对应的任务描述 + 输出格式要求。

3. gpt.value
   训练标签：<answer>{"time": 12.3}</answer>。
```

模型内部看到的不是 MP4 路径字符串，而是：

```text
视频文件
  -> processor 抽帧/缩放
  -> Vision Encoder
  -> visual tokens
  -> 通过 <video> 位置插入 Qwen3 decoder
```

#### 主要用于数据处理的字段

这些字段通常不作为自然语言直接喂给模型：

| 字段 | 用途 |
| --- | --- |
| `id` / `meta.id` | 样本追踪、结果回写、问题定位 |
| `infos.task_name` | 区分任务 |
| `infos.task_type` | 数据统计、过滤、样本分析；对应任务描述已写入 prompt |
| `infos.task_type_en` | 英文映射和下游分析，不替代 prompt 中的任务描述 |
| `infos.image_path` | 路径、审计和 bad case 分析 |
| `infos.video_info` | 视频时长、尺寸等元信息 |
| `infos.answer_obj` | 标签留存和评测对齐 |
| `fps` / `max_frames` / `max_pixels` | 视频预处理和 token/帧预算 |
| `image_size` / `input_size` | 尺寸记录和预处理配置 |

最重要的边界是：

```text
task_type 本身不是模型唯一依据。
模型真正看到的是 prompt 中的完成态规则 + task_prompt，
而视频内容来自 videos[0]。
```

### 为什么这样设计

#### 用 `conversations` 表示监督关系

```text
human = 问题和任务约束
gpt   = 目标输出
```

这样可以直接用于 SFT 的 teacher forcing：模型根据用户轮生成助手轮，loss 主要落在答案 token 上。

#### 用 `<video>` 和 `videos` 分离模态位置与文件资源

```text
<video>：告诉模型视频在对话中的位置
videos：告诉数据处理器从哪里读取视频
```

文本不直接塞视频内容，路径也不直接替代模态占位符，二者配合才能完成多模态拼接。

#### 输出只保留时间 JSON

当前无 CoT 的 prompt 版本，目标是让模型稳定学习：

```text
视频 + 业务规则
  -> 完成态第一次成立的时间
  -> <answer>{"time": ...}</answer>
```

这样做的原因：

- 标签定义单一，loss 目标清晰。
- 输出容易解析和评测。
- 不让模型在训练初期把容量浪费在长解释上。
- 后续如果做 CoT 蒸馏，可以替换 prompt 版本和答案格式，但仍保留 `<video>` + `conversations` + `videos` 这套外层结构。

#### 用 `for_disk` 适配训练环境

同一份数据需要在不同挂载盘上运行。路径转换只改资源路径，不改任务 prompt、视频内容和答案标签，避免因为存储环境不同产生数据语义差异。

### 一句话面试答案

> 我们给 Qwen3-VL 的关键帧 SFT 数据采用项目内部的 JSONL 多模态对话格式，每行一个样本，核心字段是 `conversations` 和 `videos`。用户轮以 `<video>` 占位符开头，后面是关键帧完成态规则、task prompt 和输出格式；`videos[0]` 提供实际视频文件；助手轮输出 `<answer>{"time": 秒数}</answer>` 作为监督标签。`infos`、`meta`、`fps`、`max_frames` 和 `max_pixels` 主要用于样本追踪、任务统计、视频预处理和磁盘路径适配，不是全部直接拼进模型文本。这样设计是为了把视频模态位置、任务约束和监督答案明确分开，同时保持训练格式稳定、输出可解析、不同机器路径可复用。

## 面试应对

### Qwen3-VL 的关键帧训练样本是什么格式？

回答思路：先说 JSONL 一行一条，再抓住 `conversations`、`<video>`、`videos[0]` 和答案格式。

回答模板：

我们的关键帧数据是一行一个 JSON 的多模态对话样本，核心字段是 `conversations` 和 `videos`。`conversations` 里有一轮 human 和一轮 gpt：human 内容以 `<video>` 占位符开头，后面是完成态定义、排除条件、任务描述和输出约束；`videos[0]` 是对应的视频文件路径；gpt 是监督答案，固定输出 `<answer>{"time": 秒数}</answer>`。`infos` 里还保留 task_type、fps、max_frames、视频尺寸和样本 ID，用于数据处理、追踪和评测。

### `<video>` 和 `videos[0]` 分别有什么作用？

回答思路：区分“对话中的模态位置”和“实际资源路径”。

回答模板：

`<video>` 是 human 文本中的模态占位符，表示视频特征要插入对话的这个位置；`videos[0]` 是数据处理器实际读取的视频文件路径。两者必须一一对应，前者解决“视频在序列中的位置”，后者解决“从哪里取视频”。processor 会读取 `videos[0]`，按 fps、最大帧数和像素预算采样，再把视觉 token 插入 `<video>` 对应的位置。

### `infos` 里的 task_type、fps、max_frames 会直接作为模型输入吗？

回答思路：区分已经写进 prompt 的任务语义和仅供 pipeline 使用的元数据。

回答模板：

不完全是。`task_type` 会参与生成 task prompt，任务的完成态规则最终会写进 human 文本，所以模型能看到的是完整任务描述，而不是只看到一个字段名。`fps`、`max_frames`、`max_pixels` 主要控制视频 processor 的采样和 token 预算；`id`、`video_info`、`answer_obj` 主要用于数据追踪、统计和评测，通常不会作为自然语言直接拼到 prompt 里。

### 为什么答案用 `<answer>{"time": ...}</answer>`？

回答思路：从可解析、监督目标单一、评测对齐和后续扩展回答。

回答模板：

关键帧任务最终只需要一个时间点，所以答案采用 `<answer>{"time": 秒数}</answer>` 的固定结构。这样监督目标单一，SFT 更容易收敛；推理结果可以直接解析成 JSON，方便计算 frame error、time error 和 PASS；同时用标签包住答案，可以减少模型输出解释、Markdown 或额外文本。后续如果加入结构化 CoT，只需要扩展答案内部字段或替换 prompt 版本，外层的多模态对话格式不需要改变。
