# 关键帧检测：CoT 蒸馏与 RL 方案

## 知识点解析

### 概述

本文整理关键帧检测中的 [CoT](<../../02_大模型/应用与问题/CoT.md>) 数据蒸馏方案。这里的重点不是“让模型多写解释”，而是把关键帧任务从单点时间监督升级为：

```text
任务规则理解
  -> 视频原语时间轴
  -> 候选边界前后证据
  -> 首次完成态判断
  -> 隐藏校验和过滤
  -> 高质量训练样本
```

现成方案已经落在代码仓库：

```text
/mlx_devbox/users/yutianhao/mllm-data/data_prepare/video_data_distillation/
```

核心任务名是：

```text
eval_keyframe
```

核心 graph：

```text
extract_keyframe_primitives
  -> generate_keyframe_result
  -> eval_keyframe_finalize
```

### 为什么关键帧任务需要 CoT 蒸馏

关键帧检测不是普通视频问答，也不是只预测一个时间戳。它要判断的是：

```text
给定操作录屏 + task_type 标注规则
找到第一张满足完成态的帧
```

很多错误不是模型完全看不到目标，而是边界判断不稳：

- 把加载中、过渡态、骨架屏误判为完成态。
- 看到主体内容出现就提前截，没有等核心区域稳定。
- 等到了更晚的稳定帧，违反“第一帧原则”。
- 看错 UI 区域，比如关注了角标、浮层、非目标组件。
- 被二次刷新、内容替换、局部动效干扰。
- 人工 GT 本身有噪声，模型如果强行模仿会学到脏标签。

原始 [SFT](<../../03_训练优化与对齐/后训练与对齐/SFT 监督微调.md>) 只给 `<answer>{"time": ...}</answer>`，告诉模型“答案在哪里”，但没有告诉模型：

- 为什么前一帧不能选。
- 为什么当前帧是第一次满足。
- 为什么后一段只是稳定延续。
- 哪个 UI 区域才是目标证据。
- 哪些变化是排除项或豁免项。

CoT 蒸馏的价值是把这些人工标注逻辑显式化，让学生模型学到“边界判断过程”。

### 方案演进

#### 第一版：单段 CoT

第一版直接让教师模型围绕 GT 生成完整时间片：

```text
GT 前未完成
  -> GT 附近完成
  -> GT 后稳定
  -> answer
```

问题：

- 时间切片不够细，容易把 GT 前后大段内容合并。
- 容易出现“因为 GT 是 3.47s 所以这里是完成态”的来源泄漏。
- 部分 CoT 是事后合理化，不一定真的来自视觉证据。
- 如果人工 GT 错，CoT 会围绕错误答案编解释。

#### 第二版：粗筛 + 精筛

第二版开始模拟人工标注路径：

```text
完整视频粗筛
  -> 找候选完成窗口
  -> 候选窗口高帧率精筛
  -> 比较 GT 前 / GT / GT 后
```

核心思想：

- 粗筛阶段低 FPS 看完整视频，避免丢全局上下文。
- 精筛阶段只看候选窗口附近，高 FPS 比较边界帧。
- 对 GT 错误样本允许模型判断并 relabel，但要进入人工抽检或过滤。

典型例子：

```text
0.4s：过渡动画，文字模糊，not_satisfied
0.5s：仍有重影，uncertain
0.6s：文字清晰稳定，satisfied
0.7s：后续稳定，stable_after
```

#### 第三版：Video Primitive + 问题驱动推理

当前代码里的现成方案是第三版，核心是两步式：

```text
第一步：抽任务无关的视频原语
第二步：基于任务规则和原语做关键帧推理
```

第一步不看任务答案、不看 GT、不判断关键帧，只生成客观视频时间轴：

```json
{
  "temporal_primitives": [
    {
      "primitive_id": "state_001",
      "primitive_type": "state",
      "start_time": 0.00,
      "end_time": 0.30,
      "content": "手机主屏幕处于稳定状态，显示多个应用图标。"
    },
    {
      "primitive_id": "event_001",
      "primitive_type": "event",
      "start_time": 0.30,
      "end_time": 1.20,
      "content": "用户启动应用，页面从桌面切换到启动页并开始加载。"
    }
  ]
}
```

第二步才结合 task_type 标准、内部候选边界和过滤后的原语，输出结构化关键帧推理：

```json
{
  "key_time": 6.97,
  "task_type": "商城首页加载耗时",
  "label_quality": "clean",
  "task_understanding": "定位商城首页首屏完成加载并稳定的最早时间点。",
  "target_region": "商城首页首屏商品流、权益栏、底部导航栏",
  "required_evidence": [
    "页面主干定型",
    "核心商品图文真实渲染",
    "后续没有二次刷新替换"
  ],
  "candidate_observation": [
    {
      "time": 6.53,
      "status": "not_satisfied",
      "evidence": "商品区域仍在刷新，部分图片为空白或占位。"
    },
    {
      "time": 6.97,
      "status": "satisfied",
      "evidence": "刷新后的商品图、权益栏和底部导航稳定显示。"
    },
    {
      "time": 10.00,
      "status": "stable_after",
      "evidence": "后续页面保持稳定，无二次刷新或核心替换。"
    }
  ],
  "boundary_check": "6.53s 前仍有内容刷新，6.97s 首次满足完成态，后续稳定，因此 6.97s 是首个完成边界。"
}
```

### 现有代码链路

代码入口：

```text
/mlx_devbox/users/yutianhao/mllm-data/data_prepare/video_data_distillation/run_distillation.py
```

任务注册：

```text
/mlx_devbox/users/yutianhao/mllm-data/data_prepare/video_data_distillation/tasks.py
```

Graph：

```text
/mlx_devbox/users/yutianhao/mllm-data/data_prepare/video_data_distillation/graphs/eval_keyframe_graph.py
```

三个节点：

| 节点 | 文件 | 职责 |
| --- | --- | --- |
| `extract_keyframe_primitives` | `nodes/eval_keyframe_primitive_node.py` | 抽取任务无关 Video Primitive Script，并做质量过滤 |
| `generate_keyframe_result` | `nodes/eval_keyframe_generate_node.py` | 基于任务规则、内部候选边界和原语生成关键帧 CoT |
| `eval_keyframe_finalize` | `nodes/eval_keyframe_finalize_node.py` | 做隐藏校验、泄漏/一致性过滤，构造训练样本或 reject 样本 |

核心 prompt 和 parser：

```text
/mlx_devbox/users/yutianhao/mllm-data/data_prepare/video_data_distillation/skills/eval_keyframe_skill.py
```

补充 prompt：

```text
/mlx_devbox/users/yutianhao/mllm-data/data_prepare/evaluation_keyframe/prompts/cot_visual_primitives_v1.txt
/mlx_devbox/users/yutianhao/mllm-data/data_prepare/evaluation_keyframe/prompts/no_cot_v1.txt
```

### eval_keyframe 三阶段详解

#### 阶段一：抽取视频原语

系统 prompt 要求教师模型完整观看视频，并输出任务无关的 Video Primitive Script。

关键约束：

- 不考虑下游问题、任务规则、人工 GT 或关键帧答案。
- 只描述客观可见内容。
- 按“稳定状态 -> 完整变化事件 -> 新稳定状态”切分。
- 静止画面合并为一个状态。
- 同一次加载/刷新/转场合并为一个事件。
- 时间覆盖完整视频时间轴。
- 禁止输出“完成态”“关键帧”“GT”“候选边界”等任务判断。

这一阶段解决的是：先让模型稳定“看懂视频时间轴”，而不是一上来就做任务答案。

#### 阶段二：原语质量过滤

代码会调用 quality prompt，对视频原语做质量审核。

样本级字段：

```json
{
  "sample_keep": true,
  "sample_score": 8.0,
  "video_quality": 8.0,
  "static_time_ratio": 0.25,
  "sample_issues": []
}
```

单条 primitive 字段：

```json
{
  "primitive_id": "state_001",
  "keep": true,
  "overall_score": 8.0,
  "scores": {
    "temporal_alignment": 8,
    "visual_grounding": 8,
    "segment_boundary": 8,
    "timeline_completeness": 8,
    "writing_quality": 8
  },
  "issues": []
}
```

关键阈值：

```text
video_quality_threshold=7.0
primitive_score_threshold=7.0
```

低质量视频、时间轴缺口、描述幻觉、边界混乱、含 GT/答案泄漏的原语会被过滤。

#### 阶段三：生成关键帧训练样本

生成阶段输入：

- 原始视频。
- task_type 和任务规则。
- 内部候选边界 `gt_time`。
- 过滤后的 Video Primitive Script。

注意：`gt_time` 用于内部对齐精确帧，但输出中不能说“GT/人工标注/参考答案/内部候选”。这一步是最容易出问题的地方。

输出字段：

| 字段 | 作用 |
| --- | --- |
| `key_time` | 最终关键帧时间 |
| `label_quality` | `clean` / `ambiguous` / `insufficient_visual_evidence` |
| `task_understanding` | 当前 task_type 要定位什么 |
| `target_region` | 应重点观察的 UI 区域 |
| `required_evidence` | 完成态必须满足的证据 |
| `candidate_observation` | 边界前、边界处、边界后的证据 |
| `exclusion_check` | 骨架屏、转圈、位移、二刷、豁免项检查 |
| `boundary_check` | 为什么该时间是首次完成边界 |
| `teacher_process` | 训练时希望学生学到的判断要点 |
| `reasoning_trace` | 引用 2-3 个相关原语形成推理闭环 |

### 隐藏校验与过滤

`finalize` 阶段会把样本分桶：

```text
gt_correct / relabeled / reject
```

当前默认配置更偏保守：

```text
disable_relabel=true
use_gt_as_training_answer=false
gt_time_tolerance=0.05
accept_label_quality=clean,ambiguous
reject_on_warnings=false
```

含义：

- 默认不输出 relabel 样本。
- 允许 `clean` 和轻微 `ambiguous`。
- `gt_time_tolerance=0.05` 表示 hidden check 中允许 0.05s 误差。
- 如果推理链否定最终答案、缺少 before/answer/after 证据、出现来源泄漏，会进入 reject。

典型 fatal warning 包括：

- `key_time does not match accepted GT time`
- `candidate_observation should contain at least 3 frames`
- `missing not_satisfied evidence before answer`
- `missing satisfied evidence at answer`
- `missing boundary_check`
- `reasoning_trace contains label leakage`
- `reasoning_trace contradicts answer`

### 运行方式

当前 `run_distillation.py` 已写死默认任务为 `eval_keyframe`：

```bash
cd /mlx_devbox/users/yutianhao/mllm-data
python3 -m data_prepare.video_data_distillation.run_distillation
```

脚本当前关键配置：

```text
args.task = "eval_keyframe"
eval_keyframe_call_method=cli
model_name=gpt-5.5
source_model_name=gpt-5.5
video_input_strategy=local
cli_timeout=1800
reasoning_effort=low
disable_relabel=true
use_gt_as_training_answer=false
gt_time_tolerance=0.05
cli_max_images=100
gt_alignment_retry=true
cli_frame_max_pixels=0
cli_image_jpeg_quality=100
cli_overlay_timestamps=true
accept_label_quality=clean,ambiguous
reject_on_warnings=false
video_quality_threshold=7.0
primitive_score_threshold=7.0
```

三台开发机分片跑时：

```bash
export DISTILLATION_NUM_SHARDS=3
export DISTILLATION_SHARD_INDEX=0  # 另外两台设 1 / 2

cd /mlx_devbox/users/yutianhao/mllm-data
python3 -m data_prepare.video_data_distillation.run_distillation
```

### 输出产物

每条输出包含：

| 字段 | 含义 |
| --- | --- |
| `bucket` | `gt_correct` / `relabeled` / `reject` |
| `training_record` | 可进入训练集的最终样本 |
| `reject_record` | 被拒样本的审计信息 |
| `eval_keyframe_result` | 原语、生成结果、隐藏校验、warnings 等完整审计 |
| `debug_info` | 运行调试信息 |

训练时真正需要的是 `training_record`，审计和 debug 字段不要混进训练数据。

### 和 RFT/RL 的衔接

CoT SFT 的定位是冷启动：

```text
Structured CoT SFT
  -> 让模型学会合法格式、证据引用和 before/answer/after 判断
```

RL / [GRPO](<../../03_训练优化与对齐/后训练与对齐/GRPO 组相对策略优化.md>) 的定位是进一步优化可验证目标：

```text
R = R_time + R_format + R_boundary_evidence - P_length
```

| Reward | 说明 |
| --- | --- |
| `R_time` | `abs(pred_time - gt_time) < 0.1s`，或按帧误差分段奖励 |
| `R_format` | 是否输出可解析结构，`<answer>{"time": ...}</answer>` 是否合法 |
| `R_boundary_evidence` | 是否包含 before 未完成、answer 首次完成、after 稳定证据 |
| `P_length` | 惩罚冗长、重复、循环、无效推理 |

实际训练顺序建议：

```text
Direct SFT
  -> 少量高质量 Structured CoT SFT
  -> 小规模 RL/GRPO
  -> 分业务线/分指标回归评测
```

不要跳过 Direct SFT 直接上长 CoT，也不要把所有 CoT 全量混入训练。CoT 数据应该先经过过滤、分桶、抽样和人工抽检。

### 关键设计原则

#### 1. CoT 不是越长越好

长 CoT 会增加 token 成本，也容易把模型带偏。文档里已有评论指出：冗余推理价值不高，相似语义太多会影响模型学习。

更好的方式是：

```text
短 summary + 关键证据 + 边界判断
```

#### 2. 第一阶段必须任务无关

Video Primitive 阶段不能知道答案。否则模型会从答案反推视觉描述，导致训练样本看似合理但不可泛化。

#### 3. 推理必须绑定证据

不能只写“页面已经完成”。要说明：

- 哪个区域完成。
- 哪个元素从未完成到完成。
- 为什么前一帧不能选。
- 后面有没有二次刷新或替换。

#### 4. GT 只能内部校准，不能泄漏

禁止出现：

```text
GT
人工标注
参考答案
内部候选
给定时间
标签质量
```

这些词一旦进入训练数据，学生模型会学到“答案来源”，而不是学视觉边界。

#### 5. 脏数据要隔离

人工 GT 有噪声。模型判断出 GT 错时，不应直接混进训练集。应进入：

```text
reject / relabel 待审
  -> 人工抽检
  -> 达标后再回流
```

### 与相关工作的对应

现有方案不是凭空设计，主要复用了这些机制：

| 相关方向 | 可复用点 | 当前任务迁移 |
| --- | --- | --- |
| Open-o3-Video / STGR-CoT | 显式时间和空间证据 | `target_region`、`candidate_observation`、`boundary_check` |
| Video Primitive | `state/event` 视频脚本 | 第一阶段任务无关原语时间轴 |
| LongVT / VITAL | global-to-local 找证据 | 低 FPS 粗筛 + 候选窗口高 FPS 精筛 |
| TVG-R1 / Time-R1 | temporal reward | 帧误差、early/late/invalid penalty |
| Visual Primitives / OmniParser | 视觉区域绑定 | 后续加入 `box` / `region_id` |

当前最值得做的是 P0/P1：

```text
P0: before/current/after 结构化 CoT
P0: key_frame-1 / key_frame / key_frame+1 hard negative
P0: target_region 文本字段
P1: OCR / OmniParser evidence_region
P1: frame-level RL reward
```

暂不建议一开始做：

- 完整 tool-calling。
- 全量 box/mask 标注。
- 完整 Open-o3-Video 复现。
- 从零训练 GUI grounding 模型。

### 易错点

- 把 CoT 蒸馏说成“让模型写理由”，没有讲两步式原语和隐藏校验。
- 忽略第一阶段必须任务无关，导致 GT 泄漏。
- 把所有长 CoT 都混进训练，不做过滤。
- 只看 CoT 文本是否像人写的，不校验视觉证据是否真实存在。
- 只优化解释质量，不看最终 `key_time` 的帧误差。
- 不区分 `training_record` 和审计字段，把中间过程污染训练数据。
- 人工 GT 错误样本直接回流，导致模型学到错误边界。

## 面试应对

### 关键帧项目里的 CoT 蒸馏方案是怎么设计的？

回答思路：不要只讲“生成思维链”，要讲清两步式：任务无关视频原语 + 任务驱动边界推理 + hidden filter。

回答模板：

关键帧项目里的 CoT 蒸馏不是简单让教师模型写一段解释，而是做成了两步式数据生成。第一步先抽任务无关的 Video Primitive Script，只描述视频里客观发生的状态和事件，不看任务答案、不看 GT，也不判断关键帧。第二步再把 task_type 标注规则、过滤后的原语和内部候选边界交给教师模型，让它输出 `task_understanding`、`target_region`、`candidate_observation`、`boundary_check` 和最终 `key_time`。最后还有 finalize 阶段做隐藏校验，检查时间是否对齐、是否有 GT 泄漏、是否有 before/answer/after 证据，只有通过过滤的 `training_record` 才进入训练集。

### 为什么不能直接把 GT 给教师模型生成 CoT？

回答思路：核心是避免来源泄漏和事后合理化，让模型学视觉证据而不是学答案来源。

回答模板：

不能直接把 GT 暴露给教师模型，因为这样很容易生成“答案反推式”的 CoT。模型可能不是根据视觉证据判断这个时间点，而是围绕给定 GT 编一个看起来合理的解释。这样的数据会让学生模型学到答案来源和模板化理由，而不是学会边界判断。当前方案里 GT 只作为内部候选边界用于精确对齐，输出里禁止出现 GT、人工标注、参考答案、内部候选等来源性表达，并且要用前后视觉证据证明为什么这里是第一满足完成态。

### Video Primitive 阶段为什么要和任务判断解耦？

回答思路：说明它的作用是先稳定看懂视频时间轴，减少直接推理时的幻觉和任务污染。

回答模板：

Video Primitive 阶段的目标是先让模型客观描述视频时间轴，比如哪个时间段是稳定页面，哪个时间段是点击、加载、刷新、页面跳转。它不应该知道下游 task_type 的答案，也不应该判断关键帧。这样做的好处是把“看懂视频”和“按规则判完成态”拆开：第一步降低视频理解幻觉，第二步再根据任务规则引用少量相关原语做边界判断。否则模型一上来就看任务和答案，很容易只围绕答案写解释，泛化性和可审计性都会变差。

### CoT 数据怎么过滤？

回答思路：从原语质量、结构合法、时间对齐、证据闭环、泄漏检查五类过滤讲。

回答模板：

过滤分几层。第一层是 Video Primitive 质量过滤，看时间轴是否完整、状态和事件边界是否清楚、视觉描述是否可验证，低于阈值的样本或原语会被丢掉。第二层是生成结果结构检查，要求 `key_time`、`candidate_observation`、`boundary_check`、`reasoning_trace` 等字段完整且可解析。第三层是时间对齐检查，预测时间必须在容忍范围内对齐内部候选边界。第四层是证据闭环，必须包含边界前未完成、边界处完成、边界后稳定或反证。第五层是泄漏检查，禁止出现 GT、人工标注、参考答案这类来源表达。只有通过这些过滤的 `training_record` 才能进入训练集。

### CoT SFT 和 RL 在这个项目里怎么衔接？

回答思路：SFT 负责冷启动格式和证据引用，RL 负责优化可验证目标。

回答模板：

我会把 CoT SFT 作为 RL 前的冷启动。Direct SFT 先保证模型有基础关键帧判断能力；Structured CoT SFT 再用少量高质量样本教模型输出合法格式、引用目标区域、比较边界前后证据。RL 或 GRPO 放在后面，用可验证 reward 继续优化，比如时间误差 `R_time`、格式合法 `R_format`、边界证据 `R_boundary_evidence`，再加一个长度惩罚 `P_length` 防止模型输出冗长无效推理。这样训练目标不会停留在“解释写得像”，而是同时约束答案准确、格式稳定和证据真实。

### 如果人工 GT 本身是错的，CoT 蒸馏怎么处理？

回答思路：强调不能强行围绕错 GT 生成训练样本，要进入 relabel/reject 和人工抽检。

回答模板：

如果人工 GT 本身是错的，不能强行让教师模型围绕这个 GT 写 CoT，否则会把脏标签放大。更合理的做法是让教师模型独立判断边界，如果发现视觉证据不支持 GT，就把样本打到 relabel 或 reject。当前默认配置比较保守，`disable_relabel=true`，也就是不直接输出 relabel 训练样本，而是把这类样本作为审计或人工复查对象。只有确认模型重标准确率足够高，或者经过人工抽检后，才考虑把 relabel 样本回流训练。
