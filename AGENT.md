# AGENT.md

本文件给后续进入该工作区的 AI Agent 使用。当前路径 `/mlx_devbox/users/yutianhao` 不是单一代码仓库，而是用户的研究工作区。最重要的前提是：不要把所有目录平铺理解为“用户的工作”。这里有主线项目、baseline/依赖仓库、数据工程项目、模型缓存和个人工具，它们的角色不同。

## 用户当前真正的工作

用户当前主线是 `Syner-Attack/`：围绕论文 **“Syner-Attack: Decoupling Cross-modal Alignment for Transferable Adversarial Attacks on Vision-Language Models”** 做返修、补实验、结果整理和代码支撑。

这项工作的核心目标不是一般性维护多模态代码，而是回应 ACM MM 2026 review 中的关键质疑：

- 原论文对商业/现代 MLLM 的评估太定性，需要补充定量 ASR、样本量、协议和对比。
- “cross-modal decoupling / 跨模态解耦”的定义和理论解释不够严谨，需要形式化定义和实证指标。
- 方法被认为是现有攻击技术的组合，需要用机制分析、消融和互补性证据重新支撑 novelty。
- baseline 对比公平性、扰动预算、超参数和复现来源需要讲清楚。
- 需要补充计算开销、防御评估、VGA/扰动预算/协同效应消融、语义保持验证等结果。

用户的实际需求通常是：帮忙把这些 reviewer concern 转化成可运行实验、可靠表格、论文可用叙述和可复现脚本。Agent 应优先服务这条返修主线，而不是泛泛整理仓库。

## 主线项目：Syner-Attack

`Syner-Attack/` 是用户自己的当前工作区。重要文件：

- `5562_Syner_Attack_Decoupling_C.pdf`、`paper_extracted_text.txt`：论文原稿和提取文本。
- `review_summary.md`：review 意见汇总，说明被拒核心原因。
- `revision_guide.md`：返修路线图，按 P0/P1/P2/P3 列出需要补的实验和写作策略。
- `Attacker.py`：Syner-Attack 核心攻击实现，包括图像侧 feature/alignment loss、输入多样性、SIA、TI kernel，以及文本侧 visual-guided attack 调用。
- `eval.py`：传统 VLP 检索评估入口，围绕 ALBEF/TCL/CLIP 等模型做 ITR 结果。
- `eval_mllm_attack.py`：现代 MLLM 定量评估入口，支持开源 MLLM 和商业 API，包含 `image_only` / `image_and_text` 模式、ASR 统计、GLEAM/SA-AET baseline 数字对齐。
- `eval_defense.py`：防御评估入口，覆盖 JPEG、bit-depth、Gaussian blur/noise、random resize padding 等输入预处理防御。
- `scripts/fresh_generate_attacks.py`：生成 clean / Syner / GLEAM 对齐样本，当前用于 MSCOCO sample100/sample500 实验。
- `scripts/eval_mllm_clip_proxy_gleam_style.py`：按 GLEAM 风格的 CLIP-proxy 协议评估 LLaVA、Qwen2-VL、InstructBLIP 等开源 MLLM。
- `scripts/eval_openai_clip_proxy_existing.py`：对 GPT-4o 等商业模型做 CLIP-proxy ASR 评估。
- `scripts/eval_gleam_style_defense_itr.py`、`scripts/eval_gleam_style_defense_itr_with_syner_text.py`：防御下的 ITR 评估，分别对应不同文本口径。
- `scripts/summarize_compute_cost.py`：整理 GLEAM 与 Syner-Attack 的 wall-clock、sec/img、显存、forward pass 估计。
- `scripts/build_paper_aligned_reports.py`：把 500 样本结果整理成论文表格，显式对齐 GLEAM Table 3/Table 4。
- `test_outputs/`：当前实验产物，包含 sample100/sample500 的 Syner/GLEAM 对抗图、MLLM 评估结果、防御结果、日志和论文表格草稿。

进入 `Syner-Attack/` 做任务时，要先读 `revision_guide.md` 和相关脚本，再判断用户是在补哪类 reviewer concern。

## Baseline / 依赖角色划分

不要把下面这些目录误判成当前主线。它们主要是 baseline、外部依赖或复现资源。

- `GLEAM/`：最直接、最重要的 baseline。用户当前大量实验是在与 GLEAM 对齐协议和结果，尤其是 GLEAM Table 3 的 MLLM CLIP-proxy ASR、Table 4 的防御下 ALBEF ITR 结果。`GLEAM/NURBSAttacker.py` 有本地改动，但整体仍应视为 baseline/对照方法，不是用户的新方法主体。
- `SA-AET`：不是单独目录，但作为论文和报告里的 baseline 数字出现，常与 GLEAM 一起在 MLLM 表中对齐。
- `TransferAttack/`：图像分类迁移攻击框架，属于外部方法库/参考代码。除非用户明确要求，不要在这里实现 Syner-Attack 主线逻辑。
- `OpenAttack/`：文本攻击工具库，属于外部依赖/参考库。Syner-Attack 的文本攻击思想与 BERT-Attack/OpenAttack 相关，但 `OpenAttack/` 本身不是当前论文主线。
- `BLIP/`：VLP/MLLM baseline 或模型参考。README 标明该仓库 deprecated。可用于补 BLIP/BLIP-2 相关 baseline，但不要把它当用户当前方法。
- `hf_models/`：本地模型权重缓存，用于跑开源 MLLM 和 CLIP proxy，包括 Qwen2-VL、LLaVA、CLIP、InstructBLIP 等。只读使用，不能当代码仓库修改。
- `mllm-data/`：用户另一个数据/评测工程项目，主要处理 MLLM 数据生成、关键帧、交互体验、bad case 分析等。它不是 Syner-Attack 论文主线；只有当任务涉及数据处理、JSONL/图片/视频评测数据时才进入。
- `feishu_todo_bot/`：个人飞书 TODO 机器人后端，是独立工具项目，不属于 Syner-Attack 研究主线。

## 用户在做什么

从当前文件和实验产物看，用户正在做一轮论文返修式工作：

1. 用 `Syner-Attack` 生成新的对抗样本，并同时生成/整理 GLEAM 对照样本。
2. 在 MSCOCO sample100/sample500 上跑 ALBEF 检索、防御、MLLM CLIP-proxy 评估。
3. 把开源 MLLM 从定性案例升级为定量表格，目标包括 LLaVA-1.5-7B、Qwen2-VL-7B-Instruct、InstructBLIP-Vicuna-7B。
4. 尽量按 GLEAM 论文协议对齐，避免 reviewer 继续质疑比较不公平。
5. 产出论文可直接使用的 LaTeX table、中文/英文结果总结、rebuttal 或 revision 文案。
6. 对“协同双流攻击”的有效性补证据：image-only vs image-and-text、VGA 消融、alignment disruption、计算成本、防御鲁棒性、语义保持。

因此，Agent 在回答和改代码时应主动围绕“这能否支撑返修/回应审稿人”判断优先级。

## 当前优先级

优先级来自 `Syner-Attack/revision_guide.md`：

- P0：现代/商业 MLLM 定量评估；技术新颖性论证；跨模态解耦的严格定义。
- P1：比较公平性澄清；计算开销比较；防御评估。
- P2：扰动预算、VGA、IG 步数、协同效应消融；语义保持验证；新增 stronger/recent baselines。
- P3：威胁模型讨论；跨任务双向迁移。

如果用户没有明确说明任务，优先假设是在推进 P0/P1 的返修证据，而不是做无关重构。

## Git 与改动边界

根目录本身不是 git 仓库，以下子目录是独立 git 仓库：

- `Syner-Attack/`：当前主线，已有大量未跟踪实验脚本、数据、报告和修改。编辑前必须先看目标文件。
- `GLEAM/`：baseline，有本地改动和模型缓存。不要随意重置。
- `mllm-data/`：独立数据项目，有未提交改动。
- `TransferAttack/`、`OpenAttack/`、`BLIP/`、`external_repos/NRP/`、`playground/merlin-demo/`：外部/辅助仓库。

开始任何改动前，在目标子仓库运行：

```bash
git status --short
```

不要回滚、覆盖或清理未确认属于自己的改动。不要删除 `__pycache__/`、`checkpoints/`、`data/`、`test_outputs/` 等生成物，除非用户明确要求。

## 环境与常用命令

不同项目依赖栈不一致，不要在整个工作区统一安装依赖。进入具体项目后再安装或运行。

### Syner-Attack

主线工作优先进入这里。常用入口在根目录和 `scripts/`：

```bash
cd /mlx_devbox/users/yutianhao/Syner-Attack
python eval.py
python eval_defense.py
python eval_mllm_attack.py
bash scripts/run_fresh_smoke_worker.sh
bash scripts/run_fresh_500_worker.sh
bash scripts/run_mllm_llava_smoke_worker.sh
```

具体参数和路径常写在脚本内。修改前先读目标脚本，不要假设默认数据路径可用。涉及论文结果时优先查看 `test_outputs/` 下是否已有对应 sample100/sample500 产物，避免重复跑昂贵任务。

### GLEAM

GLEAM 是 baseline/对照方法。只在需要复现、对齐协议或生成对照样本时进入。

README 中建议：

```bash
cd /mlx_devbox/users/yutianhao/GLEAM
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

运行前检查 `configs/*.yaml` 中的数据根目录、checkpoint 路径和模型路径，尤其是 Flickr30K/MSCOCO、ALBEF/TCL/CLIP 相关配置。

### TransferAttack

README 中的典型命令：

```bash
cd /mlx_devbox/users/yutianhao/TransferAttack
pip install -r requirements.txt
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet50 --attack mifgsm --model=resnet50
python main.py --input_dir ./path/to/data --output_dir adv_data/mifgsm/resnet50 --eval
```

依赖中固定了 CUDA 11.6 版 PyTorch。不要和 GLEAM 的 CUDA 12.1 / PyTorch 2.1 环境混用。

### mllm-data

README 中建议使用 Poetry：

```bash
cd /mlx_devbox/users/yutianhao/mllm-data
pip install poetry
poetry install
poetry run python data_generate/interactive_data_generate/generate_interactive_data.py --model doubao
poetry run python data_generate/interactive_data_generate/eval_for_gpt_result.py --test_type element_overlap_base --model doubao
```

运行前通常需要设置模型 API key，例如 Azure OpenAI、豆包、Gemini 等。不要把密钥写入代码或文档。常见系统依赖问题：`ImportError: libGL.so.1` 可通过安装 `libgl1-mesa-glx` 解决。

### feishu_todo_bot

FastAPI 服务，依赖在 `requirements.txt`：

```bash
cd /mlx_devbox/users/yutianhao/feishu_todo_bot
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

接口：

- `GET /healthz`
- `POST /webhook/feishu/events`

配置来自 `.env`，参考 `.env.example`。真实 `.env` 可能包含飞书应用 secret、verification token、encrypt key、LLM key、Bitable token 等，不要打印或提交。

## 数据、模型和产物注意事项

- `hf_models/` 保存本地 Hugging Face 模型权重，如 Qwen2-VL、LLaVA、CLIP、InstructBLIP；这些文件很大，通常只读使用。
- `GLEAM/checkpoints/`、`BLIP/checkpoints/`、`Syner-Attack/bert-base-uncased`、`Syner-Attack/data/` 等是实验依赖或产物，不要随意删除。
- 许多脚本依赖 `/mnt/bn/...`、ByteNAS、HDFS 或 Merlin 开发机环境。运行前确认路径存在，必要时先做小样本 smoke test。
- 处理 JSONL、Excel、图片、视频和评测输出时，优先保留原始文件，新增转换结果或中间文件应放到已有 `outputs/`、`test_outputs/`、`tmp/` 等目录。
- 不要把大模型权重、数据集、二进制产物加入 git。

## 编码约定

- Python 代码优先保持项目现有风格；不要跨项目做统一重构。
- `OpenAttack/` 使用 Black，行宽 88。
- `mllm-data/` 是脚本型数据项目，尽量保持函数边界清楚、路径参数显式、不要把私有路径硬编码到通用函数中。
- `feishu_todo_bot/` 使用 FastAPI + Pydantic 配置，新增配置应走 `app/config.py` 和 `.env.example`，不要散落 `os.getenv`。
- 对研究脚本的修改要尽量可复现：记录关键参数、数据路径、模型名、输出路径。
- 对长时间/GPU/外部 API 任务，先运行小样本或 dry run，避免直接全量启动。

## 安全与隐私

- 不要泄露 `api.md`、`.env`、命令历史或日志中的 token/key。
- 不要在最终回复里粘贴密钥，即使用户文件里已经存在。
- 涉及飞书、豆包、Azure、Gemini、OpenRouter 等 API 时，只说明需要设置哪些环境变量，不展示真实值。
- 处理内部 ByteDance / Merlin / ByteNAS 路径时，不要把敏感数据内容外传；只总结结构和必要路径。

## Agent 工作方式

1. 先判断目标项目，不要默认根目录是仓库。
2. 读取 README、配置文件、入口脚本和当前 git 状态后再改。
3. 修改前确认是否已有用户改动；如果同一文件里有用户改动，要在其基础上最小化编辑。
4. 优先用 `rg` 搜索，用项目已有脚本和配置。
5. 对代码改动，能运行轻量检查就运行；涉及大模型或 GPU 的任务先给出 smoke test 或明确说明未全量验证。
6. 最终回复要说明改了哪些文件、跑了哪些检查、哪些检查因为环境/数据/成本没有运行。
