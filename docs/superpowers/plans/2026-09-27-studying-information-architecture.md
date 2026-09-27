# Studying Information Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让每个通用知识点在仓库中只有一个权威正文，并通过索引、面试、项目和笔试视图复用。

**Architecture:** 保留13个编号一级目录，收紧01至06的知识所有权；优先解决强化学习、损失/激活、训练优化与数据实验的重复归属。07、12、13等视图与应用目录只引用权威知识卡，不复制定义。

**Tech Stack:** Markdown、相对链接、Git、Python标准库校验脚本

**Spec:** `docs/superpowers/specs/2026-09-27-studying-information-architecture-design.md`

## Global Constraints

- 不丢失现有知识：旧卡独有内容必须先迁入权威卡，再删除旧路径。
- 保留现有13个一级目录和编号。
- 不修改当前未提交的`13_笔试`内容，除非仅更新因本次迁移产生的链接。
- 唯一所有权：一个概念只保留一张权威正文；README、面试题、项目卡和笔试题只提供导航或场景差异。
- 使用`apply_patch`编辑正文；批量文件移动与机械链接替换可用Shell。
- 每个任务结束后检查局部链接，最终检查全库链接、孤立卡和`git diff --check`。

---

### Task 1: 建立迁移清单与内容保护基线

**Files:**
- Modify: `.planning/2026-09-27-untitled-b4678278/findings.md`
- Modify: `.planning/2026-09-27-untitled-b4678278/progress.md`

**Interfaces:**
- Consumes: 已批准的信息架构设计和当前工作区。
- Produces: 旧路径到新路径映射、候选卡章节清单、基线文件数与链接数。

- [ ] **Step 1: 记录目标文件的标题、章节和引用方**

运行`rg '^#{1,4} '`及全库Markdown链接检索，覆盖强化学习、损失函数、激活函数、LSTM、混合职责卡和训练总览。

- [ ] **Step 2: 建立内容迁移检查表**

逐节记录来源、目标和处理方式：原文保留、合并改写或只保留链接。任何没有目标的章节阻止删除源文件。

- [ ] **Step 3: 记录基线**

记录Markdown文件数、本地链接数、断链、孤立卡及当前`git status --short`。

---

### Task 2: 合并强化学习基础

**Files:**
- Modify: `01_机器学习基础/概率与序列决策/强化学习基础.md`
- Modify: `01_机器学习基础/概率与序列决策/README.md`
- Delete: `03_训练优化与对齐/后训练与对齐/RL 强化学习基础.md`
- Modify: `03_训练优化与对齐/后训练与对齐/README.md`
- Modify: `03_训练优化与对齐/README.md`
- Modify: `学习路线总览.md`
- Modify: 所有引用旧RL路径的Markdown文件

**Interfaces:**
- Consumes: 01中的Bandit、动态规划、SARSA、Q-learning、DQN；03中的Return、V/Q/A、策略梯度、baseline、TD/GAE、采样校正和终止边界。
- Produces: `01_机器学习基础/概率与序列决策/强化学习基础.md`作为唯一RL基础入口。

- [ ] **Step 1: 合并两张RL卡**

以03的详细推导为主体，补入01独有的Bandit、动态规划、SARSA、Q-learning和DQN内容；去除卡内指向自身旧副本的链接。

- [ ] **Step 2: 更新所有引用**

将PPO、GRPO、RLHF、学习路线和README中的旧路径统一指向01权威卡。

- [ ] **Step 3: 删除03重复卡并验证**

确认源卡所有章节已映射后删除，运行局部链接检查和关键词检查。

---

### Task 3: 归位通用神经网络组件

**Files:**
- Move: `03_训练优化与对齐/参数/常见分类损失函数.md` → `01_机器学习基础/深度学习基础/常见分类损失函数.md`
- Move: `03_训练优化与对齐/参数/常见回归损失函数.md` → `01_机器学习基础/深度学习基础/常见回归损失函数.md`
- Move: `03_训练优化与对齐/参数/常见激活函数.md` → `01_机器学习基础/深度学习基础/常见激活函数.md`
- Move: `03_训练优化与对齐/参数/GeLU.md` → `01_机器学习基础/深度学习基础/GeLU.md`
- Move: `02_大模型/基础架构/LSTM.md` → `01_机器学习基础/深度学习基础/LSTM.md`
- Rename: `03_训练优化与对齐/参数/` → `03_训练优化与对齐/超参数与优化器/`
- Modify: 相关README及所有引用方

**Interfaces:**
- Consumes: 当前五张通用组件卡及03参数目录。
- Produces: 01拥有通用网络组件，03只拥有训练参数和优化器。

- [ ] **Step 1: 移动通用组件**

保持正文内容不变地移动五张卡，随后修正卡内相对链接。

- [ ] **Step 2: 重命名03参数目录**

保留训练超参数、SFT参数和Optimizer三张权威卡，更新README标题与说明。

- [ ] **Step 3: 批量修正引用并验证**

更新01、02、03、05、07和根索引中的引用；验证旧路径命中为零且新链接可达。

---

### Task 4: 拆解01的优化、数据与实验混合职责

**Files:**
- Create: `01_机器学习基础/深度学习基础/参数初始化与数值稳定性.md`
- Modify: `03_训练优化与对齐/超参数与优化器/Optimizer 优化器.md`
- Modify: `03_训练优化与对齐/训练稳定性/Loss异常与收敛排查.md`
- Modify: `04_评测实验与数据质量/数据工程与数据质量.md`
- Modify: `04_评测实验与数据质量/指标与统计计算.md`
- Modify: `04_评测实验与数据质量/模型评测与实验设计.md`
- Delete: `01_机器学习基础/优化、数据与实验方法/训练优化、数据质量与实验设计.md`
- Delete: `01_机器学习基础/优化、数据与实验方法/README.md`
- Modify: `01_机器学习基础/README.md`
- Modify: `04_评测实验与数据质量/README.md`

**Interfaces:**
- Consumes: 混合卡中的优化、稳定性、数据、校准和实验章节。
- Produces: 各章节进入01、03、04的唯一权威卡，不保留混合正文。

- [ ] **Step 1: 迁移基础内容**

将Xavier/He初始化、稳定Softmax、交叉熵数值边界整理到新卡；保留公式、适用条件和易错点。

- [ ] **Step 2: 补齐03训练内容**

核对Optimizer及Loss排查卡是否完整覆盖SGD/AdamW、学习率、warmup、gradient clipping、混合精度和NaN诊断；只补缺失内容。

- [ ] **Step 3: 补齐04数据与实验内容**

分别迁移特征工程/泄漏/不平衡、校准/阈值、A-B/SRM/CUPED/多重检验，避免同一段同时进入多张卡。

- [ ] **Step 4: 删除混合卡**

逐节检查表全部完成后删除混合目录，并更新01、03、04索引。

---

### Task 5: 归位领域笔试训练

**Files:**
- Create: `03_训练优化与对齐/笔试训练/README.md`
- Create: `03_训练优化与对齐/笔试训练/训练优化与稳定性训练.md`
- Create: `04_评测实验与数据质量/笔试训练/README.md`
- Move: `01_机器学习基础/笔试训练/数据质量与实验设计训练.md` → `04_评测实验与数据质量/笔试训练/数据质量与实验设计训练.md`
- Modify: `01_机器学习基础/笔试训练/深度学习与训练优化训练.md`
- Modify: `01_机器学习基础/笔试训练/README.md`
- Modify: `03_训练优化与对齐/README.md`
- Modify: `04_评测实验与数据质量/README.md`

**Interfaces:**
- Consumes: 01中混合了基础组件与训练工程的题集。
- Produces: 每个题集只考对应知识目录，答案链接权威卡。

- [ ] **Step 1: 移动数据与实验题集**

保持题面和答案不变，移动至04并修正链接。

- [ ] **Step 2: 拆分深度学习与训练优化题集**

MLP、反向传播、激活、归一化、Dropout、初始化和基础数值题留在01；AdamW、warmup、梯度裁剪、混合精度、NaN和梯度累积题移入03。

- [ ] **Step 3: 更新三个训练索引并验证题号**

为拆分后的题集重新连续编号，逐题核对答案与解析，不改变知识结论。

---

### Task 6: 收敛02和03的跨章总览

**Files:**
- Rename: `02_大模型/大模型预训练与推理基础.md` → `02_大模型/大模型预训练与生成基础.md`
- Modify: `02_大模型/README.md`
- Modify: `03_训练优化与对齐/模型训练学习手册_预训练到后训练.md`
- Modify: `03_训练优化与对齐/README.md`
- Delete: `03_训练优化与对齐/训练优化与大模型训练.md`
- Modify: `07_面试体系/专业知识面试题（算法）.md`
- Modify: `README.md`
- Modify: `学习路线总览.md`

**Interfaces:**
- Consumes: 02的预训练/生成机制总览，03的训练路线与训练工程总览。
- Produces: 02回答“模型如何形成和生成”，03回答“训练项目如何执行”，根学习路线负责跨章顺序。

- [ ] **Step 1: 收紧02总览**

保留Tokenizer、预训练目标、数据配比、Scaling Law、长上下文、生成采样和推理时扩展；运行时框架、调度和部署只链接05。

- [ ] **Step 2: 将03长手册改为路线导航**

按任务定义、底座选择、数据、SFT、偏好、RLVR、评测、部署组织入口；删除重复正文前，逐节确认02至05存在对应权威卡。

- [ ] **Step 3: 合并短训练总览**

把知识入口并入03 README，把三组面试问答迁入07算法面试索引，然后删除短总览卡。

---

### Task 7: 全仓库索引与完整性验证

**Files:**
- Modify: `README.md`
- Modify: `学习路线总览.md`
- Modify: 各受影响目录README及引用文件

**Interfaces:**
- Consumes: Tasks 2至6产生的最终路径。
- Produces: 与实际目录一致的仓库地图和无断链导航。

- [ ] **Step 1: 更新根目录地图和职责说明**

补齐01的实际子目录、02笔试训练、03超参数与优化器及03/04笔试训练。

- [ ] **Step 2: 验证内容保护**

对所有被删除源文件生成章节映射；确认每个独有章节已进入目标文件，所有原知识卡均有去向。

- [ ] **Step 3: 验证链接与索引**

解析全仓库Markdown相对链接，要求本次影响范围断链为0；检查所有非README卡片至少被一个README或导航卡引用。

- [ ] **Step 4: 验证重复与格式**

扫描跨一级目录同名卡、高相似正文和旧路径；运行`git diff --check`并审阅`git status --short`。

- [ ] **Step 5: 提交实现**

仅在全部检查通过后提交本次信息架构改动；不夹带`13_笔试`此前未提交内容。
