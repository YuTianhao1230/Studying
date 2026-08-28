# Beam Search

## 知识点解析

### 概述

Beam Search 是自回归生成任务中的一种解码算法。它不像 Greedy Search 那样每一步只保留当前概率最高的一个 token，而是每一步保留 `beam size` 个累计得分最高的候选序列，再继续向后扩展。

它的核心思想是：**用有限宽度的搜索，缓解逐 token 贪心选择带来的局部最优问题。**

### 背景与使用场景

自回归模型生成序列时，理论上需要在所有可能序列中找到概率最大的输出，但完整搜索空间会随着序列长度指数级增长，无法穷举。

Greedy Search 每一步只选概率最高的 token，计算便宜，但早期选错后无法回退。Beam Search 在效率和搜索质量之间折中，每一步保留多个候选路径，适合目标相对确定、希望输出稳定的任务，例如机器翻译、摘要、结构化生成。

对于开放式对话、创意写作、多样化生成，Beam Search 不一定合适，因为它倾向于选择高概率、保守、模板化的输出。这类场景通常更常用 top-k、top-p 或 temperature sampling。

### 方法原理

假设 `beam size = k`。生成时，Beam Search 会维护 k 个候选序列。每一步对每个候选序列扩展下一个 token，计算扩展后的累计 log probability，然后从所有扩展结果中选出得分最高的 k 个继续保留。

因为 log probability 会随着序列变长不断累加，Beam Search 经常需要 length penalty 或长度归一化，避免模型系统性偏向过短或过长的序列。

常见打分形式可以理解为：候选序列得分等于 token log probability 的累积，再根据长度惩罚做校正。最终输出通常选择完成序列中得分最高的一个。

### 优势与局限

Beam Search 的优势是比 Greedy Search 更稳，能保留多个可能路径，降低局部最优风险。在翻译、摘要等确定性较强任务中，它常能带来更高质量输出。

它的限制也明显。beam size 越大，计算和显存开销越高；过大的 beam 可能让输出变得保守、重复、缺乏多样性；如果模型本身概率分布有偏，Beam Search 也可能放大这种偏差。因此实际使用时需要调 beam size、length penalty、重复惩罚等参数，并结合任务指标验证。

## 面试应对

### Beam Search 的基本原理是什么？

回答思路：先和 greedy 对比，再讲 beam size、候选扩展和分数保留。

回答模板：

Beam Search 是一种自回归解码算法。Greedy 每一步只保留当前概率最高的一个 token，而 Beam Search 每一步保留 beam size 个累计得分最高的候选序列，再继续扩展。这样可以减少局部最优风险，但计算成本更高，也可能让生成结果更保守。

### Beam Search 为什么可能比 Greedy 更好？

回答思路：说明 greedy 的局部最优问题，再说明 beam 保留多个路径。

回答模板：

Greedy 每一步做局部最优选择，早期一旦选错，后面无法恢复。Beam Search 同时保留多个候选路径，可以让某些前期概率略低但整体更好的序列在后续反超，因此在翻译、摘要等任务中常比 greedy 更稳。

### Beam size 越大越好吗？

回答思路：从搜索空间、计算成本和生成质量三个角度回答。

回答模板：

不是。beam size 变大可以扩大搜索空间，但计算和显存成本也会上升，而且过大的 beam 可能生成更模板化、更保守的答案。实际要根据任务调参，翻译和摘要可能适合 beam search，开放对话常常更适合 top-p 或 temperature sampling。

### 为什么 Beam Search 需要 length penalty？

回答思路：说明 log probability 累加会受序列长度影响。

回答模板：

Beam Search 通常累加每个 token 的 log probability，序列长度会影响总分。如果不做长度归一化，模型可能偏向过短或过长的序列。length penalty 用来校正长度偏好，让不同长度候选更公平比较。

### Beam Search 适合哪些生成任务？

回答思路：区分确定性任务和开放式生成。

回答模板：

Beam Search 适合目标相对明确、希望输出稳定高概率结果的任务，比如机器翻译、摘要、结构化生成。它不一定适合开放式聊天或创意写作，因为这类任务更需要多样性，通常会使用 top-k、top-p 或 temperature sampling。
