# Decoder-only vs Encoder-Decoder

## 直接回答

这是一个非常经典的问题。在早期的自然语言处理中，**Encoder-Decoder（如 [T5](<../模型细节/里程碑模型/T5.md>)）** 和 **Encoder-only（如 [BERT](<../模型细节/里程碑模型/BERT.md>)）** 非常流行；当前公开架构的许多通用生成式大模型则采用 **Decoder-only**。

为了让你听得最直白，我们还是用之前的“加工工序”和“写作”来打比方。

## 关键分析

### 结构上的本质区别

先看这两种“工厂”是怎么运作的：

*   **Encoder-Decoder（双塔结构）：**
    *   **Encoder（编码器）**：负责“理解”。它像一个翻译官，把输入的话（比如中文）全看完，揉碎了变成一堆复杂的语义向量。
    *   **Decoder（解码器）**：负责“生成”。它看着 Encoder 给出的语义，再一个词一个词地往外吐（比如翻译成英文）。
    *   **例子**：做中英文翻译、总结长文章。

*   **Decoder-only（单塔结构）：**
    *   只有 **Decoder**。它不分理解还是生成，它的任务只有一个：**根据目前看到的词，预测下一个词是什么。**
    *   **例子**：就像写小说，写了上文接下文。

### 为什么许多通用生成式大模型选择 Decoder-only？

#### ① **推理效率：输入编码与缓存范围不同**
两种架构都能利用缓存，差异不在于 Encoder-Decoder “不能缓存”。
*   **Decoder-only** 在 Prefill 阶段用同一套因果 Decoder 处理 Prompt，随后缓存各层已有 Prompt 和生成 token 的 K/V。生成新 token 时只需计算新位置，并让它读取此前全部 K/V。
*   **Encoder-Decoder** 先用双向 Encoder 把输入编码一次；生成阶段缓存 Decoder 已生成前缀的自注意力 K/V，同时通过 Cross-Attention 读取固定的 Encoder 输出。工程实现还可以预先计算并缓存 Cross-Attention 使用的源端 K/V，不必每步重跑 Encoder。
*   若输入长度为 $S$、输出长度为 $T$，忽略层数和常数后，Encoder-Decoder 的注意力交互可写成 $O(S^2+T^2+ST)$；Decoder-only 把输入和输出放入同一因果序列，整体与 $O((S+T)^2)$ 同阶。实际延迟取决于 Encoder/Decoder 层数、参数分配、输入输出长度和内核实现，不能仅凭存在 Cross-Attention 就断言计算必然大得多。
*   缓存占用也要按范围比较：Decoder-only 通常在每个 Decoder 层缓存 Prompt 与已生成 token；Encoder-Decoder 保存源端表示或源端 K/V，并在 Decoder 自注意力层缓存已生成 token。哪一种更省显存取决于具体结构和 $S:T$ 比例。

#### ② **任务的统一性：一切皆可接龙**
*   以前我们觉得：翻译是翻译，阅读理解是阅读理解。
*   但 [GPT](<../模型细节/里程碑模型/GPT.md>) 证明了：**所有的任务都可以转化成文字接龙**。
    *   问答任务：输入“问题是什么？”，接龙“答案是...”。
    *   翻译任务：输入“‘苹果’的英文是”，接龙“Apple”。
*   既然“文字接龙”能搞定一切，那我就不需要专门搞一个 Encoder 去费劲“理解”了。Decoder 在学的过程中，自然而然就学会了理解。

#### ③ **训练效率与规模化（Scaling Law）**
*   **Decoder-only 结构更简单**。在大规模分布式训练（用几万张显卡跑）的时候，结构越简单，越容易优化，出 Bug 的概率越小。
*   实验发现：在同样的算力下，**把所有的参数都堆给 Decoder**，比分出一半给 Encoder、一半给 Decoder，最后生成的效果往往更好。

#### ④ **低秩问题（更深层的学术原因）**
有一些研究认为（比如 Google 的研究）：
*   **Encoder-only/Encoder-Decoder** 在处理非常长的文本时，容易出现信息“退化”或者特征重叠（Rank Loss）。
*   **Decoder-only** 因为采用了“因果掩码”（Causal Masking，即每个词只能看左边，不能看右边），这种强制的约束反而让它在模型变大时，学习到的特征更有区分度。

### 直白总结：为什么它赢了？

我们用**写文章**来对比：

*   **Encoder-Decoder 像先读题再作答**：题目由 Encoder 完整阅读一次，Decoder 写每个词时都可以参考这份固定的输入表示。
*   **Decoder-only 像把题目和答案写在同一条续写序列中**：先处理完整 Prompt，再沿同一条因果序列继续生成。

**结果是：**
Decoder-only 用一套主干统一训练和生成，适合开放式生成、上下文学习和通用接口；Encoder-Decoder 对输入做双向编码并显式分离源端与目标端，在翻译、摘要等条件生成任务中仍有明确价值。架构选择应结合任务形式、参数预算、输入输出长度和部署实现，而不是把 Cross-Attention 本身视为必然更昂贵。

## 面试应对

### 为什么主流通用生成式大模型多采用 Decoder-only？

回答思路：先从自回归目标说明结构匹配，再讲任务统一、参数利用和工程扩展，最后补充这不是所有任务的唯一最优解。

回答模板：

主流通用生成式大模型多采用 Decoder-only，首先是因为因果自注意力与 next-token prediction 完全匹配，Prompt、示例和答案可以统一放进一个 token 序列中训练和生成。其次，模型只有一套主干，参数分配、数据组织和分布式扩展更直接，生成时可以复用 Prompt 与历史生成 token 的 KV Cache。Encoder-Decoder 也会一次性编码输入、缓存 Decoder 前缀，并可缓存 Cross-Attention 的源端 K/V；它的成本来自额外的 Encoder、Cross-Attention 及相应状态，但总成本和缓存大小取决于层数、参数分配及输入输出长度。对开放式生成和上下文学习，Decoder-only 通常更简洁；对输入输出边界明确的条件生成任务，Encoder-Decoder 仍可能更合适。

### Decoder-only 与 Encoder-Decoder 的数据流和训练目标有什么区别？

回答思路：用可见范围、信息流和损失位置三个维度比较。

回答模板：

Decoder-only 把输入和输出拼成同一序列，每个位置通过因果 Mask 只能读取当前位置及左侧 token，通常在目标位置计算 next-token loss。Encoder-Decoder 先用双向自注意力把完整输入编码成一组 source states，再由 Decoder 通过因果自注意力读取已生成前缀，并通过 Cross-Attention 读取 Encoder 输出。前者把理解和生成统一为续写，后者显式分离输入理解和条件生成，因此翻译、摘要等任务的源端和目标端边界更清楚。

### 什么场景更适合 Encoder-Decoder，Decoder-only 又有哪些边界？

回答思路：先给选型条件，再说明两类架构的代价，避免把流行度当成绝对优势。

回答模板：

当任务具有稳定且明确的“输入到输出”映射，例如机器翻译、摘要、文本改写或结构化转换时，Encoder-Decoder 可以先双向编码完整输入，再让 Decoder 专注生成，任务归纳偏置更直接。Decoder-only 更适合开放式生成、多轮对话、代码补全和需要 in-context learning 的统一接口，但它用因果方式处理整个上下文，未必能以最直接的方式获得双向输入表示，而且长 Prompt 仍会带来 Prefill 和 KV Cache 成本。因此实际选择要看任务形式、训练生态、延迟和部署约束，而不是认为 Decoder-only 在所有条件下都更优。
