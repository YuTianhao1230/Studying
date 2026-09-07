# NLP与大语言模型

## 知识点解析

### 概述

NLP 与大语言模型关注机器如何表示、理解和生成自然语言，核心链路从文本清洗、Tokenization、Embedding 和序列建模，发展到预训练、指令微调、对齐与生成。面试中需要说明传统 NLP 与 LLM 的能力边界、训练目标和评测方式，而不是只罗列模型名称。

#### 高频问题

1. Tokenization 是什么？BPE、WordPiece、SentencePiece 有什么区别？
2. Transformer 的 Encoder、Decoder、Encoder-Decoder 架构分别适合什么任务？
3. Self-Attention 的计算过程是什么？为什么要除以 `sqrt(d_k)`？
4. Multi-Head Attention 为什么有效？
5. 位置编码有什么作用？绝对位置编码、RoPE、ALiBi 有什么区别？
6. GPT、BERT、T5 的训练目标和适用场景有什么不同？
7. 语言模型的预训练目标是什么？Causal LM 和 Masked LM 区别是什么？
8. 大模型为什么会出现幻觉？如何缓解？
9. Prompt Engineering、Instruction Tuning、SFT、RLHF、DPO 分别是什么？
10. RAG 的基本流程是什么？如何评估和优化 RAG？
11. Function Calling / Tool Use 的核心难点是什么？
12. 多模态大模型如何处理图像、视频和文本？

### 核心知识点

- NLP 基础：分词、词向量、语言模型、序列标注、文本分类、生成任务。
- Transformer：Q/K/V、Attention Score、Mask、残差连接、FFN、Norm、位置编码。
- 模型家族：BERT、GPT、T5、LLaMA、Qwen、Mistral、DeepSeek 等。
- 训练阶段：预训练、SFT、偏好学习、RLHF、DPO、拒绝采样、蒸馏。
- 推理技术：temperature、top-k、top-p、beam search、repetition penalty、KV cache。
- 长上下文：位置外推、RoPE scaling、Attention 优化、上下文压缩。
- RAG：query rewrite、retrieval、rerank、context packing、generation、citation、faithfulness。
- Agent：规划、工具调用、状态管理、记忆、反思、任务分解、安全边界。
- 多模态：视觉编码器、投影层、图文对齐、视频采样、时序建模。

#### 回答要点

- Attention 通过 Q 和 K 的相似度得到权重，再对 V 加权求和。
- 除以 `sqrt(d_k)` 是为了控制点积方差，避免 softmax 输入过大导致梯度过小。
- 多头注意力允许模型在不同子空间关注不同关系，例如局部、全局、语义、位置关系。
- GPT 是自回归生成模型，BERT 是双向表征模型，T5 将任务统一为 text-to-text。
- 幻觉来源包括训练分布缺陷、解码随机性、知识过期、上下文冲突、模型缺少不确定性表达。
- RAG 的瓶颈常在检索召回、切片粒度、排序质量、上下文污染和答案忠实性。

### 常见追问

- KV cache 为什么能加速自回归推理？
- RoPE 为什么适合相对位置信息建模？
- SFT 和 DPO 的数据格式有什么区别？
- 如何判断一个 RAG 错误是检索错还是生成错？
- 多模态模型中图像 token 数过多会带来什么问题？

## 面试应对

### GPT、BERT 和 T5 有什么区别？

回答思路：从结构、训练目标和任务形态回答。

回答模板：

BERT 是 Encoder-only 模型，使用双向上下文和掩码语言模型预训练，适合理解、分类和抽取；GPT 是 Decoder-only 模型，使用因果掩码做下一个 Token 预测，天然适合生成；T5 是 Encoder-Decoder 模型，把不同任务统一成文本到文本，适合翻译、摘要等条件生成。当前通用大模型多采用 Decoder-only，主要因为训练目标统一、规模化简单，并能通过提示把多种任务转成生成。

### RAG 问题如何区分是检索错还是生成错？

回答思路：分阶段保存结果并分别评测。

回答模板：

我会先检查正确证据是否出现在召回结果中；没有出现说明是解析、切片、Embedding 或召回覆盖问题。正确证据已召回但排序靠后，则是 Rerank 或上下文截断问题。证据已经进入最终上下文，模型仍回答错误，才主要属于生成、指令遵循或上下文冲突问题。评测时要分别记录 Recall@K、排序指标、引用支持率和最终答案正确率。

### Tokenizer 为什么会影响模型能力和成本？

回答思路：从序列长度、词表参数和不同领域表示回答。

回答模板：

Tokenizer 决定文本如何被切成离散 Token，同一段内容切得越碎，序列越长，训练和推理成本越高；词表越大，Embedding 和输出层参数也越多。它还会影响数字、代码、多语言和专业术语的表示效率。因此 Tokenizer 需要在词表规模、序列长度和领域覆盖之间折中，并保持训练、微调和推理阶段一致。
