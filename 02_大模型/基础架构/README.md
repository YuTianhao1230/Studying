# 基础架构

本目录整理大模型底层结构和关键模块，重点是理解为什么当前 LLM 多采用 Decoder-only Transformer，以及 RoPE、GQA、MoE 等结构如何影响训练和推理。

循环序列建模的对照基础见 [LSTM](<../../01_机器学习基础/深度学习基础/LSTM.md>)；KV Cache、Prefill/Decode 等运行时机制归入 [05_推理部署与系统](<../../05_推理部署与系统/README.md>)。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [Transformer.md](<Transformer.md>) | Transformer 总体结构、Attention、FFN、残差和归一化。 |
| [Self-Attention.md](<Self-Attention.md>) | 自注意力机制、QKV、复杂度和长上下文瓶颈。 |
| [Autoregressive Model.md](<Autoregressive Model.md>) | 自回归建模、next-token prediction 和生成式解码。 |
| [Decoder-only vs Encoder-Decoder.md](<Decoder-only vs Encoder-Decoder.md>) | Decoder-only 架构在生成式大模型中的优势和取舍。 |
| [Pre-Norm vs Post-Norm.md](<Pre-Norm vs Post-Norm.md>) | Pre-Norm/Post-Norm 对深层训练稳定性和收敛的影响。 |
| [RMSNorm.md](<RMSNorm.md>) | RMSNorm 与 LayerNorm 的区别及训练效率影响。 |
| [RoPE.md](<RoPE.md>) | 旋转位置编码的原理、外推和长上下文影响。 |
| [GQA.md](<GQA.md>) | MHA、MQA、GQA 的结构差异，以及 GQA 如何降低 KV Cache 和提升推理吞吐。 |
| [MLA.md](<MLA.md>) | Multi-head Latent Attention 的压缩 KV 表示和推理优化思路。 |
| [Dense Model.md](<Dense Model.md>) | 稠密模型的含义，以及与 MoE 的结构和成本差异。 |
| [MoE.md](<MoE.md>) | 专家混合模型的稀疏激活、路由和训练/推理权衡。 |
