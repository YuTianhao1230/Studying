# 推理工程

本目录整理大模型推理和部署优化方法，重点理解线上服务中的吞吐、延迟、显存、批处理和底层执行优化。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [推理框架总览.md](<推理框架总览.md>) | 推理框架生态和选型总览。 |
| [模型部署与推理工程.md](<模型部署与推理工程.md>) | 模型部署链路、服务化和推理工程关键问题。 |
| [Serving.md](<Serving.md>) | 模型 serving 的请求处理、扩缩容和线上稳定性。 |
| [vLLM.md](<vLLM.md>) | vLLM、PagedAttention、continuous batching 和服务接口。 |
| [TensorRT_LLM.md](<TensorRT_LLM.md>) | TensorRT-LLM 的编译优化和高性能推理。 |
| [KV_Cache与Prefill_Decode.md](<KV_Cache与Prefill_Decode.md>) | KV Cache、prefill/decode 阶段和显存吞吐瓶颈。 |
| [Batching.md](<Batching.md>) | 静态 batching、dynamic batching、continuous batching 的区别。 |
| [Speculative_Decoding.md](<Speculative_Decoding.md>) | 推测解码用小模型加速大模型生成的机制。 |
| [量化.md](<量化.md>) | 权重量化、KV 量化和低比特推理的收益与风险。 |
| [CUDA_Graph.md](<CUDA_Graph.md>) | CUDA Graph 降低 launch overhead 的原理和适用条件。 |
| [CUDA与Triton基础.md](<CUDA与Triton基础.md>) | CUDA/Triton 算子开发和 GPU 编程基础。 |
| [算子.md](<算子.md>) | 算子概念、融合和性能优化基础。 |
| [推理优化方法_并行策略.md](<推理优化方法_并行策略.md>) | 推理中的张量并行、流水并行、批处理和缓存优化。 |
| [ms-swift.md](<ms-swift.md>) | ModelScope ms-swift 在微调、推理和评测中的使用定位。 |

## 学习路线

1. 先看 [模型部署与推理工程.md](<模型部署与推理工程.md>)、[Serving.md](<Serving.md>) 和 [推理框架总览.md](<推理框架总览.md>)。
2. 再看 [KV_Cache与Prefill_Decode.md](<KV_Cache与Prefill_Decode.md>)、[Batching.md](<Batching.md>) 和 [vLLM.md](<vLLM.md>)。
3. 接着看 [量化.md](<量化.md>)、[Speculative_Decoding.md](<Speculative_Decoding.md>)、[TensorRT_LLM.md](<TensorRT_LLM.md>)。
4. 最后看 [CUDA与Triton基础.md](<CUDA与Triton基础.md>)、[CUDA_Graph.md](<CUDA_Graph.md>) 和 [算子.md](<算子.md>)，理解底层性能优化。
