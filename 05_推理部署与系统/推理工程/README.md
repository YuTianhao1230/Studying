# 推理工程

本目录整理大模型推理和部署优化方法，重点理解线上服务中的吞吐、延迟、显存、批处理和底层执行优化。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [模型部署与推理工程.md](<模型部署与推理工程.md#模型部署与推理工程>) | 模型部署链路、服务化和推理工程关键问题。 |
| [Serving.md](<Serving.md#serving>) | 模型 serving 的请求处理、扩缩容和线上稳定性。 |
| [推理框架总览.md](<推理框架总览.md#推理框架总览>) | 推理框架生态和选型总览。 |
| [ms-swift.md](<ms-swift.md#ms-swift>) | ModelScope ms-swift 在微调、推理和评测中的使用定位。 |
| [KV_Cache与Prefill_Decode.md](<KV_Cache与Prefill_Decode.md#kvcache与prefilldecode>) | KV Cache、prefill/decode 阶段和显存吞吐瓶颈。 |
| [Batching.md](<Batching.md#batching>) | 静态 batching、dynamic batching、continuous batching 的区别。 |
| [vLLM.md](<vLLM.md#vllm>) | vLLM、PagedAttention、continuous batching 和服务接口。 |
| [量化.md](<量化.md#量化>) | 权重量化、KV 量化和低比特推理的收益与风险。 |
| [Speculative_Decoding.md](<Speculative_Decoding.md#speculativedecoding>) | 推测解码用小模型加速大模型生成的机制。 |
| [TensorRT_LLM.md](<TensorRT_LLM.md#tensorrtllm>) | TensorRT-LLM 的编译优化和高性能推理。 |
| [推理优化方法_并行策略.md](<推理优化方法_并行策略.md#推理优化方法并行策略>) | 推理中的张量并行、流水并行、批处理和缓存优化。 |
| [CUDA与Triton基础.md](<CUDA与Triton基础.md#cuda与triton基础>) | CUDA/Triton 算子开发和 GPU 编程基础。 |
| [CUDA_Graph.md](<CUDA_Graph.md#cudagraph>) | CUDA Graph 降低 launch overhead 的原理和适用条件。 |
| [算子.md](<算子.md#算子>) | 算子概念、融合和性能优化基础。 |
