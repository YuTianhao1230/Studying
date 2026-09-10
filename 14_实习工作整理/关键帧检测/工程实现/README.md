# 工程实现

本目录处理训练、推理、数据格式和上线链路的工程细节。完整训练方案只引用结论，不重复展开这些实现参数。

| 文件 | 内容 |
| --- | --- |
| [Qwen3-VL关键帧数据格式.md](<Qwen3-VL关键帧数据格式.md>) | JSONL、`conversations`、`<video>`、`videos[0]`、`infos` 和标签格式。 |
| [ms-swift关键帧训练与推理.md](<ms-swift关键帧训练与推理.md>) | ms-swift、Qwen3.5、full SFT、DeepSpeed、推理分片和参数。 |
| [部署上线与容量评估.md](<部署上线与容量评估.md>) | 接口、下载瓶颈、并发压测、QPM 和线上回测。 |
