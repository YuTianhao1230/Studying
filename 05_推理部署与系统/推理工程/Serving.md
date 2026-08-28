# Serving

## 一句话解释

Serving 是把模型能力部署成可被业务稳定调用的在线服务，包括接口、扩缩容、监控、限流、灰度、回滚和故障处理。

## Serving 和推理有什么区别

推理是模型计算：

```text
输入 -> 模型 -> 输出
```

Serving 是线上服务系统：

```text
用户请求
  -> 网关
  -> 权限 / 限流
  -> 模型服务
  -> 监控 / 日志
  -> 返回结果
```

推理关注模型怎么算，Serving 关注服务怎么稳定地对外提供能力。

## Serving 系统通常包括什么

- API Server：HTTP、gRPC、OpenAI-compatible API。
- Model Runtime：vLLM、TensorRT-LLM、xLLM 等推理运行时。
- Scheduler：请求调度和 batching。
- Load Balancer：负载均衡。
- Autoscaler：自动扩缩容。
- Monitor：延迟、吞吐、错误率、GPU 使用率监控。
- Logger：请求、响应、错误、trace。
- Config Manager：模型版本、参数、灰度配置。
- Guardrail：输入输出安全检查。

## 关键指标

- 可用性：服务是否稳定可调用。
- 延迟：P50、P95、P99。
- 吞吐：QPS、tokens/s。
- 成本：单请求成本、GPU 利用率。
- 质量：模型回答准确率、格式正确率、用户满意度。
- 稳定性：超时率、失败率、重试率。

## 常见线上问题

- 请求排队导致 P99 延迟升高。
- KV cache 占满导致 OOM。
- 上游请求量突增导致超时。
- 模型版本不一致导致输出异常。
- tokenizer 或 prompt 模板版本不匹配。
- 输出解析失败。
- 服务重启后冷启动耗时过长。

## 面试可能怎么问

1. 模型 Serving 系统怎么设计？
2. 推理服务如何做限流和降级？
3. 如何监控一个 LLM 服务？
4. P99 延迟突然升高怎么排查？
5. 如何做模型灰度发布和回滚？

