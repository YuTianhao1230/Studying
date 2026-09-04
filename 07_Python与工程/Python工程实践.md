# Python 工程实践

## 知识点解析

### 概述

Python 工程实践关注如何把一次性实验脚本改造成可测试、可观测、可恢复的程序。核心包括并发模型选择、结构化日志、异常与重试、配置管理、单元测试、大文件处理和断点续跑。面试中应结合任务特点解释取舍，而不是只背库名。

### GIL 与并发模型

CPython 的 GIL 保证同一进程内同一时刻通常只有一个线程执行 Python 字节码。它简化了对象内存管理，但会限制纯 Python CPU 密集任务的多线程并行。

| 模型 | 适用场景 | 主要代价 |
| --- | --- | --- |
| 多线程 | 网络、磁盘等 IO 密集任务 | 共享状态需要同步，CPU 密集加速有限 |
| 多进程 | CPU 密集任务、故障隔离 | 进程启动、序列化和内存开销 |
| asyncio | 大量可异步等待的 IO 任务 | 依赖异步库，阻塞调用会卡住事件循环 |
| 批处理/向量化 | NumPy、PyTorch 等底层算子 | 需要重新组织数据和计算 |

许多 NumPy、PyTorch 和 C 扩展会在底层计算时释放 GIL，因此不能简单地认为“Python 多线程永远不能并行”。

### 线程池与超时

```python
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_io_tasks(items, worker, max_workers=8):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            try:
                results[item] = future.result(timeout=30)
            except Exception as exc:
                results[item] = {"error": str(exc)}
    return results
```

生产代码还需要限制提交队列、区分可重试异常、记录失败项，并避免多个线程无锁写同一个文件。

### 结构化日志

日志至少应包含时间、级别、任务 ID、样本 ID、阶段、耗时和错误类型。不要记录密钥、完整身份信息或不必要的原始数据。

```python
import json
import logging
import time


logger = logging.getLogger(__name__)


def process(item_id: str) -> None:
    started_at = time.monotonic()
    try:
        # business logic
        logger.info(json.dumps({
            "event": "item_completed",
            "item_id": item_id,
            "latency_ms": round((time.monotonic() - started_at) * 1000, 2),
        }))
    except Exception:
        logger.exception("item_failed", extra={"item_id": item_id})
        raise
```

使用 `logger.exception` 可以保留堆栈。库代码不应随意配置根 Logger，日志级别和 Handler 应由应用入口统一设置。

### 重试与幂等

只有瞬时错误适合重试，例如超时、限流和短暂网络故障。参数错误、权限错误和确定性业务错误重试没有意义。

```python
import random
import time


def retry(operation, attempts=4, base_delay=0.5):
    for attempt in range(attempts):
        try:
            return operation()
        except (TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay + random.uniform(0, delay * 0.1))
```

写操作重试必须配合幂等键、唯一约束或状态检查，否则一次超时可能导致重复写入。指数退避和随机抖动用于避免大量客户端同时重试。

### 配置与可复现

配置应与代码分离，并在任务开始时保存最终生效值：

- 命令行参数负责单次运行覆盖。
- 配置文件保存可复用实验配置。
- 环境变量保存密钥和部署差异。
- 代码记录 Git Commit 和依赖版本。
- 数据、模型和 Prompt 使用明确版本。
- 随机种子覆盖 Python、NumPy 和训练框架。

日志中可以记录环境变量名称，但不能记录密钥值。

### 大文件与流式处理

不要把大文件一次性读入内存。逐行读取并分批处理：

```python
def batched_lines(path: str, batch_size: int):
    batch = []
    with open(path, "r", encoding="utf-8") as source:
        for line in source:
            batch.append(line.rstrip("\n"))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

输出优先写临时文件，完成校验后再原子重命名。并发写入可采用单独 Writer、分片文件后合并，或使用支持事务的存储系统。

### 断点续跑

断点续跑需要明确任务粒度和完成状态。仅记录“处理到第几行”在输入顺序变化时并不可靠，优先使用稳定样本 ID。

```python
import json
from pathlib import Path


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as source:
        return {
            json.loads(line)["item_id"]
            for line in source
            if line.strip()
        }
```

可靠设计还应包含：

- 单条结果幂等写入。
- 成功状态与结果一起持久化。
- 失败项单独记录，支持定向重试。
- 输入版本变化时禁止错误复用旧进度。
- 多进程场景避免竞争写同一进度文件。

### 单元测试

测试优先覆盖纯函数、边界条件、异常路径和曾经出现的 bug：

```python
import pytest


def normalize_score(value: float) -> float:
    if not 0 <= value <= 100:
        raise ValueError("score must be in [0, 100]")
    return value / 100


def test_normalize_score():
    assert normalize_score(75) == 0.75


@pytest.mark.parametrize("value", [-1, 101])
def test_normalize_score_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        normalize_score(value)
```

单元测试验证局部逻辑，集成测试验证模块协作，端到端测试验证真实链路，回归测试防止修复过的问题再次出现。涉及随机模型输出时，应测试结构、范围和不变量，避免断言脆弱的完整文本。

### 研究代码工程化

建议逐步拆分：

```text
配置
  -> 数据读取与校验
  -> 模型与推理
  -> 训练循环
  -> 评测指标
  -> 日志和产物
  -> 命令行入口
```

优先抽离稳定边界，不要为了形式一次性重构全部代码。每次重构前保留可复现基线，重构后使用相同输入比较关键指标和产物。

## 面试应对

### Python 多线程、多进程和 asyncio 怎么选？

回答思路：先区分 CPU 与 IO，再说明共享状态和依赖约束。

回答模板：

IO 密集任务可以使用多线程，因为线程等待网络或磁盘时其他线程可以继续工作；纯 Python CPU 密集任务受 GIL 影响，通常使用多进程或把计算下沉到 NumPy、PyTorch 等底层实现。面对大量支持异步接口的网络请求，可以使用 asyncio，用较少线程管理很多等待任务。选择时还要考虑对象能否序列化、共享状态、错误隔离和第三方库是否真正支持异步。

### 如何设计一个可靠的批处理脚本？

回答思路：覆盖校验、幂等、重试、进度、日志和产物。

回答模板：

我会先校验输入 Schema 和版本，并为每条任务定义稳定 ID。处理过程按批次执行，单条结果幂等落盘，同时记录成功、失败、耗时和错误类型。只对超时、限流等瞬时错误做带退避的有限重试，权限和参数错误直接失败。进度基于任务 ID 而不是行号，重启后跳过已完成项；输出先写临时文件，校验后再发布。这样脚本即使中断也能恢复，失败项也可以单独重跑。

### 单元测试、集成测试和回归测试有什么区别？

回答思路：按验证范围和目的区分。

回答模板：

单元测试验证函数或类的局部逻辑，运行快，适合覆盖边界和异常；集成测试验证数据、模型、存储或外部接口之间能否正确协作；端到端测试从真实入口验证完整链路；回归测试则把历史 bug 和关键能力固定下来，防止后续修改重新破坏。实际项目中我会让大量稳定逻辑由单元测试覆盖，再用少量集成和端到端测试保护关键链路。
