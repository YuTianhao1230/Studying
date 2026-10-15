# heapq 堆队列模块

## 知识点解析

### 概述

`heapq` 是 Python 标准库中的堆操作模块。它直接把普通列表作为堆使用，默认维护小根堆，不需要创建专门的堆对象。

堆适合需要频繁取得当前最小值或最大值的场景，例如缓存淘汰、任务调度、Top K、多路归并和最短路径。堆顶查询是 `O(1)`，插入和弹出堆顶是 `O(log n)`；堆的结构原理见[堆与优先队列](<../../10_计算机基础知识/数据结构与算法/堆与优先队列.md#堆与优先队列>)。

### 小根堆基本操作

```python
import heapq

heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 8)

print(heap[0])           # 2，只查看堆顶
minimum = heapq.heappop(heap)
print(minimum)           # 2，弹出堆顶
```

`heapq` 只保证 `heap[0]` 是最小值以及父节点不大于子节点，不保证列表整体有序。因此不能把堆的内部排列当成升序结果；需要完整有序结果时应使用 `sorted(heap)`，但这会额外花费 `O(n log n)`。

### 常用 API

| 操作 | 含义 | 复杂度 |
| --- | --- | --- |
| `heapq.heapify(data)` | 把已有列表原地调整成小根堆 | `O(n)` |
| `heapq.heappush(heap, x)` | 插入元素并恢复堆性质 | `O(log n)` |
| `heapq.heappop(heap)` | 弹出并返回最小元素 | `O(log n)` |
| `heap[0]` | 查看最小元素，不删除 | `O(1)` |
| `heapq.heapreplace(heap, x)` | 先弹出旧堆顶，再插入 `x` | `O(log n)` |
| `heapq.heappushpop(heap, x)` | 比较新元素和旧堆顶，返回较小者并在堆中保留较大者 | `O(log n)` |
| `heapq.nsmallest(k, data)` | 返回最小的 K 个元素 | 适合 K 相对较小 |
| `heapq.nlargest(k, data)` | 返回最大的 K 个元素 | 适合 K 相对较小 |

`heapify` 会直接修改传入的列表：

```python
data = [5, 2, 8]
heapq.heapify(data)
print(data[0])  # 2
```

### heapreplace 与 heappushpop

两者都把一次弹出和一次插入合并为一个操作，但执行语义不同。假设当前小根堆的堆顶是 `3`，新元素是 `1`：

```python
heap = [3, 5, 8]
removed = heapq.heapreplace(heap, 1)

print(removed)  # 3
print(heap[0])  # 1
```

`heapreplace` 强制移除旧堆顶，再插入新元素。它适合固定容量 Top K 中已经确认“新元素应该进入候选集合”的情况。

```python
heap = [3, 5, 8]
removed = heapq.heappushpop(heap, 1)

print(removed)  # 1
print(heap[0])  # 3
```

`heappushpop` 比较新元素和旧堆顶，返回较小者，把较大者保留在堆中。新元素比堆顶还小时，可能直接返回新元素，原堆不变。

### 大根堆

为了兼容不同版本的 Python 和在线评测环境，常用做法是存入相反数，把“原值最大”转换成“相反数最小”：

```python
heap = []
for value in [5, 2, 8]:
    heapq.heappush(heap, -value)

maximum = -heapq.heappop(heap)
print(maximum)  # 8
```

如果堆中保存元组，只对需要反转排序方向的数值取负。例如按照 `score` 降序、`pos` 升序弹出：

```python
heapq.heappush(heap, (-score, pos))
negative_score, pos = heapq.heappop(heap)
score = -negative_score
```

若同分时还要让 `pos` 较大的元素先弹出，则使用 `(-score, -pos)`。负号改变的是对应字段的排序方向。

### 元组与多级优先级

`heapq` 使用 Python 的元组比较规则：先比较第一个字段，相同再比较第二个字段，依次类推。

```python
heap = []
heapq.heappush(heap, (5.0, 3))
heapq.heappush(heap, (2.0, 8))
heapq.heappush(heap, (2.0, 1))

print(heapq.heappop(heap))  # (2.0, 1)
```

上述元组表示先按 `score` 升序，再按 `pos` 升序。它与以下排序键表达的是同一套优先级：

```python
sorted(records, key=lambda pos: (records[pos][0], pos))
```

区别在于全量排序一次需要 `O(n log n)`，而堆每次插入或弹出只需要 `O(log n)`。

当元组前面的优先级字段完全相同时，Python 会继续比较后续字段。如果后续对象不支持大小比较，可能触发 `TypeError`。任务调度中常加入单调递增的序号，避免直接比较任务对象：

```python
heapq.heappush(heap, (priority, sequence, task))
```

### 固定容量缓存淘汰

缓存容量超限时，若要求淘汰“分数最低、同分时位置最小”的记录，可以让堆保存 `(score, pos)`：

```python
records = {}
heap = []

records[pos] = (score, key, value)
heapq.heappush(heap, (score, pos))

if len(records) > capacity:
    _, removed_pos = heapq.heappop(heap)
    del records[removed_pos]
```

字典负责保存完整记录，堆只保存淘汰优先级。每次加入和淘汰是 `O(log K)`，其中 `K` 是缓存容量；若每次超限后重新排序全部缓存，则一次淘汰会退化为 `O(K log K)`。

上述写法成立的前提是每个 `pos` 只插入一次。如果同一 `pos` 的分数允许更新，堆中会留下旧元组，需要使用惰性删除。

### 惰性删除

`heapq` 不支持根据位置直接删除或修改任意元素。常见做法是把新状态继续压入堆，弹出时再判断它是否仍然有效：

```python
while heap:
    score, pos = heapq.heappop(heap)
    if pos in records and records[pos][0] == score:
        break
```

旧状态仍会暂时占用空间，但每个状态最多入堆和出堆一次，因此整体复杂度通常仍可按总入堆次数分析。若无效状态长期无法到达堆顶，堆占用可能增长，需要结合业务规模决定是否定期重建。

### 复杂度与选型

| 需求 | 方法 | 单次取得并删除最小值 |
| --- | --- | --- |
| 数据很少，代码优先 | `sorted` 后取首项 | `O(n log n)` |
| 只执行一次最小值查询 | `min` | `O(n)` |
| 频繁插入并弹出最小值 | `heapq` | `O(log n)` |
| 同时维护完整记录 | 字典加 `heapq` | 字典查找平均 `O(1)`；堆操作 `O(log n)` |

`heapq` 适合只关心当前极值，不要求数据始终完整有序的场景。如果需要按任意位置删除、范围查询或维护完整有序序列，应考虑平衡树、排序容器或其他数据结构。

### 易错点

- `heapq` 默认是小根堆；跨版本实现大根堆常用负数。
- `heap[0]` 是堆顶，但 `heap` 整体不一定有序。
- `heappop`、`heapreplace` 对空堆操作会抛出 `IndexError`。
- 元组从左到右比较；相同优先级下应明确第二排序字段。
- 使用负数模拟大根堆时，只对需要降序的字段取负。
- `heapify` 原地修改列表，不会返回一个新堆。
- 元素优先级发生更新时，不能直接假设旧堆项自动失效。

## 笔试常考

### 典型题型

| 题型 | 识别信号 | 解法 |
| --- | --- | --- |
| 最低优先级淘汰 | 频繁插入并删除最小值 | 小根堆保存优先级 |
| 最高优先级任务 | 频繁取得最大值 | 负数模拟大根堆 |
| 同分二级规则 | 同分时按编号或时间处理 | 元组保存多个排序字段 |
| 最大 K 个元素 | 数据流、K 远小于 n | 大小为 K 的小根堆 |
| 元素允许更新 | 同一编号可能重复出现 | 字典保存最新状态，堆使用惰性删除 |

### 解题步骤

1. 判断需要频繁取得最小值还是最大值。
2. 确定堆中保存原值、相反数还是元组。
3. 把所有同分规则按优先级从左到右放入元组。
4. 明确堆是否固定容量，以及超限时淘汰哪一端。
5. 检查元素是否会更新；会更新时增加有效性校验。
6. 根据题目要求决定最终结果是否还需要额外排序。

### 易错判断

- `heapq.heappush(heap, x)` 不会让整个列表完全升序。
- 求最大的 K 个元素应维护大小为 K 的小根堆，因为堆顶是候选中最应该被淘汰的最小值。
- `(-score, pos)` 表示 `score` 降序、`pos` 升序；`(-score, -pos)` 才表示两个字段都降序。

## 面试应对

### Python heapq 的常用操作和复杂度是什么？

回答思路：说明默认小根堆、底层列表、核心 API 和复杂度。

回答模板：

Python 的 `heapq` 直接在列表上维护小根堆。`heap[0]` 可以用 O(1) 时间查看最小值，`heappush` 和 `heappop` 分别以 O(log n) 完成插入和弹出，`heapify` 可以用 O(n) 时间把现有列表原地建堆。堆只保证堆顶和父子大小关系，不保证内部列表整体有序。

### heapq 如何实现大根堆和复合优先级？

回答思路：解释负数映射和元组从左到右比较。

回答模板：

为了兼容常见 Python 和在线评测环境，我通常通过存储相反数实现大根堆，弹出后再取负还原。复合优先级可以使用元组，例如 `(score, pos)` 表示先按分数升序，同分再按位置升序；如果某个数值字段需要降序，就对该字段取负。若优先级相同且任务对象不可比较，我会加入单调递增序号避免比较对象本身。

### heapreplace 和 heappushpop 有什么区别？

回答思路：按执行顺序和适用场景对比。

回答模板：

`heapreplace` 会强制先弹出旧堆顶，再插入新元素，所以返回的一定是原堆中的元素；`heappushpop` 会先比较新元素和旧堆顶，返回其中较小者，并把较大者留在堆里。固定容量 Top K 中，如果已经判断新元素应进入候选集合，适合用 `heapreplace`；如果希望把新元素也纳入比较后再决定是否保留，适合用 `heappushpop`。

### heapq 中的元素允许更新时怎么处理？

回答思路：说明不能原地高效更新任意堆项，引出字典和惰性删除。

回答模板：

`heapq` 不支持按标识直接高效修改任意元素。常见做法是用字典保存每个标识的最新状态，更新时把新状态再次入堆；弹出时对照字典检查该堆项是否仍有效，无效就继续弹出。这叫惰性删除。它避免了在堆中线性查找旧元素，但要注意无效项可能增加空间占用。

## 关联阅读

- [堆与优先队列](<../../10_计算机基础知识/数据结构与算法/堆与优先队列.md#堆与优先队列>)
- [高频算法模板：堆与 Top K](<../算法刷题/高频算法模板.md#堆与-top-k>)
