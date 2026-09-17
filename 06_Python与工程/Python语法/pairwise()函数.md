# pairwise()函数

## 知识点解析

### 概述

`pairwise()` 是 Python `itertools` 模块中的一个函数，它在 **Python 3.10 版本**中被正式引入标准库。它的功能非常直接且实用：**将一个可迭代对象（如列表、字符串、元组等）中的元素，两两相邻地配对起来**。

### 核心功能

`pairwise(iterable)` 会返回一个迭代器（iterator），这个迭代器会生成一系列的元组（tuple），每个元组包含原序列中相邻的两个元素。

举个例子：

-   如果输入是 `['A', 'B', 'C', 'D']`
-   `pairwise()` 会依次生成：`('A', 'B')`, `('B', 'C')`, `('C', 'D')`

**关键点**：
1.  它总是生成 `n-1` 个配对，其中 `n` 是原始序列的长度。
2.  如果原始序列的元素少于2个，`pairwise()` 不会生成任何内容。

### 如何使用

你需要先从 `itertools` 模块中导入它。

```python
from itertools import pairwise

### 示例 1: 使用列表
letters = ['A', 'B', 'C', 'D']
pairs = pairwise(letters)

print(list(pairs))  # 将迭代器转换为列表以便查看
### 输出: [('A', 'B'), ('B', 'C'), ('C', 'D')]

### 示例 2: 使用字符串 (这正是罗马数字转换代码中的用法)
s = "PYTHON"
for char1, char2 in pairwise(s):
    print(f"当前对: {char1}, {char2}")

### 输出:
### 当前对: P, Y
### 当前对: Y, T
### 当前对: T, H
### 当前对: H, O
### 当前对: O, N

### 示例 3: 使用元组
numbers = (1, 2, 3, 4, 5)
for num1, num2 in pairwise(numbers):
    print(f"{num1} + {num2} = {num1 + num2}")

### 输出:
### 1 + 2 = 3
### 2 + 3 = 5
### 3 + 4 = 7
### 4 + 5 = 9

### 示例 4: 序列长度小于2的情况
short_list = [100]
print(list(pairwise(short_list))) # 输出: []

empty_list = []
print(list(pairwise(empty_list))) # 输出: []
```

### `pairwise()` 的工作原理 (内部实现)

如果你想了解 `pairwise()` 是如何工作的，可以看一下官方文档中给出的等效实现代码。这有助于你更深刻地理解它。

```python
from itertools import tee

def pairwise_equivalent(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)  # 创建两个独立的迭代器 a 和 b，它们都指向同一个原始序列
    next(b, None)         # 将迭代器 b 向前推进一个元素
    return zip(a, b)      # 将 a 和 b 像拉链一样合并起来
```

让我们用 `s = "PYTHON"` 来分解这个过程：

1.  **`a, b = tee("PYTHON")`**:
    *   `tee()` 函数会创建两个独立的迭代器，`a` 和 `b`。
    *   此时，`a` 指向 `P Y T H O N`。
    *   `b` 也指向 `P Y T H O N`。

2.  **`next(b, None)`**:
    *   `next()` 函数会获取迭代器的下一个元素。
    *   我们对迭代器 `b` 调用 `next()`，使其向前移动一步。
    *   现在，`a` 仍然指向 `P Y T H O N`。
    *   而 `b` 已经消耗了第一个元素 'P'，它现在指向 `Y T H O N`。

3.  **`return zip(a, b)`**:
    *   `zip()` 函数会将多个迭代器“压缩”在一起，每次从每个迭代器中取出一个元素，组成一个元组。
    *   `zip()` 会一直工作，直到最短的那个迭代器耗尽。
    *   过程如下：
        -   `zip` 从 `a` 取出 `P`，从 `b` 取出 `Y` -> 生成 `('P', 'Y')`
        -   `zip` 从 `a` 取出 `Y`，从 `b` 取出 `T` -> 生成 `('Y', 'T')`
        -   `zip` 从 `a` 取出 `T`，从 `b` 取出 `H` -> 生成 `('T', 'H')`
        -   ...以此类推...
        -   `zip` 从 `a` 取出 `O`，从 `b` 取出 `N` -> 生成 `('O', 'N')`
        -   迭代器 `b` 已经结束了（因为它比 `a` 短一个元素）。`zip` 停止工作。

这就是 `pairwise()` 实现的巧妙之处。

### 如果你的 Python 版本低于 3.10

如果你的 Python 版本（例如 3.8 或 3.9）没有内置的 `itertools.pairwise`，你可以自己实现一个等效的函数，最简单的方式就是使用上面提到的 `tee` 和 `zip`。

```python
from itertools import tee, zip_longest # 通常用zip就行

def my_pairwise(iterable):
    """
    一个在 Python < 3.10 版本中模拟 pairwise 的函数。
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

### 然后就可以像使用官方 pairwise 一样使用它
s = "PYTHON"
for char1, char2 in my_pairwise(s):
    print(f"{char1}{char2}", end=" ") # 输出: PY YT TH HO ON
```

### 总结

`pairwise()` 是一个用于处理“滑动窗口”大小为2的场景的绝佳工具。在罗马数字转换的问题中，它优雅地提供了相邻字符的比较，使得算法的核心逻辑（`ans += x if x >= y else -x`）可以非常简洁地实现。当你需要比较一个序列中前后相邻的元素时，第一时间就应该想到 `itertools.pairwise()`。
## 面试应对

### pairwise()函数 是什么？

回答思路：点明它把序列相邻元素两两配对，是大小为 2 的滑动窗口，天然适合相邻比较。

回答模板：

`pairwise()` 用于把序列转成相邻元素对，例如 `[a,b,c]` 变成 `(a,b)、(b,c)`，常用于相邻差值、区间边界和路径边统计。

### pairwise()函数 适合什么场景？

回答思路：抓住它 3.10 才入标准库、老版本用 zip(nums, nums[1:]) 替代，且少于 2 个元素不产生 pair。

回答模板：

Python 3.10 起 `itertools.pairwise` 可直接使用；老版本可以用 `zip(nums, nums[1:])`。要注意空序列或单元素序列不会产生任何 pair。

### pairwise()函数 常见坑是什么？

回答思路：抓住版本依赖、返回一次性迭代器只能遍历一遍、n 个元素只有 n-1 个对这几个点。

回答模板：

pairwise 的坑首先是版本：它 Python 3.10 才进标准库，老环境得自己用 `zip(it, it[1:])` 或 `tee` 实现。其次它返回的是一次性迭代器，`list()` 或遍历一遍后就空了，不能重复用。还有边界要记牢：n 个元素只产生 n-1 个对，空序列和单元素序列直接返回空、循环体一次都不执行，写滑动窗口逻辑时别默认它至少跑一轮。另外 `zip(nums, nums[1:])` 这种替代写法只适用于可切片的序列，对普通迭代器不成立。
