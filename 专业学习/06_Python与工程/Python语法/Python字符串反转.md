# Python 字符串反转

## 知识点解析

### 概述

Python 字符串是不可变序列，反转操作不会修改原字符串，而是产生新字符串。常用方法包括切片 `s[::-1]`、`reversed()` 配合 `join()`，以及先转换为列表再原地反转。

### 使用切片

```python
s = "abcde"
reversed_s = s[::-1]

print(reversed_s)  # edcba
```

切片形式是 `s[start:stop:step]`。当步长为 `-1` 且省略起止位置时，遍历方向从右向左，因此 `s[::-1]` 得到反转后的新字符串。

该方法最简洁，适合直接反转整个字符串。

### 使用 `reversed()` 和 `join()`

```python
s = "abcde"
reversed_s = "".join(reversed(s))

print(reversed_s)  # edcba
```

`reversed(s)` 返回反向迭代器，不会直接返回字符串；`"".join(...)` 再把迭代产生的字符拼接为新字符串。

当代码需要明确表达“反向迭代”，或者需要先对反向迭代结果进行其他处理时，这种写法更合适。

### 转换为列表后原地反转

```python
s = "abcde"
chars = list(s)
chars.reverse()
reversed_s = "".join(chars)

print(reversed_s)  # edcba
```

`list.reverse()` 会原地修改列表，并返回 `None`：

```python
chars = ["a", "b", "c"]
result = chars.reverse()

print(result)  # None
```

这种方法步骤较多，但如果后续还要修改字符列表，使用可变列表会更方便。

### 方法对比

| 方法 | 返回结果 | 是否原地修改 | 适用场景 |
| --- | --- | --- | --- |
| `s[::-1]` | 新字符串 | 否 | 简洁地反转整个字符串 |
| `"".join(reversed(s))` | 新字符串 | 否 | 强调反向迭代或组合迭代操作 |
| `list(s)`、`reverse()`、`join()` | 新字符串 | 只原地修改中间列表 | 反转前后还需修改字符 |

三种写法都需要构造长度与原字符串同阶的新结果，因此整体时间复杂度为 O(n)，最终结果的空间复杂度为 O(n)。

### 边界与易错点

1. 字符串没有 `reverse()` 方法，因为字符串不可变。
2. `reversed(s)` 返回迭代器，不是字符串，也不支持直接按字符串使用。
3. `list.reverse()` 返回 `None`，不能写成 `chars = chars.reverse()`。
4. 空字符串和单字符字符串使用上述方法都能正常返回自身内容。
5. 上述方法按 Python 字符串中的码点顺序反转。对于由多个码点组成的用户感知字符，例如部分重音字符或 Emoji 组合，结果可能不符合视觉上的“逐字反转”；此时需要按 Unicode 字素簇处理。

## 面试应对

### Python 字符串反转有哪些常见写法？

回答思路：给出三种方法，再比较返回类型、可变性和使用场景。

回答模板：

最常见的写法是 `s[::-1]`，它通过负步长切片返回一个新字符串。也可以使用 `"".join(reversed(s))`，其中 `reversed()` 返回反向迭代器，再由 `join()` 拼接。若还需要修改字符，可以先用 `list(s)` 转成列表，调用 `reverse()` 原地反转，再拼接回字符串。字符串本身不可变，所以不能直接调用 `s.reverse()`。

### `reverse()` 和 `reversed()` 有什么区别？

回答思路：从调用形式、返回值和是否修改原对象三个维度比较。

回答模板：

`list.reverse()` 是列表方法，会原地修改列表并返回 `None`。`reversed(obj)` 是内置函数，接收支持反向迭代的对象并返回迭代器，不修改原对象。反转字符串时通常使用 `reversed(s)` 再配合 `join()`，因为字符串没有 `reverse()` 方法。
