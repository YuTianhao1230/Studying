# ACM 基础输入输出

## 概述

ACM/OJ 题目中，程序直接从标准输入读取数据，把答案输出到标准输出。最基本的做法是：先读第一行确定数据规模，再按题目给出的格式读取后续内容。输入中是空格还是逗号，决定了 `split()` 使用的分隔符；输入中是数字还是字符串，决定了是否需要 `int()` 转换。

## 一、按行读取矩阵

题目输入：

```text
4
1 2 3 4
1 2 3 4
1 2 3 4
1 2 3 4
```

含义是第一行给出矩阵大小 `n=4`，后面有 `n` 行，每行有 `n` 个整数。

```python
n = int(input())
matrix = []

for _ in range(n):
    row = list(map(int, input().split()))
    matrix.append(row)

print(matrix)
```

读取后：

```python
[
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
    [1, 2, 3, 4],
]
```

如果只需要访问元素，可以直接在循环中处理，不一定要保存整个矩阵：

```python
n = int(input())

for _ in range(n):
    row = list(map(int, input().split()))
    # 处理当前行 row
```

如果矩阵是 `n` 行、`m` 列，第一行通常写成：

```text
n m
```

对应代码：

```python
n, m = map(int, input().split())
matrix = [list(map(int, input().split())) for _ in range(n)]
```

## 二、输入使用逗号分隔

如果每行是：

```text
1,2,3,4
```

不能使用默认的 `split()`，需要明确指定逗号：

```python
row = list(map(int, input().split(",")))
```

完整例子：

```python
n = int(input())
matrix = []

for _ in range(n):
    row = list(map(int, input().split(",")))
    matrix.append(row)

print(matrix)
```

如果逗号后面还有空格，例如：

```text
1, 2, 3, 4
```

可以先去掉每个数字两侧的空白：

```python
row = [int(x.strip()) for x in input().split(",")]
```

因此，常见分隔符的写法是：

```python
input().split()       # 空格或其他连续空白
input().split(",")    # 逗号
input().split(";")    # 分号
```

## 三、输入两行字符串

题目输入：

```text
hello
world
```

直接按行读取：

```python
s1 = input()
s2 = input()

print(s1)
print(s2)
```

如果题目要求把两行字符串拼接：

```python
s1 = input()
s2 = input()

print(s1 + s2)
```

如果要求比较两个字符串：

```python
s1 = input()
s2 = input()

if s1 == s2:
    print("Yes")
else:
    print("No")
```

如果字符串本身可能包含空格，仍然使用 `input()`，不要使用 `split()`：

```text
hello python
machine learning
```

```python
s1 = input()
s2 = input()
```

`input()` 读取整行；`input().split()` 会把一行拆成多个字段。

## 四、最常用的输入写法

### 一行一个整数

```python
n = int(input())
```

### 一行多个整数

```python
a, b, c = map(int, input().split())
```

### 一行整数列表

```python
numbers = list(map(int, input().split()))
```

### 一行字符串

```python
s = input()
```

### 按行读取多行数据

```python
n = int(input())
lines = [input() for _ in range(n)]
```

## 五、最常用的输出写法

### 输出一个值

```python
print(answer)
```

### 输出多个值

```python
print(a, b, c)
```

输出：

```text
a b c
```

### 输出列表

```python
print(*numbers)
```

也可以写成：

```python
print(" ".join(map(str, numbers)))
```

### 输出多行结果

```python
answers = ["Yes", "No", "Yes"]
print("\n".join(answers))
```

不要输出题目没有要求的文字，例如：

```python
print("答案是:", answer)  # ACM 题目通常不能这样输出
```

只输出：

```python
print(answer)
```

## 六、输入格式和代码的对应关系

| 题目输入 | Python 写法 |
| --- | --- |
| `4` | `n = int(input())` |
| `1 2 3 4` | `list(map(int, input().split()))` |
| `1,2,3,4` | `list(map(int, input().split(",")))` |
| `hello world` | `s = input()` |
| 后面有 `n` 行 | `for _ in range(n): ...` |

记忆方式：

```text
数字 -> int
一行多个数字 -> split + map(int, ...)
逗号分隔 -> split(",")
整行字符串 -> input()
多行数据 -> for 循环
```

## 面试应对

### 空格和逗号分隔的输入有什么区别？

回答模板：

`input().split()` 默认按连续空白分隔，例如空格；如果输入使用逗号，就写成 `input().split(",")`。分割后得到的内容还是字符串，数字还需要通过 `map(int, ...)` 转换。

### 如何读取一个 `n×n` 矩阵？

回答模板：

先读取第一行的 `n`，然后循环 `n` 次，每次读取一行并用 `list(map(int, input().split()))` 转成整数列表。如果输入是逗号分隔，就把 `split()` 改成 `split(",")`。

### 如何读取两行字符串？

回答模板：

直接调用两次 `input()`，分别保存到两个字符串变量。如果字符串可能包含空格，不能用 `split()`，因为 `input()` 才会保留整行内容。
