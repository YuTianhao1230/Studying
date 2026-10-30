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

## 三、输入带方括号的列表

如果输入内容本身带有方括号和逗号，例如：

```text
[1, 2, 3, 4]
```

这不是普通的空格分隔输入，不能直接使用 `input().split()`。可以使用 `ast.literal_eval()` 将列表字面量解析为 Python 列表：

```python
import ast

numbers = ast.literal_eval(input())

print(numbers)  # [1, 2, 3, 4]
```

`numbers` 的类型是 `list`，其中的元素是整数。如果确定列表中只有四个元素，可以拆包成四个变量：

```python
import ast

a, b, c, d = ast.literal_eval(input())

print(a, b, c, d)  # 1 2 3 4
```

如果需要输出成逗号分隔的形式：

```python
import ast

numbers = ast.literal_eval(input())
print(*numbers, sep=", ")
```

输出：

```text
1, 2, 3, 4
```

`ast.literal_eval()` 只解析数字、字符串、列表、字典等 Python 字面量，比直接使用可能执行任意代码的 `eval()` 更安全。ACM/OJ 题目通常保证输入格式合法，因此一般不需要额外处理解析异常。

如果输入没有方括号，只是：

```text
1 2 3 4
```

仍然使用空格分隔的写法：

```python
numbers = list(map(int, input().split()))
```

## 四、输入两行字符串

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

## 五、最常用的输入写法

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

## 六、数据量较大或格式特殊的输入方法

### 快速按行读取

输入规模较大且主要是数字时：

```python
import sys

input = sys.stdin.buffer.readline

n, m = map(int, input().split())
numbers = list(map(int, input().split()))
```

`buffer.readline()` 返回 `bytes`。`int()`、`float()` 和 `split()` 可以直接处理字节串；需要使用字符串方法时再调用 `decode()`：

```python
text = input().decode().strip()
```

如果每行是普通文本，也可以使用文本流：

```python
import sys

input = sys.stdin.readline
text = input().strip()
```

### 一次读取全部空白分隔字段

输入全部由空白分隔、无需保留行结构时：

```python
import sys

data = list(map(int, sys.stdin.buffer.read().split()))
```

也可以使用迭代器依次取值：

```python
import sys

tokens = iter(sys.stdin.buffer.read().split())
n = int(next(tokens))
m = int(next(tokens))
numbers = [int(next(tokens)) for _ in range(n)]
```

这种方法适合大规模纯 token 输入，不适合必须保留整行空格的字符串。

### 读取到 EOF

题目没有给测试用例数，需要处理到文件结束时：

```python
import sys

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    numbers = list(map(int, line.split()))
    # 处理当前测试用例
```

若每个测试用例占固定多行，可以在循环中继续读取后续行：

```python
import sys

input = sys.stdin.buffer.readline

while True:
    line = input()
    if not line:
        break
    n = int(line)
    numbers = list(map(int, input().split()))
```

### 一行包含多种类型

先按字段拆分，再分别转换：

```python
parts = input().split()
n = int(parts[0])
rate = float(parts[1])
name = parts[2].decode()  # 使用 buffer.readline() 时需要解码
```

如果使用普通 `input()` 或 `sys.stdin.readline`，字符串字段不需要 `decode()`。

### 保留行内空格

命令或正文可能包含空格时，先读取整行，再按第一个空格切分：

```python
operation, _, content = input().rstrip("\n").partition(" ")
```

使用字节流时：

```python
line = input().decode().rstrip("\n")
operation, _, content = line.partition(" ")
```

## 七、最常用的输出写法

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

结果较多时先收集字符串，最后一次输出：

```python
import sys

answers = []
for value in numbers:
    answers.append(str(value))

sys.stdout.write("\n".join(answers))
```

### 固定小数位

```python
print(f"{answer:.2f}")
```

输出多个固定精度浮点数：

```python
print(" ".join(f"{value:.4f}" for value in values))
```

### 指定分隔符

```python
print(*numbers, sep=",")
```

如果题目要求逗号后带空格：

```python
print(*numbers, sep=", ")
```

不要输出题目没有要求的文字，例如：

```python
print("答案是:", answer)  # ACM 题目通常不能这样输出
```

只输出：

```python
print(answer)
```

## 八、输入格式和代码的对应关系

| 题目输入 | Python 写法 |
| --- | --- |
| `4` | `n = int(input())` |
| `1 2 3 4` | `list(map(int, input().split()))` |
| `1,2,3,4` | `list(map(int, input().split(",")))` |
| `[1, 2, 3, 4]` | `ast.literal_eval(input())` |
| `hello world` | `s = input()` |
| 后面有 `n` 行 | `for _ in range(n): ...` |
| 大规模按行输入 | `input = sys.stdin.buffer.readline` |
| 全部是空白分隔字段 | `sys.stdin.buffer.read().split()` |
| 未给测试用例数 | `for line in sys.stdin` |
| 多行结果 | `"\n".join(answers)` |

记忆方式：

```text
数字 -> int
一行多个数字 -> split + map(int, ...)
逗号分隔 -> split(",")
整行字符串 -> input()
多行数据 -> for 循环
```
