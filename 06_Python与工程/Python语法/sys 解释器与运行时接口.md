# sys 解释器与运行时接口

## 知识点解析

### 概述

`sys` 是 Python 标准库中与解释器运行时直接交互的模块。它提供命令行参数、标准输入输出流、进程退出、模块搜索路径、递归限制、解释器版本和运行环境等信息。

```python
import sys
```

`sys` 关注的是“当前 Python 解释器和进程如何运行”，不是通用操作系统功能：

- 文件和目录操作通常使用 `pathlib` 或 `os`。
- 进程创建和命令执行使用 `subprocess`。
- 结构化命令行解析优先使用 `argparse`。
- 对象内存分析使用 `tracemalloc`、`pympler` 等工具，而不是只依赖 `sys.getsizeof()`。

### 命令行参数：sys.argv

`sys.argv` 是字符串列表，保存启动脚本时传入的命令行参数。通常：

- `sys.argv[0]` 是脚本名或启动入口。
- `sys.argv[1:]` 是用户提供的参数。

假设 `demo.py` 内容为：

```python
import sys

print(sys.argv)
```

执行：

```bash
python demo.py train --epochs 10
```

可能输出：

```python
["demo.py", "train", "--epochs", "10"]
```

所有参数最初都是字符串，需要自行转换：

```python
epochs = int(sys.argv[3])
```

直接访问固定下标前必须检查参数数量，否则可能触发 `IndexError`。参数较多、带默认值或需要帮助信息时，应使用 `argparse`，不要手写复杂的 `sys.argv` 解析逻辑。

### 标准输入、输出和错误流

`sys` 暴露三个标准文本流：

| 对象 | 方向 | 常见用途 |
| --- | --- | --- |
| `sys.stdin` | 输入 | 从终端或重定向文件读取 |
| `sys.stdout` | 正常输出 | 输出程序结果 |
| `sys.stderr` | 错误输出 | 输出诊断和错误信息 |

按行读取：

```python
line = sys.stdin.readline()
```

`readline()` 会保留行末换行符；到达文件末尾时返回空字符串：

```python
line = sys.stdin.readline()
if line == "":
    print("EOF")
```

读取到 EOF（End of File，文件结束）：

```python
for line in sys.stdin:
    line = line.strip()
    if line:
        print(line)
```

写入标准输出时不会自动补换行：

```python
sys.stdout.write("hello\n")
```

错误信息可单独写入标准错误流：

```python
sys.stderr.write("invalid argument\n")
```

在 shell 中，`stdout` 和 `stderr` 可以分别重定向，因此库和命令行工具应把正常结果与诊断信息分开。

### 文本流与二进制缓冲区

`sys.stdin`、`sys.stdout` 是文本流；它们的 `.buffer` 属性访问底层二进制流：

```python
data = sys.stdin.buffer.read()
sys.stdout.buffer.write(b"ok\n")
```

主要区别：

| 接口 | 读取或写入类型 | 适用场景 |
| --- | --- | --- |
| `sys.stdin.readline()` | `str` | 普通文本 |
| `sys.stdin.buffer.readline()` | `bytes` | 大规模 token 输入、二进制处理 |
| `sys.stdout.write()` | `str` | 文本输出 |
| `sys.stdout.buffer.write()` | `bytes` | 二进制输出 |

不要混淆 `str` 和 `bytes`：

```python
text = sys.stdin.buffer.readline().decode().strip()
sys.stdout.write(text + "\n")
```

算法题中的高性能输入模板、EOF 处理和输出缓冲策略见[ACM 基础输入输出](<ACM输入输出.md#acm-基础输入输出>)。

### 退出程序：sys.exit

```python
sys.exit()
sys.exit(0)
sys.exit("invalid input")
```

`sys.exit()` 的机制是抛出 `SystemExit` 异常：

- `sys.exit(0)` 通常表示正常结束。
- 非零整数通常表示异常结束。
- 传入字符串时，字符串通常会输出到标准错误流，并以非零状态结束。

因为它抛出的是异常，所以 `finally` 块仍会执行：

```python
try:
    sys.exit(1)
finally:
    print("cleanup")
```

在可复用函数和库代码中，不应随意调用 `sys.exit()`，否则调用方难以恢复。更好的方式是返回结果或抛出业务异常，由最外层命令行入口决定退出码。

模块顶层不能直接使用 `return`，因此脚本常见两种写法：

```python
# 写法一
if invalid:
    raise SystemExit
```

```python
# 写法二
def main():
    if invalid:
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

第二种写法便于测试 `main()`，也能集中管理退出码。

### 模块搜索路径：sys.path

`sys.path` 是导入模块时搜索目录的列表：

```python
import sys

for path in sys.path:
    print(path)
```

当执行 `import package` 时，导入系统会结合 `sys.path`、内置模块和导入钩子寻找目标模块。常见路径来源包括脚本目录、当前环境配置、标准库、第三方包目录和 `PYTHONPATH`。

临时插入路径：

```python
sys.path.insert(0, "/path/to/project")
```

这种方式适合临时脚本和排查，不适合作为长期工程依赖管理方案。长期项目应使用正确的包结构、虚拟环境和安装配置，避免在业务代码中散落路径修改。

`sys.modules` 是已加载模块的缓存映射：

```python
import sys

"math" in sys.modules
```

模块首次成功导入后通常会记录在 `sys.modules` 中，后续导入可以复用模块对象。直接修改 `sys.modules` 会改变导入行为，除测试替换和导入系统扩展外不应随意使用。

### 递归深度：getrecursionlimit 与 setrecursionlimit

```python
limit = sys.getrecursionlimit()
sys.setrecursionlimit(2000)
```

Python 设置递归深度限制，是为了防止无限递归耗尽 C 调用栈。超过限制通常抛出 `RecursionError`。

提高限制不等于解决算法问题：

- 深度本身过大时，仍可能导致进程栈溢出。
- 指数级递归不会因为提高限制而降低时间复杂度。
- 图搜索和树遍历可以根据规模改成显式栈迭代。

只应在确认递归深度可控、算法复杂度合理并且运行环境允许时适度调整。

### 解释器和平台信息

常用运行时信息：

| 属性 | 含义 |
| --- | --- |
| `sys.version` | 可读的 Python 版本字符串 |
| `sys.version_info` | 可比较的版本信息元组 |
| `sys.executable` | 当前 Python 解释器路径 |
| `sys.platform` | 平台标识，如 `linux`、`win32` |
| `sys.byteorder` | 本机字节序，`little` 或 `big` |
| `sys.maxsize` | 平台相关的最大 `Py_ssize_t` 值 |
| `sys.prefix` | 当前 Python 环境前缀 |
| `sys.base_prefix` | 基础 Python 安装前缀 |

版本判断使用 `version_info`，不要解析 `version` 字符串：

```python
if sys.version_info >= (3, 10):
    print("Python 3.10+")
```

判断是否位于虚拟环境中的一种常见方式：

```python
in_venv = sys.prefix != sys.base_prefix
```

平台差异较复杂时优先使用 `platform` 模块或功能检测，不要只依赖 `sys.platform` 猜测所有能力。

### 对象大小：sys.getsizeof

```python
size = sys.getsizeof(obj)
```

`getsizeof()` 返回对象自身占用的浅层大小，通常不递归统计它引用的其他对象：

```python
data = [1, 2, 3]
sys.getsizeof(data)
```

该结果主要包含列表容器自身及其槽位，不等于列表和三个整数的完整内存总量。不同 Python 实现、版本和构建方式的结果也可能不同。

因此：

- 比较同一环境中相似对象的浅层开销时可以使用 `getsizeof()`。
- 分析真实内存峰值、分配来源或对象引用图时，应使用 `tracemalloc`、分析器或进程级监控。

### 常用接口速查

| 接口 | 用途 | 主要风险 |
| --- | --- | --- |
| `sys.argv` | 获取命令行原始参数 | 参数仍是字符串，可能下标越界 |
| `sys.stdin/stdout/stderr` | 标准流读写 | 文本与 bytes 混用 |
| `sys.exit()` | 终止脚本并设置退出状态 | 深层函数退出整个程序 |
| `sys.path` | 查看或临时调整导入搜索路径 | 隐藏包结构和环境问题 |
| `sys.modules` | 查看已加载模块缓存 | 修改会影响导入行为 |
| `sys.setrecursionlimit()` | 调整递归深度限制 | 过大可能造成栈溢出 |
| `sys.version_info` | 版本判断 | 不应改用字符串比较 |
| `sys.getsizeof()` | 查看对象浅层大小 | 不代表完整内存占用 |

### 适用边界与易错点

1. `sys.argv[0]` 通常是脚本入口，不是第一个业务参数。
2. `sys.stdin.readline()` 会保留换行符，常配合 `strip()` 或 `rstrip("\n")`。
3. `sys.stdin.buffer` 返回 `bytes`，字符串操作前可能需要 `decode()`。
4. `sys.stdout.write()` 不会自动追加换行。
5. `sys.exit()` 实际抛出 `SystemExit`，`finally` 仍会执行。
6. 提高递归限制不能修复指数复杂度，也不能保证不会栈溢出。
7. `sys.getsizeof()` 只报告浅层大小。
8. 直接修改 `sys.path` 可以临时解决导入问题，但可能掩盖错误的包结构。

## 关联阅读

- [ACM 基础输入输出](<ACM输入输出.md#acm-基础输入输出>)
- [argparse 命令行参数解析](<../常用库/argparse.md#argparse>)
- [Python 官方文档：sys](https://docs.python.org/3/library/sys.html)
