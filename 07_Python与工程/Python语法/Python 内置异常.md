Python 的异常被组织成一个层次结构。几乎所有常见的异常都继承自一个基类 `Exception`。下面我将列出一些最常见的异常类型，并按功能分组，附上简单的代码示例来说明它们何时会发生。

---

### 1. 与变量、属性和名称相关的错误

这些错误通常发生在你试图使用一个不存在或不合法的标识符时。

*   **`AttributeError`**
    *   **触发条件**：试图访问或设置一个对象上不存在的属性。
    *   **示例**：
        ```python
        my_list = [1, 2, 3]
        # 列表没有 'append_item' 这个方法 (正确的是 'append')
        my_list.append_item(4)  # AttributeError: 'list' object has no attribute 'append_item'
        ```

*   **`NameError`**
    *   **触发条件**：使用了尚未被定义的变量或函数名。
    *   **示例**：
        ```python
        print(non_existent_variable)  # NameError: name 'non_existent_variable' is not defined
        ```

### 2. 与数据类型和值相关的错误

这类错误发生在你向函数或操作传递了不兼容的类型或不合法的值。

*   **`TypeError`**
    *   **触发条件**：对一个对象执行了其类型不支持的操作。
    *   **示例**：
        ```python
        # 字符串和整数不能直接相加
        result = "hello" + 5  # TypeError: can only concatenate str (not "int") to str
        ```

*   **`ValueError`**
    *   **触发条件**：函数收到的参数类型正确，但其值不合适或不合法。
    *   **示例**：
        ```python
        # 'abc' 是字符串类型，但其值无法转换为整数
        num = int("abc")  # ValueError: invalid literal for int() with base 10: 'abc'
        ```

### 3. 与容器（列表、字典等）访问相关的错误

这些错误在你试图访问容器中不存在的元素时发生。

*   **`IndexError`**
    *   **触发条件**：使用了无效的索引来访问序列（如列表、元组）中的元素（通常是索引超出了范围）。
    *   **示例**：
        ```python
        my_list = [10, 20, 30]
        print(my_list[3])  # IndexError: list index out of range
        ```

*   **`KeyError`**
    *   **触发条件**：在字典中使用了不存在的键（key）来查找值。
    *   **示例**：
        ```python
        my_dict = {"name": "Alice", "age": 25}
        print(my_dict["city"])  # KeyError: 'city'
        ```

### 4. 与文件和 I/O (输入/输出) 相关的错误

这些错误在处理文件或其他外部资源时很常见。

*   **`FileNotFoundError`**
    *   **触发条件**：试图打开一个不存在的文件。
    *   **示例**：
        ```python
        with open("a_file_that_does_not_exist.txt", "r") as f:
            content = f.read()  # FileNotFoundError: [Errno 2] No such file or directory: 'a_file_that_does_not_exist.txt'
        ```

*   **`PermissionError`**
    *   **触发条件**：因为权限不足，无法读取或写入文件/目录。
    *   **示例**：
        ```python
        # 假设 C:\Windows 是一个受保护的系统目录
        with open("C:/Windows/system.log", "w") as f:
            f.write("test")  # PermissionError: [Errno 13] Permission denied: 'C:/Windows/system.log'
        ```

### 5. 其他常见错误

*   **`ZeroDivisionError`**
    *   **触发条件**：除法运算中，除数为零。
    *   **示例**：
        ```python
        result = 10 / 0  # ZeroDivisionError: division by zero
        ```

*   **`ImportError` / `ModuleNotFoundError`**
    *   **触发条件**：`import` 语句无法找到指定的模块。(`ModuleNotFoundError` 是 `ImportError` 的一个子类，更具体)。
    *   **示例**：
        ```python
        import non_existent_module  # ModuleNotFoundError: No module named 'non_existent_module'
        ```

*   **`SyntaxError` / `IndentationError`**
    *   **特殊说明**：这些是语法错误，通常**不能被 `try...except` 捕获**，因为它们在代码运行之前（解析阶段）就会导致程序失败。
    *   **`SyntaxError` 示例**：
        ```python
        print("hello"  # 缺少右括号，语法错误
        ```
    *   **`IndentationError` 示例**：
        ```python
        def my_func():
        print("hello") # 缩进错误
        ```

---

### 异常处理中的层次结构

了解异常的继承关系非常重要。例如，你可以用一个 `except` 块捕获多种类型的错误，或者捕获它们的父类。

```python
try:
    # ... 一些可能出错的代码 ...
    risky_operation() 
except FileNotFoundError:
    print("文件未找到！")
except (KeyError, IndexError):
    print("访问了不存在的键或索引！")
except Exception as e:
    # Exception 是大多数错误的基类，可以捕获上面未明确指定的其他运行时错误
    # 但要小心使用，因为它可能掩盖你没预料到的 bug
    print(f"发生了未预料的错误: {e}")
```

**总结表格**

| 异常名称 (`Exception Name`) | 中文解释 | 触发场景 |
| :--- | :--- | :--- |
| `AttributeError` | 属性错误 | 访问对象不存在的属性 |
| `NameError` | 名称错误 | 使用未定义的变量 |
| `TypeError` | 类型错误 | 对变量执行了其类型不支持的操作 |
| `ValueError` | 值错误 | 传入的参数值不合法 |
| `IndexError` | 索引错误 | 列表/元组的索引越界 |
| `KeyError` | 键错误 | 字典中不存在的键 |
| `FileNotFoundError` | 文件未找到错误 | 试图打开不存在的文件 |
| `ZeroDivisionError` | 除零错误 | 除数为零 |
| `ImportError` | 导入错误 | 无法导入指定的模块 |
