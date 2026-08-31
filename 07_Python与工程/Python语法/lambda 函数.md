# lambda 函数

## 知识点解析

### 概述

lambda 是用来创建匿名函数的关键字，语法是 `lambda 参数: 表达式`，函数体只能是单个表达式、其结果即为返回值，常用于给 sorted、map、filter 等传一个一次性的简单函数。

### `lambda` 是什么？

`lambda` 是一种用来创建**匿名函数**（anonymous function）的关键字。

*   **匿名**：意味着这个函数没有正式的名字（不像用 `def` 定义的函数）。
*   **简洁**：它通常只有一行，用于实现一些简单的功能。
*   **本质**：它就是一个函数对象，可以像其他函数一样被传来传去。

把它想象成一个“一次性”的、用完就扔的迷你函数。

### `lambda` 的语法

它的结构非常简单：

```python
lambda arguments: expression
```

*   `lambda`: 固定的关键字。
*   `arguments`: 参数列表，和普通函数的参数一样，可以有多个，用逗号隔开。
*   `:`: 分隔符。
*   `expression`: **一个单独的表达式**。这个表达式的计算结果就是函数的返回值。

**关键点：`lambda` 函数体只能是一个表达式，不能是语句（比如 `if/else` 块、`for` 循环、`print()` 等）。**

#### 与普通函数的对比

一个简单的加法函数，用两种方式来写：

**普通函数 (`def`)**
```python
def add(x, y):
    return x + y

result = add(3, 5)  # result is 8
```

**`lambda` 函数**
```python
add_lambda = lambda x, y: x + y

result = add_lambda(3, 5)  # result is 8
```
在这个例子里，`add_lambda` 只是一个变量名，它指向了 `lambda` 创建的那个匿名函数对象。

### `lambda` 怎么用？（核心应用场景）

`lambda` 的威力并不在于像上面那样给它命名后使用，而是在于**将它作为参数传递给其他高阶函数**（Higher-order Functions），比如 `sorted()`, `map()`, `filter()` 等。

#### 场景一：自定义排序 `sorted()`

这是 `lambda` 最最常见的用途。`sorted()` 函数有一个 `key` 参数，你可以提供一个函数，告诉 `sorted` 按照什么规则来排序。

**示例：按列表中元组的第二个元素（年龄）排序**

```python
students = [('Alice', 25), ('Bob', 20), ('Charlie', 22)]

### 不使用 lambda，需要先定义一个函数
def get_age(student_tuple):
    return student_tuple[1]

sorted_students = sorted(students, key=get_age)
print(sorted_students)  # [('Bob', 20), ('Charlie', 22), ('Alice', 25)]

### 使用 lambda，代码更紧凑
sorted_students_lambda = sorted(students, key=lambda student: student[1])
print(sorted_students_lambda) # [('Bob', 20), ('Charlie', 22), ('Alice', 25)]
```
`lambda student: student[1]` 创建了一个临时函数，它接收一个元组 `student`，并返回它的第二个元素 `student[1]`。`sorted` 就用这个返回值作为排序的依据。

#### 场景二：处理数据 `map()`

`map(function, iterable)` 函数会对 `iterable`（如列表）中的每一个元素应用 `function`。

**示例：将列表中的每个数字都平方**

```python
numbers = [1, 2, 3, 4, 5]

### 使用 map 和 lambda
squared_numbers_iterator = map(lambda x: x * x, numbers)
squared_numbers_list = list(squared_numbers_iterator) # map返回的是迭代器，需要转为列表

print(squared_numbers_list) # [1, 4, 9, 16, 25]
```
> **提示**：虽然这个例子很好地展示了 `map` 和 `lambda`，但在 Python 中，使用列表推导式通常更受欢迎且可读性更高：`squared = [x * x for x in numbers]`。

#### 场景三：筛选数据 `filter()`

`filter(function, iterable)` 会筛选出 `iterable` 中所有让 `function` 返回 `True` 的元素。

**示例：筛选出列表中的所有偶数**
```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8]

### 使用 filter 和 lambda
even_numbers_iterator = filter(lambda x: x % 2 == 0, numbers)
even_numbers_list = list(even_numbers_iterator)

print(even_numbers_list) # [2, 4, 6, 8]
```
> **提示**：同样，列表推导式在这里也很好用：`evens = [x for x in numbers if x % 2 == 0]`。

### `lambda` 的限制与最佳实践

*   **只能有一个表达式**：这是它最大的限制。你不能在 `lambda` 中写多行逻辑。但你可以使用三元运算符来实现简单的条件判断。
    ```python
    # 如果 x > 10 返回 'big', 否则返回 'small'
    f = lambda x: 'big' if x > 10 else 'small'
    print(f(15)) # 'big'
    print(f(5))  # 'small'
    ```

*   **保持简洁**：`lambda` 的设计初衷就是为了简洁。如果你的 `lambda` 表达式写得非常复杂，难以阅读，那这就是一个明确的信号：**你应该使用 `def` 来定义一个常规函数**。代码的可读性通常比紧凑性更重要。

### 总结

| | **普通函数 (`def`)** | **`lambda` 函数** |
| :--- | :--- | :--- |
| **名称** | 必须有名字 | 匿名，没有名字 |
| **函数体** | 可以包含多行语句和表达式 | **只能包含一个表达式** |
| **返回值** | 使用 `return` 语句显式返回 | **表达式的结果被隐式返回** |
| **用途** | 复杂的、需要复用的逻辑 | 简单的、一次性的功能，常作为高阶函数的参数 |
| **最佳场景** | 定义程序的核心功能模块 | `sorted`, `map`, `filter` 的 `key` 或 `function` 参数 |
## 面试应对

### lambda 函数 是什么？

回答思路：点明它是单表达式的匿名函数，最常作为 `key=`/`map`/`filter` 的即用即弃回调。

回答模板：

`lambda` 是 Python 的匿名函数语法，适合定义短小的一次性函数，常见于 `key=`、`map`、`filter` 等场景。

### lambda 函数 适合什么场景？

回答思路：抓住它只能写单个表达式这条界限，逻辑一超过一行就该换 def 具名函数。

回答模板：

它只能写表达式，不能写复杂语句。工程里如果逻辑超过一行，最好定义具名函数，方便阅读、测试和调试。

### lambda 函数 常见坑是什么？

回答思路：抓住循环里闭包延迟绑定这个经典坑，以及只能单表达式、匿名难调试。

回答模板：

lambda 最经典的坑是循环里的闭包延迟绑定：像 `[lambda: i for i in range(3)]` 里所有函数最后都会返回 2，因为它们引用的是同一个变量 i，而不是当时的值；要用默认参数 `lambda i=i: i` 把当前值固定下来。另外它只能是单个表达式，写不了 if/else 语句块、for、赋值这些，逻辑一旦超过一行就该改用 def；而且 lambda 是匿名的，报错栈里只显示 `<lambda>`，不好调试。
