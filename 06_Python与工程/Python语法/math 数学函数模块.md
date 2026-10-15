# math 数学函数模块

## 知识点解析

### 概述

`math` 是 Python 标准库中的实数数学函数模块，提供取整、平方根、指数、对数、三角函数、组合数、最大公约数、浮点判断和数学常量等能力。

```python
import math

radius = 2
area = math.pi * radius**2
```

`math` 主要面向整数和实数标量。它不是复数库，也不是向量化数值计算库：

- 复数数学函数使用 `cmath`。
- 高精度十进制运算使用 `decimal.Decimal`。
- 数组、矩阵和批量向量化计算通常使用 NumPy。

### 常量

| 常量 | 含义 |
| --- | --- |
| `math.pi` | 圆周率 π |
| `math.e` | 自然对数底 e |
| `math.tau` | τ，即 `2 * pi` |
| `math.inf` | 正无穷，负无穷可写为 `-math.inf` |
| `math.nan` | NaN（Not a Number，非数值） |

无穷可以参与大小比较，但 NaN 与任何值比较相等都为假，包括它自身：

```python
math.inf > 10**100       # True
math.nan == math.nan     # False
math.isnan(math.nan)     # True
```

判断 NaN 应使用 `math.isnan()`，不能使用 `==`。

### 取整、拆分与余数

| 函数 | 含义 | 示例 |
| --- | --- | --- |
| `math.ceil(x)` | 向正无穷取整 | `ceil(-2.7) == -2` |
| `math.floor(x)` | 向负无穷取整 | `floor(-2.7) == -3` |
| `math.trunc(x)` | 向 0 截断 | `trunc(-2.7) == -2` |
| `math.modf(x)` | 返回小数部分和整数部分，二者都是浮点数 | `modf(3.25) == (0.25, 3.0)` |
| `math.fmod(x, y)` | 浮点余数，结果符号跟随 `x` | `fmod(-7, 3) == -1.0` |

`floor()`、`trunc()` 和 `int()` 在负数上不同：

```python
math.floor(-2.7)  # -3
math.trunc(-2.7)  # -2
int(-2.7)         # -2
```

整数取模通常使用 `%`。`math.fmod()` 更接近 C 标准库的浮点余数语义，不能与 `%` 机械互换。

### 幂、平方根、指数与对数

| 函数 | 含义 | 关键边界 |
| --- | --- | --- |
| `math.sqrt(x)` | 浮点平方根 | 实数域要求 `x >= 0` |
| `math.isqrt(n)` | 非负整数平方根向下取整 | 返回精确整数 |
| `math.pow(x, y)` | 浮点幂 | 参数通常转换为浮点数 |
| `math.exp(x)` | \(e^x\) | 大值可能溢出 |
| `math.expm1(x)` | \(e^x-1\) | `x` 接近 0 时比 `exp(x)-1` 精确 |
| `math.log(x)` | 自然对数 | 要求 `x > 0` |
| `math.log2(x)` | 以 2 为底的对数 | 要求 `x > 0` |
| `math.log10(x)` | 以 10 为底的对数 | 要求 `x > 0` |
| `math.log1p(x)` | \(\ln(1+x)\) | `x` 接近 0 时比 `log(1+x)` 精确 |

`math.sqrt()` 返回浮点数，`math.isqrt()` 返回整数：

```python
math.sqrt(10)   # 3.162277...
math.isqrt(10)  # 3
```

算法题枚举整数约数时优先使用 `isqrt()`，避免浮点精度和类型转换：

```python
factors = []
for d in range(1, math.isqrt(n) + 1):
    if n % d == 0:
        factors.append(d)
        if d * d != n:
            factors.append(n // d)
```

`math.pow(x, y)` 与内置 `pow(x, y)` 不完全相同：

- `math.pow()` 面向浮点数学，结果通常是 `float`。
- 内置 `pow(a, b, mod)` 支持整数模幂，适合快速幂和密码学计算。

```python
math.pow(2, 10)  # 1024.0
pow(2, 10)       # 1024
pow(2, 10, 1000) # 24
```

### 数论与组合数学

| 函数 | 含义 | 典型场景 |
| --- | --- | --- |
| `math.factorial(n)` | \(n!\) | 排列组合 |
| `math.comb(n, k)` | 组合数 \(C_n^k\) | 从 n 个元素选 k 个 |
| `math.perm(n, k)` | 排列数 \(P_n^k\) | 从 n 个元素有序选 k 个 |
| `math.gcd(*integers)` | 最大公约数 | 约分、周期合并 |
| `math.lcm(*integers)` | 最小公倍数 | 周期同步 |
| `math.prod(iterable)` | 连乘 | 维度乘积、组合计算 |

```python
math.factorial(5)  # 120
math.comb(5, 2)    # 10
math.perm(5, 2)    # 20
math.gcd(12, 18)   # 6
math.lcm(12, 18)   # 36
math.prod([2, 3, 4])  # 24
```

这些函数使用整数精确计算，不应先转成浮点数再计算。

### 三角函数与角度

`sin()`、`cos()`、`tan()` 的输入单位是弧度，不是角度：

```python
angle = math.radians(30)
math.sin(angle)  # 约等于 0.5
```

| 函数 | 含义 |
| --- | --- |
| `math.sin(x)`、`math.cos(x)`、`math.tan(x)` | 三角函数，输入为弧度 |
| `math.asin(x)`、`math.acos(x)`、`math.atan(x)` | 反三角函数，返回弧度 |
| `math.atan2(y, x)` | 根据 `(x, y)` 所在象限计算方向角 |
| `math.radians(deg)` | 角度转弧度 |
| `math.degrees(rad)` | 弧度转角度 |

二维向量逆时针旋转角度 \(\theta\)：

```python
new_x = x * math.cos(theta) - y * math.sin(theta)
new_y = x * math.sin(theta) + y * math.cos(theta)
```

相比 `atan(y / x)`，`atan2(y, x)` 能区分象限，也避免 `x == 0` 时直接做除法。

### 距离、范数与高精度求和

```python
math.hypot(3, 4)          # 5.0
math.dist((0, 0), (3, 4)) # 5.0
math.fsum([0.1, 0.1, 0.1])
```

- `hypot(*coordinates)` 计算欧氏范数。
- `dist(p, q)` 计算两个同维点之间的欧氏距离。
- `fsum(iterable)` 对浮点数进行更高精度的求和，降低普通 `sum()` 的累计舍入误差。

`fsum()` 仍然返回浮点数，不会把二进制浮点运算变成精确十进制运算。

### 浮点比较与特殊值判断

二进制浮点数不适合直接用 `==` 判断计算结果：

```python
0.1 + 0.2 == 0.3             # False
math.isclose(0.1 + 0.2, 0.3) # True
```

`isclose()` 的判断条件可概括为：

\[
|a-b| \le \max(\text{rel\_tol}\times\max(|a|,|b|),\text{abs\_tol})
\]

```python
math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)
```

- `rel_tol` 是相对误差容忍度，适合数值尺度较大时。
- `abs_tol` 是绝对误差容忍度，比较接近 0 的数时通常需要显式设置。

```python
math.isclose(1e-12, 0.0, abs_tol=1e-9)  # True
```

特殊值判断：

| 函数 | 判断内容 |
| --- | --- |
| `math.isfinite(x)` | 既不是无穷，也不是 NaN |
| `math.isinf(x)` | 正无穷或负无穷 |
| `math.isnan(x)` | NaN |

### 异常、定义域与选型边界

常见错误包括：

- `math.sqrt(-1)`、`math.log(0)` 会在实数域抛出 `ValueError`。
- `math.exp()` 的结果超过浮点范围时可能抛出 `OverflowError`。
- 三角函数输入默认是弧度，忘记转换会得到数值上合法但语义错误的结果。
- `math.nan` 不能用普通相等比较检测。
- 大整数精确开方不应使用 `n ** 0.5`，应使用 `math.isqrt()`。

选型对比：

| 需求 | 推荐工具 |
| --- | --- |
| 标量实数数学函数 | `math` |
| 复数函数 | `cmath` |
| 严格十进制精度和舍入 | `decimal` |
| 分数精确运算 | `fractions.Fraction` |
| 数组和矩阵批量计算 | NumPy |
| 基本绝对值、舍入、最大最小值 | 内置 `abs`、`round`、`max`、`min` |

## 笔试常考

### 高频识别

| 题目信号 | 常用方法 |
| --- | --- |
| 枚举约数到平方根 | `math.isqrt(n)` |
| 最大公约数、最小公倍数 | `math.gcd()`、`math.lcm()` |
| 组合数、排列数 | `math.comb()`、`math.perm()` |
| 浮点近似相等 | `math.isclose()` |
| 角度参与三角函数 | 先用 `math.radians()` |
| 判断 NaN、无穷 | `math.isnan()`、`math.isinf()` |

### 易错点

1. `math.sqrt(9)` 返回 `3.0`，不是整数 `3`。
2. `math.isqrt(10)` 返回 `3`，不是四舍五入结果。
3. `math.floor(-2.7)` 是 `-3`，`int(-2.7)` 是 `-2`。
4. `math.sin(30)` 把 `30` 当作弧度，不是 30 度。
5. `math.nan == math.nan` 为假。
6. 比较接近 0 的浮点数时，只设置 `rel_tol` 往往不够，需要设置 `abs_tol`。

## 面试应对

### `sqrt()` 和 `isqrt()` 有什么区别？

回答思路：从输入类型、返回类型、精度和使用场景回答。

回答模板：

`math.sqrt()` 计算实数平方根并返回浮点数，适合普通数值计算；`math.isqrt()` 只接受非负整数，返回平方根向下取整后的精确整数。算法题枚举约数、判断完全平方数或处理大整数时，我会优先使用 `isqrt()`，避免浮点精度问题。

### 为什么浮点数比较常用 `math.isclose()`？

回答思路：先说明二进制表示误差，再说明相对和绝对容忍度。

回答模板：

很多十进制小数不能被二进制浮点数精确表示，所以计算结果直接使用 `==` 可能失败。`math.isclose()` 会综合相对误差和绝对误差判断两个值是否足够接近。数值尺度较大时关注 `rel_tol`，与零比较或处理很小的数时要设置合适的 `abs_tol`。

### `math`、`cmath`、`decimal` 和 NumPy 如何选择？

回答思路：按数据域、精度和数据规模区分。

回答模板：

标量实数函数使用 `math`；输入可能是复数时使用 `cmath`；金额或规则明确的十进制舍入使用 `decimal`；需要对数组、矩阵或批量数据做向量化计算时使用 NumPy。选择依据主要是数值域、精度要求和是否需要批量计算。

### `atan()` 和 `atan2()` 有什么区别？

回答思路：说明象限信息和除零问题。

回答模板：

`atan(y / x)` 只看到比值，可能丢失象限信息，而且 `x` 为零时无法直接相除。`atan2(y, x)` 同时接收两个坐标，可以根据符号确定正确象限，也能处理坐标轴上的情况，因此计算二维方向角时通常使用 `atan2()`。

## 关联阅读

- [Python 数值运算与转换](<Python数值运算与转换.md#python-数值运算与转换>)
- [Python 官方文档：math](https://docs.python.org/3/library/math.html)
