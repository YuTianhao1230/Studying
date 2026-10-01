# torch.unsqueeze()

## 知识点解析

### 概述

### 核心概念：它到底做了什么？

`torch.unsqueeze()` 的作用是：**在张量的指定位置插入一个大小为 1 的新维度。**

简单来说，就是给你的数据“升维”。你可以把它想象成：

*   **给数据套上一层新的括号 `[]`**。
*   **增加一个轴（axis）**。

这个操作**不会改变张量中元素的数量和值**，仅仅是改变了张量的“形状”和看待它的方式。

### 语法和参数

`unsqueeze()` 有两种使用形式，效果完全一样：

1.  **函数形式**: `torch.unsqueeze(input_tensor, dim)`
2.  **方法形式 (更常用)**: `input_tensor.unsqueeze(dim)`

它只有一个关键参数：

*   **`dim` (int)**: 你希望**插入新维度的位置（索引）**。这个参数的值可以是 `[-input.dim() - 1, input.dim()]` 范围内的整数。

**`dim` 参数的理解是关键：**
`dim` 指的是新维度的索引。我们用一个简单的例子来理解。

假设我们有一个1维张量（一个向量）：
```python
import torch

x = torch.tensor([1, 2, 3])
print(f"原始张量 x: {x}")
print(f"原始形状: {x.shape}") 
### torch.Size([3])
```

#### `dim=0` (在最前面插入)

```python
x_unsqueezed_0 = x.unsqueeze(0)

print(f"\nx.unsqueeze(0): {x_unsqueezed_0}")
print(f"新形状: {x_unsqueezed_0.shape}")
### torch.Size([1, 3])
```
**发生了什么？**
*   原始形状是 `(3)`。
*   我们在**位置 0** 插入了一个新维度。
*   新形状变成了 `(1, 3)`。
*   你可以看到，原始的 `[1, 2, 3]` 被**套上了一层新的括号**，变成了 `[[1, 2, 3]]`。这现在是一个1行3列的矩阵。

#### `dim=1` (在中间插入)

```python
x_unsqueezed_1 = x.unsqueeze(1)

print(f"\nx.unsqueeze(1): {x_unsqueezed_1}")
print(f"新形状: {x_unsqueezed_1.shape}")
### torch.Size([3, 1])
```
**发生了什么？**
*   原始形状是 `(3)`。
*   我们在**位置 1** 插入了一个新维度。
*   新形状变成了 `(3, 1)`。
*   你可以看到，`[1, 2, 3]` 中的**每个元素自己都被套上了一层括号**，变成了 `[[1], [2], [3]]`。这现在是一个3行1列的矩阵。

#### 使用负数索引
`dim` 也可以是负数，这在编程中非常方便。`-1` 表示倒数第一个位置，`-2` 表示倒数第二个，以此类推。

对于我们的一维张量 `x`，`dim=1` 和 `dim=-1` 是等价的，因为都是在最后一个维度之后插入。
```python
x_unsqueezed_neg1 = x.unsqueeze(-1)

print(f"\nx.unsqueeze(-1): {x_unsqueezed_neg1}")
print(f"新形状: {x_unsqueezed_neg1.shape}")
### torch.Size([3, 1])，和 dim=1 结果一样
```
```
以上所有的输出：
原始张量 x: tensor([1, 2, 3])
原始形状: torch.Size([3])

x.unsqueeze(0): tensor([[1, 2, 3]])
新形状: torch.Size([1, 3])

x.unsqueeze(1): tensor([[1],
        [2],
        [3]])
新形状: torch.Size([3, 1])

x.unsqueeze(-1): tensor([[1],
        [2],
        [3]])
新形状: torch.Size([3, 1])

```

### 为什么它如此重要？（核心应用场景）

`unsqueeze()` 的重要性体现在它解决了深度学习中频繁出现的“维度不匹配”问题。

#### 场景一：为模型添加 Batch 维度（最最常见的用法！）

**问题**: 深度学习模型（如 `nn.Conv2d`, `nn.Linear`）通常被设计为处理**一批 (batch)** 数据，而不是单个样本。因此，它们的输入期望是一个带有 "batch" 维度的张量。

例如，一个典型的图像模型期望的输入形状是 `[Batch, Channels, Height, Width]`，即 `[B, C, H, W]`。

但是，当你从数据集中加载**单个**图像时，它的形状通常是 `[C, H, W]`。

**解决方案**: 使用 `unsqueeze(0)` 在最前面添加一个大小为 1 的 batch 维度。

```python
### 假设我们加载了一张 3 通道、224x224 的图像
single_image = torch.rand(3, 224, 224) 
print(f"单张图片的形状: {single_image.shape}") # torch.Size([3, 224, 224])

### 模型无法处理这个形状，我们需要一个 batch 维度
### model(single_image)  # <-- 这会报错！

### 使用 unsqueeze(0) 在第 0 维添加 batch 维度
batched_image = single_image.unsqueeze(0)
print(f"添加 batch 维度后的形状: {batched_image.shape}") # torch.Size([1, 3, 224, 224])

### 现在可以安全地把它喂给模型了
### model(batched_image) # <-- OK!
```
**`squeeze()` 的作用**: 相应地，当模型输出一个 batch 的结果（例如形状为 `[1, 10]` 的分类得分），而你只想获取单个样本的结果时，可以使用 `squeeze(0)` 来移除 batch 维度，得到 `[10]`。

#### 场景二：实现广播（Broadcasting）

**问题**: 当你希望对两个形状不同的张量进行运算（如加、减、乘、除）时，需要遵循 PyTorch 的广播机制。有时，你需要手动调整维度来使广播能够正确发生。

**例子**: 你有一个 `(3, 4)` 的矩阵，和一个 `(3,)` 的向量。你希望将向量中的每个元素分别加到矩阵的每一行上。直接相加会出错（在某些旧版本或不同库中可能行为不一，但通常不符合预期）。

```python
matrix = torch.ones(3, 4)
vector = torch.tensor([10, 20, 30])

### matrix.shape -> torch.Size([3, 4])
### vector.shape -> torch.Size([3])

### print(matrix + vector) # <-- 会报错，因为维度不匹配
### RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1

### 解决方案：将 vector 的形状从 (3,) 变为 (3, 1)
vector_unsqueezed = vector.unsqueeze(1)
print(f"\nVector 的新形状: {vector_unsqueezed.shape}") # torch.Size([3, 1])

### 现在可以进行广播了
### (3, 4) + (3, 1) -> PyTorch 会将 (3, 1) 的列复制 4 次，变成 (3, 4) 再相加
result = matrix + vector_unsqueezed
print("广播相加后的结果:\n", result)
```
输出：
```
Vector 的新形状: torch.Size([3, 1])
广播相加后的结果:
 tensor([[11., 11., 11., 11.],
        [21., 21., 21., 21.],
        [31., 31., 31., 31.]])
```
通过 `unsqueeze(1)`，我们明确告诉 PyTorch：“这个向量是列向量，请将它广播到 `matrix` 的所有列上。”

### `unsqueeze()` vs. `view()`/`reshape()`

初学者可能会混淆这几个函数。

*   **`unsqueeze(dim)`**: **只负责添加**一个大小为 1 的新维度。它非常专一、意图明确。
*   **`view(shape)` / `reshape(shape)`**: **重新组织**张量中的所有元素以匹配新的形状。它们更通用，但前提是新旧形状的**元素总数必须完全一致**。

虽然 `x.unsqueeze(0)` 的效果和 `x.view(1, *x.shape)` 是一样的，但使用 `unsqueeze(0)` 的代码可读性更高，更能清晰地表达“我正在添加一个 batch 维度”这个意图。

### 总结与黄金法则

*   **核心功能**: 在指定位置 `dim` 添加一个大小为 1 的新维度。
*   **黄金法则**: **如果你需要给数据升维，但不想改变现有维度的顺序和大小，就用 `unsqueeze()`。**
*   **最常用法**: `tensor.unsqueeze(0)`，用于给单个样本数据添加一个 batch 维度，以便送入模型。
*   **它的搭档**: `squeeze()`，用于移除大小为 1 的维度，是 `unsqueeze()` 的逆操作。
## 面试应对

### torch.unsqueeze() 做了什么，dim 应该怎样理解？

回答思路：说明它在结果形状的 `dim` 位置插入长度为 1 的轴，并用正、负索引举例。

完整模板：

`torch.unsqueeze()` 在指定位置插入一个长度为 1 的维度，不改变元素数量和排列。例如形状 `[3, 4]` 的张量执行 `unsqueeze(0)` 后是 `[1, 3, 4]`，执行 `unsqueeze(1)` 后是 `[3, 1, 4]`，执行 `unsqueeze(-1)` 后是 `[3, 4, 1]`。对一个 `n` 维张量，`dim` 可以取 `[-n-1, n]`，负索引适合表达从末尾插入。

### unsqueeze 为什么通常不复制数据，会影响自动求导吗？

回答思路：解释它返回共享底层存储的视图，形状元数据改变但计算图仍然保留。

完整模板：

`unsqueeze` 通常返回与原张量共享底层存储的 view，只修改尺寸和步长等形状信息，因此开销很小。它是可微的形状操作，不会截断计算图，反向传播时梯度会按对应形状映射回原张量。因为共享存储，对 view 做原地写入可能影响原张量；如果还涉及需要梯度的叶子张量，原地操作也可能触发 autograd 的限制。

### 用 unsqueeze 配合广播时如何确定插入位置？

回答思路：先写出目标形状，再按广播从尾维对齐的规则补长度为 1 的维度。

完整模板：

广播从最后一个维度向前比较，每一维必须相等，或者其中一个为 1。比如 `[B, C, H, W]` 的特征要乘 `[C]` 的通道权重，不能直接依赖尾维对齐，而应把权重变成 `[1, C, 1, 1]`，可以连续使用 `unsqueeze(0).unsqueeze(-1).unsqueeze(-1)`。我会先写出输入和目标形状，再决定在哪些位置补 1，避免把 batch、channel 和 sequence 维混淆。

### unsqueeze、squeeze、view 和 reshape 有什么区别？

回答思路：按“增删单位维”和“重排整体形状”区分，并指出 `squeeze()` 不指定维度的风险。

完整模板：

`unsqueeze(dim)` 只增加一个长度为 1 的维度，`squeeze(dim)` 只在该维长度为 1 时删除它；`view()` 和 `reshape()` 则可以在元素总数不变的前提下重组整体形状。`view()` 通常要求内存布局兼容，`reshape()` 必要时可以创建副本。工程中我会用 `unsqueeze` 明确表达添加 batch 或广播维；删除维度时优先写 `squeeze(dim)`，避免无参数 `squeeze()` 把 batch size 为 1 等仍有语义的维度一起删掉。
