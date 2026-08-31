# torch.inference_mode()

## 知识点解析

### 概述

`torch.inference_mode()` 是 PyTorch 1.9 版本引入的一个上下文管理器。

简单来说，它是 `torch.no_grad()` 的**升级版**或**更彻底的纯净版**。

如果你的目的仅仅是**推理（Inference）**——即跑一次模型得到结果，完全不需要反向传播，那么官方现在的建议是：**总是优先使用 `torch.inference_mode()`**。

### 核心区别：它比 `no_grad` 多关了什么？

要理解它为什么更快，得先知道 PyTorch 默默在后台做了什么。

*   **`torch.no_grad()`**：告诉 PyTorch 引擎“别记录计算图了，我不需要反向传播”。这省去了构建计算图的开销。
*   **`torch.inference_mode()`**：除了做 `no_grad` 做的事之外，它还关闭了 **“视图追踪（View Tracking）”** 和 **“版本计数（Version Counter）”**。

#### 什么是版本追踪？
当你对一个 Tensor 进行操作时，PyTorch 会悄悄记录这个 Tensor 被修改了多少次（版本号）。这主要是为了防止你做 In-place 操作（原地修改）导致梯度计算出错。
而在 `inference_mode` 下，PyTorch 假设你根本不会算梯度，所以它连“版本号”都懒得记了。

### 带来的好处

1.  **更快的速度**：少了很多后台的簿记工作（Bookkeeping），C++层面的开销更小。虽然在小模型上不明显，但在高频调用或深度优化场景下会有性能提升。
2.  **更省内存**：生成的 Tensor 更加轻量化。
3.  **更安全（报错更明确）**：在 `inference_mode` 下生成的 Tensor，如果你试图在外面强行给它开梯度或者做某些奇怪的反向传播操作，PyTorch 会直接报错，而不是产生不可预知的静默错误。

### 写法对比

写法和 `no_grad` 完全一样，可以作为 `with` 块，也可以作为装饰器。

#### 作为上下文管理器（最常用）
```python
import torch

model.eval()  # 别忘了这个，把 Dropout/BatchNorm 设为测试模式

### 以前你写 no_grad
### with torch.no_grad():
### pred = model(x)

### 现在推荐写 inference_mode
with torch.inference_mode():
    pred = model(x)
```

#### 作为装饰器
```python
@torch.inference_mode()
def predict(data):
    return model(data)
```

### 什么时候**不能**用它？

虽然它是 `no_grad` 的上位替代，但有一种极少数的情况不能替代：

如果你在 `with` 块里计算了一个 Tensor，然后你**后面又想把这个 Tensor 用到另一个需要求导的计算图中**。

*   `torch.no_grad()` 出来的 Tensor 只是 `requires_grad=False`。
*   `torch.inference_mode()` 出来的 Tensor 是一种特殊的“推理张量”，它被限制得更死。如果你试图在一个开启了梯度的计算中使用这种 Tensor，可能会报错 `RuntimeError: Inference tensors cannot be saved for backward...`。

### 总结

*   **`torch.no_grad()`**：经典老牌，兼容性最好。如果你要在不求导的代码块里搞一些骚操作（后续又要接回求导流程），用它。
*   **`torch.inference_mode()`**：**现代标准**。只要你是做**模型验证（Validation）、测试（Test）或 生产环境部署（Deployment）**，请无脑使用它。它更快、更省、更纯粹。
## 面试应对

### torch.inference_mode() 是什么？

回答思路：点明它是 no_grad 的升级版，除关 autograd 还禁掉版本计数和视图追踪，专为纯推理提速。

回答模板：

`torch.inference_mode()` 是 PyTorch 的推理上下文管理器，会关闭 autograd，并进一步禁用一些版本计数和视图追踪开销。它比 `torch.no_grad()` 更激进，适合纯推理场景。

### torch.inference_mode() 适合什么场景？

回答思路：抓住验证/线上服务/离线生成这类纯推理才用，输出还要再参与求导就不能放进去。

回答模板：

推理评测、线上服务、离线生成都适合用它；如果后续还要对输出继续参与梯度计算，就不应该放在这个上下文里。

### torch.inference_mode() 常见坑是什么？

回答思路：抓住推理张量不能回传做 backward、以及它不替代 model.eval() 这两个关键坑。

回答模板：

inference_mode 最大的坑是它产生的是特殊的“推理张量”，一旦你想把这个输出再放回一个需要求导的计算图（比如继续训练或二次反传），会直接报 `Inference tensors cannot be saved for backward`，这种场景要退回用 `torch.no_grad()`。另一个常见错误是以为它能替代 `model.eval()`：它只关 autograd，不会自动切换 Dropout/BatchNorm 的行为，所以推理前仍要单独调 `model.eval()`。
