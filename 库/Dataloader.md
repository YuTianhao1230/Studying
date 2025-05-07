#  DataLoader的常用参数以及如何使用它。

这里主要以 PyTorch 为例，因为它是 `DataLoader` 概念非常突出的框架。`torch.utils.data.DataLoader` 的构造函数有很多参数，我们来介绍一些最常用和重要的：

1.  **`dataset` (必须)**：
    *   **作用**：这是 `DataLoader` 要加载的数据集对象。这个对象通常是你自定义的 `torch.utils.data.Dataset` 类的实例，或者 PyTorch 内置的一些数据集 (如 `torchvision.datasets.MNIST`)。
    *   **类型**：`Dataset` 对象。
    *   **要求**：`Dataset` 类必须实现 `__len__()` 方法（返回数据集大小）和 `__getitem__(idx)` 方法（根据索引 `idx` 返回一个数据样本）。

2.  **`batch_size` (可选)**：
    *   **作用**：指定每个批次加载多少个样本。
    *   **类型**：`int`。
    *   **默认值**：`1`。
    *   **说明**：这是最常用的参数之一。例如，`batch_size=32` 表示每次从数据集中取出32个样本组成一个批次。

3.  **`shuffle` (可选)**：
    *   **作用**：是否在每个 epoch 开始时打乱数据顺序。
    *   **类型**：`bool`。
    *   **默认值**：`False`。
    *   **说明**：在训练时，通常设置为 `True`，有助于模型学习到更通用的特征，防止过拟合。在验证或测试时，通常设置为 `False`，因为顺序不影响评估结果，且保持顺序有助于调试或可复现性。

4.  **`num_workers` (可选)**：
    *   **作用**：用于数据加载的子进程数量。
    *   **类型**：`int`。
    *   **默认值**：`0`。
    *   **说明**：
        *   `0` 表示数据将在主进程中加载（单进程）。
        *   大于 `0` 的值表示使用指定数量的子进程并行加载数据。这可以显著加快数据准备速度，尤其是在数据预处理比较耗时或模型在 GPU 上训练时，可以避免 CPU 成为瓶颈。
        *   设置多少合适？通常可以设置为 CPU 的核心数，但需要实验找到最佳值，过多的 `num_workers` 可能会因为进程间通信开销而降低效率。

5.  **`pin_memory` (可选)**：
    *   **作用**：如果为 `True`，`DataLoader` 会在返回张量之前将它们复制到 CUDA 的固定内存（pinned memory）中。
    *   **类型**：`bool`。
    *   **默认值**：`False`。
    *   **说明**：当使用 GPU 训练时，将数据从 CPU 内存传输到 GPU 显存是一个耗时操作。使用固定内存可以加快这个传输速度。通常在 `num_workers > 0` 且数据最终要传输到 GPU 时设置为 `True`。

6.  **`drop_last` (可选)**：
    *   **作用**：如果数据集大小不能被 `batch_size` 整除，最后一个批次可能会比 `batch_size` 小。如果设置为 `True`，则丢弃这个不完整的最后一个批次。
    *   **类型**：`bool`。
    *   **默认值**：`False`。
    *   **说明**：在某些情况下，模型可能要求输入的批次大小严格一致，这时可以将此参数设为 `True`。

7.  **`collate_fn` (可选)**：
    *   **作用**：一个自定义函数，用于将从 `Dataset` 中获取的多个样本（一个列表）合并成一个批次。
    *   **类型**：可调用对象 (callable)。
    *   **默认值**：`None` (使用 PyTorch 的默认合并逻辑，通常是将样本中的张量堆叠起来)。
    *   **说明**：当你的数据样本包含不同长度的序列（例如文本数据）或其他需要特殊处理的结构时，默认的 `collate_fn` 可能无法工作。这时你需要提供一个自定义的 `collate_fn` 来实现例如填充 (padding) 等操作，将它们整理成形状一致的张量批次。

8.  **`sampler` (可选)**：
    *   **作用**：定义从数据集中提取样本的策略。如果指定了 `sampler`，则 `shuffle` 参数必须为 `False` (或者不设置，默认为 `False`)。
    *   **类型**：`torch.utils.data.Sampler` 的子类实例。
    *   **说明**：`Sampler` 提供了更灵活的采样方式，例如 `RandomSampler` (随机采样，`shuffle=True` 内部就是用它), `SequentialSampler` (顺序采样), `WeightedRandomSampler` (带权重的随机采样，用于处理类别不平衡问题) 等。

---

**如何使用 `DataLoader`？**

下面是一个基本的使用流程和示例：

**步骤 1：准备你的 `Dataset`**

首先，你需要一个 `Dataset` 对象。它可以是 PyTorch 内置的，也可以是你自己定义的。

```python
import torch
from torch.utils.data import Dataset, DataLoader

# 1. 自定义一个简单的 Dataset
class MyCustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回一个样本（通常是 (特征, 标签) 对）
        sample_data = self.data[idx]
        sample_target = self.targets[idx]
        return sample_data, sample_target

# 假设我们有一些数据
# 特征数据 (例如100个样本，每个样本有10个特征)
features = torch.randn(100, 10)
# 标签数据 (例如100个样本，每个样本有一个标签)
labels = torch.randint(0, 2, (100,)) # 假设是二分类任务的标签

# 实例化你的 Dataset
my_dataset = MyCustomDataset(features, labels)
```

**步骤 2：实例化 `DataLoader`**

使用上面定义的 `my_dataset` 和一些参数来创建 `DataLoader`。

```python
# 2. 实例化 DataLoader
batch_size = 16
num_workers = 2 # 根据你的CPU核心数调整

# 训练用的 DataLoader，通常需要打乱
train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,       # 打乱数据
    num_workers=num_workers, # 使用2个子进程加载数据
    pin_memory=True,    # 如果使用GPU，可以设为True
    drop_last=False     # 不丢弃最后一个不完整的批次
)

# 验证或测试用的 DataLoader，通常不需要打乱
# 假设我们用同一个数据集做演示，实际中验证集和训练集是分开的
val_loader = DataLoader(
    dataset=my_dataset, # 实际应为 val_dataset
    batch_size=batch_size,
    shuffle=False,      # 不需要打乱
    num_workers=num_workers,
    pin_memory=True
)
```

**步骤 3：在训练/评估循环中迭代 `DataLoader`**

`DataLoader` 是一个可迭代对象，你可以像遍历列表一样遍历它，每次迭代会产出一个批次的数据。

```python
# 3. 在训练循环中使用 DataLoader
num_epochs = 5
for epoch in range(num_epochs):
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")

    # 训练阶段
    # model.train() # 将模型设置为训练模式 (如果使用如Dropout, BatchNorm等层)
    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        # batch_features 的形状通常是 [batch_size, feature_dim1, feature_dim2, ...]
        # batch_labels 的形状通常是 [batch_size] 或 [batch_size, num_classes]

        # 打印一些信息
        if batch_idx == 0 and epoch == 0: # 只在第一个epoch的第一个batch打印形状
            print(f"  Train Batch {batch_idx+1}:")
            print(f"    Features shape: {batch_features.shape}") # 应该是 torch.Size([16, 10])
            print(f"    Labels shape: {batch_labels.shape}")     # 应该是 torch.Size([16])

        # 在这里进行模型的前向传播、计算损失、反向传播、优化器更新等操作
        # outputs = model(batch_features)
        # loss = criterion(outputs, batch_labels)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        if (batch_idx + 1) % 5 == 0: # 每5个batch打印一次进度
             print(f"  Train Batch {batch_idx+1}/{len(train_loader)} processed.")
    print("Training for this epoch finished.")

    # 评估阶段 (可选)
    # model.eval() # 将模型设置为评估模式
    # with torch.no_grad(): # 在评估时不需要计算梯度
    #     for batch_features_val, batch_labels_val in val_loader:
    #         # 进行评估...
    #         pass
    # print("Validation for this epoch finished.")

print("--- Training complete ---")
```





## 参考网址

PyTorch入门必学：DataLoader（数据迭代器）参数解析与用法合集      https://blog.csdn.net/qq_41813454/article/details/134903615
