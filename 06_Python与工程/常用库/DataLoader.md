# DataLoader

## 知识点解析

### 概述

这里主要以 PyTorch 为例，因为它是 `DataLoader` 概念非常突出的框架。`torch.utils.data.DataLoader` 的构造函数有很多参数，我们来介绍一些最常用和重要的：

1.  **`dataset` (必须)**：
    *   **作用**：这是 `DataLoader` 要加载的数据集对象。这个对象通常是你自定义的 `torch.utils.data.Dataset` 类的实例，或者 PyTorch 内置的一些数据集 (如 `torchvision.datasets.MNIST`)。
    *   **类型**：map-style `Dataset` 或 `IterableDataset` 对象。
    *   **要求**：map-style `Dataset` 通常实现 `__len__()` 和 `__getitem__(idx)`；流式或无法随机索引的数据可用 `IterableDataset`，主要实现 `__iter__()`，不要求实现前两者。

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
    *   **作用**：如果为 `True`，`DataLoader` 会在返回张量之前将它们复制到 CPU 页锁定内存（pinned memory）中。
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

**如何使用 `DataLoader`？**

下面是一个基本的使用流程和示例：

**步骤 1：准备你的 `Dataset`**

首先，你需要一个 `Dataset` 对象。它可以是 PyTorch 内置的，也可以是你自己定义的。

```python
import torch
from torch.utils.data import Dataset, DataLoader

### 自定义一个简单的 Dataset
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

### 假设我们有一些数据
### 特征数据 (例如100个样本，每个样本有10个特征)
features = torch.randn(100, 10)
### 标签数据 (例如100个样本，每个样本有一个标签)
labels = torch.randint(0, 2, (100,)) # 假设是二分类任务的标签

### 实例化你的 Dataset
my_dataset = MyCustomDataset(features, labels)
```

**步骤 2：实例化 `DataLoader`**

使用上面定义的 `my_dataset` 和一些参数来创建 `DataLoader`。

```python
### 实例化 DataLoader
batch_size = 16
num_workers = 2 # 根据你的CPU核心数调整

### 训练用的 DataLoader，通常需要打乱
train_loader = DataLoader(
    dataset=my_dataset,
    batch_size=batch_size,
    shuffle=True,       # 打乱数据
    num_workers=num_workers, # 使用2个子进程加载数据
    pin_memory=True,    # 如果使用GPU，可以设为True
    drop_last=False     # 不丢弃最后一个不完整的批次
)

### 验证或测试用的 DataLoader，通常不需要打乱
### 假设我们用同一个数据集做演示，实际中验证集和训练集是分开的
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
### 在训练循环中使用 DataLoader
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

### 参考网址

PyTorch入门必学：DataLoader（数据迭代器）参数解析与用法合集      https://blog.csdn.net/qq_41813454/article/details/134903615
## 面试应对

### Dataset、Sampler、collate_fn 和 DataLoader 分别负责什么？

回答思路：沿着“产生索引、读取样本、组装批次、调度迭代”的数据流说明职责。

完整模板：

`Dataset` 定义如何按索引读取一个样本；`Sampler` 决定以什么顺序产生样本索引；`BatchSampler` 把索引组成批次；`collate_fn` 把一组样本整理成模型需要的批张量；`DataLoader` 负责协调这些组件，并提供批量迭代、多进程预取和内存固定等能力。把职责拆开后，类别均衡采样、变长样本组批和并行加载都可以独立定制。

### num_workers 应该如何设置，为什么不是越大越好？

回答思路：从 CPU 预处理、进程通信、内存和存储吞吐四方面分析，并强调用吞吐实测选值。

完整模板：

`num_workers=0` 表示在主进程加载，便于调试；大于 0 时使用子进程并行准备数据，可以覆盖 GPU 计算期间的读取和预处理时间。但 worker 增多也会增加进程启动、序列化、内存占用和存储竞争，过大反而可能降低吞吐，甚至耗尽共享内存。我会观察 GPU 等待和每秒样本数，从 0、2、4 等值逐步测试；多轮训练还可以结合 `persistent_workers=True` 减少每个 epoch 重建进程的开销。

### pin_memory=True 为什么可能加速 GPU 训练？

回答思路：说明 pinned memory 是不可分页的 CPU 内存，并解释它与异步拷贝的配合关系及使用条件。

完整模板：

`pin_memory=True` 会让 DataLoader 把返回的 CPU 张量放入页锁定内存，CUDA 可以更高效地从这类内存传输数据。训练循环中再使用 `batch.to(device, non_blocking=True)`，才有机会让主机到 GPU 的拷贝与计算重叠。它主要适用于 CUDA 训练，不会把数据直接放进显存，也不保证所有任务都加速，因为固定内存和复制本身也有开销，需要结合吞吐测试。

### 如何保证 shuffle 和多进程加载可复现？

回答思路：分别处理主采样顺序、worker 内随机增强和分布式每轮洗牌三个随机性来源。

完整模板：

我会固定 Python、NumPy 和 PyTorch 的随机种子，并给 DataLoader 传入固定种子的 `torch.Generator` 来控制采样顺序。多进程下，每个 worker 会获得不同的 PyTorch 初始种子；如果 `Dataset` 或数据增强还使用 NumPy、Python `random` 等随机源，需要在 `worker_init_fn` 中根据 worker seed 分别初始化。分布式训练应使用 `DistributedSampler`，DataLoader 不再同时设置 `shuffle=True`，并在每个 epoch 调用 `sampler.set_epoch(epoch)`，让各进程得到不重叠且每轮变化的样本顺序。

### 变长样本如何组 batch，drop_last 应该何时使用？

回答思路：说明默认 collate 只能堆叠同形状张量，再区分 padding、保留列表和丢弃尾批次的用途。

完整模板：

默认 `collate_fn` 会沿 batch 维堆叠张量，因此文本序列、检测框等形状不同的样本会报错。此时应自定义 `collate_fn`：文本可以 padding 并同时返回长度或 attention mask，检测任务可以把不同数量的标注保留为列表。`drop_last=True` 只负责丢弃最后一个不足 `batch_size` 的批次，不解决变长样本问题；它适合要求固定批大小、使用 BatchNorm 或分布式对齐训练步数的场景，验证和测试通常应保留尾批次，避免漏算样本。
