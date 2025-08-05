### 1. `torchvision.models` (模型库)

`torchvision.models` 模块包含了众多预训练好的、顶尖的计算机视觉模型，如 ResNet, VGG, MobileNet, Vision Transformer (ViT) 等。这些模型都在大型数据集（通常是 ImageNet）上进行了训练，学会了提取通用的图像特征。我们可以直接使用它们，或者在此基础上进行微调（Fine-tuning）以适应我们自己的任务。

#### **学什么 (What to Learn)**

1.  **加载预训练模型**: 如何从仓库中拉取一个模型架构，并加载它已经训练好的权重。
2.  **新旧 API 对比**: 理解为什么新的 `weights` 参数比旧的 `pretrained=True` 更好。
    *   **旧 `pretrained=True`**: 简单直接，但不够灵活，无法指定权重的版本或来源。
    *   **新 `weights` API**: 更现代、更推荐的方式。它允许你精确指定权重的版本（例如 `ResNet18_Weights.IMAGENET1K_V1`），并且这些权重对象还附带了元数据，比如它是在哪个数据集上训练的、推荐的预处理步骤（transforms）是什么。
3.  **查看模型结构**: 加载模型后，如何打印出它的所有层级结构。这对于理解模型和后续修改至关重要。
4.  **修改模型 (迁移学习)**: 核心技能。预训练模型的最后通常是一个全连接层（`Linear` 或 `fc`），其输出维度等于原始任务的类别数（如 ImageNet 的 1000 类）。在你的任务中，比如一个 10 分类问题，你需要将这个层替换成一个输出为 10 的新层。

#### **怎么学 (How to Learn)**

让我们通过代码来实践以上几点。

```python
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights

# --- 1. 加载不同的预训练模型 (使用新的 weights API) ---
print("="*20 + " 加载 ResNet-18 " + "="*20)
# 加载 ResNet-18，并使用 ImageNet V1 版本的预训练权重
weights_resnet = ResNet18_Weights.IMAGENET1K_V1
model_resnet = models.resnet18(weights=weights_resnet)

# 新的 weights API 还自带了推荐的预处理变换！
# 这非常方便，确保你的输入数据处理方式和模型训练时一致
preprocess_resnet = weights_resnet.transforms()
print("ResNet-18 推荐的预处理:\n", preprocess_resnet)


print("\n" + "="*20 + " 加载 MobileNet V2 " + "="*20)
# 加载 MobileNet V2
weights_mobilenet = MobileNet_V2_Weights.IMAGENET1K_V2
model_mobilenet = models.mobilenet_v2(weights=weights_mobilenet)


# --- 2. 查看模型结构 ---
print("\n" + "="*20 + " ResNet-18 原始结构 " + "="*20)
print(model_resnet)
# 注意看最后一行：(fc): Linear(in_features=512, out_features=1000, bias=True)

print("\n" + "="*20 + " MobileNet V2 原始结构 " + "="*20)
print(model_mobilenet)
# 注意看最后一部分：(classifier): Sequential(...) 包含一个 Linear(in_features=1280, out_features=1000, bias=True)


# --- 3. 修改模型的最后一层以适应10分类任务 ---
print("\n" + "="*20 + " 修改 ResNet-18 " + "="*20)

# 获取 ResNet-18 全连接层的输入特征数
num_features_resnet = model_resnet.fc.in_features
print(f"ResNet-18 fc 层的输入特征数: {num_features_resnet}")

# 定义新的类别数
num_classes = 10

# 替换原来的 fc 层
model_resnet.fc = torch.nn.Linear(num_features_resnet, num_classes)

print("\n" + "="*20 + " ResNet-18 修改后结构 " + "="*20)
print(model_resnet)
# 现在看最后一行，out_features 已经变成 10 了！
# (fc): Linear(in_features=512, out_features=10, bias=True)


print("\n" + "="*20 + " 修改 MobileNet V2 " + "="*20)
# 对于 MobileNet V2，分类层在 `classifier` 属性里
num_features_mobilenet = model_mobilenet.classifier[1].in_features
print(f"MobileNet V2 分类层的输入特征数: {num_features_mobilenet}")

model_mobilenet.classifier[1] = torch.nn.Linear(num_features_mobilenet, num_classes)

print("\n" + "="*20 + " MobileNet V2 修改后结构 " + "="*20)
# 打印 classifier 部分来确认修改
print(model_mobilenet.classifier)
# 现在看最后一个 Linear 层，out_features 也是 10 了！
```

---

### 2. `torchvision.datasets` (数据集)

这个模块提供了对主流计算机视觉数据集的便捷访问，如 MNIST, CIFAR10, ImageNet 等。更重要的是，它提供了一个标准接口（`torch.utils.data.Dataset`），让你能够轻松加载自己的数据集。

#### **学什么 (What to Learn)**

1.  **使用内置数据集**: 如何用一行代码下载并加载标准数据集。关键参数包括 `root` (数据存放路径), `train` (选择训练集或测试集), `download` (是否自动下载), และ `transform` (应用在数据上的变换)。
2.  **加载自定义数据集**: 这是实际项目中最常用的技能。你需要创建一个类，继承 `torch.utils.data.Dataset`，并实现三个核心方法：
    *   `__init__(self, ...)`: 构造函数。通常在这里完成初始化工作，比如读取所有图片文件的路径、加载标签等。
    *   `__len__(self)`: 返回数据集中的样本总数。
    *   `__getitem__(self, idx)`: 核心方法。根据索引 `idx`，加载对应的图片和标签，对图片进行变换（transform），然后返回 `(image, label)` 对。

#### **怎么学 (How to Learn)**

**练习1: 加载 CIFAR10**

```python
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# 定义一个简单的变换，将图片转为 Tensor
transform = transforms.ToTensor()

# 加载 CIFAR10 训练集
cifar10_train = datasets.CIFAR10(
    root='./data',      # 数据将下载到 ./data 目录下
    train=True,         # 加载训练集
    download=True,      # 如果 ./data 下没有，就自动下载
    transform=transform # 应用变换
)

# 查看数据集大小
print(f"CIFAR10 训练集大小: {len(cifar10_train)}")

# 获取第一个样本
image, label = cifar10_train[0]
print(f"第一个样本的类型: Image - {type(image)}, Label - {type(label)}")
print(f"图片张量的形状: {image.shape}") # [C, H, W] -> [3, 32, 32]
print(f"标签: {label}") # 这是一个数字，代表类别索引
```

**练习2: 为“猫狗大战”数据集编写 CustomDataset**

假设你的数据存放结构如下：
```
/path/to/your/data/cats_vs_dogs/
├── cat.0.jpg
├── cat.1.jpg
...
├── dog.0.jpg
├── dog.1.jpg
...
```

```python
import os
from torch.utils.data import Dataset
from PIL import Image

class CatDogDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含所有图片的目录路径.
            transform (callable, optional): 应用于样本的可选变换.
        """
        self.root_dir = root_dir
        self.transform = transform
        # 获取所有图片文件名
        self.img_files = os.listdir(root_dir)
        
    def __len__(self):
        # 返回数据集大小
        return len(self.img_files)

    def __getitem__(self, idx):
        # 1. 获取图片路径
        img_path = os.path.join(self.root_dir, self.img_files[idx])
        
        # 2. 加载图片
        image = Image.open(img_path).convert("RGB") # 确保是 RGB 格式
        
        # 3. 从文件名获取标签
        # 'cat.0.jpg' -> 'cat' -> 0
        # 'dog.0.jpg' -> 'dog' -> 1
        label_name = self.img_files[idx].split('.')[0]
        label = 0 if label_name == 'cat' else 1
        
        # 4. 应用变换 (如果定义了)
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 使用你的 CustomDataset
# 假设你的猫狗图片在一个名为 'cats_and_dogs_images' 的文件夹里
# (你需要自己创建一个这样的文件夹并放入几张猫狗图片来运行此代码)
# if not os.path.exists('cats_and_dogs_images'):
#     os.makedirs('cats_and_dogs_images')
#     print("请在 'cats_and_dogs_images' 文件夹中放入一些 'cat.x.jpg' 和 'dog.x.jpg' 格式的图片")

# my_transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor()
# ])

# cat_dog_dataset = CatDogDataset(root_dir='cats_and_dogs_images', transform=my_transform)

# if len(cat_dog_dataset) > 0:
#     # 像使用内置数据集一样使用它
#     img, lbl = cat_dog_dataset[0]
#     print(f"自定义数据集中第一个样本: ")
#     print(f"图片形状: {img.shape}")
#     print(f"标签: {lbl} (0=cat, 1=dog)")
# else:
#     print("自定义数据集中没有找到图片。")
```
*注意：你需要先从 Kaggle 下载 "Dogs vs. Cats" 数据集，并整理成上述的文件结构才能运行 `CatDogDataset` 的示例。*

---

### 3. `torchvision.transforms` (图像变换)

这个模块是数据预处理和数据增强的工具箱。
*   **数据预处理**: 确保数据格式和数值范围符合模型输入要求（如转为 Tensor, 归一化）。
*   **数据增强**: 在训练期间对图像进行随机变换（如翻转、裁剪、颜色抖动），以增加数据的多样性，提高模型的泛化能力，防止过拟合。

#### **学什么 (What to Learn)**

1.  **常用变换**:
    *   `Resize(size)`: 将输入图片调整到指定尺寸。
    *   `ToTensor()`: **至关重要**。将 PIL Image 或 NumPy `ndarray` (H x W x C) 转换为 PyTorch Tensor (C x H x W)，并将像素值从 `[0, 255]` 缩放到 `[0.0, 1.0]`。
    *   `Normalize(mean, std)`: 用给定的均值和标准差对 Tensor 进行归一化。公式是 `output = (input - mean) / std`。`mean` 和 `std` 通常是 ImageNet 数据集的统计值，以匹配预训练模型。
    *   `RandomHorizontalFlip(p=0.5)`: 以概率 `p` 水平翻转图片。
    *   `RandomRotation(degrees)`: 在 `(-degrees, +degrees)` 范围内随机旋转图片。
    *   `ColorJitter(...)`: 随机改变图片的亮度、对比度、饱和度和色调。
2.  **`transforms.Compose([...])`**: 将多个变换操作串联成一个流水线。列表中的变换会按顺序依次执行。

#### **怎么学 (How to Learn)**

让我们加载一张本地图片，对它应用不同的变换流水线，并观察结果。

你需要准备一张名为 `my_image.jpg` 的图片放在代码同级目录下。

```python
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 确保你有一张名为 'my_image.jpg' 的图片
# try:
#     # 1. 加载一张本地图片
#     img = Image.open('my_image.jpg')
# except FileNotFoundError:
#     print("错误: 'my_image.jpg' 未找到。请在脚本目录放置一张图片。")
#     # 创建一个虚拟图片以便代码能运行
#     img = Image.new('RGB', (200, 150), color = 'red')

# 2. 定义不同的变换流水线

# 流水线A: 用于验证或测试的"标准预处理"
# (调整尺寸 -> 转为张量 -> 归一化)
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet等模型常使用224x224的输入
    transforms.ToTensor(),
    # 使用ImageNet的均值和标准差进行归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 流水线B: 用于训练的"数据增强"
# (调整尺寸 -> 随机翻转 -> 颜色抖动 -> 转为张量 -> 归一化)
augment_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 50%概率翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15), # 随机旋转-15到+15度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. 应用变换并观察输出

# 应用标准预处理
processed_tensor = preprocess_transform(img)
print(f"标准预处理后的张量形状: {processed_tensor.shape}")
print(f"张量类型: {processed_tensor.dtype}")
print(f"张量数值范围 (大约): Min={processed_tensor.min():.2f}, Max={processed_tensor.max():.2f}")


# 为了可视化，我们需要一个反归一化的函数
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m) # 逆操作: t = t * std + mean
    return tensor

def show_tensor_image(tensor):
    # 反归一化
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    denormalized_tensor = denormalize(tensor.clone(), mean, std)
    
    # 将张量转换回 PIL Image 以便显示
    image = transforms.ToPILImage()(denormalized_tensor)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# 显示经过标准预处理的图片
# print("\n显示标准预处理后的图片:")
# show_tensor_image(processed_tensor)

# 多次应用数据增强变换，观察每次结果的不同
# print("\n多次应用数据增强，观察随机效果:")
# for i in range(3):
#     augmented_tensor = augment_transform(img)
#     print(f"第 {i+1} 次增强:")
#     show_tensor_image(augmented_tensor)

```

### 总结

这三个模块协同工作，构成了典型的 PyTorch 视觉任务流程：

1.  使用 **`torchvision.datasets`** (或你自己的 `Dataset` 子类) 来准备和加载数据集。
2.  在加载数据的同时，通过 **`torchvision.transforms`** 定义的 `transform` 流水线对每张图片进行预处理和数据增强。
3.  将处理好的数据输入到从 **`torchvision.models`** 加载并修改过的模型中，进行训练或推理。

掌握了这三者，你就掌握了用 PyTorch 处理绝大多数计算机视觉问题的基础。
