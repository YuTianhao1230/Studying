# 常用函数

本目录整理 PyTorch/Python 中容易在训练和推理代码里反复遇到的小函数、小技巧。

## 内容索引

| 文件 | 内容说明 |
| --- | --- |
| [.detach().md](<.detach().md>) | detach 的计算图切断语义，以及日志、评测、显存泄漏场景。 |
| [torch.inference_mode().md](<torch.inference_mode().md>) | inference_mode 和 no_grad 的区别及推理加速场景。 |
| [enumerate.md](<enumerate.md>) | enumerate 在循环中同时获取索引和值的用法。 |
| [折叠注释.md](<折叠注释.md>) | 代码编辑器中的折叠注释和可读性技巧。 |

## 学习路线

1. 先看 [.detach().md](<.detach().md>) 和 [torch.inference_mode().md](<torch.inference_mode().md>)，理解训练/推理图管理。
2. 再看 [enumerate.md](<enumerate.md>)，补 Python 循环习惯。
3. 最后看 [折叠注释.md](<折叠注释.md>)，作为工程可读性补充。
