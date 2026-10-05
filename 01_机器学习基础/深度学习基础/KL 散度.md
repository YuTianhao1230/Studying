# KL 散度

## 知识点解析

### 一句话理解

**KL 散度（Kullback-Leibler Divergence）衡量：真实数据来自分布 $P$，却使用分布 $Q$ 去描述或编码时，会多付出多少信息代价。**

它比较的是两个**概率分布**，而不是两个普通数值或向量。KL 散度越小，说明 $Q$ 越接近 $P$；等于 0 时，两者几乎处处相同。

### 定义

对于离散分布：

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_i P(i)\log\frac{P(i)}{Q(i)}
=\mathbb{E}_{x\sim P}\left[\log\frac{P(x)}{Q(x)}\right].
$$

对于连续分布：

$$
D_{\mathrm{KL}}(P\|Q)
=\int p(x)\log\frac{p(x)}{q(x)}\,dx.
$$

公式中的两个位置含义不同：

- $P$：参考分布、目标分布或数据分布，决定在哪些区域计算期望。
- $Q$：被比较、被优化的近似分布。
- $\log\frac{P(i)}{Q(i)}$：事件 $i$ 在两个分布下的信息差。

若使用自然对数，单位是 **nat**；若使用以 2 为底的对数，单位是 **bit**。

### 直觉与例子

设：

$$
P=[0.5,0.5],\qquad Q=[0.9,0.1].
$$

则：

$$
D_{\mathrm{KL}}(P\|Q)
=0.5\log\frac{0.5}{0.9}
+0.5\log\frac{0.5}{0.1}
\approx 0.511.
$$

$P$ 认为两个事件同样常见，但 $Q$ 严重低估了第二个事件，因此会受到较大惩罚。特别地，当 $P(i)>0$ 而 $Q(i)=0$ 时：

$$
D_{\mathrm{KL}}(P\|Q)=+\infty.
$$

这表示 $Q$ 认为一个实际可能发生的事件“绝不可能发生”，编码或建模会彻底失败。约定 $P(i)=0$ 的项贡献为 0。

### 核心性质

#### 1. 非负性

$$
D_{\mathrm{KL}}(P\|Q)\ge 0.
$$

当且仅当 $P=Q$（几乎处处成立）时取 0。虽然公式中的单个求和项可以为负，但总和不会为负。

#### 2. 不对称

通常：

$$
D_{\mathrm{KL}}(P\|Q)\ne D_{\mathrm{KL}}(Q\|P).
$$

原因是期望所依据的分布不同：

- **正向 KL：$D_{\mathrm{KL}}(P\|Q)$**  
  更关心 $Q$ 是否覆盖了 $P$ 的高概率区域。漏掉 $P$ 的某个主要模式会受到很大惩罚，因此常表现为 **mode-covering**。
- **反向 KL：$D_{\mathrm{KL}}(Q\|P)$**  
  更关心 $Q$ 采样到的位置在 $P$ 下是否合理。为了避免落入 $P$ 的低概率区域，$Q$ 可能只集中于一个主要模式，因此常表现为 **mode-seeking**。

例如，当 $P$ 是相距很远的双峰分布，而 $Q$ 只能表示单峰分布时，最小化正向 KL 往往让 $Q$ 尽量覆盖两个峰；最小化反向 KL 往往让 $Q$ 选择其中一个峰。

#### 3. 不是严格的距离

KL 散度不满足对称性，也不满足三角不等式，因此不能称为严格的数学距离。需要对称度量时，可以考虑 Jensen-Shannon 散度：

$$
M=\frac{P+Q}{2},
$$

$$
D_{\mathrm{JS}}(P\|Q)
=\frac12D_{\mathrm{KL}}(P\|M)
+\frac12D_{\mathrm{KL}}(Q\|M).
$$

### KL 散度与交叉熵的关系

熵、交叉熵和 KL 散度分别为：

$$
H(P)=-\sum_i P(i)\log P(i),
$$

$$
H(P,Q)=-\sum_i P(i)\log Q(i),
$$

$$
D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P).
$$

训练时若目标分布 $P$ 固定，则 $H(P)$ 与模型参数无关，因此：

$$
\arg\min_Q H(P,Q)
=\arg\min_Q D_{\mathrm{KL}}(P\|Q).
$$

这就是分类任务中“最小化交叉熵”等价于“让预测分布逼近目标分布”的原因。

当标签是 one-hot 分布，真实类别为 $y$ 时，$H(P)=0$，因此：

$$
D_{\mathrm{KL}}(P\|Q)=H(P,Q)=-\log Q(y).
$$

对于软标签，$H(P)$ 通常不为 0；交叉熵和 KL 的数值不同，但在固定 $P$ 时具有相同的最优解和梯度。

## 典型应用

### 1. 知识蒸馏

在[知识蒸馏](<../../03_训练优化与对齐/后训练与对齐/Knowledge Distillation 知识蒸馏.md#knowledge-distillation-知识蒸馏>)中，教师分布作为目标 $P_T$，学生分布作为近似 $P_S$：

$$
\mathcal{L}_{\mathrm{KD}}
=T^2D_{\mathrm{KL}}\left(P_T^{(T)}\|P_S^{(T)}\right),
$$

$$
P_T^{(T)}=\operatorname{softmax}\left(\frac{z_T}{T}\right),
\qquad
P_S^{(T)}=\operatorname{softmax}\left(\frac{z_S}{T}\right).
$$

- 温度 $T>1$ 会让概率分布更平滑，暴露类别之间的相似关系。
- 除以 $T$ 会使 logits 梯度缩小，经典蒸馏通常乘回 $T^2$ 补偿梯度尺度。
- 实际总损失通常还会加入学生对真实标签的交叉熵：

$$
\mathcal{L}
=\alpha\mathcal{L}_{\mathrm{hard}}
+(1-\alpha)\mathcal{L}_{\mathrm{KD}}.
$$

### 2. 变分自编码器

VAE 使用近似后验 $q_\phi(z\mid x)$ 逼近难以直接计算的真实后验，同时约束它不要偏离先验 $p(z)$：

$$
\mathcal{L}_{\mathrm{VAE}}
=\mathcal{L}_{\mathrm{recon}}
+\beta D_{\mathrm{KL}}\left(q_\phi(z\mid x)\|p(z)\right).
$$

重构项要求隐变量保留输入信息，KL 项把隐空间约束得连续、规则并便于从先验采样。若 KL 权重过大，可能出现后验坍缩，即编码器几乎不再利用输入信息。

### 3. 强化学习与大模型对齐

RLHF 等方法常约束当前策略 $\pi_\theta$ 不要偏离参考策略 $\pi_{\mathrm{ref}}$：

$$
\max_\theta\ 
\mathbb{E}[r(x,y)]
-\beta D_{\mathrm{KL}}
\left(\pi_\theta(\cdot\mid x)\|
\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

奖励项推动模型获得更高奖励，KL 惩罚项限制策略更新幅度，避免模型为了钻奖励模型的漏洞而严重偏离原始能力。具体算法可能使用采样估计、逐 token 近似或不同方向的 KL，阅读实现时必须确认定义。

## PyTorch 实现

`torch.nn.functional.kl_div` 的参数约定容易写反：

- `input` 默认必须是 **log-probability**。
- `target` 默认是普通 **probability**。
- `reduction="batchmean"` 才与“各样本 KL 求和后取 batch 均值”的数学定义一致。

```python
import torch.nn.functional as F

temperature = 4.0

teacher_prob = F.softmax(teacher_logits / temperature, dim=-1)
student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)

loss_kd = F.kl_div(
    student_log_prob,
    teacher_prob,
    reduction="batchmean",
) * temperature**2
```

这里计算的是：

$$
D_{\mathrm{KL}}(P_{\mathrm{teacher}}\|P_{\mathrm{student}}),
$$

尽管函数调用中学生分布写在第一个参数位置。原因是 PyTorch 的第一个参数表示公式中的 $\log Q$，第二个参数才表示目标分布 $P$。

若目标也以对数概率传入，需要设置 `log_target=True`。实际训练应优先使用 `log_softmax`，避免先计算很小的概率再取对数造成数值不稳定。

## 常见误区

1. **“KL 越大，两个分布的欧氏距离就越大。”**  
   不一定。KL 对低概率事件和支持集是否覆盖非常敏感，与普通几何距离含义不同。

2. **“交换 $P$、$Q$ 不影响结果。”**  
   错误。两个方向的数值、梯度和优化行为通常都不同。

3. **“KL 散度可以直接比较 logits。”**  
   错误。KL 比较的是归一化概率分布，应先通过 Softmax/LogSoftmax 得到概率或对数概率。

4. **“交叉熵和 KL 完全相等。”**  
   只有在目标熵 $H(P)=0$ 等特殊情况下数值才相等。固定目标分布时，两者只相差常数 $H(P)$，所以优化目标等价。

5. **“PyTorch `kl_div` 的第一个参数就是公式中的 $P$。”**  
   错误。第一个参数默认是被拟合分布的对数概率 $\log Q$，第二个参数是目标分布 $P$。

## 面试应对

### 什么是 KL 散度？

回答模板：

KL 散度衡量用分布 $Q$ 近似目标分布 $P$ 时产生的额外信息代价，定义为 $P$ 分布下 $\log(P/Q)$ 的期望。它非负，两个分布相同时为 0，但不对称且不满足三角不等式，因此不是严格的距离。

### KL 散度为什么不对称？

回答模板：

$D_{\mathrm{KL}}(P\|Q)$ 在 $P$ 下取期望，重点惩罚 $Q$ 漏掉 $P$ 的高概率区域；交换方向后改为在 $Q$ 下取期望，惩罚重点随之改变。正向 KL 通常偏向覆盖多个模式，反向 KL 通常偏向选择高概率模式。

### KL 散度和交叉熵是什么关系？

回答模板：

$D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P)$。当目标分布 $P$ 固定时，其熵 $H(P)$ 是常数，所以最小化交叉熵与最小化 KL 散度等价。one-hot 标签的熵为 0，此时两者数值也相等。

### 为什么知识蒸馏要使用温度？

回答模板：

较高温度会软化教师输出，显露非真实类别之间的相对关系，这些信息是 one-hot 标签没有提供的。由于温度缩放会使梯度减小，经典蒸馏通常在 KL 损失外乘 $T^2$，再与真实标签的交叉熵加权求和。
