import math
from decimal import Decimal, ROUND_HALF_EVEN

def format_half_even(x):
    val = Decimal(str(x)).quantize(
        Decimal("0.000000"), rounding=ROUND_HALF_EVEN
    )  # 银行家舍入后的值
    if val == 0:
        val = Decimal("0.000000")
    return format(val, ".6f")

if __name__ == "__main__":
    N = int(input())

    beta1 = 0.9  # 一阶矩衰减率
    beta2 = 0.999  # 二阶矩衰减率
    lr = 0.001  # 学习率
    wd = 0.01  # 权重衰减
    eps = 1e-8  # 数值稳定项

    w = [0.0, 0.0, 0.0]  # 模型参数
    m = [0.0, 0.0, 0.0]  # 一阶矩
    v = [0.0, 0.0, 0.0]  # 二阶矩
    p1 = 1.0  # beta1 的当前次幂
    p2 = 1.0  # beta2 的当前次幂

    for _ in range(N):
        x1, x2, y = map(float, input().split())
        pred = w[0] * x1 + w[1] * x2 + w[2]  # 当前预测值
        err = 2.0 * (pred - y)  # 均方误差导数的公共项
        g = [err * x1, err * x2, err]  # 参数梯度

        p1 *= beta1
        p2 *= beta2
        nw = [0.0] * 3  # 更新后的参数

        for i, gi in enumerate(g):
            m[i] = beta1 * m[i] + (1.0 - beta1) * gi
            v[i] = beta2 * v[i] + (1.0 - beta2) * gi * gi
            mh = m[i] / (1.0 - p1)  # 偏差修正后的一阶矩
            vh = v[i] / (1.0 - p2)  # 偏差修正后的二阶矩
            nw[i] = w[i] - lr * (
                mh / (math.sqrt(vh) + eps)
                + wd * w[i]
            )

        w = nw

    print(" ".join(format_half_even(x) for x in w))
