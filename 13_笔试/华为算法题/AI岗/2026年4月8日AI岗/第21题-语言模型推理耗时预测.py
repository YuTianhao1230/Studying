from decimal import Decimal, ROUND_HALF_EVEN

def format_half_even(x):
    val = Decimal(str(x)).quantize(
        Decimal("0.00"), rounding=ROUND_HALF_EVEN
    )  # 银行家舍入后的值
    if val == 0:
        val = Decimal("0.00")
    return format(val, ".2f")

if __name__ == "__main__":
    m = int(input())
    N = int(input())
    alpha = float(input())
    a = [list(map(int, input().split())) for _ in range(m)]  # 训练样本

    mn = [min(row[j] for row in a) for j in range(3)]  # 各特征最小值
    mx = [max(row[j] for row in a) for j in range(3)]  # 各特征最大值
    rg = [mx[j] - mn[j] for j in range(3)]  # 各特征极差

    gram = [[0.0] * 4 for _ in range(4)]  # X 转置乘 X
    xy = [0.0] * 4  # X 转置乘 y

    for row in a:
        x = [1.0]  # 加入偏置项后的归一化特征
        for j in range(3):
            if rg[j] == 0:
                x.append(0.0)
            else:
                x.append((row[j] - mn[j]) / rg[j])
        y = row[3]

        for i in range(4):
            xy[i] += x[i] * y
            for j in range(4):
                gram[i][j] += x[i] * x[j]

    inv = 1.0 / m  # 样本数倒数
    for i in range(4):
        xy[i] *= inv
        for j in range(4):
            gram[i][j] *= inv

    w = [0.0] * 4  # 归一化空间中的模型参数
    for _ in range(N):
        g = [
            sum(gram[i][j] * w[j] for j in range(4)) - xy[i]
            for i in range(4)
        ]
        w = [wi - alpha * gi for wi, gi in zip(w, g)]

    ans = [0.0] * 4  # 还原到原特征空间的参数
    for j in range(3):
        if rg[j] != 0:
            ans[j + 1] = w[j + 1] / rg[j]
    ans[0] = w[0] - sum(ans[j + 1] * mn[j] for j in range(3))

    print(" ".join(format_half_even(x) for x in ans))
