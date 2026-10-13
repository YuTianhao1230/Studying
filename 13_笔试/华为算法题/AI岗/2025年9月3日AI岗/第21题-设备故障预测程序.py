import math
import statistics

def parse_number(x):
    try:
        val = float(x)  # 解析后的数值
    except ValueError:
        return None
    return None if math.isnan(val) else val

def is_valid(col, val):
    if val is None:
        return False
    if col in (0, 1):
        return val >= 0
    if col in (2, 3):
        return 0 <= val <= 1000
    return 0 <= val <= 20

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)  # 避免负数分支计算 exp(-x) 溢出
    return e / (1.0 + e)

if __name__ == "__main__":
    N = int(input())
    kept = []  # 标签有效的训练样本

    for _ in range(N):
        parts = input().strip().split(",")
        if len(parts) != 7:
            continue
        try:
            label = int(parts[6])
        except ValueError:
            continue
        if label not in (0, 1):
            continue
        kept.append((parts[1:6], label))

    cols = [[] for _ in range(5)]  # 各特征列的有效值
    for vals, _ in kept:
        for j, raw in enumerate(vals):
            val = parse_number(raw)
            if is_valid(j, val):
                cols[j].append(val)

    means = []  # 各列有效值均值
    mids = []  # 各列有效值中位数
    for vals in cols:
        if vals:
            means.append(sum(vals) / len(vals))
            mids.append(float(statistics.median(vals)))
        else:
            means.append(0.0)
            mids.append(0.0)

    def clean(raw_vals):
        res = []  # 清洗后的五个特征
        for j, raw in enumerate(raw_vals):
            val = parse_number(raw)
            if val is None:
                res.append(means[j])
            elif not is_valid(j, val):
                res.append(mids[j])
            else:
                res.append(val)
        return res

    train = [(clean(vals), label) for vals, label in kept]  # 清洗后训练集
    weights = [0.0] * 5  # 逻辑回归权重
    bias = 0.0

    if train:
        n = len(train)  # 有效训练样本数
        for _ in range(100):
            gb = 0.0  # 偏置梯度和
            gw = [0.0] * 5  # 各权重梯度和
            for X, label in train:
                z = bias + sum(weights[j] * X[j] for j in range(5))
                err = sigmoid(z) - label
                gb += err
                for j in range(5):
                    gw[j] += err * X[j]

            bias -= 0.01 * gb / n
            for j in range(5):
                weights[j] -= 0.01 * gw[j] / n

    M = int(input())
    for _ in range(M):
        parts = input().strip().split(",")
        X = clean(parts[1:6])
        z = bias + sum(weights[j] * X[j] for j in range(5))
        print(1 if sigmoid(z) >= 0.5 else 0)
