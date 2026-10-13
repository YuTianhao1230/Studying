import math

if __name__ == "__main__":
    N, V, B, lam = input().split()
    N, V, B = int(N), int(V), int(B)
    lam = float(lam)  # lam：题目中的 lambda
    loss = []  # 每个样本的交叉熵
    H = []  # 每个样本的分布熵
    c = []  # 每个样本的代价

    for _ in range(N):
        row = input().split()  # 当前样本的一行数据
        logits = list(map(float, row[:V]))
        y = int(row[V])
        c.append(int(row[V + 1]))
        mx = max(logits)  # 最大 logit，用于稳定计算
        e = [math.exp(x - mx) for x in logits]  # 平移后的指数值
        s = sum(e)  # 指数值之和
        p = [x / s for x in e]  # softmax 概率
        loss.append(-math.log(p[y]))
        H.append(-sum(x * math.log(x) for x in p))

    dp = [float("-inf")] * (B + 1)  # dp[b]：代价为 b 时的最大交叉熵
    dp[0] = 0.0
    take = [bytearray(B + 1) for _ in range(N)]  # 记录背包选择
    for i, (ci, li) in enumerate(zip(c, loss)):
        for b in range(B, ci - 1, -1):
            cur = dp[b - ci] + li  # 选择当前样本后的交叉熵
            if cur > dp[b]:
                dp[b] = cur
                take[i][b] = 1

    b = max(range(B + 1), key=lambda x: dp[x])  # 最优方案使用的代价
    h = 0.0  # 所选样本的熵之和
    for i in range(N - 1, -1, -1):
        if take[i][b]:
            h += H[i]
            b -= c[i]

    ans = sum(loss) + lam * h  # 最终训练损失
    print(f"{ans:.2f}")
