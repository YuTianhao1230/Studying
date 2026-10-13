import math
from decimal import Decimal, ROUND_HALF_UP

if __name__ == "__main__":
    n, d, b = map(int, input().split())
    X = [list(map(float, input().split())) for _ in range(n)]
    W1 = list(map(float, input().split()))
    W2 = list(map(float, input().split()))

    A = []  # 各连续块的注意力分数
    w = sum(W2)  # W2 的元素和
    scale = math.sqrt(d)  # 注意力缩放因子

    for l in range(0, n, b):
        r = min(l + b, n)  # 当前块的右边界
        size = r - l  # 当前块的向量数
        sums = [0.0] * d  # 当前块各维度之和
        for i in range(l, r):
            for j, x in enumerate(X[i]):
                sums[j] += x

        h = [x / size for x in sums]
        s = sum(x * y for x, y in zip(W1, h)) + 2.0
        z = max(0.0, s)
        A.append((z * w + d) / scale)

    if len(A) < 2:
        print(0)
        raise SystemExit

    total = sum(A)  # 全部块的分数和
    pre = 0.0  # 切分点左侧分数和
    best = float("-inf")  # 当前最优的较小段和
    for i in range(len(A) - 1):
        pre += A[i]
        best = max(best, min(pre, total - pre))

    ans = Decimal(str(best * 100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    print(ans)
