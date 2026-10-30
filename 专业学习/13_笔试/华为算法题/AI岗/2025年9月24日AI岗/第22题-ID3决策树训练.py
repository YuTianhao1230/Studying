import math
import sys

def entropy(c0, c1):
    total = c0 + c1  # 当前样本总数
    ans = 0.0  # 信息熵
    for cnt in (c0, c1):
        if cnt:
            p = cnt / total
            ans -= p * math.log2(p)
    return ans

if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    n, m = map(int, input().split())

    X = []  # 训练特征
    y = []  # 训练标签
    for _ in range(n):
        row = list(map(int, input().split()))
        X.append(row[:m])
        y.append(row[m])

    def build(ids, fs, default=0):
        if not ids:
            return default

        c1 = sum(y[i] for i in ids)  # 标签 1 的数量
        c0 = len(ids) - c1  # 标签 0 的数量
        major = 1 if c1 > c0 else 0  # 当前多数标签

        if c0 == 0 or c1 == 0:
            return y[ids[0]]
        if not fs:
            return major

        base = entropy(c0, c1)  # 划分前的信息熵
        best = -1  # 信息增益最大的特征
        gain_max = 0.0  # 当前最大信息增益
        eps = 1e-12  # 浮点比较误差

        for f in fs:
            cnt = [[0, 0], [0, 0]]  # 两个分支的标签计数
            for i in ids:
                cnt[X[i][f]][y[i]] += 1

            cond = 0.0  # 按当前特征划分后的条件熵
            for b0, b1 in cnt:
                size = b0 + b1
                if size:
                    cond += size / len(ids) * entropy(b0, b1)
            gain = base - cond
            if gain > gain_max + eps:
                gain_max = gain
                best = f

        if best == -1:
            return major

        left = []  # 特征值为 0 的样本
        right = []  # 特征值为 1 的样本
        for i in ids:
            if X[i][best] == 0:
                left.append(i)
            else:
                right.append(i)

        nxt = tuple(f for f in fs if f != best)  # 子树可用特征
        return (
            best,
            build(left, nxt, major),
            build(right, nxt, major),
        )

    tree = build(list(range(n)), tuple(range(m)))

    def predict(query):
        node = tree
        while isinstance(node, tuple):
            f, left, right = node  # 当前划分特征及左右子树
            node = left if query[f] == 0 else right
        return node

    q = int(input())
    out = []  # 各查询的预测结果
    for _ in range(q):
        query = list(map(int, input().split()))
        out.append(str(predict(query)))
    print("\n".join(out))
