from collections import Counter

if __name__ == "__main__":
    k, m, n, s = map(int, input().split())
    query = list(map(float, input().split()))  # 待分类样本
    samples = []  # 距离、输入顺序和标签

    for i in range(m):
        row = list(map(float, input().split()))
        X = row[:n]  # 当前训练样本特征
        label = int(row[n])
        dist = sum(
            (query[j] - X[j]) ** 2
            for j in range(n)
        )
        samples.append((dist, i, label))

    samples.sort()
    near = samples[:k]  # 前 k 个近邻
    cnt = Counter(label for _, _, label in near)  # 各标签出现次数
    top = max(cnt.values())  # 最高出现次数

    ans = next(  # 并列时最先出现的标签
        label
        for _, _, label in near
        if cnt[label] == top
    )
    print(ans, top)
