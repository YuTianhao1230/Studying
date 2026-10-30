import math

def round_half_even(x):
    return int(round(x))

if __name__ == "__main__":
    N, L, K = map(int, input().split())
    feat = []  # 每组信号的三维频率特征
    for _ in range(N):
        a = list(map(float, input().split()))  # 当前信号
        mag = []  # (DFT 幅值, 频率下标)
        for k in range(1, L // 2):
            re = 0.0  # DFT 实部
            im = 0.0  # DFT 虚部
            for t, x in enumerate(a):
                theta = 2.0 * math.pi * k * t / L
                re += x * math.cos(theta)
                im -= x * math.sin(theta)
            mag.append((math.hypot(re, im), k))
        mag.sort(key=lambda x: (-x[0], x[1]))
        feat.append(tuple(k for _, k in mag[:3]))

    ids = list(map(int, input().split()))  # 初始中心下标
    cen = [feat[i] for i in ids]  # 当前聚类中心
    for _ in range(100):
        group = [[] for _ in range(K)]  # 各中心包含的特征
        for x in feat:
            c = min(
                range(K),
                key=lambda i: sum((x[d] - cen[i][d]) ** 2 for d in range(3)),
            )
            group[c].append(x)

        ncen = []  # 更新后的聚类中心
        for i, g in enumerate(group):
            if not g:
                ncen.append(cen[i])
            else:
                ncen.append(
                    tuple(round_half_even(sum(x[d] for x in g) / len(g)) for d in range(3))
                )
        if ncen == cen:
            break
        cen = ncen

    for c in sorted(cen):
        print(*c)
