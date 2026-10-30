import math
from decimal import Decimal, ROUND_HALF_EVEN

if __name__ == "__main__":
    n, k = map(int, input().split())
    pts = [tuple(map(int, input().split())) for _ in range(n)]  # 坐标点

    cen = [  # 各簇中心
        (float(pts[i][0]), float(pts[i][1]))
        for i in range(k)
    ]
    belong = [-1] * n  # 每个点所属的簇

    while True:
        nxt = []  # 本轮重新分配的簇编号
        for x, y in pts:
            best = 0  # 最近簇编号
            dis = float("inf")  # 当前最短平方距离
            for c, (cx, cy) in enumerate(cen):
                cur = (x - cx) ** 2 + (y - cy) ** 2
                if cur < dis:
                    dis = cur
                    best = c
            nxt.append(best)

        if nxt == belong:
            break
        belong = nxt

        sx = [0] * k  # 各簇横坐标和
        sy = [0] * k  # 各簇纵坐标和
        cnt = [0] * k  # 各簇样本数
        for (x, y), c in zip(pts, belong):
            sx[c] += x
            sy[c] += y
            cnt[c] += 1

        for c in range(k):
            if cnt[c]:
                cen[c] = (
                    sx[c] / cnt[c],
                    sy[c] / cnt[c],
                )

    mem = [[] for _ in range(k)]  # 各簇包含的点编号
    for i, c in enumerate(belong):
        mem[c].append(i)
    valid = [c for c in range(k) if mem[c]]  # 非空簇编号

    dist = [[0.0] * n for _ in range(n)]  # 点对欧氏距离
    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = dist[j][i] = math.dist(pts[i], pts[j])

    score = [0.0] * n  # 每个点的轮廓系数
    if len(valid) > 1:
        for i, own in enumerate(belong):
            same = mem[own]  # 同簇点编号
            if len(same) <= 1:
                continue

            a = sum(dist[i][j] for j in same if j != i) / (len(same) - 1)
            b = min(
                sum(dist[i][j] for j in mem[c]) / len(mem[c])
                for c in valid
                if c != own
            )
            den = max(a, b)  # 轮廓系数分母
            if den:
                score[i] = (b - a) / den

    worst = valid[0]  # 平均轮廓系数最低的簇
    low = float("inf")  # 当前最低平均轮廓系数
    for c in valid:
        avg = sum(score[i] for i in mem[c]) / len(mem[c])
        if avg < low:
            low = avg
            worst = c

    ids = mem[worst]  # 最差簇中的点编号
    cx = Decimal(sum(pts[i][0] for i in ids)) / Decimal(len(ids))
    cy = Decimal(sum(pts[i][1] for i in ids)) / Decimal(len(ids))
    cx = cx.quantize(Decimal("0.00"), rounding=ROUND_HALF_EVEN)
    cy = cy.quantize(Decimal("0.00"), rounding=ROUND_HALF_EVEN)
    print(f"{cx},{cy}")
