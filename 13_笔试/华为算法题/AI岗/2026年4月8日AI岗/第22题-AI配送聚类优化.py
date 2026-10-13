import math

if __name__ == "__main__":
    K, N, speed = map(int, input().split())
    pts = [tuple(map(float, input().split())) for _ in range(N)]  # 包裹坐标

    if K >= N:
        cen = pts[:]
    else:
        cen = sorted(pts, key=lambda p: p[0] ** 2 + p[1] ** 2)[:K]

        for _ in range(50):
            sx = [0.0] * K  # 各簇横坐标和
            sy = [0.0] * K  # 各簇纵坐标和
            cnt = [0] * K  # 各簇点数

            for x, y in pts:
                c = 0  # 最近中心编号
                dis = float("inf")  # 到最近中心的距离平方
                for i, (cx, cy) in enumerate(cen):
                    cur = (x - cx) ** 2 + (y - cy) ** 2
                    if cur < dis:
                        dis = cur
                        c = i
                sx[c] += x
                sy[c] += y
                cnt[c] += 1

            ncen = cen[:]  # 更新后的中心
            for i in range(K):
                if cnt[i]:
                    ncen[i] = (
                        sx[i] / cnt[i],
                        sy[i] / cnt[i],
                    )

            move = sum(math.dist(a, b) for a, b in zip(cen, ncen))  # 中心总位移
            cen = ncen
            if move < 1e-4:
                break

    cen.sort(key=lambda p: math.hypot(p[0], p[1]))
    dis = 0.0  # 完整配送路程
    pre = (0.0, 0.0)  # 上一个服务中心
    for p in cen:
        dis += math.dist(pre, p)
        pre = p
    dis += math.dist(pre, (0.0, 0.0))

    ans = dis * 3600 / speed  # 配送总秒数
    print(math.floor(ans))
