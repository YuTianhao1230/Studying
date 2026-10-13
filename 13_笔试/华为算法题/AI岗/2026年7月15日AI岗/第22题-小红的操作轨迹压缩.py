
if __name__ == "__main__":
    n, m = map(int, input().split())
    K = int(input())
    r, c = n // 2, m // 2  # 当前行、列
    st = (r, c)  # 起点
    val = {}  # 每个被修改位置的累计值

    for _ in range(K):
        move, op, param = input().split()
        if move == "U":
            r = max(0, r - 1)
        elif move == "D":
            r = min(n - 1, r + 1)
        elif move == "L":
            c = max(0, c - 1)
        else:
            c = min(m - 1, c + 1)
        x = int(param)  # 本次增减量
        if op == "I":
            val[(r, c)] = val.get((r, c), 0) + x
        elif op == "D":
            val[(r, c)] = val.get((r, c), 0) - x

    pts = [pos for pos, x in val.items() if x != 0]  # 必须访问的修改位置
    if not pts:
        print(abs(st[0] - r) + abs(st[1] - c))
        raise SystemExit

    q = len(pts)  # 必须访问的位置数
    full = (1 << q) - 1  # 全部位置均已访问的状态
    INF = 10**9  # 不可达状态
    dp = [[INF] * q for _ in range(1 << q)]  # dp[mask][i]：停在 i 的最短路程
    for i, pos in enumerate(pts):
        dis = abs(st[0] - pos[0]) + abs(st[1] - pos[1])  # 起点到 pos 的距离
        if dis == 0:
            dis = 1 if n == 1 or m == 1 else 2
        dp[1 << i][i] = dis

    for mask in range(1 << q):
        for last in range(q):
            cur = dp[mask][last]  # 当前状态的最短路程
            if cur == INF:
                continue
            for nxt in range(q):
                if mask >> nxt & 1:
                    continue
                dis = abs(pts[last][0] - pts[nxt][0]) + abs(pts[last][1] - pts[nxt][1])
                nmask = mask | (1 << nxt)  # 加入 nxt 后的状态
                dp[nmask][nxt] = min(dp[nmask][nxt], cur + dis)

    ans = INF  # 回到原轨迹终点的最短总路程
    for last, pos in enumerate(pts):
        tail = abs(pos[0] - r) + abs(pos[1] - c)  # 最后一段距离
        ans = min(ans, dp[full][last] + tail)
    print(ans)
