
if __name__ == "__main__":
    N, dist, Wthreshold = map(int, input().split())
    a = [tuple(map(int, input().split())) for _ in range(N)]  # 各基站数据

    adj = [[False] * N for _ in range(N)]  # 基站邻接关系
    sw = [0] * N  # 各基站邻域负载和
    for i in range(N):
        x1, y1, _, _, _ = a[i]
        for j in range(N):
            x2, y2, _, w, _ = a[j]
            if abs(x1 - x2) + abs(y1 - y2) <= dist:
                adj[i][j] = True
                sw[i] += w

    key = [i for i in range(N) if sw[i] >= Wthreshold]  # 关键基站编号
    key.sort(key=lambda i: a[i][2])

    dp = [a[i][4] for i in range(N)]  # 以各基站结尾的最大用户数
    ans = 0
    for p, v in enumerate(key):
        t = a[v][2]
        for u in key[:p]:
            if a[u][2] >= t:
                continue
            if adj[u][v]:
                dp[v] = max(dp[v], dp[u] + a[v][4])
                ans = max(ans, dp[v])

    print(ans)
