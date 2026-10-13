
if __name__ == "__main__":
    n, m, C = map(int, input().split())
    a = [list(map(int, input().split())) for _ in range(n)]
    inf = 10**18  # 无穷大初值
    ans = inf

    for x in range(1, C + 1):
        dp = [inf] * m  # 当前行各位置的最小代价
        for i in range(n):
            for j in range(m):
                cost = abs(a[i][j] - x)
                if i == 0 and j == 0:
                    dp[j] = cost
                elif i == 0:
                    dp[j] = dp[j - 1] + cost
                elif j == 0:
                    dp[j] += cost
                else:
                    dp[j] = min(dp[j], dp[j - 1]) + cost
        ans = min(ans, dp[-1])
    print(ans)
