
if __name__ == "__main__":
    N, K = map(int, input().split())
    a = list(map(int, input().split()))  # 待量化序列
    K = min(K, N)
    pre = [0]  # 序列前缀和
    for x in a:
        pre.append(pre[-1] + x)

    def cost(l, r):
        mid = (l + r - 1) // 2  # 区间下中位数下标
        med = a[mid]  # 区间下中位数
        lc = med * (mid - l) - (pre[mid] - pre[l])  # 左半区间误差
        rc = pre[r] - pre[mid + 1] - med * (r - mid - 1)  # 右半区间误差
        return lc + rc

    INF = 10**30  # 不可达状态
    dp = [INF] * (N + 1)  # 上一轮的最小误差
    dp[0] = 0

    for k in range(1, K + 1):
        ndp = [INF] * (N + 1)  # 当前轮的最小误差

        def compute(l, r, ql, qr):
            if l > r:
                return
            mid = (l + r) // 2
            best = INF  # 当前状态的最小误差
            pos = -1  # 当前状态的最优切分点
            hi = min(mid - 1, qr)  # 可枚举切分点上界
            for j in range(ql, hi + 1):
                cur = dp[j] + cost(j, mid)  # 从 j 切分的总误差
                if cur < best:
                    best = cur
                    pos = j
            ndp[mid] = best
            compute(l, mid - 1, ql, pos)
            compute(mid + 1, r, pos, qr)

        compute(k, N, k - 1, N - 1)
        dp = ndp

    print(dp[N])
