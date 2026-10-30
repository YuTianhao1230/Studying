
if __name__ == "__main__":
    n, m = map(int, input().split())
    a = list(map(int, input().split()))  # 各地砖的宝物价值
    if 3 * m < n:
        print(-1)
        raise SystemExit

    M = min(m, n)  # 实际使用的回合数
    neg = -10**18  # 不可达状态
    pre = [0]  # 上一回合可达位置的最大价值
    pre_l = 0  # pre[0] 对应的位置

    for k in range(1, M + 1):
        rest = M - k  # 剩余回合数
        low = max(k, n - 3 * rest)  # 本轮最低可行位置
        high = min(3 * k, n - rest)  # 本轮最高可行位置
        cur = [neg] * (high - low + 1)  # 本轮状态
        size = len(pre)  # 上一轮状态数

        for pos in range(low, high + 1):
            i = pos - 1 - pre_l  # 前进一步对应的状态下标
            best = pre[i] if 0 <= i < size else neg
            if 0 <= i - 1 < size and pre[i - 1] > best:
                best = pre[i - 1]
            if 0 <= i - 2 < size and pre[i - 2] > best:
                best = pre[i - 2]
            cur[pos - low] = best + a[pos - 1]

        pre = cur
        pre_l = low

    print(pre[n - pre_l])
