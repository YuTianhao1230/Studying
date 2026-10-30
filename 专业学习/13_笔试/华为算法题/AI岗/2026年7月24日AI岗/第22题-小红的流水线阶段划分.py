from collections import deque

if __name__ == "__main__":
    n, p, T = map(int, input().split())
    a = list(map(int, input().split()))  # a[i]：第 i+1 层的计算时间
    comm = list(map(int, input().split())) if n > 1 else []

    pre = [0] * (n + 1)  # pre[i]：前 i 层的计算时间之和
    for i, x in enumerate(a, start=1):
        pre[i] = pre[i - 1] + x

    INF = 10**30  # 不可达状态

    def dfs(s):
        # dfs(s)[i]：前 i 层恰好划分成 s 个阶段的最小通信开销
        if s == 1:
            dp = [INF] * (n + 1)
            for i in range(1, n + 1):
                if pre[i] > T:
                    break
                dp[i] = 0  # 只有一个阶段，没有切分和通信开销
            return dp

        last = dfs(s - 1)  # 少划分一个阶段时的答案
        dp = [INF] * (n + 1)
        q = deque()  # 按转移开销递增保存合法切分点

        for i in range(s, n + 1):
            j = i - 1

            if last[j] < INF:
                cost = last[j] + comm[j - 1]
                while q:
                    k = q[-1]
                    if last[k] + comm[k - 1] < cost:
                        break
                    q.pop()
                q.append(j)

            # 最后一段为第 j+1 到 i 层，计算时间不能超过 T
            while q and pre[i] - pre[q[0]] > T:
                q.popleft()

            if q:
                j = q[0]
                dp[i] = last[j] + comm[j - 1]

        return dp

    ans = dfs(p)[n]
    print(ans if ans < INF else -1)
