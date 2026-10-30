
if __name__ == "__main__":
    T = input().strip()
    S = input().strip()
    dp = [0] * (len(S) + 1)  # 当前行各前缀的 LCS 长度

    for x in T:
        diag = 0  # 更新前的左上角状态
        for j, y in enumerate(S, 1):
            old = dp[j]  # 更新前的上方状态
            if x == y:
                dp[j] = diag + 1
            elif dp[j - 1] > dp[j]:
                dp[j] = dp[j - 1]
            diag = old

    print(len(T) - dp[-1])
