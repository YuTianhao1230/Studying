from decimal import Decimal, ROUND_HALF_UP

if __name__ == "__main__":
    H, W, K = map(int, input().split())
    I = [list(map(float, input().split())) for _ in range(H)]
    P = [list(map(float, input().split())) for _ in range(K)]

    r = K // 2
    dp = [0.0] * H  # 当前列各行的最大路径和

    for j in range(W):
        E = [0.0] * H  # 第 j 列各位置的能量
        for i in range(H):
            val = 0.0  # 位置 (i, j) 的相关结果
            for u in range(K):
                x = i + u - r  # 对应的图像行
                if not 0 <= x < H:
                    continue
                for v in range(K):
                    y = j + v - r  # 对应的图像列
                    if 0 <= y < W:
                        val += P[u][v] * I[x][y]
            E[i] = val

        if j == 0:
            dp = E
            continue

        ndp = [0.0] * H  # 转移后的新一列状态
        for i in range(H):
            pre = dp[i]  # 可到达当前行的最佳前驱
            if i > 0:
                pre = max(pre, dp[i - 1])
            if i + 1 < H:
                pre = max(pre, dp[i + 1])
            ndp[i] = pre + E[i]
        dp = ndp

    ans = Decimal(str(max(dp))).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
    print(ans)
