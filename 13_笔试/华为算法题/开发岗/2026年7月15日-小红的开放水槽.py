
if __name__ == "__main__":
    n = int(input())
    h = list(map(int, input().split()))
    right = h[:]  # 从当前位置开始的后缀最大高度
    for i in range(n - 2, -1, -1):
        right[i] = max(right[i], right[i + 1])

    ans = 0
    left = h[0]  # 当前位置左侧的最大高度
    for i in range(n - 1):
        left = max(left, h[i])
        ans += min(left, right[i + 1])
    print(ans)
