
if __name__ == "__main__":
    N, M = map(int, input().split())
    pos = sorted(map(int, input().split()))  # 排序后的目标点
    ans = 0
    i = 0
    while i < N:
        x = pos[i]  # 最左侧未覆盖点
        j = i  # 当前信号站的位置下标
        while j + 1 < N and pos[j + 1] <= x + M:
            j += 1
        right = pos[j] + M  # 当前信号站的右覆盖边界
        i = j + 1
        while i < N and pos[i] <= right:
            i += 1
        ans += 1
    print(ans)
