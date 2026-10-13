if __name__ == "__main__":
    row = input().strip()  # 原始输入行
    a = [int(x.strip()) for x in row.split(",")]  # 各任务优先级
    n = len(a)  # 任务数

    left = [0] * n  # 从左侧约束得到的 Token 数
    for i, x in enumerate(a):
        if x <= 0:
            continue
        left[i] = 1
        if (
            i > 0
            and a[i - 1] > 0
            and x > a[i - 1]
        ):
            left[i] = left[i - 1] + 1

    right = [0] * n  # 从右侧约束得到的 Token 数
    for i in range(n - 1, -1, -1):
        if a[i] <= 0:
            continue
        right[i] = 1
        if (
            i + 1 < n
            and a[i + 1] > 0
            and a[i] > a[i + 1]
        ):
            right[i] = right[i + 1] + 1

    print(sum(max(left[i], right[i]) for i in range(n)))
