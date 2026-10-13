
if __name__ == "__main__":
    n, m, h = map(int, input().split())

    if h <= m:
        s = h * (h + 1) // 2  # 矩阵 V 的单行元素和
    else:
        s = m * (m + 1) // 2 + (h - m) * m

    print(n * s)
