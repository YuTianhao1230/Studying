def solve():
    n, m, h = map(int, input().split())

    if h <= m:
        row_sum = h * (h + 1) // 2
    else:
        row_sum = m * (m + 1) // 2 + (h - m) * m

    print(n * row_sum)

if __name__ == "__main__":
    solve()
