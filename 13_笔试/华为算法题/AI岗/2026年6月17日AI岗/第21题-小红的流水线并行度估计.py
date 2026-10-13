import math

if __name__ == "__main__":
    row = input().split()  # 输入字段
    if len(row) != 4 or any(not x.isascii() or not x.isdigit() or int(x) <= 0 for x in row):
        print("-1 -1")
        raise SystemExit
    L, C, m, G = map(int, row)
    low = (L + C - 1) // C  # 并行度下界

    fac = []  # L 的全部约数
    for x in range(1, math.isqrt(L) + 1):
        if L % x == 0:
            fac.append(x)
            if x * x != L:
                fac.append(L // x)

    can = [x for x in fac if low <= x <= G]  # 合法并行度
    if not can:
        print("-1 -1")
        raise SystemExit
    p = min(can)
    rate = (p - 1) / (m + p - 1)  # 最小气泡率
    print(f"{p} {rate:.4f}")
