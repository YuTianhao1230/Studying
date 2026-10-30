import math

if __name__ == "__main__":
    T, L, D, b, m = input().split()
    T, L, D, m = int(T), int(L), int(D), int(m)
    b = float(b)
    x = list(map(float, input().split()))  # 原始向量

    s = max(1.0, T / L)
    nb = b * s ** (D / (D - 2))  # 题目中的新基频 b'
    ans = []  # 旋转后的向量
    for i in range(D // 2):
        theta = m * nb ** (-2 * i / D)
        cos, sin = math.cos(theta), math.sin(theta)
        ans.extend(
            (
                x[2 * i] * cos - x[2 * i + 1] * sin,
                x[2 * i] * sin + x[2 * i + 1] * cos,
            )
        )

    out = []  # 格式化后的输出
    for v in ans:
        if abs(v) < 0.00005:
            v = 0.0
        out.append(f"{v:.4f}")
    print(" ".join(out))
