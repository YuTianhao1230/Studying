if __name__ == "__main__":
    s = input().strip()
    n = len(s)  # 原字符串长度
    pi = [0] * n  # KMP 前缀函数
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j

    d = n - pi[-1]  # 候选最短周期
    if d < n and n % d == 0:
        t = s[:d]
        k = n // d
        print(t + str(k))
    else:
        print(s)
