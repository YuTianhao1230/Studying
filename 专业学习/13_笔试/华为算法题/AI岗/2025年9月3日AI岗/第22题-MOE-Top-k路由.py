
if __name__ == "__main__":
    n, m, p, k = map(int, input().split())
    probs = list(map(float, input().split()))  # 各专家概率

    if n % m != 0 or p > m:
        print("error")
        raise SystemExit

    g = n // m  # 每组专家数
    if p * g < k:
        print("error")
        raise SystemExit

    reps = []  # 各组的代表概率和组号
    for i in range(m):
        l = i * g
        r = l + g
        reps.append((max(probs[l:r]), i))

    reps.sort(key=lambda x: (-x[0], x[1]))
    chosen = {i for _, i in reps[:p]}  # 被选中的组号

    cand = [  # 候选专家编号
        j
        for i in chosen
        for j in range(i * g, (i + 1) * g)
    ]
    cand.sort(key=lambda i: (-probs[i], i))
    ans = sorted(cand[:k])
    print(*ans)
