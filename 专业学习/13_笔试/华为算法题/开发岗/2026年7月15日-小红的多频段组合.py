
if __name__ == "__main__":
    N, K = map(int, input().split())
    bw = list(map(int, input().split()))  # 各载波带宽
    bands = [value.lower() for value in input().split()]
    C = int(input())
    pairs = []  # 冲突频段对
    for _ in range(C):
        x, y = input().split()
        pairs.append((x.lower(), y.lower()))

    reps = {}  # 每个频段保留的代表载波
    for i, (band, val) in enumerate(zip(bands, bw)):
        old = reps.get(band)
        if old is None or val > old[1]:
            reps[band] = (i, val)

    cand = sorted(  # 按原编号排列的候选载波
        ((i, val, band) for band, (i, val) in reps.items()),
        key=lambda x: x[0],
    )
    pos = {band: p for p, (_, _, band) in enumerate(cand)}  # 频段候选位置
    ban = [0] * len(cand)  # 各候选的冲突位掩码
    for x, y in pairs:
        if x in pos and y in pos:
            a = pos[x]
            b = pos[y]
            ban[a] |= 1 << b
            ban[b] |= 1 << a

    best = -1  # 当前最大总带宽
    ans = ()  # 当前最优编号序列

    def search(start, mask, total, ids):
        global best, ans
        cur = tuple(ids)
        if total > best or (total == best and cur < ans):
            best = total
            ans = cur
        if len(ids) == K:
            return
        for p in range(start, len(cand)):
            if ban[p] & mask:
                continue
            i, val, _ = cand[p]
            ids.append(i)
            search(
                p + 1,
                mask | (1 << p),
                total + val,
                ids,
            )
            ids.pop()

    search(0, 0, 0, [])
    print(best)
    print(*ans)
