
if __name__ == "__main__":
    N, E, K, C = map(int, input().split())
    cnt = [0] * E  # cnt[e]：专家 e 已接收的任务数

    for _ in range(N):
        a = list(map(int, input().split()))  # 当前任务对各专家的评分

        ids = list(range(E))  # 所有专家编号
        ids.sort(key=lambda e: a[e], reverse=True)  # 按评分从高到低排序
        ids = ids[:K]  # 取前 K 名专家

        for e in ids:
            if cnt[e] < C:
                cnt[e] += 1

    ans = sum(x * x for x in cnt)  # 专家负载平方和
    print(ans)
    print(*cnt, sep=" ")
