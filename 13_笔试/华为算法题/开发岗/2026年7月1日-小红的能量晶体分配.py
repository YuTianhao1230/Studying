import heapq

if __name__ == "__main__":
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = sorted(map(int, input().split()), reverse=True)

    heap = [(x, x) for x in a]  # 各角色的下一项惩罚及基础系数
    heapq.heapify(heap)
    cost = []  # 全局最小的 m 个惩罚
    for _ in range(m):
        val, x = heapq.heappop(heap)
        cost.append(val)
        heapq.heappush(heap, (val + x, x))

    ans = 0
    for energy, penalty in zip(b, cost):
        if energy <= penalty:
            break
        ans += energy - penalty
    print(ans)
