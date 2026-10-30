
import heapq


if __name__ == "__main__":
    N, D, K = map(int, input().split())
    w = list(map(int, input().split()))
    t = list(map(int, input().split()))
    heap = []  # (-评分, -对象编号)，模拟大根堆
    for _ in range(N):
        row = list(map(int, input().split()))  # 当前对象的一行数据
        id, a = row[0], row[1:]
        score = sum(w[j] * (t[j] - a[j]) ** 2 for j in range(D))
        heapq.heappush(heap, (-score, -id))
        if len(heap) > K:
            heapq.heappop(heap)

    ans = [(-score, -id) for score, id in heap]
    ans.sort()
    print(*(id for _, id in ans))
