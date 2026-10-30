import heapq

if __name__ == "__main__":
    pre = list(map(int, input().split()))  # 前序遍历
    ino = list(map(int, input().split()))  # 中序遍历
    K = int(input())
    pos = {x: i for i, x in enumerate(ino)}  # 节点的中序位置
    top = []  # 当前最大的 K 个完整大小
    stack = [(0, 0, len(ino), 0)]  # 前序根、中序区间、父路径和

    while stack:
        root, l, r, parent = stack.pop()
        if l >= r:
            continue
        val = pre[root]  # 当前根节点增量
        total = parent + val  # 当前节点完整大小
        if total <= 0:
            continue
        if len(top) < K:
            heapq.heappush(top, total)
        elif total > top[0]:
            heapq.heapreplace(top, total)

        mid = pos[val]  # 根节点的中序位置
        size = mid - l  # 左子树节点数
        if mid + 1 < r:
            stack.append((root + size + 1, mid + 1, r, total))
        if l < mid:
            stack.append((root + 1, l, mid, total))

    print(*sorted(top)) if top else print("null")
