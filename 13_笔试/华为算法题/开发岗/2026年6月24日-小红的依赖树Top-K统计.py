import heapq
import sys

def solve():
    input = sys.stdin.buffer.readline
    preorder = list(map(int, input().split()))
    inorder = list(map(int, input().split()))
    k = int(input())
    inorder_index = {value: index for index, value in enumerate(inorder)}
    top_k = []
    stack = [(0, 0, len(inorder), 0)]

    while stack:
        root_pre, left, right, parent_sum = stack.pop()
        if left >= right:
            continue
        root_value = preorder[root_pre]
        path_sum = parent_sum + root_value
        if path_sum <= 0:
            continue
        if len(top_k) < k:
            heapq.heappush(top_k, path_sum)
        elif path_sum > top_k[0]:
            heapq.heapreplace(top_k, path_sum)

        root_in = inorder_index[root_value]
        left_size = root_in - left
        if root_in + 1 < right:
            stack.append((root_pre + left_size + 1, root_in + 1, right, path_sum))
        if left < root_in:
            stack.append((root_pre + 1, left, root_in, path_sum))

    print(*sorted(top_k)) if top_k else print("null")

if __name__ == "__main__":
    solve()
