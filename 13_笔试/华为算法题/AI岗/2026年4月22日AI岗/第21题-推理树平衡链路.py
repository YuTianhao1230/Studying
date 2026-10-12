import sys
from collections import defaultdict, deque

def solve():
    tokens = sys.stdin.buffer.read().split()
    if not tokens:
        return
    token_count = int(tokens[0])
    level_order = tokens[1 : token_count + 1]

    values = [int(level_order[0])]
    children = [[-1, -1]]
    queue = deque([0])
    position = 1
    while queue and position < token_count:
        parent = queue.popleft()
        for side in range(2):
            if position >= token_count:
                break
            token = level_order[position]
            position += 1
            if token == b"None":
                continue
            node = len(values)
            values.append(int(token))
            children.append([-1, -1])
            children[parent][side] = node
            queue.append(node)

    prefix_frequency = defaultdict(int)
    prefix_frequency[0] = 1
    answer = 0
    stack = [(0, 0, False)]

    while stack:
        node, parent_sum, leaving = stack.pop()
        current_sum = parent_sum + values[node]
        if leaving:
            prefix_frequency[current_sum] -= 1
            continue

        answer += prefix_frequency[current_sum]
        if values[node] == 0:
            answer -= 1

        prefix_frequency[current_sum] += 1
        stack.append((node, parent_sum, True))
        right = children[node][1]
        left = children[node][0]
        if right != -1:
            stack.append((right, current_sum, False))
        if left != -1:
            stack.append((left, current_sum, False))

    print(answer)

if __name__ == "__main__":
    solve()
