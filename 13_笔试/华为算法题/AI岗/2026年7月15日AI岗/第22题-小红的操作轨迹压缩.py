import sys

def solve():
    input = sys.stdin.buffer.readline
    rows, columns = map(int, input().split())
    operation_count = int(input())
    row, column = rows // 2, columns // 2
    start = (row, column)
    values = {}

    for _ in range(operation_count):
        move, operation, parameter = input().split()
        if move == b"U":
            row = max(0, row - 1)
        elif move == b"D":
            row = min(rows - 1, row + 1)
        elif move == b"L":
            column = max(0, column - 1)
        else:
            column = min(columns - 1, column + 1)
        amount = int(parameter)
        if operation == b"I":
            values[(row, column)] = values.get((row, column), 0) + amount
        elif operation == b"D":
            values[(row, column)] = values.get((row, column), 0) - amount

    final_position = (row, column)
    targets = [position for position, value in values.items() if value != 0]
    if not targets:
        print(abs(start[0] - row) + abs(start[1] - column))
        return

    target_count = len(targets)
    full_mask = (1 << target_count) - 1
    infinity = 10**9
    dp = [[infinity] * target_count for _ in range(1 << target_count)]
    for i, target in enumerate(targets):
        distance = abs(start[0] - target[0]) + abs(start[1] - target[1])
        if distance == 0:
            distance = 1 if rows == 1 or columns == 1 else 2
        dp[1 << i][i] = distance

    for mask in range(1 << target_count):
        for last in range(target_count):
            current = dp[mask][last]
            if current == infinity:
                continue
            for nxt in range(target_count):
                if mask >> nxt & 1:
                    continue
                distance = abs(targets[last][0] - targets[nxt][0]) + abs(targets[last][1] - targets[nxt][1])
                new_mask = mask | (1 << nxt)
                dp[new_mask][nxt] = min(dp[new_mask][nxt], current + distance)

    answer = infinity
    for last, target in enumerate(targets):
        tail = abs(target[0] - row) + abs(target[1] - column)
        answer = min(answer, dp[full_mask][last] + tail)
    print(answer)

if __name__ == "__main__":
    solve()
