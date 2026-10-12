from collections import deque
import sys

def solve():
    input = sys.stdin.buffer.readline
    needs = list(map(int, input().split()))
    target = int(input())
    window_count = int(input())
    queue = deque(range(len(needs)))
    elapsed = 0
    while queue:
        serving = [queue.popleft() for _ in range(min(window_count, len(queue)))]
        elapsed += 1
        for customer in serving:
            needs[customer] -= 1
            if needs[customer] == 0:
                if customer == target:
                    print(elapsed)
                    return
            else:
                queue.append(customer)

if __name__ == "__main__":
    solve()
