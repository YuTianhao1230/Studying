import heapq
import sys

def solve():
    input = sys.stdin.buffer.readline
    _, m = map(int, input().split())
    coefficients = list(map(int, input().split()))
    crystals = sorted(map(int, input().split()), reverse=True)

    heap = [(coefficient, coefficient) for coefficient in coefficients]
    heapq.heapify(heap)
    penalties = []
    for _ in range(m):
        penalty, coefficient = heapq.heappop(heap)
        penalties.append(penalty)
        heapq.heappush(heap, (penalty + coefficient, coefficient))

    answer = 0
    for energy, penalty in zip(crystals, penalties):
        if energy <= penalty:
            break
        answer += energy - penalty
    print(answer)

if __name__ == "__main__":
    solve()
