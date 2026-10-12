import math
import sys

def solve():
    parts = sys.stdin.readline().split()
    if len(parts) != 4 or any(not part.isascii() or not part.isdigit() or int(part) <= 0 for part in parts):
        print("-1 -1")
        return
    layers, capacity, micro_batches, cards = map(int, parts)
    minimum_parallelism = (layers + capacity - 1) // capacity

    divisors = []
    for divisor in range(1, math.isqrt(layers) + 1):
        if layers % divisor == 0:
            divisors.append(divisor)
            if divisor * divisor != layers:
                divisors.append(layers // divisor)

    feasible = [value for value in divisors if minimum_parallelism <= value <= cards]
    if not feasible:
        print("-1 -1")
        return
    parallelism = min(feasible)
    bubble = (parallelism - 1) / (micro_batches + parallelism - 1)
    print(f"{parallelism} {bubble:.4f}")

if __name__ == "__main__":
    solve()
