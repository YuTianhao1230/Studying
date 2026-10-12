import sys

def solve():
    input = sys.stdin.buffer.readline
    object_count, dimension, k = map(int, input().split())
    weights = list(map(int, input().split()))
    target = list(map(int, input().split()))
    ranked = []
    for _ in range(object_count):
        data = list(map(int, input().split()))
        identifier, features = data[0], data[1:]
        score = sum(weight * (expected - actual) ** 2 for weight, expected, actual in zip(weights, target, features))
        ranked.append((score, identifier))
    ranked.sort()
    print(*(identifier for _, identifier in ranked[:k]))

if __name__ == "__main__":
    solve()
