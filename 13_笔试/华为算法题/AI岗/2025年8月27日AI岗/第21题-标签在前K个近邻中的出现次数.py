import sys
from collections import Counter

def solve():
    input = sys.stdin.buffer.readline
    k, sample_count, dimension, _ = map(int, input().split())
    query = list(map(float, input().split()))
    samples = []

    for order in range(sample_count):
        row = list(map(float, input().split()))
        features = row[:dimension]
        label = int(row[dimension])
        distance = sum(
            (query[index] - features[index]) ** 2
            for index in range(dimension)
        )
        samples.append((distance, order, label))

    samples.sort()
    neighbors = samples[:k]
    counts = Counter(label for _, _, label in neighbors)
    highest_count = max(counts.values())

    predicted = next(
        label
        for _, _, label in neighbors
        if counts[label] == highest_count
    )
    print(predicted, highest_count)

if __name__ == "__main__":
    solve()
