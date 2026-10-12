import math
import sys

def round_half_even(value):
    return int(round(value))

def solve():
    input = sys.stdin.buffer.readline
    signal_count, length, cluster_count = map(int, input().split())
    features = []
    for _ in range(signal_count):
        signal = list(map(float, input().split()))
        magnitudes = []
        for frequency in range(1, length // 2):
            real = 0.0
            imaginary = 0.0
            for time, value in enumerate(signal):
                angle = 2.0 * math.pi * frequency * time / length
                real += value * math.cos(angle)
                imaginary -= value * math.sin(angle)
            magnitudes.append((math.hypot(real, imaginary), frequency))
        magnitudes.sort(key=lambda item: (-item[0], item[1]))
        features.append(tuple(frequency for _, frequency in magnitudes[:3]))

    initial_indices = list(map(int, input().split()))
    centers = [features[index] for index in initial_indices]
    for _ in range(100):
        groups = [[] for _ in range(cluster_count)]
        for feature in features:
            chosen = min(
                range(cluster_count),
                key=lambda index: sum((feature[d] - centers[index][d]) ** 2 for d in range(3)),
            )
            groups[chosen].append(feature)

        new_centers = []
        for index, group in enumerate(groups):
            if not group:
                new_centers.append(centers[index])
            else:
                new_centers.append(tuple(round_half_even(sum(point[d] for point in group) / len(group)) for d in range(3)))
        if new_centers == centers:
            break
        centers = new_centers

    for center in sorted(centers):
        print(*center)

if __name__ == "__main__":
    solve()
