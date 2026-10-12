import math
import sys

def solve():
    input = sys.stdin.buffer.readline
    actual_length, trained_length, dimension, base, position = input().split()
    actual_length = int(actual_length)
    trained_length = int(trained_length)
    dimension = int(dimension)
    base = float(base)
    position = int(position)
    vector = list(map(float, input().split()))

    scale = max(1.0, actual_length / trained_length)
    adjusted_base = base * scale ** (dimension / (dimension - 2))
    answer = []
    for i in range(dimension // 2):
        angle = position * adjusted_base ** (-2 * i / dimension)
        cosine, sine = math.cos(angle), math.sin(angle)
        left, right = vector[2 * i], vector[2 * i + 1]
        answer.extend((left * cosine - right * sine, left * sine + right * cosine))

    formatted = []
    for value in answer:
        if abs(value) < 0.00005:
            value = 0.0
        formatted.append(f"{value:.4f}")
    print(" ".join(formatted))

if __name__ == "__main__":
    solve()
