import sys
from decimal import Decimal, ROUND_HALF_EVEN

def format_half_even(value):
    rounded = Decimal(str(value)).quantize(
        Decimal("0.00"), rounding=ROUND_HALF_EVEN
    )
    if rounded == 0:
        rounded = Decimal("0.00")
    return format(rounded, ".2f")

def solve():
    input = sys.stdin.buffer.readline
    sample_count = int(input())
    iterations = int(input())
    learning_rate = float(input())
    rows = [list(map(int, input().split())) for _ in range(sample_count)]

    minimums = [
        min(row[feature] for row in rows) for feature in range(3)
    ]
    maximums = [
        max(row[feature] for row in rows) for feature in range(3)
    ]
    ranges = [
        maximums[feature] - minimums[feature] for feature in range(3)
    ]

    gram = [[0.0] * 4 for _ in range(4)]
    target_product = [0.0] * 4

    for row in rows:
        vector = [1.0]
        for feature in range(3):
            if ranges[feature] == 0:
                vector.append(0.0)
            else:
                vector.append(
                    (row[feature] - minimums[feature]) / ranges[feature]
                )
        target = row[3]

        for first in range(4):
            target_product[first] += vector[first] * target
            for second in range(4):
                gram[first][second] += vector[first] * vector[second]

    inverse_count = 1.0 / sample_count
    for first in range(4):
        target_product[first] *= inverse_count
        for second in range(4):
            gram[first][second] *= inverse_count

    weights = [0.0] * 4
    for _ in range(iterations):
        gradients = [
            sum(
                gram[first][second] * weights[second]
                for second in range(4)
            )
            - target_product[first]
            for first in range(4)
        ]
        weights = [
            weight - learning_rate * gradient
            for weight, gradient in zip(weights, gradients)
        ]

    restored = [0.0] * 4
    for feature in range(3):
        if ranges[feature] != 0:
            restored[feature + 1] = (
                weights[feature + 1] / ranges[feature]
            )
    restored[0] = weights[0] - sum(
        restored[feature + 1] * minimums[feature]
        for feature in range(3)
    )

    print(" ".join(format_half_even(value) for value in restored))

if __name__ == "__main__":
    solve()
