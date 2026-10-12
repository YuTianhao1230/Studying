import math
import sys
from decimal import Decimal, ROUND_HALF_UP

def solve():
    input = sys.stdin.buffer.readline
    vector_count, dimension, block_size = map(int, input().split())
    vectors = [
        list(map(float, input().split())) for _ in range(vector_count)
    ]
    weight_1 = list(map(float, input().split()))
    weight_2 = list(map(float, input().split()))

    scores = []
    weight_2_sum = sum(weight_2)
    scale = math.sqrt(dimension)

    for start in range(0, vector_count, block_size):
        end = min(start + block_size, vector_count)
        current_size = end - start
        dimension_sums = [0.0] * dimension
        for vector_id in range(start, end):
            for index, value in enumerate(vectors[vector_id]):
                dimension_sums[index] += value

        hidden = [
            dimension_sum / current_size
            for dimension_sum in dimension_sums
        ]
        linear_score = (
            sum(
                weight * value
                for weight, value in zip(weight_1, hidden)
            )
            + 2.0
        )
        activated = max(0.0, linear_score)
        scores.append(
            (activated * weight_2_sum + dimension) / scale
        )

    if len(scores) < 2:
        print(0)
        return

    total = sum(scores)
    prefix = 0.0
    best = float("-inf")
    for split in range(len(scores) - 1):
        prefix += scores[split]
        best = max(best, min(prefix, total - prefix))

    answer = Decimal(str(best * 100)).quantize(
        Decimal("1"), rounding=ROUND_HALF_UP
    )
    print(answer)

if __name__ == "__main__":
    solve()
