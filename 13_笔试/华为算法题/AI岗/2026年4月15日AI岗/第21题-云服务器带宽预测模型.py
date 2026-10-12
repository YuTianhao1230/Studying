import math
import sys
from decimal import Decimal, ROUND_HALF_EVEN

def format_half_even(value):
    rounded = Decimal(str(value)).quantize(
        Decimal("0.000000"), rounding=ROUND_HALF_EVEN
    )
    if rounded == 0:
        rounded = Decimal("0.000000")
    return format(rounded, ".6f")

def solve():
    input = sys.stdin.buffer.readline
    sample_count = int(input())

    beta_1 = 0.9
    beta_2 = 0.999
    learning_rate = 0.001
    weight_decay = 0.01
    epsilon = 1e-8

    parameters = [0.0, 0.0, 0.0]
    first_moments = [0.0, 0.0, 0.0]
    second_moments = [0.0, 0.0, 0.0]
    beta_1_power = 1.0
    beta_2_power = 1.0

    for _ in range(sample_count):
        feature_1, feature_2, target = map(float, input().split())
        prediction = (
            parameters[0] * feature_1
            + parameters[1] * feature_2
            + parameters[2]
        )
        error_twice = 2.0 * (prediction - target)
        gradients = [
            error_twice * feature_1,
            error_twice * feature_2,
            error_twice,
        ]

        beta_1_power *= beta_1
        beta_2_power *= beta_2
        new_parameters = [0.0] * 3

        for index, gradient in enumerate(gradients):
            first_moments[index] = (
                beta_1 * first_moments[index]
                + (1.0 - beta_1) * gradient
            )
            second_moments[index] = (
                beta_2 * second_moments[index]
                + (1.0 - beta_2) * gradient * gradient
            )
            corrected_first = first_moments[index] / (
                1.0 - beta_1_power
            )
            corrected_second = second_moments[index] / (
                1.0 - beta_2_power
            )
            new_parameters[index] = parameters[index] - learning_rate * (
                corrected_first / (math.sqrt(corrected_second) + epsilon)
                + weight_decay * parameters[index]
            )

        parameters = new_parameters

    print(" ".join(format_half_even(value) for value in parameters))

if __name__ == "__main__":
    solve()
