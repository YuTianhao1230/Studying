import math
import statistics
import sys

def parse_number(value):
    try:
        number = float(value)
    except ValueError:
        return None
    return None if math.isnan(number) else number

def is_valid(column, value):
    if value is None:
        return False
    if column in (0, 1):
        return value >= 0
    if column in (2, 3):
        return 0 <= value <= 1000
    return 0 <= value <= 20

def sigmoid(value):
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)

def solve():
    input = sys.stdin.readline
    raw_count = int(input())
    retained = []

    for _ in range(raw_count):
        parts = input().strip().split(",")
        if len(parts) != 7:
            continue
        try:
            label = int(parts[6])
        except ValueError:
            continue
        if label not in (0, 1):
            continue
        retained.append((parts[1:6], label))

    valid_by_column = [[] for _ in range(5)]
    for values, _ in retained:
        for column, raw in enumerate(values):
            value = parse_number(raw)
            if is_valid(column, value):
                valid_by_column[column].append(value)

    means = []
    medians = []
    for values in valid_by_column:
        if values:
            means.append(sum(values) / len(values))
            medians.append(float(statistics.median(values)))
        else:
            means.append(0.0)
            medians.append(0.0)

    def clean(raw_values):
        result = []
        for column, raw in enumerate(raw_values):
            value = parse_number(raw)
            if value is None:
                result.append(means[column])
            elif not is_valid(column, value):
                result.append(medians[column])
            else:
                result.append(value)
        return result

    training = [(clean(values), label) for values, label in retained]
    weights = [0.0] * 5
    bias = 0.0

    if training:
        sample_count = len(training)
        for _ in range(100):
            bias_gradient = 0.0
            gradients = [0.0] * 5
            for features, label in training:
                linear = bias + sum(
                    weights[index] * features[index] for index in range(5)
                )
                error = sigmoid(linear) - label
                bias_gradient += error
                for index in range(5):
                    gradients[index] += error * features[index]

            bias -= 0.01 * bias_gradient / sample_count
            for index in range(5):
                weights[index] -= 0.01 * gradients[index] / sample_count

    prediction_count = int(input())
    for _ in range(prediction_count):
        parts = input().strip().split(",")
        features = clean(parts[1:6])
        linear = bias + sum(
            weights[index] * features[index] for index in range(5)
        )
        print(1 if sigmoid(linear) >= 0.5 else 0)

if __name__ == "__main__":
    solve()
