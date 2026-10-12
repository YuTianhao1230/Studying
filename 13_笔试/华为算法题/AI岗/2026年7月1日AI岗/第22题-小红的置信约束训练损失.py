import math
import sys

def solve():
    input = sys.stdin.buffer.readline
    sample_count, class_count, budget, coefficient = input().split()
    sample_count = int(sample_count)
    class_count = int(class_count)
    budget = int(budget)
    coefficient = float(coefficient)
    losses = []
    entropies = []
    costs = []

    for _ in range(sample_count):
        data = input().split()
        logits = list(map(float, data[:class_count]))
        label = int(data[class_count])
        costs.append(int(data[class_count + 1]))
        maximum = max(logits)
        exponentials = [math.exp(value - maximum) for value in logits]
        total = sum(exponentials)
        probabilities = [value / total for value in exponentials]
        losses.append(-math.log(probabilities[label]))
        entropies.append(-sum(p * math.log(p) for p in probabilities))

    dp = [float("-inf")] * (budget + 1)
    dp[0] = 0.0
    take = [bytearray(budget + 1) for _ in range(sample_count)]
    for i, (cost, loss) in enumerate(zip(costs, losses)):
        for used in range(budget, cost - 1, -1):
            candidate = dp[used - cost] + loss
            if candidate > dp[used]:
                dp[used] = candidate
                take[i][used] = 1

    used = max(range(budget + 1), key=lambda value: dp[value])
    selected_entropy = 0.0
    for i in range(sample_count - 1, -1, -1):
        if take[i][used]:
            selected_entropy += entropies[i]
            used -= costs[i]

    answer = sum(losses) + coefficient * selected_entropy
    print(f"{answer:.2f}")

if __name__ == "__main__":
    solve()
