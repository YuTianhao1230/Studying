import sys
from decimal import Decimal, ROUND_HALF_UP

def pareto(states):
    best_fp_by_tp = {}
    for true_positive, false_positive in states:
        old = best_fp_by_tp.get(true_positive)
        if old is None or false_positive < old:
            best_fp_by_tp[true_positive] = false_positive

    frontier = []
    best_fp = float("inf")
    for true_positive in sorted(best_fp_by_tp, reverse=True):
        false_positive = best_fp_by_tp[true_positive]
        if false_positive < best_fp:
            frontier.append((true_positive, false_positive))
            best_fp = false_positive
    return frontier

def solve():
    input = sys.stdin.buffer.readline
    node_count, sample_count, dimension = map(int, input().split())
    nodes = [None]
    for _ in range(node_count):
        nodes.append(tuple(map(int, input().split())))

    features = []
    labels = []
    for _ in range(sample_count):
        row = list(map(int, input().split()))
        features.append(row[:dimension])
        labels.append(row[dimension])

    def subtree_states(node_id, sample_ids):
        left, right, feature, threshold, node_label = nodes[node_id]
        positive_count = sum(labels[index] for index in sample_ids)
        negative_count = len(sample_ids) - positive_count

        if node_label == 1:
            states = [(positive_count, negative_count)]
        else:
            states = [(0, 0)]

        if left == 0:
            return states

        left_ids = []
        right_ids = []
        feature_index = feature - 1
        for index in sample_ids:
            if features[index][feature_index] <= threshold:
                left_ids.append(index)
            else:
                right_ids.append(index)

        left_states = subtree_states(left, left_ids)
        right_states = subtree_states(right, right_ids)
        states.extend(
            (left_tp + right_tp, left_fp + right_fp)
            for left_tp, left_fp in left_states
            for right_tp, right_fp in right_states
        )
        return pareto(states)

    total_positive = sum(labels)
    root_states = subtree_states(1, list(range(sample_count)))

    best_numerator = 0
    best_denominator = 1
    for true_positive, false_positive in root_states:
        numerator = 2 * true_positive
        denominator = total_positive + true_positive + false_positive
        if denominator and (
            numerator * best_denominator
            > best_numerator * denominator
        ):
            best_numerator = numerator
            best_denominator = denominator

    result = (
        Decimal(best_numerator) / Decimal(best_denominator)
    ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    print(result)

if __name__ == "__main__":
    solve()
