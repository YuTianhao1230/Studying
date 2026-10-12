import math
import sys

def entropy(zero_count, one_count):
    total = zero_count + one_count
    result = 0.0
    for count in (zero_count, one_count):
        if count:
            probability = count / total
            result -= probability * math.log2(probability)
    return result

def solve():
    sys.setrecursionlimit(1_000_000)
    input = sys.stdin.buffer.readline
    sample_count, feature_count = map(int, input().split())

    features = []
    labels = []
    for _ in range(sample_count):
        row = list(map(int, input().split()))
        features.append(row[:feature_count])
        labels.append(row[feature_count])

    def build(sample_ids, remaining_features, fallback_label=0):
        if not sample_ids:
            return fallback_label

        one_count = sum(labels[index] for index in sample_ids)
        zero_count = len(sample_ids) - one_count
        majority_label = 1 if one_count > zero_count else 0

        if zero_count == 0 or one_count == 0:
            return labels[sample_ids[0]]
        if not remaining_features:
            return majority_label

        base_entropy = entropy(zero_count, one_count)
        best_feature = -1
        best_gain = 0.0
        epsilon = 1e-12

        for feature_id in remaining_features:
            branch_counts = [[0, 0], [0, 0]]
            for sample_id in sample_ids:
                value = features[sample_id][feature_id]
                branch_counts[value][labels[sample_id]] += 1

            conditional_entropy = 0.0
            for branch_zero, branch_one in branch_counts:
                branch_size = branch_zero + branch_one
                if branch_size:
                    conditional_entropy += (
                        branch_size
                        / len(sample_ids)
                        * entropy(branch_zero, branch_one)
                    )
            gain = base_entropy - conditional_entropy
            if gain > best_gain + epsilon:
                best_gain = gain
                best_feature = feature_id

        if best_feature == -1:
            return majority_label

        left_ids = []
        right_ids = []
        for sample_id in sample_ids:
            if features[sample_id][best_feature] == 0:
                left_ids.append(sample_id)
            else:
                right_ids.append(sample_id)

        next_features = tuple(
            feature_id
            for feature_id in remaining_features
            if feature_id != best_feature
        )
        return (
            best_feature,
            build(left_ids, next_features, majority_label),
            build(right_ids, next_features, majority_label),
        )

    tree = build(
        list(range(sample_count)),
        tuple(range(feature_count)),
    )

    def predict(query):
        node = tree
        while isinstance(node, tuple):
            feature_id, left, right = node
            node = left if query[feature_id] == 0 else right
        return node

    query_count = int(input())
    output = []
    for _ in range(query_count):
        query = list(map(int, input().split()))
        output.append(str(predict(query)))
    print("\n".join(output))

if __name__ == "__main__":
    solve()
