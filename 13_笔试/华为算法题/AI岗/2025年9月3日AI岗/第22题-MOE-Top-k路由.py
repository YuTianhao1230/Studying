import sys

def solve():
    input = sys.stdin.buffer.readline
    expert_count, group_count, selected_groups, top_k = map(
        int, input().split()
    )
    probabilities = list(map(float, input().split()))

    if (
        expert_count % group_count != 0
        or selected_groups > group_count
    ):
        print("error")
        return

    group_size = expert_count // group_count
    if selected_groups * group_size < top_k:
        print("error")
        return

    representatives = []
    for group in range(group_count):
        start = group * group_size
        end = start + group_size
        representative_probability = max(probabilities[start:end])
        representatives.append((representative_probability, group))

    representatives.sort(key=lambda item: (-item[0], item[1]))
    chosen_groups = {
        group for _, group in representatives[:selected_groups]
    }

    candidates = [
        expert
        for group in chosen_groups
        for expert in range(
            group * group_size,
            (group + 1) * group_size,
        )
    ]
    candidates.sort(key=lambda expert: (-probabilities[expert], expert))
    answer = sorted(candidates[:top_k])
    print(*answer)

if __name__ == "__main__":
    solve()
