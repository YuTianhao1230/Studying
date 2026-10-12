import sys

def solve():
    input = sys.stdin.buffer.readline
    task_count, expert_count, top_k, capacity = map(int, input().split())
    loads = [0] * expert_count

    for _ in range(task_count):
        scores = list(map(int, input().split()))

        selected = sorted(
            range(expert_count),
            key=lambda expert: (-scores[expert], expert),
        )[:top_k]

        for expert in selected:
            if loads[expert] < capacity:
                loads[expert] += 1

    load_square_sum = sum(load * load for load in loads)
    print(load_square_sum)
    print(*loads)

if __name__ == "__main__":
    solve()
