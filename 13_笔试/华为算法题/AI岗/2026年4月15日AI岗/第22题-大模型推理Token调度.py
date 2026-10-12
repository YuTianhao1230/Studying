import sys

def solve():
    line = sys.stdin.buffer.readline().decode().strip()
    priorities = [int(value.strip()) for value in line.split(",")]
    task_count = len(priorities)

    left = [0] * task_count
    for index, priority in enumerate(priorities):
        if priority <= 0:
            continue
        left[index] = 1
        if (
            index > 0
            and priorities[index - 1] > 0
            and priority > priorities[index - 1]
        ):
            left[index] = left[index - 1] + 1

    right = [0] * task_count
    for index in range(task_count - 1, -1, -1):
        if priorities[index] <= 0:
            continue
        right[index] = 1
        if (
            index + 1 < task_count
            and priorities[index + 1] > 0
            and priorities[index] > priorities[index + 1]
        ):
            right[index] = right[index + 1] + 1

    print(
        sum(
            max(left[index], right[index])
            for index in range(task_count)
        )
    )

if __name__ == "__main__":
    solve()
