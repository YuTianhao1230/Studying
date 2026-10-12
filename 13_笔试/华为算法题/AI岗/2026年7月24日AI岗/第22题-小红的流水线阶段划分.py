import sys
from collections import deque

def solve():
    input = sys.stdin.buffer.readline
    layer_count, stage_count, time_limit = map(int, input().split())
    layer_times = list(map(int, input().split()))
    communication = (
        list(map(int, input().split())) if layer_count > 1 else []
    )

    prefix = [0] * (layer_count + 1)
    for index, value in enumerate(layer_times, start=1):
        prefix[index] = prefix[index - 1] + value

    infinity = 10**30

    previous = [infinity] * (layer_count + 1)
    for end in range(1, layer_count + 1):
        if prefix[end] > time_limit:
            break
        previous[end] = 0

    if stage_count == 1:
        print(previous[layer_count] if previous[layer_count] < infinity else -1)
        return

    for stages in range(2, stage_count + 1):
        current = [infinity] * (layer_count + 1)

        candidates = deque()

        for end in range(stages, layer_count + 1):
            cut = end - 1

            if previous[cut] < infinity:
                candidate_cost = previous[cut] + communication[cut - 1]
                while candidates:
                    last = candidates[-1]
                    last_cost = previous[last] + communication[last - 1]
                    if last_cost < candidate_cost:
                        break
                    candidates.pop()
                candidates.append(cut)

            while (
                candidates
                and prefix[end] - prefix[candidates[0]] > time_limit
            ):
                candidates.popleft()

            if candidates:
                best_cut = candidates[0]
                current[end] = (
                    previous[best_cut] + communication[best_cut - 1]
                )

        previous = current

    answer = previous[layer_count]
    print(answer if answer < infinity else -1)

if __name__ == "__main__":
    solve()
