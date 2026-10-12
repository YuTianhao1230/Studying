import sys

def solve():
    input = sys.stdin.buffer.readline
    tile_count, turn_limit = map(int, input().split())
    values = list(map(int, input().split()))
    if 3 * turn_limit < tile_count:
        print(-1)
        return

    moves = min(turn_limit, tile_count)
    negative = -10**18
    previous = [0]
    previous_low = 0

    for used in range(1, moves + 1):
        remaining = moves - used
        low = max(used, tile_count - 3 * remaining)
        high = min(3 * used, tile_count - remaining)
        current = [negative] * (high - low + 1)
        previous_length = len(previous)

        for position in range(low, high + 1):
            index = position - 1 - previous_low
            best = previous[index] if 0 <= index < previous_length else negative
            if 0 <= index - 1 < previous_length and previous[index - 1] > best:
                best = previous[index - 1]
            if 0 <= index - 2 < previous_length and previous[index - 2] > best:
                best = previous[index - 2]
            current[position - low] = best + values[position - 1]

        previous = current
        previous_low = low

    print(previous[tile_count - previous_low])

if __name__ == "__main__":
    solve()
