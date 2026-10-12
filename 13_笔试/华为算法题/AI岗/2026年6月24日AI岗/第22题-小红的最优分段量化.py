import sys

def solve():
    input = sys.stdin.buffer.readline
    n, segment_limit = map(int, input().split())
    values = list(map(int, input().split()))
    segment_limit = min(segment_limit, n)
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + value)

    def cost(left, right):
        middle = (left + right - 1) // 2
        median = values[middle]
        left_cost = median * (middle - left) - (prefix[middle] - prefix[left])
        right_cost = prefix[right] - prefix[middle + 1] - median * (right - middle - 1)
        return left_cost + right_cost

    infinity = 10**30
    previous = [infinity] * (n + 1)
    previous[0] = 0

    for segments in range(1, segment_limit + 1):
        current = [infinity] * (n + 1)

        def compute(left, right, opt_left, opt_right):
            if left > right:
                return
            middle = (left + right) // 2
            best_value = infinity
            best_split = -1
            upper = min(middle - 1, opt_right)
            for split in range(opt_left, upper + 1):
                candidate = previous[split] + cost(split, middle)
                if candidate < best_value:
                    best_value = candidate
                    best_split = split
            current[middle] = best_value
            compute(left, middle - 1, opt_left, best_split)
            compute(middle + 1, right, best_split, opt_right)

        compute(segments, n, segments - 1, n - 1)
        previous = current

    print(previous[n])

if __name__ == "__main__":
    solve()
