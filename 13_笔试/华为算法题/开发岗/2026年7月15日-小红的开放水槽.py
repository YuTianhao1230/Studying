import sys

def solve():
    input = sys.stdin.buffer.readline
    count = int(input())
    heights = list(map(int, input().split()))
    suffix_max = heights[:]
    for index in range(count - 2, -1, -1):
        suffix_max[index] = max(suffix_max[index], suffix_max[index + 1])

    answer = 0
    left_max = heights[0]
    for index in range(count - 1):
        left_max = max(left_max, heights[index])
        answer += min(left_max, suffix_max[index + 1])
    print(answer)

if __name__ == "__main__":
    solve()
