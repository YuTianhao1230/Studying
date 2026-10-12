import sys

def solve():
    input = sys.stdin.buffer.readline
    n = int(input())
    activities = [tuple(map(int, input().split())) for _ in range(n)]
    activities.sort(key=lambda item: item[1])
    answer = 0
    last_end = -1
    for start, end in activities:
        if start >= last_end:
            answer += 1
            last_end = end
    print(answer)

if __name__ == "__main__":
    solve()
