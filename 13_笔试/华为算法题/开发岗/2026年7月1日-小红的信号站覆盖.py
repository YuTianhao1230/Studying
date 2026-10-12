import sys

def solve():
    input = sys.stdin.buffer.readline
    n, radius = map(int, input().split())
    positions = sorted(map(int, input().split()))
    answer = 0
    i = 0
    while i < n:
        leftmost = positions[i]
        station_index = i
        while station_index + 1 < n and positions[station_index + 1] <= leftmost + radius:
            station_index += 1
        right_boundary = positions[station_index] + radius
        i = station_index + 1
        while i < n and positions[i] <= right_boundary:
            i += 1
        answer += 1
    print(answer)

if __name__ == "__main__":
    solve()
