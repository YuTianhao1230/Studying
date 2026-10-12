import sys

def solve():
    input = sys.stdin.buffer.readline
    rows, columns, color_count = map(int, input().split())
    grid = [list(map(int, input().split())) for _ in range(rows)]
    infinity = 10**18
    answer = infinity

    for target_color in range(1, color_count + 1):
        dp = [infinity] * columns
        for row in range(rows):
            for column in range(columns):
                cost = abs(grid[row][column] - target_color)
                if row == 0 and column == 0:
                    dp[column] = cost
                elif row == 0:
                    dp[column] = dp[column - 1] + cost
                elif column == 0:
                    dp[column] += cost
                else:
                    dp[column] = min(dp[column], dp[column - 1]) + cost
        answer = min(answer, dp[-1])
    print(answer)

if __name__ == "__main__":
    solve()
