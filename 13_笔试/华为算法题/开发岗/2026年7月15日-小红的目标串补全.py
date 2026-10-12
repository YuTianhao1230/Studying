import sys

def solve():
    input = sys.stdin.buffer.readline
    target = input().strip()
    source = input().strip()
    dp = [0] * (len(source) + 1)

    for target_character in target:
        diagonal = 0
        for index, source_character in enumerate(source, 1):
            old = dp[index]
            if target_character == source_character:
                dp[index] = diagonal + 1
            elif dp[index - 1] > dp[index]:
                dp[index] = dp[index - 1]
            diagonal = old

    print(len(target) - dp[-1])

if __name__ == "__main__":
    solve()
