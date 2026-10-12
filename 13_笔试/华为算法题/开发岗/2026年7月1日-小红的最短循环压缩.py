import sys

def solve():
    s = sys.stdin.readline().strip()
    n = len(s)
    prefix = [0] * n
    for i in range(1, n):
        j = prefix[i - 1]
        while j > 0 and s[i] != s[j]:
            j = prefix[j - 1]
        if s[i] == s[j]:
            j += 1
        prefix[i] = j

    period = n - prefix[-1]
    if period < n and n % period == 0:
        print(s[:period] + str(n // period))
    else:
        print(s)

if __name__ == "__main__":
    solve()
