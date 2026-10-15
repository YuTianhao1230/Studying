
if __name__ == "__main__":
    n, m = map(int, input().split())
    K = int(input())
    x, y = m // 2, n // 2  # 当前坐标：x 为行，y 为列
    sx, sy = x, y  # 起点
    val = [[0] * n for _ in range(m)]  # m 行 n 列，每个坐标的累计值

    for _ in range(K):
        move, op, param = input().split()
        if move == "U":
            x = max(0, x - 1)
        elif move == "D":
            x = min(m - 1, x + 1)
        elif move == "L":
            y = max(0, y - 1)
        else:
            y = min(n - 1, y + 1)
        param = int(param)  # 本次增减量
        if op == "I":
            val[x][y] += param
        elif op == "D":
            val[x][y] -= param

    pts = [(i, j) for i in range(m) for j in range(n) if val[i][j] != 0]  # 必须访问的修改位置
    if not pts:
        print(abs(sx - x) + abs(sy - y))
        raise SystemExit

    cnt = len(pts)
    INF = 10**9
    visited = [False] * cnt
    memo = {}

    def dfs(last):
        if all(visited):
            return abs(pts[last][0] - x) + abs(pts[last][1] - y)

        state = (last, tuple(visited))
        if state in memo:
            return memo[state]

        best = INF
        for nxt in range(cnt):
            if visited[nxt]:
                continue
            dis = abs(pts[last][0] - pts[nxt][0]) + abs(pts[last][1] - pts[nxt][1])
            visited[nxt] = True
            best = min(best, dis + dfs(nxt))
            visited[nxt] = False

        memo[state] = best
        return best

    ans = INF
    for first, pos in enumerate(pts):
        dis = abs(sx - pos[0]) + abs(sy - pos[1])
        if dis == 0:
            dis = 1 if n == 1 or m == 1 else 2
        visited[first] = True
        ans = min(ans, dis + dfs(first))
        visited[first] = False
    print(ans)
