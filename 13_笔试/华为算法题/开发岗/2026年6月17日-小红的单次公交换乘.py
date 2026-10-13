
if __name__ == "__main__":
    n = int(input())
    lines = []  # 各条公交线路
    pos = []  # 每条线路中站点到位置的映射
    for _ in range(n):
        data = list(map(int, input().split()))
        m = data[0]
        route = data[1 : m + 1]  # 当前线路的站点序列
        lines.append(route)
        pos.append({x: i for i, x in enumerate(route)})
    A, B = map(int, input().split())

    inf = 10**9  # 不可达标记
    ans = inf
    for i, first in enumerate(lines):
        a = pos[i].get(A)  # 起点在线路 i 中的位置
        if a is None:
            continue
        b = pos[i].get(B)  # 终点在线路 i 中的位置
        if b is not None and b >= a:
            ans = min(ans, b - a + 1)

        for j in range(n):
            b = pos[j].get(B)
            if b is None:
                continue
            for k in range(a, len(first)):
                x = first[k]  # 换乘站
                p = pos[j].get(x)  # 换乘站在线路 j 中的位置
                if p is not None and p <= b:
                    cnt = k - a + b - p + 1
                    ans = min(ans, cnt)

    print(-1 if ans == inf else ans)
