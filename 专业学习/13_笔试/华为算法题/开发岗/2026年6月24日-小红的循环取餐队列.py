from collections import deque

if __name__ == "__main__":
    a = list(map(int, input().split()))  # 各顾客需求量
    n = len(a)
    k = int(input())
    m = int(input())
    q = deque(range(n))  # 等待队列
    time = 0  # 已经过的秒数
    while q:
        cur = [q.popleft() for _ in range(min(m, len(q)))]  # 本秒服务批次
        time += 1
        for i in cur:
            a[i] -= 1
            if a[i] == 0:
                if i == k:
                    print(time)
                    raise SystemExit
            else:
                q.append(i)
