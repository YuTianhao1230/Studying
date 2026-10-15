import heapq

K,N = map(int,input().split())
rec = {}  # pos -> (score, Key, Value)
hq = []

for _ in range(N):
    input1 = input().split()
    if input1[0] == 'ADD':
        pos = int(input1[1])
        score = float(input1[2])
        m = input()
        n = input()
        rec[pos] = (score,m,n)
        heapq.heappush(hq, (score, pos))
        if len(rec) > K:
            _, rm = heapq.heappop(hq)
            del rec[rm]
            print(f"PRUNED {rm}")
    else:
        print(len(rec))
        ids = sorted(rec)
        for pos in ids:
            score, m, n = rec[pos]
            print(pos,score)
            print(m)
            print(n)
