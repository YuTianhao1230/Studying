import heapq

if __name__ == "__main__":
    K, N = map(int, input().split())
    rec = {}  # pos -> (score 原文, Key, Value)
    hp = []  # 按 score 淘汰缓存项的小根堆
    out = []  # 按顺序保存输出行

    for _ in range(N):
        cmd = input().split()  # 当前操作
        if cmd[0] == "ADD":
            pos = int(cmd[1])
            score = cmd[2]
            key = " ".join(input().split())
            value = " ".join(input().split())
            rec[pos] = (score, key, value)
            heapq.heappush(hp, (float(score), pos))
            if len(rec) > K:
                _, rm = heapq.heappop(hp)  # rm：被淘汰的位置
                del rec[rm]
                out.append(f"PRUNED {rm}")
        else:
            out.append(str(len(rec)))
            for pos in sorted(rec):
                score, key, value = rec[pos]
                out.extend((f"{pos} {score}", key, value))

    print("\n".join(out))
