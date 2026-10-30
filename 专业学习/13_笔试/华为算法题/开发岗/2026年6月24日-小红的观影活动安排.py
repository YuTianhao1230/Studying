
if __name__ == "__main__":
    n = int(input())
    acts = [tuple(map(int, input().split())) for _ in range(n)]  # 活动区间
    acts.sort(key=lambda x: x[1])
    ans = 0
    last = -1  # 上一个已选活动的结束时间
    for s_i, e_i in acts:
        if s_i >= last:
            ans += 1
            last = e_i
    print(ans)
