from decimal import Decimal, ROUND_HALF_UP

def pareto(states):
    best = {}  # 每个 TP 对应的最小 FP
    for tp, fp in states:
        old = best.get(tp)
        if old is None or fp < old:
            best[tp] = fp

    ans = []  # 未被支配的状态
    min_fp = float("inf")  # 更大 TP 下出现过的最小 FP
    for tp in sorted(best, reverse=True):
        fp = best[tp]
        if fp < min_fp:
            ans.append((tp, fp))
            min_fp = fp
    return ans

if __name__ == "__main__":
    N, M, K = map(int, input().split())
    nodes = [None]
    for _ in range(N):
        nodes.append(tuple(map(int, input().split())))

    X = []  # 验证样本特征
    y = []  # 验证样本标签
    for _ in range(M):
        row = list(map(int, input().split()))
        X.append(row[:K])
        y.append(row[K])

    def subtree_states(node_id, ids):
        left, right, feature, threshold, label = nodes[node_id]
        pos = sum(y[i] for i in ids)  # 当前节点的正样本数
        neg = len(ids) - pos  # 当前节点的负样本数

        if label == 1:
            states = [(pos, neg)]
        else:
            states = [(0, 0)]

        if left == 0:
            return states

        lids = []  # 进入左子树的样本编号
        rids = []  # 进入右子树的样本编号
        f = feature - 1  # 转为从 0 开始的特征编号
        for i in ids:
            if X[i][f] <= threshold:
                lids.append(i)
            else:
                rids.append(i)

        ls = subtree_states(left, lids)  # 左子树可行状态
        rs = subtree_states(right, rids)  # 右子树可行状态
        states.extend(
            (ltp + rtp, lfp + rfp)
            for ltp, lfp in ls
            for rtp, rfp in rs
        )
        return pareto(states)

    P = sum(y)  # 验证集正样本总数
    states = subtree_states(1, list(range(M)))  # 根节点可行状态

    up = 0  # 最优 F1 的分子
    down = 1  # 最优 F1 的分母
    for tp, fp in states:
        a = 2 * tp
        b = P + tp + fp
        if b and a * down > up * b:
            up = a
            down = b

    ans = (Decimal(up) / Decimal(down)).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )
    print(ans)
