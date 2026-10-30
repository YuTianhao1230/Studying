import sys
from collections import defaultdict, deque

if __name__ == "__main__":
    data = sys.stdin.buffer.read().split()  # 全部输入 token
    if not data:
        raise SystemExit
    n = int(data[0])
    seq = data[1 : n + 1]  # 层序序列

    val = [int(seq[0])]  # 各非空节点的值
    child = [[-1, -1]]  # 各节点的左右孩子编号
    q = deque([0])  # 待连接孩子的节点
    pos = 1  # 当前层序 token 下标
    while q and pos < n:
        fa = q.popleft()  # 当前父节点
        for side in range(2):
            if pos >= n:
                break
            token = seq[pos]
            pos += 1
            if token == b"None":
                continue
            u = len(val)  # 新节点编号
            val.append(int(token))
            child.append([-1, -1])
            child[fa][side] = u
            q.append(u)

    cnt = defaultdict(int)  # 当前路径上各前缀和的出现次数
    cnt[0] = 1
    ans = 0  # 和为 0 的非空链路数
    st = [(0, 0, False)]  # (节点, 父路径和, 是否退出)

    while st:
        u, pre, leave = st.pop()
        cur = pre + val[u]  # 根到当前节点的路径和
        if leave:
            cnt[cur] -= 1
            continue

        ans += cnt[cur]
        if val[u] == 0:
            ans -= 1

        cnt[cur] += 1
        st.append((u, pre, True))
        right = child[u][1]  # 右孩子编号
        left = child[u][0]  # 左孩子编号
        if right != -1:
            st.append((right, cur, False))
        if left != -1:
            st.append((left, cur, False))

    print(ans)
