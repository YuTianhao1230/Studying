import math
from collections import Counter, defaultdict

if __name__ == "__main__":
    N = int(input())
    docs = [  # 按时间顺序保存的文档词频
        Counter(input().strip().lower().split())
        for _ in range(N)
    ]
    K = int(input())
    P = int(input())

    qs = defaultdict(list)  # 按窗口右端点归组的查询
    for i in range(P):
        parts = input().strip().lower().split()
        t = int(parts[0])
        q = Counter(parts[1:])
        r = min(t, N - 1)  # 当前查询窗口右端点
        qs[r].append((i, q))

    ans = [-1] * P
    limit = 0.6  # 余弦相似度阈值
    eps = 1e-12  # 浮点比较误差

    for r, queries in qs.items():
        l = r - K + 1  # 当前查询窗口左端点
        win = docs[l : r + 1]  # 当前文档窗口

        df = Counter()  # 当前窗口的文档频率
        for doc in win:
            df.update(doc.keys())

        def idf(word):
            return math.log((K + 1) / (df.get(word, 0) + 1)) + 1.0

        info = []  # 各文档的向量模长和时间权重
        inv = defaultdict(list)  # 词到文档权重的倒排索引
        for i, doc in enumerate(win):
            norm2 = 0.0  # 当前文档向量模长的平方
            for word, freq in doc.items():
                val = freq * idf(word)
                norm2 += val * val
                inv[word].append((i, val))
            info.append((math.sqrt(norm2), (i + 1) / K))

        for qid, q in queries:
            qv = {}  # 查询的 TF-IDF 向量
            qnorm2 = 0.0  # 查询向量模长的平方
            for word, freq in q.items():
                val = freq * idf(word)
                qv[word] = val
                qnorm2 += val * val

            if qnorm2 == 0:
                continue

            dots = [0.0] * K  # 查询与各文档的点积
            for word, qval in qv.items():
                for i, dval in inv.get(word, ()):
                    dots[i] += qval * dval

            qnorm = math.sqrt(qnorm2)  # 查询向量模长
            best = -1.0  # 当前最高加权得分
            doc_id = -1  # 当前最佳文档编号

            for i, dot in enumerate(dots):
                dnorm, weight = info[i]
                if dnorm == 0:
                    continue
                cos = dot / (qnorm * dnorm)  # 原始余弦相似度
                if cos + eps < limit:
                    continue

                score = cos * weight
                if score > best + eps:
                    best = score
                    doc_id = l + i

            ans[qid] = doc_id

    print(*ans)
