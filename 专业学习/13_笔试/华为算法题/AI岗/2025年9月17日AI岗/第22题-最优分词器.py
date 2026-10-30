
if __name__ == "__main__":
    text = input().strip()
    n = int(input())

    words = []  # 词典中的词
    scores = []  # 各词的分值
    ids = {}  # 词到编号的映射

    for i in range(n):
        word, score = input().split()
        words.append(word)
        scores.append(int(score))
        ids[word] = i

    m = int(input())
    bonus = {}  # 相邻词对的转移加分

    for _ in range(m):
        prev, nxt, val = input().split()
        if prev in ids and nxt in ids:
            bonus[(ids[prev], ids[nxt])] = int(val)

    end_mark = "#"  # Trie 中的单词结束标记
    trie = {}  # 词典前缀树
    for i, word in enumerate(words):
        node = trie  # 当前 Trie 节点
        for ch in word:
            node = node.setdefault(ch, {})
        node[end_mark] = i

    L = len(text)  # 待切分字符串长度
    dp = [{} for _ in range(L + 1)]  # 位置和末词对应的最高分

    for l in range(L):
        if l > 0 and not dp[l]:
            continue

        node = trie
        for r in range(l, L):
            ch = text[r]
            if ch not in node:
                break

            node = node[ch]
            if end_mark not in node:
                continue

            cur = node[end_mark]  # 当前匹配词的编号
            if l == 0:
                val = scores[cur]
            else:
                pre = max(  # 接入当前词前的最高分
                    score + bonus.get((last, cur), 0)
                    for last, score in dp[l].items()
                )
                val = pre + scores[cur]

            old = dp[r + 1].get(cur)  # 同状态原有分数
            if old is None or val > old:
                dp[r + 1][cur] = val

    print(max(dp[L].values()) if dp[L] else 0)
