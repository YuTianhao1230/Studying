import sys

def solve():
    input = sys.stdin.readline

    text = input().strip()
    n = int(input())

    words = []
    scores = []
    word_to_id = {}

    for word_id in range(n):
        word, score = input().split()
        words.append(word)
        scores.append(int(score))
        word_to_id[word] = word_id

    m = int(input())
    bonus = {}

    for _ in range(m):
        prev_word, next_word, value = input().split()

        if prev_word in word_to_id and next_word in word_to_id:
            prev_id = word_to_id[prev_word]
            next_id = word_to_id[next_word]
            bonus[(prev_id, next_id)] = int(value)

    terminal = "#"
    trie = {}
    for word_id, word in enumerate(words):
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[terminal] = word_id

    length = len(text)

    dp = [{} for _ in range(length + 1)]

    for start in range(length):
        if start > 0 and not dp[start]:
            continue

        node = trie
        for end in range(start, length):
            char = text[end]
            if char not in node:
                break

            node = node[char]
            if terminal not in node:
                continue

            current_id = node[terminal]

            if start == 0:
                candidate = scores[current_id]
            else:
                best_prev = max(
                    prev_score + bonus.get((prev_id, current_id), 0)
                    for prev_id, prev_score in dp[start].items()
                )
                candidate = best_prev + scores[current_id]

            old_score = dp[end + 1].get(current_id)
            if old_score is None or candidate > old_score:
                dp[end + 1][current_id] = candidate

    print(max(dp[length].values()) if dp[length] else 0)

if __name__ == "__main__":
    solve()
