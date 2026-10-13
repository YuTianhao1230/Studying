
if __name__ == "__main__":
    B = int(input())  # 预算
    n = int(input())
    items = []  # 商品的类别、价格和满意度
    for _ in range(n):
        _, cat, price, score = map(int, input().split())
        items.append((cat - 1, price, score))

    best = 0

    def search(i, price, score, cats):
        global best
        if i == n:
            cut = 30 if cats.bit_count() >= 3 else 20  # 每满 200 的减免额
            paid = price - (price // 200) * cut  # 折后价格
            if paid <= B:
                best = max(best, score)
            return
        search(i + 1, price, score, cats)
        cat, p, s = items[i]
        search(
            i + 1,
            price + p,
            score + s,
            cats | (1 << cat),
        )

    search(0, 0, 0, 0)
    print(best)
