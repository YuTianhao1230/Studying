import sys

def solve():
    input = sys.stdin.buffer.readline
    budget = int(input())
    item_count = int(input())
    items = []
    for _ in range(item_count):
        _, category, price, satisfaction = map(int, input().split())
        items.append((category - 1, price, satisfaction))

    best = 0

    def search(index, price, satisfaction, categories):
        nonlocal best
        if index == item_count:
            discount = 30 if categories.bit_count() >= 3 else 20
            paid = price - (price // 200) * discount
            if paid <= budget:
                best = max(best, satisfaction)
            return
        search(index + 1, price, satisfaction, categories)
        category, item_price, item_satisfaction = items[index]
        search(
            index + 1,
            price + item_price,
            satisfaction + item_satisfaction,
            categories | (1 << category),
        )

    search(0, 0, 0, 0)
    print(best)

if __name__ == "__main__":
    solve()
