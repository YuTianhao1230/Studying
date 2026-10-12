import sys

def solve():
    data = list(map(int, sys.stdin.buffer.read().split()))
    budget, capacity, transport_fee, unit_cost, count = data[:5]
    heights = data[5 : 5 + count]

    def feasible(target):
        soil = sum(max(0, target - height) for height in heights)
        trucks = (soil + capacity - 1) // capacity
        return soil * unit_cost + trucks * transport_fee <= budget

    low = min(heights)
    high = low + budget // unit_cost + 1
    while low + 1 < high:
        middle = (low + high) // 2
        if feasible(middle):
            low = middle
        else:
            high = middle
    print(low)

if __name__ == "__main__":
    solve()
