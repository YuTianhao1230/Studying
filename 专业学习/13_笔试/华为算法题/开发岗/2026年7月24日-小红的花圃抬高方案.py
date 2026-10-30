import sys

if __name__ == "__main__":
    data = list(map(int, sys.stdin.buffer.read().split()))
    B, C, F, U, n = data[:5]
    h = data[5 : 5 + n]  # 各花圃初始高度

    def feasible(H):
        soil = sum(max(0, H - x) for x in h)  # 所需土量
        cars = (soil + C - 1) // C  # 所需车辆数
        return soil * U + cars * F <= B

    low = min(h)  # 可行下界
    high = low + B // U + 1  # 不可行上界
    while low + 1 < high:
        mid = (low + high) // 2
        if feasible(mid):
            low = mid
        else:
            high = mid
    print(low)
