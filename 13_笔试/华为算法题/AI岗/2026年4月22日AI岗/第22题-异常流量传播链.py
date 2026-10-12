import sys

def solve():
    input = sys.stdin.buffer.readline
    station_count, distance_limit, weight_threshold = map(
        int, input().split()
    )
    stations = [tuple(map(int, input().split())) for _ in range(station_count)]

    adjacent = [[False] * station_count for _ in range(station_count)]
    neighborhood_weight = [0] * station_count
    for first in range(station_count):
        x_1, y_1, _, _, _ = stations[first]
        for second in range(station_count):
            x_2, y_2, _, weight, _ = stations[second]
            if abs(x_1 - x_2) + abs(y_1 - y_2) <= distance_limit:
                adjacent[first][second] = True
                neighborhood_weight[first] += weight

    critical = [
        index
        for index in range(station_count)
        if neighborhood_weight[index] >= weight_threshold
    ]
    critical.sort(key=lambda index: stations[index][2])

    best_ending = [stations[index][4] for index in range(station_count)]
    answer = 0
    for target_position, target in enumerate(critical):
        target_time = stations[target][2]
        for source in critical[:target_position]:
            if stations[source][2] >= target_time:
                continue
            if adjacent[source][target]:
                best_ending[target] = max(
                    best_ending[target],
                    best_ending[source] + stations[target][4],
                )
                answer = max(answer, best_ending[target])

    print(answer)

if __name__ == "__main__":
    solve()
