import sys

def solve():
    input = sys.stdin.buffer.readline
    line_count = int(input())
    lines = []
    positions = []
    for _ in range(line_count):
        data = list(map(int, input().split()))
        route = data[1:]
        lines.append(route)
        positions.append({station: index for index, station in enumerate(route)})
    start, destination = map(int, input().split())

    infinity = 10**9
    answer = infinity
    for i, first_line in enumerate(lines):
        start_index = positions[i].get(start)
        if start_index is None:
            continue
        destination_index = positions[i].get(destination)
        if destination_index is not None and destination_index >= start_index:
            answer = min(answer, destination_index - start_index + 1)

        for j in range(line_count):
            destination_index = positions[j].get(destination)
            if destination_index is None:
                continue
            for transfer_index in range(start_index, len(first_line)):
                transfer = first_line[transfer_index]
                second_index = positions[j].get(transfer)
                if second_index is not None and second_index <= destination_index:
                    count = transfer_index - start_index + destination_index - second_index + 1
                    answer = min(answer, count)

    print(-1 if answer == infinity else answer)

if __name__ == "__main__":
    solve()
