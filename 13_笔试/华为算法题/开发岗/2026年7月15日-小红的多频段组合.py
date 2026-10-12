import sys

def solve():
    input = sys.stdin.buffer.readline
    carrier_count, limit = map(int, input().split())
    bandwidths = list(map(int, input().split()))
    bands = [value.decode().lower() for value in input().split()]
    conflict_count = int(input())
    conflict_pairs = []
    for _ in range(conflict_count):
        first, second = input().split()
        conflict_pairs.append((first.decode().lower(), second.decode().lower()))

    representative = {}
    for index, (band, bandwidth) in enumerate(zip(bands, bandwidths)):
        old = representative.get(band)
        if old is None or bandwidth > old[1]:
            representative[band] = (index, bandwidth)

    candidates = sorted(
        ((index, bandwidth, band) for band, (index, bandwidth) in representative.items()),
        key=lambda item: item[0],
    )
    band_to_position = {band: position for position, (_, _, band) in enumerate(candidates)}
    conflicts = [0] * len(candidates)
    for first, second in conflict_pairs:
        if first in band_to_position and second in band_to_position:
            a = band_to_position[first]
            b = band_to_position[second]
            conflicts[a] |= 1 << b
            conflicts[b] |= 1 << a

    best_bandwidth = -1
    best_indices = ()

    def search(start, chosen_mask, total, indices):
        nonlocal best_bandwidth, best_indices
        current = tuple(indices)
        if total > best_bandwidth or (
            total == best_bandwidth and current < best_indices
        ):
            best_bandwidth = total
            best_indices = current
        if len(indices) == limit:
            return
        for position in range(start, len(candidates)):
            if conflicts[position] & chosen_mask:
                continue
            index, bandwidth, _ = candidates[position]
            indices.append(index)
            search(
                position + 1,
                chosen_mask | (1 << position),
                total + bandwidth,
                indices,
            )
            indices.pop()

    search(0, 0, 0, [])
    print(best_bandwidth)
    print(*best_indices)

if __name__ == "__main__":
    solve()
