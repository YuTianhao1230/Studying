import heapq
import sys

def solve():
    input = sys.stdin.buffer.readline
    capacity, operation_count = map(int, input().split())
    records = {}
    prune_heap = []
    output = []

    for _ in range(operation_count):
        parts = input().split()
        if parts[0] == b"ADD":
            position = int(parts[1])
            score_text = parts[2].decode()
            score = float(score_text)
            key = b" ".join(input().split()).decode()
            value = b" ".join(input().split()).decode()
            records[position] = (score_text, key, value)
            heapq.heappush(prune_heap, (score, position))
            if len(records) > capacity:
                _, removed = heapq.heappop(prune_heap)
                del records[removed]
                output.append(f"PRUNED {removed}")
        else:
            output.append(str(len(records)))
            for position in sorted(records):
                score_text, key, value = records[position]
                output.extend((f"{position} {score_text}", key, value))

    print("\n".join(output))

if __name__ == "__main__":
    solve()
