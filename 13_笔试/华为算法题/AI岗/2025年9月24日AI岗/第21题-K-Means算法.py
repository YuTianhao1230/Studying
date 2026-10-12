import math
import sys
from decimal import Decimal, ROUND_HALF_EVEN

def solve():
    input = sys.stdin.buffer.readline
    point_count, cluster_count = map(int, input().split())
    points = [tuple(map(int, input().split())) for _ in range(point_count)]

    centers = [
        (float(points[index][0]), float(points[index][1]))
        for index in range(cluster_count)
    ]
    assignments = [-1] * point_count

    while True:
        new_assignments = []
        for x, y in points:
            best_cluster = 0
            best_distance = float("inf")
            for cluster_id, (center_x, center_y) in enumerate(centers):
                distance = (x - center_x) ** 2 + (y - center_y) ** 2
                if distance < best_distance:
                    best_distance = distance
                    best_cluster = cluster_id
            new_assignments.append(best_cluster)

        if new_assignments == assignments:
            break
        assignments = new_assignments

        sum_x = [0] * cluster_count
        sum_y = [0] * cluster_count
        counts = [0] * cluster_count
        for (x, y), cluster_id in zip(points, assignments):
            sum_x[cluster_id] += x
            sum_y[cluster_id] += y
            counts[cluster_id] += 1

        for cluster_id in range(cluster_count):
            if counts[cluster_id]:
                centers[cluster_id] = (
                    sum_x[cluster_id] / counts[cluster_id],
                    sum_y[cluster_id] / counts[cluster_id],
                )

    members = [[] for _ in range(cluster_count)]
    for point_id, cluster_id in enumerate(assignments):
        members[cluster_id].append(point_id)
    nonempty_clusters = [
        cluster_id
        for cluster_id in range(cluster_count)
        if members[cluster_id]
    ]

    distances = [[0.0] * point_count for _ in range(point_count)]
    for first in range(point_count):
        for second in range(first + 1, point_count):
            distance = math.dist(points[first], points[second])
            distances[first][second] = distance
            distances[second][first] = distance

    silhouettes = [0.0] * point_count
    if len(nonempty_clusters) > 1:
        for point_id, own_cluster in enumerate(assignments):
            own_members = members[own_cluster]
            if len(own_members) <= 1:
                continue

            inner_distance = sum(
                distances[point_id][other]
                for other in own_members
                if other != point_id
            ) / (len(own_members) - 1)

            nearest_other = min(
                sum(
                    distances[point_id][other]
                    for other in members[cluster_id]
                )
                / len(members[cluster_id])
                for cluster_id in nonempty_clusters
                if cluster_id != own_cluster
            )
            denominator = max(inner_distance, nearest_other)
            if denominator:
                silhouettes[point_id] = (
                    nearest_other - inner_distance
                ) / denominator

    worst_cluster = nonempty_clusters[0]
    worst_score = float("inf")
    for cluster_id in nonempty_clusters:
        score = sum(
            silhouettes[point_id] for point_id in members[cluster_id]
        ) / len(members[cluster_id])
        if score < worst_score:
            worst_score = score
            worst_cluster = cluster_id

    selected = members[worst_cluster]
    center_x = Decimal(sum(points[index][0] for index in selected)) / Decimal(
        len(selected)
    )
    center_y = Decimal(sum(points[index][1] for index in selected)) / Decimal(
        len(selected)
    )
    center_x = center_x.quantize(Decimal("0.00"), rounding=ROUND_HALF_EVEN)
    center_y = center_y.quantize(Decimal("0.00"), rounding=ROUND_HALF_EVEN)
    print(f"{center_x},{center_y}")

if __name__ == "__main__":
    solve()
