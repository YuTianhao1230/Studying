import math
import sys

def solve():
    input = sys.stdin.buffer.readline
    requested_centers, point_count, speed = map(int, input().split())
    points = [tuple(map(float, input().split())) for _ in range(point_count)]

    if requested_centers >= point_count:
        centers = points[:]
    else:
        centers = sorted(
            points, key=lambda point: point[0] ** 2 + point[1] ** 2
        )[:requested_centers]

        for _ in range(50):
            sum_x = [0.0] * requested_centers
            sum_y = [0.0] * requested_centers
            counts = [0] * requested_centers

            for x, y in points:
                best_cluster = 0
                best_distance = float("inf")
                for cluster_id, (center_x, center_y) in enumerate(centers):
                    distance = (x - center_x) ** 2 + (y - center_y) ** 2
                    if distance < best_distance:
                        best_distance = distance
                        best_cluster = cluster_id
                sum_x[best_cluster] += x
                sum_y[best_cluster] += y
                counts[best_cluster] += 1

            new_centers = centers[:]
            for cluster_id in range(requested_centers):
                if counts[cluster_id]:
                    new_centers[cluster_id] = (
                        sum_x[cluster_id] / counts[cluster_id],
                        sum_y[cluster_id] / counts[cluster_id],
                    )

            movement = sum(
                math.dist(old_center, new_center)
                for old_center, new_center in zip(centers, new_centers)
            )
            centers = new_centers
            if movement < 1e-4:
                break

    centers.sort(key=lambda point: math.hypot(point[0], point[1]))
    total_distance = 0.0
    previous = (0.0, 0.0)
    for center in centers:
        total_distance += math.dist(previous, center)
        previous = center
    total_distance += math.dist(previous, (0.0, 0.0))

    total_seconds = total_distance * 3600 / speed
    print(math.floor(total_seconds))

if __name__ == "__main__":
    solve()
