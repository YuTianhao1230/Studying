import sys
from decimal import Decimal, ROUND_HALF_UP

def solve():
    input = sys.stdin.buffer.readline
    height, width, kernel_size = map(int, input().split())
    image = [list(map(float, input().split())) for _ in range(height)]
    kernel = [
        list(map(float, input().split())) for _ in range(kernel_size)
    ]

    radius = kernel_size // 2
    dynamic = [0.0] * height

    for column in range(width):
        energy = [0.0] * height
        for row in range(height):
            value = 0.0
            for kernel_row in range(kernel_size):
                image_row = row + kernel_row - radius
                if not 0 <= image_row < height:
                    continue
                for kernel_column in range(kernel_size):
                    image_column = column + kernel_column - radius
                    if 0 <= image_column < width:
                        value += (
                            kernel[kernel_row][kernel_column]
                            * image[image_row][image_column]
                        )
            energy[row] = value

        if column == 0:
            dynamic = energy
            continue

        new_dynamic = [0.0] * height
        for row in range(height):
            best_previous = dynamic[row]
            if row > 0:
                best_previous = max(best_previous, dynamic[row - 1])
            if row + 1 < height:
                best_previous = max(best_previous, dynamic[row + 1])
            new_dynamic[row] = best_previous + energy[row]
        dynamic = new_dynamic

    answer = Decimal(str(max(dynamic))).quantize(
        Decimal("0.1"), rounding=ROUND_HALF_UP
    )
    print(answer)

if __name__ == "__main__":
    solve()
