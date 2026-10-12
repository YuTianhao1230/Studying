import ast
import math
import sys
from decimal import Decimal, ROUND_HALF_UP

def format_number(value):
    rounded = Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )
    if rounded == 0:
        return "0"
    return format(rounded, "f").rstrip("0").rstrip(".")

def statistics(values):
    size = len(values)
    mean = sum(values) / size

    if size == 1:
        standard_deviation = 0.0
        slope = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (
            size - 1
        )
        standard_deviation = math.sqrt(variance)

        mean_x = (size - 1) / 2
        numerator = sum(
            (index - mean_x) * (value - mean)
            for index, value in enumerate(values)
        )
        denominator = sum(
            (index - mean_x) ** 2 for index in range(size)
        )
        slope = numerator / denominator

    return (
        mean,
        standard_deviation,
        min(values),
        max(values),
        slope,
    )

def solve_line(line):
    try:
        values, windows = ast.literal_eval(line)
    except (SyntaxError, ValueError, TypeError):
        return ["[]"]

    if (
        not isinstance(values, list)
        or not isinstance(windows, list)
        or not values
        or not windows
        or any(
            not isinstance(window, int) or window <= 0
            for window in windows
        )
    ):
        return ["[]"]

    maximum_window = max(windows)
    if len(values) < maximum_window:
        return ["[]"]

    output = []
    for right in range(maximum_window - 1, len(values)):
        row = []
        for window in windows:
            current = values[right - window + 1 : right + 1]
            row.extend(format_number(value) for value in statistics(current))
        output.append("[" + ", ".join(row) + "]")
    return output

def solve():
    output = []
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if line:
            output.extend(solve_line(line))
    print("\n".join(output))

if __name__ == "__main__":
    solve()
