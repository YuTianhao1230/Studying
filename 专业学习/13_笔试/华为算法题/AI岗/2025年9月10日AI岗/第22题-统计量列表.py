import ast
import math
import sys
from decimal import Decimal, ROUND_HALF_UP

def format_number(x):
    val = Decimal(str(x)).quantize(  # 四舍五入后的数值
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )
    if val == 0:
        return "0"
    return format(val, "f").rstrip("0").rstrip(".")

def statistics(a):
    n = len(a)  # 当前窗口长度
    mean = sum(a) / n

    if n == 1:
        std = 0.0  # 样本标准差
        slope = 0.0
    else:
        var = sum((x - mean) ** 2 for x in a) / (n - 1)
        std = math.sqrt(var)

        mx = (n - 1) / 2  # 横坐标均值
        up = sum((i - mx) * (x - mean) for i, x in enumerate(a))
        down = sum((i - mx) ** 2 for i in range(n))
        slope = up / down

    return (
        mean,
        std,
        min(a),
        max(a),
        slope,
    )

def solve_line(line):
    try:
        a, ws = ast.literal_eval(line)  # 数值序列和窗口序列
    except (SyntaxError, ValueError, TypeError):
        return ["[]"]

    if (
        not isinstance(a, list)
        or not isinstance(ws, list)
        or not a
        or not ws
        or any(not isinstance(w, int) or w <= 0 for w in ws)
    ):
        return ["[]"]

    W = max(ws)  # 最长窗口
    if len(a) < W:
        return ["[]"]

    out = []  # 当前测试用例的输出行
    for r in range(W - 1, len(a)):
        row = []  # 当前右边界的全部统计量
        for w in ws:
            cur = a[r - w + 1 : r + 1]  # 当前窗口数据
            row.extend(format_number(x) for x in statistics(cur))
        out.append("[" + ", ".join(row) + "]")
    return out

if __name__ == "__main__":
    out = []  # 所有测试用例的输出行
    for raw in sys.stdin:
        line = raw.strip()
        if line:
            out.extend(solve_line(line))
    print("\n".join(out))
