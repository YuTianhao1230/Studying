
if __name__ == "__main__":
    n = int(input())
    text = []  # 当前文本字符
    undo = []  # 可撤销操作栈
    redo = []  # 可重做操作栈

    def apply(act):
        kind, s = act  # 操作类型和操作内容
        if kind == "append":
            text.extend(s)
        else:
            text.pop()

    def reverse(act):
        kind, s = act  # 操作类型和操作内容
        if kind == "append":
            del text[-len(s):]
        else:
            text.append(s)

    for _ in range(n):
        op = input()  # 当前操作
        if op.startswith("APPEND "):
            act = ("append", op[7:])  # 有效编辑记录
            apply(act)
            undo.append(act)
            redo.clear()
        elif op == "POP":
            if text:
                act = ("pop", text[-1])
                apply(act)
                undo.append(act)
                redo.clear()
        elif op == "UNDO":
            if undo:
                act = undo.pop()
                reverse(act)
                redo.append(act)
        elif op == "REDO" and redo:
            act = redo.pop()
            apply(act)
            undo.append(act)

    print("".join(text))
