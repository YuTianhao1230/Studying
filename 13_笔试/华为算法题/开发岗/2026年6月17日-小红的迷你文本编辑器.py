import sys

def solve():
    input = sys.stdin.buffer.readline
    operation_count = int(input())
    text = []
    undo_stack = []
    redo_stack = []

    def apply(action):
        kind, content = action
        if kind == "append":
            text.extend(content)
        else:
            text.pop()

    def reverse(action):
        kind, content = action
        if kind == "append":
            del text[-len(content):]
        else:
            text.append(content)

    for _ in range(operation_count):
        operation = input().decode().rstrip("\n")
        if operation.startswith("APPEND "):
            action = ("append", operation[7:])
            apply(action)
            undo_stack.append(action)
            redo_stack.clear()
        elif operation == "POP":
            if text:
                action = ("pop", text[-1])
                apply(action)
                undo_stack.append(action)
                redo_stack.clear()
        elif operation == "UNDO":
            if undo_stack:
                action = undo_stack.pop()
                reverse(action)
                redo_stack.append(action)
        elif operation == "REDO" and redo_stack:
            action = redo_stack.pop()
            apply(action)
            undo_stack.append(action)

    print("".join(text))

if __name__ == "__main__":
    solve()
