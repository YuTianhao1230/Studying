# 数据结构 class 最小示例

## 链表

链表通常拆成两个 class：`ListNode` 表示节点，`LinkedList` 表示链表本身。

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, val):
        node = ListNode(val)
        if self.head is None:
            self.head = node
            return

        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node

    def to_list(self):
        ans = []
        cur = self.head
        while cur:
            ans.append(cur.val)
            cur = cur.next
        return ans


ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
print(ll.to_list())  # [1, 2, 3]
```

面试刷题里也经常只需要节点类：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

## 栈

Python 里最小实现可以直接封装 `list`。

```python
class Stack:
    def __init__(self):
        self.data = []

    def push(self, x):
        self.data.append(x)

    def pop(self):
        return self.data.pop()

    def top(self):
        return self.data[-1]

    def empty(self):
        return len(self.data) == 0
```

## 队列

队列用 `collections.deque`，避免 `list.pop(0)` 的 O(n) 移动成本。

```python
from collections import deque


class Queue:
    def __init__(self):
        self.data = deque()

    def push(self, x):
        self.data.append(x)

    def pop(self):
        return self.data.popleft()

    def front(self):
        return self.data[0]

    def empty(self):
        return len(self.data) == 0
```

## 二叉树节点

树题里通常只需要定义节点，不一定要单独写 `BinaryTree`。

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

## 最小记忆

- 节点类负责保存值和指针：`val`、`next`、`left`、`right`。
- 容器类负责管理整体结构：`head`、`push`、`pop`、`append`。
- 刷题时如果平台已经给了 `ListNode` 或 `TreeNode`，不要重复定义，直接按题目给定结构写函数。
