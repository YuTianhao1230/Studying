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

## 面试应对

### 为什么链表通常拆成 ListNode 和 LinkedList 两个类？

回答思路：区分节点的数据模型与链表的容器行为，并说明刷题平台为什么常只提供节点类。

完整模板：

`ListNode` 只描述一个节点，保存 `val` 和指向下一节点的 `next`；`LinkedList` 管理整条链表，保存 `head`，并提供插入、删除和遍历等操作。这样拆分后，节点之间的连接关系与容器级操作职责清晰。算法题通常直接给出头节点，考查的是指针操作，因此往往只需要 `ListNode`，不必再封装 `LinkedList`。

### Python 中为什么用 list 实现栈、用 deque 实现队列？

回答思路：从操作位置和时间复杂度解释选择，明确指出 `list.pop(0)` 的代价。

完整模板：

栈只在尾部入栈和出栈，Python `list` 的 `append()` 和 `pop()` 平均都是 `O(1)`，可以直接使用。队列需要从头部出队，如果使用 `list.pop(0)`，后续元素都要前移，时间复杂度是 `O(n)`；`collections.deque` 的 `append()` 和 `popleft()` 都是 `O(1)`，因此更适合实现队列。

### 最小二叉树节点为什么只需要 val、left 和 right？

回答思路：说明二叉树算法通常通过根节点访问整棵树，父指针、树容器和其他字段应按题目需求再增加。

完整模板：

普通二叉树节点只要保存当前值 `val`，以及左右孩子引用 `left`、`right`，从根节点就能递归或迭代访问整棵树。父指针、节点高度或整棵树的容器类都不是通用必需信息，只有题目明确要求向上访问、维护平衡或封装增删操作时才添加。最小定义可以减少无关状态，也更符合刷题平台常见的函数签名。

### 手写数据结构类时如何处理空结构和接口边界？

回答思路：先约定空栈、空队列和空链表的行为，再保证每个操作的复杂度符合预期。

完整模板：

我会先明确接口和空结构约定，例如 `pop()`、`top()` 或 `front()` 在为空时是抛异常、返回哨兵值，还是由调用方先检查 `empty()`。然后检查边界：链表插入要单独处理空头节点，队列不能用 `list.pop(0)`，指针修改前要保存后继节点。最后说明复杂度，例如栈顶操作和 `deque` 两端操作为 `O(1)`，单链表按值查找或尾插在没有尾指针时为 `O(n)`。
