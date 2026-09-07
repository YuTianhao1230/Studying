# SQL高频题

## 知识点解析

### 概述

本文整理 SQL 中的多表连接、分组聚合、Top N、去重、窗口函数、条件聚合、连续日期和 NULL 处理。

### 执行顺序

```text
FROM/JOIN -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY -> LIMIT
```

WHERE 过滤分组前的行，HAVING 过滤分组后的聚合结果。

### 高频题型

- join：多表关联，找匹配或不匹配记录。
- group by：分组统计数量、总和、平均值。
- having：筛选聚合结果。
- top N：整体 Top N 或分组 Top N。
- 去重：`distinct` 或窗口函数。
- 窗口函数：`row_number`、`rank`、`dense_rank`。

### 解题顺序

拿到 SQL 题先判断：

1. 输出的一行代表什么粒度，是用户、订单还是部门。
2. 数据来自哪些表，连接键是什么。
3. 先过滤哪些明细行。
4. 是否需要分组聚合。
5. 是否需要组内排序、累计或与前后行比较。
6. NULL、重复行和并列名次如何处理。

复杂 SQL 建议先用公共表表达式 CTE 分步骤构造，先保证每一层粒度正确，再考虑性能优化。

### 多表连接

找“有记录”的对象通常使用 `INNER JOIN`，找“没有记录”的对象通常使用 `LEFT JOIN ... IS NULL` 或 `NOT EXISTS`。

```sql
SELECT u.user_id
FROM users AS u
LEFT JOIN orders AS o
    ON u.user_id = o.user_id
WHERE o.order_id IS NULL;
```

一对多连接会放大行数。连接后再统计用户数时，必须判断是否需要 `COUNT(DISTINCT user_id)`，否则容易重复计数。

### 分组聚合

```sql
SELECT
    department_id,
    COUNT(*) AS employee_count,
    AVG(salary) AS average_salary
FROM employees
WHERE status = 'active'
GROUP BY department_id
HAVING COUNT(*) >= 5;
```

`WHERE` 过滤参与分组的明细行，`HAVING` 过滤分组后的聚合结果。SELECT 中没有聚合的字段通常必须出现在 `GROUP BY` 中。

### 窗口函数区别

- `row_number`：不管是否并列，连续编号。
- `rank`：并列同名次，后续名次跳号。
- `dense_rank`：并列同名次，后续名次不跳号。

### 分组 Top N

```sql
WITH ranked AS (
    SELECT
        employee_id,
        department_id,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department_id
            ORDER BY salary DESC
        ) AS row_num
    FROM employees
)
SELECT employee_id, department_id, salary
FROM ranked
WHERE row_num <= 3;
```

- 每组严格取 N 条：`ROW_NUMBER`。
- 并列第 N 名都保留且后续名次跳号：`RANK`。
- 并列名次保留且不跳号：`DENSE_RANK`。

### 去重保留一条

例如每个用户只保留最新记录：

```sql
WITH ranked AS (
    SELECT
        records.*,
        ROW_NUMBER() OVER (
            PARTITION BY user_id
            ORDER BY updated_at DESC, record_id DESC
        ) AS row_num
    FROM records
)
SELECT *
FROM ranked
WHERE row_num = 1;
```

排序条件必须能稳定决定唯一顺序，因此时间相同时再加入主键。

### 连续日期与留存

连续登录类问题常用：

```text
日期 - ROW_NUMBER()
```

连续日期减去连续编号后会得到相同分组键，再按用户和分组键聚合。留存题则先得到用户首次日期，再判断后续目标日期是否出现。

### 条件聚合

同一分组中统计多个条件：

```sql
SELECT
    user_id,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_count,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count
FROM tasks
GROUP BY user_id;
```

### NULL 易错点

- 判断 NULL 使用 `IS NULL`，不能使用 `= NULL`。
- `COUNT(*)` 统计行数，`COUNT(column)` 不统计 NULL。
- `NOT IN` 的子查询如果包含 NULL，结果可能不符合直觉，优先考虑 `NOT EXISTS`。
- 聚合函数通常忽略 NULL，但 `COUNT(*)` 除外。

### 常见考法与检查项

- 多表连接后是否出现一对多重复。
- 分组粒度是否与输出粒度一致。
- Top N 是否正确处理并列。
- 时间范围是否左闭右开，避免日期边界重复。
- 去重时保留哪一条，排序条件是否确定。
- NULL 是否影响比较、计数和反连接。

## 面试应对

### WHERE 和 HAVING 有什么区别？

回答思路：过滤时机、聚合函数。

回答模板：

WHERE 是分组前过滤行，作用在原始记录上，一般不能直接使用聚合函数；HAVING 是 GROUP BY 分组之后过滤聚合结果，可以使用 COUNT、SUM 这类聚合函数。简单说，过滤明细行用 WHERE，过滤分组统计结果用 HAVING。优化时应尽量把能提前过滤的条件放到 WHERE，减少后续分组数据量。

### 分组 Top N 怎么写？

回答思路：窗口函数分区排序。

回答模板：

分组 Top N 通常用窗口函数。先用 `row_number()` 或 `rank()` 按分组字段 `partition by`，再按指标 `order by` 排序，得到每组内的排名，最后在外层筛选排名小于等于 N 的记录。如果要严格取 N 条用 `row_number`，如果并列名次都保留可以用 `rank` 或 `dense_rank`。

### LEFT JOIN 后行数为什么变多？

回答思路：说明一对多连接的行扩张。

回答模板：

LEFT JOIN 会为左表记录保留所有匹配结果。如果右表同一个连接键有多条记录，左表的一行就会被展开成多行，因此总行数可能增加。排查时我会先检查连接键在两侧是否唯一，再确认输出需要明细粒度还是实体粒度；如果只需要判断存在性，可以使用 EXISTS，如果需要统计实体数则考虑先聚合右表或使用 DISTINCT。

### NOT IN 和 NOT EXISTS 有什么区别？

回答思路：重点说明 NULL 语义。

回答模板：

两者都可以表达反连接，但 `NOT IN` 的子查询一旦包含 NULL，三值逻辑可能让整个条件变成 UNKNOWN，导致结果为空或不符合预期。`NOT EXISTS` 按关联条件判断是否存在匹配行，NULL 处理通常更直观。因此面对可空字段时我更倾向于使用 `NOT EXISTS`，同时结合执行计划判断性能。
