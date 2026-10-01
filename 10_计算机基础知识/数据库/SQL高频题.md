# SQL高频题

## 知识点解析

### 概述

本文整理 SQL 中的多表连接、分组聚合、Top N、去重、窗口函数、条件聚合、连续日期和 NULL 处理。除特别说明外，下面的可执行示例使用 PostgreSQL 14+；示例表名和列名在代码前说明。

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

`LAG(col, n)` 读取窗口排序后当前行之前第 n 行的值，`LEAD(col, n)` 读取之后第 n 行的值，常用于计算环比、相邻事件间隔和状态变化。窗口的 `PARTITION BY` 决定分区，`ORDER BY` 决定分区内顺序。window frame 主要限定窗口聚合以及 `FIRST_VALUE`、`LAST_VALUE`、`NTH_VALUE` 等函数使用的行范围，不限制 `LAG`、`LEAD` 按整个 partition 顺序计算偏移的语义。

例如 `payments(user_id, payment_id, paid_at, amount)` 的用户累计支付额：

```sql
SELECT
    user_id,
    payment_id,
    paid_at,
    amount,
    SUM(amount) OVER (
        PARTITION BY user_id
        ORDER BY paid_at, payment_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS cumulative_amount,
    LAG(amount) OVER (
        PARTITION BY user_id
        ORDER BY paid_at, payment_id
    ) AS previous_amount,
    LEAD(amount) OVER (
        PARTITION BY user_id
        ORDER BY paid_at, payment_id
    ) AS next_amount
FROM payments;
```

累计窗口应明确写出 `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`。仅写 `ORDER BY` 时，数据库默认 frame 可能是 `RANGE`；排序键并列时，`RANGE` 会把同一 peer group 一起纳入，结果可能不是“逐物理行累计”。次级键 `payment_id` 也让顺序稳定。这里的 frame 约束 `SUM` 的累计范围，不会把 `LAG`、`LEAD` 的偏移查找限制在该 frame 内。

### 分组 Top N

```sql
WITH ranked AS (
    SELECT
        employee_id,
        department_id,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department_id
            ORDER BY salary DESC, employee_id
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

因此“每组前 3 条”和“每组前三名”不是同一问题：前者通常固定最多 3 行，后者若第三名并列可能返回超过 3 行。选函数前必须确认并列规则。

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

`DISTINCT` 只能删除选定列完全相同的重复行，不能表达“每个用户保留最新一行”。这类带保留规则的去重应先用 `ROW_NUMBER()` 按业务键分区，再按时间和唯一键降序排序。若要执行物理删除，可先用相同 CTE 查出 `row_num > 1` 的主键，核对后再删除，避免边查边删时规则不稳定。

### 连续日期与留存

以下示例假设 PostgreSQL 表 `login_events(user_id, login_at)`，其中 `login_at` 为 `date` 或 `timestamp`。先按“用户 + 自然日”去重，否则同一天多次登录会打断编号。连续日期减去连续编号后得到相同分组键，再按用户和分组键聚合：

```sql
WITH login_days AS (
    SELECT DISTINCT user_id, CAST(login_at AS date) AS login_date
    FROM login_events
),
numbered AS (
    SELECT
        user_id,
        login_date,
        login_date - CAST(
            ROW_NUMBER() OVER (
                PARTITION BY user_id
                ORDER BY login_date
            ) AS integer
        ) AS streak_group
    FROM login_days
)
SELECT
    user_id,
    MIN(login_date) AS streak_start,
    MAX(login_date) AS streak_end,
    COUNT(*) AS consecutive_days
FROM numbered
GROUP BY user_id, streak_group
HAVING COUNT(*) >= 3
ORDER BY user_id, streak_start;
```

留存题要先固定 cohort 和目标时间窗。下面计算“首次登录次日仍登录”的 D1 留存；同一用户同一天多条事件只计一次：

```sql
WITH login_days AS (
    SELECT DISTINCT user_id, CAST(login_at AS date) AS login_date
    FROM login_events
),
first_login AS (
    SELECT user_id, MIN(login_date) AS cohort_date
    FROM login_days
    GROUP BY user_id
)
SELECT
    f.cohort_date,
    COUNT(*) AS cohort_users,
    COUNT(DISTINCT d.user_id) AS retained_users_d1,
    ROUND(
        100.0 * COUNT(DISTINCT d.user_id) / NULLIF(COUNT(*), 0),
        2
    ) AS retention_rate_d1_pct
FROM first_login AS f
LEFT JOIN login_days AS d
    ON d.user_id = f.user_id
   AND d.login_date = f.cohort_date + 1
GROUP BY f.cohort_date
ORDER BY f.cohort_date;
```

若口径是“第 1 天及以后回来”或“24 小时滚动窗口”，连接条件会不同，必须先确认自然日/小时、时区、首日是否记为 D0 以及分母是否按注册还是首次活跃用户定义。

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

例如要找没有订单的用户，若 `orders.user_id` 可能为 `NULL`，以下 `NOT EXISTS` 写法不会被无关的空值污染：

```sql
SELECT u.user_id
FROM users AS u
WHERE NOT EXISTS (
    SELECT 1
    FROM orders AS o
    WHERE o.user_id = u.user_id
);
```

`u.user_id NOT IN (SELECT o.user_id FROM orders o)` 在子查询结果含 `NULL` 时，对未匹配用户的判断会成为 `UNKNOWN`，`WHERE` 不会保留这些行。只有能证明子查询列非空，或在子查询中显式排除 `NULL` 时，`NOT IN` 才不会遇到这个语义问题。

### 常见考法与检查项

- 多表连接后是否出现一对多重复。
- 分组粒度是否与输出粒度一致。
- Top N 是否正确处理并列。
- 时间范围是否左闭右开，避免日期边界重复。
- 去重时保留哪一条，排序条件是否确定。
- NULL 是否影响比较、计数和反连接。

## 笔试常考

- 补全分组 Top N：先用窗口函数按组排名，再在外层过滤；根据“固定 N 行”或“并列名次全保留”选择 `ROW_NUMBER`、`RANK`、`DENSE_RANK`。
- 推导排名输出：值为 `100, 100, 90` 时，`ROW_NUMBER` 为 `1,2,3`，`RANK` 为 `1,1,3`，`DENSE_RANK` 为 `1,1,2`。
- 补全连续日期 SQL：先按用户和日期去重，再使用“日期减行号”分岛，最后按用户和岛分组。
- 判断累计窗口输出：明确 `PARTITION BY`、稳定排序键和 `ROWS` frame，注意默认 `RANGE` 对并列排序值的影响；frame 主要影响窗口聚合及 `FIRST_VALUE`、`LAST_VALUE` 等函数。
- 使用 `LAG`/`LEAD` 计算相邻记录差值时，先确定分区、顺序，以及首尾行返回 `NULL` 或默认值的处理方式；其偏移按 partition 顺序计算，不受 frame 边界限制。
- 最新行去重要用 `ROW_NUMBER` 表达保留规则，并用唯一键打破时间并列；`DISTINCT` 不能选择“最新”记录。
- 留存题先定义 cohort、D0/D1、自然日或滚动窗口和分母，再按用户日期去重，避免重复事件放大人数。
- 子查询可能含 `NULL` 时，推导 `NOT IN` 的三值逻辑；反连接通常用关联的 `NOT EXISTS` 更直接。

## 面试应对

### 排名和分组 Top N 怎么写？

回答思路：窗口函数分区排序，并先确认并列时返回多少行。

回答模板：

分组 Top N 通常先在 CTE 中用窗口函数按分组字段 `PARTITION BY`、按指标 `ORDER BY`，再在外层筛选排名小于等于 N。严格每组最多 N 行用 `ROW_NUMBER`，并补充唯一键保证顺序稳定；并列第 N 名都保留则用 `RANK` 或 `DENSE_RANK`，两者区别是后续名次是否跳号。

### 连续登录天数怎么计算？

回答思路：按日去重、连续日期分岛、分组统计。

回答模板：

我会先把登录时间转换到统一业务时区并按“用户 + 日期”去重，然后按用户和日期排序生成 `ROW_NUMBER`。连续日期减去连续行号会得到相同的分组键，再按用户和该键聚合，就能得到每段连续登录的开始日、结束日和天数。最后用 `HAVING` 过滤达到要求的连续段。

### 每个用户如何只保留最新一条记录？

回答思路：窗口分区、降序、稳定打破并列。

回答模板：

我会用 `ROW_NUMBER()` 按用户分区，按更新时间降序排序，并在时间相同时再按主键降序，最后保留 `row_num = 1`。这样保留规则确定且可复现。`DISTINCT` 只能去掉整行相同的数据，不能表达“保留最新”，所以不适合这个需求；执行物理删除前还会先查询待删主键并核对。

### D1 留存率怎么计算？

回答思路：先定义 cohort 和口径，再去重并做条件关联。

回答模板：

我会先确认 D0 是注册日还是首次活跃日，D1 是次自然日还是 24 小时窗口，并统一时区。然后按用户和日期去重，求每个用户的 cohort 日期，再左连接该用户 cohort 加一天的活跃记录。分母是 cohort 去重用户数，分子是 D1 仍活跃的去重用户数。使用左连接可以保留未留存用户，最后按 cohort 日期分组计算比例。

### 窗口函数相比自连接有什么优势？

回答思路：从表达力、行粒度、扫描成本和适用边界回答。

回答模板：

窗口函数能在保留明细行的同时完成排名、累计和前后行比较，通常比自连接更直接。例如环比可以用 `LAG`，累计值可以用带 `ROWS` frame 的 `SUM OVER`，避免为了找前一行或历史集合多次连接同一张表。frame 主要约束窗口聚合以及 `FIRST_VALUE`、`LAST_VALUE` 等函数的取值范围，`LAG`、`LEAD` 的偏移仍按整个 partition 的排序计算。窗口函数仍可能产生大分区排序，选择时要结合索引、数据量和执行计划。
