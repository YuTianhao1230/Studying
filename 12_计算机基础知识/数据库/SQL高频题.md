# SQL高频题

## 知识点解析

### 概述

SQL 是央国企、银行科技岗和数据/后端岗位笔试的高性价比模块。重点掌握 join、group by、having、Top N、去重和窗口函数。

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

### 窗口函数区别

- `row_number`：不管是否并列，连续编号。
- `rank`：并列同名次，后续名次跳号。
- `dense_rank`：并列同名次，后续名次不跳号。

## 面试应对

### WHERE 和 HAVING 有什么区别？

回答思路：过滤时机、聚合函数。

回答模板：

WHERE 是分组前过滤行，作用在原始记录上，一般不能直接使用聚合函数；HAVING 是 GROUP BY 分组之后过滤聚合结果，可以使用 COUNT、SUM 这类聚合函数。简单说，过滤明细行用 WHERE，过滤分组统计结果用 HAVING。优化时应尽量把能提前过滤的条件放到 WHERE，减少后续分组数据量。

### 分组 Top N 怎么写？

回答思路：窗口函数分区排序。

回答模板：

分组 Top N 通常用窗口函数。先用 `row_number()` 或 `rank()` 按分组字段 `partition by`，再按指标 `order by` 排序，得到每组内的排名，最后在外层筛选排名小于等于 N 的记录。如果要严格取 N 条用 `row_number`，如果并列名次都保留可以用 `rank` 或 `dense_rank`。
