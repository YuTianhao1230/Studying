# Shell文本处理

## 知识点解析

### 概述

Shell 文本处理常用于日志分析、批量处理和笔试命令题。核心工具包括管道、重定向、`grep`、`awk`、`sed`、`sort`、`uniq`、`wc`。

### 管道和重定向

管道 `|` 把前一个命令的输出作为后一个命令的输入。重定向 `>` 覆盖写文件，`>>` 追加写文件，`2>` 重定向错误输出。常见用法：

```bash
cat app.log | grep ERROR
grep ERROR app.log > error.log
grep ERROR app.log 2> err.txt
```

### grep

`grep` 用于按模式搜索文本。常用参数：

| 参数 | 作用 |
| --- | --- |
| `-n` | 显示行号 |
| `-i` | 忽略大小写 |
| `-v` | 反向匹配 |
| `-r` | 递归搜索目录 |
| `-E` | 使用扩展正则 |

### awk

`awk` 适合按列处理文本。默认按空白字符分隔，`$1` 表示第一列，`$NF` 表示最后一列。

```bash
awk '{print $1, $NF}' access.log
awk -F',' '{sum += $3} END {print sum}' data.csv
```

### sed

`sed` 适合流式替换和按行处理。

```bash
sed 's/old/new/g' file.txt
sed -n '10,20p' file.txt
```

### 排序和统计

常见组合：

```bash
grep ERROR app.log | wc -l
awk '{print $1}' access.log | sort | uniq -c | sort -nr
```

第一条统计错误行数，第二条统计访问 IP 次数并按次数倒序。

## 笔试常考

- `grep`、`awk`、`sed` 的用途区别。
- 管道和重定向符号含义。
- 统计文件行数、关键词出现次数。
- 提取日志某一列并排序去重。
- `sort | uniq -c` 的组合用法。

## 面试应对

### grep、awk、sed 怎么区分？

回答思路：一句话讲清各自职责。

回答模板：

`grep` 主要用于查找包含某个模式的行，适合过滤；`awk` 主要用于按列处理和统计，适合结构化文本；`sed` 主要用于按行替换、删除、打印，适合流式编辑。实际排查日志时经常把它们用管道组合起来，比如先用 grep 过滤错误日志，再用 awk 提取字段，最后 sort 和 uniq 统计频次。
