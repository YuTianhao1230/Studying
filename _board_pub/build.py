#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Studying 看板 - 内容构建脚本

作用：扫描上级目录（Studying/）下所有 .md 笔记，生成 assets/data.js 供看板读取。
用法：python build.py    （或双击同级目录下的「刷新数据.bat」）

约定：
  - 内容 = Studying/ 下的 .md 笔记（随便增删改）
  - 框架 = _board/ 下的 index.html / assets/*.js / assets/*.css（不用动）
  内容变了只需要重新运行本脚本，框架一行都不用改。
"""

import os
import re
import json
import shutil
import hashlib
import datetime
import sys

# ---------------------------------------------------------------- 路径配置

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # Studying/ 根目录
OUT_JS = os.path.join(HERE, 'assets', 'data.js')
INDEX_HTML = os.path.join(HERE, 'index.html')
ASSET_FILES = ['assets/app.js', 'assets/style.css', 'assets/md.js', 'assets/data.js']

EXCLUDE_DIRS = {'.git', '_board', '_board_pub', '_board_v2', 'node_modules', '.workbuddy', 'assets', '__pycache__'}
EXCLUDE_FILES = set()

# 放在内容根目录（不属于任何编号分类）的笔记，归入这个虚拟分类
ROOT_CAT = '00_索引'
CAT_NAME_MAP = {ROOT_CAT: '知识库索引'}

# ---------------------------------------------------------------- 排序：与 Windows 资源管理器保持一致

# 直接用 Python 的 sorted() 会得到「Unicode 码点序」：中文按汉字编码排，
# 结果和资源管理器（中文按拼音、数字按数值：2 < 10）完全对不上，用户会觉得"看板顺序和我文件夹不一样"。
# 所以 Windows 下走系统自带的 StrCmpLogicalW（资源管理器同款），其它平台回退到普通字符串序。
_natkey = None
try:
    import ctypes
    from ctypes import wintypes
    from functools import cmp_to_key

    _StrCmpLogicalW = ctypes.windll.shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
    _StrCmpLogicalW.restype = ctypes.c_int
    _natkey = cmp_to_key(lambda a, b: _StrCmpLogicalW(a, b))
except Exception:
    _natkey = None


def nat_sorted(seq, keyfn=lambda x: x):
    """按「文件夹里看到的顺序」排序（Windows 自然排序）；非 Windows 回退普通字符串序。"""
    if _natkey is not None:
        return sorted(seq, key=lambda x: _natkey(keyfn(x)))
    return sorted(seq, key=keyfn)


def nat_path_key(rel):
    """路径排序键：逐级比较，保证 'a/README.md' 与 'a/子目录/x.md' 像资源管理器那样混排。"""
    parts = rel.split('/')
    return tuple(_natkey(p) for p in parts) if _natkey is not None else tuple(parts)


# ---------------------------------------------------------------- 工具函数


def read_text(path):
    for enc in ('utf-8', 'utf-8-sig', 'gbk'):
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def strip_num_prefix(name):
    """01_机器学习基础 -> 机器学习基础"""
    return re.sub(r'^\d+[_.\-\s]*', '', name)


def title_from_body(body, filename):
    m = re.search(r'^#\s+(.+?)\s*$', body, re.M)
    if m:
        t = m.group(1).strip()
        t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
        if t.lower().endswith('.md'):
            t = t[:-3]          # "AGENT.md" -> "AGENT"
        return t
    return os.path.splitext(filename)[0]


CODE_BLOCK_RE = re.compile(r'```.*?```', re.S)
INLINE_CODE_RE = re.compile(r'`([^`]*)`')


def plain_text(md, limit=None):
    """把 markdown 压成纯文本，用于摘要和搜索索引"""
    t = CODE_BLOCK_RE.sub(' ', md)
    t = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', t)          # 图片
    t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)       # 链接保留文字
    t = INLINE_CODE_RE.sub(r'\1', t)
    t = re.sub(r'<[^>]+>', ' ', t)                       # html 标签
    t = re.sub(r'^\s{0,3}#{1,6}\s+', '', t, flags=re.M)  # 标题符号
    t = re.sub(r'^\s{0,3}[-*+]\s+', '', t, flags=re.M)   # 列表符号
    t = re.sub(r'^\s{0,3}>\s?', '', t, flags=re.M)       # 引用符号
    t = re.sub(r'^\s*\|[\s:\-|]+\|\s*$', '', t, flags=re.M)  # 表格分隔行
    t = t.replace('|', ' ')
    t = re.sub(r'^\s*---+\s*$', ' ', t, flags=re.M)
    t = re.sub(r'\*\*(.+?)\*\*', r'\1', t)
    t = re.sub(r'\*(.+?)\*', r'\1', t)
    t = re.sub(r'~~(.+?)~~', r'\1', t)
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\n{2,}', '\n', t).strip()
    if limit:
        t = t[:limit]
    return t


def count_words(md):
    """中文按字计，英文按词计，取一个阅读量级的估算值"""
    t = plain_text(md)
    cn = len(re.findall(r'[\u4e00-\u9fff]', t))
    en = len(re.findall(r'[A-Za-z]+', t))
    return cn + en


def make_summary(md, maxlen=110):
    """取正文第一段有意义的文字作为摘要"""
    lines = md.split('\n')
    buf = []
    in_code = False
    for ln in lines:
        s = ln.strip()
        if s.startswith('```'):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not s:
            if buf:
                break
            continue
        if s.startswith('#'):                      # 跳过标题
            if buf:
                break
            continue
        if re.match(r'^\|[\s:\-|]+\|?$', s):        # 表格分隔
            continue
        if re.match(r'^[-=]{3,}$', s):              # 分割线
            continue
        buf.append(s)
        if len(' '.join(buf)) > maxlen * 2:
            break
    text = plain_text(' '.join(buf))
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 12:
        # 兜底：正文开头是表格/列表等结构时，退化为取正文纯文本
        rest = re.sub(r'^#\s+.*$', '', md, count=1, flags=re.M)
        text = re.sub(r'\s+', ' ', plain_text(rest)).strip()

    if len(text) > maxlen:
        cut = text[:maxlen]
        # 尽量在标点处断开
        m = re.search(r'[。；;！!？?，,、]', cut[::-1])
        if m and len(cut) - m.start() > maxlen * 0.5:
            cut = cut[:len(cut) - m.start()]
        text = cut.rstrip('，,、；; ') + '…'
    return text


SECTION_KEYS = ['知识点解析', '面试应对', '直接回答', '面试官真正想听到什么', '面试常见追问']


def detect_sections(md):
    found = []
    for ln in md.split('\n'):
        m = re.match(r'^##\s+(.+?)\s*$', ln)
        if m:
            name = m.group(1).strip()
            if name in SECTION_KEYS and name not in found:
                found.append(name)
    return found


# ---------------------------------------------------------------- 本地图片

IMG_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.bmp'}
IMG_OUT = os.path.join(HERE, 'assets', 'img')
_img_map = {}                                   # 源相对路径 -> 输出文件名（同图多处引用只复制一次）
_img_stats = {'copied': 0, 'reused': 0, 'missing': 0}
_missing_imgs = []
IMG_RE = re.compile(r'!\[([^\]]*)\]\(\s*(<[^>]+>|[^)\s]+)\s*(?:"[^"]*")?\s*\)')


def rewrite_images(body, md_path):
    """把正文里的「本地相对路径图片」复制到 assets/img/，并改写成看板可访问路径。

    笔记里的 ![alt](assets/x.png) 是相对该 md 文件自己的，而看板跑在 _board 根目录，
    直接渲染必然 404，所以统一平铺复制成唯一文件名再改写。外链 / data: / 锚点原样保留。
    """
    md_dir = os.path.dirname(md_path)

    def repl(m):
        alt, src = m.group(1), m.group(2)
        s = src.strip().strip('<>').strip()
        if re.match(r'^(https?:)?//', s) or s.startswith('data:') or s.startswith('#'):
            return m.group(0)
        ext = os.path.splitext(s)[1].lower()
        if ext not in IMG_EXT:
            return m.group(0)
        src_path = os.path.normpath(os.path.join(md_dir, s))
        if not os.path.isfile(src_path):
            _img_stats['missing'] += 1
            _missing_imgs.append(os.path.relpath(src_path, ROOT).replace('\\', '/'))
            return m.group(0)
        key = os.path.relpath(src_path, ROOT).replace('\\', '/')
        if key in _img_map:
            _img_stats['reused'] += 1
            return '![%s](assets/img/%s)' % (alt, _img_map[key])
        out_name = hashlib.md5(key.encode('utf-8')).hexdigest()[:10] + ext
        out_path = os.path.join(IMG_OUT, out_name)
        need = True
        if os.path.isfile(out_path):
            need = os.path.getsize(out_path) != os.path.getsize(src_path)
        os.makedirs(IMG_OUT, exist_ok=True)
        if need:
            shutil.copy2(src_path, out_path)
            _img_stats['copied'] += 1
        else:
            _img_stats['reused'] += 1
        _img_map[key] = out_name
        return '![%s](assets/img/%s)' % (alt, out_name)

    return IMG_RE.sub(repl, body)


def collect_docs():
    docs = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = nat_sorted([d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith('.')])
        for fn in nat_sorted(filenames):
            if not fn.lower().endswith('.md'):
                continue
            if fn in EXCLUDE_FILES:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT).replace('\\', '/')
            parts = rel.split('/')
            cat = parts[0] if len(parts) > 1 else ROOT_CAT
            # sub 取「分类之下到文件之上」的全部子目录，按 "/".join。
            # 例：02_大模型/模型细节/里程碑模型/GPT.md -> "模型细节/里程碑模型"
            sub = '/'.join(parts[1:-1]) if len(parts) > 2 else ''
            body = read_text(full)
            body = rewrite_images(body, full)
            st = os.stat(full)
            mtime = datetime.datetime.fromtimestamp(st.st_mtime)
            docs.append({
                'id': rel,
                'title': title_from_body(body, fn),
                'cat': cat,
                'sub': sub,
                'name': os.path.splitext(fn)[0],
                'isReadme': fn.lower() == 'readme.md',
                'words': count_words(body),
                'mtime': mtime.strftime('%Y-%m-%d'),
                'ts': int(st.st_mtime),
                # 内容指纹：用于前端检测「这篇笔记自上次阅读后是否更新过」
                'hash': hashlib.md5(body.encode('utf-8')).hexdigest()[:12],
                'summary': make_summary(body),
                'sections': detect_sections(body),
                'body': body,
            })
    docs.sort(key=lambda d: nat_path_key(d['id']))
    return docs


def build_categories(docs):
    """分类树：cat -> sub -> (递归) sub。sub 字段本身是全路径（如 '关键帧检测/任务与数据治理'）。"""
    order = {}
    for d in docs:
        c = d['cat'] or '未分类'
        if c not in order:
            order[c] = {'name': CAT_NAME_MAP.get(c, strip_num_prefix(c)), 'count': 0, 'words': 0, 'nReadme': 0, 'subs': {}, 'order': len(order)}
        o = order[c]
        # 计数只算「知识点笔记」：README 是目录索引页，混进篇数会让分类卡数字和分类页列表对不上
        # （实测 57 篇的「大模型」点进去只有 47 条，差的那 10 篇就是 README）。
        if d['isReadme']:
            o['nReadme'] += 1
        else:
            o['count'] += 1
            o['words'] += d['words']
        s = d['sub']
        if not s:
            continue
        # 按层级逐层插入；用「全路径」作 dict key，确保同名不同级不会冲突
        levels = s.split('/')
        cur = o['subs']
        path = []
        for lv in levels:
            path.append(lv)
            key = '/'.join(path)
            if key not in cur:
                cur[key] = {'name': lv, 'count': 0, 'words': 0, 'nReadme': 0, 'subs': {}}
            # 只统计「直接放在本目录下的」文档：子目录里的文档归到子目录自己的节点。
            # 之前是逐层累加，导致 chip 显示「行测 40 篇」、点进去却只有 1 条。
            if d['sub'] == key:
                if d['isReadme']:
                    cur[key]['nReadme'] += 1
                else:
                    cur[key]['count'] += 1
                    cur[key]['words'] += d['words']
            cur = cur[key]['subs']

    def to_list(level):
        out = []
        for k, v in nat_sorted(level.items(), lambda kv: kv[1]['name']):
            entry = {'name': v['name'], 'count': v['count'], 'words': v['words'], 'nReadme': v['nReadme']}
            if v['subs']:
                entry['subs'] = to_list(v['subs'])
            out.append(entry)
        return out

    cats = []
    for cid, o in order.items():
        cats.append({
            'id': cid,
            'name': o['name'],
            'order': o['order'],
            'count': o['count'],
            'words': o['words'],
            'nReadme': o['nReadme'],
            'subs': to_list(o['subs']),
        })
    cats.sort(key=lambda c: _natkey(c['id']) if _natkey is not None else c['id'])
    for i, c in enumerate(cats):
        c['order'] = i
    return cats


def parse_routes(docs):
    """解析「学习路线总览.md」的复习路线：## 路线名 + 1. [标题](<路径>)：描述"""
    src = next((d for d in docs if d['id'] == '学习路线总览.md'), None)
    if not src:
        return []
    routes, cur = [], None
    for line in src['body'].split('\n'):
        m = re.match(r'^##\s+(.+?)\s*$', line)
        if m:
            cur = {'name': m.group(1).strip(), 'intro': '', 'items': []}
            routes.append(cur)
            continue
        if cur is None:
            continue
        mi = re.match(r'^\s*\d+\.\s+(.*)$', line)
        if not mi:
            # 路线下的第一段说明文字作为简介
            if line.strip() and not line.startswith('#') and not cur['intro'] and not cur['items']:
                cur['intro'] = line.strip()
            continue
        content = mi.group(1).strip()
        links = re.findall(r'\[([^\]]*)\]\(<?([^)>]+)>?\)', content)
        desc = re.sub(r'\[[^\]]*\]\(<?[^)>]+>?\)', '', content)
        desc = re.sub(r'^[\s：:，,、和]+|[\s，,、]+$', '', desc)
        items = []
        for text, path in links:
            pid = path.strip('<>').lstrip('./')
            if pid.startswith('/') or pid.startswith('http'):
                continue
            items.append({'text': text.strip(), 'doc': pid if pid in {d['id'] for d in docs} else None})
        if not items:
            items = [{'text': content, 'doc': None}]
        cur['items'].append({'desc': desc, 'links': items})
    return routes


def stamp_index():
    """给 index.html 里的 4 个框架资源引用打上内容指纹（?v=xxxxxxxxxx）。

    否则浏览器可能拿「旧的 app.js + 新的 data.js」这种混搭缓存，
    表现为侧栏子目录点击异常、新旧功能半生效等诡异现象。
    每次 build.py 都会重算指纹，内容一变 URL 就变，浏览器必然重新下载。
    """
    if not os.path.isfile(INDEX_HTML):
        return None
    h = hashlib.md5()
    for rel in ASSET_FILES:
        p = os.path.join(HERE, *rel.split('/'))
        if os.path.isfile(p):
            with open(p, 'rb') as f:
                h.update(f.read())
    ver = h.hexdigest()[:10]

    html = read_text(INDEX_HTML)
    for rel in ASSET_FILES:
        base = os.path.basename(rel)
        html = re.sub(r'assets/' + re.escape(base) + r'(\?v=[^"\']*)?',
                      'assets/' + base + '?v=' + ver, html)
    with open(INDEX_HTML, 'w', encoding='utf-8', newline='') as f:
        f.write(html)
    return ver


def main():
    if not os.path.isdir(ROOT):
        print('[x] 找不到内容根目录:', ROOT)
        return 1

    docs = collect_docs()
    cats = build_categories(docs)
    # 口径：total / words 只数「知识点笔记」，README 索引页单列（totalReadme），
    # 否则「57 篇」的分类点进去只有 47 条，用户会以为笔记丢了。
    notes = [d for d in docs if not d['isReadme']]
    total_words = sum(d['words'] for d in notes)

    data = {
        'meta': {
            'root': os.path.basename(ROOT),
            'generatedAt': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total': len(notes),
            'totalReadme': len(docs) - len(notes),
            'categories': len(cats),
            'words': total_words,
            'routes': parse_routes(docs),
        },
        'cats': cats,
        'docs': docs,
    }

    js = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    # 防止正文里出现 </script> 破坏页面
    js = js.replace('</', '<\\/')

    os.makedirs(os.path.dirname(OUT_JS), exist_ok=True)
    with open(OUT_JS, 'w', encoding='utf-8') as f:
        f.write('/* 本文件由 build.py 自动生成，请勿手动编辑。内容更新请重跑 build.py */\n')
        f.write('window.STUDY_DATA = ')
        f.write(js)
        f.write(';\n')

    size_kb = os.path.getsize(OUT_JS) / 1024
    ver = stamp_index()
    print('[√] 已生成 %s' % os.path.relpath(OUT_JS, ROOT).replace('\\', '/'))
    n_readme = len(docs) - len(notes)
    print('    笔记 %d 篇 / 分类 %d 个 / 约 %s 字 / %.0f KB' % (len(notes), len(cats), f'{total_words:,}', size_kb))
    if n_readme:
        print('    （另有 %d 篇 README 索引页，不计入篇数与字数）' % n_readme)
    if ver:
        print('    资源指纹 index.html -> ?v=%s（每次重建都会变，强制浏览器刷新缓存）' % ver)
    for c in cats:
        print('    - %-14s %2d 篇' % (c['name'], c['count']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
