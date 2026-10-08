/* md.js —— 零依赖 Markdown 渲染器
 * 支持：标题 / 代码块(带复制) / 表格 / 列表 / 引用 / 分割线 / 图片 / 链接
 *       行内：加粗 斜体 删除线 行内代码 公式
 * 站内 .md 相对链接会自动转成看板内部跳转
 */
(function (global) {
  'use strict';

  var KEYWORDS = {
    python: 'def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|try|except|finally|with|lambda|yield|pass|break|continue|raise|assert|global|nonlocal|self|None|True|False|print|range|len|int|str|float|list|dict|set|tuple|is|async|await',
    bash: 'if|then|else|fi|for|do|done|while|case|esac|function|echo|export|cd|sudo|apt|pip|python|git|docker|conda|ls|mkdir|rm|cp|mv|cat|grep|sed|awk|curl|wget|chmod|source',
    json: 'true|false|null',
    javascript: 'const|let|var|function|return|if|else|for|while|class|new|await|async|import|from|export|default|try|catch|finally|throw|typeof|null|undefined|true|false|this|switch|case|break|continue|extends|super|of|in',
    java: 'public|private|protected|class|interface|extends|implements|static|final|void|int|long|double|float|boolean|char|String|new|return|if|else|for|while|try|catch|throw|throws|import|package|this|super|null|true|false|abstract|enum',
    cpp: 'int|long|double|float|char|bool|void|return|if|else|for|while|class|struct|public|private|protected|const|static|new|delete|template|typename|namespace|using|include|define|nullptr|true|false|auto|size_t|std|switch|case|break|continue|enum|virtual|inline|extern|unsigned|short|this|try|catch|throw|operator',
    c: 'int|long|double|float|char|void|return|if|else|for|while|struct|union|enum|const|static|sizeof|include|define|typedef|NULL|switch|case|break|continue|unsigned|short|this|do|goto|malloc|free|printf',
    yaml: 'true|false|null|on|off|yes|no',
    sql: 'select|from|where|group|by|order|having|join|left|right|inner|outer|on|insert|into|values|update|set|delete|create|table|as|and|or|not|null|count|sum|avg|max|min|distinct|limit|with|case|when|then|else|end',
    go: 'func|package|import|var|const|type|struct|interface|map|chan|go|defer|return|if|else|for|range|switch|case|default|break|continue|nil|true|false|string|int|error|make|new|len|cap|append'
  };

  var ALIAS = { py: 'python', sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash', js: 'javascript', 'c++': 'cpp', 'g++': 'cpp', yml: 'yaml', golang: 'go' };

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  function slug(text) {
    return 'h-' + String(text).toLowerCase()
      .replace(/[`*_~\[\]()#]/g, '')
      .replace(/[^\w\u4e00-\u9fff]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'h';
  }

  /* ---------------- 代码高亮（极简，保守匹配，出错也不影响阅读） ---------------- */
  function highlight(code, lang) {
    var esc = escapeHtml(code);
    var kw = KEYWORDS[ALIAS[lang] || lang];
    var parts = [];
    var re;
    if (ALIAS[lang] || lang === 'python' || lang === 'yaml' || lang === 'bash' || lang === 'sh') {
      re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|\b(\d+\.?\d*)\b/;
    } else {
      re = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)|("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\b(\d+\.?\d*)\b/;
    }
    var pattern = new RegExp(re.source + (kw ? '|\\b(' + kw + ')\\b' : ''), 'g');
    var last = 0, m;
    while ((m = pattern.exec(esc)) !== null) {
      if (m.index > last) parts.push(esc.slice(last, m.index));
      if (m[1]) parts.push('<span class="tk-cmt">' + m[1] + '</span>');
      else if (m[2]) parts.push('<span class="tk-str">' + m[2] + '</span>');
      else if (m[3]) parts.push('<span class="tk-num">' + m[3] + '</span>');
      else if (m[4]) parts.push('<span class="tk-kw">' + m[4] + '</span>');
      last = m.index + m[0].length;
      if (m[0] === '') pattern.lastIndex++;
    }
    parts.push(esc.slice(last));
    return parts.join('');
  }

  /* ---------------- 行内解析 ---------------- */
  function inline(text, ctx) {
    var codeStash = [];
    var linkStash = [];
    var PH_CODE = '\u0000C', PH_LINK = '\u0000L';

    // 1. 行内代码保护（用占位符 C+idx+C），原文保留避免被 escapeHtml 干扰
    text = String(text).replace(/```([^`\n]+)```/g, function (_, c) {
      codeStash.push('<code class="ic">' + escapeHtml(c.trim()) + '</code>');
      return PH_CODE + (codeStash.length - 1) + 'C\u0000';
    });
    text = text.replace(/`([^`]+)`/g, function (_, c) {
      codeStash.push('<code class="ic">' + escapeHtml(c) + '</code>');
      return PH_CODE + (codeStash.length - 1) + 'C\u0000';
    });

    // 2. 链接保护（用占位符 L+idx+L），在 escapeHtml 之前抓出：原 href 不会含 &lt;&gt;，
 //    且允许 label/href 中有空格、尖括号包裹、URL 片段等。
    text = text.replace(/\[([^\]]*)\]\(([^)]+)\)/g, function (_, label, href) {
      var inner = resolveLink(href, ctx);
      linkStash.push({ label: label, href: href, inner: inner });
      return PH_LINK + (linkStash.length - 1) + 'L\u0000';
    });

    // 3. 转义（链接与代码已被占位保护，括号/尖括号不再干扰字符类）
    text = escapeHtml(text);

    // 4. 图片（极少用，保持原字符类）
    text = text.replace(/!\[([^\]]*)\]\(([^)]+?)(?:\s+&quot;[^)]*&quot;)?\)/g, function (_, alt, src) {
      return '<img class="md-img" src="' + src + '" alt="' + alt + '" loading="lazy" onerror="this.classList.add(\'img-fail\')">';
    });

    // 5. 加粗 / 斜体 / 删除线，清理源文件中未配对的残留标记
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*\*/g, '');
    text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
    text = text.replace(/~~([^~]+)~~/g, '<del>$1</del>');
    text = text.replace(/```/g, '');

    // 6. 公式
    text = text.replace(/\$\$([\s\S]+?)\$\$/g, '<span class="math math-block">$$$1$$</span>');
    text = text.replace(/(^|[^\\$])\$([^$\n]+?)\$/g, '$1<span class="math">$$$2$$</span>');

    // 7. 还原链接（此阶段 label 已被 escapeHtml 转义，需要再 escape 一次保证安全）
    text = text.replace(/\u0000L(\d+)L\u0000/g, function (_, i) {
      var lnk = linkStash[+i];
      var label = escapeHtml(lnk.label);
      if (lnk.inner) return '<a class="md-link internal" href="#/doc/' + encodeURI(lnk.inner) + '">' + label + '</a>';
      var h = lnk.href;
      if (/^(https?:)?\/\//.test(h) || h.indexOf('#') === 0 || h.indexOf('mailto:') === 0) {
        return '<a class="md-link" href="' + escapeHtml(h) + '" target="_blank" rel="noopener">' + label + '</a>';
      }
      return '<a class="md-link broken" title="链接目标不存在">' + label + '</a>';
    });

    // 8. 还原行内代码
    text = text.replace(/\u0000C(\d+)C\u0000/g, function (_, i) { return codeStash[+i]; });
    return text;
  }

  /* 把笔记里的相对 md 链接解析成看板内 doc id（接收原始 href，未 escape） */
  function resolveLink(href, ctx) {
    if (/^(https?:)?\/\//.test(href) || href.indexOf('#') === 0) return null;
    var clean = href.replace(/^<|>$/g, '').trim();
    if (!/\.md($|\?)/i.test(clean)) return null;
    if (!ctx || !ctx.docId) return null;
    var base = ctx.docId.split('/').slice(0, -1);
    var segs = clean.split('/');
    for (var i = 0; i < segs.length; i++) {
      if (segs[i] === '.' || segs[i] === '') continue;
      if (segs[i] === '..') base.pop();
      else base.push(segs[i]);
    }
    var id = base.join('/');
    return ctx.exists && ctx.exists[id] ? id : null;
  }

  /* 预处理：修复源文件里常见的围栏书写瑕疵
   * 例："### SFT 数据长什么样```json"  ->  "### SFT 数据长什么样" + "```json"
   *     "正文```python"               ->  "正文" + "```python"
   */
  function fixFences(md) {
    var lines = String(md).replace(/\r\n?/g, '\n').split('\n');
    var out = [];
    for (var i = 0; i < lines.length; i++) {
      var m = lines[i].match(/^(.*\S)\s*(```|~~~)\s*([A-Za-z0-9+#_-]*)\s*$/);
      if (m && !/^\s*(```|~~~)/.test(lines[i])) {
        out.push(m[1]);
        out.push(m[2] + m[3]);
      } else {
        out.push(lines[i]);
      }
    }
    return out.join('\n');
  }

  /* ---------------- 块级解析 ---------------- */
  function render(md, ctx) {
    ctx = ctx || {};
    var lines = fixFences(md).split('\n');
    var out = [];
    var i = 0;
    var n = lines.length;
    var guard = 0;               // 保险丝：任何情况下都不允许死循环
    var LIMIT = n * 4 + 2000;

    while (i < n) {
      if (++guard > LIMIT) { out.push('<p class="md-p">（内容过长或格式异常，已截断）</p>'); break; }
      var line = lines[i];

      // 空行
      if (!line.trim()) { i++; continue; }

      // 代码块
      var fence = line.match(/^\s*(```|~~~)\s*([A-Za-z0-9+#_-]*)\s*$/);
      if (fence) {
        var lang = (fence[2] || '').toLowerCase();
        var buf = [];
        i++;
        while (i < n && !new RegExp('^\\s*' + (fence[1] === '```' ? '```' : '~~~') + '\\s*$').test(lines[i])) {
          buf.push(lines[i]); i++;
        }
        i++; // 跳过结束围栏
        var code = buf.join('\n');
        var label = lang === 'mermaid' ? 'Mermaid 图' : (lang || 'text');
        out.push(
          '<div class="code-block">' +
          '<div class="code-head"><span class="code-lang">' + escapeHtml(label) + '</span>' +
          '<button class="code-copy" type="button" data-copy>复制</button></div>' +
          '<pre class="code-pre"><code data-raw="' + escapeHtml(code).replace(/"/g, '&quot;') + '">' +
          (lang === 'text' || !lang ? escapeHtml(code) : highlight(code, lang)) +
          '</code></pre></div>'
        );
        continue;
      }

      // 分割线
      if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) { out.push('<hr>'); i++; continue; }

      // 标题
      var h = line.match(/^(#{1,6})\s+(.*?)\s*$/);
      if (h) {
        var lv = h[1].length;
        var txt = h[2].replace(/#+\s*$/, '');
        var id = slug(txt);
        out.push('<h' + lv + ' id="' + id + '" class="md-h md-h' + lv + '">' + inline(txt, ctx) + '</h' + lv + '>');
        i++; continue;
      }

      // 表格
      if (/^\s*\|/.test(line) && i + 1 < n && /^\s*\|[\s:\-|]+\|?\s*$/.test(lines[i + 1])) {
        var rows = [];
        while (i < n && /^\s*\|/.test(lines[i])) { rows.push(lines[i]); i++; }
        var cells = function (r) {
          var s = r.trim();
          s = s.replace(/^\|/, '').replace(/\|$/, '');
          return s.split('|').map(function (c) { return c.trim(); });
        };
        var head = cells(rows[0]);
        var html = '<div class="table-wrap"><table class="md-table"><thead><tr>';
        head.forEach(function (c) { html += '<th>' + inline(c, ctx) + '</th>'; });
        html += '</tr></thead><tbody>';
        for (var r = 2; r < rows.length; r++) {
          var cs = cells(rows[r]);
          html += '<tr>';
          for (var c = 0; c < head.length; c++) html += '<td>' + inline(cs[c] || '', ctx) + '</td>';
          html += '</tr>';
        }
        html += '</tbody></table></div>';
        out.push(html);
        continue;
      }

      // 引用
      if (/^\s*>/.test(line)) {
        var qb = [];
        while (i < n && /^\s*>/.test(lines[i])) { qb.push(lines[i].replace(/^\s*>\s?/, '')); i++; }
        out.push('<blockquote class="md-quote">' + block(qb.join('\n'), ctx) + '</blockquote>');
        continue;
      }

      // 列表（含嵌套）
      if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
        var lb = [];
        var baseIndent = line.match(/^\s*/)[0].length;
        while (i < n) {
          var ln = lines[i];
          if (/^\s*([-*+]|\d+[.)])\s+/.test(ln)) { lb.push(ln); i++; continue; }
          if (/^\s{2,}\S/.test(ln) && ln.match(/^\s*/)[0].length > baseIndent && lb.length) {
            lb.push(ln); i++; continue;   // 续行 / 子项
          }
          if (/^\s*$/.test(ln) && i + 1 < n && /^\s*([-*+]|\d+[.)])\s+|\s{2,}\S/.test(lines[i + 1])) {
            lb.push(''); i++; continue;
          }
          break;
        }
        out.push(parseList(lb, ctx));
        continue;
      }

      // HTML 注释
      if (/^\s*<!--/.test(line)) {
        while (i < n && !/-->/.test(lines[i])) i++;
        i++; continue;
      }

      // 普通段落（连续非空行合并）
      var pb = [];
      var pStart = i;
      while (i < n) {
        var pl = lines[i];
        if (!pl.trim()) break;
        // 注意：以 ``` 开头但不成对围栏的行（如缩进的行内代码 ```xxx```）按普通文本处理
        var isLoneFence = /^\s*(```|~~~)/.test(pl) && !/^\s*(```|~~~)\s*[A-Za-z0-9+#_-]*\s*$/.test(pl);
        if ((/^\s*(```|~~~)/.test(pl) && !isLoneFence) || /^#{1,6}\s/.test(pl) || /^\s*>/.test(pl) ||
            /^\s*([-*+]|\d+[.)])\s+/.test(pl) || /^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(pl) ||
            /^\s*\|/.test(pl)) break;
        pb.push(pl); i++;
      }
      if (i === pStart) { pb.push(lines[i]); i++; }   // 强制推进，防止单行卡死
      out.push('<p class="md-p">' + inline(pb.join('\n'), ctx).replace(/\n/g, '<br>') + '</p>');
    }

    return out.join('\n');
  }

  function parseList(lines, ctx) {
    var html = '';
    var i = 0;
    var n = lines.length;
    var guard = 0, LIMIT = n * 4 + 2000;
    function indent(l) { return l.match(/^\s*/)[0].length; }
    function ordered(l) { return /^\s*\d+[.)]\s/.test(l); }

    while (i < n) {
      if (++guard > LIMIT) break;
      if (!lines[i].trim()) { i++; continue; }
      var isOl = ordered(lines[i]);
      var tag = isOl ? 'ol' : 'ul';
      var base = indent(lines[i]);
      var items = [];
      while (i < n) {
        var l = lines[i];
        if (!l.trim()) {
          if (i + 1 < n && (indent(lines[i + 1]) > base || ordered(lines[i + 1]) === isOl && indent(lines[i + 1]) >= base)) { i++; continue; }
          break;
        }
        if (indent(l) < base) break;
        if (ordered(l) !== isOl && indent(l) === base) break;
        if (indent(l) === base && /^\s*([-*+]|\d+[.)])\s+/.test(l)) {
          items.push([l.replace(/^\s*([-*+]|\d+[.)])\s+/, '')]);
          i++;
        } else if (items.length) {
          items[items.length - 1].push(l);
          i++;
        } else { i++; }
      }
      var body = '';
      items.forEach(function (it) {
        var first = it[0];
        var rest = it.slice(1);
        var sub = rest.filter(function (x) { return x.trim(); });
        var subHtml = '';
        if (sub.length) {
          var minInd = Math.min.apply(null, sub.map(indent));
          subHtml = parseList(sub.map(function (x) { return x.slice(minInd); }), ctx);
        }
        var hasBlock = /\n\s*```/.test(first);
        body += '<li>' + inline(first, ctx) + subHtml + '</li>';
        if (hasBlock) { /* 列表里的代码块按行内处理，够用 */ }
      });
      html += '<' + tag + ' class="md-list">' + body + '</' + tag + '>';
    }
    return html;
  }

  function block(md, ctx) { return render(md, ctx); }

  /* 提取目录大纲 */
  function toc(md) {
    var res = [];
    var inCode = false;
    String(md).replace(/\r\n?/g, '\n').split('\n').forEach(function (l) {
      if (/^\s*(```|~~~)/.test(l)) { inCode = !inCode; return; }
      if (inCode) return;
      var m = l.match(/^(#{2,4})\s+(.*?)\s*$/);
      if (m) res.push({ level: m[1].length, text: m[2].replace(/[*`~]/g, ''), id: slug(m[2]) });
    });
    return res;
  }

  global.MD = { render: render, toc: toc, escapeHtml: escapeHtml, slug: slug };
})(window);
