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

  /* ---------------- 公式（KaTeX） ----------------
     笔记里有 160+ 篇、2300+ 处 LaTeX（含 aligned/bmatrix/cases 环境与 \text{中文}）。
     统一走 KaTeX 渲染；**任何失败都回退成原始文本**，绝不把红色报错块丢给用户：
       · 解析失败（典型：正文里的 awk '$9 >= 500 {..}' 被 $..$ 规则误判成公式）
       · KaTeX 没加载上（离线 / 资源缺失）
     displayMode 由调用方指定：true = 块级公式（$$..$$，居中独立成行）。 */
  function renderMath(tex, display) {
    var src = String(tex).replace(/^\s+|\s+$/g, '');
    var katex = (typeof window !== 'undefined') ? window.katex : null;
    if (katex && typeof katex.renderToString === 'function') {
      try {
        return katex.renderToString(src, {
          displayMode: !!display,
          throwOnError: true,   // 抛错 -> 走 catch 回退，而不是渲染 KaTeX 自带的红色错误块
          strict: false,        // 容忍笔记里 \text{中文} 之类非严格写法
          trust: false
        });
      } catch (e) { /* 落到下面的原始文本回退 */ }
    }
    return '<code class="math-fallback" title="公式渲染失败，按原文显示">' +
      escapeHtml((display ? '$$' : '$') + src + (display ? '$$' : '$')) + '</code>';
  }

  function slug(text) {
    return 'h-' + String(text).toLowerCase()
      .replace(/[`*_~\[\]()#]/g, '')
      .replace(/[^\w\u4e00-\u9fff]+/g, '-')
      .replace(/^-+|-+$/g, '') || 'h';
  }

  /* 短哈希：给每个块生成稳定的 data-bkey（基于「块类型 + 源码」）。内容不变 -> bkey 不变；内容改了 -> bkey 变。 */ function cyrb53(str, seed) { var h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed; for (var i = 0, ch; i < str.length; i++) { ch = str.charCodeAt(i); h1 = Math.imul(h1 ^ ch, 2654435761); h2 = Math.imul(h2 ^ ch, 1597334677); } h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507); h1 ^= Math.imul(h2 ^ (h2 >>> 13), 3266489909); h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507); h2 ^= Math.imul(h1 ^ (h1 >>> 13), 3266489909); return 4294967296 * (2097151 & h2) + (h1 >>> 0); } function bkey(type, src) { var s = String(type) + '|' + String(src); return 'bk' + (cyrb53(s, 1)).toString(36) + (cyrb53(s, 2)).toString(36); }
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
    var imgStash = [];
    var PH_CODE = '\u0000C', PH_LINK = '\u0000L', PH_IMG = '\u0000I';

    // 1. 行内代码保护（用占位符 C+idx+C），原文保留避免被 escapeHtml 干扰
    text = String(text).replace(/```([^`\n]+)```/g, function (_, c) {
      codeStash.push('<code class="ic">' + escapeHtml(c.trim()) + '</code>');
      return PH_CODE + (codeStash.length - 1) + 'C\u0000';
    });
    text = text.replace(/`([^`]+)`/g, function (_, c) {
      codeStash.push('<code class="ic">' + escapeHtml(c) + '</code>');
      return PH_CODE + (codeStash.length - 1) + 'C\u0000';
    });

    // 1.5 图片（必须早于链接规则：否则 ![a](b) 里的 [a](b) 会先被链接规则吃掉，图片永远渲染不出来）
    //     支持 ![alt](url)、![alt](<含空格路径>)、![alt](url "标题")，src/alt 在此阶段转义后用占位符保护
    text = text.replace(/!\[([^\]]*)\]\(\s*(<[^>]+>|[^)\s]+)\s*(?:"[^"]*")?\s*\)/g, function (_, alt, src) {
      var s = src.replace(/^</, '').replace(/>$/, '').trim();
      imgStash.push('<img class="md-img" src="' + escapeHtml(s) + '" alt="' + escapeHtml(alt) + '" loading="lazy" onerror="this.classList.add(\'img-fail\')">');
      return PH_IMG + (imgStash.length - 1) + 'I\u0000';
    });

    // 2. 链接保护（用占位符 L+idx+L），在 escapeHtml 之前抓出：原 href 不会含 &lt;&gt;，
 //    且允许 label/href 中有空格、尖括号包裹、URL 片段等。
    text = text.replace(/\[([^\]]*)\]\(([^)]+)\)/g, function (_, label, href) {
      var inner = resolveLink(href, ctx);
      linkStash.push({ label: label, href: href, inner: inner });
      return PH_LINK + (linkStash.length - 1) + 'L\u0000';
    });

    // 2.5 公式（**必须在 escapeHtml 之前**提取：LaTeX 要拿原始字符，转义后的 &lt; 会污染公式源码。
    //     行内代码里的 $（如 `$NF`）已在步骤 1 被占位保护，不会被公式规则抓走。）
    var mathStash = [];
    var PH_MATH = '\u0000M';
    text = text.replace(/\$\$([^$\n]+?)\$\$/g, function (_, tex) {
      mathStash.push(renderMath(tex, false));
      return PH_MATH + (mathStash.length - 1) + 'M\u0000';
    });
    text = text.replace(/(^|[^\\$])\$([^$\n]+?)\$/g, function (_, pre, tex) {
      mathStash.push(renderMath(tex, false));
      return pre + PH_MATH + (mathStash.length - 1) + 'M\u0000';
    });

    // 3. 转义（链接、代码与公式已被占位保护，括号/尖括号不再干扰字符类）
    text = escapeHtml(text);
    // 4. 图片已在 1.5 阶段用占位符提取完毕，此处不再处理（占位符不含 &<>，不会被本阶段规则破坏）

    // 5. 加粗 / 斜体 / 删除线，清理源文件中未配对的残留标记
    text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*\*/g, '');
    text = text.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, '$1<em>$2</em>');
    text = text.replace(/~~([^~]+)~~/g, '<del>$1</del>');
    text = text.replace(/```/g, '');

    // 6. 还原公式（KaTeX 输出的 HTML 自带转义，直接放回即可；回退分支也已 escapeHtml）
    text = text.replace(/\u0000M(\d+)M\u0000/g, function (_, i) { return mathStash[+i]; });

    // 7. 还原链接（此阶段 label 已被 escapeHtml 转义，需要再 escape 一次保证安全）
    text = text.replace(/\u0000L(\d+)L\u0000/g, function (_, i) {
      var lnk = linkStash[+i];
      var label = escapeHtml(lnk.label);
      if (lnk.inner) {
        // inner 可能带 #锚点（resolveLink 剥出来的）；# 不能进 encodeURI(id) 段，分开拼
        var pp = lnk.inner.split('#');
        var ih = '#/doc/' + encodeURI(pp[0]) + (pp[1] ? '#' + encodeURIComponent(pp[1]) : '');
        return '<a class="md-link internal" href="' + ih + '">' + label + '</a>';
      }
      var h = lnk.href;
      if (/^(https?:)?\/\//.test(h) || h.indexOf('#') === 0 || h.indexOf('mailto:') === 0) {
        return '<a class="md-link" href="' + escapeHtml(h) + '" target="_blank" rel="noopener">' + label + '</a>';
      }
      return '<a class="md-link broken" title="链接目标不存在">' + label + '</a>';
    });

    // 8. 还原图片
    text = text.replace(/\u0000I(\d+)I\u0000/g, function (_, i) { return imgStash[+i]; });

    // 9. 还原行内代码
    text = text.replace(/\u0000C(\d+)C\u0000/g, function (_, i) { return codeStash[+i]; });
    return text;
  }

  /* 把笔记里的相对 md 链接解析成看板内 doc id（接收原始 href，未 escape）。
     2026-10-05：支持 GitHub 风格锚点 `路径.md#小节`（用户「完善知识卡片引用」commit
     全库 2489 处这种写法）——剥掉 #锚点 再判断目标存在，inner 返回 `id#锚点`；
     带 ?query 的同样剥掉。锚点能否定位由 app.js 的宽松标题匹配负责。 */
  function resolveLink(href, ctx) {
    if (/^(https?:)?\/\//.test(href)) return null;
    var clean = href.replace(/^<|>$/g, '').trim();
    // 本页锚点：[标签](<#小节>) 跳到当前文档对应标题（GitHub 同款行为）
    if (clean.indexOf('#') === 0) {
      var sa = clean.slice(1);
      if (!sa || !ctx || !ctx.docId || (ctx.exists && !ctx.exists[ctx.docId])) return null;
      return ctx.docId + '#' + sa;
    }
    var anchor = '';
    var hIdx = clean.indexOf('#');
    if (hIdx >= 0) { anchor = clean.slice(hIdx + 1); clean = clean.slice(0, hIdx); }
    var qIdx = clean.indexOf('?');
    if (qIdx >= 0) clean = clean.slice(0, qIdx);
    clean = clean.replace(/\\/g, '/').replace(/\/+/g, '/');
    if (!/\.md$/i.test(clean)) return null;
    if (!ctx || !ctx.docId) return null;
    var base = ctx.docId.split('/').slice(0, -1);
    var segs = clean.split('/');
    for (var i = 0; i < segs.length; i++) {
      if (segs[i] === '.' || segs[i] === '') continue;
      if (segs[i] === '..') base.pop();
      else base.push(segs[i]);
    }
    var id = base.join('/');
    if (ctx.exists && !ctx.exists[id]) return null;
    return anchor ? id + '#' + anchor : id;
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
    if (!ctx._slugSeen) ctx._slugSeen = {}; // 同名标题编号（GitHub 式）：首次 x，之后 x-1、x-2…
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
        var rawLines = code.split('\n');
        var lineHtml = rawLines.map(function (l, idx) {
          var h = (lang === 'text' || !lang) ? escapeHtml(l) : highlight(l, lang);
          return '<span class="code-line" data-bkey="' + bkey('cl' + idx, l) + '">' + h + '</span>';
        }).join('');
        out.push(
          '<div class="code-block">' +
          '<div class="code-head"><span class="code-lang">' + escapeHtml(label) + '</span>' +
          '<button class="code-copy" type="button" data-copy>复制</button></div>' +
          '<pre class="code-pre"><code data-raw="' + escapeHtml(code).replace(/"/g, '&quot;') + '">' +
          lineHtml +
          '</code></pre></div>'
        );
        continue;
      }

      // 数学块 $$..$$（可跨行）：必须在「普通段落」之前拦截，
      // 否则多行 aligned 会被合并成一行、换行被替换成 <br>，KaTeX 无法解析。
      if (/^\s*\$\$/.test(line)) {
        var mRest = line.replace(/^\s*\$\$/, '');
        var mTex = [];
        if (/\$\$\s*$/.test(mRest) && mRest.replace(/\$\$\s*$/, '').trim()) {
          mTex.push(mRest.replace(/\$\$\s*$/, ''));   // 单行形式 $$x$$
          i++;
        } else {
          if (mRest.trim()) mTex.push(mRest);
          i++;
          while (i < n && !/\$\$\s*$/.test(lines[i])) { mTex.push(lines[i]); i++; }
          if (i < n) { mTex.push(lines[i].replace(/\$\$\s*$/, '')); i++; }
        }
        out.push('<div class="md-math" data-bkey="' + bkey('math', mTex.join('\n')) + '">' +
          renderMath(mTex.join('\n'), true) + '</div>');
        continue;
      }

      // 分割线
      if (/^\s*([-*_])\s*\1\s*\1[\s\-*_]*$/.test(line)) { out.push('<hr data-bkey="' + bkey('hr', '') + '">'); i++; continue; }

      // 标题
      var h = line.match(/^(#{1,6})\s+(.*?)\s*$/);
      if (h) {
        var lv = h[1].length;
        var txt = h[2].replace(/#+\s*$/, '');
        var id = slug(txt);
        // GitHub 式去重：同名标题第二次出现加 -1，第三次加 -2……保证 id 唯一
        // 注意：必须先记下基准 id 再改 id，否则计数器会加到带后缀的新键上
        var bid = id;
        if (ctx._slugSeen[bid] !== undefined) { id = bid + '-' + ctx._slugSeen[bid]; ctx._slugSeen[bid]++; }
        else ctx._slugSeen[bid] = 1;
        out.push('<h' + lv + ' id="' + id + '" class="md-h md-h' + lv + '" data-bkey="' + bkey('h' + lv, txt) + '">' + inline(txt, ctx) + '</h' + lv + '>');
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
        var html = '<div class="table-wrap"><table class="md-table"><thead><tr data-bkey="' + bkey('th', head.join('|')) + '">';
        head.forEach(function (c) { html += '<th>' + inline(c, ctx) + '</th>'; });
        html += '</tr></thead><tbody>';
        for (var r = 2; r < rows.length; r++) {
          var cs = cells(rows[r]);
            html += '<tr data-bkey="' + bkey('td', (cs || []).join('|')) + '">';
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
            /^\s*\$\$/.test(pl) ||
            /^\s*\|/.test(pl)) break;
        pb.push(pl); i++;
      }
      if (i === pStart) { pb.push(lines[i]); i++; }   // 强制推进，防止单行卡死
      out.push('<p class="md-p" data-bkey="' + bkey('p', pb.join('\n')) + '">' + inline(pb.join('\n'), ctx).replace(/\n/g, '<br>') + '</p>');
    }

    return out.join('\n');
  }

  /* 行的缩进列数 / 是否有序列表标记（模块级，parseList 与 renderRest 共用） */
  function indent(l) { return l.match(/^\s*/)[0].length; }
  function ordered(l) { return /^\s*\d+[.)]\s/.test(l); }

  /* 列表项的「延续行」渲染：缩进挂在列表项下面的内容。
     2026-10-05 修复：以前延续行被直接递归 parseList，没有列表标记的行会被
     静默丢弃 —— 77 篇笔记里缩进在列表项下的 448 个 $$ 公式块就这样消失了
     （还留下空 <ul>）。现在按内容类型分派：
       · $$..$$ 块（可跨行）→ 块级公式 <div class="md-math">
       · 连续的列表行      → 递归 parseList（子列表）
       · 普通文本行        → 段落 <p class="md-p">（以前也被吞） */
  function renderRest(rows, ctx) {
    var out = '';
    var bufList = [];
    var bufPara = [];
    function flushList() {
      if (!bufList.length) return;
      var minInd = Math.min.apply(null, bufList.map(indent));
      out += parseList(bufList.map(function (x) { return x.slice(minInd); }), ctx);
      bufList = [];
    }
    function flushPara() {
      if (!bufPara.length) return;
      var minInd = Math.min.apply(null, bufPara.map(indent));
      out += '<p class="md-p">' + inline(bufPara.map(function (x) { return x.slice(minInd); }).join('\n'), ctx).replace(/\n/g, '<br>') + '</p>';
      bufPara = [];
    }
    for (var k = 0; k < rows.length; k++) {
      var ln = rows[k];
      if (!ln.trim()) { flushPara(); continue; }   // 空行只断开段落；列表行隔着空行仍算同一组
      if (/^\s*\$\$/.test(ln)) {                    // $$ 块（可跨行）：整块剥出来
        flushList(); flushPara();
        var tex = [];
        var head = ln.replace(/^\s*\$\$/, '');
        if (/\$\$\s*$/.test(head) && head.replace(/\$\$\s*$/, '').trim()) {
          tex.push(head.replace(/\$\$\s*$/, ''));   // 单行形式 $$x$$
        } else {
          if (head.trim()) tex.push(head);
          k++;
          while (k < rows.length && !/\$\$\s*$/.test(rows[k])) { tex.push(rows[k]); k++; }
          if (k < rows.length) tex.push(rows[k].replace(/\$\$\s*$/, ''));
        }
        out += '<div class="md-math" data-bkey="' + bkey('math', tex.join('\n')) + '">' + renderMath(tex.join('\n'), true) + '</div>';
        continue;
      }
      if (/^\s*([-*+]|\d+[.)])\s+/.test(ln)) { flushPara(); bufList.push(ln); }
      else { flushList(); bufPara.push(ln); }
    }
    flushList(); flushPara();
    return out;
  }

  function parseList(lines, ctx) {
    var html = '';
    var i = 0;
    var n = lines.length;
    var guard = 0, LIMIT = n * 4 + 2000;

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
      if (!items.length) continue;   // 防御：整组没有列表行时不输出空 <ul></ul>
      items.forEach(function (it) {
        var first = it[0];
        var rest = it.slice(1);
        var liSrc = first + '\n' + rest.join('\n');
        // 延续行交给 renderRest：块级公式 / 子列表 / 普通段落按类型分派
        body += '<li data-bkey="' + bkey('li', liSrc) + '">' + inline(first, ctx) + renderRest(rest, ctx) + '</li>';
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
