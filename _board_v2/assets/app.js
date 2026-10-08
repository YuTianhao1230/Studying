/* app.js —— 看板主逻辑（框架文件，内容更新无需改动这里）
 * 内容来源：assets/data.js（由 build.py 生成）
 */
(function () {
  'use strict';

  var D = window.STUDY_DATA;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var esc = function (s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  var fn = function (n) { return (n || 0).toLocaleString('zh-CN'); };

  if (!D) {
    document.body.innerHTML = '<div class="loading"><div><b>没有读到数据</b><br><br>请先运行 <code>build.py</code> 或双击「刷新数据.bat」生成 assets/data.js</div></div>';
    return;
  }

  /* ---------------- 数据预处理 ---------------- */
  var docs = D.docs;
  var cats = D.cats;
  var byId = {};
  var catById = {};
  docs.forEach(function (d, i) {
    d.i = i;
    byId[d.id] = d;
    d._search = (d.title + ' ' + d.summary + ' ' + d.id + ' ' + d.body).toLowerCase();
  });
  cats.forEach(function (c) { catById[c.id] = c; });

  var PALETTE = ['#6366f1', '#0ea5e9', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#3b82f6', '#84cc16'];
  var catColor = {};
  cats.forEach(function (c, i) { catColor[c.id] = PALETTE[i % PALETTE.length]; });

  var idSet = {};
  docs.forEach(function (d) { idSet[d.id] = 1; });

  var routes = (D.meta && D.meta.routes) || [];

  /* ---------------- 状态 ---------------- */
  var state = {
    view: 'dashboard',   // dashboard | routes | route | cat | doc | search
    cat: null,
    doc: null,
    route: null,
    q: '',
    toc: true,
    expanded: {}
  };

  try {
    state.toc = localStorage.getItem('board.toc') !== '0';
  } catch (e) {}

  var theme = (function () {
    try { return localStorage.getItem('board.theme'); } catch (e) { return null; }
  })();
  if (!theme) theme = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', theme);

  /* ---------------- 侧边栏 ---------------- */
  function buildNav(filter) {
    filter = (filter || '').trim().toLowerCase();
    var html = '';

    var homeActive = state.view === 'dashboard' && !filter ? ' active' : '';
    html += '<div class="nav-item' + homeActive + '" data-go="dashboard"><span class="ico">▦</span><span>总览看板</span></div>';
    if (!filter) {
      var routesActive = state.view === 'routes' || state.view === 'route' ? ' active' : '';
      html += '<div class="nav-item' + routesActive + '" data-go="routes"><span class="ico">➜</span><span>复习路线</span><span class="cnt">' + routes.length + '</span></div>';
    }

    var list = cats;
    if (filter) {
      list = cats.map(function (c) {
        var subs = c.subs.filter(function (s) { return s.name.toLowerCase().indexOf(filter) > -1; });
        var hitDocs = docs.filter(function (d) {
          return d.cat === c.id && (d._search.indexOf(filter) > -1);
        });
        var selfHit = c.name.toLowerCase().indexOf(filter) > -1;
        if (!selfHit && !subs.length && !hitDocs.length) return null;
        return { id: c.id, name: c.name, count: c.count, words: c.words, subs: subs, docs: hitDocs };
      }).filter(Boolean);
    }

    list.forEach(function (c) {
      var isOpen = filter ? true : (state.expanded[c.id] || (state.cat === c.id) || (state.doc && state.doc.cat === c.id));
      var active = state.view === 'cat' && state.cat === c.id ? ' active' : '';
      html += '<div class="nav-item' + (isOpen ? ' open' : '') + active + '" data-cat="' + esc(c.id) + '">' +
        '<span class="caret">▶</span>' +
        '<span class="dot" style="background:' + catColor[c.id] + '"></span>' +
        '<span class="t">' + esc(c.name) + '</span>' +
        '<span class="cnt">' + c.count + '</span></div>';

      html += '<div class="sub-group"' + (isOpen ? '' : ' hidden') + '>';

      if (filter && c.docs && c.docs.length) {
        c.docs.slice(0, 60).forEach(function (d) {
          html += docLink(d);
        });
        if (c.docs.length > 60) html += '<div class="nav-sub" style="padding:4px 12px;font-size:11.5px;color:var(--text-3)">…还有 ' + (c.docs.length - 60) + ' 篇，按回车看全部</div>';
      } else {
        // 子目录：默认折叠，仅「当前正在阅读的笔记所在子目录」自动展开
        c.subs.forEach(function (s) {
          var subActive = state.doc && state.doc.cat === c.id && state.doc.sub === s.name;
          var sOpen = filter ? true : (subActive || state.expanded[c.id + '/' + s.name] === true);
          html += '<div class="nav-item nav-sub' + (sOpen ? ' open' : '') + '" data-sub="' + esc(c.id) + '|' + esc(s.name) + '">' +
            '<span class="caret">▶</span><span class="t">' + esc(s.name) + '</span>' +
            '<span class="cnt">' + s.count + '</span></div>';
          html += '<div class="sub-group"' + (sOpen ? '' : ' hidden') + '>';
          docs.filter(function (d) { return d.cat === c.id && d.sub === s.name && !d.isReadme; })
            .forEach(function (d) { html += docLink(d); });
          html += '</div>';
        });
        // 直接在分类根下的文件（平铺，不套分组标题；README 索引页不占笔记位）
        var roots = docs.filter(function (d) { return d.cat === c.id && !d.sub && !d.isReadme; });
        if (roots.length) {
          roots.forEach(function (d) { html += docLink(d); });
        }
      }
      html += '</div>';
    });

    if (filter && !list.length) html += '<div style="padding:14px 12px;color:var(--text-3);font-size:12.5px">没有匹配的笔记</div>';

    $('#nav').innerHTML = html;
  }

  function docLink(d) {
    var a = state.doc && state.doc.id === d.id ? ' active' : '';
    return '<div class="nav-doc' + a + '" data-doc="' + esc(d.id) + '" title="' + esc(d.name) + ' · ' + esc(d.id) + '"><span class="t">' + esc(d.name) + '</span></div>';
  }

  /* ---------------- 路由 ---------------- */
  function parseHash() {
    var h = decodeURIComponent(location.hash.replace(/^#/, ''));
    if (!h || h === '/') return { view: 'dashboard' };
    if (h.indexOf('/doc/') === 0) {
      var id = decodeURIComponent(h.slice(5));
      return byId[id] ? { view: 'doc', doc: byId[id] } : { view: 'dashboard', missing: id };
    }
    if (h.indexOf('/cat/') === 0) {
      var cid = decodeURIComponent(h.slice(5));
      return catById[cid] ? { view: 'cat', cat: catById[cid] } : { view: 'dashboard' };
    }
    if (h.indexOf('/search') === 0) {
      var q = (h.split('?')[1] || '').replace(/^q=/, '');
      return { view: 'search', q: decodeURIComponent(q) };
    }
    if (h === '/routes') return { view: 'routes' };
    if (h.indexOf('/route/') === 0) {
      var rn = decodeURIComponent(h.slice(7));
      var rr = routes.filter(function (r) { return r.name === rn; })[0];
      return rr ? { view: 'route', route: rr } : { view: 'routes' };
    }
    return { view: 'dashboard' };
  }

  function render() {
    var r = parseHash();
    state.view = r.view;
    state.cat = r.cat || null;
    state.route = r.route || null;
    state.doc = r.doc || null;
    state.q = r.q || '';

    if (r.missing) toast('找不到这篇笔记：' + r.missing);

    buildNav(r.view === 'search' ? state.q : '');
    if (r.view === 'search') $('#searchInput').value = state.q;

    if (r.view === 'doc') renderDoc(r.doc);
    else if (r.view === 'cat') renderCat(r.cat);
    else if (r.view === 'search') renderSearch(state.q);
    else if (r.view === 'routes') renderRoutes();
    else if (r.view === 'route') renderRoute(r.route);
    else renderDashboard();

    updateTocBtn();
    $('#toc').hidden = !(state.toc && r.view === 'doc');
  }

  /* ---------------- 首页 ---------------- */
  function renderDashboard() {
    $('.view').classList.add('view-wide');
    $('.view-inner').classList.remove('view-wide');
    $('#toc').hidden = true;

    var m = D.meta;
    var maxWords = Math.max.apply(null, cats.map(function (c) { return c.words; }));

    var html = '';
    html += '<div class="hero"><h1>' + esc(m.root) + ' 知识看板</h1>' +
      '<p>' + m.total + ' 篇笔记 · ' + m.categories + ' 个知识域 · ' + fn(m.words) + ' 字 · ' + esc(m.generatedAt) + ' 更新</p></div>';

    html += '<div class="stat-grid">' +
      card(m.total, '笔记总数') +
      card(m.categories, '知识分类') +
      card(fn(m.words), '总字数') +
      card(cats.length ? fn(Math.round(m.words / m.total)) : 0, '篇均字数') +
      '</div>';

    // 复习路线：来自《学习路线总览.md》，按目标组织的学习主线
    if (routes.length) {
      html += '<div class="sec-title">复习路线<span class="sec-sub">按目标顺序学习</span></div><div class="route-grid">';
      routes.forEach(function (r, idx) {
        var no = ('0' + (idx + 1)).slice(-2);
        var first = r.items.slice(0, 4).map(function (it) {
          return it.links[0] ? it.links[0].text.replace(/\.md$/i, '') : it.desc;
        });
        html += '<div class="route-card" data-route="' + esc(r.name) + '">' +
          '<div class="route-no">' + no + '</div>' +
          '<div class="route-main">' +
          '<div class="route-name">' + esc(r.name) + '</div>' +
          '<div class="route-meta">' + r.items.length + ' 步' + (r.intro ? ' · ' + esc(r.intro) : '') + '</div>' +
          '<div class="route-preview">' + esc(first.join(' → ')) + '</div>' +
          '</div></div>';
      });
      html += '</div>';
    }

    html += '<div class="sec-title">知识域</div><div class="cat-grid">';
    cats.forEach(function (c) {
      html += '<div class="cat-card" data-cat="' + esc(c.id) + '" style="--c:' + catColor[c.id] + '">' +
        '<h3><span class="dot" style="background:' + catColor[c.id] + '"></span>' + esc(c.name) + '</h3>' +
        '<div class="meta">' + c.count + ' 篇 · ' + fn(c.words) + ' 字</div>' +
        '<div class="subs">' + c.subs.map(function (s) { return '<span class="chip">' + esc(s.name) + ' ' + s.count + '</span>'; }).join('') + '</div>' +
        '<div class="bar"><i style="width:' + Math.max(6, Math.round(c.words / maxWords * 100)) + '%"></i></div>' +
        '</div>';
    });
    html += '</div>';

    // 总览导航：内容根目录下的文档（如学习路线总览、AGENT.md）
    var rootDocs = docs.filter(function (d) { return d.cat === '00_索引'; });
    if (rootDocs.length) {
      html += '<div class="sec-title">总览导航</div><div class="guide-grid">';
      rootDocs.forEach(function (d) {
        html += '<div class="guide-card" data-doc="' + esc(d.id) + '">' +
          '<div class="guide-main">' +
          '<div class="guide-tt">' + esc(d.name) + '</div>' +
          '<div class="guide-desc">' + esc(d.summary || '（无摘要）') + '</div>' +
          '<div class="guide-meta">' + esc(d.mtime) + ' · ' + fn(d.words) + ' 字</div>' +
          '</div></div>';
      });
      html += '</div>';
    }

    var recent = docs.slice().sort(function (a, b) { return b.ts - a.ts; }).slice(0, 10);
    html += '<div class="sec-title">最近更新</div><div class="recent-list">';
    recent.forEach(function (d) {
      html += '<div class="recent-item" data-doc="' + esc(d.id) + '">' +
        '<span class="dot" style="background:' + catColor[d.cat] + '"></span>' +
        '<span class="t">' + esc(d.name) + '</span>' +
        '<span class="p">' + esc(catName(d.cat)) + (d.sub ? ' / ' + esc(d.sub) : '') + ' · ' + esc(d.mtime) + '</span></div>';
    });
    html += '</div>';

    setView(html, '总览');
  }

  function card(num, label) {
    return '<div class="stat-card"><div class="stat-num">' + esc(num) + '</div><div class="stat-label">' + label + '</div></div>';
  }
  function catName(id) { return catById[id] ? catById[id].name : (id || '未分类'); }

  /* ---------------- 分类页 ---------------- */
  function renderCat(c) {
    $('.view').classList.add('view-wide');
    $('#toc').hidden = true;
    var list = docs.filter(function (d) { return d.cat === c.id && !d.isReadme; });

    var html = '';
    html += '<div class="hero" style="background:linear-gradient(135deg,' + catColor[c.id] + ',' + shade(catColor[c.id], -22) + ')">' +
      '<h1>' + esc(c.name) + '</h1><p>' + c.count + ' 篇 · ' + fn(c.words) + ' 字 · ' + c.subs.length + ' 个子主题</p></div>';

    var groups = {};
    list.forEach(function (d) { (groups[d.sub || '其他'] = groups[d.sub || '其他'] || []).push(d); });

    Object.keys(groups).forEach(function (g) {
      html += '<div class="sec-title">' + esc(g) + ' <span style="font-weight:400;text-transform:none;letter-spacing:0">' + groups[g].length + ' 篇</span></div>';
      html += '<div class="res-list" style="margin-bottom:22px">';
      groups[g].forEach(function (d) {
        html += '<div class="res-item" data-doc="' + esc(d.id) + '">' +
          '<h4>' + esc(d.name) + '<span class="path">' + esc(d.mtime) + ' · ' + fn(d.words) + ' 字</span></h4>' +
          '<p>' + esc(d.summary || '（无摘要）') + '</p></div>';
      });
      html += '</div>';
    });

    setView(html, c.name);
  }

  /* ---------------- 搜索页 ---------------- */
  function renderSearch(q) {
    $('.view').classList.add('view-wide');
    $('#toc').hidden = true;
    var key = (q || '').trim().toLowerCase();
    if (!key) {
      setView('<div class="res-empty"><div class="big">⌕</div>输入关键词开始搜索<br><span style="font-size:12.5px">支持标题、正文、路径全文匹配</span></div>', '搜索');
      return;
    }

    var res = [];
    docs.forEach(function (d) {
      var p = d._search.indexOf(key);
      if (p < 0) return;
      var ti = d.title.toLowerCase().indexOf(key);
      var si = d.summary.toLowerCase().indexOf(key);
      var snippet;
      if (si > -1) snippet = d.summary;
      else snippet = plain(d.body).substr(Math.max(0, p - d.title.length - d.summary.length - 40), 200);
      res.push({ d: d, score: (ti > -1 ? 0 : 0) + (ti === 0 ? 0 : 1) + (si > -1 ? 1 : 3), snippet: snippet.trim() });
    });
    res.sort(function (a, b) { return a.score - b.score || b.d.ts - a.d.ts; });

    var html = '<div class="sec-title">搜索「' + esc(q) + '」· 命中 ' + res.length + ' 篇</div>';
    if (!res.length) {
      html += '<div class="res-empty"><div class="big">∅</div>没有找到相关笔记<br><span style="font-size:12.5px">试试更短的关键词，或换用英文术语</span></div>';
    } else {
      html += '<div class="res-list">';
      res.slice(0, 200).forEach(function (r) {
        html += '<div class="res-item" data-doc="' + esc(r.d.id) + '">' +
          '<h4>' + hl(r.d.name, q) + '<span class="path">' + esc(r.d.id) + '</span></h4>' +
          '<p>' + hl(r.snippet, q) + '</p></div>';
      });
      html += '</div>';
      if (res.length > 200) html += '<div style="padding:16px;color:var(--text-3);font-size:13px;text-align:center">仅显示前 200 条，请细化关键词</div>';
    }
    setView(html, '搜索：' + q);
  }

  function plain(md) {
    return md.replace(/```[\s\S]*?```/g, ' ').replace(/[#>*`|\[\]]/g, ' ').replace(/\s+/g, ' ');
  }
  function hl(text, q) {
    var t = esc(text);
    if (!q) return t;
    var k = esc(q).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return t.replace(new RegExp(k, 'gi'), function (m) { return '<mark>' + m + '</mark>'; });
  }

  /* ---------------- 复习路线 ---------------- */
  function renderRoutes() {
    $('.view').classList.add('view-wide');
    $('#toc').hidden = true;
    var html = '<div class="hero" style="background:linear-gradient(135deg,#0f766e,#0d9488)"><h1>复习路线</h1>' +
      '<p>按目标组织的学习主线，来自《学习路线总览.md》。选定一条路线，按步骤顺序推进。</p></div>';
    html += '<div class="route-grid route-grid-lg">';
    routes.forEach(function (r, idx) {
      var no = ('0' + (idx + 1)).slice(-2);
      var first = r.items.slice(0, 4).map(function (it) { return it.links[0] ? it.links[0].text.replace(/\.md$/i, '') : it.desc; });
      html += '<div class="route-card" data-route="' + esc(r.name) + '">' +
        '<div class="route-no">' + no + '</div>' +
        '<div class="route-main">' +
        '<div class="route-name">' + esc(r.name) + '</div>' +
        '<div class="route-meta">' + r.items.length + ' 步' + (r.intro ? ' · ' + esc(r.intro) : '') + '</div>' +
        '<div class="route-preview">' + esc(first.join(' → ')) + '</div>' +
        '</div></div>';
    });
    html += '</div>';
    setView(html, '复习路线');
  }

  function renderRoute(route) {
    $('.view').classList.remove('view-wide');
    $('#toc').hidden = true;
    var html = '<div class="doc-head">' +
      '<h1 class="doc-title">' + esc(route.name) + '</h1>' +
      (route.intro ? '<p class="route-intro">' + esc(route.intro) + '</p>' : '') +
      '<div class="doc-meta">' + route.items.length + ' 步 · 来自《学习路线总览》' +
      ' · <a class="md-link" href="#/doc/' + encodeURI('学习路线总览.md') + '">查看源文档</a></div></div>';

    html += '<div class="route-steps">';
    route.items.forEach(function (it, idx) {
      var linksHtml, descHtml = '';
      var realLinks = it.links.filter(function (l) { return l.doc; });
      if (realLinks.length) {
        linksHtml = realLinks.map(function (l) {
          return '<a class="step-link" href="#/doc/' + encodeURI(l.doc) + '">' + esc(l.text.replace(/\.md$/i, '')) + '</a>';
        }).join('<span class="step-and">+</span>');
        if (it.desc) descHtml = '<div class="step-desc">' + esc(it.desc) + '</div>';
      } else {
        // 任务型步骤（无文档链接，如「第 N 天：…」）
        linksHtml = '<span class="step-link task">' + esc(it.desc || it.links[0].text) + '</span>';
      }
      html += '<div class="route-step">' +
        '<div class="step-no">' + (idx + 1) + '</div>' +
        '<div class="step-main">' +
        '<div class="step-links">' + linksHtml + '</div>' +
        descHtml +
        '</div></div>';
    });
    html += '</div>';

    setView(html, route.name);
  }

  /* ---------------- 阅读页 ---------------- */
  /* 剥掉正文开头的 H1 标题行：页头已用真实文件名作标题，避免题目下面再出现一个题目。
     仅当正文第一个非空行是 "# ..." 时剥离；H1 出现在中间或其他位置的统统保留。 */
  function stripTopH1(body) {
    var lines = String(body || '').replace(/\r\n/g, '\n').split('\n');
    for (var i = 0; i < lines.length; i++) {
      if (!lines[i].trim()) continue;
      if (/^#\s+\S/.test(lines[i].trim())) lines.splice(i, 1);
      break;
    }
    return lines.join('\n');
  }

  function renderDoc(doc) {
    if (!doc) return;
    $('.view').classList.remove('view-wide');
    var body = stripTopH1(doc.body);
    var ctx = { docId: doc.id, exists: idSet };

    var html = '<div class="doc-head">';
    html += '<h1 class="doc-title">' + esc(doc.name) + '</h1>';
    html += '<div class="doc-meta">' +
      '<span class="tag">' + esc(catName(doc.cat)) + '</span>' +
      (doc.sub ? '<span class="tag plain">' + esc(doc.sub) + '</span>' : '') +
      '<span>' + fn(doc.words) + ' 字</span><span>·</span><span>更新于 ' + esc(doc.mtime) + '</span>' +
      '<span>·</span><span class="path-hint" title="' + esc(doc.id) + '">' + esc(catName(doc.cat) + (doc.sub ? ' / ' + doc.sub : '')) + '</span>' +
      '</div></div>';

    html += '<div class="doc-body">' + MD.render(body, ctx) + '</div>';

    // 上下篇
    var prev = docs[doc.i - 1], next = docs[doc.i + 1];
    html += '<div class="doc-nav">';
    html += prev ? '<a href="#/doc/' + encodeURI(prev.id) + '"><span class="lab">← 上一篇</span><span class="tt">' + esc(prev.name) + '</span></a>'
                 : '<a style="opacity:.4;pointer-events:none"><span class="lab">← 上一篇</span><span class="tt">已是第一篇</span></a>';
    html += next ? '<a href="#/doc/' + encodeURI(next.id) + '" style="text-align:right"><span class="lab">下一篇 →</span><span class="tt">' + esc(next.name) + '</span></a>'
                 : '<a style="opacity:.4;pointer-events:none;text-align:right"><span class="lab">下一篇 →</span><span class="tt">已是最后一篇</span></a>';
    html += '</div>';

    setView(html, doc.name);

    // 目录
    var tocList = MD.toc(body);
    var tocHtml = '<div class="toc-title">目录</div>';
    if (!tocList.length) tocHtml += '<div style="padding:6px 8px;color:var(--text-3);font-size:12px">（无子标题）</div>';
    tocList.forEach(function (t) {
      tocHtml += '<a href="#' + t.id + '" class="lv' + t.level + '" data-toc="' + esc(t.id) + '">' + esc(t.text) + '</a>';
    });
    $('#toc').innerHTML = tocHtml;
    $('#toc').hidden = !state.toc;
    $('.view').scrollTop = 0;
    bindTocScroll();
  }

  function setView(html, crumbText) {
    var v = $('#viewInner');
    v.innerHTML = html;
    $('#crumbMain').textContent = crumbText || '';
    $('.view').scrollTop = 0;
  }

  /* ---------------- TOC 滚动联动 ---------------- */
  function bindTocScroll() {
    var view = $('.view');
    var links = $$('#toc a[data-toc]');
    if (!links.length) return;
    var els = links.map(function (a) { return { a: a, el: document.getElementById(a.getAttribute('data-toc')) }; }).filter(function (x) { return x.el; });

    function onScroll() {
      var top = view.getBoundingClientRect().top + 90;
      var cur = null;
      for (var i = 0; i < els.length; i++) {
        if (els[i].el.getBoundingClientRect().top <= top) cur = els[i];
        else break;
      }
      if (!cur && els.length) cur = els[0];
      links.forEach(function (a) { a.classList.remove('on'); });
      if (cur) cur.a.classList.add('on');
    }
    view.onscroll = onScroll;
    onScroll();
  }

  function scrollToId(id) {
    var el = document.getElementById(id);
    var view = $('.view');
    if (!el || !view) return;
    var y = el.getBoundingClientRect().top - view.getBoundingClientRect().top + view.scrollTop - 70;
    view.scrollTo({ top: y, behavior: 'smooth' });
  }

  /* ---------------- 交互 ---------------- */
  function toast(msg) {
    var t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout(t._tm);
    t._tm = setTimeout(function () { t.classList.remove('show'); }, 2200);
  }

  function shade(hex, p) {
    var n = parseInt(hex.slice(1), 16);
    var r = (n >> 16) + p, g = ((n >> 8) & 255) + p, b = (n & 255) + p;
    r = Math.max(0, Math.min(255, r)); g = Math.max(0, Math.min(255, g)); b = Math.max(0, Math.min(255, b));
    return '#' + ((r << 16) | (g << 8) | b).toString(16).padStart(6, '0');
  }

  function go(hash) {
    if (location.hash === hash) render();
    else location.hash = hash;
  }

  /* 事件绑定 */
  document.addEventListener('click', function (e) {
    var el;

    // 代码块复制
    if ((el = e.target.closest('[data-copy]'))) {
      var code = el.closest('.code-block').querySelector('code');
      var text = code.textContent;
      var done = function () {
        el.textContent = '已复制';
        el.classList.add('done');
        setTimeout(function () { el.textContent = '复制'; el.classList.remove('done'); }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallbackCopy(text, done); });
      } else fallbackCopy(text, done);
      return;
    }

    // 打开笔记
    if ((el = e.target.closest('[data-doc]'))) { go('#/doc/' + encodeURI(el.getAttribute('data-doc'))); return; }

    // 复习路线
    if ((el = e.target.closest('[data-route]'))) { go('#/route/' + encodeURIComponent(el.getAttribute('data-route'))); return; }
    if ((el = e.target.closest('[data-go="routes"]'))) { go('#/routes'); return; }

    // 分类：侧栏折叠 / 卡片跳转
    if ((el = e.target.closest('.nav-item[data-cat]'))) {
      var cid = el.getAttribute('data-cat');
      // 搜索过滤态下点击直接进分类页，否则折叠
      if (state.q) { go('#/cat/' + encodeURI(cid)); return; }
      state.expanded[cid] = !state.expanded[cid];
      var grp = el.nextElementSibling;
      if (grp && grp.classList.contains('sub-group')) grp.hidden = !state.expanded[cid];
      el.classList.toggle('open', !!state.expanded[cid]);
      return;
    }
    if ((el = e.target.closest('.cat-card[data-cat]'))) { go('#/cat/' + encodeURI(el.getAttribute('data-cat'))); return; }
    if ((el = e.target.closest('.nav-item[data-sub]'))) {
      var key = el.getAttribute('data-sub');
      state.expanded[key] = state.expanded[key] === false ? true : false;
      var g2 = el.nextElementSibling;
      if (g2 && g2.classList.contains('sub-group')) g2.hidden = !state.expanded[key];
      el.classList.toggle('open', !!state.expanded[key]);
      return;
    }
    if ((el = e.target.closest('[data-go="dashboard"]'))) { go('#/'); return; }

    // TOC 跳转
    if ((el = e.target.closest('#toc a[data-toc]'))) {
      e.preventDefault();
      scrollToId(el.getAttribute('data-toc'));
      return;
    }

    // 站内 md 链接
    if ((el = e.target.closest('a.md-link.internal'))) {
      e.preventDefault();
      go(el.getAttribute('href'));
      return;
    }
  });

  function fallbackCopy(text, cb) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); cb(); } catch (e) { toast('复制失败'); }
    document.body.removeChild(ta);
  }

  /* 搜索框 */
  var si = $('#searchInput');
  var tm;
  si.addEventListener('input', function () {
    var v = si.value.trim();
    clearTimeout(tm);
    tm = setTimeout(function () {
      if (state.view !== 'search') buildNav(v);   // 侧栏实时过滤
    }, 140);
  });
  si.addEventListener('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      var v = si.value.trim();
      if (v) go('#/search?q=' + encodeURIComponent(v));
      else go('#/');
      si.blur();
    }
    if (e.key === 'Escape') { si.value = ''; buildNav(''); si.blur(); }
  });

  /* 顶栏按钮 */
  $('#themeBtn').onclick = function () {
    theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('board.theme', theme); } catch (e) {}
    this.textContent = theme === 'dark' ? '☀' : '☾';
    this.title = theme === 'dark' ? '切换到浅色' : '切换到深色';
  };
  $('#themeBtn').textContent = theme === 'dark' ? '☀' : '☾';

  function updateTocBtn() {
    var b = $('#tocBtn');
    b.classList.toggle('on', state.toc);
    b.style.display = state.view === 'doc' ? '' : 'none';
  }
  $('#tocBtn').onclick = function () {
    state.toc = !state.toc;
    $('#toc').hidden = !state.toc || state.view !== 'doc';
    updateTocBtn();
    try { localStorage.setItem('board.toc', state.toc ? '1' : '0'); } catch (e) {}
  };

  $('#menuBtn').onclick = function () { $('#sidebar').classList.toggle('open'); };
  $('#brand').onclick = function () { go('#/'); $('#sidebar').classList.remove('open'); };

  /* 快捷键 */
  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault(); si.focus(); si.select(); return;
    }
    if (e.key === '/' && document.activeElement !== si && !/INPUT|TEXTAREA/.test(document.activeElement.tagName)) {
      e.preventDefault(); si.focus(); return;
    }
    if (e.key === 'Escape' && document.activeElement === si) { si.blur(); return; }
    // 阅读页 ← / → 翻篇
    if (state.view === 'doc' && state.doc && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
      if (e.key === 'ArrowLeft') { var p = docs[state.doc.i - 1]; if (p) go('#/doc/' + encodeURI(p.id)); }
      if (e.key === 'ArrowRight') { var nx = docs[state.doc.i + 1]; if (nx) go('#/doc/' + encodeURI(nx.id)); }
    }
  });

  /* 回到顶部 */
  var toTop = $('#toTop');
  $('.view').addEventListener('scroll', function () {
    toTop.classList.toggle('show', $('.view').scrollTop > 700);
  });
  toTop.onclick = function () { $('.view').scrollTo({ top: 0, behavior: 'smooth' }); };

  window.addEventListener('hashchange', render);
  render();
})();
