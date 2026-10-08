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

  /* ---------------- 标记状态（已读 / 标黄 / 更新快照，localStorage 持久化） ---------------- */
  var READ_KEY = 'board.read.v1', READ_BAK = 'board.read.bak', MARK_KEY = 'board.marks.v1', SNAP_KEY = 'board.snap.v1', LAST_KEY = 'board.last.v1';
  function lsGet(k, d) { try { var v = localStorage.getItem(k); return v ? JSON.parse(v) : (d === undefined ? {} : d); } catch (e) { return d === undefined ? {} : d; } }
  function lsSet(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; } }   // 返回是否成功：配额满不能静默
  // 已读主记录 + 镜像备份：主 key 损坏/被清时用备份恢复（2026-10-05 前后本地存储偶发丢失的保险）
  var readMap = lsGet(READ_KEY, {});
  if (!Object.keys(readMap).length) {
    var _bak = lsGet(READ_BAK, {});
    if (Object.keys(_bak).length) { readMap = _bak; lsSet(READ_KEY, readMap); }
  }
  function backupRead() { try { localStorage.setItem(READ_BAK, localStorage.getItem(READ_KEY) || ''); } catch (e) {} }
  var markMap = lsGet(MARK_KEY, {});
  var snapMap = lsGet(SNAP_KEY, {});
  // 继续阅读：{ _last: docId, <docId>: { pct: 0..1, t: ts } }
  var lastMap = lsGet(LAST_KEY, {});
  var pendingBkey = null;
  // 跨笔记锚点跳转：#/doc/<id>#<github式锚点>（2026-10-05，「完善知识卡片引用」commit 全库 2489 处）
  var pendingAnchor = null;
  // 只有从「接着上次看」卡片进入时才恢复滚动位置，避免普通打开被强行拉到中途
  var pendingResume = false;

  function saveLastPos(id, pct) {
    if (!id) return;
    lastMap._last = id;
    lastMap[id] = { pct: pct, t: Date.now() };
    lsSet(LAST_KEY, lastMap);
  }
  // 已读状态改为「用户手动控制」（不再是打开即自动标记）。
  // readMap[id] = { r:0|1, h:<内容指纹基线>, t:<时间戳> }
  //   r = 是否标为已读（手动切换）
  //   h = 检测「有更新」的基线指纹；标为已读时冻结为当时内容，取消已读时保留以便继续检测
  // snapMap[id] = 标为已读时的「块指纹串」（bk1|bk2|…），用于「高亮变更处」对比。
  //   2026-10-05 修复：旧版存整篇正文（全库标完 ≈3.7MB，撑爆 localStorage 5MB 配额，
  //   setItem 抛 QuotaExceeded 被静默吞掉 → 已读标记实际没存上，刷新后全变未读）。
  //   改存渲染后的 data-bkey 集合（对比语义 100% 不变，体积降约 65%，全库 ≈1.2MB）。
  function isRead(id) { var s = readMap[id]; return !!(s && s.r === 1); }
  function isUpdated(d) { var s = readMap[d.id]; return !!(s && s.r === 1 && d.hash && s.h && s.h !== d.hash); }

  /* 快照格式（v2）：把一篇正文渲染后各块的 data-bkey 用 | 连接。
     与 toggleDiff 的对比口径完全同源（都是 data-bkey），不存文本、只存指纹。 */
  function snapshotFromSource(body, docId) {
    try {
      var tmp = document.createElement('div');
      tmp.innerHTML = MD.render(stripTopH1(String(body || '')), { docId: docId, exists: idSet });
      var keys = [];
      $$('[data-bkey]', tmp).forEach(function (el) { keys.push(el.getAttribute('data-bkey')); });
      return keys.join('|');
    } catch (e) { return null; }
  }
  function isBkeySnap(v) { return typeof v === 'string' && v.indexOf('bk') === 0; }
  // 惰性迁移：读到旧格式（整篇正文）时转成 v2 并写回
  function getSnap(id) {
    var v = snapMap[id];
    if (v == null || v === '') return null;
    if (isBkeySnap(v)) return v;
    if (typeof v === 'string') {
      var c = snapshotFromSource(v, id);
      if (c != null && c.length) { snapMap[id] = c; lsSet(SNAP_KEY, snapMap); return c; }
      delete snapMap[id]; lsSet(SNAP_KEY, snapMap);
    }
    return null;
  }
  // 配额告急时按「标记时间最旧优先」淘汰一半快照（快照可牺牲，已读标记绝不牺牲）
  function pruneSnap(frac) {
    var ids = Object.keys(snapMap);
    if (ids.length < 8) return false;
    ids.sort(function (a, b) { return ((readMap[a] && readMap[a].t) || 0) - ((readMap[b] && readMap[b].t) || 0); });
    var n = Math.max(1, Math.floor(ids.length * (frac || 0.5)));
    for (var i = 0; i < n; i++) delete snapMap[ids[i]];
    return true;
  }
  function setRead(doc, val) {
    var s = readMap[doc.id] || {};
    var wasRead = !!s.r;
    var oldHash = s.h || null;
    if (val) {
      readMap[doc.id] = { r: 1, h: doc.hash || '', t: Date.now() };
    } else {
      readMap[doc.id] = { r: 0, h: doc.hash || oldHash || '', t: Date.now() };  // 保留 hash 基线，取消已读后仍可检测更新
    }
    // 先写关键的小块（已读状态），再写可牺牲的大块（快照）；快照失败也不能连累已读标记
    var ok = lsSet(READ_KEY, readMap);
    if (ok) backupRead();
    var warn = null;
    if (val) {
      snapMap[doc.id] = snapshotFromSource(doc.body, doc.id);   // 冻结「上次已读快照」（v2 块指纹）
      if (!lsSet(SNAP_KEY, snapMap)) {
        pruneSnap(0.5);
        if (!lsSet(SNAP_KEY, snapMap)) warn = '⚠ 本地存储空间不足，变更对比快照未保存（已读标记不受影响）';
      }
    }
    var changed = !!(wasRead && doc.hash && oldHash && oldHash !== doc.hash);
    return { changed: changed, readAt: wasRead ? s.t : null, nowRead: !!val, warn: warn };
  }

  /* ---------------- 侧边栏 ---------------- */
  // 顶部导航图标：内联 SVG，替代原来的 ▦ / ➜ / 📌（emoji 与符号字形跨平台不一致）
  var ICO = {
    home: '<svg class="icn" viewBox="0 0 24 24"><rect x="4" y="4" width="7" height="7" rx="1.5"/><rect x="13" y="4" width="7" height="7" rx="1.5"/><rect x="4" y="13" width="7" height="7" rx="1.5"/><rect x="13" y="13" width="7" height="7" rx="1.5"/></svg>',
    route: '<svg class="icn" viewBox="0 0 24 24"><circle cx="6" cy="18" r="2.5"/><circle cx="18" cy="6" r="2.5"/><path d="M8.5 16.5c4-4 5.5-8 7-8.5"/></svg>',
    mark: '<svg class="icn" viewBox="0 0 24 24"><path d="M7 4h10v17l-5-4-5 4z"/></svg>',
    chevron: '<svg class="icn chev" viewBox="0 0 24 24"><path d="M9 5l7 7-7 7"/></svg>'
  };
  // 分类折叠三角（统一用 SVG，避免 Unicode ▶ 在不同字体下大小/基線不一）
  function caret() { return '<span class="caret">' + ICO.chevron + '</span>'; }

  function buildNav(filter) {
    filter = (filter || '').trim().toLowerCase();
    var html = '';

    var homeActive = state.view === 'dashboard' && !filter ? ' active" aria-current="page' : '';
    html += '<div class="nav-item' + homeActive + '" data-go="dashboard"><span class="ico">' + ICO.home + '</span><span>总览看板</span></div>';
    if (!filter) {
      var routesActive = state.view === 'routes' || state.view === 'route' ? ' active" aria-current="page' : '';
      html += '<div class="nav-item' + routesActive + '" data-go="routes"><span class="ico">' + ICO.route + '</span><span>复习路线</span><span class="cnt">' + routes.length + '</span></div>';
      var mkCount = 0;
      for (var mid in markMap) if (markMap[mid]) mkCount += Object.keys(markMap[mid]).length;
      var marksActive = state.view === 'marks' ? ' active" aria-current="page' : '';
      html += '<div class="nav-item' + marksActive + '" data-go="marks"><span class="ico">' + ICO.mark + '</span><span>我的标注</span>' + (mkCount ? '<span class="cnt">' + mkCount + '</span>' : '') + '</div>';
    }

    var list = cats;
    if (filter) {
      list = cats.map(function (c) {
        var subs = c.subs.filter(function (s) {
          if (s.name.toLowerCase().indexOf(filter) > -1) return true;
          if (s.subs) return s.subs.some(function (gs) { return gs.name.toLowerCase().indexOf(filter) > -1; });
          return false;
        });
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
      var active = state.view === 'cat' && state.cat === c.id ? ' active" aria-current="page' : '';
      html += '<div class="nav-item' + (isOpen ? ' open' : '') + active + '" data-cat="' + esc(c.id) + '">' +
        caret() +
        '<span class="dot" style="--dc:' + catColor[c.id] + '"></span>' +
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
          var sFull = s.name;
          var docSub = (state.doc && state.doc.cat === c.id) ? (state.doc.sub || '') : '';
          // 展开条件：当前笔记在该子目录的任何层级之下
          var subActive = docSub === sFull || docSub.indexOf(sFull + '/') === 0;
          var sOpen = filter ? true : (subActive || state.expanded[c.id + '|' + sFull] === true);
          html += '<div class="nav-item nav-sub' + (sOpen ? ' open' : '') + '" data-sub="' + esc(c.id) + '|' + esc(sFull) + '">' +
            caret() + '<span class="t">' + esc(s.name) + '</span>' +
            '<span class="cnt">' + s.count + '</span></div>';
          html += '<div class="sub-group"' + (sOpen ? '' : ' hidden') + '>';
          if (s.subs) {
            // 嵌套：再渲染一层孙目录
            s.subs.forEach(function (gs) {
              var gsFull = sFull + '/' + gs.name;
              var gsActive = state.doc && state.doc.cat === c.id && state.doc.sub === gsFull;
              var gsOpen = filter ? true : (gsActive || state.expanded[c.id + '|' + gsFull] === true);
              html += '<div class="nav-item nav-sub nav-gs' + (gsOpen ? ' open' : '') + '" data-sub="' + esc(c.id) + '|' + esc(gsFull) + '">' +
                caret() + '<span class="t">' + esc(gs.name) + '</span>' +
                '<span class="cnt">' + gs.count + '</span></div>';
              html += '<div class="sub-group"' + (gsOpen ? '' : ' hidden') + '>';
              subDocsOf(c.id, gsFull).forEach(function (d) { html += docLink(d); });
              html += '</div>';
            });
            // 该子目录下仍有「直接挂在这里（非孙级）」的笔记时，单独列在孙目录之后
            var directHere = subDocsOf(c.id, sFull);
            directHere.forEach(function (d) { html += docLink(d); });
          } else {
            // 单层：直接列笔记
            subDocsOf(c.id, sFull).forEach(function (d) { html += docLink(d); });
          }
          html += '</div>';
        });
        // 直接在分类根下的文件（平铺，不套分组标题）；README 索引页排第一条
        var roots = subDocsOf(c.id, '');
        if (roots.length) {
          roots.forEach(function (d) { html += docLink(d); });
        }
      }
      html += '</div>';
    });

    if (filter && !list.length) html += '<div style="padding:14px 12px;color:var(--text-3);font-size:12.5px">没有匹配的笔记</div>';

    $('#nav').innerHTML = html;
  }

  /* 某个子目录下的全部文档（含该目录的 README 索引页）。
     README 是"这目录里有什么"的导航页，必须排在第一条；其余保持文件夹顺序。
     Array.prototype.sort 稳定，所以 README 归位后剩下的顺序不会乱。 */
  function subDocsOf(cid, sub) {
    var out = docs.filter(function (d) { return d.cat === cid && d.sub === sub; });
    out.sort(function (a, b) { return (b.isReadme ? 1 : 0) - (a.isReadme ? 1 : 0); });
    return out;
  }

  function docLink(d) {
    var a = state.doc && state.doc.id === d.id ? ' active" aria-current="page' : '';
    var flag = '';
    if (isRead(d.id)) {
      if (isUpdated(d)) flag = '<span class="read-flag upd" data-readflag="' + esc(d.id) + '" title="已读，且上次阅读后有更新（点击标为未读）">●</span>';
      else flag = '<span class="read-flag read" data-readflag="' + esc(d.id) + '" title="已读（点击标为未读）">✓</span>';
    } else {
      flag = '<span class="read-flag unread" data-readflag="' + esc(d.id) + '" title="未读（点击标为已读）">○</span>';
    }
    // README 是目录索引页：全库 88 篇都叫 README，侧栏里 88 条同名会糊成一片。
    // 显示成「所在目录名 + 索引」标签，一眼看得出是哪层的索引，也不计入篇数。
    var t = esc(d.name);
    var idxTag = '';
    if (d.isReadme) {
      var seg = String(d.id).split('/');
      var dir = seg.length > 1 ? seg[seg.length - 2] : d.sub;
      t = esc(dir || d.name);
      idxTag = '<span class="idx-tag" title="「' + t + '」目录的索引页，不计入篇数">索引</span>';
    }
    return '<div class="nav-doc' + a + '" data-doc="' + esc(d.id) + '" title="' + esc(d.name) + ' · ' + esc(d.id) + '"><span class="t">' + t + '</span>' + idxTag + flag + '</div>';
  }

  /* ---------------- 路由 ---------------- */
  function parseHash() {
    var h = decodeURIComponent(location.hash.replace(/^#/, ''));
    if (!h || h === '/') return { view: 'dashboard' };
    if (h.indexOf('/doc/') === 0) {
      // 2026-10-05：支持 `#/doc/<id>#<小节锚点>`（笔记里 GitHub 风格的 .md#锚点 链接跳转）
      var dh = h.slice(5).split('#');
      var id = dh[0];
      return byId[id] ? { view: 'doc', doc: byId[id], anchor: dh[1] || '' } : { view: 'dashboard', missing: id };
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
    if (h === '/marks') return { view: 'marks' };
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

    if (r.view === 'doc') { pendingAnchor = r.anchor || ''; renderDoc(r.doc); }
    else if (r.view === 'cat') renderCat(r.cat);
    else if (r.view === 'search') renderSearch(state.q);
    else if (r.view === 'routes') renderRoutes();
    else if (r.view === 'route') renderRoute(r.route);
    else if (r.view === 'marks') renderMarks();
    else renderDashboard();

    updateTocBtn();
    syncToc();
  }

  /* ---------------- 首页 ---------------- */
  function renderDashboard() {
    $('.view').classList.add('view-wide');
    $('.view-inner').classList.remove('view-wide');
    syncToc();

    var m = D.meta;
    var maxWords = Math.max.apply(null, cats.map(function (c) { return c.words; }));

    var html = '';
    var readN = docs.filter(function (d) { return isRead(d.id); }).length;
    var updN = docs.filter(function (d) { var s = readMap[d.id]; return s && s.r && d.hash && s.h && s.h !== d.hash; }).length;
    html += '<div class="hero"><h1>' + esc(m.root) + ' 知识看板</h1>' +
      '<p>' + m.total + ' 篇笔记' + (m.totalReadme ? '（另有 ' + m.totalReadme + ' 篇目录索引页）' : '') + ' · ' + m.categories + ' 个知识域 · ' + fn(m.words) + ' 字 · ' + esc(m.generatedAt) + ' 更新 · 已读 ' + readN + ' / ' + m.total + (updN ? ' · <b>' + updN + ' 篇有更新</b>' : '') + '</p></div>';

    // 接着上次看：486 篇的库，每次进来重新找位置是真实痛点
    var lastId = lastMap._last;
    var lastDoc = lastId ? byId[lastId] : null;
    if (lastDoc && !state.q) {
      var lp = (lastMap[lastId] && lastMap[lastId].pct) || 0;
      var lpPct = Math.round(lp * 100);
      html += '<div class="resume-card" data-doc="' + esc(lastDoc.id) + '" title="继续上次的阅读位置">' +
        '<div class="resume-main">' +
        '<div class="resume-lab">接着上次看</div>' +
        '<div class="resume-tt">' + esc(lastDoc.name) + '</div>' +
        '<div class="resume-path">' + esc(catName(lastDoc.cat)) + (lastDoc.sub ? ' / ' + esc(lastDoc.sub.split('/').join(' / ')) : '') + '</div>' +
        '</div>' +
        '<div class="resume-bar"><div class="track"><i style="width:' + Math.min(100, Math.max(2, lpPct)) + '%"></i></div>' +
        '<div class="resume-pct">' + lpPct + '%</div></div>' +
        '</div>';
    }

    // 统计卡：从静态计数升级为「有信息量」——已读进度 + 本周新增
    var weekAgo = Date.now() - 7 * 864e5;
    var weekN = docs.filter(function (d) { return !d.isReadme && d.ts && d.ts * 1000 > weekAgo; }).length;
    var readPct = m.total ? Math.round(readN / m.total * 100) : 0;
    html += '<div class="stat-grid">' +
      card(m.total, '笔记总数', '') +
      card(m.categories, '知识分类', '') +
      card(readPct + '%', '已读进度', 'acc') +
      card(weekN ? '+' + weekN : '0', '本周新增', weekN ? 'pos' : '') +
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
        '<div class="meta">' + c.count + ' 篇 · ' + fn(c.words) + ' 字' + (c.nReadme ? ' · 含 ' + c.nReadme + ' 索引' : '') + '</div>' +
        '<div class="subs">' + c.subs.map(function (s) {
          // chip 上的数字 = 点进这个子目录会列出的条数（只算直接放在该目录下的笔记），
          // 子目录自己的笔记归它自己的 chip，所以这里是「不累计」口径，和数字对得上。
          var t = esc(s.name);
          if (s.count) t += ' ' + s.count + ' 篇';
          if (s.nReadme) t += '<span class="chip-idx" title="/' + esc(s.name) + ' 目录下有 ' + s.nReadme + ' 篇索引页（README），不计入篇数">+' + s.nReadme + ' 索引</span>';
          return '<span class="chip">' + t + '</span>';
        }).join('') + '</div>' +
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
        '<span class="p">' + esc(catName(d.cat)) + (d.sub ? ' / ' + esc(d.sub.split('/').join(' / ')) : '') + ' · ' + esc(d.mtime) + '</span></div>';
    });
    html += '</div>';

    setView(html, '总览');
  }

  function renderMarks() {
    $('.view').classList.add('view-wide');
    syncToc();
    var ids = Object.keys(markMap).filter(function (id) { return markMap[id] && Object.keys(markMap[id]).length; });
    var total = 0;
    ids.forEach(function (id) { total += Object.keys(markMap[id]).length; });
    var html = '<div class="hero" style="--hc:#d97706"><h1>我的标注</h1><p>你在各篇笔记里标黄的重点（共 ' + total + ' 处）</p></div>';
    if (!ids.length) {
      html += '<div class="res-empty"><div class="big">✎</div>还没有标注<br><span style="font-size:12.5px">在阅读页把鼠标移到段落 / 代码行上，点 ✎ 即可标黄做笔记</span></div>';
    } else {
      ids.forEach(function (id) {
        var d = byId[id]; if (!d) return;
        var mk = markMap[id];
        html += '<div class="sec-title">' + esc(d.name) + ' <span style="font-weight:400;text-transform:none;letter-spacing:0">' + Object.keys(mk).length + ' 处</span></div>';
        html += '<div class="res-list">';
        Object.keys(mk).forEach(function (k) {
          var m = mk[k];
          html += '<div class="res-item marks-item" data-doc="' + esc(id) + '" data-bkey="' + esc(k) + '"><p>' + esc((m.t || '').slice(0, 160)) + '</p></div>';
        });
        html += '</div>';
      });
    }
    setView(html, '我的标注');
  }

  // 统计卡：标签在上、数字在下（原为数字在上，信息层级不对）
  function card(num, label, cls) {
    return '<div class="stat-card' + (cls ? ' ' + cls : '') + '">' +
      '<div class="stat-label">' + label + '</div>' +
      '<div class="stat-num">' + esc(num) + '</div></div>';
  }
  function catName(id) { return catById[id] ? catById[id].name : (id || '未分类'); }
  // 可读路径：知识域 / 子目录，而不是原始文件 id
  function docPath(d) { return catName(d.cat) + (d.sub ? ' / ' + d.sub.split('/').join(' / ') : ''); }

  /* ---------------- 全文检索（搜索页与命令面板共用） ---------------- */
  function doSearch(q) {
    var key = (q || '').trim().toLowerCase();
    if (!key) return [];
    var res = [];
    docs.forEach(function (d) {
      var p = d._search.indexOf(key);
      if (p < 0) return;
      var ti = d.title.toLowerCase().indexOf(key);
      var si = d.summary.toLowerCase().indexOf(key);
      var snippet;
      if (si > -1) snippet = d.summary;
      else snippet = plain(d.body).substr(Math.max(0, p - d.title.length - d.summary.length - 40), 200);
      res.push({ d: d, score: (ti === 0 ? 0 : 1) + (si > -1 ? 1 : 3), snippet: snippet.trim() });
    });
    res.sort(function (a, b) { return a.score - b.score || b.d.ts - a.d.ts; });
    return res;
  }

  /* ---------------- 分类页 ---------------- */
  function renderCat(c) {
    $('.view').classList.add('view-wide');
    syncToc();
    var list = docs.filter(function (d) { return d.cat === c.id && !d.isReadme; });

    // 顶层子主题数 = c.subs 长度；若有嵌套，再把孙级计入"主题数"用 avg 表述
    var subTopicCount = c.subs.length;
    var html = '';
    html += '<div class="hero" style="--hc:' + catColor[c.id] + '">' +
      '<h1>' + esc(c.name) + '</h1><p>' + c.count + ' 篇 · ' + fn(c.words) + ' 字 · ' + subTopicCount + ' 个子主题</p></div>';

    c.subs.forEach(function (s) {
      var sFull = s.name;
      var subDocs = list.filter(function (d) { return d.sub === sFull; });
      // s.count 只统计直接放在该目录下的笔记；孙目录的笔记归各自的子标题，所以这里不写「累计」
      html += '<div class="sec-title">' + esc(s.name) +
        (s.count ? ' <span style="font-weight:400;text-transform:none;letter-spacing:0">' + s.count + ' 篇</span>' : '') +
        (s.nReadme ? ' <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3)">' + s.nReadme + ' 索引</span>' : '') +
        '</div>';
      if (s.subs) {
        // 嵌套：列出每个孙目录
        s.subs.forEach(function (gs) {
          var gsFull = sFull + '/' + gs.name;
          var gsDocs = list.filter(function (d) { return d.sub === gsFull; });
          html += '<div class="sec-subtitle">' + esc(gs.name) +
            (gs.count ? ' <span style="font-weight:400;text-transform:none;letter-spacing:0">' + gs.count + ' 篇</span>' : '') +
            (gs.nReadme ? ' <span style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-3)">' + gs.nReadme + ' 索引</span>' : '') +
            '</div>';
          html += '<div class="res-list" style="margin-bottom:18px">';
          gsDocs.forEach(function (d) { html += renderResItem(d); });
          html += '</div>';
        });
        if (subDocs.length) {
          html += '<div class="sec-subtitle" style="color:var(--text-3)">该目录下其它 <span style="font-weight:400;text-transform:none">' + subDocs.length + ' 篇</span></div>';
          html += '<div class="res-list" style="margin-bottom:22px">';
          subDocs.forEach(function (d) { html += renderResItem(d); });
          html += '</div>';
        }
      } else {
        html += '<div class="res-list" style="margin-bottom:22px">';
        subDocs.forEach(function (d) { html += renderResItem(d); });
        html += '</div>';
      }
    });

    setView(html, c.name);
  }

  function renderResItem(d) {
    return '<div class="res-item" data-doc="' + esc(d.id) + '">' +
      '<h4>' + esc(d.name) + '<span class="path">' + esc(d.mtime) + ' · ' + fn(d.words) + ' 字</span></h4>' +
      '<p>' + esc(d.summary || '（无摘要）') + '</p></div>';
  }

  /* ---------------- 搜索页 ---------------- */
  function renderSearch(q) {
    $('.view').classList.add('view-wide');
    syncToc();
    var key = (q || '').trim().toLowerCase();
    if (!key) {
      setView('<div class="res-empty"><div class="big">⌕</div>输入关键词开始搜索<br><span style="font-size:12.5px">支持标题、正文、路径全文匹配</span></div>', '搜索');
      return;
    }

    var res = doSearch(q);

    // 分组：标题命中优先，其余归为正文命中；路径显示可读的分类/子目录，不再是原始 id
    var pathOf = docPath;
    function itemHtml(r) {
      return '<div class="res-item" data-doc="' + esc(r.d.id) + '">' +
        '<h4>' + hl(r.d.name, q) + '<span class="path">' + esc(pathOf(r.d)) + '</span></h4>' +
        '<p>' + hl(r.snippet, q) + '</p></div>';
    }
    var titleHits = [], bodyHits = [];
    res.forEach(function (r) {
      var inTitle = r.d.name.toLowerCase().indexOf(key) > -1 || (r.d.title || '').toLowerCase().indexOf(key) > -1;
      (inTitle ? titleHits : bodyHits).push(r);
    });

    var html = '<div class="sec-title">搜索「' + esc(q) + '」· 命中 ' + res.length + ' 篇</div>';
    if (!res.length) {
      html += '<div class="res-empty"><div class="big">∅</div>没有找到相关笔记<br><span style="font-size:12.5px">试试更短的关键词，或换用英文术语</span></div>';
    } else {
      var shown = 0;
      if (titleHits.length) {
        html += '<div class="res-group">标题命中 · ' + titleHits.length + ' 篇</div><div class="res-list">';
        titleHits.slice(0, 100).forEach(function (r) { html += itemHtml(r); shown++; });
        html += '</div>';
      }
      if (bodyHits.length) {
        html += '<div class="res-group">正文命中 · ' + bodyHits.length + ' 篇</div><div class="res-list">';
        bodyHits.slice(0, Math.max(0, 200 - shown)).forEach(function (r) { html += itemHtml(r); shown++; });
        html += '</div>';
      }
      if (res.length > shown) html += '<div style="padding:16px;color:var(--text-3);font-size:13px;text-align:center">仅显示前 ' + shown + ' 条，请细化关键词</div>';
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
    syncToc();
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
    syncToc();
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
    var prevSnap = getSnap(doc.id);
    var wasRead = isRead(doc.id);
    var upd = isUpdated(doc);
    var changed = wasRead && upd;
    var readAt = (readMap[doc.id] || {}).t || null;
    $('.view').classList.remove('view-wide');
    var body = stripTopH1(doc.body);
    var ctx = { docId: doc.id, exists: idSet };

    // README 的正文里，篇名被 stripTopH1 吃掉了（避免重复 h1），这里补回目录名当标题
    var title = doc.name;
    if (doc.isReadme) {
      var seg = String(doc.id).split('/');
      var dir = seg.length > 1 ? seg[seg.length - 2] : doc.sub;
      title = dir || doc.name;
    }
    var html = '<div class="doc-head">';
    // doc-title 挂上「被剥掉的顶层 H1」的 slug id：笔记里 [x › 卡片](<path.md#卡片名>) 这类
    // GitHub 锚点指向 H1，剥掉后正文里没有对应标题 —— 挂在补回的页头标题上，锚点就能命中页首。
    var h1Text = (String(doc.body).match(/^\s*#\s+(.+?)\s*$/) || [])[1] || title;
    html += '<h1 class="doc-title" id="' + MD.slug(h1Text) + '">' + esc(title) +
      (doc.isReadme ? '<span class="idx-tag" style="margin-left:10px;vertical-align:3px" title="目录索引页">索引</span>' : '') +
      '</h1>';
    // sub 可能是多级路径（"关键帧检测/任务与数据治理"），每级都单独显示为 tag；path-hint 用全路径加空格分隔
    var subTags = '';
    if (doc.sub) {
      subTags = doc.sub.split('/').map(function (s) { return '<span class="tag plain">' + esc(s) + '</span>'; }).join('');
    }
    var pathHint = catName(doc.cat) + (doc.sub ? ' / ' + doc.sub.split('/').join(' / ') : '');
    html += '<div class="doc-meta">' +
      '<span class="tag">' + esc(catName(doc.cat)) + '</span>' +
      subTags +
      '<span>' + fn(doc.words) + ' 字</span><span>·</span><span>更新于 ' + esc(doc.mtime) + '</span>' +
      '<span>·</span><span class="path-hint" title="' + esc(doc.id) + '">' + esc(pathHint) + '</span>' +
      '</div>';
    html += '<button class="read-toggle' + (wasRead ? ' on' : '') + '" id="readToggle" type="button" title="手动标记是否已读">' + (wasRead ? '✓ 已读' : '○ 标为已读') + '</button>';
    html += '</div>';

    if (changed) {
      var readDate = readAt ? new Date(readAt).toLocaleDateString('zh-CN') : '';
      html += '<div class="update-banner" id="updateBanner">' +
        '<span class="ub-ico">📌</span>' +
        '<span class="ub-text">本文自你上次阅读后已有内容更新' + (readDate ? '（' + readDate + '）' : '') + '</span>' +
        '<button class="ub-btn" id="ubDiff" type="button">高亮变更处</button>' +
        '<button class="ub-btn ghost" id="ubIgnore" type="button">知道了</button>' +
        '</div>';
    }
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

    setupHighlights(doc);
    bindReadToggle(doc);
    // 记录「上次打开的是哪篇」（保留已有进度，不在此处清零），供首页「接着上次看」使用
    saveLastPos(doc.id, (lastMap[doc.id] && lastMap[doc.id].pct) || 0);
    if (changed) bindUpdateBanner(doc, prevSnap);
    if (pendingBkey) {
      var pe = $('.doc-body [data-bkey="' + pendingBkey + '"]');
      if (pe) pe.scrollIntoView({ block: 'center' });
      pendingBkey = null;
    }
    // GitHub 风格锚点定位：笔记里的 [x › 小节](<path.md#anchor>)。
    // 看板标题 id 带 h- 前缀且标点规则与 GitHub 不同（：/、等 GitHub 删、看板转 -），
    // 所以用宽松比对：双方去掉 h- 前缀、去掉 - 和空格、转小写后相等即命中。
    if (pendingAnchor) {
      var norm = function (s) { return String(s).toLowerCase().replace(/[-\s]/g, ''); };
      var want = norm(decodeURIComponent(pendingAnchor));
      var cands = $$('.doc-body h1[id], .doc-body h2[id], .doc-body h3[id], .doc-body h4[id], .doc-body h5[id], .doc-body h6[id], .doc-head .doc-title[id]');
      var tgt = null;
      for (var ci = 0; ci < cands.length; ci++) {
        if (norm((cands[ci].getAttribute('id') || '').replace(/^h-/, '')) === want) { tgt = cands[ci]; break; }
      }
      if (tgt) {
        tgt.scrollIntoView({ block: 'start' });
        tgt.classList.add('anchor-hit');
        setTimeout(function () { tgt.classList.remove('anchor-hit'); }, 2600);
      } else {
        toast('未找到小节锚点：' + pendingAnchor);
      }
      pendingAnchor = null;
    }

    // 目录
    var tocList = MD.toc(body);
    var tocHtml = '<div class="toc-title">目录</div>';
    if (!tocList.length) tocHtml += '<div style="padding:6px 8px;color:var(--text-3);font-size:12px">（无子标题）</div>';
    tocList.forEach(function (t) {
      tocHtml += '<a href="#' + t.id + '" class="lv' + t.level + '" data-toc="' + esc(t.id) + '">' + esc(t.text) + '</a>';
    });
    $('#toc').innerHTML = tocHtml;
    syncToc();
    bindTocScroll();

    // 滚动位置：只有从「接着上次看」进入时才恢复，避免普通打开被强行拉到中途
    var v = $('.view');
    var saved = lastMap[doc.id] ? lastMap[doc.id].pct : 0;
    if (pendingResume && saved > 0.01) {
      pendingResume = false;
      requestAnimationFrame(function () {
        v.scrollTop = saved * (v.scrollHeight - v.clientHeight);
        updateReadProgress();
      });
    } else {
      v.scrollTop = 0;
      updateReadProgress();
    }
  }

  /* 顶部 2px 阅读进度条 + 把进度写回 lastMap */
  function updateReadProgress() {
    var v = $('.view');
    var bar = $('#readProgress');
    var max = v.scrollHeight - v.clientHeight;
    var pct = max > 0 ? v.scrollTop / max : 0;
    if (pct < 0) pct = 0; else if (pct > 1) pct = 1;
    if (bar) bar.style.width = (pct * 100) + '%';
    if (state.view === 'doc' && state.doc) saveLastPos(state.doc.id, pct);
  }

  function setupHighlights(doc) {
    var body = $('.doc-body'); if (!body) return;
    var blocks = $$('[data-bkey]', body);
    var marks = markMap[doc.id] || {};
    blocks.forEach(function (el) {
      if (!el.querySelector(':scope > .hl-btn')) {
        var b = document.createElement('button');
        b.type = 'button'; b.className = 'hl-btn'; b.textContent = '✎';
        b.title = '标黄 / 取消标黄';
        b.addEventListener('click', function (ev) {
          ev.stopPropagation(); ev.preventDefault();
          toggleMark(doc.id, el.getAttribute('data-bkey'), el);
        });
        el.appendChild(b);
      }
      if (marks[el.getAttribute('data-bkey')]) el.classList.add('hl');
    });
  }
  function toggleMark(docId, key, el) {
    if (!markMap[docId]) markMap[docId] = {};
    var txt = el.textContent.replace(/\s+/g, ' ').trim().slice(0, 200);
    if (markMap[docId][key]) {
      delete markMap[docId][key];
      el.classList.remove('hl');
      toast('已取消标黄');
    } else {
      markMap[docId][key] = { t: txt, at: Date.now() };
      el.classList.add('hl');
      toast('已标黄 ✓');
    }
    lsSet(MARK_KEY, markMap);
    refreshMarksBadge();
  }
  function refreshMarksBadge() {
    var cnt = 0; for (var k in markMap) if (markMap[k]) cnt += Object.keys(markMap[k]).length;
    var item = $('.nav-item[data-go="marks"]');
    if (!item) return;
    var old = item.querySelector('.cnt');
    if (cnt) {
      if (old) old.textContent = cnt;
      else item.insertAdjacentHTML('beforeend', '<span class="cnt">' + cnt + '</span>');
    } else if (old) old.remove();
  }
  function bindReadToggle(doc) {
    var btn = $('#readToggle'); if (!btn) return;
    btn.onclick = function () {
      var nowRead = !isRead(doc.id);
      var res = setRead(doc, nowRead);
      btn.textContent = nowRead ? '✓ 已读' : '○ 标为已读';
      btn.classList.toggle('on', nowRead);
      buildNav();                 // 刷新侧栏 ✓/●/○ 标记
      toast((res && res.warn) || (nowRead ? '已标为已读' : '已标为未读'));
    };
  }
  function bindUpdateBanner(doc, prevSnap) {
    var bd = $('#ubDiff'); if (bd) bd.onclick = function () { toggleDiff(doc, prevSnap); };
    var bi = $('#ubIgnore'); if (bi) bi.onclick = function () { var b = $('#updateBanner'); if (b) b.style.display = 'none'; };
  }
  function toggleDiff(doc, prevSnap) {
    var body = $('.doc-body'); if (!body) return;
    var old = {};
    // v2 快照本身就是 data-bkey 集合（'bk1|bk2|…'），直接拆开对比，不再重渲染
    if (prevSnap != null) String(prevSnap).split('|').forEach(function (k) { if (k) old[k] = 1; });
    var on = body.classList.toggle('show-diff');
    var n = 0;
    $$('[data-bkey]', body).forEach(function (el) {
      var k = el.getAttribute('data-bkey');
      var isNew = on && prevSnap != null && !old[k];
      el.classList.toggle('diff', isNew);
      if (isNew) n++;
    });
    if (!on) toast('已取消变更高亮');
    else if (prevSnap == null) toast('未找到上次阅读的快照');
    else toast('高亮了 ' + n + ' 处变更（左侧色条）');
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

    // 从「我的标注」跳转到对应笔记并滚动到该块
    if ((el = e.target.closest('.marks-item'))) {
      pendingBkey = el.getAttribute('data-bkey');
      go('#/doc/' + encodeURI(el.getAttribute('data-doc')));
      return;
    }

    // 手动切换「已读 / 未读」（侧栏 ✓/●/○，点一下即切换，不跳转）
    if ((el = e.target.closest('[data-readflag]'))) {
      e.preventDefault(); e.stopPropagation();
      var fid = el.getAttribute('data-readflag');
      var fd = byId[fid];
      if (fd) {
        var nowRead = !isRead(fid);
        var fres = setRead(fd, nowRead);
        buildNav();
        toast((fres && fres.warn) || (nowRead ? '已标为已读' : '已标为未读'));
      }
      return;
    }

    // 「接着上次看」：进入文档后恢复到上次滚动位置
    if ((el = e.target.closest('.resume-card'))) {
      pendingResume = true;
      go('#/doc/' + encodeURI(el.getAttribute('data-doc')));
      closeDrawer();
      return;
    }

    // 打开笔记
    if ((el = e.target.closest('[data-doc]'))) { closeDrawer(); go('#/doc/' + encodeURI(el.getAttribute('data-doc'))); return; }

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
      // 正确切换：undefined→true（首次点击即展开），true→false，false→true
      state.expanded[key] = state.expanded[key] === true ? false : true;
      var g2 = el.nextElementSibling;
      if (g2 && g2.classList.contains('sub-group')) g2.hidden = !state.expanded[key];
      el.classList.toggle('open', !!state.expanded[key]);
      return;
    }
    if ((el = e.target.closest('[data-go="marks"]'))) { go('#/marks'); return; }
    if ((el = e.target.closest('[data-go="dashboard"]'))) { go('#/'); return; }

    // TOC 跳转
    if ((el = e.target.closest('#toc a[data-toc]'))) {
      e.preventDefault();
      scrollToId(el.getAttribute('data-toc'));
      // 窄屏目录是右上角浮层，选中后自动收起（宽屏常驻，不动）
      if (tocNarrowMQ.matches) {
        state.tocNarrow = false;
        syncToc();
      }
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
  // 主题切换：图标是内联 SVG（月亮/太阳），由 CSS 按 data-theme 显示，不能用 textContent 覆盖
  function setTheme(next) {
    theme = next;
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('board.theme', theme); } catch (e) {}
    $('#themeBtn').title = theme === 'dark' ? '切换到浅色' : '切换到深色';
  }
  $('#themeBtn').onclick = function () { setTheme(theme === 'dark' ? 'light' : 'dark'); };
  $('#searchBtn').onclick = function () { openPalette(); };

  // 目录显隐：宽屏常驻右侧；窄屏（手机 / 大幅缩放）默认收起，由按钮显式唤出
  var tocNarrowMQ = (window.matchMedia && window.matchMedia('(max-width: 1023px)')) || { matches: false };
  function tocShouldShow() {
    if (state.view !== 'doc') return false;
    return tocNarrowMQ.matches ? !!state.tocNarrow : !!state.toc;
  }
  function syncToc() {
    $('#toc').hidden = !tocShouldShow();
    updateTocBtn();
  }
  function updateTocBtn() {
    var b = $('#tocBtn');
    b.classList.toggle('on', tocShouldShow());
    b.style.display = state.view === 'doc' ? '' : 'none';
  }
  $('#tocBtn').onclick = function () {
    if (tocNarrowMQ.matches) {
      state.tocNarrow = !state.tocNarrow;   // 窄屏只影响本次会话，不写入偏好
    } else {
      state.toc = !state.toc;
      try { localStorage.setItem('board.toc', state.toc ? '1' : '0'); } catch (e) {}
    }
    syncToc();
  };

  $('#menuBtn').onclick = function () {
    if ($('#sidebar').classList.contains('open')) closeDrawer(); else openDrawer();
  };
  $('#brand').onclick = function () { go('#/'); closeDrawer(); };

  /* ---------------- 展开 / 折叠全部 ---------------- */
  function expandAll(open) {
    cats.forEach(function (c) {
      state.expanded[c.id] = open;
      c.subs.forEach(function (s) {
        state.expanded[c.id + '|' + s.name] = open;
        if (s.subs) s.subs.forEach(function (gs) {
          state.expanded[c.id + '|' + s.name + '/' + gs.name] = open;
        });
      });
    });
    buildNav();
    toast(open ? '已展开全部' : '已折叠全部');
  }
  $('#expandBtn').onclick = function () { expandAll(true); };
  $('#collapseBtn').onclick = function () { expandAll(false); };

  /* ---------------- 帮助面板 ---------------- */
  function toggleHelp(force) {
    var m = $('#helpMask');
    m.hidden = (force === undefined) ? !m.hidden : !force;
  }
  $('#helpBtn').onclick = function () { toggleHelp(); };
  $('#helpMask').onclick = function (e) { if (e.target === this) toggleHelp(false); };

  /* ---------------- 命令面板（Ctrl/Cmd + K） ---------------- */
  var pal = { open: false, items: [], idx: 0 };
  var pi = $('#paletteInput');

  function openPalette() {
    $('#paletteMask').hidden = false;
    pal.open = true;
    pi.value = state.q || '';
    renderPalette(pi.value);
    pi.focus();
    pi.select();
  }
  function closePalette() {
    $('#paletteMask').hidden = true;
    pal.open = false;
    pi.blur();
  }
  function renderPalette(q) {
    var list = $('#paletteList');
    var key = (q || '').trim().toLowerCase();
    if (!key) {
      list.innerHTML = '<div class="pal-empty">输入关键词，在 ' + fn(docs.length) + ' 篇笔记的标题与正文中检索</div>';
      pal.items = []; pal.idx = 0;
      return;
    }
    var res = doSearch(q).slice(0, 40);
    var titleHits = [], bodyHits = [];
    res.forEach(function (r) {
      var inTitle = r.d.name.toLowerCase().indexOf(key) > -1 || (r.d.title || '').toLowerCase().indexOf(key) > -1;
      (inTitle ? titleHits : bodyHits).push(r);
    });
    pal.items = titleHits.concat(bodyHits);
    pal.idx = 0;
    if (!pal.items.length) {
      list.innerHTML = '<div class="pal-empty">没有找到「' + esc(q) + '」<br><span style="font-size:12px">试试更短的关键词，或换用英文术语</span></div>';
      return;
    }
    var html = '', i = 0;
    function group(name, arr) {
      if (!arr.length) return;
      html += '<div class="pal-group">' + name + ' · ' + arr.length + ' 篇</div>';
      arr.forEach(function (r) {
        html += '<div class="pal-item" data-pal="' + i + '" data-doc="' + esc(r.d.id) + '">' +
          '<div class="t">' + hl(r.d.name, q) + '</div>' +
          '<div class="m">' + esc(docPath(r.d)) + '</div></div>';
        i++;
      });
    }
    group('标题命中', titleHits);
    group('正文命中', bodyHits);
    list.innerHTML = html;
    markPal();
  }
  function markPal() {
    var items = $$('#paletteList .pal-item');
    items.forEach(function (el, i) { el.classList.toggle('on', i === pal.idx); });
    var on = items[pal.idx];
    if (on && on.scrollIntoView) on.scrollIntoView({ block: 'nearest' });
  }
  function movePal(d) {
    if (!pal.items.length) return;
    pal.idx = (pal.idx + d + pal.items.length) % pal.items.length;
    markPal();
  }
  function palOpen(all) {
    var r = pal.items[pal.idx];
    var q = pi.value.trim();
    closePalette();
    if (!r) { if (q) go('#/search?q=' + encodeURIComponent(q)); return; }
    if (all) go('#/search?q=' + encodeURIComponent(q));
    else go('#/doc/' + encodeURI(r.d.id));
  }

  pi.addEventListener('input', function () { renderPalette(pi.value); });
  pi.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); movePal(1); return; }
    if (e.key === 'ArrowUp') { e.preventDefault(); movePal(-1); return; }
    if (e.key === 'Enter') { e.preventDefault(); palOpen(e.metaKey || e.ctrlKey); return; }
    if (e.key === 'Escape') { e.preventDefault(); closePalette(); return; }
  });
  $('#paletteMask').addEventListener('click', function (e) {
    // 面板内的点击不能冒泡到 document 上的 [data-doc] 处理器，否则会导航两次
    e.stopPropagation();
    if (e.target === this) { closePalette(); return; }
    var el = e.target.closest('.pal-item');
    if (!el) return;
    pal.idx = parseInt(el.getAttribute('data-pal'), 10) || 0;
    palOpen(false);
  });

  /* ---------------- 窄屏抽屉 ---------------- */
  function closeDrawer() { $('#sidebar').classList.remove('open'); $('#backdrop').hidden = true; }
  function openDrawer() { $('#sidebar').classList.add('open'); $('#backdrop').hidden = false; }
  $('#backdrop').onclick = closeDrawer;

  /* ---------------- 快捷键 ---------------- */
  function isTyping() {
    var a = document.activeElement;
    return !!(a && /INPUT|TEXTAREA|SELECT/.test(a.tagName));
  }
  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (pal.open) closePalette(); else openPalette();
      return;
    }
    if (e.key === '?' || (e.shiftKey && e.key === '/')) { e.preventDefault(); toggleHelp(); return; }
    if (e.key === 'Escape') {
      if (pal.open) { closePalette(); return; }
      if (!$('#helpMask').hidden) { toggleHelp(false); return; }
      if (!$('#backdrop').hidden) { closeDrawer(); return; }
      if (document.activeElement === si) { si.blur(); }
      return;
    }
    if (isTyping()) return;
    if (e.key === '/') { e.preventDefault(); si.focus(); return; }
    if (e.key === 't' || e.key === 'T') { setTheme(theme === 'dark' ? 'light' : 'dark'); return; }
    if (e.key === 'e' || e.key === 'E') { expandAll(true); return; }
    if (e.key === 'q' || e.key === 'Q') { expandAll(false); return; }
    // 阅读页翻篇：k 上一篇 / j 下一篇（vim 习惯），方向键同义
    if (state.view === 'doc' && state.doc) {
      var t = null;
      if (e.key === 'k' || e.key === 'K' || e.key === 'ArrowLeft') t = docs[state.doc.i - 1];
      if (e.key === 'j' || e.key === 'J' || e.key === 'ArrowRight') t = docs[state.doc.i + 1];
      if (t) go('#/doc/' + encodeURI(t.id));
    }
  });

  /* 回到顶部 */
  var toTop = $('#toTop');
  $('.view').addEventListener('scroll', function () {
    toTop.classList.toggle('show', $('.view').scrollTop > 700);
    updateReadProgress();
  });
  toTop.onclick = function () { $('.view').scrollTo({ top: 0, behavior: 'smooth' }); };

  window.addEventListener('hashchange', render);

  /* ---------------- 旧快照空闲迁移 + 进度导出/导入 ---------------- */
  // 2026-10-05 前标已读留下的 v1 快照（整篇正文）在空闲时批量转成 v2 块指纹，释放 localStorage 空间
  setTimeout(function () {
    var stale = [];
    for (var k in snapMap) if (!isBkeySnap(snapMap[k]) && typeof snapMap[k] === 'string') stale.push(k);
    if (!stale.length) return;
    function step() {
      var batch = stale.splice(0, 10);
      batch.forEach(function (id) {
        var c = snapshotFromSource(snapMap[id], id);
        if (c != null && c.length) snapMap[id] = c; else delete snapMap[id];
      });
      lsSet(SNAP_KEY, snapMap);
      if (stale.length) setTimeout(step, 80);
    }
    step();
  }, 1500);

  // 导出：已读 + 标黄 + 快照 + 阅读位置 → JSON 下载（换浏览器 / 清缓存后可恢复）
  function exportProgress() {
    try {
      var payload = { kind: 'studying-board-progress', v: 2, at: new Date().toISOString(), read: readMap, marks: markMap, snaps: snapMap, last: lastMap };
      var blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'studying-progress-' + new Date().toISOString().slice(0, 10) + '.json';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 3000);
      var n = Object.keys(readMap).filter(function (k) { return readMap[k] && readMap[k].r === 1; }).length;
      toast('已导出进度（已读 ' + n + ' 篇）');
    } catch (e) { toast('导出失败：' + e.message); }
  }
  // 导入：覆盖恢复（导出文件是完整备份）
  function importProgress(file) {
    var fr = new FileReader();
    fr.onload = function () {
      try {
        var p = JSON.parse(String(fr.result));
        if (!p || p.kind !== 'studying-board-progress') { toast('文件不对：请选择本看板导出的进度文件'); return; }
        if (p.read && typeof p.read === 'object') { readMap = p.read; lsSet(READ_KEY, readMap); backupRead(); }
        if (p.marks && typeof p.marks === 'object') { markMap = p.marks; lsSet(MARK_KEY, markMap); }
        if (p.snaps && typeof p.snaps === 'object') { snapMap = p.snaps; lsSet(SNAP_KEY, snapMap); }
        if (p.last && typeof p.last === 'object') { lastMap = p.last; lsSet(LAST_KEY, lastMap); }
        var n = Object.keys(readMap).filter(function (k) { return readMap[k] && readMap[k].r === 1; }).length;
        buildNav();
        refreshMarksBadge();
        toast('已恢复进度：已读 ' + n + ' 篇');
      } catch (e) { toast('导入失败：文件损坏或格式不对'); }
    };
    fr.readAsText(file);
  }
  var bkMenu = $('#backupMenu'), bkWrap = $('#backupWrap'), importFile = $('#importFile');
  var bex = bkMenu ? $('[data-bk="export"]', bkMenu) : null;
  var bim = bkMenu ? $('[data-bk="import"]', bkMenu) : null;
  if ($('#backupBtn') && bkMenu && bkWrap && importFile && bex && bim) {
    $('#backupBtn').onclick = function (e) { e.stopPropagation(); bkMenu.classList.toggle('open'); };
    document.addEventListener('click', function (e) {
      if (!bkWrap.contains(e.target)) bkMenu.classList.remove('open');
    });
    bex.onclick = function () { bkMenu.classList.remove('open'); exportProgress(); };
    bim.onclick = function () { bkMenu.classList.remove('open'); importFile.click(); };
    importFile.onchange = function () { if (importFile.files && importFile.files[0]) importProgress(importFile.files[0]); importFile.value = ''; };
  }

  render();
})();
