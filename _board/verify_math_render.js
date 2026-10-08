/* 端到端渲染校验：md.js + KaTeX 渲染全库真实笔记的公式 */
const fs = require('fs');
const B = 'C:/Users/10433/Desktop/Studying/_board';

global.window = {};
global.window.katex = require(B + '/assets/katex/katex.min.js');
require(B + '/assets/md.js');
const MD = global.window.MD;      // md.js 是 (function(global){...})(window)，所以挂在 window 上

global.window.STUDY_DATA = null;
eval(fs.readFileSync(B + '/assets/data.js', 'utf8'));
const docs = window.STUDY_DATA.docs;

console.log('KaTeX 已加载: ' + (!!global.window.katex) + '  MD 已导出: ' + (!!MD));

let fail = 0;
function bad(m) { fail++; if (fail <= 25) console.log('  FAIL ' + m); }

let katexDocs = 0, katexTotal = 0, errTotal = 0, fallbackTotal = 0, rawLeak = 0;
let blockHtml = 0;

docs.forEach(function (d) {
  let html;
  try { html = MD.render(d.body || '', { docId: d.id, exists: {} }); }
  catch (e) { bad('渲染抛错 [' + d.name + '] ' + e.message.slice(0, 70)); return; }

  const kn = (html.match(/class="katex/g) || []).length;
  const eErr = (html.match(/katex-error/g) || []).length;
  const eFb = (html.match(/math-fallback/g) || []).length;
  const eMathBlock = (html.match(/class="md-math"/g) || []).length;
  if (kn) { katexDocs++; katexTotal += kn; }
  errTotal += eErr; fallbackTotal += eFb; blockHtml += eMathBlock;
  if (eErr) bad('出现 katex-error（红色报错块）: ' + d.name + ' ×' + eErr);

  // 未渲染的原始 LaTeX 泄漏：必须排除「代码块」与「行内代码」里的 $（如 `$NF`、表格里的 `world$`），
  // 那些本就不该被公式规则处理，不算泄漏。
  const bodyNoCode = String(d.body || '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`[^`\n]*`/g, '');
  const pendingBlock = (bodyNoCode.match(/\$\$[\s\S]+?\$\$/g) || []).length;
  const pendingInline = (bodyNoCode.match(/(^|[^\\$])\$[^$\n]+?\$/gm) || []).length;
  const pending = pendingBlock + pendingInline;
  if (pending > 0) {
    // 公式数量应能对上：成功渲染的 katex 节点 + 回退的 math-fallback
    if (kn + eFb === 0) { rawLeak++; bad('公式既没渲染也没回退: ' + d.name + '（正文残留 ' + pending + ' 处 $）'); }
  }
});

console.log('\n含 KaTeX 输出的笔记: ' + katexDocs + ' / ' + docs.length);
console.log('KaTeX 节点总数: ' + katexTotal);
console.log('块级公式容器 <div class="md-math">: ' + blockHtml);
console.log('katex-error（必须 0）: ' + errTotal);
console.log('math-fallback 回退次数: ' + fallbackTotal);
console.log('原始 LaTeX 泄漏笔记: ' + rawLeak);

// 抽查一篇：块级公式必须完整还原 aligned 环境（不能被 <br> 压平）
const target = docs.find(d => /aligned/.test(d.body || ''));
if (target) {
  const html = MD.render(target.body, { docId: target.id, exists: {} });
  const at = html.indexOf('class="md-math"');
  console.log('\n=== 抽查 aligned 块级公式：' + target.name + ' ===');
  if (at < 0) { bad('未找到块级公式容器'); }
  else {
    const seg = html.slice(at, at + 4000);
    console.log('  含 katex-display: ' + (/katex-display/.test(seg) ? '是' : '否'));
    console.log('  含 <br> 污染: ' + (/<br>/.test(seg.slice(0, seg.indexOf('</div>'))) ? '是（异常）' : '否（正确）'));
    console.log('  块级公式数（md-math）: ' + (html.match(/class="md-math"/g) || []).length +
      '  / 源文件 $$ 数: ' + Math.floor((target.body.match(/\$\$/g) || []).length / 2));
  }
}

// 抽查 awk 误判场景：Shell 笔记里代码块的 $9 不应被当公式
const sh = docs.find(d => /Shell/.test(d.name) || /awk/.test(d.body || ''));
if (sh) {
  const html = MD.render(sh.body, { docId: sh.id, exists: {} });
  const codeArea = html.match(/<code data-raw="[\s\S]*?<\/code>/);
  console.log('\n=== 抽查 awk 代码块（不应被公式化）===');
  if (codeArea) console.log('  代码块含 $9: ' + (/\$9/.test(codeArea[0]) ? '是（保留原文，正确）' : '否'));
}

// === 回退机制验证：绝不能让页面出现红色报错块 ===
console.log('\n=== 回退机制 ===');
// (1) KaTeX 存在，但内容不是合法 LaTeX（awk 里的 $9..$9 被 $..$ 规则误判）
const awkLine = "awk '$9 >= 500 {count[$9]++} END {print}' access.log";
const h1 = MD.render(awkLine, {});
console.log('  awk 误判 -> 含 katex-error: ' + (/katex-error/.test(h1) ? '有（异常）' : '无（正确）'));
console.log('  awk 误判 -> 已回退原文: ' + (/math-fallback/.test(h1) ? '是' : '否'));

// (2) KaTeX 完全没加载（离线 / 资源缺失）
delete require.cache[require.resolve(B + '/assets/md.js')];
global.window = {};
require(B + '/assets/md.js');
const MD2 = global.window.MD;
const h2 = MD2.render('行内 $a^2+b^2=c^2$ 与块级：\n\n$$\n\\frac{1}{2}\n$$\n', {});
console.log('  KaTeX 缺失 -> 全部回退原文: ' + ((h2.match(/math-fallback/g) || []).length === 2 ? '是（2/2）' : '否（' + (h2.match(/math-fallback/g) || []).length + '/2）'));
console.log('  KaTeX 缺失 -> 无红色报错: ' + (/katex-error/.test(h2) ? '有（异常）' : '无（正确）'));
if (!/math-fallback/.test(h2)) bad('KaTeX 缺失时未回退');

console.log('\nFAIL=' + fail);
process.exit(fail ? 1 : 0);
