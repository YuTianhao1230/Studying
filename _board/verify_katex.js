/* 验证 KaTeX 能否渲染库里真实出现的公式（含中文 \text、aligned/bmatrix/cases、\bmod 等） */
const fs = require('fs');
const katexPath = 'C:/Users/10433/Desktop/Studying/_board/assets/katex/katex.min.js';
const katex = require(katexPath);

global.window = {};
eval(fs.readFileSync('C:/Users/10433/Desktop/Studying/_board/assets/data.js', 'utf8'));
const docs = window.STUDY_DATA.docs.filter(d => !d.isReadme);

// 收集所有公式样本
const blocks = [], inlines = [];
docs.forEach(function (d) {
  const b = d.body || '';
  (b.match(/\$\$[\s\S]+?\$\$/g) || []).forEach(x => blocks.push(x.slice(2, -2)));
  (b.match(/(^|[^\\$])\$([^$\n]+?)\$/g) || []).forEach(x => inlines.push(x.replace(/^[^$]*\$/, '').replace(/\$$/, '')));
});
console.log('样本：块级 ' + blocks.length + ' 处，行内 ' + inlines.length + ' 处\n');

let errBlock = [], errInline = [];
blocks.forEach(function (tex) {
  try {
    const h = katex.renderToString(tex, { displayMode: true, throwOnError: true, strict: false, trust: false });
    if (/katex-error/.test(h)) errBlock.push({ tex: tex.slice(0, 60), why: 'katex-error class' });
  } catch (e) { errBlock.push({ tex: tex.slice(0, 60), why: e.message.slice(0, 90) }); }
});
inlines.forEach(function (tex) {
  try {
    const h = katex.renderToString(tex, { displayMode: false, throwOnError: true, strict: false, trust: false });
    if (/katex-error/.test(h)) errInline.push({ tex: tex.slice(0, 50), why: 'katex-error class' });
  } catch (e) { errInline.push({ tex: tex.slice(0, 50), why: e.message.slice(0, 90) }); }
});

console.log('块级失败: ' + errBlock.length + ' / ' + blocks.length);
errBlock.slice(0, 12).forEach(e => console.log('  ✗ [' + e.why + ']  ' + e.tex.replace(/\n/g, '⏎')));
console.log('\n行内失败: ' + errInline.length + ' / ' + inlines.length);
errInline.slice(0, 15).forEach(e => console.log('  ✗ [' + e.why + ']  ' + e.tex));

// 中文 \text 渲染抽查
console.log('\n=== 中文 \\text 渲染抽查 ===');
try {
  const h = katex.renderToString('\\text{增长量}=\\frac{\\text{现期}\\times r}{1+r}', { displayMode: true, throwOnError: true });
  console.log('  输出含中文: ' + (/增长量/.test(h) ? '是' : '否'));
  console.log('  输出长度: ' + h.length);
} catch (e) { console.log('  ✗ ' + e.message); }

process.exit((errBlock.length + errInline.length) ? 1 : 0);
