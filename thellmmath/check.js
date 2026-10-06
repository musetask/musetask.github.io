#!/usr/bin/env node
/* 校验 thellmmath 页面：标签闭合、id 唯一、数学定界符配对、HTML 转义、KaTeX 渲染、图片/链接存在、每个公式的必备结构 */
const fs = require('fs');
const path = require('path');
const katex = require('../assets/katex/katex.min.js');

const ROOT = __dirname;
const SRC = path.join(ROOT, '_src');
const files = fs.readdirSync(SRC).filter(f => f.endsWith('.body.html')).sort();
const VOID = ['img', 'br', 'hr', 'meta', 'link', 'input', 'source'];

function decode(s) {
  return s.replace(/&lt;/g, '<').replace(/&gt;/g, '>')
          .replace(/&nbsp;/g, ' ').replace(/&amp;/g, '&')
          .replace(/&quot;/g, '"').replace(/&#39;/g, "'");
}
function stripTags(html) {
  return html.replace(/<(script|style)[\s\S]*?<\/\1>/gi, ' ')
             .replace(/<[a-zA-Z/!?][^>]*>/g, ' ');
}
/* 与 KaTeX 官方 auto-render.min.js 同规则：跳过 \\ 转义对，避免 \\[2pt] 被当成左定界符 */
function extractMath(text, lefts) {
  const out = [];
  let pos = 0, guard = 0;
  while (guard++ < 200000) {
    let open = -1, i = pos;
    while (i < text.length) {
      let hit = -1;
      for (const L of lefts) if (text.slice(i, i + L.left.length) === L.left) { hit = i; break; }
      if (hit !== -1) { open = hit; break; }
      if (text.charAt(i) === '\\') { i += 2; continue; }
      i++;
    }
    if (open === -1) break;
    let di = -1;
    for (let k = 0; k < lefts.length; k++) if (text.slice(open, open + lefts[k].left.length) === lefts[k].left) di = k;
    const right = lefts[di].right;
    let j = open + lefts[di].left.length, depth = 0, close = -1;
    while (j < text.length) {
      if (depth === 0 && text.slice(j, j + right.length) === right) { close = j; break; }
      const c = text.charAt(j);
      if (c === '\\') { j += 2; continue; }
      if (c === '{') depth++; else if (c === '}') depth = Math.max(0, depth - 1);
      j++;
    }
    if (close === -1) { out.push({ tex: null, at: open, around: text.slice(Math.max(0, open - 60), open + 60) }); break; }
    out.push({ tex: text.slice(open + lefts[di].left.length, close), display: lefts[di].display });
    pos = close + right.length;
  }
  return out;
}
const DELIMS = [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }];

let totalErr = 0;
const lines = [];

for (const f of files) {
  const raw = fs.readFileSync(path.join(SRC, f), 'utf8');
  const errs = [];

  /* 1. 标签闭合 */
  const tags = {};
  for (const m of raw.matchAll(/<(\/?)([a-zA-Z][a-zA-Z0-9]*)\b[^>]*?(\/?)>/g)) {
    const [, close, name, selfClose] = m;
    const n = name.toLowerCase();
    if (VOID.includes(n) || selfClose) continue;
    tags[n] = (tags[n] || 0) + (close ? -1 : 1);
  }
  for (const [n, c] of Object.entries(tags)) if (c !== 0) errs.push(`标签未闭合 <${n}> 差 ${c}`);

  /* 2. id 唯一 */
  const ids = [...raw.matchAll(/\sid="([^"]+)"/g)].map(m => m[1]);
  const dup = [...new Set(ids.filter((v, i) => ids.indexOf(v) !== i))];
  if (dup.length) errs.push(`重复 id: ${dup.join(', ')}`);

  /* 3. 数学检查：先在原始源码里查裸 < >，再解码实体后交给 KaTeX */
  const text = stripTags(raw);
  const spans = extractMath(text, DELIMS);
  let nMath = 0;
  for (const s of spans) {
    if (s.tex === null) { errs.push(`定界符未闭合: …${s.around.replace(/\n/g, ' ')}…`); continue; }
    nMath++;
    if (/[<>]/.test(s.tex)) errs.push(`数学里有裸 < > 未转义: ${s.tex.slice(0, 70)}`);
    const tex = decode(s.tex);
    try {
      katex.renderToString(tex, { displayMode: s.display, throwOnError: true, strict: false });
    } catch (e) {
      errs.push(`KaTeX 报错 [${tex.slice(0, 75).replace(/\n/g, ' ')}] → ${String(e.message).slice(0, 110)}`);
    }
  }

  /* 4. 链接与图片 */
  for (const m of raw.matchAll(/<div class="origin">\s*<img src="([^"]+)"/g))
    if (!fs.existsSync(path.resolve(ROOT, m[1]))) errs.push(`图片不存在: ${m[1]}`);
  for (const m of raw.matchAll(/href="(\.\.\/thellm\/[^"]*\.html)(#[^"]*)?"/g))
    if (!fs.existsSync(path.resolve(ROOT, m[1]))) errs.push(`原书页面不存在: ${m[1]}`);

  /* 5. 跨页锚点 */
  for (const m of raw.matchAll(/href="([0-9a-z-]+\.html)#(eq-[^"]+)"/g)) {
    const abs = path.join(ROOT, m[1]);
    if (!fs.existsSync(abs)) { errs.push(`跨页文件不存在: ${m[1]}`); continue; }
    if (!fs.readFileSync(abs, 'utf8').includes(`id="${m[2]}"`)) errs.push(`跨页锚点失效: ${m[1]}#${m[2]}`);
  }

  /* 6. 每个公式 section 的必备结构（class 含 supp 的是补充推导，不要求原图/①②） */
  const secs = raw.split(/<section class="fml/).slice(1);
  for (const s of secs) {
    const head = s.slice(0, 260);
    const id = (head.match(/\sid="([^"]+)"/) || [, '(无 id)'])[1];
    const isSupp = /\ssupp"/.test(head);
    const body = s.split(/<section class="fml"|<section id=/)[0];
    if (!isSupp && !/id="eq-/.test(head)) errs.push(`section id 不是 eq-* : ${id}（补充推导请加 class="fml supp"）`);
    if (!isSupp && !/<div class="origin">/.test(body)) errs.push(`${id}: 缺少原书公式图片`);
    if (!/<table class="sym">/.test(body)) errs.push(`${id}: 缺少符号表`);
    // 推导：要么用 <ol class="steps">，要么用"第 N 步"分段
    const hasSteps = /<ol class="steps"/.test(body);
    const hasStepP = (body.match(/<strong>第\s*\d+\s*步|<b>第\s*\d+\s*步/g) || []).length >= 2;
    if (!hasSteps && !hasStepP) errs.push(`${id}: 缺少推导步骤`);
    if (!isSupp && (!/<h3>[①②③]\s*公式/.test(body) || !/<h3>[①②③]\s*符号/.test(body)))
      errs.push(`${id}: 缺少"① 公式"或"② 符号"小标题`);
    if (!/<div class="note">/.test(body)) errs.push(`${id}: 缺少 note 直觉框`);
  }

  totalErr += errs.length;
  lines.push({ f, errs, n: secs.length, math: nMath, size: raw.length });
}

for (const r of lines) {
  console.log(`[${r.errs.length ? 'FAIL' : ' ok '}] ${r.f}  公式=${r.n} 数学=${r.math} chars=${r.size}`);
  for (const e of r.errs.slice(0, 20)) console.log('        - ' + e);
  if (r.errs.length > 20) console.log(`        … 另有 ${r.errs.length - 20} 条`);
}
console.log(`\n共 ${lines.length} 个源文件，${totalErr} 个问题。`);
process.exit(totalErr ? 1 : 0);