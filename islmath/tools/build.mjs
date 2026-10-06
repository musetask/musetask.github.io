/* ===================================================================
   ISLP 公式推导详解 — 构建脚本
   src/*.md  --(KaTeX 预渲染)-->  静态 HTML（完全离线，无运行时依赖）
   用法: node tools/build.mjs [--check]
   =================================================================== */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { render as tex } from "./tex.mjs";
import { SOURCES } from "./manifest.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(HERE);
const SRC = path.join(ROOT, "src");

const RE_TAG = /\s*\\tag\*?\s*\{([^}]*)\}/g;

const MATH_ERRORS = [];
let CUR = "";

/* ------------------------------------------------------------- 工具 */

const esc = (s) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

function slug(s, prefix) {
  const t = s.toLowerCase()
    .replace(/<[^>]+>/g, "")
    .replace(/[（(].*?[）)]/g, "")
    .replace(/[^a-z0-9一-鿿]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return (prefix || "s") + "-" + (t || "x");
}

/* ------------------------------------------------------------- 数学 */

function renderOrReport(s, display) {
  try {
    return tex(s, display);
  } catch (e) {
    MATH_ERRORS.push({
      ch: CUR, tex: s,
      msg: String(e.message).split("\n")[0].slice(0, 200),
    });
    return '<span class="katex-error" style="color:#b3261e">'
      + esc("公式错误: " + s) + "</span>";
  }
}

/* ------------------------------------------------------------- 行内 */

function inline(s) {
  const keep = [];
  const put = (html) => { keep.push(html); return '\u0001' + (keep.length - 1) + '\u0001'; };

  // 代码
  s = s.replace(/`([^`]+)`/g, (_, c) => put("<code>" + esc(c) + "</code>"));
  // 行内公式（行内不支持 \tag，静默去掉）
  s = s.replace(/\$([^$\n]+?)\$/g,
    (_, t) => put(renderOrReport(t.replace(RE_TAG, ""), false)));
  // 链接
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g,
    (_, t, u) => put('<a href="' + esc(u) + '">' + t + "</a>"));
  s = esc(s);
  // 强调
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[\s(（])\*([^*\n]+)\*/g, "$1<em>$2</em>");
  return s.replace(/\u0001(\d+)\u0001/g, (_, i) => keep[+i]);
}

/* ------------------------------------------------------------- 块 */

const RE = {
  h: /^(#{1,4})\s+(.*)$/,
  ul: /^[-*]\s+(.*)$/,
  ol: /^(\d+)[.)]\s+(.*)$/,
  bq: /^>\s?(.*)$/,
  eq: /^\$\$\s*(.*?)\s*\$\$$/,
  tab: /^\|/,
  hr: /^(---|___|\*\*\*)\s*$/,
  src: /^@src\s+(.*)$/,
};

function blocks(lines, ctx) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const ln = lines[i];

    if (!ln.trim()) { i++; continue; }

    if (RE.hr.test(ln)) { out.push("<hr>"); i++; continue; }

    let m = ln.match(RE.h);
    if (m) {
      const lvl = m[1].length;
      let txt = m[2];
      let id = null;
      const a = txt.match(/\s*\{#([^}]+)\}\s*$/);
      if (a) { id = a[1]; txt = txt.slice(0, a.index); }
      const body = inline(txt);
      const aid = id || ctx.unique(slug(txt, "s"));
      ctx.heads.push({ lvl, txt, id: aid });
      // 条目标题 "(3.4) …" 里的编号用于自动补公式编号
      const nm = RE_HEADNUM.exec(stripTags(txt).trim());
      if (lvl === 3 && nm) { ctx.eqNum = nm[1]; ctx.tagUsed = false; }
      else if (lvl <= 2) { ctx.eqNum = null; }
      out.push(`<h${lvl} id="${aid}" class="sec" data-t="${esc(stripTags(txt))}">`
        + body + `<a class="anchor" href="#${aid}" aria-label="本节链接">#</a></h${lvl}>`);
      i++;
      continue;
    }

    m = ln.match(RE.src);
    if (m) { out.push(`<div class="src">${inline(m[1])}</div>`); i++; continue; }

    if (ln.trim() === "$$") {                       // 多行显示公式
      const buf = [];
      i++;
      while (i < lines.length && lines[i].trim() !== "$$") buf.push(lines[i++]);
      i++;
      out.push(eqBox(buf.join("\n"), ctx));
      continue;
    }
    m = ln.match(RE.eq);
    if (m) { out.push(eqBox(m[1], ctx)); i++; continue; }

    if (RE.bq.test(ln)) {                           // 引用 / 提示块
      const buf = [];
      while (i < lines.length && (RE.bq.test(lines[i]) ||
        (lines[i].trim() && buf.length && !isBlockStart(lines[i])))) {
        const t = lines[i].match(RE.bq);
        buf.push(t ? t[1] : lines[i]);
        i++;
      }
      out.push("<blockquote>" + blocks(buf, ctx) + "</blockquote>");
      continue;
    }

    if (RE.tab.test(ln)) {                          // 表格
      const rows = [];
      while (i < lines.length && RE.tab.test(lines[i])) rows.push(lines[i++]);
      out.push(table(rows));
      continue;
    }

    if (RE.ul.test(ln) || RE.ol.test(ln)) {          // 列表
      const ordered = RE.ol.test(ln);
      const items = [];
      while (i < lines.length) {
        const t = lines[i].match(ordered ? RE.ol : RE.ul);
        if (!t) {
          if (lines[i].trim() && lines[i].startsWith("  ") && items.length) {
            items[items.length - 1] += " " + lines[i].trim(); i++; continue;
          }
          break;
        }
        items.push(ordered ? t[2] : t[1]);
        i++;
      }
      const tag = ordered ? "ol" : "ul";
      const cls = ordered ? ' class="steps"' : ' class="bl"';
      out.push(`<${tag}${cls}>` + items.map((x) => "<li>" + inline(x) + "</li>").join("")
        + `</${tag}>`);
      continue;
    }

    // 段落
    const buf = [ln];
    i++;
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
      buf.push(lines[i++]);
    }
    const raw = buf.join(" ").trim();
    const cls = labClass(raw);
    const body = inline(raw);
    const rest = stripTags(body.replace(/<strong>.*?<\/strong>/, ""))
      .replace(/[\s:：.。，、]/g, "");
    const headOnly = cls && !rest;
    out.push(headOnly
      ? `<p class="labhead ${cls}">${inline(/^\*\*(.+?)\*\*[：:]?/.exec(raw)[1])}</p>`
      : `<p${cls ? ' class="' + cls + '"' : ""}>${body}</p>`);
  }
  return out.join("\n");
}

function isBlockStart(ln) {
  return RE.h.test(ln) || RE.ul.test(ln) || RE.ol.test(ln) || RE.bq.test(ln) ||
    RE.eq.test(ln) || RE.tab.test(ln) || RE.hr.test(ln) || RE.src.test(ln) ||
    ln.trim() === "$$";
}

const LABS = [
  ["基础", "lab"], ["推导", "lab der"], ["推导过程", "lab der"],
  ["结论", "lab key"], ["含义", "lab key"], ["注意", "lab warn"],
  ["易错点", "lab warn"], ["备注", "lab warn"], ["条件", "lab der"],
  ["步骤", "lab der"], ["小结", "lab key"], ["验证", "lab der"],
];
function labClass(raw) {
  const m = raw.match(/^\*\*(.+?)\*\*/);
  if (!m) return null;
  const t = m[1].replace(/[：:]\s*$/, "").trim();
  for (const [k, c] of LABS) if (t.startsWith(k)) return c;
  return null;
}

const RE_HEADNUM = /^[（(]\s*([\d.]+)\s*[)）]/;

/* 公式框：把 \tag 抽出来做成右上角标注，这样在窄屏横向滚动时编号不会被推出视野。
   条目标题写成 "(3.4) …" 时，若正文第一个显示公式没有 \tag，则自动补上该编号。 */
function eqBox(texsrc, ctx) {
  const t = texsrc.trim();
  ctx.eqs.push(t);
  let tag = null;
  const body = t.replace(RE_TAG, (_, x) => {
    if (x.trim() && x.trim() !== "*") tag = x.trim();
    return "";
  });
  if (!tag && ctx.eqNum && !ctx.tagUsed) { tag = "(" + ctx.eqNum + ")"; }
  ctx.tagUsed = true;
  return `<div class="m-eq${tag ? " taged" : ""}">`
    + (tag ? `<span class="eqtag">${esc(tag)}</span>` : "")
    + renderOrReport(body, true) + "</div>";
}

function stripTags(s) { return s.replace(/<[^>]+>/g, ""); }

function table(rows) {
  const cells = rows
    .filter((r) => !/^\|[\s:|-]+\|$/.test(r.replace(/\s/g, "")))
    .map((r) => r.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
  if (!cells.length) return "";
  const head = cells[0];
  const body = cells.slice(1);
  const th = "<tr>" + head.map((c) => `<th>${inline(c)}</th>`).join("") + "</tr>";
  const tb = body.map((r) => "<tr>" + r.map((c) => `<td>${inline(c)}</td>`).join("")
    + "</tr>").join("");
  return `<table class="tb">${th}${tb}</table>`;
}

/* ------------------------------------------------------------- 页面 */

function renderMd(file, ctx) {
  const txt = fs.readFileSync(file, "utf8").replace(/\r\n/g, "\n");
  return blocks(txt.split("\n"), ctx);
}

function ctxNew(ch) {
  const seen = {};
  CUR = ch;
  return {
    heads: [], eqs: [], eqNum: null, tagUsed: false,
    unique(base) {
      const n = seen[base] || 0;
      seen[base] = n + 1;
      return n ? `${base}-${n + 1}` : base;
    },
  };
}

const HEAD = (t, d) => `<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="generator" content="islmath-build">
<title>${esc(t)}</title>
<meta name="description" content="${esc(d)}">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23476b85'/%3E%3Cpath d='M7 24V8h6a5 5 0 0 1 0 10H7m10-10 8 16m0-16-8 16' fill='none' stroke='%23fff' stroke-width='2.2' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<link rel="stylesheet" href="style.css">
<script>(function(){try{var t=localStorage.getItem('islp-theme');document.documentElement.setAttribute('data-theme',t==='dark'?'dark':'light');}catch(e){}})();</script>
</head>
<body>
<a class="skip" href="#main">跳到正文</a>
<div class="progress" id="progress"></div>
<header class="bar">
<a class="nav menu" href="index.html" aria-label="目录">☰</a>
<span class="title" id="barTitle">${esc(t)}</span>
<button class="nav" id="themeToggle" type="button" aria-label="切换深色模式">◐</button>
</header>
<main id="main">
`;

const FOOT = (pager) => `</main>
<nav class="pager" aria-label="章节导航">
${pager}
</nav>
<button class="totop" id="totop" type="button" aria-label="回到顶部">↑</button>
<script src="app.js"></script>
</body>
</html>
`;

function pager(prev, next) {
  const a = (c, cls) => c
    ? `<a class="${cls}" href="${c.out}"><span class="dir">${cls === "prev" ? "← 上一章" : "下一章 →"}</span><span class="t">${esc(c.short)}</span></a>`
    : '<span class="void"></span>';
  return a(prev, "prev") + a(next, "next");
}

const RE_H3EQ = /^[（(]\s*[\d.]+\s*[)）]|^公式/;

/* `###` 开头写成 "(3.4) …" 或 "公式 …" 的视为公式条目，其余视为普通小节 */
function isEqHead(h) { return h.lvl === 3 && RE_H3EQ.test(stripTags(h.txt).trim()); }

function chToc(heads) {
  const eqs = heads.filter((h) => h.lvl === 3);
  if (eqs.length < 3) return "";
  const eq = eqs.filter(isEqHead).length;
  const allEq = eq * 5 >= eqs.length * 3;   // 多数条目是公式条目时就叫「公式一览」
  const li = eqs.map((h) => `<li><a href="#${h.id}">${inline(h.txt)}</a></li>`).join("");
  return `<details class="chidx"><summary>`
    + `${allEq ? "本章公式一览" : "本页小节一览"}（${eqs.length} 条）</summary>`
    + `<ol class="bl">${li}</ol></details>`;
}

/* ------------------------------------------------------------- 原书链接校验 */

// 原书网页版是同级的 ../isl；构建时把每章 HTML 的 id 集合缓存下来，
// 用来验证 @src 行指向的锚点真实存在（避免手写锚点时打错）。
const bookIds = new Map();

function loadBookIds(file) {
  if (!file || bookIds.has(file)) return bookIds.get(file);
  let ids = null;
  try {
    const html = fs.readFileSync(path.join(ROOT, "..", "isl", file), "utf8");
    ids = new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]));
  } catch (e) { /* 原书 HTML 不在时跳过校验 */ }
  bookIds.set(file, ids);
  return ids;
}

function checkSrcLinks(ch, src) {
  for (const m of src.matchAll(/\.\.\/isl\/([^)\s#]+\.html)#([^\s)]+)/g)) {
    const ids = loadBookIds(m[1]);
    if (ids && !ids.has(m[2])) {
      console.error(`  ! ${ch.n}: 原书锚点不存在 → ../isl/${m[1]}#${m[2]}`);
      process.exitCode = 1;
    }
  }
}

/* ------------------------------------------------------------- 主流程 */

function main() {
  const pages = [];
  const search = [];

  for (const ch of SOURCES) {
    const ctx = ctxNew(ch.num);
    let body = "";
    for (const f of ch.files || [ch.n + ".md"]) {
      const p = path.join(SRC, f);
      if (!fs.existsSync(p)) {
        console.error("  ! 缺少源文件:", path.relative(ROOT, p));
        continue;
      }
      const raw = fs.readFileSync(p, "utf8");
      checkSrcLinks(ch, raw);
      body += renderMd(p, ctx) + "\n";
    }

    // 防护：条目标题里的 (N.M) 必须在 manifest 里出现过，防止自编编号
    if (ch.book) {
      try {
        const mf = JSON.parse(fs.readFileSync(
          path.join(SRC, `manifest_${ch.n}.json`), "utf8"));
        const known = new Set(mf.equations.filter((e) => e.num)
          .map((e) => e.num.replace(/[()]/g, "")));
        for (const h of ctx.heads) {
          const t = stripTags(h.txt);
          for (const m of t.matchAll(/\((\d+\.\d+)\)/g)) {
            if (!known.has(m[1])) {
              console.error(`  ! ${ch.n}: 标题编号 (${m[1]}) 不在 manifest 里: ${t.slice(0, 44)}`);
              process.exitCode = 1;
            }
          }
        }
      } catch (e) { /* manifest 缺失时跳过 */ }
    }

    const short = ch.cn;
    const head = `<div class="phead"><span class="cnt">${ch.kicker || ""}</span>`
      + `<span class="cur">${esc(ch.num ? "第 " + ch.num + " 章 · " + ch.cn : ch.cn)}</span>`
      + `<span class="cn">${esc(ch.en)}</span></div>`;

    const idx = pages.length;
    const prev = idx ? pages[idx - 1] : null;
    const next = pages[idx + 1] || null;
    const h1 = `<h1>${ch.num ? "第 " + ch.num + " 章 · " + esc(ch.cn) : esc(ch.cn)}</h1>`
      + `<p class="sub">${esc(ch.en)}</p>`;
    const doc = HEAD(
      `${ch.num ? ch.num + " · " : ""}${ch.cn} — ISLP 公式推导详解`,
      `${ch.cn}：${ch.en} 全部公式的逐步推导与前置知识`,
    ) + head + h1 + "\n" + chToc(ctx.heads) + "\n" + body + "\n"
      + FOOT(pager(prev, next));

    fs.writeFileSync(path.join(ROOT, ch.out), doc);
    pages.push({ ...ch, short, nheads: ctx.heads.length, neqs: ctx.eqs.length,
      neqheads: ctx.heads.filter((h) => h.lvl === 3).length });

    for (const h of ctx.heads) {
      search.push({
        l: h.lvl, f: ch.out, a: h.id,
        n: ch.num || "", t: stripTags(h.txt),
        p: ch.pages || "",
        k: (stripTags(h.txt) + " " + ch.cn + " " + ch.en).toLowerCase(),
      });
    }
    console.log(`  ${ch.out.padEnd(42)} 标题 ${String(ctx.heads.length).padStart(3)}`
      + `  公式 ${String(ctx.eqs.length).padStart(3)}`
      + `  条目 ${String(ctx.heads.filter((h) => h.lvl === 3).length).padStart(3)}`);
  }

  /* -------- 目录页 -------- */
  const rows = search.map((r) => ({
    level: r.l, file: r.f, id: r.a, num: r.n, t: r.t, pages: r.p, key: r.k,
  }));
  const body = [];
  body.push("<h1>ISLP 公式推导详解</h1>");
  body.push('<p class="sub">An Introduction to Statistical Learning — '
    + "全书数学公式的逐步推导</p>");
  body.push('<p class="byline">面向有大学数学基础、但计算细节已经生疏的读者</p>');
  body.push('<p class="note">每个公式给出「前置知识 → 分步推导 → 结论与易错点」，'
    + "可回跳原书对照</p>");
  const nav = pages.map((p) => {
    const label = p.num ? `第 ${p.num} 章 · ${p.cn}` : p.cn;
    return `<div class="tocbox"><a class="ttl" href="${p.out}">`
      + `<span class="n">${p.num || "·"}</span><span class="t">${esc(label)}</span>`
      + `<span class="pg">${p.neqheads} 式</span></a>`
      + `<p class="en">${esc(p.en)}${p.pages ? " · 原书 p." + p.pages : ""}</p></div>`;
  }).join("");
  body.push('<div class="chlist">' + nav + "</div>");
  body.push('<div class="finder"><input id="q" type="search" '
    + `placeholder="搜索公式 / 章节 / 关键词…（共 ${search.length} 条）" `
    + 'autocomplete="off" spellcheck="false" aria-label="搜索"></div>');
  body.push('<ol class="toc" id="toc"></ol>');
  body.append?.call?.(null);
  body.push('<p class="empty" id="empty" hidden>没有匹配的条目</p>');
  body.push("<script>var TOC=" + JSON.stringify(rows) + ";</script>");
  body.push('<p class="credit">内容依据 Gareth James 等 '
    + "<em>An Introduction to Statistical Learning</em>（2023，613 页）整理，"
    + "并对照 <a href=\"../isl/index.html\">原书网页版</a> 逐式核对。</p>");

  fs.writeFileSync(path.join(ROOT, "index.html"),
    HEAD("ISLP 公式推导详解", "《统计学习导论》全书公式推导与数学基础")
    + body.join("\n") + "\n" + FOOT(pager(null, pages[0])));

  const total = pages.reduce((a, p) => a + p.neqs, 0);
  console.log(`\nindex.html  条目 ${search.length} · 公式 ${total}`);

  if (MATH_ERRORS.length) {
    console.error(`\n!! KaTeX 错误 ${MATH_ERRORS.length} 处:`);
    for (const e of MATH_ERRORS) {
      console.error(`   [ch${e.ch}] ${e.msg}\n     ${e.tex.replace(/\n/g, " ").slice(0, 160)}`);
    }
    if (!process.argv.includes("--check")) process.exitCode = 1;
  } else {
    console.log("KaTeX: 全部公式渲染通过");
  }
}

main();
