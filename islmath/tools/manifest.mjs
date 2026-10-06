/* 为每个章节生成「公式清单」，供撰写者核对公式原貌。
   用法: node tools/manifest.mjs            # 生成 src/manifest_chNN.json
*/
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.dirname(HERE);            // islmath/
const BOOK = path.join(ROOT, "..", "isl");   // 原书网页版

export const SOURCES = [
  { n: "prereq", num: "", book: null, kicker: "全书通用",
    cn: "数学基础", en: "Mathematical Prerequisites",
    out: "00-prerequisites.html", pages: "",
    files: ["prereq-1-linear-algebra.md", "prereq-2-calculus.md",
            "prereq-3-probability.md", "prereq-4-optimization.md"] },
  { n: "ch01", num: "1", book: "02-introduction.html",
    cn: "导论", en: "Introduction", out: "01-introduction.html", pages: "12–24" },
  { n: "ch02", num: "2", book: "03-statistical-learning.html",
    cn: "统计学习概览", en: "An Overview of Statistical Learning",
    out: "02-statistical-learning.html", pages: "25–77" },
  { n: "ch03", num: "3", book: "04-linear-regression.html",
    cn: "线性回归", en: "Linear Regression", out: "03-linear-regression.html", pages: "78–143" },
  { n: "ch04", num: "4", book: "05-classification.html",
    cn: "分类", en: "Classification", out: "04-classification.html", pages: "144–208" },
  { n: "ch05", num: "5", book: "06-resampling-methods.html",
    cn: "重采样方法", en: "Resampling Methods",
    out: "05-resampling-methods.html", pages: "209–236" },
  { n: "ch06", num: "6", book: "07-linear-model-selection-and-regularization.html",
    cn: "线性模型选择与正则化",
    en: "Linear Model Selection and Regularization",
    out: "06-linear-model-selection-and-regularization.html", pages: "237–296" },
  { n: "ch07", num: "7", book: "08-moving-beyond-linearity.html",
    cn: "超越线性模型", en: "Moving Beyond Linearity",
    out: "07-moving-beyond-linearity.html", pages: "297–337",
    files: ["ch07.md", "ch07b.md"] },
  { n: "ch08", num: "8", book: "09-tree-based-methods.html",
    cn: "基于树的方法", en: "Tree-Based Methods",
    out: "08-tree-based-methods.html", pages: "338–373" },
  { n: "ch09", num: "9", book: "10-support-vector-machines.html",
    cn: "支持向量机", en: "Support Vector Machines",
    out: "09-support-vector-machines.html", pages: "374–405" },
  { n: "ch10", num: "10", book: "11-deep-learning.html",
    cn: "深度学习", en: "Deep Learning", out: "10-deep-learning.html", pages: "406–474" },
  { n: "ch11", num: "11", book: "12-survival-analysis-and-censored-data.html",
    cn: "生存分析与删失数据", en: "Survival Analysis and Censored Data",
    out: "11-survival-analysis-and-censored-data.html", pages: "475–508" },
  { n: "ch12", num: "12", book: "13-unsupervised-learning.html",
    cn: "无监督学习", en: "Unsupervised Learning",
    out: "12-unsupervised-learning.html", pages: "509–562" },
  { n: "ch13", num: "13", book: "14-multiple-testing.html",
    cn: "多重检验", en: "Multiple Testing", out: "13-multiple-testing.html", pages: "563–602" },
];

const strip = (s) =>
  s.replace(/<[^>]+>/g, "")
    .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
    .replace(/&nbsp;/g, " ").replace(/&#8722;/g, "-")
    .replace(/\s+/g, " ")
    .trim();

const clip = (s, k) => (s.length > k ? s.slice(0, k) + "…" : s);

export function manifest(ch) {
  const html = fs.readFileSync(path.join(BOOK, ch.book), "utf8");
  const lines = html.split("\n");
  const out = [];
  let sec = "";
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    const h = ln.match(/<h([23]) id="([^"]+)"[^>]*>(.*?)<\/h\3?>/);
    const hh = ln.match(/^<h([23]) id="([^"]+)"[^>]*>(.*?)<\/h\1>$/);
    if (hh) {
      const txt = strip(hh[3].replace(/<a class="anchor".*?<\/a>/, "")
        .replace(/<span class="num">([^<]*)<\/span>/, "$1 "));
      if (hh[1] === "2") sec = txt; else sec = sec + " › " + txt;
    }
    if (!ln.includes('class="eq"')) continue;
    const id = (ln.match(/id="(eq-[\d-]+)"/) || [])[1] || null;
    const num = (ln.match(/<span class="eqn">([^<]*)<\/span>/) || [])[1] || null;
    const img = (ln.match(/src="assets\/([^"]+)"/) || [])[1] || null;
    const page = img ? Number(/p(\d+)/.exec(img)[1]) : null;
    const prev = [], next = [];
    for (let j = i - 1, k = 0; j >= 0 && k < 4; j--) {
      const t = strip(lines[j]);
      if (t) prev.unshift(t);
      k++;
    }
    for (let j = i + 1, k = 0; j < lines.length && k < 4; j++) {
      const t = strip(lines[j]);
      if (t) next.push(t);
      k++;
    }
    out.push({
      id, num, page, img, section: sec,
      link: `../isl/${ch.book}${id ? "#" + id : ""}`,
      before: clip(prev.join(" "), 500),
      after: clip(next.join(" "), 700),
    });
  }
  return out;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const only = process.argv[2];
  // 原书网页版的解析器漏掉了一些编号公式，从 PDF 核对过的补充清单合并进来
  const EXTRA = JSON.parse(
    fs.readFileSync(path.join(ROOT, "src", "manifest_extra.json"), "utf8"));
  for (const ch of SOURCES) {
    if (!ch.book) continue;
    if (only && ch.n !== only) continue;
    const m = manifest(ch);
    const extra = (EXTRA[ch.num] && !EXTRA[ch.num]._说明) ? EXTRA[ch.num] : {};
    for (const [num, page] of Object.entries(extra)) {
      if (m.some((e) => e.num === `(${num})`)) continue;
      m.push({
        id: `eq-${num.replace(/\./g, "-")}`, num: `(${num})`, page: Number(page),
        img: null, section: "（原书有编号，网页版解析器未切图）",
        link: `../isl/${ch.book}#eq-${num.replace(/\./g, "-")}`,
        before: "", after: "", fromPdf: true,
      });
    }
    const f = path.join(ROOT, "src", `manifest_${ch.n}.json`);
    fs.writeFileSync(f, JSON.stringify({
      chapter: ch.num, cn: ch.cn, en: ch.en, book: ch.book,
      count: m.length, equations: m,
    }, null, 1));
    console.log(ch.n, ch.en, m.length, "equations ->",
      path.relative(ROOT, f));
  }
}
