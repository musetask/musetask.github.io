/* 逐条对齐：manifest 的每个公式是否都有一个对应的 ### 条目。
   node tools/audit.mjs ch03       # 单章详细清单
   node tools/audit.mjs            # 全部章节汇总 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SOURCES } from "./manifest.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(path.dirname(HERE), "src");
const only = process.argv[2];

for (const ch of SOURCES) {
  if (!ch.book) continue;
  if (only && ch.n !== only) continue;
  const mf = JSON.parse(fs.readFileSync(path.join(SRC, `manifest_${ch.n}.json`)));
  const files = ch.files || [ch.n + ".md"];
  if (files.some((f) => !fs.existsSync(path.join(SRC, f)))) continue;
  const src = files.map((f) => fs.readFileSync(path.join(SRC, f), "utf8")).join("\n");

  // 把源文件切成 ### 条目
  const parts = src.split(/^###\s+/m).slice(1)
    .map((t) => t.split(/\n(?=##\s)/)[0]);
  const heads = [...src.matchAll(/^###\s+(.+)$/gm)].map((m) => m[1].trim());

  // 每个条目的「覆盖集合」：正文里提到的原书编号 + 自身标题编号
  const bodies = parts;
  const coverOf = (i) => {
    const b = bodies[i] || "";
    const s = new Set();
    const h = heads[i].replace(/\s*\{#[^}]*\}\s*$/, "");
    const hn = /^[（(]\s*([\d.]+)/.exec(h);
    if (hn) s.add(hn[1]);
    for (const m of b.matchAll(/\((\d+\.\d+)\)/g)) s.add(m[1]);
    for (const m of b.matchAll(/\\tag\*?\{(\d+\.\d+)\}/g)) s.add(m[1]);
    return s;
  };
  const covers = heads.map((_, i) => coverOf(i));

  const numbered = mf.equations.filter((e) => e.num);
  const unnum = mf.equations.filter((e) => !e.num);
  const missNum = numbered.filter((e) => {
    const n = e.num.replace(/[()]/g, "");
    return !covers.some((c) => c.has(n));
  });
  // 无编号公式：按顺序与未使用的条目对照
  const usedForNum = new Set();
  for (const c of covers) for (const v of c) usedForNum.add(v);
  const spare = heads.length - new Set(
    mf.equations.filter((e) => e.num).map((e) => e.num.replace(/[()]/g, ""))).size;

  console.log("=" .repeat(64));
  console.log(`${ch.n}  条目 ${heads.length}  编号 ${numbered.length}  无编号 ${unnum.length}`
    + `  覆盖缺口 ${missNum.length}`);
  if (missNum.length) {
    console.log("  未覆盖编号: " + missNum.map((e) => e.num).join(", "));
  }
  if (unnum.length > spare && spare >= 0) {
    console.log(`  无编号公式 ${unnum.length} 个，多出的条目 ${spare} 个 —— 需人工确认`);
  }
  if (only) {
    console.log("\n条目清单:");
    heads.forEach((h, i) => console.log(`  ${String(i + 1).padStart(3)}  ${h}`));
  }
}