/* 覆盖率检查：每章的公式条目数、编号是否与原书清单一致。
   node tools/check.mjs          # 全部章节
   node tools/check.mjs ch04     # 单章 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SOURCES } from "./manifest.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(path.dirname(HERE), "src");
const only = process.argv[2];

let bad = 0;
const totEntries = { n: 0 };
const rows = [];

for (const ch of SOURCES) {
  if (!ch.book) continue;
  if (only && ch.n !== only) continue;
  const mf = JSON.parse(fs.readFileSync(path.join(SRC, `manifest_${ch.n}.json`)));
  const files = ch.files || [ch.n + ".md"];
  const gone = files.filter((f) => !fs.existsSync(path.join(SRC, f)));
  if (gone.length) {
    rows.push([ch.n, "缺文件", gone.join(","), String(mf.count)]);
    bad++; continue;
  }
  const src = files.map((f) => fs.readFileSync(path.join(SRC, f), "utf8")).join("\n");

  // 所有三级标题 = 公式条目
  const heads = [...src.matchAll(/^###\s+(.+)$/gm)].map((m) => m[1].trim());
  // 标题里出现过的原书编号（正文括注与 \tag 都算）
  const seen = new Set();
  for (const m of src.matchAll(/\((\d+\.\d+)\)/g)) seen.add(m[1]);
  for (const m of src.matchAll(/\\tag\*?\{(\d+\.\d+)\}/g)) seen.add(m[1]);

  const missing = [];
  for (const e of mf.equations) {
    if (!e.num) continue;
    const n = e.num.replace(/[()]/g, "");
    if (!seen.has(n)) missing.push(n);
  }
  const flagged = missing.length ? `缺 ${missing.length}: ${missing.slice(0, 8).join(",")}`
    + (missing.length > 8 ? "…" : "") : "齐";
  if (missing.length) bad++;
  rows.push([ch.n, String(heads.length), `${mf.count} 式 / ${seen.size} 号`, flagged]);
  totEntries.n += heads.length;
}

const w = [6, 8, 22, 40];
console.log("章".padEnd(w[0]) + "条目".padEnd(w[1]) + "原书".padEnd(w[2]) + "编号");
console.log("-".repeat(70));
for (const r of rows) {
  console.log(r[0].padEnd(w[0]) + r[1].padEnd(w[1]) + r[2].padEnd(w[2]) + r[3]);
}
console.log("-".repeat(70));
console.log("合计公式条目 " + totEntries.n);
process.exitCode = bad ? 1 : 0;