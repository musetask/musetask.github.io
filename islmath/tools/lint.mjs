/* 结构检查：每个 ### 条目是否齐备 基础/推导/结论，锚点是否唯一。
   node tools/lint.mjs */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const SRC = path.join(path.dirname(path.dirname(fileURLToPath(import.meta.url))), "src");
let bad = 0;
const anchors = new Map();

const only = process.argv[2];
for (const f of fs.readdirSync(SRC)
  .filter((x) => x.endsWith(".md")).sort()
  .filter((x) => !only || x.startsWith(only))) {
  const src = fs.readFileSync(path.join(SRC, f), "utf8");
  // 前置知识页是「概念参考」，不要求逐公式的 基础/推导/结论 结构
  const ref = f.startsWith("prereq-");
  if (ref) {
    const h2 = (src.match(/^###\s+/gm) || []).length;
    console.log(f.padEnd(34) + `小节 ${String(h2).padStart(3)}  `
      + (h2 >= 4 ? "✓ 概念参考页" : "✗ 小节过少"));
    if (h2 < 4) bad++;
    continue;
  }
  const parts = src.split(/^###\s+/m).slice(1);
  const headAll = [...src.matchAll(/^###\s+(.+)$/gm)].map((m) => m[1]);
  // chNNb.md 是同一章的后半部分（拼接进 chNN.md），不重复章节首尾两节
  const part = /ch\d+b\.md$/.test(f);
  const noRoute = !part && !/^##\s+本章数学路线/m.test(src);
  const noTable = !part && !/^##\s+章末速查/m.test(src);
  const probs = [];
  if (noRoute) probs.push("缺 ## 本章数学路线");
  if (noTable) probs.push("缺 ## 章末速查");
  headAll.forEach((h, i) => {
    const id = (h.match(/\{#([^}]+)\}/) || [])[1];
    if (!id) probs.push(`条目 ${i + 1} 无锚点: ${h.slice(0, 40)}`);
    else if (anchors.has(id)) probs.push(`锚点重复 ${id}（${anchors.get(id)} 与 ${f}）`);
    else anchors.set(id, f);
    const body = parts[i] || "";
    const seg = body.split(/\n(?=##\s)/)[0];
    for (const lab of ["基础", "推导", "结论"]) {
      if (!new RegExp(`\\*\\*${lab}`, "u").test(seg)) {
        probs.push(`${h.slice(0, 34)} 缺 **${lab}**`);
      }
    }
  });
  console.log(f.padEnd(34) + `条目 ${String(headAll.length).padStart(3)}  `
    + (probs.length ? "✗ " + probs.length + " 处" : "✓"));
  for (const p of probs.slice(0, 6)) console.log("     - " + p);
  if (probs.length) bad += probs.length;

  // \text{...} 里不能再嵌 $...$：KaTeX 会把它当普通字符丢掉，下标会消失
  for (const m of src.matchAll(/\\text\{[^}]*\$[^}]*\}/g)) {
    console.log(f.padEnd(34) + "✗ \\text{} 内嵌了 $…$："
      + m[0].replace(/\s+/g, " ").slice(0, 50));
    bad++;
  }
}
console.log("-".repeat(60));
console.log(bad ? `共 ${bad} 处问题` : "全部条目结构齐备");
process.exitCode = bad ? 1 : 0;