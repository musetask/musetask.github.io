/* 校验 LaTeX 能否被 KaTeX 渲染。
   node tools/texcheck.mjs '<tex>'        # 单行，多条用 § 分隔
   node tools/texcheck.mjs -f src/chNN.md # 抽出文件里所有 $...$ / $$...$$ 逐条检查
   退出码非 0 表示有公式无法渲染 */
import fs from "node:fs";
import { render } from "./tex.mjs";

let bad = 0, n = 0;

function check(tex, where) {
  const t = tex.trim();
  if (!t) return;
  n++;
  try {
    render(t, true);
  } catch (e) {
    bad++;
    const one = String(e.message).split("\n")[0];
    console.log(`FAIL${where ? " [" + where + "]" : ""}: `
      + t.replace(/\n/g, " ").slice(0, 110));
    console.log(`      ${one.slice(0, 140)}`);
  }
}

const a = process.argv.slice(2);
if (a[0] === "-f") {
  const src = fs.readFileSync(a[1], "utf8").replace(/\\n/g, "\n");
  for (const m of src.matchAll(/\$\$([\s\S]+?)\$\$/g)) check(m[1], "$$");
  const rest = src.replace(/\$\$[\s\S]+?\$\$/g, "");
  for (const m of rest.matchAll(/\$([^$\n]+?)\$/g)) check(m[1], "$");
} else {
  for (const t of a.join(" ").split("§")) check(t);
}
console.log(`检查 ${n} 条，${bad} 条无法渲染`);
process.exitCode = bad ? 1 : 0;
