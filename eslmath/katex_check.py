#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用站点自带的 KaTeX 把每个 src/*.md 里的公式全部渲染一遍，
检查有没有 KaTeX 解析错误。纯 Node + 全站共享的 ../assets/katex，不联网、不写文件。

用法： node katex_check.js
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
VENDOR = os.path.join(HERE, os.pardir, "assets", "katex")

JS = r"""
const fs = require('fs');
const path = require('path');
const katexDir = process.argv[2];
const katex = require(path.join(katexDir, 'katex.min.js'));
const files = JSON.parse(process.argv[3]);
let bad = [], nDisp = 0, nInline = 0;
for (const f of files) {
  const md = fs.readFileSync(f, 'utf8');
  // display math: $$ ... $$
  const dispRe = /\$\$([\s\S]*?)\$\$/g;
  let m;
  while ((m = dispRe.exec(md))) {
    nDisp++;
    let tex = m[1].replace(/\\?\s*\\eqno\{[^}]*\}/g, '').trim();
    if (!tex) continue;
    try { katex.renderToString(tex, {throwOnError: true, displayMode: true}); }
    catch (e) { bad.push([path.basename(f), 'display', tex.slice(0, 90), e.message]); }
  }
  // inline math: $...$ not part of $$
  const inlineRe = /(^|[^$])\$([^$\n]{1,400}?)\$/g;
  while ((m = inlineRe.exec(md))) {
    nInline++;
    const tex = m[2].replace(/\\?\s*\\eqno\{[^}]*\}/g, '').trim();
    if (!tex) continue;
    try { katex.renderToString(tex, {throwOnError: true, displayMode: false}); }
    catch (e) { bad.push([path.basename(f), 'inline', tex.slice(0, 90), e.message]); }
  }
}
console.log(JSON.stringify({nDisp, nInline, bad}));
"""

js_path = os.path.join(HERE, "_katex_check.js")
with open(js_path, "w", encoding="utf-8") as fh:
    fh.write(JS)

files = sorted(glob.glob(os.path.join(SRC, "*.md")))
try:
    out = subprocess.run(["node", js_path, VENDOR, json.dumps(files)],
                         capture_output=True, text=True, timeout=600)
finally:
    os.remove(js_path)

if out.returncode != 0:
    print("node 失败:\n" + out.stderr[-3000:])
    sys.exit(1)
try:
    r = json.loads(out.stdout.strip().split("\n")[-1])
except Exception:
    print("无法解析输出:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    sys.exit(1)

print("display %d 个 / inline %d 个 / 错误 %d 个" % (r["nDisp"], r["nInline"], len(r["bad"])))
seen = set()
for f, kind, tex, msg in r["bad"]:
    key = (f, msg[:60])
    if key in seen:
        continue
    seen.add(key)
    print("  %-32s %-7s %s" % (f, kind, tex.replace("\n", " ")))
    print("      → %s" % msg.replace("\n", " "))
if len(r["bad"]) > len(seen):
    print("  （另有 %d 条同类错误已折叠）" % (len(r["bad"]) - len(seen)))