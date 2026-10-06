#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读校验：把 src/NN-*.md 跑一遍转换，检查是否留下未解析的标记。
用法： python3 check.py 03-linear-regression.md [...]   （不给参数则检查全部）
不会写任何文件。"""

import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_site as B

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")

CJK = re.compile(r"[\u4e00-\u9fff]")


def main():
    args = sys.argv[1:]
    files = [os.path.join(SRC, a) if not os.path.sep in a else a for a in args] \
        if args else sorted(glob.glob(os.path.join(SRC, "*.md")))
    bad = 0
    for f in files:
        text = open(f, encoding="utf-8").read()
        pid = os.path.splitext(os.path.basename(f))[0]
        if text.startswith("---"):
            fm, body = B.parse_fm(text)
            pid = fm.get("id") or pid
            is_shell = True
        else:
            # 分片：没有 front matter，也没有一级标题
            body, is_shell = text, False
        try:
            h = B.md_to_html(body, pid)
        except Exception as e:
            print("!! %s: 转换异常 %r" % (os.path.basename(f), e))
            bad += 1
            continue
        msgs = []
        if h.count("$$") % 2:
            msgs.append("$$ 不成对")
        for pat, why in [(r"<p>[\s\S]{0,200}?^\s*>", "blockquote 未闭合"),
                         (r"<p>[^<]*\*\*", "** 未解析"),
                         (r"<li>[^<]*\*\*", "列表里 ** 未解析"),
                         (r"\\begin\{(cases|aligned)\}", ""),
                         (r"\|\s*\|", "")]:
            m = re.search(pat, h, re.M)
            if m and why:
                msgs.append("%s: …%s…" % (why, h[max(0, m.start() - 40):m.end() + 40]
                                           .replace("\n", " ")))
        # begin/end 配对
        for env in set(re.findall(r"\\begin\{(\w+\*?)\}", h)):
            if len(re.findall(r"\\begin\{%s\}" % re.escape(env), h)) != \
               len(re.findall(r"\\end\{%s\}" % re.escape(env), h)):
                msgs.append("\\begin{%s} 与 \\end 不配对" % env)
        nl = len(re.findall(r"\\left(?![a-zA-Z])", h))
        nr = len(re.findall(r"\\right(?![a-zA-Z])", h))
        if nl != nr:
            msgs.append("\\left 与 \\right 不配对 (%d vs %d)" % (nl, nr))
        # 每个 \\eqno 里的编号格式
        for t in re.findall(r"\\eqno\{([^}]*)\}", body):
            if not re.match(r"^[0-9]+[.\-0-9A-Za-z()]*$", t.strip()):
                msgs.append("eqno 编号可疑: %r" % t)
        # 必备标题 id（分片由外壳承担一级标题）
        if is_shell and not re.search(r'^#\s+\S+.*\{#s-', body, re.M):
            msgs.append("一级标题缺少 {#id}")
        h2 = re.findall(r"^##\s+(.*)$", body, re.M)
        if h2 and not all("{#" in x for x in h2):
            msgs.append("有二级标题缺 {#id}: %s"
                        % [x for x in h2 if "{#" not in x][:3])
        if not is_shell and re.search(r'^#\s+', body, re.M):
            msgs.append("分片不应写一级标题（外壳已有）")
        words = len(CJK.findall(body))
        eqs = len(re.findall(r"\$\$", body)) // 2
        asides = body.count("> **")
        flag = "OK " if not msgs else "!! "
        print("%s%-34s 汉字 %5d  公式块 %3d  提示框 %3d" % (flag, os.path.basename(f),
                                                           words, eqs, asides))
        for x in msgs:
            print("     - " + x)
        bad += bool(msgs)
    print("\n%s" % ("全部通过" if not bad else "%d 个文件有待修问题" % bad))


if __name__ == "__main__":
    main()