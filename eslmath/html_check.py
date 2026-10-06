#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建产物的静态体检（只读）。查三类历史上真实出现过的问题：

1. 公式块里的 LaTeX 没做 HTML 转义 —— `\\sum_{k<\\ell}` 会让浏览器把 `<l}` 当标签开头，
   整条公式被吞进属性，表现为「公式不显示」+「页面横向溢出」。
2. `.eq` 块嵌套（通常是 1 的次生症状）。
3. `<a class="src">` 指向的锚点在 ../esl/*.html 里不存在（死链）。

用法： python3 html_check.py
"""
import glob
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ESL = os.path.join(os.path.dirname(HERE), "esl")

EQ_RE = re.compile(r'<div class="escroll">\n\$\$\n(.*?)\n\$\$\n</div>', re.S)


def eq_nesting(body):
    """统计 .eq 是否嵌套。用一个栈记住每个未闭合 div 是不是 .eq。"""
    stack = []
    nested = 0
    for m in re.finditer(r'<div\b([^>]*)>|</div>', body):
        tag = m.group(0)
        if tag.startswith("</"):
            if stack:
                stack.pop()
            continue
        attrs = m.group(1) or ""
        cls = re.search(r'class="([^"]*)"', attrs)
        classes = cls.group(1).split() if cls else []
        is_eq = "eq" in classes
        if is_eq and any(x for x in stack):
            nested += 1
        stack.append(is_eq)
    return nested


def main():
    files = sorted(p for p in glob.glob(os.path.join(HERE, "*.html"))
                   if os.path.basename(p) not in ("index.html", "toc.html"))
    problems = 0
    for f in files:
        s = open(f, encoding="utf-8").read()
        name = os.path.basename(f)
        msgs = []

        # 1) 转义检查
        n = 0
        for m in EQ_RE.finditer(s):
            raw = m.group(1)
            if "<" in raw or ">" in raw:
                n += 1
                if n == 1:
                    msgs.append("公式块未转义 < >：%s" % re.sub(r"\s+", " ", raw[:60]))
        if n > 1:
            msgs.append("（共 %d 个公式块未转义）" % n)

        # 1b) \eqno 逃出公式块 —— 说明有 $$ 写在段落/列表项里，没被解析成公式块
        esc = [m.span() for m in re.finditer(r'<div class="escroll">', s)]
        stray = [m for m in re.finditer(r'\\eqno', s)
                 if not any(a < m.start() < b for a, b in esc)]
        if stray:
            txt = re.sub(r"<[^>]+>", "", s[stray[0].start() - 90:stray[0].start() + 40])
            msgs.append("%d 个 \\eqno 逃到公式块之外（公式被写在段落/列表项里）：…%s"
                        % (len(stray), re.sub(r"\s+", " ", txt)))

        # 2) 嵌套
        nest = eq_nesting(s)
        if nest:
            msgs.append("有 %d 个嵌套的 .eq" % nest)

        # 2b) 非法 href
        #     LaTeX 里的 [..](..)（矩阵、\frac 参数）被 Markdown 当成链接就是这么来的，
        #     会产生既不存在、又指向不存在文件的可点击坏链。
        #     注意：行内 $ 残留**不能**在这里查 —— 产物 HTML 里本来就含 $，
        #     KaTeX 是运行时渲染的；那条检查只能在浏览器里做。
        for m in re.finditer(r'<a href="([^"]+)"', s):
            h = m.group(1)
            path = h.partition("#")[0]
            if not path:
                continue                      # 纯页内锚点
            if path.startswith(("http", "../", "./", "mailto:")):
                continue
            if re.search(r"\.[a-z0-9]{1,5}$", path):
                if not os.path.exists(os.path.join(HERE, path)):
                    msgs.append("链接指向不存在的文件：%s" % path)
                continue
            msgs.append("非法 href（多半是公式里的 [..](..) 被当成链接）：%s" % h[:60])
            break

        # 3) 死链
        dead = []
        for m in re.finditer(r'<a class="src" href="\.\./esl/([^"#]+)#([^"]+)"', s):
            tgt, anc = m.group(1), m.group(2)
            path = os.path.join(ESL, tgt)
            if not os.path.exists(path):
                dead.append("%s 不存在" % tgt)
            else:
                body = open(path, encoding="utf-8").read()
                if ('id="%s"' % anc) not in body:
                    dead.append("%s#%s" % (tgt, anc))
        if dead:
            msgs.append("死链 %d 条：%s" % (len(dead), "、".join(sorted(set(dead))[:6])))

        eqn = len(EQ_RE.findall(s))
        print("%s%-12s 公式块 %4d%s"
              % ("OK " if not msgs else "!! ", name, eqn,
                 "" if not msgs else ""))
        for x in msgs:
            print("     - " + x)
        problems += bool(msgs)
    print("\n%s" % ("全部通过" if not problems else "%d 个页面有待修问题" % problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())