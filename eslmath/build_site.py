#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eslmath — ESL 公式推导手册构建脚本

  src/NN-slug.md  (front matter + Markdown 子集 + KaTeX 数学)
        │
        ├─ md → html 转换（自定义极小子集，见 md_to_html）
        └─ 套上与 esl 站点一致的外壳（顶栏 / 底部章节导航 / 进度条 / 深色模式）

用法：
    python3 build_site.py            # 构建全部
    python3 build_site.py m03        # 只构建某页（调试用）
"""

import glob
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
SITE_TITLE = "ESL 数学推导"
BOOK = "The Elements of Statistical Learning"

# --------------------------------------------------------------------------
# front matter
# --------------------------------------------------------------------------
FM_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)


INCLUDE_RE = re.compile(r"^[ \t]*<!--\s*include\s+([^\s]+?)\s*-->[ \t]*$", re.M)


def expand_includes(text, depth=0):
    """把 <!--include foo.md --> 就地展开成 foo.md 的内容（相对当前 md 文件）。"""
    if depth > 5:
        raise SystemExit("include 嵌套过深")
    if not INCLUDE_RE.search(text):
        return text

    def rep(m):
        path = m.group(1)
        full = path if os.path.isabs(path) else os.path.join(SRC, path)
        if not os.path.exists(full):
            print("  ! include 缺失，先跳过: %s" % path)
            return ""
        return expand_includes(open(full, encoding="utf-8").read(), depth + 1)

    return INCLUDE_RE.sub(rep, text)


def parse_fm(text):
    m = FM_RE.match(text)
    if not m:
        raise SystemExit("missing front matter (--- ... ---)")
    body = text[m.end():]
    fm = {}
    order = []
    key = None
    for line in m.group(1).split("\n"):
        if not line.strip():
            continue
        km = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
        if km:
            key = km.group(1)
            fm[key] = km.group(2).strip().strip('"').strip("'")
            order.append(key)
        elif key:                      # 续行（列表）
            fm[key] += "\n" + line.strip()
    return fm, body


# --------------------------------------------------------------------------
# inline markdown
# --------------------------------------------------------------------------
INLINE_MATH_RE = re.compile(r"\$[^$\n]{1,600}?\$")

# 全局的行内公式暂存池；每次构建一个文件时由 main() 清空。
MATH_STASH = []


def protect_math(s, stash):
    r"""把 `$...$` 行内公式换成占位符，让 Markdown 规则碰不到它。

    这一步是必须的：LaTeX 里大量使用 `_`（下标）和 `[…](…)` 结构
    （矩阵、\frac 的参数、撇号差分），而 Markdown 会把 `_` 当斜体、
    把 `[a](b)` 当链接，直接在公式里做替换会把公式改坏
    （例如 (K^{-1}+\lambda I)^{-1} 会被吃成链接，$x_i$ 的下标会消失）。
    """
    def rep(m):
        stash.append(m.group(0))
        # 用 \x01 而不是 \x00：\x00 是 Markdown 规则自己的占位符，两者会撞
        return "\x01%d\x01" % (len(stash) - 1)

    return INLINE_MATH_RE.sub(rep, s)


def inline(s):
    r"""行内标记：代码 / 粗体 / 斜体 / 链接。

    `$...$` 行内公式先被保护起来（放进全局 MATH_STASH），不会被任何 Markdown 规则改写。
    """
    stash = []

    def keep(t):
        stash.append(t)
        return "\x00%d\x00" % (len(stash) - 1)

    # 顺序很重要：先摘出行内代码（`...`），再保护数学。
    # 反过来的话，`$p$` 这种「代码里写公式」的写法会被数学保护先吃掉，
    # 留下一个孤立的 $ —— 例如 prereq 里用反引号标注的 p 值记号。
    s = re.sub(r"`([^`]+)`", lambda m: keep("<code>%s</code>" % m.group(1)), s)

    math_stash = MATH_STASH
    s = protect_math(s, math_stash)

    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
               lambda m: keep('<a href="%s">%s</a>' % (m.group(2), m.group(1))), s)
    s = re.sub(r"\*\*([^*]+)\*\*", lambda m: keep("<strong>%s</strong>" % m.group(1)), s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", lambda m: keep("<em>%s</em>" % m.group(1)), s)

    def back(m):
        return stash[int(m.group(1))]

    s = re.sub(r"\x00(\d+)\x00", back, s)
    # 还原时做 HTML 转义：行内公式里同样会出现 < > &（如 $a<b$、$\alpha_t\to0$），
    # 不转义会被浏览器当成标签起始符，把整条公式连同后面正文一起吞进属性。
    s = re.sub(r"\x01(\d+)\x01",
               lambda m: html.escape(math_stash[int(m.group(1))], quote=False), s)
    return s


# --------------------------------------------------------------------------
# block level
# --------------------------------------------------------------------------
HEAD_RE = re.compile(r"^(#{1,4})\s+(.*)$")
ID_RE = re.compile(r"\s*\{#([A-Za-z0-9_-]+)\}\s*$")
EQNO_RE = re.compile(r"\\eqno\{([^}]*)\}")


def slug_to_id(s):
    s = s.strip().lower()
    s = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", s)
    return s.strip("-")


def math_block(raw, tag):
    """$$ ... $$ → 可横向滚动的公式块 + 右侧编号。"""
    # 连同 \eqno{} 前面作为 LaTeX 空格用的 "\ " 一起去掉，否则会留下悬空反斜杠
    raw = re.sub(r"\\?\s*\\eqno\{[^}]*\}", "", raw)
    raw = EQNO_RE.sub("", raw).strip()
    num = ""
    if tag:
        num = tag.group(1).strip()
    span = '<span class="tag">(%s)</span>' % html.escape(num) if num else ""
    cls = "eq"
    if len(raw) > 220:
        cls = "eq verylong"
    elif len(raw) > 110:
        cls = "eq long"
    # 必须转义：LaTeX 里常见 < > &（如 \sum_{k<\ell}、cases 的列分隔 &），
    # 不转义会被浏览器当成标签起始符，把整条公式吞进属性里。
    # 浏览器解码后 textContent 仍是原始字符，KaTeX 看到的是对的。
    safe = html.escape(raw, quote=False)
    return ('<div class="%s"><div class="escroll">\n$$\n%s\n$$\n</div>%s</div>'
            % (cls, safe, span))


LIST_ITEM_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")


def parse_list(lines, i):
    """从第 i 行开始解析（可嵌套的）列表，返回 (html, next_index)。"""
    base = len(lines[i]) - len(lines[i].lstrip())
    ordered = bool(re.match(r"^\s*\d+\.\s+", lines[i]))
    items = []            # 每个元素：("p", text) 或 ("list", html)

    def item_text(idx):
        """把一个条目从 idx 起吃到下一个同级/更高级标记为止。"""
        buf = [LIST_ITEM_RE.match(lines[idx]).group(3).strip()]
        idx += 1
        while idx < len(lines):
            cur = lines[idx]
            if not cur.strip():
                break
            m = LIST_ITEM_RE.match(cur)
            if m:
                break
            buf.append(cur.strip())
            idx += 1
        return " ".join(buf), idx

    while i < len(lines):
        cur = lines[i]
        if not cur.strip():
            # 空行后若仍是同级或更深列表则继续，否则结束
            if i + 1 < len(lines) and LIST_ITEM_RE.match(lines[i + 1]) \
                    and len(lines[i + 1]) - len(lines[i + 1].lstrip()) >= base:
                i += 1
                continue
            break
        m = LIST_ITEM_RE.match(cur)
        if not m:
            break
        ind = len(m.group(1))
        if ind < base:
            break
        if ind > base:
            # 更深：交给递归
            sub, i = parse_list(lines, i)
            items.append(("list", sub))
            continue
        is_ord = m.group(2)[0].isdigit()
        if is_ord != ordered:
            break
        txt, i = item_text(i)
        items.append(("p", txt))

        def li_content(txt):
            """渲染一个列表项的内容。

            条目的续行里可能夹着缩进的 $$ ... $$ 公式块。写作约定要求把公式
            提到顶层，但作者经常写在列表项里；这些块必须真的变成 <div class="eq">，
            否则 \\eqno{} 不会被剥掉、会在页面上显示成字面量「\\eqno」，
            而且长公式会撑宽整个页面。
            """
            if "$$" not in txt:
                return inline(txt)
            chunks = []
            pos = 0
            for m in re.finditer(r"\$\$([\s\S]*?)\$\$", txt):
                lead = txt[pos:m.start()].strip()
                if lead:
                    chunks.append("<span>%s</span>" % inline(lead))
                body = m.group(1).strip()
                if body:
                    chunks.append(math_block(EQNO_RE.sub("", body), EQNO_RE.search(body)))
                pos = m.end()
            tail = txt[pos:].strip()
            if tail:
                chunks.append("<span>%s</span>" % inline(tail))
            return "".join(chunks)

    parts = []
    for kind, val in items:
        parts.append(val if kind == "list" else li_content(val))
    tag = "ol" if ordered else "ul"
    htmlout = "<%s>%s</%s>" % (tag, "".join("<li>%s</li>" % p for p in parts), tag)
    return htmlout, i


def md_to_html(md, page_id):
    lines = md.split("\n")
    out = []
    i = 0
    n = len(lines)
    used_ids = set()

    def uniq(base):
        cand = base
        k = 2
        while cand in used_ids:
            cand = "%s-%d" % (base, k)
            k += 1
        used_ids.add(cand)
        return cand

    def flush_para(buf):
        r"""把已积累的段落行输出成 HTML。

        段落里可能夹着「行内的 $$ ... $$」（作者没把公式另起一行）。
        这时必须把它拆成独立的公式块，否则 KaTeX 会把它当行内公式，
        而且 \eqno{} 不会被剥掉、会在页面上显示成字面量「\eqno」。
        """
        if not buf:
            return
        text = " ".join(buf).strip()
        del buf[:]
        if "$$" not in text:
            out.append("<p>%s</p>" % inline(text))
            return
        pos = 0
        for m in re.finditer(r"\$\$([\s\S]*?)\$\$", text):
            lead = text[pos:m.start()].strip()
            if lead:
                out.append("<p>%s</p>" % inline(lead))
            body = m.group(1).strip()
            if body:
                tag = EQNO_RE.search(body)
                out.append(math_block(EQNO_RE.sub("", body), tag))
            pos = m.end()
        tail = text[pos:].strip()
        if tail:
            out.append("<p>%s</p>" % inline(tail))

    para = []

    while i < n:
        ln = lines[i]

        # fenced code
        if ln.startswith("```"):
            flush_para(para)
            lang = ln[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = "listing"
            out.append('<pre class="%s"><code>%s</code></pre>'
                       % (cls, html.escape("\n".join(buf))))
            continue

        # display math：$$ ... $$，可跨行，可同行，用 \eqno{} 给编号
        if ln.strip().startswith("$$"):
            flush_para(para)
            rest = ln.strip()[2:]
            if rest.endswith("$$") and rest[:-2].count("$$") == 0 and len(rest) > 2:
                # 单行 $$ ... $$
                m = EQNO_RE.search(rest)
                out.append(math_block(EQNO_RE.sub("", rest[:-2]), m))
                i += 1
                continue
            buf = [] if rest.strip() == "" else [rest]
            i += 1
            while i < n:
                cur = lines[i].strip()
                if cur == "$$":
                    i += 1
                    break
                if cur.endswith("$$"):
                    buf.append(cur[:-2])
                    i += 1
                    break
                buf.append(lines[i])
                i += 1
            raw = "\n".join(buf)
            m = EQNO_RE.search(raw)
            out.append(math_block(EQNO_RE.sub("", raw), m))
            continue

        # blockquote → aside box   > **基础知识** · 标题
        if ln.startswith(">"):
            flush_para(para)
            buf = []
            while i < n and lines[i].startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            head = ""
            rest = list(buf)
            if buf:
                first = buf[0].strip()
                mm = re.match(r"^\*\*(.+?)\*\*(.*)$", first)
                if mm:
                    kind = mm.group(1).strip()
                    tail = mm.group(2).strip()
                    # 形如 "**基础知识** · 标题"  → 标题只是标题
                    if tail.startswith(("·", "|", "：", ":")):
                        sub = tail.lstrip("·|:： ").strip()
                        head = kind + (" · " + sub if sub else "")
                        rest = buf[1:]
                        # 标题太长（超过 24 字）时把它当正文，标题留空
                        if len(sub) > 24:
                            head = kind
                            rest = [sub] + buf[1:]
                    elif not tail:
                        head = kind
                        rest = buf[1:]
                    else:
                        head = kind
                        rest = [tail] + buf[1:]
                else:
                    # 无标签的引用 = 普通段落（导语）
                    rest = buf
            if not head:
                out.append(md_to_html("\n".join(rest).strip() + "\n", page_id))
                continue
            kind_slug = {"基础知识": "basis", "结果": "key", "坑": "warn",
                         "数值": "calc", "延伸": "", "记号": ""}.get(
                             head.split(" ")[0].split("·")[0].strip(), "")
            cls = "aside %s" % kind_slug if kind_slug else "aside"
            hd = '<p class="hd">%s</p>' % inline(head) if head else ""
            inner = md_to_html("\n".join(rest).strip() + "\n", page_id) if rest else ""
            out.append('<div class="%s">%s%s</div>' % (cls, hd, inner))
            continue

        # heading
        m = HEAD_RE.match(ln)
        if m:
            flush_para(para)
            level = len(m.group(1))
            text = m.group(2).strip()
            mid = ID_RE.search(text)
            if mid:
                hid = mid.group(1)
                text = ID_RE.sub("", text)
            else:
                base = page_id[1:] if page_id.startswith("m") else "0"
                bare = re.sub(r"^[0-9]+(\.[0-9]+)*\s+", "", text)
                hid = uniq("s-%s" % base) if level == 1 else uniq(slug_to_id(bare))
            if hid in used_ids and level > 1:
                hid = uniq(hid)
            used_ids.add(hid)
            num = ""
            nm = re.match(r"^([0-9]+(?:\.[0-9]+)*(?:\.[0-9]+)?|[0-9]+)\s+(.*)$", text)
            if nm and level <= 3:
                num = '<span class="num">%s</span> ' % html.escape(nm.group(1))
                text = nm.group(2)
            cls = "anchored"
            if level == 1:
                cls = "ch anchored"
            out.append('<h%d id="%s" class="%s">%s%s</h%d>'
                       % (level, hid, cls, num, inline(text), level))
            i += 1
            continue

        # table
        if ln.startswith("|") and i + 1 < n and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            flush_para(para)
            head_cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < n and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            th = "".join("<th>%s</th>" % inline(c) for c in head_cells)
            body = []
            for r in rows:
                body.append("<tr>%s</tr>" % "".join("<td>%s</td>" % inline(c) for c in r))
            out.append('<div class="tscroll-x"><table class="fx"><thead><tr>%s</tr></thead>'
                       '<tbody>%s</tbody></table></div>' % (th, "".join(body)))
            continue

        # lists
        if re.match(r"^\s*([-*]|\d+\.)\s+", ln):
            flush_para(para)
            frag, i = parse_list(lines, i)
            out.append(frag)
            continue

        # blank
        if not ln.strip():
            flush_para(para)
            i += 1
            continue

        # horizontal rule
        if re.match(r"^-{3,}$", ln.strip()):
            flush_para(para)
            out.append("<hr>")
            i += 1
            continue

        para.append(ln.strip())
        i += 1

    flush_para(para)
    return "\n".join(out)


# --------------------------------------------------------------------------
# page shell
# --------------------------------------------------------------------------
def render(fm, body_html):
    n = fm.get("n", "")
    title = fm.get("title", "")
    en = fm.get("title_en", "")
    page = title + (" · " + en if en else "")
    desc = fm.get("desc", "")

    def nav_link(key, cls, lbl, nm):
        u = fm.get(key)
        if not u:
            return '<span class="%s void"></span>' % cls if cls else ""
        return ('<a class="%s" href="%s.html"><span class="lbl">%s</span>'
                '<span class="nm">%s</span></a>' % (cls, u, lbl, nm))

    nav = []
    if fm.get("prev"):
        nav.append('<a class="prev" href="%s.html"><span class="lbl">← 上一章</span>'
                   '<span class="nm">%s</span></a>' % (fm["prev"], fm.get("prev_title", "上一页")))
    else:
        nav.append('<a class="prev" href="index.html"><span class="lbl">Home</span>'
                   '<span class="nm">封面</span></a>')
    if fm.get("next"):
        nav.append('<a class="next" href="%s.html"><span class="lbl">下一章 →</span>'
                   '<span class="nm">%s</span></a>' % (fm["next"], fm.get("next_title", "下一页")))

    return """<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="description" content="%(desc)s">
<title>%(page)s · %(site)s</title>
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<link rel="stylesheet" href="assets/style.css">
<script defer src="../assets/katex/katex.min.js"></script>
<script defer src="../assets/katex/auto-render.min.js"></script>
<script defer src="assets/app.js"></script>
</head>
<body>
<a class="skip" href="#main">跳到正文</a>
<div class="progress" id="progress"></div>
<header class="bar">
  <button class="icon" id="themeToggle" type="button" aria-label="切换深色模式">&#9681;</button>
  <a class="title" href="index.html">%(site)s</a>
  <a class="nav" href="toc.html">目录</a>
</header>
<main id="main">
%(body)s
</main>
<nav class="chapnav"><div class="inner">
%(nav)s
</div></nav>
</body>
</html>
""" % dict(site=SITE_TITLE, page=html.escape(page), desc=html.escape(desc),
           body=body_html, nav="\n".join(nav))


def render_toc(pages):
    rows = []
    for p in pages:
        secs = "".join(
            '<li class="l2"><a href="%s.html#%s"><span class="n">%s</span>'
            '<span class="t">%s</span></a></li>'
            % (p["id"], s["id"], html.escape(s["num"] or "·"),
               html.escape(re.sub(r"[`*]", "", s["title"])))
            for s in p["sections"][1:])
        rows.append('<li class="l1"><a href="%s.html"><span class="n">%s</span>'
                    '<span class="t">%s</span></a></li>' % (p["id"], html.escape(p["n"]),
                                                            html.escape(p["title"])))
        rows.append(secs)
    return """<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>目录 · %(site)s</title>
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<link rel="stylesheet" href="assets/style.css">
<script defer src="../assets/katex/katex.min.js"></script>
<script defer src="../assets/katex/auto-render.min.js"></script>
<script defer src="assets/app.js"></script>
</head>
<body>
<a class="skip" href="#main">跳到正文</a>
<div class="progress" id="progress"></div>
<header class="bar">
  <button class="icon" id="themeToggle" type="button" aria-label="切换深色模式">&#9681;</button>
  <a class="title" href="index.html">%(site)s</a>
  <a class="nav" href="toc.html">目录</a>
</header>
<main id="main">
<h1 id="s-toc" class="ch"><span class="num">Contents</span> 目录</h1>
<p class="lede">按原书章节顺序排列，每一节都给出「基础知识 → 推导 → 结果」的完整链条。公式编号与 <a href="../esl/index.html">ESL 原文</a> 一致。</p>
<nav class="chapjump">%(jump)s</nav>
<ol class="toc">
%(rows)s
</ol>
</main>
<nav class="chapnav"><div class="inner">
<a class="prev" href="index.html"><span class="lbl">Home</span><span class="nm">封面</span></a>
<a class="next" href="%(first)s.html"><span class="lbl">开始 →</span><span class="nm">数学预备知识</span></a>
</div></nav>
</body>
</html>
""" % dict(site=SITE_TITLE, rows="\n".join(rows),
           jump="".join('<a href="%s.html">%s</a>' % (p["id"], html.escape(p["n"])) for p in pages),
           first=pages[0]["id"])


INDEX_CARDS_HEAD = "<!--CARDS-->"
INDEX_CARDS_TAIL = "<!--/CARDS-->"


def render_index(pages, totals):
    """封面页。章节卡片由已构建的页面自动生成，锚点不会和正文脱节。"""
    p = os.path.join(HERE, "index.html")
    old = ""
    if os.path.exists(p):
        old = open(p, encoding="utf-8").read()
    if INDEX_CARDS_HEAD not in old:
        return False

    cards = []
    for pg in pages:
        subs = []
        for s in pg["sections"][1:7]:
            lab = re.sub(r"[*`]", "", s["title"]).strip()
            if not lab or lab == "§":
                continue
            subs.append('<li><a href="%s.html#%s">%s</a></li>'
                        % (pg["id"], s["id"], html.escape(
                            (s["num"] + " ") if s["num"] else "") + lab))
        sub_html = ('<ul class="subs">%s</ul>' % "".join(subs)) if subs else ""
        cards.append(
            '<li class="card"><a class="chead" href="%s.html">'
            '<span class="cnum">%s</span><span class="cname">%s</span>'
            '<span class="cpg">%s</span></a>%s</li>'
            % (pg["id"], html.escape(pg["n"]), html.escape(pg["title"]),
               html.escape(pg.get("title_en", "")), sub_html))

    stats = ("%d 章 · 约 %s 个公式 · 每式给出「基础知识 → 逐步推导 → 结果与几何/概率解释」"
             % (totals["chapters"], totals["eqs"]))
    body = ('\n'.join(cards)).replace("&#39;", "'")
    # 用 lambda 做替换，避免公式里的反斜杠被当成 re.sub 的替换模板转义
    new = re.sub(re.escape(INDEX_CARDS_HEAD) + ".*?" + re.escape(INDEX_CARDS_TAIL),
                 lambda m: INDEX_CARDS_HEAD + "\n" + body + "\n" + INDEX_CARDS_TAIL,
                 old, flags=re.S)
    new = re.sub(r'<p class="stats">.*?</p>',
                 lambda m: '<p class="stats">' + stats + "</p>", new, flags=re.S)
    nxt = ('<a class="next" href="%s.html"><span class="lbl">下一章 →</span>'
           '<span class="nm">%s</span></a>'
           % (pages[0]["id"], html.escape(pages[0]["title"])))
    new = re.sub(r'<a class="next" href="[^"]*\.html"><span class="lbl">下一章 →</span>'
                 r'<span class="nm">[^<]*</span></a>',
                 lambda m: nxt, new)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(new)
    print("wrote index.html")
    return True


HEAD_RE2 = re.compile(r'^<h([123]) id="([^"]+)"[^>]*>(.*?)</h\1>$', re.M)


NUM_SPAN_RE = re.compile(r'<span class="num">(.*?)</span>')


def collect_sections(body_html, levels=(1, 2)):
    """从已生成的 HTML 里抽出 (id, 节号, 标题)。节号取自标题里的 .num 标记。"""
    secs = []
    for m in HEAD_RE2.finditer(body_html):
        if int(m.group(1)) not in levels:
            continue
        inner = m.group(3)
        num = ""
        mn = NUM_SPAN_RE.search(inner)
        if mn:
            num = re.sub(r"<[^>]+>", "", mn.group(1)).strip()
            inner = inner.replace(mn.group(0), "")
        title = re.sub(r"<[^>]+>", "", inner)
        title = re.sub(r"\s+", " ", title).strip()
        if not num:
            mm = re.match(r"^([0-9]+(?:\.[0-9]+)*)\s+(.*)$", title)
            if mm:
                num, title = mm.group(1), mm.group(2).strip()
        if not title:
            continue
        secs.append({"id": m.group(2), "num": num, "title": title})
    return secs


# --------------------------------------------------------------------------
def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    all_md = sorted(glob.glob(os.path.join(SRC, "*.md")))
    # 分片文件通过 <!--include --> 拼进外壳，不单独成页
    fragments = set()
    for f in all_md:
        fragments.update(os.path.basename(x) for x in INCLUDE_RE.findall(
            open(f, encoding="utf-8").read()))
    files = [f for f in all_md if os.path.basename(f) not in fragments]
    pages = []
    for f in files:
        text = open(f, encoding="utf-8").read()
        if not text.startswith("---"):
            print("  ! %s 还没有 front matter，先跳过" % os.path.basename(f))
            continue

        fm, body = parse_fm(expand_includes(text))
        pid = fm.get("id") or os.path.splitext(os.path.basename(f))[0]
        if only and pid != only:
            # 仍然需要 sections 以生成目录
            pages.append(dict(id=pid, n=fm.get("n", ""), title=fm.get("title", ""),
                              title_en=fm.get("title_en", ""),
                              sections=collect_sections(md_to_html(body, pid))))
            continue
        del MATH_STASH[:]
        html_body = md_to_html(body, pid)
        with open(os.path.join(HERE, pid + ".html"), "w", encoding="utf-8") as fh:
            fh.write(render(fm, html_body))
        print("wrote", pid + ".html")
        secs = [s for s in collect_sections(html_body) if s["title"] != "§" or s["num"]]
        secs = [s for s in secs if len(s["title"]) > 1 or s["num"]]
        pages.append(dict(id=pid, n=fm.get("n", ""), title=fm.get("title", ""),
                          title_en=fm.get("title_en", ""), sections=secs))
    with open(os.path.join(HERE, "toc.html"), "w", encoding="utf-8") as fh:
        fh.write(render_toc(pages))
    print("wrote toc.html")
    # 统计要算上所有分片，否则会漏掉大半公式
    eqs = sum(len(re.findall(r"\\eqno\{", open(f, encoding="utf-8").read()))
              for f in all_md)
    render_index(pages, {"chapters": max(0, len(pages) - 1), "eqs": "{:,}".format(eqs)})


if __name__ == "__main__":
    main()