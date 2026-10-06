#!/usr/bin/env python3
"""把 _src/*.body.html 里数学片段中的裸 < > & 转义成 &lt; &gt; &amp;。
按浏览器的方式逐个文本节点处理（数学不跨标签），因此与实际渲染一致。"""
import re, sys, os, glob

LEFT = [('\\[', '\\]', True), ('\\(', '\\)', False)]

def find_open(text, frm):
    i = frm
    while i < len(text):
        for l, r, d in LEFT:
            if text.startswith(l, i):
                return i, l, r, d
        if text[i] == '\\':
            i += 2; continue
        i += 1
    return -1, None, None, None

def find_close(text, frm, right):
    i, depth = frm, 0
    while i < len(text):
        if depth == 0 and text.startswith(right, i):
            return i
        c = text[i]
        if c == '\\':
            i += 2; continue
        if c == '{': depth += 1
        elif c == '}': depth = max(0, depth - 1)
        i += 1
    return -1

def escape_math_in_segment(seg):
    out, pos, n = [], 0, 0
    while True:
        o, l, r, d = find_open(seg, pos)
        if o == -1:
            out.append(seg[pos:]); break
        c = find_close(seg, o + len(l), r)
        if c == -1:
            out.append(seg[pos:o + len(l)]); pos = o + len(l); n += 1
            if n > 50: out.append(seg[pos:]); break
            continue
        out.append(seg[pos:o])
        body = seg[o + len(l):c]
        # 只转义"裸"的 & < >；已有实体（&lt; &#38; &amp; 等）保持不变
        body = re.sub(r'&(?!(?:#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);)', '&amp;', body)
        body = body.replace('<', '&lt;').replace('>', '&gt;')
        out.append(l + body + r)
        pos = c + len(r)
        n += 1
        if n > 500: out.append(seg[pos:]); break
    return ''.join(out)

# 只把真正像标签的东西当标签（后面必须跟字母 / ! ?），否则数学里的裸 < 会被误吞
TOKEN = re.compile(r'(<[a-zA-Z/!?][^>]*>)')

def process(path):
    src = open(path, encoding='utf-8').read()
    parts = TOKEN.split(src)
    changed = 0
    for i in range(0, len(parts), 2):
        seg = parts[i]
        if ('\\(' in seg) or ('\\[' in seg):
            new = escape_math_in_segment(seg)
            if new != seg:
                changed += 1
                parts[i] = new
    out = ''.join(parts)
    if out != src:
        open(path, 'w', encoding='utf-8').write(out)
    return changed

if __name__ == '__main__':
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_src', '*.body.html')))
    for f in files:
        c = process(f)
        print(f'{"fixed" if c else "  ok "}  {os.path.basename(f)}  ({c} 个文本节点)')