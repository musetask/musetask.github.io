#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把指向原文的「原文 §x.y」死链改指到最近存在的锚点（逐级回退到父节）。
只在 src/*.md 上改写，不碰构建脚本。用法： python3 fix_links.py [--dry]
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ESL = os.path.join(os.path.dirname(HERE), "esl")
DRY = "--dry" in sys.argv

LINK_RE = re.compile(r'(<a class="src" href="\.\./esl/([^"#]+)#)([^"]+)(")')


def anchors(path):
    s = open(path, encoding="utf-8").read()
    return set(re.findall(r'id="([^"]+)"', s))


def resolve(anc, have):
    """死锚点回退到最深的、真实存在的祖先。"""
    if anc in have:
        return anc, None
    parts = anc.split("-")
    for k in range(len(parts) - 1, 0, -1):
        cand = "-".join(parts[:k])
        if cand in have:
            return cand, anc
    return None, anc


def main():
    total = 0
    for f in sorted(glob.glob(os.path.join(HERE, "src", "*.md"))):
        path = os.path.join(ESL, os.path.basename(f).split("-")[0] and "")
        s = open(f, encoding="utf-8").read()
        changed = 0
        cache = {}

        def rep(m):
            nonlocal changed
            tgt, anc = m.group(2), m.group(3)
            p = os.path.join(ESL, tgt)
            if not os.path.exists(p):
                return m.group(0)
            if tgt not in cache:
                cache[tgt] = anchors(p)
            have = cache[tgt]
            good, bad = resolve(anc, have)
            if good == anc or good is None:
                return m.group(0)
            changed += 1
            return m.group(1) + good + m.group(4)

        out = LINK_RE.sub(rep, s)
        if changed:
            total += changed
            print("%-34s 修正 %d 条" % (os.path.basename(f), changed))
            if not DRY:
                open(f, "w", encoding="utf-8").write(out)
    print("合计修正 %d 条死链%s" % (total, "（dry-run，未写入）" if DRY else ""))


if __name__ == "__main__":
    main()