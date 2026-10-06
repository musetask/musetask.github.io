#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对账：src/*.md 里出现的 \\eqno{} 编号 vs 原书抽取的编号清单。

注意：第 14、18 章的原书 HTML 是第一版，编号体系不同，会被标为 SKIP。
用法： python3 coverage.py
"""
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
INV = "/private/var/folders/_h/hcgnzw6d0lb4lltt1d21klmm0000gn/T/opencode/eslmath"

FIRST_ED = {"14"}   # 第 18 章现已按第一版写，可与 .eqs 对账

# .eqs 是从 PDF 的 alt 文本抽的，会把正文里的「交叉引用」也当成公式，例如
# 第 6 章 alt 里有一句 "the similarity between the expansion (6.31) and the
# solution (5.50)"，(5.50) 其实属于第 5 章。这类条目已知为假阳性。
BENIGN = {("06", "5.50"), ("02", "2007"), ("05", "3.58")}

SLUG = {
    "02": "ch02-overview-of-supervised-learning",
    "03": "ch03-linear-methods-for-regression",
    "04": "ch04-linear-methods-for-classification",
    "05": "ch05-basis-expansions-and-regularization",
    "06": "ch06-kernel-smoothing-methods",
    "07": "ch07-model-assessment-and-selection",
    "08": "ch08-model-inference-and-averaging",
    "09": "ch09-additive-models-trees-and-related-methods",
    "10": "ch10-boosting-and-additive-trees",
    "11": "ch11-neural-networks",
    "12": "ch12-support-vector-machines-andflexible-discriminants",
    "13": "ch13-prototype-methods-and-nearest-neighbors",
    "14": "ch14-unsupervised-learning",
    "15": "ch15-random-forests",
    "16": "ch16-ensemble-learning",
    "17": "ch17-undirected-graphical-models",
    "18": "ch18-high-dimensional-problems-p-n",
}


def norm(t):
    t = t.strip().strip("()").strip()
    return t


def main():
    # 收集每章正文里出现过的编号（分片按前缀归属）
    have = {}
    for f in sorted(glob.glob(os.path.join(SRC, "*.md"))):
        name = os.path.basename(f)
        m = re.match(r"^(\d\d)", name)
        if not m:
            continue
        ch = m.group(1)
        if name.startswith("00"):
            continue
        text = open(f, encoding="utf-8").read()
        body = text.split("---", 2)[-1] if text.startswith("---") else text
        for t in re.findall(r"\\eqno\{([^}]*)\}", body):
            t = norm(t)
            if t:
                have.setdefault(ch, set()).add(t)

    total_missing = 0
    for ch in sorted(SLUG):
        inv_path = os.path.join(INV, SLUG[ch] + ".eqs")
        if not os.path.exists(inv_path):
            print("ch%s  清单缺失" % ch)
            continue
        want = set()
        for line in open(inv_path, encoding="utf-8"):
            head = line.split("\t")[0].strip()
            if re.match(r"^\((\d{1,2}\.[0-9]+)\)$", head):
                want.add(norm(head))
        got = have.get(ch, set())
        want = {t for t in want if (ch, t) not in BENIGN}
        missing = sorted(want - got, key=lambda s: [int(x) for x in s.split(".")])
        extra = sorted(got - want, key=lambda s: [int(x) for x in s.split(".")] if
                       re.match(r"^\d+\.\d+$", s) else [999])
        if ch in FIRST_ED:
            print("ch%-3s SKIP 原书本章是 ESL 1e，本站也按 1e 写（.eqs 清单不可信）"
                  " | 本站已有 %d 个编号" % (ch, len(got)))
            continue
        flag = "OK " if not missing else "!! "
        print("%sch%-3s 清单 %3d 个 | 已写 %3d | 缺 %2d %s%s"
              % (flag, ch, len(want), len(got), len(missing),
                 ("→ " + " ".join(missing)) if missing else "",
                 ("  | 多出 " + " ".join(extra[:12])) if extra else ""))
        total_missing += len(missing)
    print("\n合计缺 %d 个编号" % total_missing)


if __name__ == "__main__":
    main()