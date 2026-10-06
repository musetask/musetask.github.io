#!/usr/bin/env python3
"""组装 thellmmath 站点：读取 _src/*.body.html + meta，生成完整页面与目录。"""
import json, os, html, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, '_src')
TITLE = 'LLM 公式推导手册'
SUB = '《Large Language Models》公式推导手册'

# (slug, 章节标签, 标题, 副标题/描述, 公式数)
PAGES = [
 ('00-prereq',  '预备知识', '数学预备知识速查',
  '推导前需要重新捡起来的大学数学：信息论、概率统计、微积分、线代与最优化。', None),
 ('01-ch02-scaling', 'Chapter 2', '第 2 章 · 缩放定律',
  'Scaling Laws：幂律拟合、熵与 KL 分解、拉格朗日乘子求最优算力分配。', '2.1–2.5'),
 ('02-ch04-data', 'Chapter 4', '第 4 章 · 数据准备',
  'BPE 合并分数、UTF-8 压缩率、去重与数据配比中的统计量。', '4.1–4.2'),
 ('03-ch05-arch-a', 'Chapter 5 · 1/3', '第 5 章 · 模型架构（1/3）',
  'Transformer 主干：嵌入、注意力、Softmax、LayerNorm、FFN 与残差。', '5.1–5.16'),
 ('04-ch05-arch-b', 'Chapter 5 · 2/3', '第 5 章 · 模型架构（2/3）',
  '位置编码、归一化家族、激活函数、长上下文外推。', '5.17–5.32'),
 ('05-ch05-arch-c', 'Chapter 5 · 3/3', '第 5 章 · 模型架构（3/3）',
  'MoE 路由、注意力变体、Mamba 状态空间模型。', '5.33–5.46'),
 ('06-ch06-pt-a', 'Chapter 6 · 1/2', '第 6 章 · 预训练（1/2）',
  '预训练目标函数、训练配置、优化器与学习率调度。', '6.1–6.8'),
 ('07-ch06-pt-b', 'Chapter 6 · 2/2', '第 6 章 · 预训练（2/2）',
  '计算与通信开销、缩放与并行策略、训练稳定性。', '6.9–6.15'),
 ('08-ch07-sft', 'Chapter 7', '第 7 章 · 指令微调',
  '指令数据构造、模板、LoRA 低秩微调与量化。', '7.1–7.4'),
 ('09-ch08-align-a', 'Chapter 8 · 1/3', '第 8 章 · 人类对齐（1/3）',
  '偏好数据、奖励模型、Bradley–Terry 与 DPO 目标。', '8.1–8.16'),
 ('10-ch08-align-b', 'Chapter 8 · 2/3', '第 8 章 · 人类对齐（2/3）',
  'RLHF：PPO 的裁剪目标、KL 惩罚、价值函数与 GAE。', '8.17–8.32'),
 ('11-ch08-align-c', 'Chapter 8 · 3/3', '第 8 章 · 人类对齐（3/3）',
  '其他对齐算法：SLiC、RRHF、PPO 的变体与多目标偏好。', '8.33–8.46'),
 ('12-ch09-dec-a', 'Chapter 9 · 1/2', '第 9 章 · 解码与部署（1/2）',
  '贪心/采样/束搜索/束采样、nucleus 与 top-k 截断。', '9.1–9.10'),
 ('13-ch09-dec-b', 'Chapter 9 · 2/2', '第 9 章 · 解码与部署（2/2）',
  '投机解码、KV Cache、量化、并行推理与服务吞吐。', '9.11–9.20'),
 ('14-ch10-prompt', 'Chapter 10', '第 10 章 · 提示工程',
  '上下文学习 ICL 的形式化、示例选择的统计视角与 RAG 的条件概率解释。', '10.1'),
 ('15-ch12-eval-a', 'Chapter 12 · 1/2', '第 12 章 · 评测（1/2）',
  '自回归分解、困惑度的两种写法与通用评测指标的数学基础。', '12.1–12.8'),
 ('16-ch12-eval-b', 'Chapter 12 · 2/2', '第 12 章 · 评测（2/2）',
  'BLEU / ROUGE 的 n-gram 与 LCS 构造、Pass@k 无偏估计、Elo 与 Bradley–Terry。', '12.9–12.15'),
]

HEAD = '''<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>{title} · {sub}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='7' fill='%23a11f27'/%3E%3Ctext x='16' y='23' font-size='20' font-family='Georgia,serif' fill='%23fff' text-anchor='middle'%3E%E2%88%91%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<link rel="stylesheet" href="style.css">
<script>(function(){{try{{var t=localStorage.getItem('thellmmath-theme');document.documentElement.setAttribute('data-theme',t==='dark'?'dark':'light');}}catch(e){{}}}})();</script>
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<div class="progress" id="progress"></div>
<header class="topbar">
  <a class="btn idxbtn" href="index.html" aria-label="目录"><span aria-hidden="true">&#9776;</span> <span class="lbl">目录</span></a>
  <a class="brand" href="index.html">{sub}</a>
  <button class="btn" id="themeToggle" aria-label="切换深色模式">&#9788;</button>
</header>
<main id="main">
<article>
'''

TAIL = '''</article>
</main>
<nav class="chapnav" aria-label="上下页导航"><div class="chapnav-inner">
{nav}
</div></nav>
<script src="../assets/katex/katex.min.js"></script>
<script src="../assets/katex/auto-render.min.js"></script>
<script src="app.js"></script>
</body>
</html>
'''

def nav_html(i):
    def item(j, cls, lbl):
        slug, chno, title, desc, rng = PAGES[j]
        return (f'<a class="{cls}" href="{slug}.html"><span class="lbl">{lbl}</span>'
                f'<span class="nm">{html.escape(chno)}　{html.escape(title)}</span></a>')
    left = item(i-1, '', '上一章') if i > 0 else '<span class="none"></span>'
    right = item(i+1, 'next', '下一章') if i < len(PAGES)-1 else '<span class="none"></span>'
    return left + right

def read_body(slug):
    """读取 _src/<slug>.body.html；
    若改为 _src/<slug>.head.html + _src/<slug>.part*.body.html 分片，则按文件名顺序拼接。"""
    SRC = os.path.join(ROOT, '_src')
    main = os.path.join(SRC, slug + '.body.html')
    if os.path.exists(main):
        return open(main, encoding='utf-8').read()
    head = os.path.join(SRC, slug + '.head.html')
    pref = re.escape(slug) + r'\.part[\w-]*\.body\.html$'
    parts = sorted(f for f in os.listdir(SRC) if re.match(pref, f))
    if not os.path.exists(head) or not parts:
        return None
    buf = [open(head, encoding='utf-8').read().rstrip()]
    for p in parts:
        buf.append('')
        buf.append(open(os.path.join(SRC, p), encoding='utf-8').read().strip())
    return '\n'.join(buf) + '\n'

def build():
    for i, (slug, chno, title, desc, rng) in enumerate(PAGES):
        body = read_body(slug)
        if body is None:
            print('  skip (no body):', slug); continue
        mp = os.path.join(SRC, slug + '.meta.json')
        meta = {}
        if os.path.exists(mp):
            meta = json.load(open(mp, encoding='utf-8'))
        page_title = meta.get('title', title)
        out = HEAD.format(title=html.escape(page_title), sub=SUB)
        out += body.strip() + '\n'
        out += TAIL.format(nav=nav_html(i))
        open(os.path.join(ROOT, slug + '.html'), 'w', encoding='utf-8').write(out)
        n_eq = len(re.findall(r'<section class="fml"', body))
        n_supp = len(re.findall(r'<section class="fml supp"', body))
        print(f'  ok {slug}.html  编号公式={n_eq}  补充小节={n_supp}  chars={len(body)}')

def build_index():
    rows = []
    for slug, chno, title, desc, rng in PAGES:
        cnt = f'<span class="count">{rng}</span>' if rng else '<span class="count">速查</span>'
        rows.append(f'<li><div class="row"><span class="n">{html.escape(chno)}</span>'
                    f'<a href="{slug}.html">{html.escape(title)}</a>{cnt}</div>'
                    f'<p class="desc">{html.escape(desc)}</p></li>')
    body = f'''<h1 class="booktitle">{SUB}</h1>
<p class="byline">配套《Large Language Models》公式逐条推导 · 移动端优先</p>
<a class="start" href="00-prereq.html">从数学预备知识开始</a>
<a class="start alt" href="01-ch02-scaling.html">直接进入公式推导</a>
<ul class="index">
{chr(10).join(rows)}
</ul>
<h2 id="how">怎么用这本手册</h2>
<p>原书共出现 <strong>154 个编号公式</strong>，另有约 96 处行内数学表达式。本手册把它们按章节顺序拆成 16 页，每一条公式都给出：</p>
<ul>
<li><strong>原书公式图片</strong>——点开可放大，保证和你手里的书完全一致；</li>
<li><strong>公式的 LaTeX 重写</strong>——可复制、可搜索，脱离图片也能读；</li>
<li><strong>符号表</strong>——逐个说明每个字母的含义与量纲；</li>
<li><strong>推导步骤</strong>——从大学数学（线性代数、微积分、概率统计、优化）出发，一步一步推到书上那一步；</li>
<li><strong>直觉与坑</strong>——公式"在说什么"以及"哪里容易被误用"。</li>
</ul>
<div class="note"><b>前置要求</b>
默认读者有大学理工科数学基础，但具体计算细节已经模糊。因此每页都会临时补齐所需的数学工具（对数线性化、拉格朗日乘子、链式法则求梯度、极大似然、KL 散数……），不需要预先复习完整门课程。</div>
'''
    out = HEAD.format(title=TITLE, sub=TITLE) + body + '\n' + TAIL.format(
        nav='<span class="none"></span>')
    open(os.path.join(ROOT, 'index.html'), 'w', encoding='utf-8').write(out)
    print('  ok index.html')

if __name__ == '__main__':
    os.makedirs(SRC, exist_ok=True)
    print('building pages...')
    build()
    build_index()