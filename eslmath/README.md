# eslmath — ESL 公式推导手册

《The Elements of Statistical Learning》(Hastie / Tibshirani / Friedman) 的公式逐步推导版。
配套阅读 [../esl](../esl/index.html)（原书的移动端站点）。

- 打开 `index.html` 即可。公式由**本地** KaTeX 渲染，**离线可用**，不联网。
- 深色模式跟随系统，右上角可手动切换。
- 所有公式编号沿用原书编号（章节号.序号），可与原文逐条对照。

## 版本说明（重要）

本地 `../esl` 的 PDF 是一个**混合版**：第 1–13、15–17 章是**第二版**，
但**第 14、18 章是第一版**（第 14 章含 Association Rules / Self-Organizing Maps /
NMF / PageRank；第 18 章含 diagonal LDA / 二次正则化 / 字符串核 / 监督 PCA / FDR）。

因此本站第 14、18 章**按第一版编号写**，以便与读者手上的 `esl/` 逐条对照；
其余 16 章按第二版。两章页面里的「原文 §x.y」链接只作定位参考。

## 目录结构

```
eslmath/
├── index.html          封面（章节卡片由构建脚本自动生成）
├── toc.html            目录（自动生成）
├── mNN.html            各章推导页（自动生成）
├── prereq.html         数学预备知识
├── src/                ← 需要手工维护的内容源文件
│   ├── NN-*.md           单文件章节
│   ├── NN-*.md           外壳（front matter + <!--include --> 分片）
│   └── NN?-*.md          分片（无 front matter、无一级标题）
├── build_site.py       md → html 构建脚本
├── check.py            源文件的只读 lint（分片也支持）
├── katex_check.py      用本地 KaTeX 逐式试渲染，查解析错误
├── html_check.py       构建产物的只读体检（转义 / 嵌套 / 死链）
├── coverage.py         公式编号对账：src 里的 \eqno vs 原书清单
├── fix_links.py        把「原文 §x.y」死链回退到最近存在的锚点
├── WRITING-parts.md    大章分片约定
├── assets/             style.css / app.js
└── ../assets/katex/    全站共享的 KaTeX 0.16.11（CSS / JS / 20 个字体）
```

## 常用命令

```sh
cd eslmath

python3 check.py                      # 源文件 lint（全部）
python3 check.py 03a-least-squares.md # 只查一个分片

python3 katex_check.py                # 用本地 KaTeX 试渲染每个公式（只读）
python3 build_site.py                 # 构建 *.html + toc.html + index.html
python3 html_check.py                 # 产物体检：转义 / .eq 嵌套 / 死链
python3 coverage.py                   # 公式编号对账
python3 fix_links.py --dry            # 预览死链修正
```

修改流程：**改 `src/*.md`** → `check.py` → `build_site.py` → `html_check.py` → 浏览器看一眼。

## 写内容时必须记住的几条

1. 公式编号写在 `\eqno{3.6}` 里，构建脚本会渲染成右侧的小徽标；**编号必须用原书编号**。
2. 数学里**不要用 `*`**（会被当作斜体标记）。上标星号写 `^{\star}`。
3. **不要用 `\begin{align}`**（KaTeX 不支持）。多行用 `\begin{cases}` 或 `\begin{aligned}`。
4. 每个标题都要手写 `{#id}`，id 规则是 `s-<原书节号>`。
5. `$$` 块**不要放在列表项里** —— 会被解析成行内文本，公式不渲染且 `\eqno` 失效。
6. 分片文件不带 front matter、不写 `# ` 一级标题，由外壳 `<!--include -->` 拼装。
7. **编号以原书为准，不要相信任何人的提纲**。写之前先用
   `coverage.py` / 原书 HTML 核对编号；第 14、18 章是第一版。

## 产物体检查什么（都是真实踩过的坑）

- `html_check.py` 第 1 项「公式块未转义」：LaTeX 里的 `<`、`>`、`&`（如 `\sum_{k<\ell}`）
  必须转义后再进 HTML，否则浏览器会把 `<l}` 当标签开头，**整条公式被吞进属性**，
  表现为公式不显示 + 页面横向溢出。
- 第 2 项「嵌套的 `.eq`」：通常是第 1 项的次生症状。
- 第 3 项「死链」：`../esl/xxx.html#s-x-y` 里的锚点在原书里不存在，用 `fix_links.py` 回退。
- 第 4 项「非法 href」：LaTeX 里的 `[..](..)`（矩阵、`\frac` 参数）被 Markdown 当成链接。

## 行内公式的渲染只能在浏览器里查

产物 HTML 里**本来就含 `$...$`** —— KaTeX 是运行时渲染的，所以静态检查
（`check.py` / `html_check.py` / `katex_check.py`）都查不出「行内公式被 Markdown
破坏」这类问题。必须起服务器用浏览器打开 KaTeX 渲染完之后再看：

```sh
python3 -m http.server 8765
```

然后在浏览器里数「可见文本里的裸 `$`」，应当是 **0**。
历史上真出过的两类问题（都由 `build_site.py` 里的 `protect_math` 挡住）：

1. `$...$` 里的 `_` 被当成斜体 → 段落被吞、KaTeX 把后面的中文也当数学渲染。
2. `$...$` 里的 `[..](..)` 被当成链接 → 公式本身被改坏（`(K^{-1}+\lambda I)^{-1}`
   会显示成 `(K^{-1})+\lambda I^{-1}`，是**数值上错**的表达式）。
3. 行内公式里的 `<`、`>`、`&`（如 `$a<b$`）必须 HTML 转义，否则浏览器把 `<b$`
   当标签开头，整条公式被吞进属性。显示公式与行内公式**都要**转义。

## 已知的一处测量伪影

`m14.html` 的 `document.documentElement.scrollWidth` 会比视口宽 66px，
但**页面实际不能横向滚动**（`scrollTo(9999,0)` 后 `scrollX` 仍为 0），
`main` 上有 `overflow-x: clip`。来源是 KaTeX 为无障碍插入的隐藏 MathML
（`.katex-mathml`，1px + clip）偶尔仍被计入文档滚动宽度。
判断这一项要用「能不能滚」而不是 `scrollWidth`。