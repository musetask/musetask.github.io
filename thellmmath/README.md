# thellmmath —— 《Large Language Models》公式推导手册

把 `../llm.pdf`（Wayne Xin Zhao 等著《Large Language Models》）里出现的**全部 154 个编号公式**
逐条重写为 LaTeX，并补齐从大学数学出发的完整推导。移动端优先，配色与版式沿用 `../thellm`。

## 打开方式

需要用一个静态服务器（页面里有跨目录引用的原书图片，直接 `file://` 打开也能显示，但
`fetch`/模块在部分浏览器受限）：

```bash
cd ..            # 即包含 thellm/ 和 thellmmath/ 的目录
python3 -m http.server 8899
# 浏览器打开 http://127.0.0.1:8899/thellmmath/
```

## 目录结构

```
thellmmath/
├─ index.html            目录页（由 build.py 生成）
├─ 00-prereq.html        数学预备知识速查
├─ 01-ch02-scaling.html  第 2 章 缩放定律（2.1–2.5）
├─ …                    每个章节 1–3 页
├─ 16-ch12-eval-b.html   第 12 章 评测（12.9–12.15）
├─ style.css             版式（配色变量取自原书 #a11f27）
├─ app.js               深色模式 / 代码复制 / 阅读进度 / 公式灯箱 / KaTeX 调用
├─ ../assets/katex/   全站共享的 KaTeX 0.16.11（本地化，离线可用）
├─ build.py             把 _src/*.body.html 套壳成完整页面 + 生成目录页
├─ check.js             静态校验（见下）
├─ fix_escape.py        把数学里的裸 < > & 转义成实体
└─ _src/                各页正文片段（唯一需要手工编辑的地方）
```

## 工作流

```bash
cd thellmmath
python3 build.py     # 生成/更新所有 *.html 与 index.html
python3 fix_escape.py  # 修正数学里的裸 < > &
node check.js        # 校验，不过则退出码非 0
```

## 内容规范

见 `../GUIDE.md`。要点：

- 每条公式一个 `<section class="fml" id="eq-章-号">`，内含
  ① 公式（KaTeX 重写）② 符号表 ③ 推导步骤 ④ 直觉框 ⑤ 易错点 ⑥ 相关公式跳转。
- 书里没有编号公式、但数学上必须补的推导，用 `<section class="fml supp" id="...">`（虚线框），
  不需要原图和 ①② 小标题。
- 数学写法：行内 `\( … \)`，独立 `\[ … \]`（必须包在 `<div class="eq">` 里）。
- **数学里的 `<`、`>`、`&` 必须写成 `&lt;`、`&gt;`、`&amp;`**，否则浏览器会把 `<` 当 HTML 标签解析，
  把后面的公式整段吃掉。`fix_escape.py` 会自动修，`check.js` 会报出来。
- 长公式用 `\begin{aligned} … \end{aligned}` 手动折行；窄屏下公式块可横向滑动。
  注意 KaTeX 不支持 `align` 环境（用 `aligned`）、`\label`、`\eqref`、`$…$`。

## check.js 检查什么

1. HTML 标签是否配平、id 是否重复
2. 数学定界符 `\[ \]` `\( \)` 是否配对（跳过 `\\` 转义对，否则 `\\[2pt]` 会被误判）
3. 数学里有没有裸 `<` `>`（未转义）
4. 每一处数学能否被 KaTeX 真实渲染通过（用共享的 ../assets/katex/katex.min.js）
5. 原书公式图片、原书页面链接、跨页锚点是否有效
6. 每个公式 section 是否齐备：原图、符号表、推导步骤、`note` 直觉框

## 数据来源

- `../llm.pdf` —— 公式编号与原文上下文
- `../thellm/*.html` 与 `../thellm/assets/fig-*.png` —— 原书转出的 HTML 与公式图片
- `../catalog/page-*.txt` —— 按页面切分的"公式编号 / 图片路径 / 原文上下文"清单
- `../extract_eqs.py`、`../build_catalog.py`、`../split_catalog.py` —— 生成上面清单的脚本