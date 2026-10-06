# ISLP 公式推导详解

《An Introduction to Statistical Learning》(ISLP, 2023) 全书数学公式的**逐步推导**与
**前置知识**手册。面向的读者：学过大学数学、但具体计算细节已经忘掉的读者。

每个公式条目固定给出四段：

- **基础** —— 用到的概念的定义，一句话一条；
- **推导** —— 从定义出发的分步计算，中间表达式不省略；
- **结论** —— 这个式子在做什么，与书中叙述如何对应；
- **易错点** —— 读者真实会踩的坑。

## 页面

| 文件 | 内容 |
| --- | --- |
| `index.html` | 目录页：章节卡片 + 全文公式搜索 |
| `00-prerequisites.html` | 数学基础：线性代数、微积分、概率论、正则化与优化 |
| `01-introduction.html` | 第 1 章 记号与矩阵代数 |
| `02-statistical-learning.html` | 第 2 章 统计学习概览（含偏差–方差分解） |
| `03-linear-regression.html` | 第 3 章 线性回归（最小二乘、标准误、$t$/$F$ 检验） |
| `04-classification.html` | 第 4 章 分类（逻辑回归、LDA/QDA、朴素贝叶斯、GLM） |
| `05-resampling-methods.html` | 第 5 章 重采样方法（LOOCV 的精确推导、$k$ 折交叉验证） |
| `06-linear-model-selection-and-regularization.html` | 第 6 章 模型选择与正则化（$C_p$、ridge、LASSO、弹性网） |
| `07-moving-beyond-linearity.html` | 第 7 章 超越线性（多项式、样条、平滑、核方法） |
| `08-tree-based-methods.html` | 第 8 章 基于树的方法（分裂准则、bagging、随机森林、boosting） |
| `09-support-vector-machines.html` | 第 9 章 支持向量机（对偶推导、软间隔、核技巧） |
| `10-deep-learning.html` | 第 10 章 深度学习（反向传播、CNN、RNN、SGD） |
| `11-survival-analysis-and-censored-data.html` | 第 11 章 生存分析（Kaplan-Meier、log-rank、比例风险） |
| `12-unsupervised-learning.html` | 第 12 章 无监督学习（PCA、K-means、层次聚类、谱聚类） |
| `13-multiple-testing.html` | 第 13 章 多重检验（Bonferroni、Holm、BH 与 FDR） |

每个公式条目都能一键跳回原书网页版对照（`../isl/`），原书页面在同一台机器的
`../isl/index.html`。

## 特点

- **完全离线**：公式在构建期由 KaTeX 预渲染成 HTML + CSS，运行时不需要任何
  JavaScript 数学库，断网可用、不闪动。
- **移动优先**：版式与配色沿用 `../isl/style.css`，单栏窄屏排版，宽屏渐进放大。
  过宽的公式横向滚动而不缩放字号。
- **深色模式**：与原书一致，`t` 键或右上角按钮切换。
- **公式编号不丢**：编号抽到公式框右上角，横向滚动时始终可见。

## 构建

内容是 Markdown + LaTeX 的一个子集，源码在 `src/`，构建脚本在 `tools/`。

```sh
npm install                 # 只装 katex（构建期用）
node tools/build.mjs        # 编译全部页面；KaTeX 出错、标题编号对不上原书、
                            # 或 @src 原书锚点不存在，都会置退出码 1
node tools/check.mjs        # 覆盖率检查：每章条目数 vs 原书公式清单
node tools/audit.mjs        # 逐条对齐：每个原书编号是否有对应条目（缺口应为 0）
node tools/lint.mjs         # 结构检查：基础/推导/结论三段、锚点唯一、
                            # \text{} 里不得内嵌 $…$、章首章末两节
node tools/texcheck.mjs -f src/chNN.md   # 只检查某章的公式能否渲染
node tools/manifest.mjs     # 重新生成公式清单（只在原书 HTML 更新后需要）
```

`src/manifest_chNN.json` 是从原书网页版自动抽出的**公式清单**：编号、页码、切图
文件名、以及公式前后的正文。撰写时对照它逐条覆盖，`tools/check.mjs` 负责核对。
`src/manifest_extra.json` 补录了**原书网页版解析器漏切**的那几个编号公式
（已从 `ISLP_website.pdf` 逐个核对），`manifest.mjs` 会把它们合并进对应清单。

撰写规范见 [AUTHORING.md](AUTHORING.md)。

## 目录

```
islmath/
├── index.html              编译产物：目录页
├── 00-*.html … 13-*.html   编译产物：各章
├── style.css               版式（在 ../isl/style.css 基础上加了数学相关增量）
├── app.js                  交互（深色模式、进度条、目录搜索、回到顶部）
├── src/                    Markdown 源文件 + manifest
└── tools/                  构建与检查脚本
```

KaTeX CSS 与字体来自全站共享的 `../assets/katex/`，本目录不再自带副本。

`node_modules/` 只在构建时需要（`npm install` 装 katex），不入库，产物本身不依赖它。