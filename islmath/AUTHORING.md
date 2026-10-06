# ISLP 公式推导详解 — 撰写规范

面向「有大学数学基础、但计算细节忘了」的读者。每个公式给出：它在原书哪里、
前置知识、分步推导、结论与易错点。

## 写作文件

`islmath/src/chNN.md`（`NN` = 01…13），由 `node tools/build.mjs` 编译成静态 HTML。
源文件是 Markdown 的一个子集 + LaTeX 数学，**离线**渲染（KaTeX 在构建期完成）。

## 每个公式条目的结构

```markdown
### (3.4) 简单线性回归系数的估计 {#eq-3-4}

@src 原书 p.84 · [3.3.2 Model Fitting](../isl/04-linear-regression.html#eq-3-4)

$$
\hat{\beta}_1 = \frac{\sum_i (x_i-\bar{x})y_i}{\sum_i (x_i-\bar{x})^2} \tag{3.4}
$$

**基础**：…（用到哪几个概念，各自的准确定义）

**推导**：

1. 第一步，…（写出中间表达式，不要跳步）
2. 第二步，…

**结论**：…

**易错点**：…
```

### 规则

- 标题层级：`#` 章标题（已由构建脚本给定，正文里不要再写 `#`）→ `##` 小节 → `###` 公式条目。
  实际写法：源文件第一行写 `## 1.1 背景` 之类的小节，公式用 `### (n.m) 名称 {#eq-n-m}`。
- 公式编号用 `\tag{n.m}`；无编号的推导中间式**不要**加 `\tag`。
- `@src` 行给出原书页码与小节链接，链接指向 `../isl/<原书页面>.html#eq-n-m`。
- 段落以 `**标签**：` 开头会被渲染成带彩色边线的块。可用标签：
  `基础`、`推导`、`推导过程`、`结论`、`含义`、`注意`、`易错点`、`备注`、`条件`、`验证`。
- 编号步骤用有序列表（渲染成圆圈序号）。
- 提示/反例用 `>` 引用块。
- 表格用标准 Markdown 管道表（首行为表头）。
- 单独一行 `---` 分隔小节。

### 数学语法

用 KaTeX，构建期渲染，产物完全离线。

预定义宏：`\R`（ℝ）、`\E`（𝔼）、`\N`（ℕ）、`\I`（𝕀）、`\Prob`（ℙ）、
`\Var`（Var）、`\Cov`（Cov）、`\sd`（sd）、`\mse`（MSE）、`\rss`（RSS）、
`\ind`（指示函数 𝟙）、`\Rank`（rank）。

已验证可用的环境与命令：`pmatrix`、`bmatrix`、`vmatrix`、`cases`、`aligned`、
`alignedat`、`array`、`split`、`substack`、`smallmatrix`；`\operatorname`、
`\hat`、`\bar`、`\tilde`、`\widehat`、`\overline`、`\mathbf`、`\boldsymbol`、
`\mathbb`、`\mathcal`、`\mathrm`、`\|`、`\lVert`、`\lvert`、`\partial`、
`\nabla`、`\arg\min`、`\arg\max`、`\min`、`\max`、`\log`、`\exp`、`\Pr`、
`\binom`、`\overset`、`\underbrace`、`\xrightarrow`、`\iff`、`\Rightarrow`、
`\mid`、`\sim`、`\approx`、`\propto`、`\equiv`、`\pm`、`\infty`、`\dim`、
`\operatorname{tr}`、`\det`、`\coloneqq`、`\varepsilon`、`\nabla_{\boldsymbol\beta}`。

`\text{}` 里可写中文。**不要**用 KaTeX 不支持的宏（`\bm`、`\Var` 之类未列出的
自定义命令都会报错）。

写完必须自检两条命令，都必须无输出错误：

```
node tools/texcheck.mjs -f src/chNN.md   # 逐条检查该文件里所有公式
node tools/build.mjs                     # 全量编译，KaTeX 出错则退出码 1
```

### 内容要求

1. **必须覆盖 manifest 里的每一条公式**（含无编号的切图）。`src/manifest_chNN.json`
   的 `equations` 数组每个元素对应原书里的一个公式：
   - `num`：公式编号，如 `(3.4)`；`null` 表示原书未编号（也要写）。
   - `page`：原书页码，用于 `@src` 行。
   - `img`：切图文件名。**用 Read 工具打开 `../isl/assets/<img>` 看原式**，
     不要凭记忆抄——公式记号（$x$ vs $X$、$\hat\beta$ vs $\beta$、$n$ vs $N$、
     $\sigma$ vs $\hat\sigma$）必须与原书一致。
   - `before` / `after`：公式前后的正文片段，提供上下文与符号说明。
   - `section`：所在小节。
   - 无编号公式的标题自己起一个描述性名字，锚点用 `{#eq-<章>-u<序号>}`。
2. **推导要能独立读懂**：不能出现「显然」「容易验证」就跳到结论。
   从定义出发，写出中间表达式。代数变形要逐步展开，不要把两三步并成一步。
3. **基础** 部分只列真正用到的，定义要准确，一句话一条。
4. **结论** 部分说明这个式子在做什么、量纲/取值范围、与书中叙述的对应关系。
5. **易错点** 写读者真实会踩的坑（符号错、方向错、适用条件、为什么不能用
   逆矩阵、什么时候该用数值解等）。没有真实坑就别硬写，可省略该段。
6. 数学正确性优先。不确定的书上细节，说明「原书表述」而不是自己编。
7. 篇幅：单个公式条目 200–600 字，重要公式（3.4、6.8、9.7、10.5 等）可以更长。
8. 中文正文，数学符号用 LaTeX。原书术语首次出现时给出中英对照。

### 章首与章末

每个 `chNN.md` 开头写一个 `##` 小节「本章数学路线」，用 3–6 条列出本章公式的
依赖顺序（谁是谁的特例、谁需要谁）。结尾写 `## 章末速查`，用表格列出本章
所有公式编号 + 一句话含义 + 用到的关键手段。

**不要**写 `#` 一级标题，也不要写章首的「本章共有 N 个公式」之类——
构建脚本会生成章标题和公式一览表。

## 构建与检查

```
node tools/texcheck.mjs -f src/chNN.md   # 只检查该文件的公式
node tools/build.mjs                     # 全量编译；KaTeX 出错则退出码 1
node tools/manifest.mjs                  # 重新生成公式清单（改原书时才需要）
```

写完一章后，用 Read 工具检查编译产物 `NN-*.html` 的开头部分，
确认标题、公式编号、`@src` 链接都正确。
