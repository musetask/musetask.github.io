=== VERIFICATION STANDARDS ===

项目：`/Users/dilfish/Downloads/isl/islmath/` —— 《An Introduction to Statistical
Learning》全书数学公式的逐步推导手册（静态网页版，移动优先，离线可用）。
风格、配色、版式参照同级目录 `/Users/dilfish/Downloads/isl/isl/`（原书网页版）。

## A. 构建与自动化检查（必须全部通过）

在 `/Users/dilfish/Downloads/isl/islmath` 下运行：

1. `node tools/build.mjs`
   - 退出码 0；末行必须是 `KaTeX: 全部公式渲染通过`。
   - 不得有 `缺少源文件`（`src/ch07.md`、`src/ch12.md` 必须在）。
   - 输出中 14 个页面（00-prerequisites … 13-multiple-testing）都应有非零标题/公式数。
2. `node tools/check.mjs`
   - 退出码 0；13 个章节行「编号」列全部为「齐」，不得有「缺文件」。
3. `for f in src/*.md; do node tools/texcheck.mjs -f $f || exit 1; done`
   - 每个文件都必须是 `0 条无法渲染`。
4. `node tools/audit.mjs`
   - 每章「覆盖缺口」必须为 0。
5. 编译产物检查：
   - 所有 `islmath/*.html` 中 `class="katex-error"` 计数为 0。
   - `grep -c 'katex-error' *.html` 全部为 0。

## B. 内容覆盖

对每个 `chNN.md`：

1. 逐条覆盖 `src/manifest_chNN.json` 的 `equations` 全部条目，**含 `num` 为 null
   的无编号公式**（无编号公式也要有独立 `###` 条目，锚点 `{#eq-NN-uK}`）。
2. 每个编号公式在正文中出现（`check.mjs` 的「齐」即为通过）。
3. 每个 `###` 条目含 `**基础**`、`**推导**`、`**结论**` 三段（`易错点` 可选）。
4. 每个文件含 `## 本章数学路线` 开头 与 `## 章末速查` 结尾（速查表为 Markdown 管道表）。
5. 公式记号与原书一致：撰写者需用 Read 工具核对 `../isl/assets/<img>` 切图。
   抽样复核至少 12 个编号公式（跨 ≥6 个章节），确认记号与原图一致
   （如 `$X$` vs `$x$`、`$\hat\beta$` vs `$\beta$`、截距写法、`\epsilon` vs `\xi`、
   先验概率 `\pi_k`、第 3 章 $S_{jj}$ 的 $n-1$、第 9 章 $\boldsymbol\beta_0$ 记法等）。
6. 抽查 ≥8 条推导的**数学正确性**：逐行验算，确认代数变形正确、
   条件（$E[\epsilon\mid X]=0$、独立性、Slater、正定性等）在文中被正确声明。

## C. 网页与移动优先

在 `http://127.0.0.1:8902/islmath/` 起服务后用浏览器验证（从
`/Users/dilfish/Downloads/isl` 目录起 `python3 -m http.server 8902`）：

1. 全部 14 个页面在 **320 / 390 / 768 / 1280px** 四个宽度下
   `document.documentElement.scrollWidth <= window.innerWidth`（无横向溢出）。
2. 无 JavaScript 控制台错误。
3. `index.html`：章节卡片齐全（14 张）、搜索框可用（输入中文关键词如「岭回归」
   「偏差」能过滤出公式级条目）、清空后恢复。
4. 深色模式：切换后正文与公式均为浅色字、深色底，可读；`localStorage` 持久化。
5. 顶部 sticky 栏、阅读进度条、回到顶部按钮、底部翻页（上一章/下一章）可用，
   且首尾两章的翻页留空占位正确。
6. 过宽的显示公式横向滚动而不撑宽页面；公式编号（`.eqtag`）始终可见
   （位于公式框右上角，不随横向滚动移出）。
7. 每页顶部的「本章公式一览 / 本页小节一览」可展开，点击条目能跳到对应锚点。

## D. 结构与链接

1. `islmath/style.css` 是 `../isl/style.css` 的增量（含其全部规则），
   数学相关样式集中在文件末尾注释块内。
2. 每个公式条目下的 `@src` 行链接指向 `../isl/<原书页面>.html[#eq-N-M]`，
   点击后原书页面能定位到对应锚点（抽查 ≥5 个）。
3. 无外网资源引用：HTML/CSS/JS 里不得出现 `http://` 或 `https://` 的
   script/style/font/img 引用（数学字体来自 `../assets/katex/fonts/`）。
4. `README.md` 存在且与实际页面列表一致；`AUTHORING.md` 存在。

## E. 交付

回报需包含：构建/检查命令的实际输出（build 末行、check 表、texcheck 汇总）、
抽样复核的公式清单与结论、以及浏览器验证中 320/390/768/1280 四宽度的溢出检查结果。
若任一项不通过，返回 FAIL 并列出具体条目与文件行号。

=== END VERIFICATION STANDARDS ===
---

## 附录：第一轮验收的 8 处问题与处理（供复核）

| # | 问题 | 处理 |
| --- | --- | --- |
| 1 | ch09 出现两个 `(9.16)` 编号（`### (9.16) 的约束块` 与 `### (9.16)` 同章） | 标题改为「用 2p 个特征时的归一化与预算约束」，并删掉 (9.11) 条目里与上方 `aligned` 块重复的 `$$…\tag{9.11}$$` |
| 2 | ch09 (9.7) 记号错：写成 `f(x*)=β₀+β₁x₁*+…`（无帽 β） | 按原书切图改为正类一侧的条件 `(9.7)`；原 (9.6) 改为负类一侧条件；测试观测记号移入 (9.6) 的「基础」段说明 |
| 3 | **ch05 杠杆值界论证错误**（`Σ_j H_ij = H_ii`、`Σ H_ij² ≤ Σ H_ij`、`h_i ≥ 1/n` 显然） | 改为「投影不增范数 ⇒ $h_i\le1$」与「$\mathrm{tr}(H)=p$ ⇒ $\min_i h_i\le p/n\le\max_i h_i$」，并给出 $h_i<1/n$ 的反例（$X=(1,1,1,10)^\top$），注明原书 $1/n$ 的说法只是粗略表述 |
| 4 | **ch09 互补松弛方向写反**（`ε_i=0 ⇒ κ_i=0`、三情形表里 `κ_i>0 ⇒ ε_i>0`） | 改为单向蕴含 `ε_i>0 ⇒ κ_i=0 ⇒ α_i=ρ`；`ε_i=0` 时 κ 自由；`α_i<C` 用 `κ_i>0 ⇒ ε_i=0`，`α_i=C` 用 `κ_i=0` |
| 5 | ch11 (11.3) 漏了原书的 $W$ 下标 | 全文统一为 $\widehat S_W$（已与原书 PDF 正文 `WS(dk)` 核对） |
| 6 | ch03 出现悬空的带撇编号引用 (3.4')/(3.3')/(3.25')/(3.25'')/(3.7') | 全部改为指向真实锚点的页内链接（`#eq-3-u4`、`#eq-3-3`、`#eq-3-u9`、`#eq-3-u10`、`#eq-3-u22`） |
| 7 | ch02 自编了原书不存在的 (2.13)/(2.15) | 原书第 2 章编号止于 (2.12)。已重排：(2.11)=Bayes 错误率、(2.12)=KNN 概率估计；原先的「逐点 Bayes 错误概率」与「KNN 分类规则」改为无编号条目 `#eq-2-u1`/`#eq-2-u2`，并删除标题里的伪编号 |
| 8 | ch04 在 (4.5) 条目内把 $\ell$ 先当似然、后又当对数似然 | 保留 (4.5) 的 $\ell$（与原书一致），对数似然改记 $\Lambda$，全章推导同步；另把 (4.5) 的乘积改回原书按类别分组的写法 |

另外新增两处防回归措施：
- `tools/build.mjs` 增加**编号校验**：条目标题里的 `(N.M)` 必须在 manifest 中存在，否则构建报警并置退出码 1。
- `tools/lint.mjs` 增加 **`\text{}` 内嵌 `$…$`** 检查（KaTeX 会静默丢弃，已在 ch04/ch12 修掉 3 处）。

以及一处**覆盖补全**：从 `ISLP_website.pdf` 逐章核对了正文里出现、但原书网页版
解析器漏切的编号公式（存于 `src/manifest_extra.json`），据此补写了 (3.34)、(3.35)、
(9.14) 三个此前缺失的条目，并把 `#eq-3-x3` 提升为正式的 (3.36)。

## 附录二：第二、三轮验收的问题与处理

| # | 问题 | 处理 |
| --- | --- | --- |
| R2-1 | **ch05 (5.2) Sherman–Morrison 恒等式错**：第 1 步写出错误等价 $\frac{w}{1+x_i^\top w}=(1-h_i)Gx_i$（实为 $=Gx_i$），导致 boxed 式与后续全错 | 改为 $\frac1{1+x_i^\top w}=1-h_i$ 与 $w=\frac{1}{1-h_i}Gx_i$；boxed 式改为 $\hat\beta-\hat\beta_{(i)}=Gx_i\bigl(y_i-\hat f_{(i)}(x_i)\bigr)$，并显式提示「不是 $(1-h_i)Gx_i$」。已用 numpy 在 3 组 $(n,p)$×5 种子独立验算 8 条恒等式全过 |
| R2-2 | ch02 两条 `@src` 锚点不存在（漏了章节编号前缀） | 改为 `#s-2-2-assessing-model-accuracy`、`#s-2-2-3-the-classification-setting`；并新增构建期 `@src` 锚点校验（见下） |
| R2-3 | ch03 残留 `(3.23'')`、`(3.35'')` 悬空引用 | 改为指向 `#eq-3-23`、`#eq-3-u12` 的页内链接 |
| R2-4 | ch11 的 $\widehat S_W$ 未全章统一 | 全章统一为 $\widehat S_W$，含速查表 |
| R2-5 | ch04 (4.5) 未改回原书的按类别分组乘积 | 改为 $\prod_{i:y_i=1}p(x_i)\prod_{i':y_{i'}=0}(1-p(x_{i'}))$，并说明它与统一指数写法等价 |
| R2-6 | ch09 预算约束的互补松弛表述过强 | 改为单向表述 |
| R3-1 | **ch09 软间隔对偶把预算常数 $C$ 误当对偶乘子** | 见下 |
| R3-2 | ch05 有个维度不成立的等式 $(\tilde X x_i)^\top\tilde{\mathbf y}$ | 改为 $(\tilde X^\top x_i)^\top\tilde{\mathbf y}$ |
| R3-3 | ch05 括注把 $1/(1-h_i)$ 数成两处 | 改为「全部来自留一残差那一步；boxed 式的 $Gx_i$ 本身不带该因子」 |

### R3-1 的详细处理（唯一一处推导层面的实质修正）

第二轮指出：原书 (9.15) 是**预算形式** $\sum_i\epsilon_i\le C$，其对偶的盒约束上标是
$\rho$（内生变量，$\rho^\ast=\max_i\alpha_i$），**一般不等于 $C$**；直接把对偶写成
$0\le\alpha_i\le C$ 会违反弱对偶。

我独立复核确认该判断成立（$n=2$、$\boldsymbol x_1=(1,-1)^\top,y_1=1$、
$\boldsymbol x_2=(1,2)^\top,y_2=-1$、$C=\tfrac13$）：

- 原问题最优值 $\tfrac{25}{162}\approx0.1543$（此时 $\beta_0\ne0$）；
- 误用 $0\le\alpha_i\le C$ 时，$\sum_i\alpha_iy_i=0$ 迫使 $\alpha_1=\alpha_2=\tfrac29$，
  对偶值 $\tfrac29\approx0.2222$ **> 原值**，违反弱对偶；
- 改用 $0\le\alpha_i\le\rho$ 后 $\rho^\ast=\tfrac{5}{27}\approx0.1852\ne C$，对偶值回到
  $\tfrac{25}{162}$，强对偶恢复。

已把整段推导重写为：**第 0 步先把预算形式换成等价的惩罚形式**
$\min\frac12\lVert\boldsymbol\beta\rVert^2+C\sum_i\epsilon_i$，
在惩罚形式下 $C$ 是给定常数，对 $\epsilon_i$ 求导直接得 $\alpha_i=C-\kappa_i$，
故 $0\le\alpha_i\le C$；同时在「易错点」里保留预算形式的对偶
（$0\le\alpha_i\le\rho$，$\rho$ 内生）与上面那个反例，供对照。
原有的 $\alpha_i=\rho-\kappa_i$ 三情形表、结论与易错点已同步更新。

## 累计新增的防回归措施

1. `tools/build.mjs`：**标题编号校验**——条目标题里的 `(N.M)` 必须在 manifest 中存在。
2. `tools/build.mjs`：**`@src` 原书锚点校验**——缓存 `../isl/*.html` 的 id 集合，
   校验每个 `@src` 链接的锚点真实存在。
3. `tools/lint.mjs`：**`\text{}` 内嵌 `$…$` 检查**（KaTeX 会静默丢弃，已修 ch04/ch12 共 3 处）。
4. `src/manifest_extra.json`：补录原书网页版**解析器漏切**的编号公式
   （2.6、2.9、3.20、3.30、3.34、3.35、3.36、6.4、9.9、9.10、9.12、9.13、9.14、13.3、13.9），
   全部从 `ISLP_website.pdf` 逐个核对；据此补写了 (3.34)、(3.35)、(9.14) 三个条目，
   并把 `#eq-3-x3` 提升为正式的 (3.36)。
