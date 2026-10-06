## 7.7 广义可加模型

@src 原书 p.313–317 · [7.7 Generalized Additive Models](../isl/08-moving-beyond-linearity.html#s-7-7-generalized-additive-models)

7.6 结尾的原书文字点明了下一步的方向：GAM 提供了「在保持可加性的前提下，把每个变量都换成非线性函数」的统一框架。本节的两个小节分别处理定量响应（7.7.1）与定性响应（7.7.2）。这一节的数学要点只有两句话：

1. 对定量响应，GAM **就是**一次普通最小二乘回归——只要 $f_j$ 用基函数展开过，第 3 章的 $\hat{\boldsymbol\beta}=(X^\top X)^{-1}X^\top\boldsymbol y$ 原封不动可用。
2. 对定性响应，把「响应均值」换成「log odds」，GAM 就成了逻辑回归 GAM，只是把 $\beta_jX_j$ 换成 $f_j(X_j)$。

---

### 被扩展的出发点：多元线性回归模型 {#eq-7-u4}

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_p x_{ip} + \epsilon_i
$$

@src 原书 p.314 · [7.7.1 GAMs for Regression Problems](../isl/08-moving-beyond-linearity.html#s-7-7-1-gams-for-regression-problems)

**基础**：这是第 3 章的多元线性回归（multiple linear regression）。$x_{ij}$ 是第 $i$ 个观测的第 $j$ 个预测子取值；$X_j$ 是第 $j$ 个变量的**总体**取值（随机变量），两者不要混用。$\epsilon_i$ 是误差项，$\E[\epsilon\mid X]=0$。

**推导**（它为什么是 GAM 的「特例」）。在这个模型里，第 $j$ 个变量的贡献被限制成 $x_{ij}$ 的一次函数：

$$
\mathbb{E}[Y\mid X=x] = \beta_0 + \sum_{j=1}^p \beta_j x_j
$$

也就是说，第 $j$ 个成分被**写死**为 $f_j(x)=\beta_jx$ 这种最简单的一元函数。想让 $f_1$ 变成「S 形」或「先升后降」，单靠调 $\beta_1$ 做不到，必须换掉「一次函数」这个函数类——这就是 7.1–7.6 各种方法存在的理由，也是 GAM 要做的事。

**结论**：式子里没有编号（原书只是在 7.7.1 开头回顾了一次），但它是后面所有 GAM 式的**参照系**：取 $f_j(x)\equiv\beta_jx$ 时，(7.15) 退回本式；取 $f_j\equiv0$（$j\ge2$）时退回简单线性回归。理解「GAM = 多元线性回归 + 每个成分换成任意一元函数」这句话，就抓住了 7.7 节的全部内容。

**易错点**：$X_j$ 与 $x_{ij}$ 的区别在本节特别重要——GAM 是关于**单个变量的函数**，所以每个成分只接收自己那一个变量（可加性的来源）。不要以为 $f_j$ 可以接收整个 $X$。

---

### (7.15) 回归的 GAM：把每个线性成分换成非线性函数 {#eq-7-15}

$$
y_i = \beta_0 + \sum_{j=1}^p f_j(x_{ij}) + \epsilon_i
= \beta_0 + f_1(x_{i1}) + f_2(x_{i2}) + \cdots + f_p(x_{ip}) + \epsilon_i \tag{7.15}
$$

@src 原书 p.314 · [7.7.1 GAMs for Regression Problems](../isl/08-moving-beyond-linearity.html#eq-7-15)

**基础**：

- **加性模型（additive model）**：响应由 $p$ 个「只依赖单个变量的函数」相加而成。
- **基函数展开**：任何足够光滑的一元函数都可以近似写成 $f_j(x)=\sum_{m=1}^{M_j}\theta_{jm}b_{jm}(x)$，其中 $b_{jm}(\cdot)$ 是**预先选定、已知**的基函数，$M_j$ 是这个成分的自由度。这正是第 7.3 节式 (7.7) 的思路。
- **最小二乘目标**：$\mathrm{RSS}=\sum_{i=1}^n\big(y_i-\beta_0-\sum_jf_j(x_{ij})\big)^2$。
- 记号提醒：ISLP 在本节只用 $\beta_0$ 与 $\beta_1$（(7.19) 里的 year 斜率）；$\theta_{jm}$ 这种「每个基函数一个系数」的记号来自 Hastie–Tibshirani《统计学习基础》，**原书此处的切图里没有它**，下面用它只是为了把「基函数展开 ⇒ 一次普通回归」这件事写清楚。

**推导**（为什么 GAM 可以直接套用第 3 章的全部工具）。分四步：

1. 把每个 $f_j$ 的基函数展开代入模型：

$$
y_i = \beta_0 + \sum_{j=1}^p\sum_{m=1}^{M_j}\theta_{jm}b_{jm}(x_{ij}) + \epsilon_i
$$

2. 令 $K=\sum_{j=1}^pM_j$，把上式看成标准线性模型 $y_i = \beta_0+\sum_{k=1}^{K}\theta_k z_{ik}+\epsilon_i$，其中第 $k$ 个「预测子」是**已知的**设计矩阵元素 $z_{ik}=b_{jm}(x_{ij})$。于是设计矩阵为

$$
X=\begin{pmatrix} 1 & b_{11}(x_{11}) & \cdots & b_{1p}(x_{1p}) \\ 1 & b_{11}(x_{21}) & \cdots & b_{1p}(x_{2p}) \\ \vdots & & & \\ 1 & b_{11}(x_{n1}) & \cdots & b_{1p}(x_{np}) \end{pmatrix},\qquad \boldsymbol\theta=(\beta_0,\theta_{11},\ldots,\theta_{pM_p})^\top
$$

3. 正规方程 $\hat X^\top(X\hat{\boldsymbol y}-X\hat{\boldsymbol\theta})=\boldsymbol 0$。若 $X$ 列满秩（要求 $n>K+1$ 且基函数线性无关），唯一解就是

$$
\hat{\boldsymbol\theta}=(X^\top X)^{-1}X^\top\boldsymbol y,\qquad
\hat{\mathrm{Var}}\big(\hat{\boldsymbol\theta}\big)=\sigma^2(X^\top X)^{-1}
$$

4. 因为 $\hat{\boldsymbol\theta}$ 是 $\boldsymbol y$ 的线性函数，$\hat f_j(x)=\sum_m\hat\theta_{jm}b_{jm}(x)$ 仍是**同一族基函数的线性组合**——用 df=4 的自然样条去拟合，拟合出来仍是 4 维空间里的东西。这正是原书那句「the entire model is just a big regression onto spline basis variables and dummy variables, all packed into one big regression matrix」。

**结论**：GAM 对回归来说**没有任何新的估计理论**，它是「设计矩阵换一种拼法」的普通最小二乘。于是第 3 章的一切照搬：$t$ 检验、整体 $F$ 检验、$R^2$、逐点标准误（即 (7.3)，Figure 7.11–7.12 的误差带就是这样来的）。唯一新增的「调节旋钮」是每个成分的自由度 $M_j$：$M_j$ 越大 $f_j$ 越弯，越容易过拟合。

**易错点**：

- 自由度 $M_j$ **不是**「$\beta_j$ 的个数」。截距 $\beta_0$ 之外，参数总数是 $1+\sum_jM_j$。判断复杂度看这个总数，不看 $p$。
- 不要用 $X^\top X$ 的闭式解硬算：当 $\sum_jM_j$ 接近 $n$（例如直接用平滑样条做 GAM）时 $X^\top X$ 病态甚至奇异，必须改用数值解（IRLS、QR/SVD，或原书用的 backfitting）。
- 可加性是硬约束：$f_j$ 只接收 $X_j$，所以 GAM 无法表示 $X_j$ 与 $X_k$ 的交互。原书在 7.7.1 的「Pros and Cons」里明确说这是主要缺点，补救办法是手工加入 $X_j\times X_k$ 或 $f_{jk}(X_j,X_k)$。

---

### (7.16) Wage 数据上的 GAM：自然样条 + 虚拟变量 {#eq-7-16}

$$
\mathrm{wage} = \beta_0 + f_1(\mathrm{year}) + f_2(\mathrm{age}) + f_3(\mathrm{education}) + \epsilon
$$

@src 原书 p.314 · [7.7.1 GAMs for Regression Problems](../isl/08-moving-beyond-linearity.html#eq-7-16)

**基础**：Wage 数据（$n=3000$，美国中部大西洋地区男性）。`year` 与 `age` 是定量变量，用自然样条（(7.9)–(7.10)，Figure 7.11 说明自由度分别为 4 与 5）；`education` 是五水平定性变量（`<HS, HS, <Coll, Coll, >Coll`），用虚拟变量做步函数（(7.4)–(7.5)）。注意原书这里写的是响应名 `wage` 而不是 $y_i$，误差项也没有下标。

**推导**（设计矩阵到底长什么样）。按 Figure 7.11 的设定逐块展开 $X$ 的列：

1. 截距：$1$ 列。
2. `year` 的自然样条，df $=4$：4 列（多项式部分 $1$ 个基 + 每个节点 1 个截断幂，$df=1+3$）。
3. `age` 的自然样条，df $=5$：5 列。
4. `education` 的 5 个水平：只能放 **4** 个虚拟变量，否则与截距共线——理由同 (7.5) 的注 2：$C_0+C_1+\cdots+C_4=1$ 恒成立，所以必须丢掉一个；参考水平（书中是 `<HS`）的效应被吸收进 $\beta_0$。

于是 $X$ 是 $n\times(1+4+5+4)=n\times14$，用 `sm.OLS` 一次拟合即可。拟合之后，第 $j$ 个成分在任意新点 $x^\ast$ 处的值由对应的基函数列取出：

$$
\hat f_j(x^\ast)=\sum_{m=1}^{M_j}\hat\theta_{jm}b_{jm}(x^\ast),\qquad
\widehat{\mathrm{Var}}\big(\hat f_j(x^\ast)\big)=x^{\ast\top}_{jm}\widehat{\mathrm{Cov}}(\hat{\boldsymbol\theta})\,x^\ast_{jm}
$$

后者与 (7.3) 完全同构（基函数代替了 $x_0^j$），这就是 Figure 7.11/7.12 里每个面板的误差带。

**结论**：原书对这张图的解读正是「加性 + 可逐个变量看」带来的好处：固定 age 与 education，wage 随 year 略升（可能是通胀）；固定其余，年纪处于中间时 wage 最高，太年轻或太老都低；固定 year 与 age，学历越高 wage 越高。三条曲线的纵轴尺度相同，所以**可以直接比较各变量的贡献大小**。

**易错点**：

- $\beta_1$ 不是 year 的边际效应。在 (7.16) 里 year 的边际效应是 $f_1$ 的**斜率**，而 $\beta_0$ 只是基准水平；把某个基函数系数当成「year 的影响」是最常见的误读。
- 定性变量必须用虚拟变量且要丢掉一个水平，否则 $X^\top X$ 奇异。
- 换成平滑样条（Figure 7.12）后**最小二乘不再适用**：平滑样条是 (7.11) 的「损失 + 惩罚」问题，其解不落在任何固定基的线性组合空间里，所以要改用 backfitting 或 `pygam`（原书用后者）。这是 (7.16) 与 Figure 7.12 之间唯一的实质差别。

---

## 7.7.2 分类问题的 GAM

@src 原书 p.316–317 · [7.7.2 GAMs for Classification Problems](../isl/08-moving-beyond-linearity.html#s-7-7-2-gams-for-classification-problems)

这一小节只有三个编号式，逻辑是一条直线：先回顾逻辑回归 (7.17)（把概率压到 log odds 上做线性），再把右边换成非线性 (7.18)，最后给出一个具体实例 (7.19)。

---

### (7.17) 逻辑回归：log odds 的线性形式（回顾） {#eq-7-17}

$$
\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \cdots + \beta_p X_p \tag{7.17}
$$

@src 原书 p.316 · [7.7.2 GAMs for Classification Problems](../isl/08-moving-beyond-linearity.html#eq-7-17)

**基础**：二元响应 $Y\in\{0,1\}$，$p(X)=\Pr(Y=1\mid X)$。**odds**（几率）$=p(X)/(1-p(X))$，即「事件发生」与「事件不发生」的赔率；取对数后称为 **log odds**（logit）。原书说左端是「the log of the odds of $P(Y=1|X)$ versus $P(Y=0|X)$」。

**推导**（为什么一定是 log odds，以及系数怎么解释）。记 $\eta=\beta_0+\sum_j\beta_jX_j$，这是第 4 章的 logistic 模型 (4.6)：

1. 由 $p(X)=\dfrac{e^{\eta}}{1+e^{\eta}}$ 出发（这是唯一保证 $p\in(0,1)$ 的软链接），计算 $1-p(X)$：

$$
1-p(X)=1-\frac{e^{\eta}}{1+e^{\eta}}=\frac{1+e^{\eta}-e^{\eta}}{1+e^{\eta}}=\frac{1}{1+e^{\eta}}
$$

2. 于是 odds $=\dfrac{e^{\eta}/(1+e^{\eta})}{1/(1+e^{\eta})}=e^{\eta}$，取对数得 $\log\frac{p}{1-p}=\eta$，即 (7.17)。
3. 取值范围自洽性：$p\in(0,1)\Rightarrow$ odds $\in(0,\infty)\Rightarrow\eta\in\R$。当 $p\to1$ 时 $\eta\to+\infty$，$p\to0$ 时 $\eta\to-\infty$——记住这个「无穷」，它解释了 (7.19) 里 `<HS` 层标准误爆掉的现象。
4. 系数解释：$X_j$ 增加一个单位，$\eta$ 增加 $\beta_j$，odds 就乘

$$
\frac{\mathrm{odds}(X_j+1)}{\mathrm{odds}(X_j)}=\frac{e^{\eta+\beta_j}}{e^{\eta}}=e^{\beta_j}
$$

倍。

**结论**：(7.17) 就是第 4 章的逻辑回归，只是写成了「左端 log odds 线性、右端预测子线性」的形式。写成这样有两个好处：左端可以是任何实数（不用管概率的 $(0,1)$ 约束），右端线性 ⇒ 第 4 章的 IRLS 求解与全部推断工具可用。

**易错点**：$\beta_j$ 是**对数几率**尺度的增量，不是概率增量。概率增量必须再过一次链接函数 $p=e^\eta/(1+e^\eta)$，而且依赖于当前 $\eta$——同一个 $\beta_j$ 在 $p$ 已接近 1 时几乎不改变概率。

---

### (7.18) 逻辑回归 GAM：把 log odds 的线性成分换成非线性 {#eq-7-18}

$$
\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + f_1(X_1) + f_2(X_2) + \cdots + f_p(X_p) \tag{7.18}
$$

@src 原书 p.316 · [7.7.2 GAMs for Classification Problems](../isl/08-moving-beyond-linearity.html#eq-7-18)

**基础**：式 (7.17) 与本章前六个小节的全部工具：**基函数展开**（把 $f_j$ 写成一组已知基的线性组合）、**logit 链接**、**似然 / IRLS 估计**。原书直言 (7.18)「has all the same pros and cons as discussed in the previous section for quantitative responses」。

**推导**（系数的解释：从 log odds 到 odds 倍数）。原书没有展开写这一步，这里补上完整链条。设只有 $X_j$ 从 $a$ 变到 $a+1$，其余变量固定，记 $\eta$ 的变化为

$$
\Delta = f_j(a+1) - f_j(a)
$$

1. **odds 的倍数**：由 (7.18)，odds $=e^{\eta}$，故

$$
\frac{\mathrm{odds}(a+1)}{\mathrm{odds}(a)}=\frac{e^{\eta+\Delta}}{e^{\eta}}=e^{\Delta}=\exp\big(f_j(a+1)-f_j(a)\big)
$$

2. **但这不是概率的倍数**：概率比多了一个链接函数的因子。逐步算：

$$
\frac{p(a+1)}{p(a)}=\frac{e^{\eta+\Delta}/(1+e^{\eta+\Delta})}{e^{\eta}/(1+e^{\eta})}=e^{\Delta}\cdot\frac{1+e^{\eta}}{1+e^{\eta+\Delta}}
$$

因为 $\Delta>0$ 时 $1+e^{\eta}>1+e^{\eta+\Delta}$，所以概率比**严格小于** odds 比（$\Delta<0$ 时相反）。当 $\eta$ 很大时这个因子趋于 1，odds 比与概率比几乎重合。

3. **退化情形**：若 $f_j$ 是线性的，$f_j(x)=\beta_jx$，则 $\Delta=\beta_j$ 与 $a$ 无关，(7.18) 恰好退回 (7.17) 的 $e^{\beta_j}$ 倍规则。
4. **可解释性的落点**：因为 $\Delta$ 依赖起点 $a$，正确的说法不是「$X_j$ 每增一个单位，odds 乘 $\exp(f_j)$ 倍」，而是「在 $a$ 附近，odds 的变化率是 $\exp(f_j(a))\,\exp(f_j'(a))$」。$f_j$ 的形状（而不是某个数）才是信息所在。

**结论**：$f_j$ 画出「在 $X_j$ 的哪一段风险上升最快」，用自由度 $df$ 控制它的弯曲程度。分类 GAM 的三个优点（自动非线性、可能更准、可逐个变量解读）与一个缺点（无交互）都与 7.7.1 相同。

**易错点**：

- 说「odds 乘 $\exp(f_j)$ 倍」漏掉了差值 $\Delta$，是错的表述。
- GAM 的 $f_j$ 本身没有「系数」意义；只有 $\exp(\Delta)$ 这种倍数、以及 $f_j$ 的**形状**才可解释。
- logit 尺度会压缩高概率段的变化：即使 $f_j$ 在高风险区很陡，概率也趋近 1 而不再上升。看图时要把纵轴从 log odds 换算回概率再下结论。

---

### (7.19) 预测「高收入」的逻辑回归 GAM {#eq-7-19}

$$
\log\left(\frac{p(X)}{1 - p(X)}\right) = \beta_0 + \beta_1 \times \mathrm{year} + f_2(\mathrm{age}) + f_3(\mathrm{education}) \tag{7.19}
$$

@src 原书 p.317 · [7.7.2 GAMs for Classification Problems](../isl/08-moving-beyond-linearity.html#eq-7-19)

**基础**：响应取二值 $\ind\{\mathrm{wage}>250\}$（$250$ 的单位是千美元，即 25 万美元）。三个成分：year 保持**线性**（相当于 $f_1(x)=\beta_1x$）；age 用 df $=5$ 的平滑样条；education 用步函数（对五个水平造虚拟变量）。这是 (7.18) 的一个具体实例。

**推导**（为什么需要后向拟合 backfitting，以及它凭什么成立）。分五步：

1. **为什么不能像 (7.16) 那样一次 OLS**：age 用的是平滑样条，它是 (7.11) 的极小化解

$$
\hat g_\lambda=\arg\min_g\left\{\sum_i\big(z_i-g(x_i)\big)^2+\lambda\int\big[g''(x)\big]^2dx\right\}
$$

其中 $z=\log\frac{p}{1-p}$。这个解不落在任何**固定基**的线性组合空间里（除非把基取成 $\hat g_\lambda$ 的主成分），所以「把它当成一列回归」不成立。
2. **后向拟合的做法**：轮流更新每个成分，其余固定。每次更新时，把当前模型**解释不掉的部分**当作伪响应，再对 $X_j$ 做一次一维拟合。原书脚注 6 给出偏残差（partial residual）的形式：对 $X_3$ 而言 $r_i=y_i-f_1(x_{i1})-f_2(x_{i2})$。
3. **数学依据**（原书只给了直觉，这里写清楚）：设目标函数为

$$
J(\beta_0,f_1,\ldots,f_p)=\sum_{i=1}^n\Big(z_i-\beta_0-\sum_{j=1}^p f_j(x_{ij})\Big)^2+\lambda\sum_{j=2}^p\int\big[f_j''(x)\big]^2dx
$$

固定 $\beta_0,f_1,f_2$，令 $r_i=z_i-\beta_0-f_1(x_{i1})-f_2(x_{i2})$，则

$$
J=\sum_{i=1}^n\big(r_i-f_3(x_{i3})\big)^2+\lambda\int\big[f_3''(x)\big]^2dx
$$

即「对 $X_3$ 做一次带相同 $\lambda$ 的平滑样条拟合」。因此**每一步更新都是在精确地极小化同一个 $J$**，只是每次只动一个成分（块坐标 / 循环坐标下降）。
4. **收敛性**：$J$ 逐步不增且有下界 $0$，故收敛到一个驻点。分类情形只要似然有唯一内点极大值（未完全分离），迭代就稳定；实践中两三轮就稳定（习题 7.9 应用题第 11 题就是把这套迭代与多元回归对比）。
5. **等价的标准做法**：对 (7.18) 直接做极大似然 / IRLS（Fisher–Scoring），也收敛到同一驻点。`pygam` 的 `LogisticGAM` 走的是后一条路；原书只提「backfitting」一词与 `pygam`。

**结论**：(7.19) 的三个面板纵轴尺度一致，所以可以直接比较贡献大小——age 与 education 的影响远大于 year。

**易错点**：

- `<HS` 层的置信区间异常宽，不是「学历与高收入无关」，而是**完全分离**：该类别里没有任何 $Y=1$，似然沿某个系数方向单调上升、估计量发散，标准误爆炸。原书的处理是剔除这些观测后重拟合（Figure 7.14），而不是解读成零效应。
- $\beta_1$ 的单位是「每年」，且它是 log odds 尺度；换算成概率要过 logit。
- 用 `simple_reg` 类的工具做「后向拟合」时，每次必须用**当前**的偏残差，不能复用上一轮的残差。

---

### 响应概率的定义 {#eq-7-u5}

$$
p(X) = \Pr(\mathrm{wage} > 250 \mid \mathrm{year},\ \mathrm{age},\ \mathrm{education})
$$

@src 原书 p.317 · [7.7.2 GAMs for Classification Problems](../isl/08-moving-beyond-linearity.html)

**基础**：条件概率。$\mathrm{wage}$ 在 Wage 数据中以**千美元**为单位，所以 250 就是 \$250,000。原书在 (7.19) 之后用「where」引出这一行，切图上它是紧跟 (7.19) 的一个无编号公式。

**推导**（它把 (7.19) 与数据、图连起来）。令 $Y_i=\ind\{\mathrm{wage}_i>250\}$，则 $p(X)=\E[Y\mid X]$。于是 (7.19) 就是一个以 $\ind\{\mathrm{wage}>250\}$ 为响应的逻辑回归 GAM：由 $\eta=\log\frac{p}{1-p}$ 可反解出

$$
p(X)=\frac{e^{\eta}}{1+e^{\eta}}=\frac{e^{\beta_0+\beta_1\,\mathrm{year}+f_2(\mathrm{age})+f_3(\mathrm{education})}}{1+e^{\beta_0+\beta_1\,\mathrm{year}+f_2(\mathrm{age})+f_3(\mathrm{education})}}
$$

Figure 7.13/7.14 画的是右端三个成分（log odds 尺度），要读「概率」需按上式换算。

**结论**：这一行明确了「预测什么」——原书要的是「一个人年收入超过 \$250,000 的概率」。样本里只有 79 个高收入者（见 7.1 节末尾对置信区间宽的解释），这是后面标准误宽、数据驱动建模不稳的根源。

**易错点**：阈值 250 是千美元而非美元；而且高收入者稀少 ⇒ 任何依赖该响应的拟合都会是高方差问题，分类 GAM 的曲线形状比系数数值更值得看。

---

## 7.8 Lab：非线性建模

@src 原书 p.317–322 · [7.8 Lab: Non-Linear Modeling](../isl/08-moving-beyond-linearity.html#s-7-8-lab-non-linear-modeling)

这一节没有编号公式（原书全是代码），数学上它把前面 6 节的方法逐个在 Wage 数据上实现一遍，值得记下的对应关系是：

1. **7.8.1 多项式与步函数**：`poly()` 实现 (7.1)，`cut()`/`step()` 实现 (7.4)–(7.6)。核心是「换 $X$ 列」——$\beta$ 的估计方式完全没变。
2. **7.8.2 样条**：`bs()` 生成 (7.9)–(7.10) 的基矩阵，`ns()` 生成自然样条基（原书提到默认 `intercept=False`，所以 `ns()` 给出的列数比名义 df 少 1，因为它丢掉了与截距重复的那一列）。于是拟合仍是 `sm.OLS`。
3. **7.8.3 平滑样条与 GAM**：自然样条 GAM 依旧是「一个大 OLS」（对应 (7.16)）；换成平滑样条则用 `pygam` 的 `LinearGAM`/`LogisticGAM`（对应 (7.19)），超参数用 `approx_lam` 由 LOOCV 选出——即把 (7.11)–(7.13) 的 $\lambda$ 选择流程搬进 GAM。原书还用 `anova()` 比较不同 df 的模型。
4. **7.8.4 局部回归**：`lowess()` 实现 (7.14)，只依赖 span 参数。

> 这一节的实践要点：同一个「GAM」在代码层面可以完全是线性代数（自然样条版本），也可以是迭代式非线性拟合（平滑样条版本）；区别只在成分 $f_j$ 的函数类。

---

## 7.9 习题中的公式

@src 原书 p.333–336 · [7.9 Exercises](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

下面 12 条全部来自习题正文（概念题第 1–5 题、应用题第 11 题），原书都不给解答，只给「要求证明/要求画图」。因此每条的写法是：**它在问什么 → 涉及哪条已讲过的结论 → 用什么代数去做**；能严格推出来的（如样条系数、曲线分段式）就把代数写全，需要额外假设的（如测试 RSS 谁更小）明确标注假设。

### 概念题 1：截断幂基给出的分段三次函数 {#eq-7-u6}

$$
f(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \beta_4 (x-\xi)^3_+
$$

@src 原书 p.333 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：截断幂基 $(x-\xi)^3_+=(x-\xi)^3\ind\{x>\xi\}$（(7.10) 的脚注）。原书的命题是：对**任意** $\beta_0,\dots,\beta_4$，$f$ 都是一个只有一个节点的三次回归样条。$M=5$ 个基函数，故 df $=5$。

**推导**（把「分段」这件事变成多项式系数比较）。$(x-\xi)^3$ 展开：

$$
(x-\xi)^3 = x^3 - 3\xi x^2 + 3\xi^2x - \xi^3
$$

1. 当 $x\le\xi$：$(x-\xi)^3_+=0$，故 $f(x)=\beta_0+\beta_1x+\beta_2x^2+\beta_3x^3$。
2. 当 $x>\xi$：代入展开式并按幂次归并，

$$
f(x)=\underbrace{(\beta_0-\beta_4\xi^3)}_{\text{常数}}
+\underbrace{(\beta_1+3\beta_4\xi^2)}_{\text{一次}}\;x
+\underbrace{(\beta_2-3\beta_4\xi)}_{\text{二次}}\;x^2
+\underbrace{(\beta_3+\beta_4)}_{\text{三次}}\;x^3
$$

**结论**：左侧就是全局三次多项式，右侧是另一个三次多项式——这正是 7.4.1 节「两个系数不同的三次多项式」的形式，只是系数已被 $\beta$ 显式决定。后续四问就变成逐阶比较导数。

**易错点**：截断只在 $x>\xi$ 一侧展开，所以 (2) 中的常数项是 $\beta_0-\beta_4\xi^3$（**负号**），二次项系数是 $\beta_2-3\beta_4\xi$（也是负号）。常数项里出现 $\xi^3$ 说明样条基**不平移不变**：$x$ 的原点变了，基函数的系数也跟着变。

### 概念题 1(a)：左侧的三次多项式 $f_1$ {#eq-7-u7}

$$
f_1(x) = a_1 + b_1 x + c_1 x^2 + d_1 x^3
$$

@src 原书 p.333 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：题 (a) 要求找出这个三次多项式，使 $f(x)=f_1(x)$ 对**所有** $x\le\xi$ 成立。$a_1,b_1,c_1,d_1$ 用四系数多项式表示（与 $f_1$ 前面的公式一致）。

**推导**：$x\le\xi$ 时 $(x-\xi)^3_+=0$，于是 $f(x)=\beta_0+\beta_1x+\beta_2x^2+\beta_3x^3$。与 $f_1$ 对比（$1,x,x^2,x^3$ 在任意区间上线性无关，故系数必须逐个相等）：

$$
a_1=\beta_0,\qquad b_1=\beta_1,\qquad c_1=\beta_2,\qquad d_1=\beta_3
$$

**结论**：节点 $\xi$ 完全不影响左段——左段就是普通三次多项式回归 (7.1) 的 $d=3$ 情形。原书强调「regardless of the values of $\beta_0,\dots,\beta_4$」，正是因为这四个系数与 $\xi$ 无关。

**易错点**：等式要对一段区间上的所有 $x$ 成立才能比系数；若只在单点 $x=\xi$ 上成立，一个方程四个未知数，是欠定的。

### 概念题 1(b)：右侧的三次多项式 $f_2$ {#eq-7-u8}

$$
f_2(x) = a_2 + b_2 x + c_2 x^2 + d_2 x^3
$$

@src 原书 p.333 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：题 (b) 要求 $f(x)=f_2(x)$ 对所有 $x>\xi$ 成立。展开 $(x-\xi)^3=x^3-3\xi x^2+3\xi^2x-\xi^3$（与上一条的推导同一行）。

**推导**：$x>\xi$ 时截断项被激活，

$$
f(x)=\beta_0+\beta_1x+\beta_2x^2+\beta_3x^3+\beta_4\big(x^3-3\xi x^2+3\xi^2x-\xi^3\big)
$$

按幂次归并后与 $f_2$ 比系数：

$$
a_2=\beta_0-\beta_4\xi^3,\qquad
b_2=\beta_1+3\beta_4\xi^2,\qquad
c_2=\beta_2-3\beta_4\xi,\qquad
d_2=\beta_3+\beta_4
$$

**结论**：由此确立了 $f$ 是**分段**三次多项式（piecewise polynomial），这是 7.4.1 节图 7.3 的「Top Left: unconstrained」情形——两段之间没有任何连续性要求。原书接着说「We have now established that $f(x)$ is a piecewise polynomial」。

**易错点**：四个系数里三个带 $\xi$、两个带 $\beta_4$，符号分别是 $-,+,-,+$；记错符号会在后面的连续性检验里暴露出来。

### 概念题 1(c)：验证连续性时用的 $f_1$ {#eq-7-u9}

$$
f_1(x) = a_1 + b_1 x + c_1 x^2 + d_1 x^3
$$

@src 原书 p.333 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：这是原书在题 (c) 之前**重述**的 $f_1$（与 (a) 的式子相同，但这次它的用途是验证连续性）。题 (c) 要证 $f_1(\xi)=f_2(\xi)$。

**推导**：代入同一个点 $x=\xi$。左段（截断项为 0）：

$$
f_1(\xi)=\beta_0+\beta_1\xi+\beta_2\xi^2+\beta_3\xi^3
$$

右段用上一条的四个系数：

$$
\begin{aligned}
f_2(\xi)&=(\beta_0-\beta_4\xi^3)+(\beta_1+3\beta_4\xi^2)\xi+(\beta_2-3\beta_4\xi)\xi^2+(\beta_3+\beta_4)\xi^3\\
&=\beta_0+\beta_1\xi+\beta_2\xi^2+\beta_3\xi^3+\beta_4\big(-\xi^3+3\xi^3-3\xi^3+\xi^3\big)\\
&=\beta_0+\beta_1\xi+\beta_2\xi^2+\beta_3\xi^3
\end{aligned}
$$

其中 $\xi$ 相关的四项恰好相消。

**结论**：$f_1(\xi)=f_2(\xi)$，即 $f$ 在 $\xi$ 处连续。切图上这一行带着一个逗号，因为原文紧接着写「That is, $f(x)$ is continuous at $\xi$」。

**易错点**：截断幂基保证了函数值在节点处天然连续（截断项在 $x=\xi$ 处取 0）；如果换成硬截断的分段多项式（7.8），连续性必须额外约束，而 (7.9) 的三次样条基把它变成了自动满足的性质。

### 概念题 1 的提示：一元三次多项式的一阶导数 {#eq-7-u10}

$$
f_1'(x) = b_1 + 2c_1 x + 3d_1 x^2
$$

@src 原书 p.333 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：逐项求导。原书在同一处提示二阶导数为 $f_1''(x)=2c_1+6d_1x$（这一行在网页版与切图里都是散文，没有单独的公式切图）。

**推导**（题 (d)(e) 的答案）：

1. 一阶导连续性：$f_1'(\xi)=\beta_1+2\beta_2\xi+3\beta_3\xi^2$；而

$$
f_2'(\xi)=(\beta_1+3\beta_4\xi^2)+2(\beta_2-3\beta_4\xi)\xi+3(\beta_3+\beta_4)\xi^2
$$

$$
=\beta_1+2\beta_2\xi+3\beta_3\xi^2+\beta_4\big(3\xi^2-6\xi^2+3\xi^2\big)=\beta_1+2\beta_2\xi+3\beta_3\xi^2
$$

2. 二阶导连续性：$f_1''(\xi)=2\beta_2+6\beta_3\xi$；

$$
f_2''(\xi)=2(\beta_2-3\beta_4\xi)+6(\beta_3+\beta_4)\xi=2\beta_2+6\beta_3\xi+\beta_4(-6\xi+6\xi)=2\beta_2+6\beta_3\xi
$$

3. 三阶导：$f_1'''=6\beta_3$ 与 $f_2'''=6(\beta_3+\beta_4)$，在 $\xi$ 处相差 $6\beta_4$，即**不连续**。

**结论**：$f$ 在节点处连续、一阶与二阶导连续、三阶导跳跃 $6\beta_4$——这恰好是 7.4.2 节「三次样条」的定义（分三次多项式 + 到 $d-1=2$ 阶导数连续），与 (7.10) 脚注「adding a term $\beta_4h(x,\xi)$ will lead to a discontinuity in only the third derivative」完全一致。

**易错点**：题目只要求到二阶；三阶导跳 $6\beta_4$ 正是「截断幂基 + 三次」这一组合的指纹。看到三阶导跳变不要以为是算错。另外 (d)(e) 必须代入**同一个** $\xi$ 才能比较。

### 概念题 2：$m$ 阶导数惩罚的平滑拟合 {#eq-7-u11}

$$
\hat g = \arg\min_g \left( \sum_{i=1}^{n} \big(y_i - g(x_i)\big)^2 + \lambda \int \left[g^{(m)}(x)\right]^2 dx \right)
$$

@src 原书 p.334 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：这是 (7.11) 的推广——惩罚项从 $g''$ 换成任意 $m$ 阶导数，且约定 $g^{(0)}=g$。题目要求对五种 $(\lambda,m)$ 组合画出 $\hat g$ 的草图：$(\infty,0),(\infty,1),(\infty,2),(\infty,3),(0,3)$。

**推导**（逐个情形，判据是「惩罚项能否被压到 0」）：

1. **$\lambda\to\infty$**：目标里的第二项被强制最小化，只能取到 $0$，即 $\int[g^{(m)}]^2=0\Rightarrow g^{(m)}\equiv0$。由 $m$ 阶导恒为零，$g$ 必须是次数 $\le m-1$ 的多项式；在此有限维类内再最小化损失项，$\hat g$ 就是**该类的最小二乘拟合**。
2. $m=0$：惩罚项是 $\lambda\int[g^{(0)}(x)]^2dx=\lambda\int g^2dx$，极限下 $g\equiv0$。故 $\hat g$ 恒为零函数（与数据无关的水平 0 线）。
3. $m=1$：$g'\equiv0\Rightarrow g$ 为常数；$\lambda\to\infty$ 时最小化 RSS 的常数是 $\bar y$。故 $\hat g$ 是水平线 $\bar y$。
4. $m=2$：$g$ 为一次多项式 ⇒ $\hat g$ 是最小二乘直线。这与 7.5.1 节的原文一致：「When $\lambda\to\infty$, $g$ will be … just be a straight line that passes as closely as possible to the training points」。
5. $m=3$：$g$ 为二次多项式 ⇒ $\hat g$ 是二次（抛物线）最小二乘拟合。
6. $\lambda=0,\ m=3$：只剩 $\min\sum_i(y_i-g(x_i))^2$，下界 0 可达（插值），所以训练 RSS $=0$，而**极小元不唯一**：任何插值这 $n$ 个点的三次多项式样条都最优（「最简单」的选择是自然三次样条）。曲线穿过每个训练点，典型过拟合。

**结论**：五种草图分别是：零函数、$\bar y$ 水平线、最小二乘直线、最小二乘抛物线、插值训练点的锯齿样条。原书让你画图，实际上是在考「$\lambda$ 控制平滑度、$m$ 控制曲线的多项式阶」这两个旋钮的正交作用。

**易错点**：$m=0$ **不是**「不惩罚」——它惩罚函数本身，极限是零函数。$\lambda=0$ 时 $\arg\min$ 不唯一，题目只要求「example sketches」，此时要说清存在多个解、原书未指定选择规则。

### 概念题 3：基函数模型 $Y = \beta_0 + \beta_1 b_1(X) + \beta_2 b_2(X) + \epsilon$ {#eq-7-u12}

$$
Y = \beta_0 + \beta_1 b_1(X) + \beta_2 b_2(X) + \epsilon
$$

@src 原书 p.334 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：题设基函数 $b_1(X)=X$、$b_2(X)=(X-1)^2I(X\ge1)$（$I(\cdot)$ 为指示函数），并**直接给定**系数估计 $\hat\beta_0=1,\ \hat\beta_1=1,\ \hat\beta_2=-2$，要求画出 $-2\le X\le2$ 上的估计曲线。这正是 (7.7) 的基函数回归形式——只不过这里系数是给定的，不用估计。

**推导**（把估计曲线写成分段多项式）：

$$
\hat f(X)=1+1\cdot X-2\,(X-1)^2I(X\ge1)
$$

1. $X<1$：$I=0$，故 $\hat f(X)=1+X$（直线，截距 $1$、斜率 $1$）。
2. $X\ge1$：展开 $-2(X-1)^2=-2X^2+4X-2$，得

$$
\hat f(X)=-2X^2+5X-1
$$

3. 接缝检查：$X=1$ 处左段给 $1+1=2$，右段给 $-2+5-1=2$，**连续**；斜率左 $1$，右 $-4\cdot1+5=1$，**一阶导也连续**（截断幂基 $(X-1)^2$ 在 $X=1$ 处的值与前两阶导都为 0）。
4. 关键点：$X=-2\Rightarrow-1$；$X=0\Rightarrow1$；$X=1\Rightarrow2$；抛物线顶点 $X^\ast=-b/(2a)=5/4$，值 $-2\cdot(25/16)+5\cdot(5/4)-1=-25/8+25/4-1=17/8=2.125$；$X=2\Rightarrow-8+10-1=1$。

**结论**：草图 = 从 $(-2,-1)$ 直线上升到 $(1,2)$，随后接一段开口向下的抛物线，在 $X=5/4$ 处达最大 $2.125$，再下降到 $X=2$ 处的 $1$。两段的斜率在接缝处相同（都是 1），所以曲线光滑。

**易错点**：$b_2$ 含 $(X-1)^2$，所以曲线在 $X=1$ 处连续且一阶光滑——不是阶梯。若把它误当成 $I(X\ge1)$（不带平方），画出来会是断崖。另一个坑是 $-2$ 乘在 $b_2$ 上而不是 $X$ 上。

### 概念题 4：两个「线性 × 指示函数」基 {#eq-7-u13}

$$
b_1(X) = I(0 \le X \le 2) - (X-1)I(1 \le X \le 2), \qquad b_2(X) = (X-3)I(3 \le X \le 4) + I(4 < X \le 5)
$$

@src 原书 p.334 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：这一行在网页版里被截断了（只残留「$b_1(X)=I(0\le X\le2)-$」），切图 `eq-p334-409` 顶部也被裁掉；此处按原书正文补全（原书把它排在同一行的两个部分，中间用逗号连接）。两个基函数都是「线性函数 × 指示函数」，即在窗口内为直线、在窗口外为 0 的**局部线性（hat-like）**基。

**推导**（逐段取值）：

| $X$ 的区间 | $b_1(X)$ | $b_2(X)$ |
| --- | --- | --- |
| $X<0$ | $0$ | $0$ |
| $0\le X<1$ | $1$ | $0$ |
| $1\le X\le 2$ | $1-(X-1)=2-X$ | $0$ |
| $2<X<3$ | $0$ | $0$ |
| $3\le X\le4$ | $0$ | $X-3$ |
| $4<X\le5$ | $0$ | $1$ |
| $X>5$ | $0$ | $0$ |

**结论**：$b_1$ 在 $[0,1]$ 上是水平 1、在 $[1,2]$ 上从 1 线性降到 0（一个「帽形」的一半的镜像）；$b_2$ 在 $[3,4]$ 上从 0 线性升到 1、在 $(4,5]$ 上保持 1。二者都**连续**（因子在端点处恰好为 0 或 1），所以由它们拟合出的曲线不会有跳跃，除非基的支撑边界本身造成跳跃。

**易错点**：区间端点写法（$0\le X\le2$ 与 $2<X$）在这里只影响单点取值，不影响曲线的连续性结论。容易看错的是 $4<X\le5$ 段：$b_2=1$ 而不是 $X-3$，因为 $(X-3)I(3\le X\le4)$ 在 $X>4$ 时已经归零。

### 概念题 4：待拟合的模型式（与概念题 3 同形） {#eq-7-u14}

$$
Y = \beta_0 + \beta_1 b_1(X) + \beta_2 b_2(X) + \epsilon
$$

@src 原书 p.334 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：与概念题 3 是**同一个公式形状**（这是基函数回归的通用形式 (7.7)），但基函数与系数估计都不同：给定的估计是 $\hat\beta_0=1,\ \hat\beta_1=1,\ \hat\beta_2=3$，要求画 $-2\le X\le6$。

**推导**（代入系数与基函数，逐段相乘相加）：

$$
\hat f(X)=1+1\cdot b_1(X)+3\cdot b_2(X)
$$

1. $X<0$：$\hat f=1$。
2. $0\le X<1$：$\hat f=1+1=2$。
3. $1\le X\le2$：$\hat f=1+(2-X)=3-X$，从 $2$ 线性降到 $1$。
4. $2<X<3$：$\hat f=1$。
5. $3\le X\le4$：$\hat f=1+3(X-3)$，从 $1$ 线性升到 $4$。
6. $4<X\le5$：$\hat f=1+3\cdot1=4$。
7. $X>5$：$\hat f=1$。

**结论**：草图要点——$X=0$ 处向上跳 $1$（$1\to2$）；$1\le X\le2$ 是斜率 $-1$ 的下降段；$X=2$ 处回到 1（连续，因为 $2-X\to0$）；$[2,3)$ 与 $(-2,0)$ 都是水平 1；$[3,4]$ 斜率 $+3$ 升到 4；$(4,5]$ 水平 4（斜率由 3 变 0，但值连续，因为 $X-3=1$ 于 $X=4$）；$X=5$ 之后向下跳 $3$（$4\to1$）。

**易错点**：$4<X\le5$ 段容易顺手写成 $1+3(X-3)$，那会把平台画成继续上升；同样，别忘了 $\hat\beta_2=3$ 是**乘在 $b_2$ 上**。

### 概念题 5：两种阶数惩罚的两条曲线 $\hat g_1,\hat g_2$ {#eq-7-u15}

$$
\hat g_1 = \arg\min_g \left( \sum_{i=1}^{n} \big(y_i - g(x_i)\big)^2 + \lambda \int \left[g^{(3)}(x)\right]^2 dx \right), \qquad \hat g_2 = \arg\min_g \left( \sum_{i=1}^{n} \big(y_i - g(x_i)\big)^2 + \lambda \int \left[g^{(4)}(x)\right]^2 dx \right)
$$

@src 原书 p.334 · [7.9 Exercises › Conceptual](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：与概念题 2 同一个目标函数，只是 $m=3$ 与 $m=4$。三小问：(a) $\lambda\to\infty$ 时哪条训练 RSS 更小；(b) $\lambda\to\infty$ 时哪条测试 RSS 更小；(c) $\lambda=0$ 时训练与测试 RSS 谁更小。

**推导**：

1. **(a) 可以严格证明**：$\lambda\to\infty$ 时 $\hat g_1\to$ 二次（次数 $\le2$）最小二乘拟合，$\hat g_2\to$ 三次（次数 $\le3$）最小二乘拟合。三次类**包含**二次类（令三次项系数为 0 即可），所以

$$
\min_{\mathcal P_3}\mathrm{RSS}\ \le\ \min_{\mathcal P_2}\mathrm{RSS}
$$

即 $\hat g_2$ 的训练 RSS 更小（或相等）。这是纯代数结论，不依赖数据。
2. **(b) 需要假设**：测试 RSS 取决于未知的真实函数 $f$，题面没给，因此无法无条件排序。常见教材答案是「$\hat g_2$ 更小」——理由是三次限制比二次限制更少偏差（例如 $f$ 本身是三次或更低次多项式时 $\hat g_2$ 无偏、$\hat g_1$ 有偏）。但这是**假设**，原书没有给出 $f$。
3. **(c) 训练与测试要分开说**：$\lambda=0$ 时目标只剩 RSS，两者的训练 RSS 都等于 0（都能插值），是平手；测试 RSS 取决于在众多插值解中挑哪一个——$\hat g_2$ 的可行类 $\{g:\ g^{(4)}\text{ 连续}\}$ 是 $\hat g_1$ 可行类 $\{g:\ g^{(3)}\text{ 连续}\}$ 的真子集，被限得更死，通常泛化更好。但同样需要「取最简单/最小范数插值解」这一额外约定才能下结论。

**结论**：这一问真正想考的只有两点——训练 RSS 由「函数类的包含关系」决定（单调可比），测试 RSS 由「偏差」决定，两者可能反向；以及「限制更强」的估计量在 $\lambda$ 大时不会自动更好。

**易错点**：不要把 (a) 的答案直接搬到 (b)。也不要把 $\lambda=0$ 说成「两条曲线相同」：训练 RSS 相同，但插值解不同、测试表现可以差很多。

### 应用题 11(d)：固定 $\beta_1$ 后的简单回归 {#eq-7-u16}

$$
Y - \text{beta1}\cdot X_1 = \beta_0 + \beta_2 X_2 + \epsilon
$$

@src 原书 p.336 · [7.9 Exercises › Applied](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：题目设定 $n=100$、响应 $Y$、两个预测子 $X_1,X_2$，但「只有简单线性回归的软件」。做法是**后向拟合**：轮流固定其余系数，只更新一个。原书把变量名写成代码字体（$\text{beta1}$），因为它同时是程序里的变量名。$\epsilon$ 无下标，同 (7.16)。

**推导**（为什么这一步在数学上是「对」的）：

1. 完整模型是 $Y_i=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\epsilon_i$。给定当前的 $\beta_1$（第 (c) 问随便取），令

$$
z_i = Y_i - \text{beta1}\cdot X_{i1}
$$

模型就变成关于 $(\beta_0,\beta_2)$ 的简单线性回归 $z_i=\beta_0+\beta_2X_{i2}+\epsilon_i$。

2. 等价性（关键一步）：对 $(\beta_0,\beta_2)$ 求极小 $\sum_i\big(Y_i-\beta_1X_{i1}-\beta_0-\beta_2X_{i2}\big)^2$，与「固定 $\beta_1$ 后在全模型上做 OLS」是**同一个**极小化问题。所以每一步更新都精确地做到了「在保持其它系数不变的前提下把这个 RSS 降到最小」。
3. 斜率的显式式子。记 $S_{jk}=\sum_i(x_{ij}-\bar x_j)(x_{ik}-\bar x_k)$，$S_{y2}=\sum_i(y_i-\bar y)(x_{i2}-\bar x_2)$。因为 $\bar z=\bar y-\text{beta1}\cdot\bar x_1$，简单回归斜率为

$$
\hat\beta_2=\frac{\sum_i(x_{i2}-\bar x_2)(z_i-\bar z)}{\sum_i(x_{i2}-\bar x_2)^2}=\frac{S_{y2}-\hat\beta_1S_{12}}{S_{22}}
$$

整理即 $\hat\beta_2S_{22}+\hat\beta_1S_{12}=S_{y2}$，正是多元线性回归中心化正规方程的**第二式**。

**结论**：(d) 一步就产出了多元回归解应满足的一个条件。习题随后要求 (e) 换另一个系数、再 (f) 循环 1000 次、(g) 与真正的多元回归对比——验证的就是「这套迭代会收敛到 $\hat{\boldsymbol\beta}=(X^\top X)^{-1}X^\top\boldsymbol y$」。

**易错点**：每次必须用**当前**的 $\hat\beta_1$ 重算 $z$，不能沿用上一轮的残差；初值任意但应避免让 $z$ 与 $X_2$ 近乎无关（否则 $S_{22}$ 之外的分母退化、$\hat\beta_2$ 剧烈跳动）。截距由 `simple_reg` 自动给出，不用单独处理。

### 应用题 11(e)：固定 $\beta_2$ 后的简单回归 {#eq-7-u17}

$$
Y - \text{beta2}\cdot X_2 = \beta_0 + \beta_1 X_1 + \epsilon
$$

@src 原书 p.336 · [7.9 Exercises › Applied](../isl/08-moving-beyond-linearity.html#s-7-9-exercises)

**基础**：与 (d) 互换角色：固定当前 $\beta_2$，令 $w_i=Y_i-\text{beta2}\cdot X_{i2}$，把 $w$ 对 $X_1$ 做简单回归，截距存为 $\beta_0$、斜率存为 $\beta_1$（**覆盖**上一轮的值）。

**推导**（两步合起来正好是正规方程组）：

1. 本步给出的斜率：$\hat\beta_1=\dfrac{S_{y1}-\hat\beta_2S_{12}}{S_{11}}$，整理得 $\hat\beta_1S_{11}+\hat\beta_2S_{12}=S_{y1}$——多元回归中心化正规方程的**第一式**。
2. 截距：简单回归的截距是 $\bar w-\hat\beta_1\bar x_1$。而 $\bar w=\bar y-\hat\beta_2\bar x_2$，所以

$$
\hat\beta_0=\bar y-\hat\beta_1\bar x_1-\hat\beta_2\bar x_2
$$

这恰好是多元回归对 $\beta_0$ 的正规方程 $\sum_i\hat\epsilon_i=0$。

3. 于是**不动点**处 (d)(e) 的两个式子同时成立，即 $\hat{\boldsymbol\theta}=(\hat\beta_0,\hat\beta_1,\hat\beta_2)$ 满足全部三条正规方程；只要 $X$ 列满秩（$S_{11}>0$、$S_{22}>0$、$S_{11}S_{22}-S_{12}^2>0$），它就**唯一地**等于多元 OLS 解。每轮 RSS 不增 ⇒ 迭代收敛到该不动点。

**结论**：后向拟合不是「近似技巧」，在两预测子的情形下它就是 Gauss–Seidel 型的正规方程迭代，且收敛到与 $(X^\top X)^{-1}X^\top\boldsymbol y$ 相同的解。原书第 (h) 问「how many backfitting iterations were required」正是要你数这个收敛步数；第 12 问（$p=100$）则说明预测子多时同样的逻辑仍成立。

**易错点**：收敛速度取决于 $X_1$ 与 $X_2$ 的相关程度——$\lvert S_{12}\rvert/\sqrt{S_{11}S_{22}}$ 越接近 1 收敛越慢。若 $X_1,X_2$ 几乎完全共线，$X^\top X$ 近奇异，迭代会颤动或发散；此时先解决共线性（第 3 章）再谈后向拟合。

---

