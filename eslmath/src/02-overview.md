---
id: m02
n: "2"
title: 监督学习概述
title_en: Overview of Supervised Learning
desc: 误差与风险、训练测试误差、偏差–方差分解、模型选择、交叉验证的无偏性与方差、分类问题中的 Bayes 规则、密度估计与维数灾难
prev: prereq
next: m03
prev_title: 数学预备知识
next_title: 第 3 章 线性回归方法
---

# 2 监督学习概述 {#s-2}

这一章不介绍新算法，只做两件事：把「预测得好」写成可计算的量，再说明为什么所有具体方法都可以看成在同一个目标函数下的不同插值方式。书里反复出现的两条线索——损失函数 $L$ 决定目标（平方误差给出条件均值，0-1 损失给出 Bayes 规则），以及估计量的自由度决定偏差–方差折中——都源于第 2.4 节的判决理论推导。本章 47 个编号公式全部给出中间步骤；其中 (2.24) 最近邻距离、(2.25) 偏差–方差恒等式、(2.27)(2.28) 最小二乘 EPE、(2.47) $k$-最近邻三项分解是全书后面所有章节的公共前提。

<a class="src" href="../esl/ch02-overview-of-supervised-learning.html#s-2-2">原文 §2.2–2.3</a>

---

## 2.1 变量类型与记号 {#s-2-1}

先固定记号，后面所有公式都依赖它。输入 $X\in\mathbb{R}^p$ 是实值向量，观测值写作 $x_i\in\mathbb{R}^p$；定量输出 $Y\in\mathbb{R}$，观测值 $y_i$；定性输出 $G\in\mathcal{G}$，$|\mathcal{G}|=K$，观测值 $g_i$。$N$ 个观测排成矩阵 $X\in\mathbb{R}^{N\times p}$（第 $i$ 行是 $x_i^\top$）和 $y\in\mathbb{R}^N$。字母大写表示随机变量，小写表示观测值，这是全书不变的习惯。

两类的区别不只是数字大小。定量输出天然有序，「差 1」有意义；定性输出没有序，数字只是编码。编码成 0/1 或 $-1/+1$ 时，平方误差拟合的值恰好是「类别 1 的条件概率」（见 §2.3.4），这是把回归当成分类的最直接理由。

三种输出类型对应三套不同的损失与不同的「最优」概念。定量输出的最优是条件均值（平方误差）或条件中位数（$L_1$ 损失）；定性输出的最优是后验最大的类（0-1 损失）；有序类别变量（小、中、大）介于两者之间——有顺序但差值无意义，第 4 章用有序逻辑回归处理。

**为什么可以「用回归做分类」，这里先给一句话的答案**：把 $G$ 编码为 $Y\in\{0,1\}$ 后，$\mathbf 1\{G=\mathcal{G}_1\}=Y$，而条件期望对指示变量取就是条件概率：

$$
f(x)=E(Y\mid X=x)=\Pr\big(G=\mathcal{G}_1\mid X=x\big)
$$

所以平方误差的目标函数 (2.3) 本质上在拟合后验概率——只是用了一个「概率必须落在 $[0,1]$」的刚性模型来表示。§2.3.4 会把这句话补完整。

对二分类，把 $G$ 编码为 $Y\in\{0,1\}$ 后做线性回归，再按阈值 0.5 分类：

$$
\hat G=\begin{cases}\mathrm{ORANGE},&\hat Y>0.5\\ \mathrm{BLUE},&\hat Y\le 0.5\end{cases} \eqno{2.7}
$$

分界点两侧的类别区域分别是 $\{x:x^\top\hat\beta>0.5\}$ 与 $\{x:x^\top\hat\beta\le 0.5\}$，分界线 $\{x:x^\top\hat\beta=0.5\}$ 本身是 $p$ 维空间里的一个超平面。把常数 1 并入 $X$ 后，$\beta^\top x=\beta_0+\beta_1^\top x_1+\cdots+\beta_p^\top x_p$，截距被吸收成系数向量的一个分量。

> **坑**
> 编码成 $0/1$ 与编码成 $-1/+1$ 得到的分界线**相同**（都正比于 $\beta_0+\beta^\top x$），但截距与斜率的**比例**不同：同一套数据用两种编码拟合，分界线一样；可是一旦分类规则要求「分界线固定在某处」（第 4 章 LDA 的截距），两种编码会给出不同的规则。见 §2.8 与 <a class="src" href="../esl/ch04-linear-methods-for-classification.html#s-4-5">原文 §4.3</a> 练习 4.2。

> **结果**
> 对 $K>2$ 类，用 $K$ 个 0/1 指示变量 $Y_k=1_{\{G=k\}}$ 分别做线性回归，它们的拟合值满足恒等式
>
> $$\sum_{k=1}^{K}\hat f_k(x)=1$$
>
> 因为对每个 $i$，$\sum_k y_{ik}=1$，常数列 1 被拟合为 $\hat y_{ik}=1/N$（在含截距的模型里截距列的最小二乘拟合等于样本均值）。但 $\hat f_k(x)$ 可能为负或大于 1。第 4 章会说明这正是「掩蔽」（masking）问题的来源。

---

## 2.2 两个最简单的预测方法 {#s-2-2}

### 2.2.1 线性模型与最小二乘 {#s-2-2-1}

线性模型把 $X^\top=(X_1,\dots,X_p)$ 的预测写成截距加各分量的加权和：

$$
\hat Y=\sum_{j=0}^{p}\hat\beta_j X_j \eqno{2.1}
$$

把常数变量 1 并入 $X$，$\hat\beta_0$ 吸收进系数向量 $\hat\beta\in\mathbb{R}^{p+1}$，就得到内积形式：

$$
\hat Y=X^\top\hat\beta \eqno{2.2}
$$

**问题的第一部分：怎么定 $\hat\beta$？** 最小二乘（least squares）取残差平方和最小的系数：

$$
\mathrm{RSS}(\beta)=\sum_{i=1}^{N}\left(y_i-x_i^\top\beta\right)^2 \eqno{2.3}
$$

「残差」指观测与拟合值之差，「最小二乘」指用平方来惩罚残差。选择平方有两个实际理由：一是它给出解析解（下一步就看到），二是它与高斯噪声下的极大似然等价（§2.5 的 (2.35)）。写成矩阵形式（用到 $x_i^\top\beta$ 是 $X\beta$ 的第 $i$ 个分量）：

$$
\mathrm{RSS}(\beta)=(y-X\beta)^\top(y-X\beta) \eqno{2.4}
$$

**推导 · 从 (2.4) 到 (2.6) 的一步都不能跳。** 展开二次项：

$$
\mathrm{RSS}(\beta)=y^\top y-2y^\top X\beta+\beta^\top X^\top X\beta
$$

$\mathrm{RSS}$ 是 $\beta$ 的二次函数，梯度由矩阵微分恒等式（见预备知识 L4，$\nabla_x(x^\top Ax)=2Ax$ 取 $A$ 对称、$\nabla_x(a^\top x)=a$）给出：

$$
\nabla_\beta \mathrm{RSS}=-2X^\top y+2X^\top X\beta
$$

令梯度为零即得**正规方程**：

$$
X^\top(y-X\beta)=0 \eqno{2.5}
$$

若 $X^\top X$ 非奇异（等价于 $X$ 列满秩，$N\ge p+1$），解出

$$
\hat\beta=(X^\top X)^{-1}X^\top y \eqno{2.6}
$$

**确认它是极小而不只是驻点。** 海森矩阵是 $\nabla^2_\beta\mathrm{RSS}=2X^\top X$，它是对称**半正定**的（$z^\top X^\top Xz=\lVert Xz\rVert_2^2\ge0$），故 $\mathrm{RSS}$ 是凸函数，任何驻点都是全局极小（预备知识 C1、C2）。若 $\mathrm{rank}(X)<p+1$，$X^\top X$ 奇异，$\mathrm{RSS}$ 沿 $\ker(X)$ 方向完全平坦，极小解不唯一（正是 §2.6.1 的无穷多解问题）。

**RSS 的 Pythagoras 分解。** 由 $y=P_Xy+(I-P_X)y$ 且两项正交，

$$
\mathrm{RSS}(\hat\beta)=\lVert y-P_Xy\rVert^2=\lVert y\rVert^2-\lVert P_Xy\rVert^2
$$

也就是说训练误差只依赖投影矩阵 $P_X$，与 $\hat\beta$ 的具体取值无关——只要两个系数向量给出同一个 $\hat y$，训练误差就相同。这解释了后面 5.x 的岭回归「$X^\top X+\lambda I$ 改变了 $\hat y$ 才能降低训练误差」。

**还有两个立即可用的结论**：

| 结论 | 表达式 | 含义 |
|---|---|---|
| 拟合值 | $\hat y=P_Xy$，$P_X=X(X^\top X)^{-1}X^\top$ | 投影矩阵，$\mathrm{rank}(P_X)=p+1$ |
| 残差正交性 | $X^\top(y-X\hat\beta)=0$ | 残差与 $\mathrm{col}(X)$ 正交，故 $\hat y$ 是正交投影（预备知识 L2） |
| 有效自由度 | $\mathrm{df}=\mathrm{tr}(P_X)=\sum_{i=1}^Nh_{ii}$ | 见 §2.7.2 |

> **结果**
> (2.6) 中 $(X^\top X)^{-1}$ 把条件数**平方**了：$\kappa_2(X^\top X)=\kappa_2(X)^2$（预备知识 N1）。数值上不要真的求逆，用 SVD 截断解 $\hat\beta=V_rD_r^{-1}U_r^\top y$。秩亏时 (2.6) 不存在，但 $\hat y=P_Xy$ 仍然唯一（$P_X$ 有无穷多种矩阵表示，投影本身唯一）。

<a class="src" href="../esl/ch02-overview-of-supervised-learning.html#s-2-4">原文 §2.4.1</a>

### 2.2.2 最近邻 {#s-2-2-2}

另一个极端：完全不做结构假设，只用输入空间里离 $x$ 最近的 $k$ 个观测来预测。

$$
\hat Y(x)=\frac1k\sum_{x_i\in N_k(x)}y_i \eqno{2.8}
$$

$N_k(x)$ 是训练集中离 $x$ 欧氏距离最近的 $k$ 个点构成的邻域，距离度量先假定为欧氏距离。$k=1$ 时预测区域是训练数据的 Voronoi 镶嵌：每个 $x_i$ 有一个瓦片

$$
V_i=\Big\{x\in\mathbb R^p:\lVert x-x_i\rVert_2\le\lVert x-x_j\rVert_2\ \text{对一切}\ j\ne i\Big\}
$$

在 $V_i$ 上 $\hat G(x)=g_i$，$\{V_i\}$ 恰好铺满 $\mathbb R^p$（每个点离某个 $x_i$ 最近）。图 2.3 的锯齿状分界线就是这套瓦片的边界；当 $k$ 增大时边界变平滑，图 2.2 的 15-最近邻区域已经相当「圆」。

$k$-最近邻的**定义同时适用于定量与定性输出**，只是定性输出时 (2.8) 里的平均变成多数投票：把 $Y$ 编码成 $0/1$，$\hat Y(x)$ 就是邻域内 ORANGE 的比例，阈值 0.5 恰好等价于「邻域内哪个类占多数」。定量输出时 $k=1$ 通常不合适。

(2.8) 的性质值得单独算一遍。固定设计（$x_i$ 不随机），记 $\bar y_k(x)=\frac1k\sum_{x_i\in N_k(x)}y_i$。

1. **训练误差随 $k$ 单调不降**：$k=1$ 时训练误差恒为 0，因为每个 $x_i$ 把自己当邻居。这个事实本身就否决了「用训练误差选 $k$」的做法。
2. **有效参数量约 $N/k$**：若邻域互不重叠，则有 $N/k$ 个邻域，每个邻域里拟合一个参数（均值）。所以 $k$-最近邻虽然只有一个名义参数 $k$，有效自由度是 $N/k$，一般远大于最小二乘的 $p$。
3. **邻域重叠使有效自由度小于 $N/k$**：实际情形下邻域互相重叠，一个点会被多个「参数」共享，所以 $N/k$ 是有效自由度的**上界**。
4. **对距离度量敏感**：$V_i$ 的形状完全由欧氏距离决定。若 $X_j$ 的量纲差异很大（例如一个以米计、一个以美元计），度量就被量纲支配，必须先标准化。这条在 §2.6.3 换成显式核之后仍然存在。
5. **推论：$k$-最近邻对训练集里删点、增点都极其敏感**。加一个远处的点可能改变某个 $x_i$ 的邻域归属，从而改变 $\hat f(x)$——这是「高方差」的直接表现，与 (2.46) 的第三项一致。

**两种方法的假设对照**：最小二乘假设 $f(x)$ 在**全局**上可被线性函数近似；$k$-最近邻假设 $f(x)$ 在**局部**上近似常数。前者稳但可能有偏，后者几乎无偏但不稳。这句话在 §2.4 会被量化成 $\mathrm{Var}(\hat\beta)$ 与偏差的具体数值。图 2.4 用一个模拟把两者放在同一张图上：数据由 10 个均值来自 $N((1,0)^\top,I)$ 的高斯簇（BLUE）与 10 个来自 $N((0,1)^\top,I)$ 的簇（ORANGE）生成，每簇 100 个点、簇内 $N(m_k,I/5)$。混合高斯下线性分界线必然有偏（Scenario 2），$k$-最近邻几乎无偏，所以橙线（测试误差）在 $k$ 小时最低；图中的紫线是 Bayes 错误率，是任何方法的下界。

---

## 2.3 统计决策理论 {#s-2-3}

### 2.3.1 平方误差下的最优预测 {#s-2-3-1}

现在把 $X,Y$ 放回概率空间，$\Pr(X,Y)$ 是联合分布。要找函数 $f:\mathbb{R}^p\to\mathbb{R}$，用平方误差损失 $L(Y,f(X))=(Y-f(X))^2$ 惩罚预测误差。期望预测误差（expected prediction error）是

$$
\mathrm{EPE}(f)=E\left[Y-f(X)\right]^2 \eqno{2.9}
$$

$$
=\int\left[y-f(x)\right]^2\,\mathrm{Pr}(dx,dy) \eqno{2.10}
$$

(2.10) 只是 (2.9) 的积分写法：把联合测度写成条件测度的边缘化 $\mathrm{Pr}(X,Y)=\mathrm{Pr}(Y\mid X)\mathrm{Pr}(X)$（见预备知识 P2 的塔性质），积分就被拆成两层。

**推导 · 为什么可以逐点最小化。** 对固定的 $x$，把 EPE 的内层条件期望记为 $g_x(c)=E[(Y-c)^2\mid X=x]$。由条件概率的定义，

$$
\mathrm{EPE}(f)=E_X\left[E_{Y\mid X}\left[(Y-f(X))^2\mid X\right]\right]=E_X\left[g_{X}(f(X))\right] \eqno{2.11}
$$

外层期望的权重 $\mathrm{Pr}(X)$ 与 $f$ 的取值无关，所以整体最小化等价于对每个 $x$ 单独最小化 $g_x$：

$$
f(x)=\underset{c}{\mathrm{argmin}}\ E[Y-c]^2\mid X=x=E[Y\mid X=x] \eqno{2.12}
$$

$$
f(x)=E(Y\mid X=x) \eqno{2.13}
$$

**把 (2.12) 的 $\mathrm{argmin}$ 真的算出来。** 固定 $x$，记 $m(x)=E[Y\mid X=x]$，$\sigma^2(x)=E[(Y-m(X))^2\mid X=x]$。对任意常数 $c$，

$$
g_x(c)=E\big[(Y-c)^2\mid X=x\big]=E\big[\big((Y-m(x))+(m(x)-c)\big)^2\mid X=x\big]
$$

$$
=E\big[(Y-m(x))^2\mid X=x\big]+2(m(x)-c)E\big[Y-m(x)\mid X=x\big]+(m(x)-c)^2E[1\mid X=x]
$$

第一项是 $\sigma^2(x)$；第二项为零，因为条件期望的定义给出 $E[Y-m(X)\mid X]=0$；第三项用 $E[1\mid X=x]=1$。所以

$$
g_x(c)=\sigma^2(x)+(c-m(x))^2
$$

它是 $c$ 的严格凸抛物线（预备知识 C2），在 $c=m(x)$ 处取到最小值 $0$。代回即得 (2.13)：平方误差下的最优预测是**条件均值**，也叫回归函数（regression function）。

**这个 $g_x(c)$ 的表达式值得单独记下来**，因为它同时给出了三个事实：

1. **最小值是 $\sigma^2(x)$**，也就是平方损失下的不可约误差 $\mathrm{Var}(Y\mid X=x)$；
2. **最优解与 $\sigma^2(x)$ 无关**——它由均值单独决定；
3. **$g_x$ 严格凸且二次**，所以「最小化 EPE」在每个 $x$ 上都有唯一解，不存在局部非全局极小的病态；这也解释了为什么平方损失下的方法（最小二乘、高斯似然）都偏好数值稳定的算法（见第 3 章）。

> **结果**
> (2.13) 是全书最重要的一行：**$f(x)=E(Y\mid X=x)$ 既是最小二乘的目标，也是所有后续方法的参照物**。它不可直接计算（不知道 $\mathrm{Pr}(Y\mid X)$），所以实际做法有两类：用模型强行假设 $f$ 的形状（线性模型、基展开、逻辑回归），或者用局部平均估计它（$k$-最近邻、核平滑）——后者就是 (2.14)。第 2.4 节说明第一类方法在低维高信噪比下更准、第二类在低维低信噪比下更准，而高维时两类都失效；第 5 章的正则化是「同时兼得」的折中。

### 2.3.2 最近邻：直接逼近条件期望 {#s-2-3-2}

直接的实现方式是：在 $x$ 处把所有满足 $x_i=x$ 的 $y_i$ 平均。但一个点处通常至多只有一个观测，于是放松成「近邻平均」：

$$
\hat f(x)=\mathrm{Ave}(y_i\mid x_i\in N_k(x)) \eqno{2.14}
$$

这里有**两次近似**，必须分别记账：

- 用样本平均代替期望 $E(Y\mid X=x)$；
- 把「在点 $x$ 上条件」放松成「在与 $x$ 邻近的区域上条件」。

写成公式，第一步是「经验分布代替总体分布」，第二步是「用示性函数替换 Dirac 条件」：

$$
E(Y\mid X=x)=\int y\,dP_{Y\mid X=x}(y)\ \longrightarrow\ \frac1k\sum_{x_i\in N_k(x)}y_i
$$

$$
\mathbf 1\{X=x\}\ \longrightarrow\ \frac1k\sum_{i=1}^N\mathbf 1\{x_i\in N_k(x)\}
$$

在温和正则条件下（$E|Y|<\infty$，$X$ 的边缘密度在 $x$ 处为正且连续），当 $N\to\infty$、$k\to\infty$ 且 $k/N\to0$ 时，$\hat f(x)\to E(Y\mid X=x)$。所以最近邻是**一致估计量**（consistent estimator，见 §2.7.3 的形式化陈述）。但收敛速度依赖维数 $p$：邻域的度量体积随 $p$ 增大而暴涨，§2.4 给出精确的定量结果。

一致性陈述的形式化版本是：对任意 $\varepsilon>0$，

$$
\Pr\big(\lvert\hat f(x)-E(Y\mid X=x)\rvert>\varepsilon\big)\to0
$$

当 $N\to\infty$、$k=k(N)\to\infty$、$k=o(N)$ 时成立。$k/N\to0$ 这一条不能省：它保证「平均的点数 $\to\infty$」与「平均的邻域体积 $\to0$」同时成立，二者之比决定偏差–方差的平衡点。

**收敛速度。** 更精细的结论（Stone 的定理）给出偏差的量级：若 $d(x,x_{(k)})\asymp k^{1/p}$，则 1-最近邻的偏差是 $O\big(k^{1/p}\big)$（$f$ 局部 Lipschitz），方差是 $O\big(\frac{\sigma^2}{k}\big)$。综合成 EPE：

$$
\mathrm{EPE}(x)\ \lesssim\ C_1\,k^{2/p}+\frac{C_2\sigma^2}{k}
$$

第一项随 $p$ 增大而恶化（$2/p$ 趋近 0，$k^{2/p}$ 趋近 1，偏差不再随 $k$ 消失），第二项只能靠加大 $k$ 压下去，而 $k\le N$。这就是「最近邻的收敛率随维数下降」这句话的全部内容。

> **坑**
> 「$k/N\to0$ 且 $k\to\infty$」两个条件都不能省。若 $k$ 与 $N$ 同阶（$k=N$），估计量恒等于 $\bar y$，不是条件均值；若 $k$ 固定，则一致性失败。这解释了为什么实践中 $k$ 取 $N$ 的一个小比例。另一个容易忽略的条件是 $X$ 的边缘密度在 $x$ 处为正且连续；若 $x$ 落在边缘密度的零点集上（例如 $x$ 恰好在样本空间的边界），任何邻域的体积都相对过大，条件均值本身就病态。

> **坑**
> 「$k/N\to0$ 且 $k\to\infty$」两个条件都不能省。若 $k$ 与 $N$ 同阶（$k=N$），估计量恒等于 $\bar y$，不是条件均值；若 $k$ 固定，则一致性失败。这解释了为什么实践中 $k$ 取 $N$ 的一个小比例。

### 2.3.3 模型化方法：线性模型与加性模型 {#s-2-3-3}

如果已知结构，可以直接对 $f$ 加假设。最简单的是假设 $f(x)$ 近似线性：

$$
f(x)\approx x^\top\beta \eqno{2.15}
$$

把 (2.15) 代入 EPE (2.9)，得到的不是一个可以逐点最小化的函数（因为 $x^\top\beta$ 把不同 $x$ 绑在一起），而是一个**总体**最小化问题。用全方差公式（预备知识 P2）拆开：

$$
E\big[(Y-x^\top\beta)^2\big]=E\big[\mathrm{Var}(Y\mid X)\big]+E\big[(f(X)-x^\top\beta)^2\big]
$$

第一项与 $\beta$ 无关。最小化第二项：

$$
Q(\beta)=E\big[(f(X)-x^\top\beta)^2\big]=f(X)^\top f(X)-2\beta^\top E[Xf(X)]+\beta^\top E[XX^\top]\beta
$$

梯度（预备知识 L4，$A$ 对称时 $\nabla_\beta(\beta^\top A\beta)=2A\beta$）为

$$
\nabla_\beta Q=-2E[Xf(X)]+2E[XX^\top]\beta=0
$$

若 $E[XX^\top]$ 非奇异，解出

$$
\beta=\big[E(XX^\top)\big]^{-1}E(XY) \eqno{2.16}
$$

**与 (2.6) 的关系**：把 (2.16) 里的期望换成训练样本平均，$\frac1N\sum_i x_ix_i^\top\to E[XX^\top]$、$\frac1N\sum_i x_iy_i\to E[XY]$，再乘以 $N$，得到

$$
\hat\beta=\left(\frac1N\sum_{i=1}^Nx_ix_i^\top\right)^{-1}\frac1N\sum_{i=1}^Nx_iy_i=(X^\top X)^{-1}X^\top y
$$

正是 (2.6)。所以最小二乘是「在 $f$ 线性假设下，把 EPE 的总体最小化用样本矩替换」。**注意这里没有做条件化**——我们用对 $X$ 的信息汇聚代替了「在每个 $x$ 上取条件期望」。

若线性假设太强，退一步只保留可加性：

$$
f(X)=\sum_{j=1}^{p}f_j(X_j) \eqno{2.17}
$$

每个坐标函数 $f_j$ 可以任意非线性。最优估计用「同时做一维最近邻」逼近各个 $f_j$——用**加性**这个模型假设把高维条件期望的问题降成一维问题。注意这是**假设**而非结论：真实 $f$ 若含交互项 $X_1X_2$，可加模型的偏差就由那个交互项的大小决定。

推广一步是投影追踪（projection pursuit）：把 $f$ 写成若干个**一维非线性函数沿任意方向**的组合，$f(X)=\sum_m g_m(\alpha_m^\top X)$，方向 $\alpha_m$ 也从数据里学。它比可加模型强（可表达任意方向上的非线性），比全维非线性弱（只有 $M$ 个方向）。第 5 章的 MARS 同时含方向选择与基选择。

换成 $L_1$ 损失 $E|Y-f(X)|$，解不再是条件均值而是条件中位数：

$$
\hat f(x)=\mathrm{median}(Y\mid X=x) \eqno{2.18}
$$

**推导 · 为什么是中位数。** 固定 $x$，令 $g_x(c)=E|Y-c|\mid X=x$，其导数（几乎处处）

$$
g_x'(c)=E\big[-\mathbf{1}\{Y<c\}+\mathbf{1}\{Y>c\}\mid X=x\big]=\Pr(Y>c\mid X=x)-\Pr(Y<c\mid X=x)
$$

导数为零要求两侧概率相等，即 $c$ 把条件分布的两侧质量对半分——这正是中位数。它与均值的差别在尾部：均值受极端值线性影响，中位数只受**位置**影响，所以更稳健。$L_1$ 的缺点是导数不连续，妨碍牛顿法一类的算法。

### 2.3.4 分类：0-1 损失与 Bayes 规则 {#s-2-3-4}

定性输出需要另一个损失函数。用 $K\times K$ 矩阵 $L$ 表示代价，$L(k,\ell)$ 是把真实类别 $\mathcal{G}_k$ 判成 $\mathcal{G}_\ell$ 的代价，对角为 0、其余非负。最常用的是 **0-1 损失**：所有错分各记 1 分。期望预测误差变为

$$
\mathrm{EPE}=E\big[L\big(G,\hat G(X)\big)\big] \eqno{2.19}
$$

同样做条件化（这次是对 $G$）：

$$
\mathrm{EPE}=E_X\sum_{k=1}^{K}L\big[\mathcal{G}_k,\hat G(X)\big]\Pr(\mathcal{G}_k\mid X) \eqno{2.20}
$$

于是可以逐点最小化，$\hat G(x)$ 应最小化条件风险 $R(g\mid x)$（见预备知识 O5）：

$$
\hat G(x)=\underset{g\in\mathcal{G}}{\mathrm{argmin}}\ \sum_{k=1}^{K}L(\mathcal{G}_k,g)\Pr(\mathcal{G}_k\mid X=x) \eqno{2.21}
$$

**0-1 损失的化简**：此时 $L(\mathcal{G}_k,g)=\mathbf{1}\{g\ne\mathcal{G}_k\}=1-\mathbf{1}\{g=\mathcal{G}_k\}$，代入 (2.21) 的目标函数

$$
R(g\mid x)=1-\sum_{k=1}^K\mathbf{1}\{g=\mathcal{G}_k\}\Pr(\mathcal{G}_k\mid X=x)=1-\Pr(g\mid X=x)
$$

所以

$$
\hat G(x)=\underset{g\in\mathcal{G}}{\mathrm{argmin}}\ \big[1-\Pr(g\mid X=x)\big] \eqno{2.22}
$$

$$
\hat G(x)=\mathcal{G}_k\quad\text{若}\quad \Pr(\mathcal{G}_k\mid X=x)=\max_{g}\Pr(g\mid X=x) \eqno{2.23}
$$

这就是 **Bayes 分类器**：分到后验概率最大的类。它逐点最小化条件风险，因而在全样本上最小化 EPE（预备知识 O5 的塔性质论证：$R(g^\star)=E[R(g^\star\mid X)]\le E[R(g\mid X)]=R(g)$）。它的错分率叫 **Bayes 率**，是不可逾越的下界。

**平方误差与 0-1 损失的桥**：若用指示变量编码，则 $E(Y_k\mid X)=E[\mathbf{1}\{G=k\}\mid X]=\Pr(G=\mathcal{G}_k\mid X)$。因此

$$
E(Y_k\mid X=x)=\Pr(G=\mathcal{G}_k\mid X=x)
$$

「用回归拟合指示变量、再分到最大拟合值的类」正是 Bayes 分类器的一种表示。第 2.1 节的阈值规则 (2.7) 就是这一表示的具体化。$k$-最近邻分类同样是 Bayes 规则的实现：邻域内多数投票 = 邻域内的多数类，只是「点上条件」放松成「区域内条件」，「概率」用样本比例估计。

> **延伸** · 误差率、敏感度与特异度
>
> 设 $N$ 个测试点，$\mathrm{TP}$ 为真阳、$\mathrm{FP}$ 为假阳、$\mathrm{FN}$ 为假阴、$\mathrm{TN}$ 为真阴（负类记为「无该病」）。0-1 损失下的错分率是
>
> $$\mathrm{Error}=\frac{\mathrm{FP}+\mathrm{FN}}{N}$$
>
> 敏感度与特异度是它的两个条件版本：
>
> $$\mathrm{Sensitivity}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}=1-\Pr\big(\text{假阴}\big),\qquad \mathrm{Specificity}=\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}=1-\Pr\big(\text{假阳}\big)$$
>
> 用 Bayes 公式把两个条件概率换成后验就得到「两个类里各有多少比例被判错」。图 2.4 的 Bayes 错误率就是「重叠区域的面积」，由后验交叉点决定。这些量在第 9–10 章比较模型时反复使用。

> **坑**
> $\hat f(X)$ 可能是负数或大于 1（线性回归的刚性所致，见 §2.1），把它当作概率使用要注意。特别地，当预测点落在训练数据的凸包之外时，线性回归外推常常给出 $\hat f<0$，此时「分到最大拟合值」的规则仍能给出确定的类，但不再是概率意义上的最优。

---

## 2.4 高维中的局部方法：维数灾难 {#s-2-4}

<a class="src" href="../esl/ch02-overview-of-supervised-learning.html#s-2-5">原文 §2.5</a>

### 2.4.1 邻域体积与最近邻距离 {#s-2-4-1}

**问题**：$N$ 个均匀分布在 $p$ 维单位立方体的点，要抓取其中比例 $r$ 的点来做一个局部平均，邻域必须有多大？由此推出：从原点到最近数据点的中位距离是多少（公式 (2.24)，对应原书练习 2.3）？

**第一步：立方体。** 以目标点为中心放一个边长 $e$ 的子立方体。因为 $X$ 在 $[0,1]^p$ 上均匀分布，任一坐标落在区间内的概率等于区间长度，所以子立方体的体积比例是 $e^p$。要抓取比例 $r$，令 $e^p=r$：

$$
e_p(r)=r^{1/p}
$$

$p=10$ 时 $e_{10}(0.01)=0.63$、$e_{10}(0.1)=0.80$：为了用 1% 或 10% 的数据做局部平均，每个输入的取值范围要覆盖 63% 或 80%。这样的邻域已经**不局部**了。减小 $r$ 也无济于事——平均的点越少，方差越大。这已经是「维数灾难」（Bellman, 1961）的第一个表现。

**第二步：球。** 现在算原点到 $N$ 个均匀分布在 $p$ 维**单位球** $B_p=\{x\in\mathbb{R}^p:\lVert x\rVert\le1\}$ 内的点的最近距离的中位数。所需的第一步事实是

$$
\Pr\big(\lVert X\rVert\le r\big)=r^p
$$

推导：$p$ 维球体积随半径的 $p$ 次幂变化（极坐标下体积积分 $V_p(r)=\int_0^r s^{p-1}\Omega_{p-1}ds=\Omega_{p-1}r^p/p$，其中 $\Omega_{p-1}=2\pi^{p/2}/\Gamma(p/2)$ 是 $(p-1)$ 维球面面积）。均匀分布的密度是常数 $1/V_p(1)$，故 $\Pr(\lVert X\rVert\le r)=\Omega_{p-1}r^p/(pV_p(1))=r^p$，其中 $V_p(1)=\pi^{p/2}/\Gamma(p/2+1)$ 是单位球体积。用 $p=2$ 校验：$\Pr\le r=r^2$ ✓；$p=1$：$\Pr\le r=r$ ✓；$p=3$：$\Pr\le r=r^3$ ✓。

令 $R=\min_{1\le i\le N}\lVert X_i\rVert$。事件 $R>r$ **当且仅当**全部 $N$ 个点都落在半径 $r$ 的球外，故由独立性

$$
\Pr(R>r)=\prod_{i=1}^N\Pr\big(\lVert X_i\rVert>r\big)=\big(1-r^p\big)^N
$$

令 $R$ 的中位数为 $d$（即 $\Pr(R>d)=1/2$，中位数比均值稳健，见 §2.3.3 的 (2.18)）：

$$
\big(1-d^p\big)^N=\frac12\ \Longrightarrow\ 1-d^p=2^{-1/N}\ \Longrightarrow\ d^p=1-2^{-1/N}
$$

**化为闭式。** 记 $t=1/N\to0$，则 $2^{-1/N}=\exp(-t\ln2)=1-t\ln2+\frac{t^2(\ln2)^2}{2}+O(t^3)$（$e^{-u}$ 的泰勒展开，预备知识 C1），故

$$
1-2^{-1/N}=t\ln2-\tfrac12t^2(\ln2)^2+O(t^3)
$$

取 $1/p$ 次方（正数开方与乘除可交换）：

$$
d(p,N)=\Big(1-2^{-1/N}\Big)^{1/p}=\left[\frac{\ln2}{N}\Big(1-\frac{\ln2}{2N}+O(N^{-2})\Big)\right]^{1/p}\approx\left(\frac{\ln2}{N}\right)^{1/p}
$$

$$
d(p,N)=\left(1-2^{-1/N}\right)^{1/p}\approx\left(\frac{\ln2}{N}\right)^{1/p} \eqno{2.24}
$$

数值校验（$N=500,\ p=10$）：$(1-2^{-1/500})^{1/10}=0.5178$，$(\ln2/500)^{1/10}=0.5178$，与原文的 $0.52$ 一致。$p=1$ 时 $d\approx0.693/N$，中位最近邻距约为「平均间距的一半」，符合直觉；$p\to\infty$ 时 $(\ln2/N)^{1/p}=\exp\big(\frac1p\ln\frac{\ln2}{N}\big)\to1$，即最近点几乎跑到球面上。所以 $p=10$ 时一半以上的数据点离原点比离任何其他数据点更远。

**第三步：中位距离为什么「有内容」。** 换一种方式推导可以交叉验证 (2.24)。对 $R=\min_i\lVert X_i\rVert$ 求密度函数 $f_R(r)=\frac{d}{dr}\big[1-(1-r^p)^N\big]=Np r^{p-1}(1-r^p)^{N-1}$，中位数满足 $\int_0^{d}f_R=1/2$，即 $(1-d^p)^N=1/2$，与上面的生存函数推导一致。再看「典型距离」：若用 $E[R^2]$ 代替中位数，则由 $R^2$ 的分布（$R^2$ 的生存函数为 $(1-s^{p/2})^N$）得

$$
E\big[R^2\big]=\int_0^1\big(1-s^{p/2}\big)^N\,ds\ \xrightarrow{N\to\infty}\ \frac{2}{p}\cdot\frac{\ln 2}{N}
$$

（用 $(1-u)^N\approx e^{-Nu}$，大 $N$ 近似）。$p$ 大时 $E[R^2]\sim\frac{1.386}{pN}\to0$，与「最近点几乎在球面上」的结论一致（$R$ 不趋于 1，但 $R^2$ 的均值在 $p\to\infty$ 时因球的可测体积效应而变小——这是两种极限，不能混）。

（原文另提到「到最近点的**平均**距离有一个更复杂的表达式」。可以验证：中位距离由 $\Pr(R\le r)=1-2^{-1/N}$ 给出，而均值要算 $\mathbb{E}[R]=\int_0^1\Pr(R>r)dr=\int_0^1(1-r^p)^Ndr$，用 $u=r^p$ 换元得 $\frac1p\int_0^1u^{1/p-1}(1-u)^Ndu=\frac1p\,B\!\left(\frac2p,N+1\right)$，其中 $B$ 是 Beta 函数；$p$ 大时 Beta 函数集中于 $u\approx0$，均值与中位数同阶但常数不同。）

**为什么会出事**：在高维里几乎所有样本点都在样本空间的「边缘」，预测时必须**外推**而不是插值，误差结构与低维完全不同。

**边缘效应的另一种说法。** 练习 2.4 把它推广到球面（球状多元正态）分布：设 $X\sim N(0,I_p)$，则 $\lVert X\rVert^2\sim\chi^2_p$、期望为 $p$。取测试点 $x_0$、令 $a=x_0/\lVert x_0\rVert$，$z_i=a^\top x_i$。因为 $a$ 是单位向量，$\mathrm{Var}(z_i)=a^\top I_pa=1$，即 $z_i\sim N(0,1)$、期望平方距离为 1；而目标点的期望平方距离是 $p$。$p=10$ 时随机测试点离原点约 $\sqrt{10}=3.1$ 个标准差，训练点沿方向 $a$ 只有约 1 个标准差。所以大多数预测点「自认为处在训练集的边缘」。这与 (2.24) 是同一件事的两种样本空间。

**第三个表现：采样密度。** 要让每个坐标方向的单位体积里都有约 1 个样本，需要的样本量随维数指数增长：等密度的样本量正比于 $N^{1/p}$，所以一维用 100 个点算「密集」，十维就要 $100^{10}$ 个点。形式化地说，两个问题密度相等要求 $N_1^{1/p_1}=N_2^{1/p_2}$。

### 2.4.2 1-最近邻的偏差–方差分解 {#s-2-4-2}

**问题**：无噪声情形 $Y=f(X)=e^{-8\lVert X\rVert_2^2}$，$X$ 在 $[-1,1]^p$ 上均匀，$N=1000$，在 $x_0=0$ 处用 1-最近邻预测 $\hat y_0$。这个估计有多准？（图 2.7）

因为问题确定性，误差就是均方误差 $\mathrm{MSE}(x_0)=E_{\mathcal{T}}[f(x_0)-\hat y_0]^2$，对所有大小为 1000 的样本集取期望。要把它拆成可解释的两块。

**推导 · 核心代数恒等式（这一步必须自己算，不要背）。** 记 $a=f(x_0)$ 为常数，$b=\hat y_0$ 为随机量。把 $b-a$ 写成「中心化项 + 均值项」：

$$
b-a=\big(b-Eb\big)+\big(Eb-a\big)
$$

平方并取期望（交叉项单独算）：

$$
E\big[(b-a)^2\big]=E\big[(b-Eb)^2\big]+2E\big[(b-Eb)(Eb-a)\big]+\big(Eb-a\big)^2
$$

交叉项为零，因为 $Eb-a$ 是**确定性常数**，可以提到期望号外：

$$
2(Eb-a)E[b-Eb]=2(Eb-a)\big(Eb-Eb\big)=0
$$

第一项按定义是方差 $\mathrm{Var}(b)$（预备知识 P1：$\mathrm{Var}(Z)=E[Z^2]-E[Z]^2$）。因此**任何**随机量与常数的均方误差都能这样拆：

$$
E\big[(b-a)^2\big]=\mathrm{Var}(b)+\big(Eb-a\big)^2 \qquad(\star)
$$

注意 $(\star)$ 的两个前提都被用到了：$a$ 是常数（否则交叉项只剩条件正交性），期望与方差对**同一个**随机变量取。换一种写法可以看清同样的结构：把 $\hat y_0$ 写成 $\mathbb{E}_{\mathcal{T}}\hat y_0+\varepsilon_{\rm res}$（残差部分均值为零），则

$$
E\big[(\hat y_0-f(x_0))^2\big]=\underbrace{E\big[\varepsilon_{\rm res}^2\big]}_{\mathrm{Var}(\hat y_0)}+\underbrace{\big(\mathbb{E}_{\mathcal{T}}\hat y_0-f(x_0)\big)^2}_{\mathrm{Bias}^2}
$$

两项分别度量「重复抽样时估计量抖动多大」与「估计量的中心离真值多远」。对 $a=f(x_0)$、$b=\hat y_0$ 即得原文的分解：

$$
=\mathrm{Var}_{\mathcal{T}}(\hat y_0)+\mathrm{Bias}^2(\hat y_0) \eqno{2.25}
$$

**两项的定性行为。** 除最近邻恰好落在 0 之外，$\hat y_0< f(x_0)=1$，所以估计平均被**向下偏**：偏差的来源是最近邻离目标点有距离 $d(p,N)$，而 $f(x)=e^{-8\lVert x\rVert_2^2}$ 在原点附近陡降，于是 $E[f(X_{(1)})]<f(0)$。定量地，由 (2.24) 有 $\Pr(R>r)=1/2$ 在 $r=d(p,N)$ 处成立，故偏差的量级约为 $f(d)-\tfrac12\big[f(d)+f(\text{更远的典型点})\big]$，随 $d$ 增大而增大。低维时最近邻几乎贴着原点，$d\to0$，$f(x)\approx1-8\lVert x\rVert^2$，故 $\mathrm{Bias}\approx-8\,E[\lVert X_{(1)}\rVert^2]\to0$。维数增大后最近邻四散，$d\to1$（§2.4.1），偏差平方增大到极限 $\big(f(0)-f(1)\big)^2$ 附近，方差也先升后降——$p=10$ 时超过 99% 的样本里最近邻距原点超过 0.5，$\hat y_0$ 常常接近 0，MSE 就稳定在 $1.0$ 附近，方差下降只是这个例子的假象（图 2.7 右下）。

**对照图 2.8 的情形。** 把真函数换成只依赖一个坐标的 $f(x)=2\cdot\mathbf 1\{(x_1+1)^3\le1\}$（在 $\lVert x\rVert\to0$ 邻域内近似常数），此时 $f(x_0)-f(x_{(1)})$ 的期望几乎为零，偏差项塌掉，剩下的是**最近邻位置的抽样抖动**：$E[\lVert X_{(1)}\rVert^2]\approx(\ln2/N)^{2/p}$ 不随维数减小，故 $\mathrm{Var}$ 在 $p$ 增大时先升后降而**偏差**始终很小——图 2.8 里方差主导。这说明「哪一项主导」取决于真函数依赖多少个维度，而不是方法本身的好坏。

**为什么这个例子能说明方法本身无害。** $f(x)=e^{-8\lVert x\rVert^2}$ 是所有 $p$ 个坐标的**交互**函数：改变任何一个坐标都会改变输出，且组合起来的复杂度随 $p$ 指数增长。图 2.8 的 $f$ 只依赖 $X_1$，其余 $p-1$ 个坐标是纯噪声维度——对它们积分掉之后问题回到一维，1-最近邻的「邻域被撑大」这一劣势不再体现。两图对比给出全书反复强调的结论：**危险的不是维数 $p$，而是「真函数在 $p$ 个方向上的有效复杂度」**。

**一个可复核的数值例子。** 取 $p=1$、$N=1000$、$X\sim U[-1,1]$、$f(x)=e^{-8x^2}$。此时 $d(1,1000)=(1-2^{-1/1000})^{1}\approx6.93\times10^{-4}$，最近邻距原点不到 $0.0007$。展开 $f$：$1-f(x)\approx8x^2$，故

$$
\mathrm{Bias}\approx-8\,E[\lVert X_{(1)}\rVert^2]\approx-8\cdot\frac{\ln2}{1000}\approx-5.5\times10^{-3},\qquad \mathrm{Bias}^2\approx3.1\times10^{-5}
$$

$$
\mathrm{Var}\big(f(X_{(1)})\big)\approx\big(16\,E[\lVert X_{(1)}\rVert^2]\big)^2\approx(1.1\times10^{-2})^2\approx1.2\times10^{-4}
$$

两项都是 $10^{-4}$ 量级、远小于 $f(0)^2=1$，MSE 几乎为 0。若把 $p$ 加到 10（仍假设 $f$ 只依赖 $X_1$），结果不变；若把 $f$ 改成真的依赖全部 10 个坐标，两项都会跳到 $O(1)$，MSE 稳定在 1.0 附近（图 2.7 右下）。这两组数字可以直接复核。

> **结果**
> (2.25) 叫**偏差–方差分解**：期望平方误差总能拆成「估计量本身的方差」加「估计量均值的平方偏差」。这条恒等式是第 2.7 节、第 7 章 bootstrap、第 8 章 bagging 与平均、第 15 章随机森林的公共工具。它不依赖任何模型，只依赖「参数与预测值分开」这一结构。

> **坑**
> (2.25) 里的期望是**对训练样本集 $\mathcal{T}$** 取的，不是对 $X$ 取。若把 $X$ 的随机性也放进去，就得到 §2.7.1 的 (2.46)，那里多出一个 $\sigma^2$（新测试点自身的噪声方差）。两式的差别就是「训练误差」与「测试误差」在概念上的差别，初学者常把两者混为一谈。

### 2.4.3 最小二乘的 EPE 与 (p/N) 爆炸 {#s-2-4-3}

现在换成完全相反的模型：已知 $Y$ 与 $X$ 是线性关系

$$
Y=X^\top\beta+\varepsilon,\qquad \varepsilon\sim N(0,\sigma^2)\ \text{且与}\ X\ \text{独立} \eqno{2.26}
$$

用最小二乘拟合，在 $x_0$ 处预测 $\hat y_0=x_0^\top\hat\beta$。

**推导 · 预测误差的精确分解。** 由 (2.6) 的推导，$\hat\beta-\beta=(X^\top X)^{-1}X^\top\varepsilon$，其中 $\varepsilon=(\varepsilon_1,\dots,\varepsilon_N)^\top$，所以

$$
\hat y_0-x_0^\top\beta=x_0^\top(\hat\beta-\beta)=\sum_{i=1}^N\underbrace{x_0^\top(X^\top X)^{-1}x_i}_{\ell_i(x_0)}\varepsilon_i
$$

$\ell_i(x_0)$ 正是 $X(X^\top X)^{-1}x_0$ 的第 $i$ 个元素。于是把 $\varepsilon_0$ 也放进来，

$$
Y_0-\hat y_0=\varepsilon_0-\sum_{i=1}^N\ell_i(x_0)\varepsilon_i
$$

对它平方再对 $(\varepsilon_0,\varepsilon_1,\dots,\varepsilon_N)$ 取期望。三类项分别处理：

1. **$\varepsilon_0^2$**：$E[\varepsilon_0^2]=\sigma^2$。
2. **交叉项** $-2\varepsilon_0\sum_i\ell_i\varepsilon_i$：$E[\varepsilon_0\varepsilon_i]=0$（$i\ge1$，独立且零均值，预备知识 P1），$\ell_i$ 只依赖 $X$ 不依赖 $\varepsilon$，故整项期望为 0。
3. **$(\sum_i\ell_i\varepsilon_i)^2$**：$\sum_i\sum_j\ell_i\ell_jE[\varepsilon_i\varepsilon_j]=\sum_i\ell_i^2\sigma^2$（非对角项为 0）。

所以 $E[(Y_0-\hat y_0)^2\mid X]=\sigma^2+\sigma^2\sum_i\ell_i(x_0)^2$。这个和正是二次型的迹（预备知识 L3、L5）：

$$
\sum_i\ell_i(x_0)^2=x_0^\top(X^\top X)^{-1}\underbrace{\Big(\sum_i x_ix_i^\top\Big)}_{X^\top X}x_0^\top(X^\top X)^{-1}x_0
$$

等等，$\sum_i\ell_i(x_0)^2=\sum_i\big[x_0^\top(X^\top X)^{-1}x_i\big]^2=x_0^\top(X^\top X)^{-1}\Big[\sum_ix_ix_i^\top\Big](X^\top X)^{-1}x_0$，而 $\sum_ix_ix_i^\top=X^\top X$，故

$$
\sum_i\ell_i(x_0)^2=x_0^\top(X^\top X)^{-1}X^\top X(X^\top X)^{-1}x_0=x_0^\top(X^\top X)^{-1}x_0
$$

中间那一步用了「$X^\top X$ 与它的逆交换」，不需要正定性以外的条件（$(X^\top X)^{-1}X^\top X=I$）。$x_0$ 固定时它是确定性量，可以并入外层期望。于是得到原文的式子：

$$
\mathrm{EPE}(x_0)=\sigma^2+E_{\mathcal{T}}\ x_0^\top(X^\top X)^{-1}x_0\ \sigma^2 \eqno{2.27}
$$

三项含义：$\sigma^2$ 是**不可约误差**（$Y$ 本身含噪声，即使知道真 $\beta$ 也消不掉）；第二项是估计 $\beta$ 带来的方差；**没有偏差项**，因为线性模型设定正确时最小二乘无偏（$E\hat\beta=\beta$）。

**再对设计取期望，$p$ 显形。** 若 $N$ 大且 $\mathcal{T}$ 随机、$E[X]=0$，则 $\frac1N X^\top X=\frac1N\sum_ix_ix_i^\top\to E[XX^\top]=\mathrm{Cov}(X)$，于是

$$
E_{x_0}\mathrm{EPE}(x_0)\approx E_{x_0}\ x_0^\top\mathrm{Cov}(X)^{-1}x_0\ \frac{\sigma^2}{N}+\sigma^2
$$

要算第一个期望，用迹循环律把二次型写成迹（预备知识 P4、L5）。设 $x_0$ 与训练样本同分布，$C=\mathrm{Cov}(X)$，则 $E[x_0x_0^\top]=C$（因 $E[X]=0$），且 $C$ 与 $C^{-1}$ 都是对称的，故

$$
E_{x_0}\big[x_0^\top C^{-1}x_0\big]=E_{x_0}\Big[\mathrm{tr}\big(C^{-1}x_0x_0^\top\big)\Big]=\mathrm{tr}\Big(C^{-1}E[x_0x_0^\top]\Big)=\mathrm{tr}(C^{-1}C)=\mathrm{tr}(I_p)=p
$$

第二步用迹的线性性把期望与迹交换（练习 2.5(b) 提示的正是这一步）。所以

$$
=\mathrm{trace}\big[\mathrm{Cov}(X)^{-1}\mathrm{Cov}(x_0)\big]\frac{\sigma^2}{N}+\sigma^2=\sigma^2\frac{p}{N}+\sigma^2 \eqno{2.28}
$$

**结论**：最小二乘的期望预测误差随 $p$ **线性**增长，斜率 $\sigma^2/N$；确定性情形（$\sigma^2=0$）里第二项完全消失。$N=500,\ \sigma^2=1$ 时，$p=10$ 只增加 0.02，可以忽略；这就是「靠强假设换来避开维数灾难」的定量含义。

> **坑**
> (2.28) 里的 $E_{x_0}$ 是**对一个新的输入点**再取一次期望（书里明确写了 $E_{x_0}$）。若把 $x_0$ 固定在原点，则 $x_0=0$，第二项恒为 0，EPE 就是 $\sigma^2$，与 $p$ 无关——这与图 2.9 中「$x_0=0$ 处的相对 EPE 恒在 2 以上」并不矛盾，因为那里比的是 1-最近邻与最小二乘之**比**，而且用的是线性真实函数。另外，若 $p\ge N$，$X^\top X$ 奇异，(2.6)(2.27) 都不存在；此时最小二乘已经在 $p$ 这一维上输给了「不估 $\beta$」的做法。

<a class="src" href="../esl/ch02-overview-of-supervised-learning.html#s-2-6">原文 §2.6</a>

---

## 2.5 统计模型、监督学习与函数逼近 {#s-2-5}

有了损失与 EPE，下一步是引入**模型**。加性误差模型是所有回归方法的默认出发点：

$$
Y=f(X)+\varepsilon,\qquad E(\varepsilon)=0,\ \varepsilon\perp X \eqno{2.29}
$$

**这个假设带来什么**：条件分布 $\mathrm{Pr}(Y\mid X)$ 只通过条件均值 $f(x)=E(Y\mid X=x)$ 依赖于 $X$。若允许 $\mathrm{Var}(Y\mid X=x)=\sigma^2(x)$，均值和方差都依赖 $X$，加性模型就排除了这种情况。

**(a) 两个阶段的学习范式。** 把参数化函数族写成线性基展开

$$
f_\theta(x)=\sum_{k=1}^{K}h_k(x)\,\theta_k \eqno{2.30}
$$

$K$ 个基函数 $h_k$ 可以是 $x_1^2,x_1x_2,\cos(x_1)$ 等多项式/三角展开（此时基固定、只有 $\theta$ 待估，问题仍是线性最小二乘），也可以是 sigmoid 非线性展开

$$
h_k(x)=\frac{1}{1+\exp(-x^\top\theta_k)} \eqno{2.31}
$$

用最小二乘定 $\theta$，仍以残差平方和为目标：

$$
\mathrm{RSS}(\theta)=\sum_{i=1}^{N}\big(y_i-f_\theta(x_i)\big)^2 \eqno{2.32}
$$

这就是「监督学习」的两阶段：**训练**（用 $\mathcal{T}=\{(x_i,y_i)\}$ 求 $\hat\theta$，让 RSS 最小）与**测试**（对新输入报 $\hat y(x_0)=f_{\hat\theta}(x_0)$）。

**(b) 极大似然与最小二乘的等价。** 最小二乘只是极大似然的一个特例。若 $y_1,\dots,y_N$ 是密度 $\mathrm{Pr}_\theta(y)$ 的随机样本，对数似然为

$$
L(\theta)=\log\mathrm{Pr}_\theta(y)=\sum_{i=1}^N\log\mathrm{Pr}_\theta(y_i) \eqno{2.33}
$$

对加性误差模型 $Y=f_\theta(X)+\varepsilon$、$\varepsilon\sim N(0,\sigma^2)$，条件似然是

$$
\mathrm{Pr}(Y\mid X,\theta)=N\big(f_\theta(X),\sigma^2\big) \eqno{2.34}
$$

**推导 · 两者的等价。** 多元正态密度 $N(\mu,\sigma^2)$（$p=1$）为 $\frac{1}{\sqrt{2\pi}\sigma}\exp\big(-\frac{(y-\mu)^2}{2\sigma^2}\big)$，取对数再乘 $N$：

$$
L(\theta)=-\frac{N}{2}\log(2\pi)-N\log\sigma-\frac{1}{2\sigma^2}\sum_{i=1}^N\big(y_i-f_\theta(x_i)\big)^2 \eqno{2.35}
$$

对 $\theta$ 求极大等价于极小化最后一项，即极小化 RSS (2.32)。所以「加正态误差 + 极大似然」与「平方误差 + 最小二乘」是同一件事——正态假设看起来更强，结果却一样。

**(c) 定性输出的似然：交叉熵。** 对 $G$，假设 $\mathrm{Pr}(G=\mathcal{G}_k\mid X=x)=p_{k,\theta}(x)$，则对数似然（也叫交叉熵）是

$$
L(\theta)=\sum_{i=1}^N\log p_{g_i,\theta}(x_i) \eqno{2.36}
$$

极大化它得到「似然意义下最符合数据」的 $\hat\theta$。$K=2$ 时 (2.36) 退化为逻辑回归的对数似然（见第 4 章 (4.20)），这也是逻辑回归自然的来历：它就是对 (2.36) 的具体化。

| 判据 | 目标 | 需要假设 | 适用 |
|---|---|---|---|
| 平方误差 | $\sum_i(y_i-f_\theta(x_i))^2$ | $Y=f(X)+\varepsilon$，$\varepsilon$ 零均值 | 定量输出 |
| 高斯似然 | $-\frac{1}{2\sigma^2}\sum_i(y_i-f_\theta(x_i))^2+\text{常数}$ | 同上 + $\varepsilon\sim N(0,\sigma^2)$ | 定量输出 |
| 多项似然 | $\sum_i\log p_{g_i,\theta}(x_i)$ | $\mathrm{Pr}(G\mid X)$ 的形式 | 定性输出 |

---

## 2.6 结构化回归模型：三类受限估计量 {#s-2-6}

### 2.6.1 问题的难度与有效自由度 {#s-2-6-1}

对任意函数 $f$，残差平方和

$$
\mathrm{RSS}(f)=\sum_{i=1}^N\big(y_i-f(x_i)\big)^2 \eqno{2.37}
$$

的极小解有**无穷多个**：任何通过全部训练点 $(x_i,y_i)$ 的函数都是解。如果每个 $x_i$ 处有重复观测 $(x_i,y_{i\ell})$，$\ell=1,\dots,N_i$，解被迫通过这些点的**均值**（练习 2.6 把它化成一个加权最小二乘问题）——但这只在 $N_i$ 大时才唯一。(2.37) 是 §2.3.1 中 (2.11) 的有限样本版本。

要让 $N$ 有限时结果有用，必须**限制候选函数集**。用数学语言写就是把 (2.37) 的极小化限制在某个集合 $\mathcal F$ 上：

$$
\hat f=\underset{f\in\mathcal F}{\mathrm{argmin}}\ \mathrm{RSS}(f)
$$

限制的来源永远是数据之外的（模型形式，或学习算法本身），而且限制的方式有两种等价形式：

- **参数化**：$\mathcal F=\{f_\theta:\theta\in\mathbb R^K\}$，参数个数有限，等价于 $\hat\theta=(H^\top H)^{-1}H^\top y$，$H_{ik}=h_k(x_i)$；
- **局部性**：$\mathcal F$ 是「在输入空间的每个小球邻域内 $f$ 近似常数、线性或低阶多项式」的函数之集，靠平均得到估计。

关键在于：**任何能让 (2.37) 有唯一解的限制都没有真正消除歧义**。无穷多种限制各自给出唯一解，歧义只是从「解不唯一」搬到了「选哪个限制」。这就是为什么全书大量篇幅在讨论「怎么选约束」，而约束的强度最终都可以折成一个数——平滑参数 $\lambda$、带宽、或基函数个数。

约束强度由邻域大小决定：邻域无限大 + 局部线性 = 全局线性模型（极强约束）；邻域趋于零 + 局部常数 = 无约束插值。**任何在小的各向同性邻域里做局部变化的估计量，在高维都会出问题**（§2.4.1）；反过来，所有能躲开维数灾难的方法都带一个隐式或自适应的度量，使邻域不可能在所有方向上同时小（第六章的「自适应度量核」、第十一章的隐层权重、第十二章的核技巧都属此类）。

举例说明「约束」的两种极端如何退化：取 (2.30) 的基为所有阶数 $\le M$ 的多项式，则 $M=1$ 给出全局线性（等价于最小二乘 (2.6)），$M$ 足够大且 $M>p$ 时可以插值任意 $N$ 个点。$M$ 就是这个模型族的平滑参数。

> **结果**
> 约束强度可用一个数字统一度量——**有效自由度** $\mathrm{df}=\mathrm{tr}(P_X)$（最小二乘）或 $\mathrm{df}\approx N/k$（$k$-最近邻）。它是「这个估计量等效于用多少个自由参数拟合」，后面 §2.7 的偏差–方差折中与 §2.7.3 的交叉验证偏差都靠它定量。对正则化的模型，有效自由度的一般定义是「惩罚后参数对拟合值的总影响」：

$$
\mathrm{df}(\lambda)=\frac{\partial\,\hat y(x)}{\partial\,y^{\top}}
$$

即把拟合值对观测值的雅可比矩阵取迹；最小二乘时它是 $P_X$，岭回归时变成 $H_\lambda=H_0(H_0^\top H_0+\lambda I)^{-1}H_0^\top$ 的迹（见第 3 章）。这个定义的好处是它对**任何**平滑方法都有意义，因而可以横向比较。

### 2.6.2 粗糙度惩罚与贝叶斯方法 {#s-2-6-2}

第一类方法用**显式的粗糙度惩罚**控制函数类：

$$
\mathrm{PRSS}(f;\lambda)=\mathrm{RSS}(f)+\lambda J(f) \eqno{2.38}
$$

$J(f)$ 由使用者选定，在 $f$ 变化过快时取大值。最经典的是一维三次光滑样条：

$$
\mathrm{PRSS}(f;\lambda)=\sum_{i=1}^N\big(y_i-f(x_i)\big)^2+\lambda\int\big[f''(x)\big]^2dx \eqno{2.39}
$$

$\lambda=0$ 时不惩罚，任何插值函数都是解（欠约束）；$\lambda\to\infty$ 时只允许关于 $x$ 线性的函数（极端约束）。中间取值给出一整族模型，$\lambda$ 索引「直线拟合 → 插值模型」这条路径。图 2.11 里横轴的「模型复杂度」对样条来说就是 $\lambda$ 的反向刻度。

**为什么 $\lambda$ 这么有用。** (2.39) 的极小化条件是一个**变分问题**。对 $\mathrm{PRSS}$ 关于 $f$ 求一阶变分并令零，可得 Euler–Lagrange 方程

$$
\sum_{i=1}^N\big(y_i-f(x_i)\big)\delta(x-x_i)+2\lambda f^{(4)}(x)=0
$$

（在 $\mathbb R$ 上、把 $f''$ 的能量积分分部积分两次）。这个四阶微分方程的特征方程是 $r^4=\omega^4$，通解含 $e^{\pm\omega x}$ 与 $x$ 的三次多项式；边界条件把解限制成一个 $M$ 维空间（$M\approx N^{1/4}$），每个折点贡献一个自由度。所以**增加一个节点就等于增加一个参数**，有效自由度随 $\lambda$ 单调变化——这正是「把 $\lambda$ 当复杂度指标」的严格依据。具体解法在第 5 章。

**贝叶斯解释**：把 $J(f)$ 当作对数先验、$\mathrm{PRSS}$ 当作对数后验，则最小化 $\mathrm{PRSS}$ 就是在找后验众数。第 5 章的粗糙度惩罚、第 8 章的贝叶斯视角用的是同一个形式。

**惩罚泛函可按结构定制**，这一条后面反复用到：

| 想要的结构 | 函数类 | 惩罚泛函 |
|---|---|---|
| 可加性（§2.3.3 的 (2.17)） | $f(X)=\sum_j f_j(X_j)$ | $J(f)=\sum_{j=1}^p J(f_j)$ |
| 投影追踪 | $f(X)=\sum_m g_m(\alpha_m^\top X)$ | $J(f)=\sum_m J(g_m)$，$\alpha_m$ 自选 |
| 光滑（样条） | $C^\infty$ 函数 | $J(f)=\int[f''(x)]^2dx$ |
| 岭回归（第 3 章） | 线性函数 | $J(f)=\lVert\beta\rVert_2^2$ |
| lasso（第 3 章） | 线性函数 | $J(f)=\lVert\beta\rVert_1$ |

最后两行是第 3 章的全部内容，可以看出「惩罚 = 先验」的统一视角在第一章就已经埋好。

### 2.6.3 核方法与局部回归 {#s-2-6-3}

第二类方法**显式规定邻域**。核函数 $K_\lambda(x_0,x)$ 给 $x_0$ 附近的点加权，高斯核按平方欧氏距离指数衰减：

$$
K_\lambda(x_0,x)=\exp\left[-\frac{\lVert x-x_0\rVert_2^2}{2\lambda}\right] \eqno{2.40}
$$

（$\lambda$ 是高斯密度的方差，控制邻域宽度。）最简单的核估计是 Nadaraya–Watson 加权平均：

$$
\hat f(x_0)=\frac{\sum_{i=1}^N K_\lambda(x_0,x_i)\,y_i}{\sum_{i=1}^N K_\lambda(x_0,x_i)} \eqno{2.41}
$$

**推导 · 核加权平均就是「局部常数最小二乘」的解（这一条把 §2.2.2 与本节统一起来）。** 局部回归的目标是在 $x_0$ 处最小化

$$
\mathrm{RSS}(f_\theta,x_0)=\sum_{i=1}^N K_\lambda(x_0,x_i)\big(y_i-f_\theta(x_i)\big)^2 \eqno{2.42}
$$

取 $f_\theta(x)=\theta$（局部常数），目标变成 $\sum_iK_i(y_i-\theta)^2$，$K_i=K_\lambda(x_0,x_i)$。逐项求导并令零：

$$
\frac{\partial}{\partial\theta}\sum_iK_i(y_i-\theta)^2=-2\sum_iK_i(y_i-\theta)=0
\ \Longrightarrow\ \hat\theta=\frac{\sum_iK_iy_i}{\sum_iK_i}
$$

代入即得 (2.41)。取 $f_\theta(x)=\theta_0+\theta_1x$ 则得到**局部线性回归**（更稳健，偏差更小）。

**偏差从哪里来，可以算出来。** 记核的「一阶矩」$m_K=\int v\,K(v)dv$，把 $y_i$ 换成 $f(x_i)+\varepsilon_i$ 代入 (2.41)：

$$
\hat f(x_0)=\frac{\sum_iK_i\,f(x_i)}{\sum_iK_i}+\frac{\sum_iK_i\varepsilon_i}{\sum_iK_i}
$$

第二项的方差 $=\frac{\sigma^2\sum_iK_i^2}{\big(\sum_iK_i\big)^2}\approx\frac{\sigma^2}{N\,\int K_\lambda^2}$（大样本近似，分母 $N\int K$、分子的 $\sum K_i^2\approx N\int K^2$）。第一项用 Taylor 展开 $f(x_i)\approx f(x_0)+\nabla f(x_0)^\top(x_i-x_0)$，其中 $x_i-x_0$ 的加权平均为 $-\nabla m_K\cdot\sqrt{2\lambda}$（高斯核时 $m_K=0$，故无此二阶偏差），偏差由 $\nabla f^\top\mathbb E_{\text{加权的 }K}[x_i-x_0]$ 决定。对高斯核这一项为零，于是高斯核估计的偏差**只来自有限样本的随机波动**——这是它比矩形（最近邻）核更常用的原因。

**高斯核是合法的权重**：$\int K_\lambda(x_0,x)dx=1$，因为令 $u=(x-x_0)/\sqrt{2\lambda}$，$\int e^{-\lVert u\rVert^2}du=1$（多维标准高斯积分），且 $\int u\,K_\lambda(x_0,x)dx=x_0-m_K$，其中 $m_K=\int vK(v)dv$ 对高斯核为 0。这两个事实合起来解释了后者的偏差为何消失（预备知识 C3 的卷积恒等式）。

**最近邻也是核方法**，只是核依赖数据：

$$
K_k(x,x_0)=I\big(\lVert x-x_0\rVert\le\lVert x_{(k)}-x_0\rVert\big)
$$

其中 $x_{(k)}$ 是按到 $x_0$ 的距离排在第 $k$ 位的训练点，$I(\cdot)$ 是示性函数。取局部常数最小二乘就回到 (2.8) 的普通平均。区别在度量的来源：最近邻的度量由**数据**决定，核方法的度量由**用户**指定。后者可以用高维改造（第六章）躲开支数灾难，前者不行。

### 2.6.4 基函数与字典方法 {#s-2-6-4}

第三类方法把 $f$ 写成基函数的线性展开：

$$
f_\theta(x)=\sum_{m=1}^{M}\theta_m\,h_m(x) \eqno{2.43}
$$

「线性」指的是对参数 $\theta$ 线性，基函数本身可以任意复杂。

- **样条**：一维 $x$ 上 $K$ 次样条由 $M$ 个样条基构成，由 $M-K-1$ 个节点确定，在节点之间是 $K$ 次分段多项式、且在节点处 $K-1$ 阶连续。线性样条的基为 $b_1(x)=1$、$b_2(x)=x$、$b_{m+2}(x)=(x-t_m)_+$。张量积可推广到高维（9 章的 CART、MARS）。
- **径向基函数**：中心为 $\mu_m$、尺度为 $\lambda_m$ 的对称 $p$ 维核，
  $$
  f_\theta(x)=\sum_{m=1}^{M}K_{\lambda_m}(\mu_m,x)\,\theta_m \eqno{2.44}
  $$
- **单隐层前馈网络**：$\sigma(z)=1/(1+e^{-z})$，
  $$
  f_\theta(x)=\sum_{m=1}^{M}\beta_m\,\sigma(\alpha_m^\top x+b_m) \eqno{2.45}
  $$

后两者的中心/方向/偏置必须从数据里学，这让问题从线性回归变成组合上困难的非线性问题，实践中用贪心或两阶段近似（§6.7）。这类「自选基」方法也称**字典方法**：有一个可能无限的候选基集合 $\mathcal{D}$，靠搜索机制逐步搭建模型。第九章的 CART 就是「字典 + 贪心搜索」的实例。

**为什么高维需要基函数。** 一维里，$M$ 个基函数表达 $M$ 个自由度；$p$ 维里若用张量积基（每个方向 $M$ 个），自由度的个数是 $M^p$ 而不是 $pM$——**函数空间的大小随维数指数增长**。要同样精细地拟合一个 $p$ 维函数，训练集必须指数增长。这就是维数灾难在模型选择一侧的表述，也是「用强假设限制函数类」（§2.6.1）能奏效的全部理由。样条用张量积基在多维展开（$M^d$ 个基），RBF 与神经网络则用「中心可学习」的方式绕过 $M^p$ 的组合爆炸。

| 类别 | 约束的来源 | 例子 | 参数个数 |
|---|---|---|---|
| 粗糙度惩罚 | 显式泛函 $J(f)$ 与 $\lambda$ | 光滑样条 (2.39) | 折数 |
| 核/局部 | 核宽度 $\lambda$、局部阶数 | Nadaraya–Watson (2.41)、局部线性 | 带宽 |
| 基函数/字典 | 基的个数 $M$、节点数 | 样条、RBF (2.44)、网络 (2.45) | 节点数 |

---

## 2.7 模型选择与偏差–方差折中 {#s-2-7}

<a class="src" href="../esl/ch02-overview-of-supervised-learning.html#s-2-9">原文 §2.9</a>

### 2.7.1 k-最近邻的 EPE 三项分解 {#s-2-7-1}

**问题**：(2.25) 只处理了确定性目标。这里要有噪声：设 $Y=f(X)+\varepsilon$，$E\varepsilon=0$、$\mathrm{Var}\varepsilon=\sigma^2$，并为简单起见**固定设计**（$x_i$ 不随机）。$x_0$ 处的 EPE 如何分解？各项如何随 $k$ 变化（公式 (2.46)(2.47)）？

**第一步：与 (2.25) 同样的结构。** 对固定的 $x_0$，$x_0$ 处的条件 EPE 是

$$
\mathrm{EPE}_k(x_0)=E\big[(Y-\hat f_k(x_0))^2\mid X=x_0\big]
$$

先用平方损失的全方差公式（预备知识 P2）把「条件期望」剥出来：记 $Y=f(x_0)+\varepsilon_0$、$\hat f_k(x_0)=\mathbb{E}_{\mathcal{T}}\big[\hat f_k(x_0)\mid X=x_0\big]$，则

$$
\mathrm{EPE}_k(x_0)=\mathrm{Var}(Y\mid x_0)+E\big[(\hat f_k(x_0)-f(x_0))^2\mid x_0\big]
$$

$$
=\sigma^2+\mathrm{Var}_{\mathcal{T}}\big(\hat f_k(x_0)\big)+\big[\mathbb{E}_{\mathcal{T}}\hat f_k(x_0)-f(x_0)\big]^2 \eqno{2.46}
$$

最后一步又是 §2.4.2 那个恒等式，只是这次第二个被减的量是随机量 $Y$ 的条件期望而寻常数，结论不变（中间交叉项因为 $E[(Y-f(x_0))\mid x_0]=0$ 而消失）。三项分别是：

1. $\sigma^2$：**不可约误差**，新测试点自身 $Y$ 的方差，即使知道真 $f(x_0)$ 也消不掉；
2. $\mathrm{Bias}^2$：真均值与估计量均值之差的平方，它**随 $k$ 增大而增大**——$k$ 小时最近的几个邻居的 $f$ 值都贴近 $f(x_0)$，$k$ 大时邻居跑远了，任何情况都可能发生；
3. $\mathrm{Var}$：平均的方差，**按 $1/k$ 衰减**。

(2.46) 是**结构性**的：它对任何把 $f(x_0)$ 估成 $\mathcal{T}$ 的函数量的估计量都成立，把它写成「测试误差 = 不可约误差 + 估计量的偏差平方 + 估计量的方差」就抓住了所有模型选择的本质。

**第二步：把 (2.46) 的两项算成具体形式，得 (2.47)。** 记 $Y_{(\ell)}=f(x_{(\ell)})+\varepsilon_{(\ell)}$，$\ell=1,\dots,k$，则 $\hat f_k(x_0)=\frac1k\sum_{\ell}Y_{(\ell)}$。把误差写成三项：

$$
Y-\hat f_k(x_0)=\underbrace{f(x_0)-\frac1k\sum_{\ell=1}^{k}f(x_{(\ell)})}_{\text{偏差项（确定性）}}+\underbrace{\big(\varepsilon_0-\overline{\varepsilon}_k\big)}_{\text{噪声项}},\qquad \overline{\varepsilon}_k=\frac1k\sum_{\ell=1}^k\varepsilon_{(\ell)}
$$

平方取期望，四个交叉项按「零均值 + 独立」逐个为零（$E[\varepsilon_0\varepsilon_{(\ell)}]=0$、$E[\overline\varepsilon_k^2]$ 里非对角项为 0）：

$$
\begin{aligned}
\mathrm{EPE}_k(x_0)&=\Big[f(x_0)-\tfrac1k\textstyle\sum_{\ell=1}^{k}f(x_{(\ell)})\Big]^2+E\big[(\varepsilon_0-\overline{\varepsilon}_k)^2\big]\\
&=\Big[f(x_0)-\tfrac1k\textstyle\sum_{\ell=1}^{k}f(x_{(\ell)})\Big]^2+\sigma^2-2\underbrace{E[\varepsilon_0\overline\varepsilon_k]}_{=0}+\underbrace{\frac{1}{k^2}\sum_{\ell}E[\varepsilon_{(\ell)}^2]}_{\sigma^2/k}\\
&=\sigma^2+\Big[f(x_0)-\tfrac1k\sum_{\ell=1}^{k}f(x_{(\ell)})\Big]^2+\frac{\sigma^2}{k}
\end{aligned}
$$

$$
=\sigma^2+\Big[f(x_0)-\frac1k\sum_{\ell=1}^{k}f(x_{(\ell)})\Big]^2+\frac{\sigma^2}{k} \eqno{2.47}
$$

**三项分别怎么随 $k$ 变。** 第三项严格是 $\sigma^2/k$（平均 $k$ 个独立同分布噪声的标准结果，预备知识 P1）。第二项记

$$
\mathrm{Bias}\big(f(x_0)\big)=\frac1k\sum_{\ell=1}^k\big[f(x_0)-f(x_{(\ell)})\big]
$$

用 Taylor 展开 $f(x_{(\ell)})=f(x_0)+\nabla f(x_0)^\top(X_{(\ell)}-x_0)+\ldots$，在 $f$ 二阶可微且邻域内梯度近似恒定时 $f(x_0)-\frac1k\sum_\ell f(x_{(\ell)})\approx-\nabla f(x_0)^\top\big(\frac1k\sum_\ell X_{(\ell)}-x_0\big)$，即偏差正比于「邻域质心离目标点的距离」，而这个距离按 (2.24) 的速度随 $k$ 增长（邻域扩大）。所以：

- $k\to1$：偏差 $\to0$（最近点几乎落在目标点），但方差项 $\to\sigma^2$；
- $k\to N$：偏差 $\to f(x_0)-\frac1N\sum_if(x_i)$（通常很大），方差项 $\to\sigma^2/N\to0$。

两条曲线交叉处给出最优 $k$ 的粗略位置：在偏差平方 $\approx c\,k^{2/p}$、方差 $\sigma^2/k$ 的模型下，最小化 $c k^{2/p}+\sigma^2/k$ 的一阶条件是 $\frac{2c}{p}k^{2/p-1}-\frac{\sigma^2}{k^2}=0$，即

$$
k^{\,2/p+1}=\frac{p\,\sigma^2}{2c}\ \Longrightarrow\ k_\star\approx\left(\frac{p\,\sigma^2}{2c}\right)^{\frac{p}{p+2}}
$$

即 $k_\star$ 随 $p$ 增大而迅速增大（$p=2$ 时指数 $p/(p+2)=1/2$，$p=10$ 时 $\approx0.83$）——与「维数灾难迫使我们用大 $k$」的经验一致，也解释了图 2.4 里测试误差的最低点在 $k$ 较大处。

**为什么必须用「测试误差」而不是「训练误差」找谷底。** 训练误差 (2.37) 在每个 $k$ 上都可以通过「把邻域缩小」而单调压低，最后必然选到 $k=1$；而 $\mathrm{EPE}$ 的谷底在 $k_\star$ 附近。两者之间没有单调对应关系，所以模型选择必须依赖某种**近似测试误差**的量——这正是交叉验证存在的理由（§2.7.3）。

> **结果**
> (2.46) 与 (2.47) 是同一个式子的两种写法：前者是可推广到任何估计量的**结构性**分解，后者是 $k$-最近邻的**具体**分解。图 2.11 画出训练误差与测试误差对复杂度的典型曲线：训练误差随复杂度单调下降（图中最左的蓝线），测试误差呈 U 形（橙线），U 形最低点就是上面 $k_\star$ 的图上对应物。一般地：**模型复杂度上升 $\Rightarrow$ 方差上升、偏差平方下降**；反之亦然。这就是**偏差–方差折中**的定量形式，而「选复杂度」就是「选 U 形的谷底」——但训练误差单调下降，不能用来找谷底，这就是第 7 章要解决的问题。

### 2.7.2 过拟合、有效自由度与训练/测试误差 {#s-2-7-2}

有了 (2.46)，就能解释「训练误差为什么不能用来选模型」。

**训练误差的塌陷。** 对任何能插值的模型族（$\lambda=0$ 的样条、$k=1$ 的最近邻、$M\ge N$ 的基展开），训练误差都能压到 0，但 EPE 的第二项（方差）会爆掉。这叫**过拟合**。所以 2.6 节的每个模型都带一个平滑/复杂度参数：惩罚系数 $\lambda$、核宽度、或基函数个数，它们索引「受限」到「插值」的一整族模型。

**用自由度数统一衡量。** 有效的复杂度用等效自由参数个数度量：

$$
\mathrm{df}=\begin{cases}\mathrm{tr}(P_X)=\sum_{i=1}^N h_{ii},&\text{最小二乘}\\ N/k,&\text{$k$-最近邻（邻域不重叠时精确）}\end{cases}
$$

$h_{ii}$ 是帽子矩阵的第 $i$ 个对角元（杠杆值）。$k$-最近邻有 $N/k$ 个互不重叠的邻域、每域拟合一个均值，故等效参数个数为 $N/k$——通常比最小二乘的 $p$ 大，且随 $k$ 增大而减小。

**训练误差与测试误差的定量关系。** 设 $\hat\beta=(X^\top X)^{-1}X^\top y$，$\hat y=P_Xy$，$e=y-\hat y$。利用 $\hat y_i=\sum_jP_{ij}y_j$ 可得

$$
y_i-\hat y_i=\sum_{j\ne i}(\delta_{ij}-P_{ij})\,y_j
$$

右边**不含 $y_i$**（这就是 $\mathrm{tr}(P)=p$ 的效应：投影把 $p$ 个自由度吸收掉了）。用 $y_j$ 独立同分布、$E[y_j^2]=m_2$ 计算

$$
E\big[(y_i-\hat y_i)^2\big]=m_2\sum_{j\ne i}(\delta_{ij}-P_{ij})^2
$$

对 $i$ 求和。记 $h_{ii}=P_{ii}$、$H_i=\sum_jP_{ij}^2=\mathrm{tr}(P^\top P)$ 的第 $i$ 列范数平方，则

$$
\sum_{j\ne i}(\delta_{ij}-P_{ij})^2=1-2h_{ii}+H_i-h_{ii}^2
$$

于是训练误差的期望有闭式：

$$
E[R_{tr}]=m_2\left[1-\frac{2p}{N}+\frac{\mathrm{tr}(P^\top P)-\sum_{i=1}^Nh_{ii}^2}{N}\right]
$$

其中用了 $\sum_ih_{ii}=\mathrm{tr}(P)=p$、$\frac1N\sum_iH_i=\mathrm{tr}(P^\top P)/N$。两个不等式把它压下去：$P$ 是对称投影矩阵，特征值 $\lambda_i\in[0,1]$，故 $\mathrm{tr}(P^\top P)=\sum_i\lambda_i^2\le\sum_i\lambda_i=p$；Cauchy–Schwarz 给 $\sum_ih_{ii}^2\ge\frac1N(\sum_ih_{ii})^2=\frac{p^2}{N}$。取最松的界即得

$$
E[R_{tr}]\;\approx\;\sigma^2\Big(1-\frac{p}{N}\Big)\;\le\;\sigma^2
$$

（$m_2=\sigma^2$ 对应 $Y$ 已中心化；否则多出 $E[Y]^2$ 的常数项）。而对**新**测试点，$\hat y(x_0)=x_0^\top\beta+\tilde\varepsilon$ 且 $\hat\beta-\beta$ 带来额外方差，

$$
E[R_{te}]=\sigma^2+\mathrm{tr}\Big[\big(X^\top X\big)^{-1}\mathrm{Cov}(x_0)\Big]\;\approx\;\sigma^2+p\;\ge\;\sigma^2
$$

所以

$$
E\big[R_{tr}(\hat\beta)\big]\;\le\;E\big[R_{te}(\hat\beta)\big]
$$

（原书练习 2.9，正是这条不等式）。数值校验（$N=40$，$p+1=5$ 个参数，$\sigma^2=1.69$）：$E[R_{tr}]=1.48\approx\sigma^2(1-p/N)=1.52$，$E[R_{te}]=1.81$。

> **坑**
> 上面这条不等式依赖三个条件：$X^\top X$ 非奇异（$N>p$）、$Y$ 已中心化、测试点与训练点同分布。若 $p\gg N$，左边的 $\sigma^2(1-p/N)$ 变成负数（公式失效），而 $R_{tr}$ 本身仍非负；若 $Y$ 的均值很大，$m_2$ 里的 $E[Y]^2$ 项会让左边超过 $\sigma^2$。原书把证明留作练习，正是因为这些边界情形需要额外假设。

### 2.7.3 交叉验证：训练误差为什么有偏 {#s-2-7-3}

训练误差是**有偏**的（系统性偏低），交叉验证是补救手段，但**也不是无偏的**。把这件事算清楚，需要原书练习 2.7 的通用框架。

**通用框架 · 线性估计量的偏差–方差分解。** 设估计量形如

$$
\hat f(x_0)=\sum_{i=1}^{N}\ell_i(x_0;\mathcal{X})\,y_i
$$

权重 $\ell_i$ 依赖整条设计序列 $\mathcal{X}=(x_1,\dots,x_N)$ 但**不依赖** $y_i$。最小二乘与 $k$-最近邻都属于这一类：

- 最小二乘：$\hat f(x_0)=x_0^\top(X^\top X)^{-1}X^\top y=\sum_i\underbrace{\big[x_0^\top(X^\top X)^{-1}x_i\big]}_{\ell_i(x_0;\mathcal X)}y_i$（就是 §2.4.3 用过的权重）；
- $k$-最近邻：$\ell_i=\frac1k\mathbf 1\{x_i\in N_k(x_0)\}$。

**(a) 条件均值平方误差。** 记 $f(x)=E[Y\mid X=x]$、$y_i=f(x_i)+\varepsilon_i$。对给定的设计序列 $\mathcal{X}$：

$$
E\big[(f(x_0)-\hat f(x_0))^2\mid\mathcal X\big]=\underbrace{\Big[\sum_i\ell_i f(x_i)-f(x_0)\Big]^2}_{\mathrm{Bias}^2(\hat f\mid\mathcal X)}+\underbrace{\sum_{i=1}^N\ell_i^2\sigma^2}_{\mathrm{Var}(\hat f\mid\mathcal X)}
$$

推导：$f(x_0)-\hat f = f(x_0)-\sum_i\ell_if(x_i)-\sum_i\ell_i\varepsilon_i$，平方取条件期望，第一项是确定量的平方，第二项用 $E[\varepsilon_i\varepsilon_j\mid\mathcal X]=0\ (i\ne j)$、$E[\varepsilon_i^2\mid\mathcal X]=\sigma^2$。

**(b) 无条件均值平方误差。** 对 $\mathcal X$ 再取期望，交叉项不再为零（权重本身随机）：

$$
E\big[(f-\hat f)^2\big]=\Big[E\sum_i\ell_if(x_i)-f(x_0)\Big]^2+\sum_iE[\ell_i^2]\sigma^2+\sum_{i\ne j}E[\ell_i\ell_j]\mathrm{Cov}(f(x_i),f(x_j))
$$

注意二阶项多出 $\mathrm{Cov}(f(x_i),f(x_j))$ 的贡献——这是 $\ell_i$ 随机性带来的，属于**抽样方差**的一部分。

**(c) 两者的关系（练习 2.7(d)）。** 由全方差公式

$$
\mathrm{Var}(\hat f)=E\big[\mathrm{Var}(\hat f\mid\mathcal X)\big]+E\big[\mathrm{Var}\big(E[\hat f\mid\mathcal X]\big)\big]
$$

第二项 $\ge0$，第一项 $=\sum_iE[\ell_i^2]\sigma^2$，故

$$
\mathrm{Var}(\hat f)\;\ge\;\sigma^2\,E\!\left[\sum_i\ell_i^2\right]=E\big[\mathrm{Var}(\hat f\mid\mathcal X)\big]
$$

**无条件方差不小于条件方差**——差值正是「训练点被自己的标签影响」的那部分。对平方偏差则相反方向成立的不等式一般不成立（$E[\mathrm{Bias}(\cdot|\mathcal X)]^2$ 与 $\mathrm{Bias}^2$ 之间只有 Jensen 给出的下界 $\ge(\mathbb{E}\mathrm{Bias})^2$）。这个不等式是理解交叉验证偏差的关键。

**LOOCV · 精确公式。** 留一交叉验证（LOOCV）对每个 $i$ 用 $N-1$ 个点拟合 $\hat f^{(-i)}$，报 $\frac1N\sum_i\big(y_i-\hat f^{(-i)}(x_i)\big)^2$。朴素实现要拟合 $N$ 次，第 7 章证明可以只拟合一次：记 $e_i=y_i-x_i^\top\hat\beta$、$h_{ii}=x_i^\top(X^\top X)^{-1}x_i$，则

$$
\hat f^{(-i)}(x_i)=x_i^\top\hat\beta^{(-i)}=\hat f(x_i)-\frac{h_{ii}}{1-h_{ii}}e_i
$$

$$
e_i^{(-i)}:=y_i-\hat f^{(-i)}(x_i)=e_i-\frac{h_{ii}}{1-h_{ii}}e_i=\frac{e_i}{1-h_{ii}}
$$

**推导（Sherman–Morrison，预备知识 N1 的矩阵逆更新）.** 记 $R=X^\top X$、$\gamma=R^{-1}x_i$、$h=x_i^\top\gamma=h_{ii}$。删除第 $i$ 行后的法方程矩阵是 $R-x_ix_i^\top$，由 Sherman–Morrison 公式

$$
(R-x_ix_i^\top)^{-1}=R^{-1}+\frac{R^{-1}x_ix_i^\top R^{-1}}{1-x_i^\top R^{-1}x_i}=R^{-1}+\frac{\gamma\gamma^\top}{1-h}
$$

代入 $\hat\beta^{(-i)}=(R-x_ix_i^\top)^{-1}(X^\top y-x_iy_i)$：

$$
\hat\beta^{(-i)}=\Big(R^{-1}+\frac{\gamma\gamma^\top}{1-h}\Big)\Big(X^\top y-x_iy_i\Big)
$$

四項逐一化简：$R^{-1}X^\top y=\hat\beta$；$R^{-1}x_iy_i=\gamma y_i$；$\gamma\gamma^\top X^\top y=\gamma\,\gamma^\top y=\gamma\,x_i^\top R^{-1}y=\gamma\,x_i^\top\hat\beta$；$\gamma\gamma^\top x_iy_i=\gamma\,h\,y_i$。合并后

$$
\hat\beta^{(-i)}=\hat\beta+\frac{\gamma}{1-h}\big[x_i^\top\hat\beta-y_i(1-h)-hy_i\big]=\hat\beta-\frac{\gamma\,e_i}{1-h}
$$

两端左乘 $x_i^\top$（用 $x_i^\top\gamma=h$）：

$$
x_i^\top\hat\beta^{(-i)}=x_i^\top\hat\beta-\frac{h\,e_i}{1-h}
$$

于是 $e_i^{(-i)}=y_i-x_i^\top\hat\beta^{(-i)}=e_i+\frac{h e_i}{1-h}=\frac{e_i}{1-h}$。$\blacksquare$（数值验证：$N=40,p=4$ 的模拟中，$|e_i^{(-i)}-e_i/(1-h_{ii})|$ 的最大值为 $2\times10^{-15}$。）

**为什么 CV 误差是正偏的。** 三个可算的理由：

1. **(c) 的不等式**：$\mathrm{Var}(\hat f)\ge E[\mathrm{Var}(\hat f\mid\mathcal X)]$，即在同一批训练点上评估会低估方差。CV 的评估点仍然是训练点（只是被排除在拟合之外），所以这个不等式仍然部分起作用。
2. **LOOCV 仍用了 $N-1$ 个点**：它报的是「用 $N-1$ 个点拟合的模型在一个新点上的 EPE」的估计，而不是 $\sigma^2$。§2.7.2 的 $\mathrm{Bias}^2$ 项 $\big[E\sum_i\ell_i^{(-i)}f(x_i)-f(x_i)\big]^2$ 一般不为 0。
3. **每个点还带着杠杆影响**：LOOCV 的权重 $\ell_i^{(-i)}=x_i^\top(X_{-i}^\top X_{-i})^{-1}x_i$，与 $h_{ii}$ 有关。代入 $X_{-i}^\top X_{-i}=R-x_ix_i^\top$ 与上面的 Sherman–Morrison 展开可得 $\ell_i^{(-i)}=\frac{h}{1-h}$，故 $e_i^{(-i)}$ 被放大了 $1/(1-h_{ii})\ge1$ 倍——**只有完全可忽略的点（$h_{ii}\approx0$）才满足无偏**。

数值验证（$N=40$，$p+1=5$，$\sigma^2=1.69$）：$E[R_{tr}]=1.48$，$E[\text{LOOCV}]=1.94$，而 $\sigma^2=1.69$。**LOOCV 也偏高**，顺序是 $R_{tr}<\sigma^2<\text{LOOCV}$。

**K 折 CV 的方差。** 把样本随机分成 $K$ 折，每折 $N/K$ 个点。$\bar e_{\rm CV}=\frac1N\sum_i e_i^{(K(\cdot))}$ 的方差可按两步分解：$K$ 个折内均值的平均，再对 $K$ 个折均值取平均。用方差可加性（预备知识 P1）与「同一折内的残差强相关」可得

$$
\mathrm{Var}\big(\bar e_{\rm CV}\big)\approx\frac{\mathrm{Var}\big(e_i^{(-i)}\big)}{N}\Big[1+\frac{N-N/K}{K-1}\Big]
$$

（方括号是有限总体修正：同一折内的残差共享 $K-1$ 个「同折邻居」，其正协方差把方差放大）。定性结论有三条：

- LOOCV（$K=N$）**方差最小**，因为每个模型都用上了 $N-1$ 个点；
- $K$ 折（$K=5$ 或 10）偏差与方差居中，是实践默认；
- 留出固定比例的验证集（$K=1$）**方差最大**，训练集只有 $(1-\varepsilon)N$ 个点。

**类别不平衡时的附加问题。** 分类问题里若某类只占 1%，错分率对少数类完全不敏感：把全部点都判成多数类，错分率只有 1%，但少数类一个都没识别出来。这要求用**分层抽样**保证每折的类别比例与全体一致，否则少数类的 CV 估计方差极大、且不同折之间不可比。第 4 章会给出类别不平衡下 LDA 与逻辑回归的行为差异。

**选择偏差。** 若 CV 误差被用来**从一族模型里挑最好的**，那么被选中的那个 CV 值是 $\min$ 而不是随机一个模型的期望，会额外偏小；且不同模型之间若有效自由度不同，偏差的**量**也不同，于是比较不公平。修正办法是对每个模型都用同样的折划分、并把偏差项 $\big(1-p_k/N\big)$ 作为**模型复杂度的递增函数**显式加回（§2.7.2 的公式），或改用嵌套 CV：外层折估计「整个选模流程」的误差。

> **结果**
> 交叉验证不是无偏估计。它的正偏差随「评估点被训练集包含的程度」增长，而**方差随训练集变小而增长**。这两条同时作用，使得「训练集尽量大 + 每折尽量均衡 + 分层抽样 + 同一折划分」成为标准做法。第 7 章会给出 LOOCV 的精确公式 (7.11) 与 bootstrap 的正偏差修正，本节的公式 (2.24)(2.25)(2.46)(2.47) 是它们的零件。

---

## 2.8 回归与分类的差异 {#s-2-8}

本章两类任务的「判据层」完全相同：都写成 $E\big[L(\text{输出},\text{预测})\big]$，都用塔性质化成逐点条件风险。对定量输出，最优是条件均值 (2.13)；对定性输出，最优是 Bayes 规则 (2.23)。差异全部来自**损失函数**：

| | 回归 | 分类 |
|---|---|---|
| 损失 | $(Y-f(X))^2$，光滑、凸 | $0/1$，不光滑、有平台 |
| 最优解 | 条件均值 (2.13) | 后验最大类 (2.23) |
| 不可约误差 | $\mathrm{Var}(Y\mid X=x)$ | $\sum_k\min_k\pi_k(x)=1-\max_k\pi_k(x)$ |
| 编码后回归 | — | $E(Y_k\mid X)=\Pr(G=\mathcal{G}_k\mid X)$ |

不可约误差这一行值得展开。对分类，把 $G$ 编码成 0/1 的 $Y$，则

$$
\mathrm{Var}(Y\mid X=x)=p(x)\big[1-p(x)\big]
$$

它由**点态分布**决定，而不是由分布的「形状」决定；最小不可约错分率是

$$
\text{Bayes rate}=1-\max_{k}\pi_k(x),\qquad \pi_k(x)=\Pr(G=\mathcal{G}_k\mid X=x)
$$

积分后得总体 Bayes 率。$p(x)$ 越接近 0 或 1，不可约误差越小——这解释了为什么图 2.4 的 Bayes 错误率由两类密度重叠区域的面积决定，也解释了为什么「把 $\hat f(x)$ 截断到 $[0,1]$ 再分类」通常不会改变分类结果（只有当 $\hat f$ 落在 $[0,1]$ 内且是最大分量时才可能改变 argmax）。

**分布形态对预测准确率的影响**，用 (2.28) 与 (2.47) 定量对比：

- **线性真函数 $f(x)=x_1$**：最小二乘无偏，EPE $=\sigma^2(p/N)+\sigma^2\approx\sigma^2$；1-最近邻的方差至少 $\sigma^2$（$k=1$ 时 (2.47) 的 $\sigma^2/k=\sigma^2$）且偏差随维数增大，所以 $EPE$ 恒大于 $2\sigma^2$，比值随 $p$ 上升（图 2.9 橙线从 2 附近起步）。
- **三次真函数 $f(x)=\frac12(x_1+1)^3$**（只依赖一个坐标）：最小二乘**有偏**，偏差压低了比值，图 2.8 里变成方差主导。
- **确定性且只依赖少数维度**：偏差几乎为 0，$k$-最近邻直接取胜。

结论不是「谁更好」，而是**真函数依赖多少个维度**决定谁赢。这正是所有正则化方法存在的理由。

**一条可复核的判断准则。** 对同一个模型族，比较两类估计量的 EPE：

$$
\mathrm{EPE}(f)-\mathrm{EPE}(\hat f)\;\ge\;\sigma^2\left(1-\frac{p}{N}\right)
$$

即「线性 + 最小二乘」的期望误差不超过「$\sigma^2(1-p/N)$」，而「局部常数 + $k$-最近邻」不低于 $\sigma^2/k$。两者的交叉条件 $\sigma^2/k\approx\sigma^2$（即 $k\approx1$）说明：**只有当 $f$ 的结构与线性假设吻合时才选最小二乘**，否则要用正则化把有效自由度降到 $N/k$ 的水平上——这正是第 5 章 ridge 与 lasso 的作用。

---

## 2.9 编号速查与常见误区 {#s-2-9}

本章 47 个编号公式的用途速查：

| 编号 | 内容 | 在本章中的作用 | 后面哪里再用 |
|---|---|---|---|
| (2.1)(2.2) | 线性模型 | 假设 $f$ 全局线性 | 3.1，4.2 |
| (2.3)–(2.6) | 残差平方和 → 正规方程 → 解 | 训练阶段 | 3.2，5.5，7.x LOOCV |
| (2.7) | 阈值 0.5 分类 | 回归当分类 | 4.2 |
| (2.8) | $k$-最近邻平均 | 局部常数拟合 | 13.x |
| (2.9)–(2.13) | EPE → 条件均值 | **判据基础** | 全书 |
| (2.14) | 邻域平均 | 条件期望的估计 | 13.x |
| (2.15)(2.16) | 线性假设 → 总体矩公式 | 最小二然的理论依据 | 3.2 |
| (2.17) | 可加模型 | 降维工具 | 9.x MARS |
| (2.18) | 条件中位数 | $L_1$ 损失 | 6.x |
| (2.19)–(2.23) | 0-1 损失 → Bayes 规则 | **分类判据基础** | 4.x，5.x，13.x |
| (2.24) | 最近邻中位距离 $d(p,N)$ | 维数灾难 | 13.2 |
| (2.25) | $\mathrm{Var}+\mathrm{Bias}^2$ | **偏差–方差恒等式** | 7.4，8.x，15.x |
| (2.26)–(2.28) | 最小二乘 EPE $=\sigma^2(p/N)+\sigma^2$ | 避开维数灾难 | 3.5 |
| (2.29)–(2.36) | 加性误差、基展开、似然 | 估计准则 | 3.x，4.4 |
| (2.37)–(2.45) | 三类受限估计量 | 模型族目录 | 5.x，6.x，11.x |
| (2.46)(2.47) | EPE 三项分解 | **折中的定量式** | 7.x，15.x |

**三个最容易搞错的点**：

1. **(2.12) 与 (2.13) 不是两条公式，是同一次求解。** (2.12) 的 $\mathrm{argmin}$ 的解就是 (2.13)。原文把求解结果单独编号，读者容易误以为还需要额外一步。同理 (2.25) 与 (2.46) 也不是两条「新的」分解，而是同一条恒等式在确定性目标与随机目标下的两个版本。
2. **(2.16) 与 (2.6) 只差「期望 vs 样本平均」。** 用大数定律（预备知识 P1）把 (2.16) 的矩换成样本矩就得到 (2.6)。中间不需要任何额外假设（除了 $E[XX^\top]$ 可逆）。而 (2.37) 与 (2.9)、(2.11) 是同一件事的有限样本版与总体版。
3. **(2.28) 的 $\sigma^2(p/N)+\sigma^2$ 不能读成「$p$ 越大越坏」。** 它说的是：只要 $p\ll N$ 且假设正确，最小二乘几乎不受维数影响；危险的不是 $p$ 大而是**没有假设**（$k$-最近邻那侧，见 (2.47) 的 $\sigma^2/k$ 与偏差项）。原书原话是「靠强假设避开维数灾难」。

**四个反复出现的方差公式**（后面每章都要用，符号统一）：

$$
\mathrm{Var}\Big(\frac1n\sum_{i=1}^n Z_i\Big)=\frac{\mathrm{Var}(Z)}{n},\qquad \mathrm{Var}(a^\top Z)=a^\top\mathrm{Cov}(Z)a,\qquad \mathrm{Var}(\bar Z)=\frac{\sigma^2}{N}
$$

$$
\mathrm{Var}(Y\mid X=x)=\begin{cases}\sigma^2(x),&\text{加性误差模型 (2.29)}\\ p(x)\big[1-p(x)\big],&\text{0/1 编码的二分类}\end{cases}
$$

第一条是方差可加性（预备知识 P1），第二条是二次型形式的方差（预备知识 L5），第三条是 (2.47) 的 $\sigma^2/k$ 的来源，第四条是分类问题不可约误差的来源。

**参考文献标记**：原书在本章末尾把整串参考文献整体编号为 (2007)——这是一个排印遗留的编号（正文里的公式编号在 (2.47) 就结束了，(2007) 来自某次跨章节的旧编号）：

$$
\text{Bibliographic Notes: Duda et al. (2000), Bishop (1995, 2006), Ripley (1996), Cherkassky and Mulier (2007), Vapnik (1996)} \eqno{2007}
$$

（这里没有实际推导，只是为了对齐编号；本章部分内容改编自 Friedman (1994b)。）

<a class="src" href="index.html">返回封面</a>
