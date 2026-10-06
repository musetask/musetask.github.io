## 14.7.3 探索性投影寻优 {#s-14-7-3}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-7-3">原文 §14.7.3</a>

投影寻优（projection pursuit, PP）不是一个新的模型，而是一个**找方向**的手段：给定 $\mathbb{R}^p$ 里的数据，投影寻优问的是「往哪个方向投影才能看清结构」。Friedman 与 Tukey（1974）提出它时的出发点是一句经验观察：**高维数据的低维投影看上去几乎都是高斯的**，所以想看见簇、长尾、离群点这些结构，就得去找那些「不像高斯」的方向。ESL 把这个观察的形式化交给 §14.7.2 的负熵 $J$，于是投影指标与 ICA 的对比函数变成同一件事。

### 14.7.3.1 为什么高维数据的投影都像高斯 {#s-14-7-3-1}

> **基础知识** · 中心极限定理（见预备知识 P1）
>
> 若 $Z_k$ 独立同分布、$E[Z_k]=0$、$\mathrm{Var}(Z_k)=\sigma^2<\infty$，则
> $$\frac{1}{\sigma\sqrt M}\sum_{k=1}^M Z_k\ \xrightarrow{d}\ N(0,1)$$
>
> 关键是：这里只用到「加了很多项」，不要求 $Z_k$ 本身高斯。

> **推导** · 把数据写成 $x=\sum_{k=1}^M a_k\xi_k$，$\xi_k$ 独立同分布、$E[\xi_k]=0$、$\mathrm{Var}(\xi_k)=\sigma^2$。对任意方向 $a\in\mathbb{R}^M$，
>
> $$\frac{a^\top x}{\mathrm{sd}(a^\top x)}=\frac{\sum_{k=1}^Ma_k\xi_k}{\sigma\sqrt{\sum_k a_k^2}}\ \xrightarrow{d}\ N(0,1)$$
>
> 因为 $\mathrm{Var}(a^\top x)=\sigma^2\sum_ka_k^2$（二次型形式的方差，见预备知识 L5），所以标准化后的系数恰好是 $a_k/\sqrt{\sum_ka_k^2}$，$\sum_k$ 它仍是 $M$ 项独立同分布的加权和，CLT 直接适用。
>
> **结论**：只要 $M$（独立成分个数）相对大，**任何**一维投影的边缘分布都与标准正态难以区分。于是
> - 「投影看上去像高斯」不是数据没有结构，而是**信息被高维平均掉了**；
> - 想看到结构，只能沿 $\sum_ka_k^4$ 之类**不随方向平均掉**的方向去看，而 $p$ 维球面上有无穷多个方向可以试。
>
> **数值** · 若 $\xi_k$ 都是 Bernoulli$(-1,1)$（两点分布，$\sigma^2=1$、四阶累积量 $\kappa_4=-2$），取 $a$ 有 $M$ 个等值分量，则 $\mathrm{Var}(a^\top w)=\sum_ka_k^2$，而四阶矩精确等于 $3+\sum_ka_k^4\kappa_4=3-2/M$，即峰度随 $M$ 增大而单调趋向高斯的 $3$。这就是「投影越随机越像高斯」的一个手算版本。

### 14.7.3.2 投影指标就是负熵 {#s-14-7-3-2}

> **基础知识** · 微分熵与 KL 散度（见预备知识 P3）
>
> 连续随机变量 $Y$ 的微分熵 $H(Y)=-\int f_Y\log f_Y\,\mathrm{d}y$；若 $H(Z)$ 有限，则相对熵非负给出
> $$J(Y):=H(Z)-H(Y)=\int f_Y\log\frac{f_Y}{f_Z}\ \ge\ 0$$
> 其中 $Z$ 与 $Y$ 同均值同方差。推导：把 $\log\frac{f_Y}{f_Z}$ 乘上 $f_Z$ 积分（$=0$，因为归一化），再用 Jensen 不等式 $\int f_Y\log(f_Y/f_Z)\le\log\int f_Y=0$。
> 单变量正态的熵精确等于 $H(Z)=\tfrac12\log(2\pi e\,\sigma^2)$：代入 $f_Z=\frac{1}{\sqrt{2\pi}\sigma}e^{-z^2/(2\sigma^2)}$，得 $H(Z)=\frac12\log(2\pi\sigma^2)+\frac12$。

> **结果** · 投影指标与 (14.86)(14.87) 的等价性
>
> 固定 $\|a\|=1$，记 $Y=a^\top X$，$\sigma^2=\mathrm{Var}(Y)$，$Z\sim N(0,\sigma^2)$。ESL §14.7.3 的原话是：投影指标「恰好是 (14.86) 里 $J(Y_j)$ 的同一形式」，其中 $Y_j=a_j^\top X$。把这句话展开成三行：
>
> 1. 由上面的 Jensen 推导，$J(Y)\ge0$，且 $J(Y)=0\iff f_Y=f_Z\iff Y$ 是高斯。
> 2. $H(Z)=\tfrac12\log(2\pi e\sigma^2)$ 只依赖 $\sigma^2$，所以 $\max_a J(a^\top X)$ 与 $\min_a H(a^\top X)$ 在方差固定时**完全等价**。
> 3. 对比函数近似 (14.87) $J(Y)\approx\bigl[E\,G(Y)-E\,G(Z)\bigr]^2$ 把「算熵」换成「算两个矩之差」。取 $G$ 为偶函数（通常 $G(y)=\exp(-y^2/2)$），只依赖 $|Y|$ 的矩，于是
> $$\max_{\lVert a\rVert=1}\ \Big[E\,G(a^\top X)-E\,G(Z)\Big]^2$$
> 就是投影寻优的**指数指标**。历史上还有对数余弦指标 $G(y)=\log\cosh y$（Huber, 1985）与直接用 $H$ 的**熵指标**（GGobi 的默认值），三者的区别只在 $G$ 的选取。
>
> 推导见上一节的 (14.86)(14.87)（[§14.7.2.2](#s-14-7-2-2)）。

> **推导** · 为什么要用「方差比」型指标（Friedman–Tukey 的原始写法）
>
> 投影寻优最原始的指标不是负熵，而是
> $$Q(a)=\frac{\mathrm{Var}(a^\top X)}{\mathrm{Inf}_a(a^\top X)},\qquad \mathrm{Inf}_a(a^\top X):=\inf_{\psi}\ \mathrm{Var}\big[\,\psi(Y)-Y\,\big]$$
> 其中 $\psi$ 取低阶（通常三次）多项式，$\mathrm{Inf}$ 是残差方差的最小值。把最优的 $\psi^\star$ 记下来，由勾股分解（见预备知识 L2 的 Pythagoras 恒等式）：
> $$\mathrm{Var}(Y)=\mathrm{Var}\big(\psi^\star(Y)\big)+\mathrm{Var}\big(Y-\psi^\star(Y)\big)=\mathrm{Var}\big(\psi^\star(Y)\big)+\mathrm{Inf}_a(a^\top X)$$
> 两边除以 $\mathrm{Inf}_a$，并记 $R^2=\mathrm{Var}(\psi^\star)/\mathrm{Var}(Y)$（这就是把 $Y$ 对三次多项式回归的判定系数），得
> $$Q(a)=\frac{1}{1-R^2(a)}$$
> 于是 $\max_a Q(a)$ **等价于** $\max_a R^2(a)$：找一个方向，使这个投影能被光滑的三次曲线**最好地解释**。$R^2$ 越大说明边缘分布越不「光滑对称」，也就是越非高斯。

> **结果** · 与 PCA 的三点对比
>
> | | 目标 | 用到的矩 | 对尺度的敏感性 |
> |---|---|---|---|
> | PCA | $\max_{\lVert a\rVert=1}\mathrm{Var}(a^\top X)$ | 只用二阶 | 不敏感（$\mathrm{Var}$ 已除掉尺度） |
> | 负熵 / EPP | $\max_{\lVert a\rVert=1}J(a^\top X)$ | 高阶（对比函数取到几阶就用几阶） | 不敏感（$J$ 对缩放不变） |
> | $Q(a)=\mathrm{Var}/\mathrm{Inf}_a$ | 同上，但换成残差解释力 | 高阶 | 不敏感（分子分母都二次齐次） |
>
> PCA 的解是 $E[XX^\top]$ 的最大特征向量（见 [§14.5.1](#s-14-5-1)），它只知道二阶矩：若 $X$ 各分量独立且都对称，PCA 找的「最大方差方向」与「最非高斯方向」毫无关系。反过来 EPP 完全不看二阶矩的绝对大小，只看一维投影的形状。
>
> **高维下的困难**有三层：(1) 目标在 $\mathbb{R}^p$ 的单位球面上非凸，多起点是必需的；(2) 每算一次 $Q(a)$ 要做一次一维三次回归（或等价地估计一维密度），代价高；(3) 方向多、样本少时容易过拟合。
>
> **神经网络估计 $\mathrm{Inf}_a$**：Eberhart、Fui 与 Lin（2000）的做法是用一个前馈网络 $\hat f_a(y)\approx \mathrm{dens}(a^\top X)$ 代替三次多项式，于是「残差方差最小」变成「最小化 $\int(y-\hat\psi(y))^2\hat f_a(y)\mathrm{d}y$」。注意这恰好是 14.7.4 的做法——用样条/网络估计密度再做对比。**投影寻优的神经网络实现与 ProDenICA 的 GAM 实现是同一个数学对象**。

> **坑** · 投影寻优**不要求**方向正交（原文明确说 "the directions $a_j$ are not constrained to be orthogonal"）。Friedman（1987）的改进是「先变换数据让它在当前方向上看起来高斯，再找下一个方向」，即**去膨胀 + 逐次去相关**，与 §14.7.4 的 $A\leftarrow UV^\top$ 正交化 + 逐个 $a_j$ 不动点迭代是同一个套路。另一个坑：EPP 的结果依赖起点与指标选取，没有唯一答案，这正是 ICA 也要面对的问题。

---

## 14.7.4 ICA 的直接方法 {#s-14-7-4}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-7-4">原文 §14.7.4</a>

§14.7.2 的快速不动点算法把每个成分当成**一维对比函数**（非高斯性的一个标量摘要）来最大化；§14.7.4 换一条路：不猜对比函数，而是**直接估计一维密度** $f_j$，再极大化似然。实现（R 包 `ProDenICA`）是「广义加性模型 + 修正的 Newton 步」。(14.88)(14.89) 的模型与倾斜高斯表示已在 [§14.7.2.5](#s-14-7-2-5) 交代，这里从 (14.90) 开始。

### 14.7.4.1 倾斜高斯模型的对数似然 (14.90) {#s-14-7-4-1}

> **基础知识** · 变量替换的密度公式
>
> 若 $X=AS$，$S=(s_1,\dots,s_p)^\top$ 独立且 $s_j=a_j^\top X$，则 $A$ 可逆时 $S=A^{-1}X$，密度按链式法则
> $$f_X(x)=\frac{f_S(A^{-1}x)}{|\det A|}$$
> $A$ 正交时 $|\det A|=1$，变换不改变密度。

> **推导** · 把 (14.90) 从模型写出来
>
> 1. 预处理：把 $X$ 白化（中心化 + 归一化到 $E[XX^\top]=I$），于是可以**假设** $A$ 正交。
> 2. 独立成分的联合密度按定义是乘积形式 (14.88)：
> $$f_S(s)=\prod_{j=1}^p f_j(s_j)$$
> 3. 对 $i=1,\dots,N$ 求似然并取对数：
> $$\ell(A,\{g_j\};X)=\sum_{i=1}^N\log f_X(x_i)=\sum_{i=1}^N\sum_{j=1}^p\Big[\log f_j(s_{ij})+1\Big]$$
> 其中 $s_{ij}=a_j^\top x_i$。代入倾斜高斯表示 (14.89) $f_j(s)=\varphi(s)e^{g_j(s)}$，$s_{ij}=a_j^\top x_i$，得
> $$\ell(A,\{g_j\};X)=\sum_{i=1}^N\sum_{j=1}^p\Big[\log\varphi(a_j^\top x_i)+g_j(a_j^\top x_i)\Big] \eqno{14.90}$$
> 4. 约束：$A$ 正交，且每个 $g_j$ 要让 $\varphi e^{g_j}$ 是密度（归一化）。

> **结果** · (14.90) 与 (14.87) 是同一件事的两种写法
>
> 取对比函数 $G(z)=-\log\varphi(z)=\tfrac12z^2+\tfrac12\log(2\pi)$（除去与 $z$ 无关的常数就是 $\tfrac12z^2$），则
> $$\frac1N\sum_{i=1}^N G(a_j^\top x_i)=\frac{1}{2N}\sum_{i=1}^N(a_j^\top x_i)^2+\text{const}=E[a_j^\top XX^\top a_j]\Big/2+\text{const}$$
> 白化下这一项对 $a_j$ **没有区分度**（$a_j^\top a_j=1$，值恒为 $\frac12$）。所以 (14.90) 里「像高斯」的那部分与方向无关，**全部方向信息都在 $g_j$ 上**——这就是下面 (14.91) 要给 $g_j$ 加两个罚项的原因，也是 (14.95) 里 $C(A)$ 只含 $\hat g_j$ 的原因。

> **坑** · 不加任何限制时 (14.90) 是**过参数化**的：$g_j$ 可以任意复杂（甚至取到使密度不存在的值），极大似然无界、原书说「没有唯一解」。必须正则化。

### 14.7.4.2 两个罚项与正则化似然 (14.91) {#s-14-7-4-2}

> **推导** · (14.91) 逐项来历
>
> 目标是极大化一个「加了对数似然减去两个罚项」的准则：
> $$\sum_{j=1}^p\Bigg[\frac1N\sum_{i=1}^N\Big[\log\varphi(a_j^\top x_i)+g_j(a_j^\top x_i)\Big]-\int\varphi(t)e^{g_j(t)}\,\mathrm{d}t\Big]-\sum_{j=1}^p\lambda_j\int\bigl[g_j''(t)\bigr]^2\,\mathrm{d}t \eqno{14.91}$$
> 逐项解释（这是原书的两条 bullet）：
>
> 1. **归一化罚** $\int\varphi(t)e^{g_j(t)}\,\mathrm{d}t$。若不加它，$g_j\equiv c$（常数）能让似然项 $\frac1N\sum_ig_j(a_j^\top x_i)=c$ 任意大。取 $g_j$ 的驻点条件（对 $g_j$ 求导、令 $\frac1N\sum_i\varphi(s_{ij})=0$ 不可能，故最优解自动满足）$\partial/\partial g_j$：$\frac1N\sum_i\varphi(s_{ij})=\int\varphi(t)e^{g_j(t)}\mathrm{d}t$，右端是密度估计的积分，**必须等于 1**。所以这一项是「把 $f_j$ 变成密度的约束」。
> 2. **粗糙度罚** $\lambda_j\int[g_j''(t)]^2\mathrm{d}t$。它保证解是以观测值 $s_{ij}=a_j^\top x_i$ 为节点的三次/四次样条。
>
> **为什么二阶导罚给出样条。** 变分问题
> $$\hat g=\arg\min_g\ \Big\{\int\ell(g(t))\,\mathrm{d}t+\lambda\int[g''(t)]^2\mathrm{d}t\Big\}$$
> 对 $g$ 求导得 $\ell'(t)-2\lambda g'''(t)=0$，即 $g'''=\ell'/(2\lambda)$：三次函数加上一个特解。加上边界条件（$g''=g'''=0$ 在端点，$g,g'$ 的插值条件）后解是**三次样条**，共 $2M-2$ 个自由度（$M$ 是节点数）。所以「二阶导罚 ↔ 样条」不是修辞，是变分法的直接结论。
>
> **性质**：$g_j$ 的极小点还满足均值 $0$、方差 $1$（Exercise 14.18），所以 $\hat f_j$ 自动是「标准化」的；$\lambda_j\to\infty$ 时 $\hat g_j\to$ 常数、$\hat f_j\to\varphi$。

### 14.7.4.3 单分量问题 (14.92) 与直方图近似 (14.93)(14.94) {#s-14-7-4-3}

> **推导** · 把 $p$ 个问题拆成一个
>
> (14.91) 对每个 $j$ 是**分离**的（$a_j$ 与 $g_j$ 配对出现），所以可以只解一个：
> $$\sum_{i=1}^N\Big[\log\varphi(s_i)+g(s_i)\Big]-\int\varphi(t)e^{g(t)}\,\mathrm{d}t-\lambda\int\bigl[g''(t)\bigr]^2\,\mathrm{d}t \eqno{14.92}$$
> 其中 $s_i=a_j^\top x_i$。困难在**第一个积分**：$t$ 上的积分没有任何闭式。用网格近似：取 $L$ 个网格点 $s^\star_\ell=s_{\min}+\ell\Delta$（原书取 $L=1000$），并把每个观测 $s_i$ 归到最近的箱里，箱内比例记为
> $$y_\ell^{\star}=\frac{\#\{s_i\in(s^\star_\ell-\Delta/2,\ s^\star_\ell+\Delta/2)\}}{N} \eqno{14.93}$$
> 注意 $\sum_{\ell=1}^Ly_\ell^{\star}=1$：每个 $s_i$ 恰好落进一个箱。于是
> $$\int\varphi(t)e^{g(t)}\,\mathrm{d}t\approx\sum_{\ell=1}^L\Delta\,\varphi(s^\star_\ell)e^{g(s^\star_\ell)}\ \qquad\text{（矩形法则）}$$
> $$\sum_{i=1}^N\Big[\log\varphi(s_i)+g(s_i)\Big]\approx\sum_{\ell=1}^Ly_\ell^{\star}\Big[\log\varphi(s^\star_\ell)+g(s^\star_\ell)\Big]$$
> 代入即得可用于 Newton 迭代的准则
> $$L=\sum_{\ell=1}^L\Bigg\{y_\ell^{\star}\Big[\log\Big(\varphi(s^\star_\ell)\Big)+g(s^\star_\ell)\Big]-\Delta\,\varphi(s^\star_\ell)e^{g(s^\star_\ell)}\Bigg\}-\lambda\int\bigl[g''(s)\bigr]^2\,\mathrm{d}s \eqno{14.94}$$

> **结果** · (14.94) 是**带 offset 的 Poisson GAM**
>
> 把上式除以 $\Delta$ 写成 Poisson 对数似然的标准形 $\sum_\ell[u_\ell\log\mu_\ell-\mu_\ell]$：
> $$u_\ell=\frac{y_\ell^{\star}}{\Delta},\qquad \mu(s)=\varphi(s)e^{g(s)}$$
> 即**响应**是箱内频率、**均值函数**是倾斜高斯 $\varphi e^g$、**offset** 是 $\log\varphi(s)$、**惩罚**是 $\lambda/\Delta$ 乘二阶导平方。原书：「这是一个广义加性样条模型，用 Newton 算法 $O(L)$ 就能拟合」——因为它就是一个 Poisson 回归（见 [第 6 章](m06.html#s-6-1) 的样条平滑与 [第 9 章](m09.html#s-9-2) 的加性模型）。

> **数值** · 网格近似的误差可控：$\int\varphi g$ 与 $\sum_\ell\Delta\varphi(s^\star_\ell)g(s^\star_\ell)$ 的差是复合梯形公式的误差，$\le\frac{L\Delta^2}{24}\sup_{t}|(tg)''|$。取 $L=1000$、$\Delta\sim\text{range}/1000$ 时远小于统计噪声。原书另外指出：虽然变分问题给出的是三次样条，实践中三次 B 样条（cubic）已足够。

### 14.7.4.4 为什么只剩 $C(A)$，以及不动点 (14.95)(14.96) {#s-14-7-4-4}

> **推导** · (14.91) 中 $\varphi$ 那一坨与 $A$ 无关（Exercise 14.19）
>
> 设 $a_j$ 是正交矩阵 $A$ 的第 $j$ 列。则
> $$\sum_{i=1}^N\log\varphi(a_j^\top x_i)=-\frac12\sum_{i=1}^N(a_j^\top x_i)^2-\frac{N}{2}\log(2\pi)=-\frac12\big\lVert A^\top X\big\rVert_F^2-\frac{Np}{2}\log(2\pi)$$
> 推导第一步只是把 $\log\varphi(z)=-\tfrac12z^2-\tfrac12\log(2\pi)$ 代进去；第二步用 Frobenius 范数的正交不变性：
> $$\big\lVert A^\top X\big\rVert_F^2=\mathrm{tr}\big(X^\top AA^\top X\big)=\mathrm{tr}\big(X^\top X\big)=\big\lVert X\big\rVert_F^2\qquad(AA^\top=I)$$
> 所以不管 $A$ 怎么在正交群上转，$\sum_i\log\varphi(a_j^\top x_i)$ 都不变。于是 (14.91) 中唯一依赖 $A$ 的是 $g_j$ 那一项：
> $$C(A)=\frac1N\sum_{j=1}^p\sum_{i=1}^N\hat g_j(a_j^\top x_i)=\sum_{j=1}^p C_j(a_j) \eqno{14.95}$$

> **结果** · $C(A)$ 就是负熵（14.86）的经验估计
>
> 对固定的 $a_j$，$C_j(a_j)=\frac1N\sum_i\hat g_j(a_j^\top x_i)$ 恰好是 (14.87) 里对比函数那一项 $E\,G(Y_j)$ 的样本均值（取 $G=\hat g_j$，且用拟合密度而非标准正态做参考）。原文：「$C(A)$ 是拟合密度与高斯之间的对数似然比，可以看成负熵 (14.86) 的估计，每个 $\hat g_j$ 是一个 (14.87) 式的对比函数。」
>
> **这就把 14.7.3 与 14.7.4 焊死了**：EPP 的投影指标与 ProDenICA 的对数似然是同一个东西。

> **推导** · (14.96) 这个不动点从哪来
>
> 固定 $\hat g_j$，对单个 $a_j$ 求梯度（见预备知识 L4）：
> $$\nabla_{a_j}C_j=\frac1N\sum_{i=1}^N\hat g_j'(a_j^\top x_i)\,x_i$$
> 加上单位模约束 $\lVert a_j\rVert=1$，用 Lagrange 乘子 $\lambda_j$（见预备知识 O1），驻点条件为
> $$\frac1N X^\top \hat g_j'(s_j)=\mu\,a_j,\qquad s_j=\frac{a_j^\top X}{\sqrt N}$$
> 即 $a_j$ 必须是「加权的协方差方向」——**与主成分 (14.51) 同形**（对照 [§14.5.1](#s-14-5-1)）：主成分用 $X$ 的二阶矩，ICA 用 $X\hat g_j'(s_j)$ 的二阶矩。原书给出的实际更新（一次「修正 Newton 步」，Exercise 14.20）把 $\hat g_j'$ 换成了 $\hat g_j$：
> $$a_j\ \longleftarrow\ E\Big\{X\hat g_j'(a_j^\top X)-E\big[\hat g_j''(a_j^\top X)\big]a_j\Big\} \eqno{14.96}$$
> 括号里的第二项就是 Newton 法对约束的投影：因为 $E[X\hat g_j'(a_j^\top X)]$ 与 $a_j$ 未必平行，Newton 步要把它拉回到切空间。若 $X$ 已白化（$E[XX^\top]=I$）且忽略 $X$ 与 $g_j(a_j^\top X)$ 的相关性（Exercise 14.20 的假设），则 $E[X\hat g_j'(a_j^\top X)]\approx E[\hat g_j'(a_j^\top X)]a_j$，第二项消掉，更新简化为
> $$a_j\ \longleftarrow\ \hat g_j'(a_j^\top X)\Big/\big\lVert\cdot\big\rVert$$
> 这正是 FastICA（[§14.7.2.3](#s-14-7-2-3)）的不动点。**所以 ProDenICA 与 FastICA 是同一算法的两个实现**——原书用 Figure 14.42 的模拟（Amari 度量）来支持这一结论。
>
> 逐个 $j$ 更新完后要做正交化：$A\leftarrow(AA^\top)^{-1/2}A$。推导：设 $A=UDV^\top$（SVD，见预备知识 L3），$U,V$ 列正交，则
> $$A(AA^\top)^{-1/2}=UDV^\top\big(VD^2V^\top\big)^{-1/2}A=UDV^\top VD^{-1}V^\top A=U\,V^\top A\ \Longrightarrow\ A\longleftarrow UV^\top$$
> 若原本 $A$ 正交，则 $UV^\top=A$，这一步是恒等；它是用来修正在交替迭代中积累的正交性误差的。

### 14.7.4.5 Amari 度量 (14.97) 与四阶矩的对照 {#s-14-7-4-5}

> **结果** · (14.97) 怎样度量两个解的接近程度
>
> 记 $r_{ij}=(A_0A^{-1})_{ij}$，$A_0$ 是生成数据的混合矩阵（已正交化），$A$ 是估计值。则
> $$d(A_0,A)=\frac1{2p}\sum_{i=1}^p\Big(\frac{\sum_{j=1}^p|r_{ij}|}{\max_j|r_{ij}|}-1\Big)+\frac1{2p}\sum_{j=1}^p\Big(\frac{\sum_{i=1}^p|r_{ij}|}{\max_i|r_{ij}|}-1\Big) \eqno{14.97}$$
> 第一项沿**行**（每个成分 $i$ 的误差被哪个成分 $j$ 分摊）、第二项沿**列**（每个观测方向 $j$ 的误差被哪个成分 $i$ 分摊）。理想时 $A_0A^{-1}$ 是置换矩阵，每行每列只有一个非零且等于 1，故两个比值都是 $1$，$d=0$。因此 $d$ 度量的是**两个可逆变换之间的双线性「质量集中程度」**，取值 $\ge0$，且对 $A$ 的行/列置换不敏感——这正是 ICA 的可识别性只到置换（及符号、尺度）的原因。

> **推导** · 为什么四阶累积量能认出 ICA 方向
>
> 这是 ICA 最经典的判据（Hyvärinen 与 Oja 2000，也是原书「departures from the Gaussian via kurtosis」那句注释所指）。设 $X=AS$ 且已正交化 $A^\top A=I$，$S$ 分量独立、$E[S_k]=0$、$\mathrm{Var}(S_k)=1$。对单位向量 $a$，令 $b=A^\top a$（$\lVert b\rVert^2=\lVert a\rVert^2=1$），$Y=a^\top X=b^\top S$。
>
> **第一步：展开四阶矩。** $E[(b^\top S)^4]=\sum_{k,l,m,n}b_kb_lb_mb_nE[S_kS_lS_mS_n]$。因各分量独立且零均值，期望为零当且仅当每个下标出现**偶数次**（否则取对应变量求导得零）。$4$ 个因子的偶数划分只有两类：
> $$\underbrace{\sum_{k}b_k^4\mu_{4k}}_{\text{全同 }(k,k,k,k)}+\underbrace{6\sum_{k<l}b_k^2b_l^2}_{(k,k,l,l)\text{ 的 }6\text{ 种排列}},\qquad \mu_{4k}=E[S_k^4]$$
> **第二步：用一阶矩条件消掉交叉项。** $\sum_{k<l}b_k^2b_l^2=\frac12\left[\big(\sum_kb_k^2\big)^2-\sum_kb_k^4\right]=\frac12\left[1-\sum_kb_k^4\right]$，故
> $$E[(b^\top S)^4]=3+\sum_{k=1}^p b_k^4\,(\mu_{4k}-3)$$
> **第三步：换成累积量。** 单位方差下 $\kappa_{4k}=E[S_k^4]-3E[S_k^2]^2=\mu_{4k}-3$，于是
> $$\boxed{\ E[(a^\top X)^4]=3+\sum_{k=1}^p b_k^4\,\kappa_{4k},\qquad b=A^\top a,\ \lVert a\rVert=1\ }$$
> **推论（可识别性）。** $d=b_k^4\ge0$、$\sum_kb_k^2=1$，故 $\sum_kb_k^4\le1$，等号当且仅当只有一个 $|b_k|=1$（其余为 0）：
> - 若所有 $\kappa_{4k}=0$（全高斯），右端恒等于 $3$，**ICA 不可识别**——这正是要用高阶统计量的原因；
> - 否则取 $a=Ae_{k^\star}$，$k^\star=\arg\max_k|\kappa_{4k}|$，则 $|E[(a^\top X)^4]-3|=|\kappa_{4k^\star}|$ 达到上界 $\max_k|\kappa_{4k}|$，且**只有这些有符号坐标向量取到**。所以最大化 $\big(E[(a^\top X)^4]-3\big)^2$（这就是 (14.87) 取 $G(y)=y^4$ 的对比函数）能精确恢复 ICA 方向。
>
> **坑** · 常见的公式 $E[(a^\top w)^4]=3\sigma^4\big(1+2\sum_ka_k^4\big)$ 形式上是错的：它对应上面第二步中把 $\sum_{k<l}b_k^2b_l^2$ 换成 $\sum_kb_k^4$ 的误算，正确写法是 $\sum_kb_k^4(\mu_{4k}-3\sigma^4)$（一般方差时 $E[S_k^2]=\sigma_k^2$，需先把 $S$ 白化）。**斜度（三阶累积量）路线同理**，但它有符号，$\sum_k b_k^3\kappa_{3k}$ 可以相消，所以教科书统一用四阶。
>
> **与 §14.7.2 的关系**：(14.95)(14.96) 用**拟合密度**给出负熵的估计，本框用**单条四阶矩**给出闭式判据。二者一致：把 $G(y)=y^4$ 代进 (14.87) 就落到本框，把 $G$ 换成样条拟合的 $\hat g_j$ 就落到 ProDenICA。原书 Figure 14.42 就是用 (14.97) 证明两种实现表现相当。

---

## 14.8 多维标度 {#s-14-8}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-8">原文 §14.8</a>

自组织映射、主曲线/主曲面、MDS 都在做「把 $\mathbb{R}^p$ 的数据搬到低维」，但**输入不同**：SOM 与主曲线需要数据点 $x_i$；MDS 只需要两两之间的**相异度** $d_{ij}$。典型场景是品酒实验：没有「向量」可用，只有一组被试对两种酒的**差异评分**，MDS 照样能画出一张地图。这就是它跟前面所有方法的本质区别。

### 14.8.1 应力函数 (14.98) 与 Sammon 映射 (14.99) {#s-14-8-1}

> **基础知识** · 应力函数（stress）
>
> 记 $\hat d_{ij}(z)=\lVert z_i-z_j\rVert$。最小二乘标度（Kruskal–Shephard scaling，简称 SMACOF）取
> $$S_M(z_1,\dots,z_N)=\sum_{i<i'}\big(d_{ij}-\hat d_{ij}(z)\big)^2 \eqno{14.98}$$
> 原书注：有些作者定义应力为 $\sqrt{S_M}$，因为不影响优化，ESL 保留平方形式以便与其它准则比较。

> **推导** · (14.98) 的梯度下降步
>
> 原书只说「用梯度下降最小化 $S_M$」。求导（预备知识 L4，注意 $\partial\lVert z_i-z_{i'}\rVert/\partial z_i=(z_i-z_{i'})/\lVert z_i-z_{i'}\rVert$）：
> $$\nabla_{z_i}S_M=2\sum_{i'\ne i}\big(\lVert z_i-z_{i'}\rVert-d_{ij'}\big)\frac{z_i-z_{i'}}{\lVert z_i-z_{i'}\rVert}=2\sum_{i'\ne i}(z_i-z_{i'})\Big[1-\frac{d_{ii'}}{\lVert z_i-z_{i'}\rVert}\Big]$$
> （$\sum_{i'\ne i}$ 把每个无序对数了两次，与 $\sum_{i<i'}$ 差一个系数 2。）所以更新是
> $$z_i\ \longleftarrow\ z_i-\gamma\sum_{i'\ne i}(z_i-z_{i'})\Big[1-\frac{d_{ii'}}{\lVert z_i-z_{i'}\rVert}\Big]$$
> 每一项的读法是「把自己往 $z_{i'}$ 拉近 $d_{ii'}$ 倍」或「推开」——这是 SMACOF 的 Weiszfeld 型迭代的来源。

> **结果** · Sammon 映射 (14.99) 是加权最小二乘
>
> $$\boxed{\ S_{Sm}(z_1,\dots,z_N)=\sum_{i\ne i'}\frac{\big(d_{ii'}-\lVert z_i-z_{i'}\rVert\big)^2}{d_{ii'}} \eqno{14.99}}$$
> 推导它与 (14.98) 的关系：把单项展开成
> $$\frac{(d-\lVert z_i-z_{i'}\rVert)^2}{d}=d-2\lVert z_i-z_{i'}\rVert+\frac{\lVert z_i-z_{i'}\rVert^2}{d}$$
> 于是 $\nabla_{z_i}S_{Sm}=\sum_{i'\ne i}\Big[\frac{2(z_i-z_{i'})}{d_{ii'}}-\frac{2(z_i-z_{i'})}{\lVert z_i-z_{i'}\rVert}\Big]$，即**权重 $1/d_{ii'}$ 的加权应力函数**。含义：$d_{ii'}$ 越小权重越大，误差被放大——所以「更看重小距离的保持」。原文：「Here more emphasis is put on preserving smaller pairwise distances.」
>
> 这条线索在 §14.9 会再次出现：局部 MDS (14.106) 的第二项 $-\tau\sum_{(i,i')\notin\mathcal N}\lVert z_i-z_{i'}\rVert$ 与 Sammon 的 $-\lVert z_i-z_{i'}\rVert$ 是同一个「推远」项。

> **坑** · (14.98) 有两个不可识别性：整体平移（$\{z_i+c\}$ 不改变应力）与整体旋转（$z\mapsto Rz$ 不改变欧氏距离）；此外若某个 $\lVert z_i-z_{i'}\rVert=0$，梯度里的 $1/\lVert\cdot\rVert$ 炸掉。实务上一律**加上 $\bar z=0$ 与 $\sum_m z_{im}^2=1$ 的归一化**（对照 [§14.5.3.3](#s-14-5-3-3) 的 Rayleigh 商做法）。另外 $S_M$ 非凸，Sammon 映射要用**SMACOF**（stress majorization）而不是朴素梯度下降——Weiszfeld 型迭代全局单调下降但只能收敛到局部极小。

### 14.8.2 经典标度的双中心化与特征分解 (14.100) {#s-14-8-2}

> **推导** · 从平方距离反推内积：双中心化
>
> 这是 MDS 最重要的一步推导。给定坐标 $z_i\in\mathbb{R}^M$，记
> $$\hat D_{ij}^2=\sum_{m=1}^M(z_{im}-z_{jm})^2=\underbrace{\sum_m z_{im}^2}_{u_i}+\underbrace{\sum_m z_{jm}^2}_{u_j}-2\underbrace{\sum_m z_{im}z_{jm}}_{g_{ij}}$$
> 其中 $u_i=\lVert z_i\rVert^2$，$g_{ij}$ 是 Gram 矩阵的元素。对**平方距离**矩阵 $a_{ij}=\hat D_{ij}^2$ 做行均值 $a_{i\cdot}=\frac1N\sum_j a_{ij}$、列均值 $a_{\cdot j}$、总均值 $a_{\cdot\cdot}$，则
> $$b_{ij}:=-\tfrac12\big(a_{ij}-a_{i\cdot}-a_{\cdot j}+a_{\cdot\cdot}\big)=g_{ij}-\bar g_{i\cdot}-\bar g_{\cdot j}+\bar g$$
> 推导（逐项代入即可，此处给行和验证）：$\sum_m z_{im}^2=u_i$，$\frac1N\sum_j u_j=\bar u$，$\frac1N\sum_jg_{ij}=\bar g_{i\cdot}$，于是
> $$a_{ij}=u_i+u_j-2g_{ij},\quad a_{i\cdot}=u_i+\bar u-2\bar g_{i\cdot},\quad a_{\cdot j}=\bar u+u_j-2\bar g_{\cdot j},\quad a_{\cdot\cdot}=2\bar u-2\bar g$$
> 四项相减：$a_{ij}-a_{i\cdot}-a_{\cdot j}+a_{\cdot\cdot}=-2g_{ij}+2\bar g_{i\cdot}+2\bar g_{\cdot j}-2\bar g$，除以 $-2$ 即得。**行列和都为零**可以直接验证：$\sum_jb_{ij}=-\tfrac12\big[r_i-\tfrac RN-r_i+\tfrac RN\big]=0$。
>
> 写成矩阵就是**双中心化**：$\boldsymbol{B}=G\boldsymbol{A}G$，$G=I-\frac1Nee^\top$，$\boldsymbol{A}=(a_{ij})$。$\boldsymbol{B}$ 是**中心化 Gram 矩阵**。

> **推导** · 特征分解给出坐标
>
> 设 $\boldsymbol{B}=\sum_{m=1}^p\lambda_m e_me_m^\top$（对称谱分解，见预备知识 L1/L3），$\lambda_1\ge\lambda_2\ge\cdots$。取
> $$z_{im}=\sqrt{\lambda_m}\,e_{im},\qquad m=1,\dots,k$$
> 则（用 $\sum_i e_{mi}=0$，因为 $e_m\perp\mathbf1$）
> $$\langle z_i-\bar z,\ z_{i'}-\bar z\rangle=\sum_{m\le k}\lambda_m(e_{mi}-\bar e_m)(e_{mi'}-\bar e_m)=\sum_{m\le k}\lambda_me_{mi}e_{mi'}=\big(\boldsymbol{B}_k\big)_{ii'}$$
> 即前 $k$ 个特征向量**精确**张成的子空间里，所有中心化内积都被还原了；剩下的误差是 $\sum_{m>k}\lambda_me_{mi}e_{mi'}$。所以 $k$ 越大应力越小，取 $k$ 个最大的特征值即最优——这就是 Exercise 14.11「$z_i$ 是 $E_kD_k$ 的**行**」的结论（$D_k=\mathrm{diag}(\sqrt{\lambda_1},\dots,\sqrt{\lambda_k})$）。

> **结果** · 经典标度 = PCA（当相似度是中心化内积时）
>
> 经典标度 (14.100) 的准则
> $$S_C(z_1,\dots,z_N)=\sum_{i\ne i'}\Big(s_{ii'}-\langle z_i-\bar z,\ z_{i'}-\bar z\rangle\Big)^2 \eqno{14.100}$$
> 在「$s_{ii'}$ 就是中心化内积 $\langle x_i-\bar x,x_{i'}-\bar x\rangle$」时，$\boldsymbol{S}$（中心化内积矩阵）$=\boldsymbol{G}\boldsymbol{X}^\top\boldsymbol{X}\boldsymbol{G}$，其特征向量就是 $X$ 的左奇异向量，于是 $z_i=(u_{1i}\sqrt{\lambda_1},\dots,u_{ki}\sqrt{\lambda_k})$——**逐字就是主成分得分**（对照 [§14.5.1](#s-14-5-1) 的 (14.51)）。
>
> **若手里是距离而不是内积**，且距离是**欧氏的**，就先用上面的双中心化把 $a_{ij}=d_{ij}^2$ 转成 $\boldsymbol{B}$，再做特征分解——这正是 Isomap 第二步要用的算法（原书注：见第 18 章 (18.31)）。若距离**不是**欧氏的（不可嵌入到任何欧氏空间），双中心化得到的 $\boldsymbol{B}$ 有负特征值，取 $\sqrt{\lambda}$ 就会出**虚数**——这是 MDS 里唯一真正麻烦的失败模式。

> **坑** · 原书明确：**经典标度与最小二乘标度不等价**——损失函数不同（内积 vs 距离），映射可以是非线性的；两者都归入**度量标度**（metric scaling，因为真的去逼近那些数值），而下面 (14.101) 的非度量标度只用**秩**。

### 14.8.3 非度量标度 (14.101) 与「度量 vs 非度量」 {#s-14-8-3}

> **结果** · 只保留秩的 Shephard–Kruskal 非度量标度
>
> 当相异度只是「谁大谁小」有意义（数值本身不可靠）时，让一个**单调递增函数** $\theta$ 去拟合：
> $$S_{NM}(z_1,\dots,z_N)=\sum_{i\ne i'}\Big[\lVert z_i-z_{i'}\rVert-\theta\big(d_{ii'}\big)\Big]^2 \eqno{14.101}$$
> 交替求解：$z_i$ 固定时，对每个 $d_{ii'}$ 求 $\theta$。求导得 $\partial S_{NM}/\partial\theta(d)=\sum_{i\ne i'\text{ 同箱}}2\big[\lVert z_i-z_{i'}\rVert-\theta\big]\cdot 1$，所以该步等价于对每个 $\theta$ 的取值点做**加权最小二乘的单调回归（isotonic regression）**，用 PAVA（pool adjacent violators algorithm）求解；$\theta$ 固定时用梯度下降更新 $z_i$。反复迭代至稳定。

### 14.8.4 与本库前文的衔接 {#s-14-8-4}

> **结果** · 三者的共同点与差别
>
> | 方法 | 输入 | 目标 | 解的形式 |
> |---|---|---|---|
> | 主曲面 / SOM | $x_i\in\mathbb{R}^p$ | 用 $p_{\rm low}$ 维流形**逼近数据** | 非线性映射 |
> | MDS（度量） | $d_{ij}$ 或 $s_{ii'}$ | 保持**全部**成对距离 / 内积 | 低维坐标 |
> | PCA | $x_i$ | 最大化二阶方差 | 线性映射 |
>
> 原书的关键辨析：「在主曲面和 SOM 里，原始特征空间中靠得近的点应该映射到流形上相邻，但特征空间中离得远的点也可能映射得很近；MDS 不太会这样，因为它显式地要保持所有成对距离。」Figure 14.43 用经典标度处理「半球」数据：两个簇被清楚分开，且红色簇更紧凑。

> **坑** · MDS 是**投影式**方法（只给坐标，不给「从低维回到高维」的模型），与 SOM/主曲面的**流形拟合**不同。经典标度在欧氏距离下等价于 PCA；非欧氏距离下才出现真正的非线性能力。

---

## 14.9 非线性降维与局部多维标度 {#s-14-9}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-9">原文 §14.9</a>

这一节处理一个具体病症：数据其实**贴着**一个内蕴的低维非线性流形（parabola、sphere、helix），但 PCA / 经典 MDS 是**线性**的。Figure 14.44 左图里橙色的点落在一条抛物线上，经典 MDS 只保住了「两端的距离」（因为欧氏距离把弯过去的距离算错了），于是判断「两端的两点应该接近」；右图用**局部** MDS 就把沿曲线的顺序保住了。原文的比喻是把流形「拉平（flattening）」。

原文介绍三种方法：**Isomap**、**LLE（局部线性嵌入）**、**Local MDS**。这一节的公式（14.102)–(14.106) 全部是后两种方法的；Isomap 是纯两步法，原书未给编号。

### 14.9.1 Isomap：用图最短路近似测地距离 {#s-14-9-1}

> **基础知识** · 图最短路与 $k$ 近邻图
>
> $k$ 近邻图：顶点为 $N$ 个样本，若 $i$ 的 $k$ 个欧氏最近邻里有 $j$（或反之），则连边，边权 $=d_{ij}$（原书用「某个小欧氏距离内的点」定义邻居，等价于取 $k$）。图上的**最短路径长度**用 Dijkstra 算法（或 Floyd–Warshall）计算。

> **推导** · 第一步：$d_{\rm geo}$ 为什么是测地距离的近似
>
> 记流形 $\mathcal{M}\subset\mathbb{R}^p$，$d_{\mathcal{M}}(x,y)$ 是 $\mathcal{M}$ 上的**内蕴（测地）距离**。建图后取
> $$d_{\rm geo}(i,j):=\min_{\gamma:\,i\to j\ \text{图上路径}}\ \sum_{\text{边}\,(u,v)\in\gamma}d_{uv}$$
> 两个方向的不等式都能证：
> 1. **$d_{\rm geo}\ge d_{\mathcal{M}}$**：任何图上路径 $\gamma$ 展开成样本点列，它在 $\mathcal{M}$ 上对应一条折线；折线长度 $\ge$ 流形上连接两端的曲线长度 $\ge d_{\mathcal{M}}$。取下确界即得。
> 2. **$d_{\rm geo}\lesssim d_{\mathcal{M}}$**：设样本足够密（$k$ 近邻半径 $\varepsilon$ 很小），则流形上任意一段长 $d_{\mathcal{M}}$ 的测地线可以沿着样本点走成 $\lceil d_{\mathcal{M}}/\varepsilon\rceil$ 段、每段长 $\le\varepsilon+O(\varepsilon^2)$ 的折线（局部 $C^2$ 曲面上的标准误差估计：弦长与弧长之差 $\approx \frac{L^3}{24\rho^2}$，$L\lesssim\varepsilon$、曲率 $\rho$ 固定，故 $\lesssim \varepsilon^3/\rho^2$）。累加得到 $d_{\rm geo}\le d_{\mathcal{M}}\big[1+O(\varepsilon^2/\rho^2)\big]$。
>
> **所以「两步」的第一步不是启发式，而是有误差界的**：采样越密，$d_{\rm geo}\to d_{\mathcal{M}}$。

> **推导** · 第二步：在 $d_{\rm geo}$ 上做经典 MDS 为什么合理
>
> 把 $a_{ij}=d_{\rm geo}(i,j)^2$ 送进 §14.8.2 的双中心化 $\boldsymbol{B}=G\boldsymbol{A}G$，取前 $k$ 个特征向量作坐标。这是「**先得到流形的内在度量，再用 MDS 把这个内在度量线性化**」。
> - 若 $\mathcal{M}$ 本身**内蕴平坦**（isometric to a subset of $\mathbb{R}^k$），第二步在 $k$ 维里**精确**（$B$ 只有 $k$ 个非零特征值）。
> - 否则 MDS 给出的是「在欧氏空间里尽量还原 $d_{\rm geo}$ 的最佳配置」——这就是它的定义，没有更强的承诺。
>
> **Isomap 与谱聚类的同构**：若第一步的图是 $k$ 近邻图，第二步的核心矩阵 $G\boldsymbol{A}G$ 与 §14.5.3 的图拉普拉斯一样只用到**图的局部信息**。原书末尾明确点出这条联系：「There are also close connections between the methods discussed here, spectral clustering (Section 14.5.3) and kernel PCA (Section 14.5.4).」

### 14.9.2 LLE：局部仿射重构 (14.102) 与谱解 (14.104) {#s-14-9-2}

> **基础知识** · 局部线性嵌入（Roweis 与 Saul, 2000）
>
> LLE 的赌注：数据局部看着像仿射的，所以「用邻居的仿射组合去重构 $x_i$」得到的权重 $w_{ik}$ 就是该点的**局部坐标系**；把这个重构关系原封不动搬到低维，就保住了局部几何。

> **推导** · 第一步：权重 (14.102)
>
> 对每个 $x_i\in\mathbb{R}^p$，找它的 $K$ 个最近邻 $\mathcal{N}(i)$，解
> $$\min_{W_{ik}}\ \Big\lVert x_i-\sum_{k\in\mathcal{N}(i)}w_{ik}x_k\Big\rVert^2\qquad\text{s.t.}\quad w_{ik}=0\ (k\notin\mathcal{N}(i)),\ \sum_{k=1}^{N}w_{ik}=1 \eqno{14.102}$$
> **推导（为什么只有一个和为 1 的约束就够）**：把 $x_k$ 换成 $x_k-x_i$、$w$ 换成 $w_{ik}-\frac1K$，约束变成 $\sum_k(w_{ik}-\frac1K)=0$，目标里 $x_i$ 消失：
> $$\min\ \sum_{k,l\in\mathcal{N}(i)}(w_{ik}-\tfrac1K)(w_{il}-\tfrac1K)\langle x_k-x_i,x_l-x_i\rangle\ \ \text{s.t.}\ \sum_k(w_{ik}-\tfrac1K)=0$$
> 这是带一次齐次约束的二次规划，$\mathcal{N}(i)$ 上的 Gram 矩阵若满秩（要求 $K<p$，原书明确说了），KKT（见预备知识 O2）给出
> $$M_iw_i=-\frac{1-\mathbf{1}^\top M_i^{-1}\mathbf{1}}{1-\mathbf{1}^\top M_i^{-1}\mathbf{1}}\ \mathbf{1}\ \mathbf{1}^\top M_i^{-1}\mathbf{1}$$
> 其中 $M_i$ 是近邻的 $K\times K$ Gram 矩阵；$W_i$ 由这个闭式给出（「$w_{ik}$ 是点 $k$ 对重构点 $i$ 的贡献」）。所以 $w$ 允许为负——它是**仿射**组合，不是凸组合。

> **推导** · 第二步与第三步：(14.103) → (14.104)
>
> 固定 $w_{ik}$，在 $\mathbb{R}^{d<p}$ 中找 $y_1,\dots,y_N$：
> $$\min_{y_i}\ \sum_{i=1}^N\sum_{k=1}^N\Big\lVert y_i-w_{ik}y_k\Big\rVert^2 \eqno{14.103}$$
> 记 $Y$ 为 $N\times d$ 的坐标矩阵、$W=(w_{ik})$ 为 $N\times N$ 的权重矩阵（$W$ 是**行随机**的：$\sum_kw_{ik}=1$），则
> $$\sum_{i,k}\lVert y_i-w_{ik}y_k\rVert^2=\sum_i\lVert (WY)_i-(WY)_{ii}\Big\rVert^2=\mathrm{tr}\big[(Y-WY)^\top(Y-WY)\big]=\mathrm{tr}\big[Y^\top(I-W)^\top(I-W)Y\big] \eqno{14.104}$$
> （第三步用了 $\mathrm{tr}(A^\top B)=\sum_{ij}A_{ij}B_{ij}$，把 $A=(Y-WY)^\top$、$B=(Y-WY)$ 代入即得。）
>
> **解**：记 $M=(I-W)^\top(I-W)$（Gram 矩阵，半正定）。对 $Y^\top Y=I_d$ 极小化 $\mathrm{tr}(Y^\top MY)$，KKT 给出 $MY=Y\Lambda$，即 **$Y$ 是 $M$ 的最小特征向量**（与 [§14.5.3.2](#s-14-5-3-2) 的 $Lv=\lambda v$ 完全同形）。
>
> **为什么要丢掉平凡特征向量。** 因为 $W\mathbf1=\mathbf1$（行随机）$\Rightarrow (I-W)\mathbf1=0\Rightarrow M\mathbf1=0$：$\mathbf1$ 是特征值 0 的特征向量。它对应的「解」是所有 $y_i$ 都相同（退化）。原书：丢弃它、保留下一个 $d$ 个。并且由于这 $d$ 个特征向量都 $\perp\mathbf1$（$M$ 对称且 $M\mathbf1=0$，零特征空间含 $\mathbf1$，最小特征子空间与之正交——严格地说需按重数处理），坐标自动满足 $\mathbf1^\top Y=0$，即**嵌入被中心化**。
>
> **为什么这能「拉平」流形**：若流形局部看是仿射的（$k\ll$ 弯曲程度），则 $x_i=\sum_kw_{ik}x_k$ 也近似成立；把同样的线性组合搬到 $y$ 上，即 $y_i\approx\sum_kw_{ik}y_k$，就意味着**每个邻域内的仿射关系被保留**。这正是「局部仿射结构」不变量的做法。

> **坑** · 需要 $K<p$ 才可能唯一（(14.102) 下）；实际还要 $M_i$ 良态。若近邻跨过流形的「褶皱」，$w$ 会失真，LLE 对邻居数 $K$ 非常敏感——原书 Figure 14.44 的实验用**局部 MDS** 而非 LLE，Figure 14.45（1965 张人脸照片）才用 LLE。

### 14.9.3 局部 MDS (14.105)(14.106) {#s-14-9-3}

> **基础知识** · 局部 MDS（Chen 与 Buja, 2008）
>
> $\mathcal{N}$ 是**对称的**近邻对集合：对 $(i,i')$，若 $i$ 在 $i'$ 的 $K$ 近邻里，或反之，就收进 $\mathcal{N}$。应力函数是
> $$S_L(z_1,\dots,z_N)=\sum_{(i,i')\in\mathcal{N}}\big(d_{ii'}-\lVert z_i-z_{i'}\rVert\big)^2+\sum_{(i,i')\notin\mathcal{N}}w\cdot\big(D-\lVert z_i-z_{i'}\rVert\big)^2 \eqno{14.105}$$
> $D$ 是很大的常数、$w$ 是小权重。含义：非近邻对被**当成很远**处理，但权重小，所以不会主宰整个应力。

> **推导** · 化简 (14.105) 得 (14.106)
>
> 取 $w\sim 1/D$，令 $D\to\infty$。第二项 $\approx \frac1D\big(D^2-2D\lVert z_i-z_{i'}\rVert+\lVert z_i-z_{i'}\rVert^2\big)\approx D-2\lVert z_i-z_{i'}\rVert$（$\mathcal{N}$ 的补集有 $O(N^2)$ 对，常数项 $O(N^2D)$ 与 $z$ 无关，可丢），故
> $$S_L(z_1,\dots,z_N)=\sum_{(i,i')\in\mathcal{N}}\big(d_{ii'}-\lVert z_i-z_{i'}\rVert\big)^2-\tau\sum_{(i,i')\notin\mathcal{N}}\lVert z_i-z_{i'}\rVert,\qquad \tau=2wD \eqno{14.106}$$
> **第一项**保持局部结构（近邻对的距离要被保留），**第二项**鼓励非近邻对离得远（等价于把非近邻压到「无穷远」，从而避免流形自我折叠）。注意第二项与 (14.99) Sammon 映射里的 $-\lVert z_i-z_{i'}\rVert$ 是同一种「推远」势能——差别只在 Sammon 有权重 $1/d$ 且求和遍及所有对。
>
> **算法**：固定 $K$ 与 $\tau$，用坐标下降在 (14.106) 上极小化。原书 Figure 14.44 右图用 $K=2$、$\tau=0.01$、多个起点（因为 (14.106) 非凸）。Chen 与 Buja (2008) 的实验显示局部 MDS 优于 Isomap 与 LLE，且特别适合**图布局**（graph layout）。

### 14.9.4 两种同期方法：LTSA 与扩散映射 {#s-14-9-4}

> **延伸** · 局部切空间对齐 LTSA（Donoho 与 Grimes, 2004；Chen 与 Buja 2008 有比较）
>
> LTSA 与 LLE 是「同一张考卷的两种答卷」：都取 $K$ 近邻，都用**局部 PCA** 找切空间，区别只在「局部坐标怎么搬到全局」。设 $E_i\in\mathbb{R}^{|\mathcal{N}_i|\times p}$ 是近邻块的行选择矩阵，$Y_i$ 是 $Y$ 的对应行块，目标是把每个近邻块 $Y_i$ 换成它的最优秩-$m$ 近似 $\hat Y_i$：
> $$\min\ \sum_{i=1}^N\lVert Y_i-\hat Y_i\rVert_F^2\qquad\text{s.t.}\ \mathrm{rank}(\hat Y_i)\le m$$
> 展开成迹形式后极小点由「$Y$ 在每个近邻块内都是局部坐标的线性函数」刻画，等价地取
> $$M=\sum_{i=1}^N\frac1{|\mathcal{N}_i|}E_i^\perp{}^\top E_i^\perp$$
> 的**最小 $m$ 个特征向量**。理由：$M$ 的零空间 = 「在每个近邻块内都能被某组局部切坐标线性表达」的方向集合，其维数恰为 $N-m$（一个 $m$ 维流形上的 $N$ 个点，每个点的局部线性函数给出 $N-m$ 个自由度），所以「最小特征向量」有且只有 $m$ 个可用。**注意这与谱聚类的差别只在「取哪一端」**：这里取最小（零空间最大），谱聚类那里也取最小（连通性）。

> **延伸** · 扩散映射（diffusion maps, Coinfman 与 Lafon）
>
> 换一个「局部距离」的定义：不直接用欧氏距离/测地距离，而用**随机游走的扩散距离**。取 $k$ 近邻图的对称归一化转移算子
> $$P=MS,\qquad M=D^{-1/2},\quad D_{ii}=\sum_jS_{ij},\quad S_{ij}=e^{-\lVert x_i-x_j\rVert^2/(\sigma^2\varepsilon)}$$
> （$\varepsilon$ 是小带宽）。记 $P$ 的特征对 $Pv_k=\lambda_kv_k$，$k=0,1,\dots$，$\lambda_0=1>v_1\ge\cdots\ge0$。用
> $$\Phi_t=\big(\sqrt{\lambda_0^t},\sqrt{\lambda_1^t}v_1,\dots,\sqrt{\lambda_{k}^t}v_k\big)$$
> 的列作坐标（$\Phi_0=\mathbf1$，$\sqrt{\lambda_0^t}v_0=\mathbf1$）。**为什么用 $\lambda^t$ 而不是 $\lambda$**：$\lambda_k$ 只是单步转移概率的收缩率，而 $\lambda_k^t$ 是 $t$ 步的收缩率；取 $t$ 使 $\lambda_1^t$ 不太小、$\lambda_2^t$ 已很小，就把「全局拓扑」（$v_0,v_1$）和「局部噪声」（$v_k,\ k\ge2$）分开。等价的**扩散核**写法是
> $$k_t(x_i,x_j)=\sum_m\lambda_m^t v_m(x_i)v_m(x_j)\ \approx\ \exp\big(t\,L(x_i,x_j)\big)\ \text{（小 }t\text{）}$$
> 其中 $L=I-P$ 是（归一化）图拉普拉斯（与 [§14.5.3.1](#s-14-5-3-1) 同一个矩阵）。这是「流形上的热核」在图上的离散化：$t\to0$ 时恢复欧氏局部度量，$t$ 大时沿流形扩散。
>
> **三种方法的共同主题**：都试图**把流形拉平**，区别只在**如何定义「局部距离」**：
> - Isomap：**测地距离**（局部欧氏 → 全局最短路）；
> - LLE / LTSA：**局部仿射关系**（局部坐标不变量）；
> - Local MDS：**局部距离 + 非局部排斥**；
> - 扩散映射：**热核 / 多步扩散距离**。
>
> **坑** · 这一整套只在**信噪比很高**时有效（原书：「they are useful for problems where signal-to-noise ratio is very high (e.g., physical systems), and are probably not as useful for observational data with lower signal-to-noise ratios」）。低信噪比下「近邻」本身就不稳定，所有局部估计都失效。

---

## 14.10 Google PageRank 算法 {#s-14-10}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-10">原文 §14.10</a>

PageRank 是本书里唯一一个「跑在全球规模上」的算法，也是**唯一**一个把无监督学习（找结构）和工程约束（$10^9$ 级稀疏矩阵、要能算完）逼到一起的例子。它与 §14.5.3 谱聚类的共同点是**都归结为某个矩阵的主特征向量**；与 §17 图模型的共同点是**都是图上的平稳测度**。

### 14.10.1 链接矩阵与递推定义 (14.107)(14.108) {#s-14-10-1}

> **基础知识** · 链接矩阵
>
> $L_{ij}=1$ 若**页面 $j$ 指向页面 $i$**，否则 0。于是
> $$c_j=\sum_{i=1}^N L_{ij}$$
> 是**页面 $j$ 的出链数**（数的是「第 $j$ 列里有几个 1」）。$\boldsymbol{D}_c=\mathrm{diag}(c_1,\dots,c_N)$，$\boldsymbol{e}$ 是 $N$ 维全 1 向量。

> **结果** · PageRank 的递推定义 (14.107)
>
> $$p_i=(1-d)+d\sum_{j=1}^N\Big(\frac{L_{ij}}{c_j}\Big)p_j \eqno{14.107}$$
> 其中 $d\in(0,1)$ 是阻尼因子，原书说「apparently set to $0.85$」。三项含义：
> - $\sum_j\frac{L_{ij}}{c_j}p_j$：**所有指向 $i$ 的页面**把 PageRank 汇总过来，页面 $j$ 只分出 $p_j/c_j$——因为它要「一票分散到所有出链」，出链越多每条分到的越少；
> - $(1-d)$：兜底常数，保证「每个页面的 PageRank 至少是 $1-d$」，也让方程组有唯一解（见 §14.10.3）。
>
> 矩阵形式：$p_i$ 对应 $(LD_c^{-1}p)_i$（**注意 $D_c^{-1}$ 在右边**，因为要按**列** $j$ 除），常数项 $(1-d)e$。所以
> $$p=(1-d)e+d\,\boldsymbol{L}\boldsymbol{D}_c^{-1}p \eqno{14.108}$$

> **推导** · 为什么 $\sum_i p_i=N$ 自动成立（Exercise 14.22(a)）
>
> (14.108) 左右两边同乘 $e^\top$。关键是 $e^\top\boldsymbol{L}\boldsymbol{D}_c^{-1}=e^\top$，因为它的第 $j$ 个分量为 $\sum_i L_{ij}/c_j=c_j/c_j=1$。所以
> $$e^\top p=(1-d)e^\top e+d\,e^\top p=(1-d)N+d\,e^\top p\ \Longrightarrow\ (1-d)e^\top p=(1-d)N\ \Longrightarrow\ e^\top p=N$$
> **不需要额外归一化**：只要 $p$ 满足 (14.108)，和就自动是 $N$（等价于平均 PageRank 为 1）。反过来，由 $e^\top p=N$ 可以把常数项改写成矩阵形式：
> $$\boxed{\ p=\Big[(1-d)\frac{ee^\top}{N}+d\,\boldsymbol{L}\boldsymbol{D}_c^{-1}\Big]p=\boldsymbol{A}p \ \eqno{14.109}}$$
> 推导：$(1-d)e=(1-d)\frac{ee^\top p}{N}$，用 $e^\top p=N$ 即得。

### 14.10.2 转移矩阵 (14.109) 的三条性质 {#s-14-10-2}

> **结果** · $\boldsymbol{A}$ 是**列随机**矩阵，且**严格正**
>
> $$\boldsymbol{A}=(1-d)\,E+d\,Q,\qquad E=\frac{ee^\top}{N},\quad Q=\boldsymbol{L}\boldsymbol{D}_c^{-1}$$
> 1. **列和为 1**：$e^\top A=(1-d)\frac{e^\top ee^\top}{N}+d\,e^\top QD_c^{-1}=e^\top$。（这里 $E$ 与 $Q$ 都列随机。）
> 2. **特征值 1**：由 1，$e^\top A=e^\top$。
> 3. **严格正**：每个元素 $\ge(1-d)/N>0$（当 $d<1$），所以 $A$ **本原**（primitive）。

> **推导** · 特征值 1 一定是最大模特征值（谱半径 $\rho=1$）
>
> $A$ 非负、列和为 1，故按定义 $\lVert A\rVert_1=\max_j\sum_i|A_{ij}|=1$，而对任意矩阵 $\rho(A)\le\lVert A\rVert_1$，所以 $\rho(A)\le1$；又由性质 2，$\rho(A)\ge1$。两边夹住得
> $$\rho(\boldsymbol{A})=1$$
> 原书说「$A$ 有一个实特征值等于 1，且它是最大特征值」，推导就在这两行里。

### 14.10.3 幂迭代 (14.110) 与收敛速率 {#s-14-10-3}

> **结果** · PageRank 的实际算法 (14.110)
>
> 从任意 $\boldsymbol{p}_0$ 出发反复迭代，每步归一化到和为 $N$：
> $$\boldsymbol{p}_k\ \longleftarrow\ \boldsymbol{A}\boldsymbol{p}_{k-1};\qquad \boldsymbol{p}_k\ \longleftarrow\ N\frac{\boldsymbol{p}_k}{\boldsymbol{e}^\top\boldsymbol{p}_k} \eqno{14.110}$$
> 固定点 $\hat p$ 就是 PageRank。**第二个归一化在数学上是恒等操作**：由 $e^\top A^k=e^\top$ 与 $e^\top A^kp_0=e^\top p_0$，$Ap_{k-1}$ 的和恒等于 $p_0$ 的和（归一化后为 $N$）。写成这种形式纯粹是为了对付浮点误差累积。

> **推导** · 为什么幂迭代收敛、收敛有多快
>
> 设 $A$ 可对角化，$\boldsymbol{1}=\sum_i\lambda_iu_iv_i^\top$（含重根时结论不变，只是误差多一个多项式因子）。由 $Au_i=\lambda_iu_i$ 得
> $$\boldsymbol{A}^k=\sum_{i=1}^N\lambda_i^ku_iv_i^\top=p\,\boldsymbol{e}^\top+\sum_{i\ge2}\lambda_i^ku_iv_i^\top+O\big(k\lvert\lambda_2\rvert^{k-1}\big)$$
> 其中第一项是「$p$ 作用在 $\mathbf1^\top p$ 上」：因为 $e^\top u_1=1$（$u_1=\mathbf1/\sqrt N$），所以 $p\,e^\top$ 在 $e^\top p$ 方向上等于恒等算子。于是
> $$\boldsymbol{p}_k-\boldsymbol{p}\ \sim\ \sum_{i\ge2}\lambda_i^ku_iv_i^\top$$
> **收敛因子是 $\lvert\lambda_2\rvert$**（谱间隙的倒数决定速度）。这是**谱聚类里完全同一个判据**（[§14.5.3.2](#s-14-5-3-2) 用 $\lvert\lambda_2\rvert/\lvert\lambda_1\rvert$ 判可分性）。原书：「Exploiting a connection with Markov chains, it can be shown that the matrix $A$ has a real eigenvalue equal to one, and one is its largest eigenvalue.」
>
> **不可对角化时**（Jordan 块）误差变成 $O(k\lvert\lambda_2\rvert^{k-1})$——阶乘因子，这正是**非正规矩阵幂迭代可能不收敛**的原因，需要阻尼压住。

> **结果** · 阻尼因子 $d$ 的三个作用（一次说清）
>
> 1. **保证唯一解**。$d<1$ 时 $A\ge(1-d)/N>0$，故 $A$ 本原。
> 2. **保证收敛**。$P$（下）本原 ⇒ 特征值 1 **简单**，其余特征值 $\lvert\lambda\rvert<1$（Perron–Frobenius，见 §14.10.4）。
> 3. **控制速度**。$\lvert\lambda_2\rvert\le d$，所以阻尼越大收敛越快但偏离「真实随机游走者」越远；$d\to1$ 链变慢且对网页结构更敏感。这是 PageRank 的经典 trade-off，也是后来 personalized PageRank / Teleportation 主题的动机。

### 14.10.4 Markov 链解释与 Perron–Frobenius {#s-14-10-4}

> **基础知识** · Markov 链与 Perron–Frobenius 定理
>
> **（a）转移矩阵与平稳分布**：$N$ 状态链的转移矩阵 $P$（行随机：$P_{ij}=\Pr(X_{t+1}=j\mid X_t=i)$）。分布向量 $\pi$（行向量）**平稳**当且仅当
> $$\pi P=\pi \iff P^\top\pi=\pi$$
> 即 $\pi$ 是 $P$ 特征值 1 的特征向量。唯一性存在当且仅当 $P$ 不可约（irreducible：从任一状态出发能以正概率到达任一其它状态）；此时逐点收敛 $P^k\to\mathbf1\pi$。若 $P$ 还是非周期的（aperiodic），则 $P^k$ 全矩阵收敛。
>
> **（b）Perron–Frobenius 定理（第一部分）**：非负矩阵 $A$ 的谱半径 $\rho=\rho(A)$ 是 $A$ 的特征值，且 $\lvert\lambda\rvert\le\rho$。若 $A$ **不可约**，则 $\rho$ 是**单特征值**。若 $A$ **不可约且非周期**（等价地 $A$ 本原、primitive，即存在 $m$ 使 $A^m>0$），则
> $$\boldsymbol{A}^k\to v w^\top\qquad(k\to\infty),\qquad \boldsymbol{A}v=\rho v$$
> 其中 $v>0$，$w^\top v=1$；对非负向量 $z$，$\lVert z-v\rVert_2/\lVert z\rVert_2\to0$。这与 (14.110) 的幂迭代完全一致。

> **推导** · 把 PageRank 写成随机游走者的平稳分布
>
> 原书给了两个版本：
> 1. **「以概率 $1-d$ 随机跳到均匀随机页面」版**：从 $j$ 到 $i$ 的转移概率 $A_{ij}=(1-d)/N+d\,L_{ij}/c_j$，这正是 §14.10.2 的 $A$（列随机，因为 $j$ 是出发地）。此时 PageRank 解 $x=p/N$ 满足 $Px=x$，$x$ 落在单纯形上。
> 2. **「把常数项写成 $1-d$」版**：即 (14.107) 本身，对应另一条（转移概率不同的）链。
>
> 原书：「Viewing PageRank as a Markov chain makes clear why the matrix $A$ has a maximal real eigenvalue of 1. Since $A$ has positive entries with each column summing to one, Markov chain theory tells us that it has a unique eigenvector with eigenvalue one, corresponding to the stationary distribution of the chain.」
>
> **严格对应**：由 §14.10.2，$A\ge(1-d)/N>0$，故 $A$ 本原；$A$ 列随机，故 $e^\top A=e^\top$ 且 $\rho(A)=1$（§14.10.2 已证）。Perron–Frobenius 给出 $1$ 是**单特征值**，$v>0$ 且唯一。取 $x=p/N$（$\sum_ix_i=1$、$x_i>0$），由 $e^\top Ap=p$（§14.10.1）可知 $p$ 落在特征值 1 的特征空间（与 $e$ 张成的方向无关，$Ap$ 的和仍是 $p$ 的和），故 $p$ 就是那个唯一正特征向量。这就是 (14.110) 的收敛性保证。
>
> **坑** · 死链（$c_j=0$ 的页面）会让 $L_{ij}/c_j$ 无定义。实务里把死链的出链当作指向所有页面（等价于把它当成一个 teleport 点），而 (14.91) 的第一项 $\int\varphi e^g$ 型「均匀兜底」在 PageRank 里就是干这件事的——**PageRank 的阻尼项在数学上等价于对死链补边**。

### 14.10.5 PageRank 作为 Gibbs / 单点更新 {#s-14-10-5}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-10">原文 §14.10</a>

<a class="src" href="../esl/ch17-undirected-graphical-models.html#s-17-2">对照 [第 17 章 §17.2](m17.html#s-17-3-4)</a>

这是 PageRank 最漂亮的一层：它的迭代式和 Gibbs 采样的单点更新**字面上是同一种东西**，差别只在「场」是不是可加的。

> **基础知识** · Gibbs 单点（热浴）更新
>
> 能量 $U(x)$ 的 Gibbs 分布 $\pi(x)\propto e^{-U(x)}$。热浴采样抽 $x_i$ 时按条件分布重抽，等价于**乘性更新**
> $$x_i\ \longleftarrow\ \frac{\exp(u_i)}{\sum_k\exp(u_k)},\qquad u_i=-\frac{\partial U}{\partial x_i}\Big|_{\text{其他坐标固定}}$$
> 二元（Ising/Potts）情形退化为 $x_i\leftarrow\sigma(u_i)$。参 [第 17 章](m17.html#s-17-3-4) 的 Potts 模型（铁磁/反铁磁耦合 $J$ 与外场 $h$）。

> **推导** · PageRank 的更新就是 softmax 单点更新
>
> 记 $Q=LD_c^{-1}$（列随机）、$\varepsilon=(1-d)/N$。PageRank 的幂迭代（用 $x=p/N$，则 $A$ 的作用变成 $x\mapsto\varepsilon+Q^\top x$）是
> $$x_i^{\star}\ \longleftarrow\ \frac{\exp(u_i}{\sum_k\exp(u_k)}\ \text{里}...\ \Rightarrow\ x_i^{\text{new}}=\big(Q^\top x\big)_i+\varepsilon$$
> 把它写成 softmax 形式只需取
> $$u_i=\log\Big[\big(Q^\top x\big)_i+\varepsilon\Big]$$
> 于是
> $$x_i^{\text{new}}=\frac{\exp(u_i)}{\sum_k\exp(u_k)}\ \Big(\text{因为 }\sum_k[(Q^\top x)_k+\varepsilon]=1,\ Q\text{ 列随机}\Big)$$
> **不动点检验**：若 $x$ 已满足 PageRank $(Q^\top x)_i=x_i-\varepsilon$，则分子 $=(x_i-\varepsilon)+\varepsilon=x_i$，分母 $=\sum_k(x_k-\varepsilon+\varepsilon)=1$，故 $x^{\text{new}}=x$。✓
>
> 所以「场」$u_i$ 是把 $x$ **线性**映射再取对数：**PageRank 是 Gibbs 更新中「耦合矩阵可加」的那个特例**。一般 Gibbs 分布的场 $u_i(x_{-i})$ 可以是任意非线性泛函；PageRank 只能取线性泛函。

> **结果** · 与 Potts 型 MRF 的精确关系
>
> 取二元 Ising/Potts 单点更新 $x_i\leftarrow\sigma(\sum_kJ_{ik}x_k+h_i)$，$x\in\{0,1\}^N$（$J$ 由 (14.107) 取 $J_{ik}=d\,L_{ik}/c_k$，$h_i=0$）。不动点满足
> $$\log\frac{x_i}{1-x_i}=d\big(Q^\top x\big)_i$$
> 对比 PageRank 的 $x_i=\varepsilon+d(Q^\top x)_i$：
>
> | | 不动点方程 | 联系 |
> |---|---|---|
> | Potts / Ising（MRF） | $\log\frac{x_i}{1-x_i}=d(Q^\top x)_i$ | 对数线性，需 $J$ **对称**（无向图） |
> | PageRank | $x_i=\varepsilon+d(Q^\top x)_i$ | 线性 + 归一化 |
>
> **PageRank 是 Potts 更新把 $\sigma$ 换成恒等映射后的高斯/高温极限**：$\sigma(u)\approx\tfrac12+\tfrac{u}{4}$（$u$ 小时），把 LHS 线性化即得 PageRank 形式，而 $\varepsilon$ 正是「$x_i$ 的先验均匀概率」。换句话说：**PageRank = 无向图模型在高温（无相互作用）极限下的 softmax 平衡点**，只是它把「相互作用」降级成了单向的 $Q$。

> **推导** · 唯一真正的结构差异：$Q$ 是**有向、非对称**的
>
> $Q_{ij}=L_{ij}/c_j$ 一般不对称（页面 $i$ 链到 $j$ 不蕴含 $j$ 链到 $i$），因此它**不是**任何无向图的权重矩阵，不能直接写成 Potts 的 $x_ix_k$ 项。若强行对称化，取 $J=\tfrac12(Q+Q^\top)$：
> $$\big[(Q+Q^\top)x\big]_i=\big(Q^\top x\big)_i+\big(Qx\big)_i$$
> 于是
> $$u_i=\log\Big[\tfrac d2\big[(Q+Q^\top)x\big]_i+\tfrac{\varepsilon}{2}(1+x_i)\Big]$$
> 拆开：$\tfrac d2(Q+Q^\top)x_i$ 是**对称耦合**（可以写成 $\tfrac d2\sum_kJ_{ik}x_k$，即 Potts 的场），$\tfrac{\varepsilon}2 x_i$ 是**自相互作用**，$\tfrac\varepsilon2$ 是常数（因为 $\sum_kx_k=1$）。
>
> **所以 PageRank 的更新规则可以看成 MRF 在「只有外场、没有 $x_ix_k$ 二次耦合」的特例下的单点 Gibbs 更新**，加上 $Q$ 的非对称部分（相当于把 $Q$ 拆成对称部分 + 一个非平衡「漂移」项）。§17 里 Markov 场的平稳分布靠的是**对称性**保证可逆性；PageRank 的链**不是可逆的**（一般 $Q_{ij}\ne Q_{ji}$），这就是为什么必须靠阻尼 $d<1$ 去保证收敛——**不可逆 ⇒ 谱间隙不能从对称性得到**。

> **坑** · 别把 PageRank 误读成 Gibbs 分布的采样：它的迭代是**确定性**的（在 $A$ 上做幂方法），而 Gibbs 是**随机**采样；两者的共同点只是「乘性单点更新」这个代数形式。若要随机化，就是 §14.10.4 的随机游走者模型（它才是真正的马尔可夫链，只是平稳分布相同）。

### 14.10.6 一个四页面的小例子 (14.111) {#s-14-10-6}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-10">原文 §14.10</a>

> **结果** · 链接矩阵与出链数
>
> $$L=\begin{pmatrix}0&0&1&0\\ 1&0&0&0\\ 1&1&0&1\\ 0&0&0&0\end{pmatrix},\qquad \boldsymbol{c}=(2,1,1,1) \eqno{14.111}$$
> 读法（$L_{ij}=1$ 表示 $j$ 指向 $i$）：第 1 页指向 3 页；第 2 页指向 1 页；第 3 页指向 1、2、4 页；第 4 页**没有**入链。列和 $\boldsymbol{c}=(2,1,1,1)$ 正是出链数（第 1 页有 2 条出链，第 2、3、4 页各 1 条）。

> **推导** · 手算 PageRank
>
> $$Q=LD_c^{-1}=\begin{pmatrix}0&0&1&0\\ \tfrac12&0&0&0\\ \tfrac12&1&0&1\\ 0&0&0&0\end{pmatrix}$$
> （每一行是把 $L$ 的第 $j$ 个分量除以 $c_j$。）代入 (14.107)，取 $d=0.85$：
> $$\begin{aligned} p_1&=0.15+0.85\,p_3\\ p_2&=0.15+0.85\tfrac12 p_1\\ p_3&=0.15+0.85\big(\tfrac12p_1+p_2+p_4\big)\\ p_4&=0.15 \end{aligned}$$
> 逐个代入：$p_1=0.15+0.85(1.58)=1.49$；$p_2=0.15+0.85\cdot0.745=0.78$；$p_3=0.15+0.85(0.745+0.78+0.15)=1.57\approx1.58$；$p_4=0.15$。故
> $$\hat p=(1.49,\ 0.78,\ 1.58,\ 0.15)$$
> **三个自查**：(1) 和 $=4.0=N$，与 §14.10.1 的 $e^\top p=N$ 一致；(2) 第 4 页**没有入链**（$L$ 的第 4 行全为 0），故求和项为空、拿到的恰是兜底常数 $1-d=0.15$，这就是原文「page 4 has no incoming links, and hence gets the minimum PageRank of 0.15」；(3) 第 3 页虽然只有一个入链（第 1 页），但那一票来自 PageRank 最高的页面之一，所以它反而最高——这正是 PageRank 的递归直觉。

> **结果** · 全章的谱论统一
>
> 把 §14.5.3（谱聚类：$L$ 的最小特征向量）、§14.5.4（核 PCA：中心化 Gram 的特征分解）、§14.8（经典 MDS：$-\tfrac12B\boldsymbol{A}$ 的特征分解）、§14.9（Isomap/LLE/LTSA：图的拉普拉斯或 $(I-W)^\top(I-W)$ 的特征向量）、§14.10（PageRank：列随机 $A$ 的特征值 1 特征向量）放在一起：**降维、聚类、图布局、网页排名在这一章里是同一个动作——对某个对称/相似矩阵做特征分解，取哪一端由物理含义决定**。谱间隙 $|\lambda_2|$ 是贯穿全章的**唯一诊断量**：它既是谱聚类的可分性判据、也是 LLE 的退化判据、还是 PageRank 的收敛速度。

> **坑** · 本分片负责的编号止于 (14.111)。后续 (14.112) 起是第 14 章习题的公式（Procrustes 加权距离、谱方法的降秩对偶、ProDenICA 的样条矩条件、MM/minorize-maximize 算法的 (14.119)(14.120) 等），不在本分片范围内。另一处易混：§17 是**无向**图模型（Markov 场，靠 $J$ 对称保证可逆），PageRank 用的 $Q=LD_c^{-1}$ 是**有向**非对称的转移矩阵，只在「取 $J=\tfrac12(Q+Q^\top)$」这一步才与 MRF 接上，且这一接法**不保留** PageRank 的原始动力学。