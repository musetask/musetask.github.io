## 3.5 子集选择 {#s-3-5}

上一节把最小二乘的三个等价形式（正规方程、QR、SVD）串了起来，但它们有一个共同的软肋：估计量 $\hat\beta_{\rm ls}=(X^\top X)^{-1}X^\top y$ 没有任何惩罚项，$p$ 接近或超过 $N$ 时 $\mathrm{Var}(\hat\beta_j)=(X^\top X)^{-1}_{jj}\sigma^2$ 会爆掉（见预备知识 L1 关于条件数的讨论、第 3 章 (3.8)）。本节的做法是往目标函数里加一个「复杂度罚款」，用**收缩**（shrinkage）换掉方差。全文围绕两个罚款写：$\ell_2$ 罚款给出岭回归，$\ell_1$ 罚款给出 lasso；$\ell_0$ 罚款（真正的子集选择）作为动机写在最前面。对应原文 <a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-5">原文 §3.5</a>。

### 3.5.1 最佳子集选择与它的两种写法 {#s-3-5-1}

**问题**：把「只允许 $k$ 个系数非零」这件事写成数学。ESL 给两种写法 (3.41)(3.42)，要证明它们给出**同一个解集**。

> **基础知识** · 拉格朗日乘子与凸对偶（见预备知识 O1）
>
> 等式约束 $\min_\beta f(\beta)$ s.t. $g(\beta)=0$ 的最优解满足 $\nabla_\beta L=0$ 且 $\lambda^\top g=0$，其中 $L=f-\lambda^\top g$。
> 不等式约束版本（KKT，见预备知识 O2）还要加**对偶可行** $\lambda\ge0$ 与**互补松弛** $\lambda_i f_i(\beta^\star)=0$。

「至多 $k$ 个 $\beta_j$ 非零」等价于 $\lVert\beta\rVert_0=\sum_{j=1}^{p}\mathbf 1\{\beta_j\ne0\}\le k$。因此两种写法是

$$
\hat\beta_{\rm subset}=\arg\min_{\beta}\ \sum_{i=1}^{N}\Big(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\Big)^2+\lambda\sum_{j=1}^{p}\mathbf 1\{\beta_j\ne0\}\ \eqno{3.41}
$$

$$
\hat\beta_{\rm subset}=\arg\min_{\beta}\ \sum_{i=1}^{N}\Big(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\Big)^2\quad\text{s.t.}\ \sum_{j=1}^{p}\mathbf 1\{\beta_j\ne0\}\le t\ \eqno{3.42}
$$

注意 (3.41) 里拟合项与 (3.2) 的 RSS 相同；$X$ 已经吸收了截距列，所以 (3.41)(3.42) 的第一项就是 $\mathrm{RSS}(\beta)=(y-X\beta)^\top(y-X\beta)$（第 3 章 (3.3)）。

> **坑** · $\lVert\beta\rVert_0$ 在 $\beta_j=0$ 处**不连续**：从 $\beta_j=10^{-6}$ 走到 $0$，罚款从 $1$ 跳到 $0$。所以 (3.41) 不是凸问题，梯度法、拉格朗日对偶、KKT 那一整套工具在这里**全都不能直接用**。这正是人们退而求其次用 $\ell_1$、$\ell_2$ 代替 $\ell_0$ 的原因。
>
> 还有：$\lVert\beta\rVert_0$ 的**单位问题**。$\beta_j$ 的单位随 $x_j$ 的单位变，把 $x_j$ 乘以 $1000$ 就改变了罚款，所以 (3.41) 不可解释、不具备「与 $y$ 同量纲」这类不变性。$\lVert\beta\rVert_2$ 有同样的病，$\lVert\beta\rVert_1$ 稍好些但也没有完全解决。

**为什么两种写法等价（约束形式 → 惩罚形式）**。这是凸分析里最常用的一次「$\lambda\leftrightarrow t$ 互换」，我按等式约束做一遍，再对不等式补上互补松弛。

1. 先做**纯等式**版本，约束取成 $\lVert\beta\rVert_0=t$，即 $\sum_{j=1}^{p}u_j-t=0$，其中 $u_j=\mathbf 1\{\beta_j\ne0\}$。拉格朗日函数

$$
L(\beta,\lambda)=\mathrm{RSS}(\beta)+\lambda\Big(\sum_{j=1}^{p}\mathbf 1\{\beta_j\ne0\}-t\Big)
$$

2. 固定 $\beta$，只对单个坐标 $\beta_j$ 求方向导数。取 $h$ 充分小使 $\beta_j+h$ 与 $\beta_j$ 同号，则 $\mathbf 1\{\beta_j+h\ne0\}=\mathbf 1\{\beta_j\ne0\}=u_j$，所以

$$
\frac{\mathrm{d}}{\mathrm{d}h}L(\beta+h e_j,\lambda)\Big|_{h=0}=\frac{\partial\,\mathrm{RSS}(\beta)}{\partial\beta_j}+0=-2x_j^\top(y-X\beta)
$$

这一项与 $\lambda$ 无关，等式约束**没有给出任何一阶信息**。若 $\beta_j=0$ 且 $u_j$ 变化（加/减一个 $\varepsilon$），$L$ 跳变 $\lambda$，故最小化要求

$$
\lambda\Big[\mathbf 1\{\beta_j+h\ne0\}-u_j\Big]\ \text{对一切 }h\text{ 非负}\ \Longrightarrow\ \lambda=0
$$

3. 所以**等式约束** $\lVert\beta\rVert_0=t$ 乘上任何 $\lambda>0$ 都让 $L$ 沿坐标轴不下降，一阶条件根本定不出 $\beta$。这暴露了一件事：$\ell_0$ 问题的解**不唯一、不规则**（可能有几十万个等价的 $\beta$），没法用对偶刻画。

**不等式版本（这才是实用的推导）**。约束写成 $f_i(\beta)\le0$，$i=1,\dots,p$，取

$$
f_j(\beta)=\lVert\beta\rVert_0-t\le0,\qquad L(\beta,\lambda)=\mathrm{RSS}(\beta)+\lambda\big(\lVert\beta\rVert_0-t\big),\ \lambda\ge0
$$

此时对 $\beta_j$ 的方向导数仍然不含 $\lambda$（因为 $\lVert\beta\rVert_0$ 局部常数），但**最优点必须在约束的边界上**：若 $\lVert\beta^\star\rVert_0<t$（严格可行），取 $\lambda=0$ 再取 $\beta=\hat\beta_{\rm ls}$ 就严格更优，矛盾。所以

$$
\lVert\hat\beta^\star\rVert_0=t,\qquad \lambda^\star=0
$$

即最优解就是 $\lVert\beta\rVert_0\le t$ 下 RSS 最小的那点，且 $\lambda$ 携带的信息为 $0$。**这解释了为什么 (3.41) 中 $\ell_0$ 版本的 $\lambda$ 没有可解释的量纲**。

4. 真正有用的联系是**通过子集内部最小二乘**建立的。设 $S_\lambda=\{j:\beta_j(\lambda)\ne0\}$，对任意 $S\subseteq\{1,\dots,p\}$，在「支撑集恰为 $S$」的约束下 RSS 的最小值是

$$
\min_{\mathrm{supp}(\beta)\subseteq S}\mathrm{RSS}(\beta)=\mathrm{RSS}\big(X_S\hat\beta_S^{\rm ls},\ y\big)\ \text{其中}\ \hat\beta_S^{\rm ls}=(X_S^\top X_S)^{-1}X_S^\top y
$$

于是 (3.41) 精确地是「在所有 $S$ 里挑使 $\mathrm{RSS}_S+\lambda|S|$ 最小的那一个」：

$$
\hat\beta_{S_\lambda}=\hat\beta^{\rm ls}_{S_\lambda},\qquad S_\lambda\in\arg\min_{S\subseteq\{1,\dots,p\}}\Big[\mathrm{RSS}_S+\lambda\lvert S\rvert\Big]
$$

（(3.41) 中还多了截距的 $\beta_0$，若 $X$ 的第一列是 $\mathbf 1$ 则它被吸收。）**为什么 $\lvert S\rvert$ 越大 $\lambda$ 越大越好**：因为 $\mathrm{RSS}_S$ 关于 $S$ 单调不增（多加一个变量只会让 $\mathrm{RSS}$ 变小或不变），所以 $\mathrm{RSS}_S+\lambda\lvert S\rvert$ 的极小点随 $\lambda$ 单调右移。$\lambda\to0$ 时选出全部 $p$ 个变量（退化为 $\hat\beta_{\rm ls}$），$\lambda\to\infty$ 时选出空集。

> **结果** · (3.41) 与 (3.42) 等价的完整含义：$t$ 固定 ⇒ 子集大小固定为 $\lfloor t\rfloor$；$\lambda$ 固定 ⇒ 存在一个 $t$，使 (3.42) 在这个 $t$ 下的解**恰好**是 (3.41) 在这个 $\lambda$ 下的解。反向不唯一（同一个 $\lambda$ 可能对应一段 $t$）。这个「$\lambda\leftrightarrow t$ 一一对应」在 §3.6 的 lasso 路径里会再次精确地出现一次。

> **坑** · 组合爆炸：$\lvert S\rvert$ 的枚举量是 $2^p$，(3.41) 的精确解需要 $O(2^p)$ 次最小二乘（每次 $O(N\lvert S\rvert^2)$）。$p=40$ 就已经 $10^{12}$ 量级。这不是实现问题而是本质困难：**没有多项式时间的精确算法**（除非 $P=NP$，因为最小二乘子集和能编码 0-1 背包问题）。第 3 章后面所有算法（lasso、elastic net、LARS、回归树）都是**贪心/连续松弛**的近似。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-5-1">原文 §3.5 子集选择</a>

### 3.5.2 岭回归：配方、SVD 与几何 {#s-3-5-2}

**问题**：把罚款从 $\lVert\beta\rVert_0$ 换成 $\lVert\beta\rVert_2^2$，解会变成什么样子？答案是有闭式解 (3.44)，并且在 SVD 基下每个方向独立收缩 (3.47)。

> **基础知识** · 二次函数的配方（见预备知识 C2）
>
> 对任意对称正定 $A$ 与向量 $b$，配方恒等式为
> $$\beta^\top A\beta-2\beta^\top b=(\beta-A^{-1}b)^\top A(\beta-A^{-1}b)-b^\top A^{-1}b$$
> 验证：右边展开得 $\beta^\top A\beta-2\beta^\top b+b^\top A^{-1}b-b^\top A^{-1}b$。因为 $A$ 对称，$A^{-1}$ 也对称，所以交叉项是 $-2\beta^\top A A^{-1}b=-2\beta^\top b$。
>
> 由此得 $\min_\beta[\beta^\top A\beta-2\beta^\top b]=A^{-1}b$，最小值 $=-b^\top A^{-1}b$。

岭回归的**惩罚形式** (3.43) 是

$$
\mathrm{RSS}(\lambda)=(y-X\beta)^\top(y-X\beta)+\lambda\beta^\top\beta\ \eqno{3.43}
$$

> **推导** · 逐项展开 (3.43)，再配方。第一步展开平方：

$$
(y-X\beta)^\top(y-X\beta)=y^\top y-y^\top X\beta-\beta^\top X^\top y+\beta^\top X^\top X\beta=y^\top y-2\beta^\top X^\top y+\beta^\top X^\top X\beta
$$

（第二项与第三项相等是因为 $X^\top y$ 是向量，$\beta^\top X^\top y=(X^\top y)^\top\beta=y^\top X\beta$ 都是标量，用了转置不改变标量这个事实。）

第二步加上 $\lambda\beta^\top\beta$，并记 $A=X^\top X+\lambda I$（$I$ 是 $p\times p$ 单位矩阵）：

$$
\mathrm{RSS}(\lambda)=y^\top y+\beta^\top\big[X^\top X+\lambda I\big]\beta-2\beta^\top X^\top y=y^\top y+\beta^\top A\beta-2\beta^\top X^\top y
$$

第三步验证 $A$ 正定：$A$ 对称；且对一切 $\beta\ne0$，

$$
\beta^\top A\beta=\lVert X\beta\rVert_2^2+\lambda\lVert\beta\rVert_2^2>0
$$

（两个非负项之和，只要 $\beta\ne0$ 第二项就 $>0$，**不需要 $X$ 满列秩**。）这是预备知识 L1 里正定性的直接应用，也是 ridge 相对最小二乘的关键优势。

第四步代入 C2 的配方，取 $b=X^\top y$：

$$
\beta^\top A\beta-2\beta^\top X^\top y=\big(\beta-A^{-1}X^\top y\big)^\top A\big(\beta-A^{-1}X^\top y\big)-y^\top X A^{-1}X^\top y
$$

于是

$$
\mathrm{RSS}(\lambda)=\underbrace{y^\top y-y^\top XA^{-1}X^\top y}_{\text{与 }\beta\text{ 无关的常数}}+\big(\beta-\beta_0\big)^\top A\big(\beta-\beta_0\big),\qquad \beta_0=A^{-1}X^\top y
$$

第五步读极小：第一项常数，第二项 $\ge0$ 且当且仅当 $\beta=\beta_0$ 取 $0$。所以

$$
\hat\beta_{\rm ridge}=(X^\top X+\lambda I)^{-1}X^\top y\ \eqno{3.44}
$$

对应的**最小惩罚 RSS 值**（后面证明 (3.42) 那种「$t$ 球面」要用）是

$$
\mathrm{RSS}_{\rm ridge}(\lambda)=y^\top y-y^\top X(X^\top X+\lambda I)^{-1}X^\top y
$$

> **结果** · (3.44) 的 $p\times p$ 求逆看着吓人，但 (3.45)–(3.47) 给出了**不用求逆**的计算方式：只要一个 SVD。这是 3.5 最重要的实用结论——岭回归的实际代价是**一次 SVD**，之后任意 $\lambda$ 都是 $O(Np)$。

> **坑** · (3.44) 的推导**完全不需要 $X^\top X$ 可逆**（这是与第 3 章 (3.6) 的最大区别）。$p>N$ 时 $X^\top X$ 秩至多 $N<p$，$(X^\top X)^{-1}$ 不存在，但 $X^\top X+\lambda I$ 对任何 $\lambda>0$ 都正定，(3.44) 仍有唯一解。这就是 ridge 在高维里「还能算」的根本原因。
>
> 但 $X^\top X+\lambda I$ 的**条件数**会随 $\lambda$ 变好（这是正则化的数值意义）：$\kappa_2(X^\top X+\lambda I)=\dfrac{d_{\max}^2+\lambda}{d_{\min}^2+\lambda}$，$\lambda$ 大时趋于 $1$。

**SVD 形式 (3.45)–(3.47)**。设 $X$ 的「薄」SVD 为

$$
X=UDV^\top\ \eqno{3.45}
$$

其中 $U$ 是 $N\times p$ 列正交矩阵（$U^\top U=I_p$），$D=\mathrm{diag}(d_1,\dots,d_p)$，$d_1\ge\cdots\ge d_p\ge0$ 是奇异值，$V$ 是 $p\times p$ 正交矩阵（$V^\top V=I_p$）。$d_j$ 也等于 $X^\top X$ 的特征值的平方根。

先算最小二乘拟合值（回顾第 3 章 (3.46)，这里给出验证）：

$$
X\hat\beta_{\rm ls}=X(X^\top X)^{-1}X^\top y=UU^\top y\ \eqno{3.46}
$$

验证：$X^\top X=VD^2V^\top$，故 $(X^\top X)^{-1}=VD^{-2}V^\top$（在 $d_j>0$ 的方向上），代入得

$$
X(X^\top X)^{-1}X^\top y=UDV^\top V D^{-2}V^\top VD U^\top y=UU^\top y
$$

（用了 $V^\top V=I$，三处相消。$UU^\top$ 是到 $\mathrm{col}(X)$ 的正交投影矩阵，见预备知识 L2。）

再算 ridge。分母 $d_j^2+\lambda$ 全为正（即使 $d_j=0$），所以逆矩阵可算：

$$
\begin{aligned}
(X^\top X+\lambda I)^{-1}&=\big(VD^2V^\top+\lambda VV^\top\big)^{-1}=V(D^2+\lambda I)^{-1}V^\top\\
&=V\,\mathrm{diag}\Big(\frac{1}{d_j^2+\lambda}\Big)_{j=1}^{p}V^\top
\end{aligned}
$$

（$D^2+\lambda I$ 对角元为 $d_j^2+\lambda>0$，逆是对角的；$I=VV^\top$ 用了 $V$ 正交。）代入 (3.44)：

$$
\hat\beta_{\rm ridge}=V\,\mathrm{diag}\Big(\frac{d_j}{d_j^2+\lambda}\Big)_{j=1}^{p}\,U^\top y,\qquad\text{即}\qquad \hat\beta_{\rm ridge}=\sum_{j=1}^{p}\frac{d_j}{d_j^2+\lambda}\,(u_j^\top y)\,v_j
$$

（中间一步：$V^\top X^\top y=V^\top V D U^\top y=DU^\top y$。）

再左乘 $X=UDV^\top$，用 $DV^\top V\gamma=\mathrm{diag}(d_j)\gamma$ 得拟合值

$$
X\hat\beta_{\rm ridge}=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\,u_j u_j^\top y\ \eqno{3.47}
$$

> **结果** · (3.47) 的含义：最小二乘拟合值 (3.46) 是把 $y$ 投影到全部 $p$ 个方向；ridge 把第 $j$ 个方向**单独**按比例 $d_j^2/(d_j^2+\lambda)$ 缩放。方向的收缩量只依赖该方向的奇异值，与 $y$ 无关。
>
> $d_j\to\infty$（该方向信号极强）时因子 $\to1$：不收缩。$d_j\to0$（噪声方向）时因子 $\to0$：完全丢掉。所以 ridge 是**按方向信噪比自动加权**的。
>
> 注意两处收缩因子**不同**：拟合值里是 $\frac{d_j^2}{d_j^2+\lambda}$，$\beta$ 的 $V$-坐标里是 $\frac{d_j}{d_j^2+\lambda}$。若把 $\beta$ 写成 $V\gamma$、把最小二乘系数写成 $V\gamma^{\rm ls}$（$\gamma_j^{\rm ls}=u_j^\top y/d_j$，即第 3 章 §3.4 的 SVD 回归系数 $z_j$），则

$$
\gamma_j=\frac{d_j^2}{d_j^2+\lambda}\,\gamma_j^{\rm ls}=\frac{1}{1+\lambda/d_j^2}\,\gamma_j^{\rm ls}
$$

**两种写法统一了**：收缩因子都是 $1/(1+\lambda/d_j^2)$，只是 $\gamma^{\rm ls}$ 里已经含了一个 $d_j$。

**几何**。约束形式 (3.42) 把 $\ell_0\le t$ 换成 $\beta^\top\beta\le t^2$，即 $\lVert\beta\rVert_2\le\sqrt t$，是 $\mathbb{R}^p$ 里的**欧氏球**（$p\ge3$ 时是球面，$p=2$ 时是圆盘）。而 RSS 作为 $\beta$ 的二次函数，其等值面满足

$$
\mathrm{RSS}(\beta)=c\iff (y-X\beta)^\top(y-X\beta)=c\iff (\beta-\hat\beta_{\rm ls})^\top X^\top X(\beta-\hat\beta_{\rm ls})=\text{常数}
$$

因为 $\mathrm{RSS}(\beta)=\mathrm{RSS}(\hat\beta_{\rm ls})+(X(\beta-\hat\beta_{\rm ls}))^\top(X(\beta-\hat\beta_{\rm ls}))$（平方项展开后交叉项 $2(y-X\hat\beta_{\rm ls})^\top X(\beta-\hat\beta_{\rm ls})=0$，用第 3 章 (3.5) 的正规方程）。所以 RSS 等值面是**以 $\hat\beta_{\rm ls}$ 为中心、主轴方向为 $X^\top X$ 特征向量（即 $V$ 的列）、半轴长与 $1/d_j$ 成反比的椭球面**。

> **结果** · $\hat\beta_{\rm ridge}$ 就是**这个椭球面族与球面 $\|\beta\|_2\le\sqrt t$ 的切点**。切点条件「梯度平行于法向」正好就是 (3.44) 的正规方程：$\nabla\mathrm{RSS}=-2X^\top(y-X\beta)$ 与 $2\lambda\beta$ 平行。
>
> 因为球面光滑（处处可微、无角点），切点唯一。**这就是 ridge 解唯一、且每个系数都非零的几何原因**——球面上没有「角点」可以把解「压」到坐标轴上。lasso 的 $\ell_1$ 球（多面体）有角点，稀疏性就来自那里（见 §3.5.4）。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-5-2">原文 §3.5 岭回归</a>

### 3.5.3 有效自由度与主成分 {#s-3-5-3}

**问题**：「岭回归的有效自由度」是多少？为什么它能跟「用了 $M$ 个主成分」画等号？

> **基础知识** · 迹与迹循环律（见预备知识 L4）
>
> $\mathrm{tr}(AB)=\mathrm{tr}(BA)$（$A$ 是 $m\times n$、$B$ 是 $n\times m$）；$\mathrm{tr}(A)=\sum_i A_{ii}$。迹是线性函数：$\mathrm{tr}(A+B)=\mathrm{tr}(A)+\mathrm{tr}(B)$。

用 $V$ 的列作坐标重新写 (3.44)。$V^\top\hat\beta_{\rm ridge}$ 给出 $\beta$ 在正交基 $v_1,\dots,v_p$ 下的坐标：

$$
V^\top\hat\beta_{\rm ridge}=\mathrm{diag}\Big(\frac{d_j}{d_j^2+\lambda}\Big)U^\top y
$$

把 $X^\top X$ 的特征分解写出来更方便（这是 (3.48)）：

$$
X^\top X=VD^2V^\top\ \eqno{3.48}
$$

其中 $d_j^2$ 是 $X^\top X$ 的第 $j$ 个特征值，$v_j$ 是对应的单位特征向量。注意 (3.48) 就是 §3.4 里用过的谱分解，**没有新东西**。

定义第 $j$ 个回归系数的随机变量

$$
z_j=\frac{v_j^\top X^\top y}{v_j^\top X^\top Xv_j}=\frac{v_j^\top X^\top y}{d_j^2}\qquad (d_j>0)
$$

这是 $v_j$ 这个方向上的「单变量最小二乘系数」（第 3 章 (3.28) 的向量版）。它把 ridge 系数表成**方向**的形式：

$$
\hat\beta_{\rm ridge}=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\,z_j\,v_j\ \text{（$z_j$ 用 }\hat z_j\text{ 代入）}
$$

**为什么 $d_j^2\mathrm{Var}(z_j)$ 就是该方向解释的方差**。在模型 $y=X\beta+\varepsilon$、$\mathrm{Var}(\varepsilon)=\sigma^2I$ 下（预备知识 L5）：

$$
\mathrm{Var}(v_j^\top X^\top y)=\mathrm{Var}\big((Xv_j)^\top y\big)
$$

而 $Xv_j$ 是 $y$ 方向上的一个回归系数，记 $\gamma=v_j^\top\beta$，则 $v_j^\top X^\top y=\gamma\,v_j^\top X^\top Xv_j+v_j^\top X^\top\varepsilon=\gamma d_j^2+v_j^\top X^\top\varepsilon$。用 (3.8) 的推导（$\mathrm{Var}(X^\top\varepsilon)=\sigma^2X^\top X$，见预备知识 L5 的高斯线性组合）得

$$
\mathrm{Var}(v_j^\top X^\top y)=\sigma^2d_j^2,\qquad
\mathrm{Var}(z_j)=\frac{\sigma^2d_j^2}{d_j^4}=\frac{\sigma^2}{d_j^2}
$$

再由 $Xv_j$ 与 $z_j$ 的关系 $\mathrm{Var}(Xv_j)=\mathrm{Var}\big(\sum_j\beta_jXv_j\big)$ 在 $v_j$ 这个坐标上的分量 $=d_j^2\mathrm{Var}(z_j)$，即得 (3.49)：

$$
d_j^2\,\mathrm{Var}(z_j)=\mathrm{Var}(Xv_j)=\sigma^2\ \eqno{3.49}
$$

直观说法：**若 $z_j$ 已知，则 $Xv_j$ 的取值完全被 $\sigma^2$ 的噪声决定**；$d_j^2\mathrm{Var}(z_j)$ 衡量这个方向上「$X$ 空间位移」的方差，即该方向真正的信号强度。$d_j=0$ 的方向是 $X$ 的零空间，$Xv_j=0$，无信号。

**有效自由度 (3.50)**。岭回归的拟合值是 $X\hat\beta_{\rm ridge}$，写成矩阵形式

$$
X\hat\beta_{\rm ridge}=\underbrace{X(X^\top X+\lambda I)^{-1}X^\top}_{=:H_\lambda}y
$$

$H_\lambda$ 是 $N\times N$ 的**线性 smoother**（对称、不一定有秩 $N$）。用 (3.47) 与 $UU^\top y$ 得

$$
H_\lambda=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\,u_ju_j^\top
$$

**推导** $\mathrm{tr}(H_\lambda)$。用迹的定义逐项算（$\mathrm{tr}(u_ju_j^\top)=u_j^\top u_j=1$，因为 $u_j$ 是单位列向量）：

$$
\begin{aligned}
\mathrm{tr}(H_\lambda)
&=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\,\mathrm{tr}(u_ju_j^\top)\\
&=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\sum_{i=1}^{N}\sum_{k=1}^{N}(u_j)_i(u_j)_k\,\mathrm{tr}(e_ie_k^\top)\\
&=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\sum_{i=1}^{N}\sum_{k=1}^{N}(u_j)_i(u_j)_k\,\delta_{ik}\qquad(\mathrm{tr}(e_ie_k^\top)=(e_i^\top e_k)=\delta_{ik})\\
&=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\sum_{i=1}^{N}(u_j)_i^2\\
&=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\ \qquad(\text{因为 }u_j^\top u_j=1)
\end{aligned}
$$

所以

$$
\mathrm{df}(\hat y)=\mathrm{tr}(H_\lambda)=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}\ \eqno{3.50}
$$

> **结果** · (3.50) 是**主成分回归（PCR）的另一种写法**。因为 $H_\lambda$ 的第 $j$ 个特征方向上的特征值是 $d_j^2/(d_j^2+\lambda)\in(0,1)$，记

$$
M_\lambda=\#\Big\{j:\ \frac{d_j^2}{d_j^2+\lambda}>\frac12\Big\}=\#\{j:\ d_j^2>\lambda\}
$$

则 (3.50) 落在 $(M_\lambda,\ M_\lambda+1)$ 之间（每个保留项贡献接近 $1$，每个丢弃项贡献接近 $0$，而 $\frac{d_j^2}{d_j^2+\lambda}-1=-\frac{\lambda}{d_j^2+\lambda}$ 各项都 $<0$）。所以 **PCR 是 ridge 的「阶梯化」版本**：PCR 只取 $M$ 个方向、保留全部；ridge 取全部方向、逐个部分保留。

具体对应关系：设 (3.49) 给出第 $j$ 个方向的信号强度 $d_j^2\mathrm{Var}(z_j)$。若 $d_j^2>\lambda$，该方向在 ridge 里保留比例 $>1/2$，同时它的信号强度 $>\lambda$；若 $d_j^2<\lambda$，保留比例 $<1/2$ 且信号强度 $<\lambda$。**这正是「按信噪比阈值截断」的含义**：$\lambda$ 就是同时扮演「截断阈值」和「收缩强度」两个角色。

> **坑** · (3.50) 要求 $d_j^2+\lambda>0$，$\lambda>0$ 时自动成立。$\lambda=0$ 时 $d_j=0$ 的项是 $0/0$，必须理解为**丢弃**（这正是 §3.7 里用 $\mathrm{Var}(z_j)$ 排序后丢掉 $\mathrm{Var}(z_j)$ 最小的那几个的做法），此时 df 应为 $\mathrm{rank}(X)$ 而不是 $p$。
>
> 另一个坑：$d_j^2$ **不是**可以观测的量（$y$ 不出现在 $d_j$ 里，只出现在 $z_j$ 里），所以「$d_j^2>\lambda$」这个截断判据在**纯 ridge** 里没有实用意义——ridge 是**连续收缩**的，不做硬截断。只有当 $X$ 被标准化且列间近乎正交时，$d_j^2\approx$ 每列的 $N$ 倍，$\lambda$ 才近似有「保留几个变量」的直觉。ESL 后面反复提醒：**ridge 与 lasso 混用时不要用 ridge 的直觉去解释 lasso 的解**。
>
> 还有：$d_j^2$ 是**未标准化**列的量。若 $x_j$ 的单位是「元」而 $x_k$ 是「米」，$d_j^2$ 的比较毫无意义。所以 ridge **必须配合标准化**才可解释（lasso 同理，见 §3.5.4 的坑）。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-5">原文 §3.5 有效自由度</a>

### 3.5.4 Lasso：软阈值与稀疏性 {#s-3-5-4}

**问题**：把 $\ell_2$ 罚款换成 $\ell_1$，(3.41) 的解会变成什么？为什么它给出稀疏？

lasso 的**约束形式** (3.51) 与**惩罚形式** (3.52)：

$$
\hat\beta_{\rm lasso}=\arg\min_{\beta}\ \sum_{i=1}^{N}\Big(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\Big)^2\quad\text{s.t.}\ \sum_{j=1}^{p}|\beta_j|\le t\ \eqno{3.51}
$$

$$
\hat\beta_{\rm lasso}=\arg\min_{\beta}\ \frac{1}{2N}\sum_{i=1}^{N}\Big(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\Big)^2+\lambda\sum_{j=1}^{p}|\beta_j|\ \eqno{3.52}
$$

两式的等价性论证**与 §3.5.1 的 lasso 版本完全平行**，而且这次真的能做成凸对偶（因为 $\lVert\beta\rVert_1$ 是凸函数）：

1. 拉格朗日函数 $L(\beta,\lambda)=\mathrm{RSS}(\beta)+\lambda(\lVert\beta\rVert_1-t)$，$\lambda\ge0$。
2. 对 $\beta_j$ 求偏导，$j\ne0$ 时 $\frac{\partial|\beta_j|}{\partial\beta_j}=\mathrm{sign}(\beta_j)$，$j=0$ 时是 $[-1,1]$ 的区间（次梯度）。所以

$$
\frac{\partial L}{\partial\beta_j}=-2x_j^\top(y-X\beta)+\lambda\,\mathrm{sign}(\beta_j)=0,\qquad j=1,\dots,p
$$

3. 最优点在边界上（若 $\lVert\hat\beta\rVert_1<t$ 则 $\lambda=0$，解退化成 $\hat\beta_{\rm ls}$，它一般不在 $\lVert\beta\rVert_1\le t$ 内）。KKT 的互补松弛（预备知识 O2）给 $\lambda^\star>0$ 且 $\lVert\hat\beta^{\rm lasso}\rVert_1=t$。
4. Slater 条件满足（取 $\beta=0$，$t>0$ 时严格可行），目标凸，所以**强对偶**成立：$\max_{\lambda\ge0}g(\lambda)=\min_\beta\mathrm{RSS}(\beta)$ s.t. $\lVert\beta\rVert_1\le t$，其中 $g(\lambda)=\inf_\beta[\mathrm{RSS}(\beta)+\lambda\lVert\beta\rVert_1]$。这就给出 (3.51) 与 (3.52) 的一一对应。

> **结果** · (3.52) 的最优解是 $\lambda$ 的**分段线性函数**：$\hat\beta(\lambda)=\hat\beta(0)+\lambda\gamma$，在 $0\le\lambda\le\lambda_{\max}$ 上（这一段上支撑集固定）。$\lambda_{\max}$ 的显式值：

$$
\lambda_{\max}=\frac{1}{N}\max_{j=1,\dots,p}\big|x_j^\top(y-\bar y\mathbf 1)\big|
$$

证明：从 (3.52) 的一阶条件，若最优解是 $\beta=0$，需要（用次梯度的 $0\in\partial$）$\big|-2x_j^\top y+\lambda s_j\big|\le\lambda$ 对某个 $|s_j|\le1$ 成立，即 $2|x_j^\top y|\le\lambda(1+|s_j|)$ 恒真，也就是 $2|x_j^\top y|\le2\lambda$，取最大的那个 $j$ 即得 $\lambda\ge\max_j|x_j^\top y|$。归一化到 (3.52) 的 $1/(2N)$ 版本得 $\lambda_{\max}=\frac1N\max_j|x_j^\top y|$；去中心后 $x_j^\top y=x_j^\top(y-\bar y\mathbf 1)$。

**为什么 lasso 稀疏：几何论证**。

$\lVert\beta\rVert_1\le t$ 的可行集 $C_t=\{\beta:\sum_j|\beta_j|\le t\}$ 是一��� $p$ 维单纯形的多面体。它有 $2^{p}$ 个顶点，恰好是坐标轴上的点

$$
\pm te_j,\qquad j=1,\dots,p
$$

（可以直接验证：$\lVert\beta\rVert_1=\sum_j|\beta_j|\ge|\beta_j|$ 对每个 $j$ 成立，所以 $\lVert\beta\rVert_1\le t$ 蕴含 $|\beta_j|\le t$，等号只在 $\beta=\pm te_j$ 达到。）

RSS 的等值面仍是中心在 $\hat\beta_{\rm ls}$ 的椭球面（§3.5.2 证明过）。现在最小化 RSS 就是**把椭球面族从 $\hat\beta_{\rm ls}$ 处膨胀到第一次碰到 $C_t$**，接触点就是 $\hat\beta_{\rm lasso}$。

- 若 $p=2$，$C_t$ 是**菱形**。RSS 椭球面第一次接触菱形的位置通常是**角点或角附近的边**。角点对应「只有一个 $\beta_j\ne0$」——**这就是稀疏性的来源**。
- 一般 $p$ 下，接触点落在哪个 facet 上由哪几个变量的「相关」决定。facet 是 $\sum_{j\in\mathcal A}s_j\beta_j=t$（$\mathcal A\subseteq\{1,\dots,p\}$，$s_j=\pm1$）。若接触点落在 facet $\mathcal A$ 的**相对内部**（即 $\beta_j\ne0$ 对所有 $j\in\mathcal A$），则 KKT 给出

$$
-2x_j^\top(y-X\hat\beta)+\lambda s_j=0\quad(j\in\mathcal A)
$$

   解出 $\hat\beta_\mathcal{A}$ 后，若某个 $k\notin\mathcal A$ 的 $|x_k^\top(y-X\hat\beta)|>\lambda$，椭球还能再膨胀一点而不越界，说明还没到最紧，**矛盾**。所以不活跃集上的变量必须满足 (3.59) 的阈值条件（§3.6 推导）。$\lVert\beta\rVert_1$ 固定时，$\mathcal A$ 越小（$\beta_j$ 越集中）越容易满足这些阈值条件——这就是 lasso 系统性偏好的解形态。

> **结果** · 球面（$\ell_2$）光滑 ⇒ 解唯一、系数全非零；多面体（$\ell_1$）有角点 ⇒ 解可以落在角上 ⇒ 系数为 0。**平滑 vs 尖角是收缩与稀疏的唯一区别**。
>
> 更定量地说：$\ell_1$ 球约束下解满足 $\mathrm{sign}(\hat\beta_j)=\mathrm{sign}\big(x_j^\top(y-X\hat\beta)\big)$（相关性方向与系数方向一致），而 $\ell_2$ 球没有这个「符号锁定」。

**收缩算子**。把 §3.6 要用的软阈值算子先定义在这里（它是 lasso 的核心输出形式，见 (3.53)）。设 $\hat\beta_j^{\rm ls}$ 是第 $j$ 个变量的**单变量**最小二乘系数（其余变量不管），则 lasso 系数的（非线性）变换形式是

$$
S(\hat\beta_j^{\rm ls},\lambda)=\mathrm{sign}\big(\hat\beta_j^{\rm ls}\big)\max\Big(\big|\hat\beta_j^{\rm ls}\big|-\lambda,\ 0\Big)=\mathrm{sign}\big(\hat\beta_j^{\rm ls}\big)\big(\big|\hat\beta_j^{\rm ls}\big|-\lambda\big)_+
$$

其中 $(a)_+=\max(a,0)$。注意**这个公式只在单变量（或正交设计）情形严格成立**；在一般的共线情形下 lasso 解不是逐坐标的软阈值（§3.6 的坐标下降是迭代地把每个坐标「拉」到正确位置）。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-5">原文 §3.5 Lasso</a>

### 3.5.5 岭、lasso 与 elastic net 的对比 {#s-3-5-5}

**问题**：(3.53) 把两者的收缩规则并排放在一起。差别到底有多大？

> **结果** · 收缩规则 (3.53)

$$
\begin{cases}
\text{Ridge}:\ \dfrac{\hat\beta_j^{\rm ridge}}{\hat\beta_j^{\rm ls}}=\dfrac{1}{1+\lambda}\ \text{（近似，$\langle x_j,x_j\rangle=N$ 归一化下）}\\[8pt]
\text{Lasso}:\ \dfrac{\hat\beta_j^{\rm lasso}}{1}=\mathrm{sign}\big(\hat\beta_j^{\rm ls}\big)\Big(\big|\hat\beta_j^{\rm ls}\big|-\lambda\Big)_+\ \text{（仅单变量情形）}
\end{cases}\ \eqno{3.53}
$$

(3.53) 的严格版本（ESL 原文的形式）是在**标准化**设计矩阵（每列平方和为 $N$）下的近似。为看清差别，做一张表：

| | Ridge $\hat\beta_j/(1+\lambda)$ | Lasso $\mathrm{sign}(\hat\beta_j)(\lvert\hat\beta_j\rvert-\lambda)_+$ |
|---|---|---|
| $\lvert\hat\beta_j\rvert>\lambda$ | 乘 $1/(1+\lambda)$，**严格为正** | $\lvert\hat\beta_j\rvert-\lambda$，同号 |
| $\lvert\hat\beta_j\rvert\le\lambda$ | $\hat\beta_j/(1+\lambda)$，**严格为正** | 精确等于 $0$ |
| 是否非线性 | 线性（关于 $\hat\beta_j$） | 分段线性、有拐点，**非线性** |
| 收缩随 $\lvert\hat\beta_j\rvert$ 的变化 | 比例恒定 $1/(1+\lambda)$ | 比例 $1-\lambda/\lvert\hat\beta_j\rvert$ 随系数增大而增大 |

三条可验证的结论：

1. **Ridge 永远不产生 0**。若 $\hat\beta_j^{\rm ls}\ne0$，则 $\hat\beta_j^{\rm ridge}=\hat\beta_j^{\rm ls}/(1+\lambda)\ne0$（因为 $1+\lambda<\infty$）。要真正置零需要 $\lambda\to\infty$。
2. **Lasso 把小系数精确压成 0**。阈值 $|\hat\beta_j^{\rm ls}|\le\lambda$ 就是 (3.59) 的 KKT 条件在单变量情形的直接后果（那个坐标的最优值确实是 $0$ 当且仅当偏导的「梯度区间」包含 $0$）。
3. **相对收缩量 ridge 更大、lasso 更温和**。取 $\lvert\hat\beta_j^{\rm ls}\rvert$ 很大时，lasso 的收缩比例 $\to1$（几乎不收缩），ridge 的比例恒为 $1/(1+\lambda)$；反之在小系数处 lasso 收缩到 0、ridge 保留一个小的正数。所以 **lasso 是「选择性收缩」，ridge 是「均匀收缩」**。ESL 的经验法则：**预测误差相同时 lasso 系数更稀疏**；**$p$ 很小且信噪比高时 ridge 的 MSE 更低**（因为 lasso 硬阈值带来的「偏差」在信号明确时是纯粹的损失）。

> **坑** · (3.53) 的两个公式**都不是**一般情况下 lasso/ridge 精确解的逐坐标表达。只有在**正交设计**（$X^\top X=N\cdot I$，即列两两正交且等长）时才严格成立；共线时 ridge 的收缩因子逐方向是 $d_j^2/(d_j^2+\lambda)$（不是 $1/(1+\lambda)$），lasso 需要迭代（§3.6）。ESL 把这个表放在正文里是为了给**直觉**，不要当定理用。
>
> 另一个坑：$\mathrm{sign}(0)$ 的定义。$S(u,\lambda)$ 在 $u=0$ 处 $S(0,\lambda)=0$，一致；但 $\mathrm{sign}(0)=0$ 会让 $\lambda\,\mathrm{sign}(\beta_j)$ 在 $\beta_j=0$ 时等于 $0$，而 KKT 需要的其实是 $-\lambda\le x_j^\top(y-X\beta)\le\lambda$（见 (3.59)）。这是初学者最容易写错的地方。

**Elastic net (3.54)**。既然 ridge 稳（唯一解、连续）而 lasso 稀疏（但可能不稳），把两个罚款加起来：

$$
\hat\beta_{\rm EN}=\arg\min_{\beta}\ \frac{1}{2N}\sum_{i=1}^{N}\Big(y_i-\beta_0-\sum_{j=1}^{p}\beta_jx_{ij}\Big)^2+\lambda\sum_{j=1}^{p}\Big[\alpha\beta_j^2+(1-\alpha)|\beta_j|\Big]\ \eqno{3.54}
$$

（ESL 把 $\alpha\beta_j^2+(1-\alpha)|\beta_j|$ 称为 elastic net 罚，$\alpha\in[0,1]$：$\alpha=1$ 退化为 ridge，$\alpha=0$ 退化为 lasso，$\alpha\in(0,1)$ 兼顾两者。）

> **推导** · 弹性网的唯一性。目标函数的可微部分（忽略 $\lVert\cdot\rVert_1$ 的次梯度）的 Hession 是

$$
\nabla^2\Big[\tfrac12 y^\top y-y^\top X\beta+\beta^\top X^\top X\beta+N\lambda\alpha\beta^\top\beta\Big]=2\big(X^\top X+N\lambda\alpha I\big)
$$

对任何 $\lambda\alpha>0$，这个矩阵**正定**（$X^\top X$ 半正定，$N\lambda\alpha I$ 正定，和必正定，见预备知识 L1）。加上凸函数 $\lambda(1-\alpha)\lVert\beta\rVert_1$ 后，总目标是**严格凸**的（严格凸的证明：任意 $\beta\ne\beta'$，取中点，二次部分严格下降、$\ell_1$ 部分不增，故中点目标严格小于两端平均）⇒ **极小点唯一**。

> **结果** · $p>N$（甚至 $N<p$ 秩亏）时：$\alpha>0$ ⇒ (3.54) 有唯一解；$\alpha=0$（纯 lasso）⇒ 解**可能不唯一**。显式反例：$N=1$，$y=1$，$X=(1,1)$（$p=2$），$\lambda=1$。目标为

$$
(1-\beta_1-\beta_2)^2+|\beta_1|+|\beta_2|
$$

   固定 $s=\beta_1+\beta_2$ 时 $\lvert\beta_1\rvert+\lvert\beta_2\rvert\ge\lvert s\rvert$，等号当且仅当 $\beta_1,\beta_2$ 同号。$s\in[0,1]$ 上目标为 $1-s+s^2$，在 $s=1/2$ 取 $3/4$；$s<0$ 上目标为 $1-s-s^2+s=1-s^2<1$ 在 $s\to0^-$… 更仔细地，$s\le0$ 时 $(1-s)^2-s=1-3s+s^2>1$。所以最小值 $3/4$ 在 $(\beta_1,\beta_2)=(1/2,0),(0,1/2),(1/4,1/4)$ 等无穷多组上取到 ⇒ **不唯一**。
>
> 这就是 elastic net 存在的第二个理由：$\alpha>0$ 的 $\ell_2$ 项提供了**唯一性**。$Zou \& Hastie (2005)$ 的命名也来自这个「net（网）」——把 lasso 的 $\ell_1$ 尖角「兜」上一层光滑的 $\ell_2$ 球。

> **坑** · elastic net 的 $\lambda$ 与 $\alpha$ **不是各自独立可解释的**。整体罚款强度是 $\sqrt{(1-\alpha)}\lambda$（见第 3 章 (3.88) 的 $\lambda^{(1-\alpha)}=\sqrt{1-\alpha}\,\lambda$ 那段讨论），只有这个组合量决定解在路径上的位置。ESL 后面明确说：不要用「lasso 的 $\lambda$」或「ridge 的 $\lambda$」的直觉去读 (3.54) 的 $\lambda$。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-5">原文 §3.5 Elastic net</a>

## 3.6 坐标下降 {#s-3-6}

§3.5 给出了 lasso 的定义 (3.52)，但没给算法——(3.52) 是个含 $\lVert\beta\rVert_1$ 的不可微凸问题，没有闭式解。本节补上三件事：**实际怎么算**（坐标下降，(3.83)(3.84)）、**解长什么样**（KKT 条件 (3.58)(3.59)）、**一条更快的算法路线**（LARS，(3.55)(3.56)），最后回答「岭回归和 lasso 的有效自由度能不能用同一套公式算」（(3.60)）。对应原文 <a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-6">原文 §3.6</a>。

### 3.6.1 坐标下降 {#s-3-6-1}

**问题**：一个具体的迭代格式，一次只更新一个坐标，几次之后收敛。

> **基础知识** · 坐标下降（见预备知识 O4）
>
> 对可分目标 $F(\beta)=\sum_{j=1}^pF_j(\beta_j)$，固定其余坐标只优化第 $j$ 个：$\beta_j\leftarrow\arg\min_tF_j(t)$。因为 $F(\beta)\le F(\beta^{\pm})$，每次迭代单调不增目标值。

ESL 采用的目标函数 (3.57)（注意这个版本不带 $1/N$，$\lambda$ 的定义与 (3.52) 差一个 $N$ 倍因子，见下面的坑）：

$$
R(\beta)=\frac{1}{2}\lVert y-X\beta\rVert_2^2+\lambda\lVert\beta\rVert_1\ \eqno{3.57}
$$

**第一步：写「残差修正量」**。更新 $\beta_j$ 时，只有 $\beta_jx_j$ 这一项变化。定义

$$
\tilde y^{(j)}=y-\sum_{k\ne j}\hat\beta_kx_k\ \in\ \mathbb{R}^N\ \text{，逐元素写为}\ \tilde y_i^{(j)}=y_i-\sum_{k\ne j}\hat\beta_kx_{ik}
$$

（注意 $\tilde y^{(j)}$ 里的 $\hat\beta_k$（$k\ne j$）是**上一次迭代**的值，$j$ 坐标被清零。用 (3.2) 的记号，这等价于 $\tilde y_i^{(j)}=y_i-\beta_0-\sum_{k\ne j}\hat\beta_kx_{ik}$，截距被放进了 $\beta_0$。）

于是

$$
R(\hat\beta^{\pm})=\frac{1}{2}\sum_{i=1}^{N}\big(\tilde y_i^{(j)}-x_{ij}\beta_j\big)^2+\lambda\sum_{k\ne j}|\hat\beta_k|+\lambda|\beta_j|
$$

固定其余坐标后，$\sum_{k\ne j}|\hat\beta_k|$ 是常数（**可分性**），只需极小化

$$
g_j(t)=\frac{1}{2}\sum_{i=1}^{N}\big(\tilde y_i^{(j)}-x_{ij}t\big)^2+\lambda|t|
$$

**第二步：对 $t$ 求导，分三种情形**。$t\ne0$ 时 $\frac{\mathrm d|t|}{\mathrm dt}=\mathrm{sign}(t)$，

$$
g_j'(t)=-x_j^\top\tilde y^{(j)}+t\,\lVert x_j\rVert_2^2+\lambda\,\mathrm{sign}(t)
$$

（$g_j''=\lVert x_j\rVert_2^2>0$，因为 $x_j$ 不为零向量，所以 $g_j$ 严格凸，最小点唯一。）

- **$t>0$**：$g_j'(t)=0$ 解出

$$
t=\frac{x_j^\top\tilde y^{(j)}-\lambda}{\lVert x_j\rVert_2^2}
$$

  这个值 $>0$ 的条件是 $x_j^\top\tilde y^{(j)}>\lambda$。

- **$t<0$**：$\mathrm{sign}(t)=-1$，

$$
t=\frac{x_j^\top\tilde y^{(j)}+\lambda}{\lVert x_j\rVert_2^2}
$$

  为负的条件是 $x_j^\top\tilde y^{(j)}<-\lambda$。

- **$t=0$**：$g_j$ 在 $0$ 处不可导，$0$ 是极小点当且仅当 $0$ 属于次梯度，即

$$
-x_j^\top\tilde y^{(j)}\in\lambda[-1,1]\iff \big|x_j^\top\tilde y^{(j)}\big|\le\lambda
$$

  这正是 (3.59) 的不等式在 $N=1$ 归一化下的样子。

**第三步：合并成软阈值算子**。设 $a=\frac{x_j^\top\tilde y^{(j)}}{\lVert x_j\rVert_2^2}$（即 $x_j$ 方向上的单变量最小二乘系数），则

$$
t^\star=\begin{cases}
a-\dfrac{\lambda}{\lVert x_j\rVert_2^2}, & a>\dfrac{\lambda}{\lVert x_j\rVert_2^2}\\[6pt]
0, & \lvert a\rvert\le\dfrac{\lambda}{\lVert x_j\rVert_2^2}\\[6pt]
a+\dfrac{\lambda}{\lVert x_j\rVert_2^2}, & a<-\dfrac{\lambda}{\lVert x_j\rVert_2^2}
\end{cases}=\mathrm{sign}(a)\Big(\lvert a\rvert-\frac{\lambda}{\lVert x_j\rVert_2^2}\Big)_+
$$

用 $\frac{1}{N}\sum$ 的记号（假设 $\lVert x_j\rVert_2^2=N$，即列已标准化）写成 ESL 的形式 (3.83)(3.84)：

$$
\tilde y_i^{(j)}=y_i-\sum_{k\ne j}\hat\beta_kx_{ik}\ \eqno{3.83}
$$

$$
\hat\beta_j\leftarrow S\Big(\frac{1}{N}\sum_{i=1}^{N}\tilde y_i^{(j)}x_{ij},\ \lambda\Big)\ \eqno{3.84}
$$


> **坑** · 原书 (3.84) 里**没有** $\lambda$ 的减项，写的就是 $S(a,\lambda)$。常见的错误推论是「软阈值算子有平移性质 $S(u-\lambda,\lambda)=S(u,\lambda)-\lambda$，所以两者等价」——**这个恒等式不成立**：
>
> $$S(1.5\lambda,\lambda)-\lambda=(1.5\lambda-\lambda)-\lambda=-0.5\lambda\ \ne\ S(0.5\lambda,\lambda)=0.5\lambda$$
>
> 真正成立的是**正齐次性** $S(cu,c\lambda)=c\,S(u,\lambda)\ (c>0)$，以及只在 $u\ge0$ 一侧成立的$S(u+\lambda,\lambda)=S(u,\lambda)+\lambda$。至于 $\frac12\lVert y-X\beta\rVert^2$ 与 $\frac1{2N}\sum_i(\cdot)^2$ 之间的归一化差异，是通过**阈值本身**被缩放来吸收的（目标里的 $\lambda/N$ 对应这里阈值 $\lambda$），不是在结果上再减一个 $\lambda$。
>
> 若不确定，直接把 $u=0$ 代进去检验：$S(0,\lambda)-\lambda=-\lambda\ne0$，而正确解在 $x_j^\top r=0$ 时必须给出 $\beta_j=0$。


其中软阈值算子 $S$ 与 (3.53) 里的写法等价：ESL 常用**去中心**的版本

$$
S(a,\lambda)=\begin{cases}
a-\lambda,& a>\lambda\\
0,& -\lambda\le a\le\lambda\\
a+\lambda,& a<-\lambda
\end{cases}\qquad\text{（这是「$a\mapsto a-\lambda\,\mathrm{sign}(a)$ 对 $|a|>\lambda$，否则 $0$」）}
$$


$$
S\big(a-\lambda,\ \lambda\big)=S\big(a,\ \lambda\big)-\lambda\qquad\text{（对一切 }a,\lambda>0\text{ 成立，逐段代入上面三个分支即可）}
$$

> **数值** · 一次迭代的代价（N2）：更新 $\beta_j$ 需要 $\sum_i x_{ij}^2$（$O(N)$，预处理后是查表）和 $\sum_i\tilde y_i^{(j)}x_{ij}$（$O(N)$）。所以**单坐标更新是 $O(N)$**，一轮扫过 $p$ 个坐标是 $O(Np)$，与一次矩阵–向量乘同阶。这比任何需要构造 $X^\top X$（$O(Np^2)$）的方法都便宜。ESL 强调这一点：lasso 的实际计算量是 $O(Np)$ per sweep。

> **坑** · 三条，都容易踩：
>
> 1. **标准化必须做**。若不标准化，$\lVert x_j\rVert_2^2$ 各不相同，阈值变成 $\lambda/\lVert x_j\rVert_2^2$，$\ell_1$ 罚的「等权」假设被破坏，解不再是「每个变量同等对待」。ESL 的所有例子都假定 $x_j$ 已中心化、标准化到 $\frac{1}{N}\sum_i x_{ij}^2=1$。
> 2. **截距必须从罚里排除**。$\lVert\beta_1\rVert_1$ 里不能含 $\beta_0$（截距），否则 (3.57) 会惩罚截距。实现上 (3.83) 里的 $\hat\beta_0$ 是单独处理的。
> 3. **收敛是线性的但很慢，且不保证唯一解**。坐标下降单调不增 $R$，凸性保证收敛到全局极小（预备知识 O4）；但若解不唯一（§3.5.5 的反例），停在哪个解取决于初始化。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-6">原文 §3.6 坐标下降</a>

### 3.6.2 Lasso 解的 KKT 条件 {#s-3-6-2}

**问题**：解 $\hat\beta$ 满足什么方程？这些方程能用来做**证书**（检查一个候选解是否为全局最优），也能解释 LARS。

把 (3.57) 写开（$y$ 已中心化、$X$ 已标准化，$\lambda>0$）：

$$
R(\beta)=\frac{1}{2}\sum_{i=1}^{N}\big(y_i-\sum_{j=1}^{p}\beta_jx_{ij}\big)^2+\lambda\sum_{j=1}^{p}|\beta_j|
$$

> **推导** · 第一阶条件。设 $\mathcal{B}=\{j:\ \hat\beta_j\ne0\}$ 为**活跃集**（active set，系数非零的变量），$\mathcal{A}=\{1,\dots,p\}\setminus\mathcal{B}$ 为零集。
>
> **（a）$\beta_j\ne0$ 时**（$j\in\mathcal{B}$）：$\lvert\beta_j\rvert$ 在 $j$ 附近可微，$\frac{\partial|\beta_j|}{\partial\beta_j}=\mathrm{sign}(\beta_j)$。用链式法则与平方项求导 $\frac{\partial}{\partial\beta_j}\frac12\sum_i(y_i-\sum_k\beta_kx_{ik})^2=-\sum_i x_{ij}(y_i-\sum_k\hat\beta_kx_{ik})=-x_j^\top(y-X\hat\beta)$，得

$$
x_j^\top(y-X\hat\beta)=\lambda\,\mathrm{sign}(\hat\beta_j),\qquad\forall j\in\mathcal{B}\ \eqno{3.58}
$$

> **（b）$\beta_j=0$ 时**（$j\in\mathcal{A}$）：$|\beta_j|$ 不可导，一阶条件要改成**次梯度**形式 $\mathbf 0\in\partial R$。$\partial|\beta_j|\big|_{\beta_j=0}=[-1,1]$，所以要求存在 $s_j\in[-1,1]$ 使

$$
-x_j^\top(y-X\hat\beta)+\lambda s_j=0
$$

> 因为 $\hat\beta_j=0$ 时 $\mathrm{sign}(\hat\beta_j)$ 无定义（若约定 $\mathrm{sign}(0)=0$ 则会错），KKT 的正确写法是 (3.59) 的**不等式**。两边不等号的来源：$-x_j^\top(y-X\hat\beta)=-\lambda s_j$ 且 $|s_j|\le1$ 给出 $\big|x_j^\top(y-X\hat\beta)\big|\le\lambda$：

$$
\big|x_k^\top(y-X\hat\beta)\big|\le\lambda,\qquad\forall k\notin\mathcal{B}\ \eqno{3.59}
$$

**与坐标下降不动点是同一组方程**。这是本节最值得记住的观察：

- (3.84) 的一步不动点条件：$\beta_j=\beta_j^{\text{new}}$。由 §3.6.1 的第三步，$\beta_j=0$ 当且仅当 $\big|\frac1N x_j^\top r\big|\le\lambda$；$\beta_j\ne0$ 时 $\frac1N x_j^\top r=\beta_j+\lambda\,\mathrm{sign}(\beta_j)$。因为 $r=y-X\hat\beta+\text{（只影响 }j\text{ 的项）}$，在不动点上 $r=y-X\hat\beta$，**乘掉 $1/N$ 的归一化**后恰好是 (3.58)(3.59)。
- 更进一步：坐标下降每扫一轮，若 $R$ 严格下降，则每一步的一阶条件都**至少在一个坐标上失效**；坐标下降终止 ⟺ 所有坐标都满足一阶条件 ⟺ KKT ⟺ 全局最优（因为 $R$ 凸，预备知识 O2）。

> **结果** · (3.58)(3.59) 是 lasso 的**最优性证书**。给定 $y,X,\lambda$，可以：① 用坐标下降求 $\hat\beta$；② 计算 $s_k=\frac{x_k^\top(y-X\hat\beta)}{\lambda}$，检查是否所有 $k\notin\mathcal{B}$ 都有 $|s_k|\le1$、所有 $j\in\mathcal{B}$ 都有 $s_j=\mathrm{sign}(\hat\beta_j)$。若全部满足，$\hat\beta$ **必是全局最优**（凸性 + KKT 充分必要）。这比「目标值下降很小就停」严格得多，也是很多 lasso 软件库（`glmnet` 等）实际使用的收敛判据。

> **坑** · 两条：
>
> 1. (3.58) 里 $\mathcal{B}$ 是**活跃集（$\hat\beta_j\ne0$）**，不是零集。ESL 的记号在 §3.6 里容易读反——(3.58) 的 `∀j∈B` 里 `sign(β_j)` 才有意义，$\beta_j=0$ 时 `sign` 无定义。所以 $\mathcal{B}$ 必须是非零集。若把 $\mathcal{B}$ 读成零集，(3.58) 退化成 $x_j^\top(y-X\hat\beta)=0$（$j\in$ 零集），那是**最小二乘**的正规方程，不是 lasso 的。
> 2. KKT 只在**凸**问题上充分。$R$ 确实是凸的（平方项凸 + $\lVert\cdot\rVert_1$ 凸，见预备知识 C3 关于 $\|x\|^1$ 凸性的讨论），所以 (3.58)(3.59) **充分且必要**。但这不含「唯一性」——$X^\top X$ 秩亏时解不唯一，KKT 仍全满足（§3.5.5 的反例）。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-6">原文 §3.6 KKT 条件</a>

### 3.6.3 LARS 与 lasso 路径 {#s-3-6-3}

**问题**：坐标下降一次 $O(Np)$，要几百次扫。LARS（Least Angle Regression）沿整条路径走，每个 $k$ 只需一次 $O(\lvert\mathcal{A}_k\rvert^3)$ 的小矩阵求解。

设 $\mathcal{A}_k$ 是第 $k$ 步的活跃集（ESL 这里用 $\mathcal{A}$，注意与上一小节 $\mathcal{B}$ 指同一个东西，**同一概念两个字母**，别被绕晕），$X_{\mathcal{A}_k}$ 是 $X$ 取 $\mathcal{A}_k$ 列的子矩阵（$N\times\lvert\mathcal{A}_k\rvert$），$X^\top_{\mathcal{A}_k}$ 是它的转置。当前拟合 $\hat y_k=X_{\mathcal{A}_k}\hat\beta_{\mathcal{A}_k}$，残差

$$
r_k=y-\hat y_k=y-X_{\mathcal{A}_k}\hat\beta_{\mathcal{A}_k}
$$

**LARS 的每一步（3.55）**：**单个最小二乘步**

$$
\delta_k=\big(X^\top_{\mathcal{A}_k}X_{\mathcal{A}_k}\big)^{-1}X^\top_{\mathcal{A}_k}r_k\ \eqno{3.55}
$$

> **推导** · 为什么 $\delta_k$ 就是「保持当前活跃集时的最优改进方向」。若我们只允许 $\mathcal{A}_k$ 里的系数变动（其余固定为 0），最小二乘残差平方是

$$
\mathrm{RSS}(\hat\beta_{\mathcal{A}_k}+\delta)=\lVert r_k-X_{\mathcal{A}_k}\delta\rVert_2^2=r_k^\top r_k-2\delta^\top X^\top_{\mathcal{A}_k}r_k+\delta^\top X^\top_{\mathcal{A}_k}X_{\mathcal{A}_k}\delta
$$

> 对 $\delta$ 求梯度置零：$X^\top_{\mathcal{A}_k}X_{\mathcal{A}_k}\delta=X^\top_{\mathcal{A}_k}r_k$。矩阵 $X^\top_{\mathcal{A}_k}X_{\mathcal{A}_k}$ 是 $\lvert\mathcal{A}_k\rvert\times\lvert\mathcal{A}_k\rvert$ 的 Gram 矩阵；因 $X_{\mathcal{A}_k}$ 列满秩（活跃集的定义）故正定（预备知识 L1），逆存在，解即 (3.55)。

然后 $\hat y_{k+1}=X_{\mathcal{A}_k}(\hat\beta_{\mathcal{A}_k}+\gamma_k\delta_k)$，$\gamma_k>0$ 是标量步长，$\hat\beta_{k+1}=\hat\beta_k+\gamma_k\delta_k$。**$\gamma_k$ 选多大？** 一直走到「某个新变量与当前残差的相关性等于当前活跃集内所有相关性的最小绝对值」为止——这就是**等相关（equicorrelation）**条件 (3.56)：

$$
x_j^\top(y-X\hat\beta)=\gamma\cdot s_j,\qquad\forall j\in\mathcal{A}_k\ \eqno{3.56}
$$

其中 $s_j=\mathrm{sign}(\hat\beta_j)$ 是每个活跃变量系数的符号，$\gamma\ge0$ 是当前残差与所有活跃预测量的**公共内积大小**。若 $|x_j^\top r|$ 的最大值落在 $j\notin\mathcal{A}_k$ 上，就把 $j$ 加入活跃集（$\mathcal{A}\leftarrow\mathcal{A}\cup\{j\}$）；若活跃集内某变量的 $\mathrm{sign}$ 要翻转，就把它踢出（对应 lasso 路径上的**变量退出**点）。

> **结果** · (3.56) **正是 lasso 的 KKT 条件 (3.58) 限制在活跃集上**。逐一核对：
>
> - (3.58) 说：对 $j\in\mathcal{B}$（= 活跃集），$x_j^\top(y-X\hat\beta)=\lambda\,\mathrm{sign}(\hat\beta_j)$。
> - (3.56) 说：对 $j\in\mathcal{A}_k$（= 同一个集合），$x_j^\top(y-X\hat\beta)=\gamma\,s_j$，其中 $s_j=\mathrm{sign}(\hat\beta_j)$。
>
> 两式形式完全一致，只差一个标量 $\gamma$ 替代 $\lambda$。**LARS 沿路径走出的折线，与 lasso 的解路径重合**。所以 LARS 是 lasso 的一个高效求解器（$p$ 中等时 $O(Np^2)$ 总代价，远少于坐标下降的数百次 $O(Np)$ 扫描 + 收敛判据的开销）。

> **坑** · LARS 与 lasso 路径只在**「一般位置」**上重合。当 $\lvert x_j^\top r\rvert$ 的最大值同时在多个变量（包括已在 $\mathcal{A}$ 里的）上取到时出现**结（knot）**，LARS 一次加/删一组变量。lasso 路径在结处通常也有折角。ESL 说明：标准 LARS 实现处理一般位置的简单结，**「lasso 的 LARS」（lasso modification / 修正 LARS）在结处需要额外处理**才能严格等于 lasso 的 argmin。这是个实打实的实现坑。
>
> 另：LARS 依赖 $X^\top_{\mathcal{A}}X_{\mathcal{A}}$ 可逆（活跃集列满秩），所以 **$\mathcal{A}$ 的大小不超过 $\mathrm{rank}(X)\le\min(N,p)$**——LARS 天然不会让解有超过 $\min(N,p)$ 个非零。这与 §3.5.5 中「$p>N$ 时 lasso 不唯一」呼应：LARS 输出的是其中一个特定解。

**$L_1$ arc-length 参数化**。LARS 走的是**分段线性路径**（每步 $\hat\beta\leftarrow\hat\beta+\gamma\delta$），但用 $\gamma$ 作横轴不好解释。ESL 用 **$L_1$ 弧长**作横轴：

$$
\text{横轴}\ =\ \sum_{j=1}^{p}\big|\hat\beta_j(\lambda)\big|\ \text{（$L_1$ arc-length）}
$$

即 lasso 路径按 $\|\hat\beta\|_1$ 归一化。因为 $\hat\beta(\lambda)$ 在 $\lambda\in[0,\lambda_{\max}]$ 上分段线性且在 $\lambda=0$ 处连续（$\hat\beta(0)=\hat\beta_{\rm ls}$，$\lVert\hat\beta_{\rm ls}\rVert_1$ 有限），$\lVert\hat\beta(\lambda)\rVert_1$ 也是 $\lambda$ 的分段线性**递减**函数（对 lasso，$\lambda$ 增大时每个非零系数向 0 移动），且**严格递减**直到 $\lambda_{\max}$ 时归零。所以 $\lVert\hat\beta\rVert_1$ 是 $\lambda$ 的一一对应单调函数，**可以反过来用 $\lVert\hat\beta\rVert_1$ 当自变量**，横轴「fraction of $L_1$ arc-length」就是 $\lVert\hat\beta(\lambda)\rVert_1/\lVert\hat\beta_{\rm ls}\rVert_1\in[0,1]$。

这样画的图有两个好处：① 纵轴 $\hat\beta_j$ 对横轴分段线性，lasso 路径和 LARS 都是折线，容易读出「哪个变量在哪个弧长处进出」；② 横轴在 $[0,1]$ 里，各变量尺度可比，不同 $p$ 的模型能放一起比（如与前向逐步回归的对比）。ESL 的图 3.11 就是这个形式，也是第 3 章末尾比较 lasso / ridge / elastic net / 逐步选择 / 前向逐步的主要工具。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-6">原文 §3.6 LARS</a>

### 3.6.4 有效自由度 (3.60) {#s-3-6-4}

**问题**：§3.5.3 用迹 $\mathrm{tr}(H_\lambda)$ 定义了岭回归的有效自由度。lasso、PCR、LARS 没有矩阵 $H$，但「有效自由度」照样要能算。怎么统一？

答案是用 (3.60)：

$$
\mathrm{df}(\hat y)=\frac{1}{\sigma^2}\sum_{i=1}^{N}\mathrm{Cov}\big(\hat y_i,\ y_i\big)\ \eqno{3.60}
$$

原文对这个 $\mathrm{Cov}$ 有一句关键的限定：**「$\mathrm{Cov}(\hat y_i,y_i)$ 指预测值 $\hat y_i$ 与对应观测值 $y_i$ 之间的**抽样**协方差」**。这一句话就是整个推导的钥匙——它意味着我们**条件于 $X$**（把 $X\beta$ 当作常数），于是 $y$ 的全部随机性都来自噪声。

> **基础知识** · 条件协方差与线性平滑器
>
> 记条件于 $X$ 的协方差为 $\mathrm{Cov}(\cdot\mid X)$。在高斯线性模型 $y=X\beta+\varepsilon$、$\varepsilon\sim N(0,\sigma^2I_N)$、$X$ 视为固定（预备知识 P4）下，
>
> $$\mathrm{Cov}(y\mid X)=\mathrm{Cov}(\varepsilon\mid X)=\sigma^2I_N$$
>
> 对任何**对称**的线性平滑器 $\hat y=Hy$（$H$ 对称即可，不必是对称幂等的），矩阵协方差公式给出
>
> $$\mathrm{Cov}(\hat y,y\mid X)=H\,\mathrm{Cov}(y\mid X)\,H^\top=\sigma^2HH^\top=\sigma^2H$$

> **推导** · **第 1 步：条件于 $X$，$y$ 的协方差就是 $\sigma^2I$**。
>
> 条件于 $X$ 时 $\beta$ 与 $\varepsilon$ 都是常数，只有观测的随机性来自 $\varepsilon$，故上式的 $\mathrm{Cov}(y\mid X)=\sigma^2I_N$（预备知识 L5 的线性变换律，取 $a=I$）。

> **第 2 步：把 $H$ 代进去**。$\hat y=Hy$ 是 $y$ 的线性变换，而 $\mathrm{Cov}(Ay,B)=A\,\mathrm{Cov}(y)\,B^\top$（可直接展开：$E[((Ay-EAy)(By-EBy)^\top)]$，展开后非零的只有 $E[(y-Ey)(y-Ey)^\top]$）。所以

$$
\mathrm{Cov}(\hat y,y\mid X)=H\,\mathrm{Cov}(y\mid X)\,H^\top=\sigma^2HH^\top=\sigma^2H
$$

> 最后一步只用了 $H$ 对称。

> **第 3 步：取对角线**。$\big[\mathrm{Cov}(\hat y,y\mid X)\big]_{ii}=\mathrm{Cov}(\hat y_i,y_i\mid X)=\sigma^2H_{ii}$，逐项就是 $\sigma^2$ 乘以帽子矩阵的第 $i$ 个对角元。求和：

$$
\sum_{i=1}^{N}\mathrm{Cov}\big(\hat y_i,\ y_i\big)=\sigma^2\sum_{i=1}^{N}H_{ii}=\sigma^2\,\mathrm{tr}(H)
$$

> **第 4 步：代回 (3.60)**，就得到

$$
\mathrm{df}(\hat y)=\frac{1}{\sigma^2}\cdot\sigma^2\,\mathrm{tr}(H)=\mathrm{tr}(H)
$$

> **这个推导的三个要点**：① 条件于 $X$，否则信号部分会混进来（见「坑」第 1 条）；② 只要求 $H$ **对称**，不要求对称幂等——所以它对岭回归、PCR、以及任何「$\hat y=Hy$ 型」的拟合都成立；③ 不要求 $HX=X$，所以 ridge 的收缩不会被破坏（$E\hat y=HX\beta$ 只是被 $\mathrm{Cov}$ 减掉的东西，不出现在结果里）。

> **结果** · (3.60) 与 §3.5.3 的迹公式完全一致，但它有两个实用优势：
>
> - **只需要 $\hat y$ 和 $y$**，不需要 $\hat\beta$、不需要矩阵 $H$、不需要 $\lambda$。所以任何方法只要能给出拟合值向量 $\hat y$，就能算有效自由度——这也是第 7 章讨论交叉验证与模型复杂度时的通用武器。
> - **数值验证**：对固定 $k$ 个预测子的线性回归，$H=P_{\mathcal A}$（$\mathcal A$ 是这 $k$ 个列张成的子空间），$\mathrm{tr}(H)=k$，即「预先指定的 $k$ 个变量恰好用掉 $k$ 个自由度」；而如果这 $k$ 个变量是用最优子集**从数据里选出来**的，$k$ 个参数背后实际用掉的自由度大于 $k$——(3.60) 正好量化了「多用掉多少」。
> - **与 Mallows $C_p$ 的关系**：$\mathrm{CMSE}\approx\frac{\sigma^2}{N}\big[\mathrm{df}(\hat y)+2\frac{(N-p)\hat\sigma^2}{\sigma^2}\big]$ 一类形式里，df 正是被 (3.60) 提供的量。

> **坑** · 三条：
>
> 1. **必须条件于 $X$（固定设计）。** 如果把 $\beta$ 也当成随机量，$\mathrm{Cov}(y)=\sigma^2I+X\beta\beta^\top X^\top$，第 2 步就变成 $H(X\beta\beta^\top X^\top+\sigma^2I)H^\top$，逐项出现信号项 $\big(\beta^\top x_i\big)\big(x_i^\top H\beta\big)$，求和不再等于 $\sigma^2\mathrm{tr}(H)$。所以 (3.60) 是**固定设计下**的定义；随机设计下要写成条件期望 $\mathbb{E}_X[\ldots]$。
> 2. **$\sigma^2$ 未知时**要用 $\hat\sigma^2$ 代替，于是 df 也带估计误差，(3.60) 变成近似。实践中取 $\hat\sigma^2=\mathrm{RSS}/(N-p)$。
> 3. **要求 $\hat y$ 是 $y$ 的线性函数**（$\hat y=Hy$，$H$ 对称）。对 lasso/LARS 这类非线性方法，(3.60) 不是恒等式而是一个**定义**：它把「$\hat y$ 跟 $y$ 有多紧地一起动」折成一个标量。好处是它对任意预测器都有定义；代价是它在非线性方法下不再等于任何显式的秩。要理解这点，请对比本节 §3.5.3 的 $\mathrm{tr}(H_\lambda)$：那里 $H$ 存在且对称幂等，两种定义恰好重合。

> **延伸** · lasso / LARS 路径上的 df。LARS 与 lasso 的解 $\hat y(\lambda)$ 在相邻 $\lambda$ 之间是**线性**的（系数路径分段线性，见 (3.90)），于是 $\hat y$ 对 $\lambda$ 的斜率 $g_m=\hat y(\lambda_{m+1})-\hat y(\lambda_m)$ 是常数。整条路径上的 $\mathrm{df}$ 也是分段线性的：在一个段内，$\hat y(\lambda)=\hat y_m-(\lambda-\lambda_m)g_m$，把 (3.60) 的 $\hat y_i$ 换成这个表达式即可得到该段上 df 的闭式，然后逐段线性插值。这是 lasso 软件报告「有效自由度」的标准做法。$\mathrm{df}$ 随 $\lambda$ 单调递增（拟合越紧 df 越大），这与 §3.6.2 中「松弛越少、约束越紧」的直觉一致。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-6">原文 §3.6 有效自由度</a>

（完）
