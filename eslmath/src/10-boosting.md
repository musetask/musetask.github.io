---
id: m10
n: "10"
title: 提升法与加性树
title_en: Boosting and Additive Trees
desc: AdaBoost 的加权误差与 margin 推导、阶段式加性建模与经验风险最小化、随机梯度提升、随机森林与提升的比较、多分类提升
prev: m09
next: m11
prev_title: 第 9 章 加性模型、树与相关方法
next_title: 第 11 章 神经网络
---

# 10 提升法与加性树 {#s-10}

第 8 章的 bagging 把很多个「弱」模型平均起来，靠的是降方差；第 9 章的树把特征空间切成矩形，靠的是在每个矩形里拟合常数。提升法（boosting）站在这两个思想的交叉点上：它把很多个弱分类器**加**起来而不是平均起来，而且加的系数 $\alpha_m$ 不是拍脑袋定的，而是由一个明确的极小化问题算出来的。全章的主线只有一句话——**提升法就是用阶段式（stagewise）的方式拟合一个加性模型，而这个加性模型的损失函数恰好是指数损失**。把这个等式说清楚之后，AdaBoost 的权重重分配、$\alpha_m$ 的闭式解、以及后来所有变体（随机梯度提升、Huber 提升、多分类提升）都变成同一套「泛函的极小分解」。

本章的公式编号与《ESL》第二版一致，共 58 个编号公式（10.1）–（10.58）。核心推导集中在三处：(10.9)–(10.15) 把 AdaBoost.M1 还原成指数损失下的阶段式加性建模；(10.16)–(10.19) 给出指数损失的总体极小元；(10.33)–(10.38) 说明梯度提升为什么可以用数值优化的语言重述提升。

> **本文与第一版的一个结构差异**
>
> 第一版 ESL 在 §10.2 有「AdaBoost 的误差分析」一节，给出 margin 界 $I(y_if\le0)$ 的上界与 $\gamma_m$ 的下界。**第二版删掉了这一节**，因此本手册按第二版的编号体系组织，这些公式**不给编号**。为了让 margin 这个概念不被丢掉，10.2 下面仍保留一节 `margin 与误差界（延伸）`，把它重新推一遍（用的是 Chebyshev 不等式，见预备知识 P1）。原书 §10.7–10.8 的内容（离货架算法、垃圾邮件例）并入 10.4 与 10.5。

---

## 10.1 提升法简介 {#s-10-1}

### 10.1.1 弱分类器与加权多数表决 {#s-10-1-1}

两分类问题，$Y\in\{-1,1\}$，分类器 $G(x)\in\{-1,1\}$。定义训练误差率与总体误差率

$$
\mathrm{err}_m=\frac{\sum_{i=1}^{N}w_i\,I(y_i\neq G_m(x_i))}{\sum_{i=1}^{N}w_i},\qquad
R(g)=E_{X,Y}\,I\big(Y\neq g(X)\big)
$$

**弱分类器**指错分率只比随机猜测（50%）好一点点的分类器，例如只有两个叶节点的分类树（decision stump）。

> **基础知识** · 为什么「多数票」原则上会越做越好
>
> 若 $M$ 个分类器**独立**地以概率 $p>1/2$ 正确，「多数票」的正确率是二项和
>
> $$\sum_{k>M/2}\binom{M}{k}p^k(1-p)^{M-k}\ \xrightarrow[\ \text{CLT}\ ]{\ \Phi\big(\sqrt M\,(2p-1)\big)\ }$$
>
> $p=\frac12+\epsilon$ 时，随着 $M\to\infty$ 这个和趋于 1（预备知识 P1）。**但 (10.1) 里各成员并不独立，而且 $\alpha_m$ 刻意偏向「本轮加权错分率低」的成员**，所以这个论证只是启发。真正的保证来自 10.3 节的指数损失推导。

提升法的做法是：反复把同一个弱学习算法作用在**被修改过的数据**上，得到一串 $G_1,G_2,\dots,G_M$，再用加权多数表决合成

$$
G(x)=\mathrm{sign}\Big\{\sum_{m=1}^{M}\alpha_m\,G_m(x)\Big\} \eqno{10.1}
$$

权重的初值是 $w_i=1/N$（等价于均匀采样），第 $m$ 轮之后把被 $G_m$ 分错的观测的权重按 $e^{\alpha_m}$ 放大，分对的按 $e^{-\alpha_m}$ 缩小，于是下一轮的弱学习器被迫去照顾前面漏掉的难样本。**提升的全部技巧就是这一句话**。

### 10.1.2 一个可以手算的例子 {#s-10-1-2}

原书用下面的确定性分类问题展示「弱学习器也能被提升」<a class="src" href="../esl/ch10-boosting-and-additive-trees.html#s-10-1">原文 §10.1</a>：

$$
Y=\begin{cases}1,&\displaystyle\sum_{j=1}^{10}X_j^2>\chi^2_{10}(0.5)\\ -1,&\text{否则}\end{cases} \eqno{10.2}
$$

其中 $\chi^2_{10}(0.5)=9.34$ 是 10 个标准正态平方和的中位数（10 个自由度的 $\chi^2$ 分布）。也就是说 $Y$ 由「到原点距离的平方」与一个阈值比较决定，**判别面是一个球面**。用 stump 作基学习器：单个 stump 的测试错分率 45.8%（几乎等于随机猜测的 50%），但 400 轮提升后降到 5.8%，而且优于一棵 244 个叶节点的大树（24.7%）。

> **数值** · 为什么 $Y$ 均匀地落在球内球外
>
> 对标准正态向量 $X\in\mathbb R^{10}$，$S=\sum_{j=1}^{10}X_j^2$ 的中位数是 9.34（中位数 $\approx$ 自由度 $=10$，差 0.66 是中位数的偏差）。因为 $S$ 的分布是 $\chi^2_{10}$，中位数略小于均值 10，所以 $P(S>9.34)\approx0.5$，两个类几乎各占一半，基学习器信息量为零。这个例子把「提升能造出非线性判别面」这件事展示得干净利落。

---

## 10.2 AdaBoost：加权误差 {#s-10-2}

AdaBoost.M1（Algorithm 10.1）的四行：初始化 $w_i=1/N$；第 $m$ 轮用权重 $w_i$ 拟合 $G_m$；算加权误差 $\mathrm{err}_m$ 与 $\alpha_m$；更新权重。写成公式：

$$
\alpha_m=\log\frac{1-\mathrm{err}_m}{\mathrm{err}_m},\qquad
\mathrm{err}_m=\frac{\sum_{i=1}^{N}w_i^{(m)}\,I\big(y_i\neq G_m(x_i)\big)}{\sum_{i=1}^{N}w_i^{(m)}} \eqno{10.13}
$$

> **坑** · $\mathrm{err}_m$ 必须用**当前轮**的权重 $w^{(m)}$，而且权重只归一化到「和为任意常数」即可——因为 $G_m$ 是「最小化加权错分」的，它只对权重成比例的平移不敏感。若 $\mathrm{err}_m\ge1/2$，$\alpha_m\le0$，表决机制失效，这也是「弱学习器必须比随机好一点」这条假设的实际来源。

### margin 与误差界（延伸）{#s-10-2-margin}

第一版 ESL 在这里有一节专门分析 margin；第二版删掉了，本手册按第二版的编号体系组织，所以这一小节**不给公式编号**。但它是理解提升稳健性最重要的一条推导，单独推一遍。

margin 是 $y_if(x_i)$，其中 $f(x)=\sum_{m=1}^M\alpha_mG_m(x)$ 是 (10.1) 里 $\mathrm{sign}$ 之前的实值函数。

**第一步：margin 是部分和。** 对固定的 $i$，

$$
y_if(x_i)=y_i\sum_{m=1}^{M}\alpha_mG_m(x_i)=\sum_{m=1}^{M}\alpha_m\underbrace{y_iG_m(x_i)}_{\in\,\{-1,+1\}}
$$

**第二步：加权意义下的 $\gamma_m$。** 在第 $m$ 轮，把训练点按概率 $p_i^{(m)}=w_i^{(m)}/\sum_jw_j^{(m)}$ 独立重抽样，记抽到的点为 $I\sim p^{(m)}$，并令 $Z_m:=y_IG_m(x_I)\in\{-1,+1\}$。**验证** $E[Z_m]$：

$$
E[Z_m]=\frac{\sum_iw_i^{(m)}y_iG_m(x_i)}{\sum_iw_i^{(m)}}=\frac{\sum_iw_i^{(m)}\big(1-2I(y_i\neq G_m(x_i))\big)}{\sum_iw_i^{(m)}}=1-2\mathrm{err}_m
$$

（用了 $y_iG_m(x_i)=1-2I(y_i\neq G_m(x_i))$。）定义

$$
\gamma_m=\frac12-\mathrm{err}_m\quad\Longrightarrow\quad E[Z_m]=2\gamma_m
$$

方差也立刻得到（预备知识 P1：$\mathrm{Var}(Z)=E[Z^2]-E[Z]^2=1-(2\gamma_m)^2$）：

$$
\mathrm{Var}(Z_m)=1-(1-2\mathrm{err}_m)^2=4\mathrm{err}_m(1-\mathrm{err}_m)=4\Big(\tfrac12-\gamma_m\Big)\Big(\tfrac12+\gamma_m\Big)=1-4\gamma_m^2
$$

> **坑** · $\mathrm{err}_m\le1/2$ 就是「$\gamma_m\ge0$」
>
> 弱学习器假设的真正含义是 $\mathrm{err}_m<1/2$，等价于 $\gamma_m>0$。$\gamma_m$ 是「$G_m$ 相对随机猜测的优势」，越大这一轮贡献越大。

**第三步：一个指示函数不等式。** 对一切实数 $z$，$I(z\le0)\le(1-z)^2$（$z\le0$ 时 $(1-z)^2\ge1$；$z>0$ 时左边为 0）。常数 1 是紧的（$z\to0^-$ 时）。

> **坑** · 常被写错的 $\frac14$ 版本
>
> 不少文献（以及一些二手笔记）把上式写成 $I(z\le0)\le\frac14(1-z)^2$。这是**错的**：取 $z=-0.5$，左边 1，右边 $\frac14\cdot2.25=0.5625<1$。$\frac14$ 的形式只在 $|y_if|$ 已经被额外约束（例如已证明所有 margin 的绝对值 $\ge1$）时才成立，不能当作一般不等式使用。要得到 $1/\gamma^2$ 的量级，靠的是 Chebyshev 而不是这个不等式。

**第四步：Chebyshev 给出 $O(1/\gamma^2)$。** 抽 $M$ 次独立样本（$Z_m^{(1)},\dots,Z_m^{(M)}$ 独立同分布），取平均 $M_m=\frac1M\sum_{k=1}^{M}Z_m^{(k)}$。独立性给出

$$
E[M_m]=2\gamma_m,\qquad \mathrm{Var}(M_m)=\frac{\mathrm{Var}(Z_m)}{M}=\frac{1-4\gamma_m^2}{M}
$$

由单边 Chebyshev 不等式 $\Pr(M_m-E[M_m]\le-t)\le\mathrm{Var}/t^2$（预备知识 P1 的集中不等式），取 $t=2\gamma_m$：

$$
\Pr\big(M_m\le0\big)\le\frac{\mathrm{Var}(M_m)}{(2\gamma_m)^2}=\frac{1-4\gamma_m^2}{4M\gamma_m^2}\le\frac{1}{4M\gamma_m^2}
$$

代回 $\mathrm{err}_m=\frac12-\gamma_m$ 逐步展开验证 $\mathrm{Var}(M_m)$：

$$
\mathrm{Var}(Z_m)=4\Big(\frac12-\gamma_m\Big)\Big(\frac12+\gamma_m\Big)=4\Big(\frac14-\gamma_m^2\Big)=1-4\gamma_m^2
$$

**第五步：换成单边 Cantelli 不等式可把界收紧。** Chebyshev 是双边界，对单边事件偏松。Cantelli 给出

$$
\Pr\big(M_m-E[M_m]\le-t\big)\le\frac{\mathrm{Var}(M_m)}{\mathrm{Var}(M_m)+t^2}=\frac{(1-4\gamma_m^2)/M}{(1-4\gamma_m^2)/M+4\gamma_m^2}
$$

在 $\gamma_m$ 小、$M$ 大时分母 $\approx4\gamma_m^2$，与 Chebyshev 的 $\frac{1-4\gamma_m^2}{4M\gamma_m^2}$ 同阶，但当 $\gamma_m\to1/2$（$G_m$ 很准）时 Cantelli 明显更好。

> **坑** · Markov 在这里给不出信息
>
> 有人会想用 Markov：$1+M_m\ge0$，故 $\Pr(M_m\le0)=\Pr(1+M_m\le1)\le E[1+M_m]=1+2\gamma_m>1$。**这个界恒大于 1，等于没说**。原因是 Markov 只用均值，而这里的均值本身就在阈值上方，$E[1+M_m]=1+2\gamma_m$ 告诉我们「平均偏向不错分」，却无法量化「有多大比例错分」。要用分布的离散程度，就必须靠方差——这就是 Chebyshev/Cantelli 的角色。

**结论**：错分事件是「平均 margin 掉到 0 以下」；平均的涨落按 $1/\sqrt M$ 缩小，而均值是常数 $2\gamma_m$。所以要让错分概率小到 $O(\varepsilon)$，需要 $M\asymp1/(\varepsilon\gamma^2)$。这就是「$\gamma_m$ 每轮只需比 $1/2$ 好一点点，总错分率就能降到任意小」的定量含义。

> **结果** · margin 分析的三条定量结论
>
> - **平均 margin 递增**：把 $f$ 拆成部分和（第一步），再对每一项用第二、三步的结论：
>
> $$E\big[y_if(x)\big]=\sum_{m=1}^{M}\alpha_m\,E\big[y_iG_m(x)\big]=\sum_{m=1}^{M}\alpha_m\cdot 2\gamma_m>0\quad\text{且随 }M\text{ 递增}$$
>
> 注意 $E[y_iG_m(x)]$ 要在**第 $m$ 轮自己的权重**下算，这正是 (10.9) 里 $w_i^{(m)}=\exp(-y_if_{m-1}(x_i))$ 的含义。
>
> - **训练错分率非凸**：因为 margin 分析只控制期望而不控制样本路径上的每一步，实际曲线会在若干轮后变非凸（Figure 10.3 的两条曲线正是这一现象：训练错分率在 250 轮左右降到 0 后不再变，而指数损失仍在下降）。
> - **指数损失的敏感性**：$e^{-yf}$ 对大负 margin 的惩罚是指数的，所以 AdaBoost 在训练后期把几乎全部权重压在少数几个「顽固」样本上——这既是它能持续降低损失的原因，也是它在标签噪声下退化的原因（对照 10.3.4 节里 deviance 对错分类只线性惩罚）。

<a class="src" href="../esl/ch10-boosting-and-additive-trees.html#s-10-2">原文 §10.2</a>

### 10.2.1 阶段式加性建模与经验风险最小化 {#s-10-2-1}

<a class="src" href="../esl/ch10-boosting-and-additive-trees.html#s-10-3">原文 §10.3</a>

提升法的核心是 (10.1) 中的**加法**结构：把分类器换成一般的基函数，加性展开写成

$$
f(x)=\sum_{m=1}^{M}\beta_m\,b(x;\gamma_m) \eqno{10.3}
$$

其中 $b(x;\gamma)\in\mathbb R$ 是由参数 $\gamma$ 刻画的简单函数，$\beta_m$ 是展开系数。神经网络的 $b(x;\gamma)=\sigma(\gamma_0+\gamma_1^\top x)$、小波的平移与尺度、MARS 的截断幂样条基，都是这个形式。

**问题**：怎样求 $\beta_m,\gamma_m$？最自然的想法是直接最小化训练集上的平均损失

$$
f̂=\underset{f}{\mathrm{argmin}}\ \sum_{i=1}^{N}L\big(y_i,f(x_i)\big) \eqno{10.5}
$$

如果 $f$ 无限制地取遍所有函数，(10.5) 的解是 $f(x_i)=y_i$，显然过拟合。真正的做法是加约束：$f$ 必须是 (10.3) 的形式。于是分两步走。

**第一步：整体极小化问题。** 在 (10.3) 上直接最小化 (10.5)：

$$
\hat f=\underset{\beta_1,\dots,\beta_M,\,\gamma_1,\dots,\gamma_M}{\mathrm{argmin}}\ \sum_{i=1}^{N}L\big(y_i,\textstyle\sum_{m=1}^{M}\beta_mb(x_i;\gamma_m)\big)
$$

这是 $M$ 个 $(\beta_m,\gamma_m)$ 的联合非线性优化，$\gamma$ 一动整个展开就变，通常无闭式解。

**第二步：单基函数的子问题。** 保持已加入的 $\beta_1b(\cdot;\gamma_1),\dots,\beta_{m-1}b(\cdot;\gamma_{m-1})$ **冻结**，只优化新的一项：

$$
(\hat\beta_m,\hat\gamma_m)=\underset{\beta,\gamma}{\mathrm{argmin}}\ \sum_{i=1}^{N}L\big(y_i,\ \beta\,b(x_i;\gamma)\big) \eqno{10.4}
$$

**阶段式加性建模**（Algorithm 10.2）就是反复解 (10.4)：$f_0(x)\equiv0$，第 $m$ 步解一次 (10.4) 得到 $(\beta_m,\gamma_m)$，然后

$$
f_m(x)=f_{m-1}(x)+\beta_m b(x;\gamma_m)
$$

**为什么这近似 (10.5)**：记 (10.5) 的目标为 $J(f)=\sum_iL(y_i,f(x_i))$。**阶段式单调性**来自「新解不劣于退化解」：取 $\beta_m=0$ 是 (10.4) 的可行点（此时新项不改变 $f_{m-1}$），而 $\hat\beta_m$ 是最优解，故

$$
J\big(\hat f_m\big)=\sum_{i=1}^{N}L\big(y_i,\hat f_{m-1}(x_i)+\hat\beta_mb(x_i;\hat\gamma_m)\big)\ \le\ \sum_{i=1}^{N}L\big(y_i,\hat f_{m-1}(x_i)\big)=J\big(\hat f_{m-1}\big)
$$

于是 $J(\hat f_M)\le\cdots\le J(\hat f_0)$，训练风险**单调不增**。

**但这只是「不劣于退化」**：(10.4) 的解空间比 (10.5) 严格小——旧项 $\beta_1,\dots,\beta_{m-1}$ 被冻结，不能重新调整。所以 $\{f_m\}$ 一般**不等于** (10.5) 的全局最优解。这就是贪心近似的全部含义，也是为什么需要 (10.41) 的收缩与早停来补偿。

平方误差损失本身是

$$
L\big(y,f(x)\big)=\big(y-f(x)\big)^2 \eqno{10.6}
$$

**它的阶段式子问题**（$L(y_i,f_{m-1}(x_i)+\beta b(x_i;\gamma))$）展开为

$$
\big(y_i-f_{m-1}(x_i)-\beta b(x_i;\gamma)\big)^2=\big(r_{im}-\beta b(x_i;\gamma)\big)^2,\qquad r_{im}=y_i-f_{m-1}(x_i) \eqno{10.7}
$$

> 对 $\beta$ 求偏导并令零：$-\sum_i b(x_i;\gamma)(r_{im}-\beta b(x_i;\gamma))=0$，得
>
> $$\beta(\gamma)=\frac{\sum_{i=1}^{N}b(x_i;\gamma)r_{im}}{\sum_{i=1}^{N}b(x_i;\gamma)^2}$$
>
> 换成矩阵记号：$\beta(\gamma)=(b_\gamma^\top b_\gamma)^{-1}b_\gamma^\top r_m$，其中 $b_\gamma$ 是 $N$ 维向量 $(b(x_1;\gamma),\dots,b(x_N;\gamma))^\top$。所以平方误差下的 (10.4) 就是**用当前残差拟合一个新基函数**——「最小二乘提升」的全部内容。

**这段推导依赖的假设**：(a) 目标对 $\beta$ 光滑；(b) $\sum_ib(x_i;\gamma)^2>0$（基函数不正交于零）；(c) 每步只降训练风险，**不保证降未来风险**——后者要靠第 10.4 节的正则化（收缩）。

---

## 10.3 指数损失与 AdaBoost 的等价 {#s-10-3}

### 10.3.1 指数损失 {#s-10-3-1}

AdaBoost.M1 用的损失函数是

$$
L(y,f(x))=\exp\big(-y\,f(x)\big) \eqno{10.8}
$$

代入阶段式加性建模的子问题 (10.4)，基函数取 $G(x)\in\{-1,1\}$：

$$
(\beta_m,G_m)=\underset{\beta,\,G}{\mathrm{argmin}}\ \sum_{i=1}^{N}w_i^{(m)}\exp\big[-\beta\,y_iG(x_i)\big] \eqno{10.9}
$$

$$
w_i^{(m)}=\exp\big(-y_i f_{m-1}(x_i)\big)
$$

**关键观察**：$w_i^{(m)}$ 只依赖 $f_{m-1}$，与待求的 $(\beta,G)$ **无关**，所以它就是第 $i$ 个观测的一个固定权重。于是 (10.9) 变成「**带权重的**指数损失极小化」——这正是 AdaBoost 里「用权重 $w_i$ 拟合分类器」那一步的含义。

### 10.3.2 两步求解：先定 G 再定 beta {#s-10-3-2}

**第一步：固定 $\beta>0$，求 $G$。** 用 $-y_iG(x_i)=2I(y_i\neq G(x_i))-1$ 把判别项和权重项分开：

$$
\sum_{i=1}^{N}w_ie^{-\beta y_iG(x_i)}=\sum_{i=1}^{N}w_ie^{-\beta}\,I(y_i=G(x_i))+\sum_{i=1}^{N}w_ie^{\beta}\,I(y_i\neq G(x_i))
$$

两边同除 $e^{-\beta}>0$，判别 $G$ 的目标与 $\beta$ 无关：

$$
G_m=\underset{G}{\mathrm{argmin}}\ \sum_{i=1}^{N}w_i^{(m)}\,I\big(y_i\neq G(x_i)\big) \eqno{10.10}
$$

也就是**最小化加权错分率**。加权错分率等于括号内两个和按权重加权，因此 $G_m$ 就是这一轮 AdaBoost 用权重 $w^{(m)}$ 训练出来的分类器。

把上式整理成「$e^{-\beta}$ 乘上 $\beta$ 的无关项」的形式，就是原书的等价写法

$$
\sum_{i=1}^{N}w_ie^{-\beta y_iG(x_i)}=e^{-\beta}\sum_{i=1}^{N}w_i+\big(e^{\beta}-e^{-\beta}\big)\sum_{i=1}^{N}w_i\,I\big(y_i\neq G(x_i)\big) \eqno{10.11}
$$

> **推导** · 核对 (10.11)
>
> 对 $y_i=G(x_i)$：右边 $=e^{-\beta}w_i+(e^\beta-e^{-\beta})\cdot0=w_ie^{-\beta}$，左边 $=w_ie^{-\beta}$ ✓。
>
> 对 $y_i\neq G(x_i)$：右边 $=e^{-\beta}w_i+(e^\beta-e^{-\beta})w_i=w_ie^{\beta}$，左边 $=w_ie^{\beta}$ ✓。
>
> 所以 (10.11) 与前一行是同一个二次型恒等式，(10.11) 的好处是把「随 $\beta$ 变化的部分」全部集中到第二项，于是可以直接对 $\beta$ 求导。

**第二步：把 $G_m$ 代回，对 $\beta$ 求极小。** 对 (10.11) 求导：

$$
\frac{d}{d\beta}\sum_iw_ie^{-\beta y_iG_m(x_i)}=-e^{-\beta}\sum_{i=1}^{N}w_i+\big(e^{\beta}+e^{-\beta}\big)\sum_{i=1}^{N}w_iI\big(y_i\neq G_m(x_i)\big)=0
$$

记 $\Sigma_c=\sum_iw_iI(y_i=G_m(x_i))$、$\Sigma_w=\sum_iw_iI(y_i\neq G_m(x_i))$，两者之和为 $W=\sum_iw_i$。方程变成 $-e^{-\beta}W+(e^\beta+e^{-\beta})\Sigma_w=0$。整理（把 $-\Sigma_w$ 从括号里拿出来）：

$$
-e^{-\beta}(\Sigma_c+\Sigma_w)+e^{-\beta}\Sigma_w+e^{\beta}\Sigma_w=-e^{-\beta}\Sigma_c+e^{\beta}\Sigma_w=0
$$

这一步没有假设，只是展开（$-e^{-\beta}\Sigma_w$ 与 $+e^{-\beta}\Sigma_w$ 抵消）。两边同除 $e^{-\beta}>0$：

$$
e^{2\beta_m}\Sigma_w=\Sigma_c
\quad\Longrightarrow\quad
e^{2\beta_m}=\frac{\Sigma_c}{\Sigma_w}=\frac{W-\Sigma_w}{\Sigma_w}=\frac{1-\mathrm{err}_m}{\mathrm{err}_m}
$$

取对数即得

$$
\hat\beta_m=\frac12\log\frac{1-\mathrm{err}_m}{\mathrm{err}_m} \eqno{10.12}
$$

> **坑** · 中间量不是 $\tanh$
>
> 直觉上容易写成「$\tanh\beta_m=\mathrm{err}_m$」，这是**错的**。实际上
>
> $$\tanh\beta_m=\frac{e^{2\beta_m}-1}{e^{2\beta_m}+1}=\frac{(1-\mathrm{err}_m)-\mathrm{err}_m}{(1-\mathrm{err}_m)+\mathrm{err}_m}=1-2\mathrm{err}_m$$
>
> 正是 $E[y_iG_m(x_i)]$ 本身（与 10.2.1 节的 $E[Z_m]$ 同一个量）。$\mathrm{err}_m=\frac12$ 时 $\tanh\beta_m=0$ 且 $\beta_m=0$，两者都对；但 $\mathrm{err}_m=0.3$ 时 $\beta_m=\frac12\log\frac{0.7}{0.3}=0.4233$，而 $\tanh\beta_m=0.4\ne0.3$。

这就是 Algorithm 10.1 的 $\alpha_m=2\beta_m$。**注意 $\alpha_m$ 只依赖 $\mathrm{err}_m$，而 $\mathrm{err}_m$ 只依赖 $G_m$**——所以 $\beta_m$ 与 $G_m$ 可以分开求，这两步分离是整个算法可实现的原因。

**第三步：更新 $f$ 与权重。** 把 $(\beta_m,G_m)$ 加进展开：

$$
f_m(x)=f_{m-1}(x)+\beta_mG_m(x)
$$

下一轮的权重定义是 $w_i^{(m+1)}=\exp(-y_if_m(x_i))$，代入上式：

$$
w_i^{(m+1)}=w_i^{(m)}\exp\big(-\beta_my_iG_m(x_i)\big) \eqno{10.14}
$$

再用 $-y_iG_m(x_i)=2I(y_i\neq G_m(x_i))-1$：

$$
w_i^{(m+1)}=w_i^{(m)}\cdot e^{\alpha_m I(y_i\neq G_m(x_i))}\cdot e^{-\beta_m} \eqno{10.15}
$$

> **推导** · 从 (10.14) 到 (10.15)
>
> 记 $I=I(y_i\ne G_m(x_i))$，则
>
> $$e^{-\beta_my_iG_m(x_i)}=e^{-\beta_m(2I-1)}=e^{-\beta_m}\cdot e^{2\beta_mI}$$
>
> 而 $2\beta_m=\alpha_m$。
>
> 因为 $\alpha_m=2\beta_m$ 且 $\sum_iw_i^{(m+1)}$ 与 $\sum_iw_i^{(m)}$ 只差一个公共因子 $e^{-\beta_m}$，而 $G_{m+1}$ 对权重成比例平移不敏感（见 (10.10) 的推导），所以 (10.15) **与 Algorithm 10.1 第 2(d) 行等价**。

> **结果** · 等价定理
>
> **AdaBoost.M1 = 指数损失下的阶段式加性建模。** 具体地，Algorithm 10.1 第 2(a) 行是「近似求解 (10.11)，从而近似求解 (10.9)」；第 2(b)(c) 行精确给出 (10.13)(10.12)；第 2(d) 行精确给出 (10.15)。
>
> **依赖的假设**：(a) 基分类器 $G_m$ 取遍全部 $\{-1,1\}$ 值函数（否则 (10.10) 只是近似）；(b) $\mathrm{err}_m<1/2$（否则 $\alpha_m\le0$）；(c) $\beta_m>0$（否则第一步「固定 $\beta>0$」不成立）。
>
> 这一等价是 Freund–Schapire 算法比 bagging 强得多的根本原因：bagging 优化的是方差，AdaBoost 优化的是**指数损失**，一个对 margin 敏感的准则。

### 10.3.3 指数损失的总体极小元 {#s-10-3-3}

指数损失好不好，要看它在**总体分布**下的极小元是什么。设 $p(x)=\Pr(Y=1\mid x)$，$Y\in\{-1,1\}$，则

$$
f^{\star}(x)=\underset{f}{\mathrm{argmin}}\ E\big[-Yf(x)\big]=\log\frac{\Pr(Y=-1\mid x)}{\Pr(Y=1\mid x)} \eqno{10.16}
$$

> **推导** · 分情形，先看无约束版本
>
> **情形 A（条件期望）**：$E[-Yf(x)\mid X=x]=-f(x)\big[p(x)-(1-p(x))\big]=f(x)\big[2p(x)-1\big]$。系数 $2p(x)-1$ 只依赖 $x$，不依赖 $f$，所以在 $p(x)\ne1/2$ 处这个目标关于 $f$ 无极小（推到 $\mp\infty$）。**因此 $E[-Yf]$ 不是可用的准则**——(10.16) 的正确读法是：$f^\star$ 同时最小化条件期望 $E[-Yf\mid x]$（即无约束下取下确界）与指数损失。
>
> **情形 B（指数损失，我们真正在意的）**：条件风险是
>
> $$E\big[e^{-Yf(X)}\big]=p(x)e^{-f(x)}+(1-p(x))e^{f(x)}$$
>
> 求导并令零（C1）：$-p(x)e^{-f(x)}+(1-p(x))e^{f(x)}=0$，即 $e^{2f(x)}=p(x)/(1-p(x))$，取对数得 $f^\star(x)=\frac12\log\frac{p(x)}{1-p(x)}$。二阶导 $p e^{-f}+(1-p)e^{f}>0$，故这是唯一的极小点。
>
> 由全期望公式（预备知识 P2），对 $f$ 的**逐点**极小化与对**总体**极小化可交换：$f$ 只能取形如 $f(x)=\eta(x)$ 的函数（$f$ 由 $x$ 决定），所以
>
> $$\underset{f}{\mathrm{argmin}}\ E_{X,Y}\big[e^{-Yf(X)}\big]\ \Longleftrightarrow\ \text{对几乎每个 }x,\ f(x)=f^\star(x)$$
>
> 唯一性：若总体极小元存在，每个 $x$ 上的条件风险都必须取到自己的最小值（否则改动那个 $x$ 的取值就能改进），而条件风险的极小元唯一。**$f^\star$ 是 log-odds 的一半**——所以取 $\mathrm{sign}(f)$ 分类是合理的（与 (10.1) 一致）。
>
> **与情形 A 的一致性**：代入 $f^\star=\frac12\log\frac{p}{1-p}$，得
>
> $$f^\star(2p-1)=\frac12\log\frac{p}{1-p}\cdot\frac{p-(1-p)}{1}=\frac{(2p-1)^2}{2p(1-p)}\ \ge\ 0$$
>
> 即 $E[-Yf^\star]=f^\star(2p-1)\ge0$，符合「总体最小化 $E[-Yf]$」的直觉（$f^\star$ 是它的约束极小元）。当 $p=\frac12$ 时上式为 0，$f^\star=0$，退化情形也对。

把 $f$ 解释成 logit 变换，则

$$
p(x)=\Pr(Y=1\mid x)=\frac{1}{1+e^{-2f(x)}} \eqno{10.17}
$$

另一个有**相同总体极小元**的准则是二项负对数似然（deviance / 交叉熵）。令 $Y'=(Y+1)/2\in\{0,1\}$，则

$$
-l\big(Y,f(x)\big)=\log\big(1+e^{-2Yf(x)}\big) \eqno{10.18}
$$

> **推导** · (10.18) 与 (10.16) 同极小元
>
> $p(x)$ 与 $f$ 的关系是 (10.17)，故
>
> $$-\ell=\big[Y'\log p+(1-Y')\log(1-p)\big]$$
>
> 代入 $p=1/(1+e^{-2f})$、$1-p=e^{-2f}/(1+e^{-2f})$。对 $Y=1$：
>
> $$-\ell=-\log p=\log(1+e^{-2f})=\log(1+e^{-2Yf})$$
>
> 对 $Y=-1$：
>
> $$-\ell=-\log(1-p)=2f+\log(1+e^{-2f})=\log(e^{2f}+1)=\log(1+e^{2f})=\log(1+e^{-2Yf})$$
>
> 两种情形都得到 (10.18) ✓

**坑**：$e^{-Yf}$ 本身**不是**任何二元随机变量的概率质量函数的对数，所以它不是真正的似然；它只是一个「总体极小元碰巧相同」的损失。

作为对比，平方误差损失的总体极小元是

$$
f^{\star}(x)=\underset{f}{\mathrm{argmin}}\ E\big[\big(Y-f(x)\big)^2\big]=E\big[Y\mid x\big]=2\Pr(Y=1\mid x)-1 \eqno{10.19}
$$

> **结果** · 为什么平方误差不适合分类
>
> (10.19) 的极小元 $E[Y\mid x]=2p-1$ 存在且唯一（因为它对 $f$ 是严格二次的），但它**不是 margin 的单调递减函数**：$L(y,f)=(y-f)^2$ 在 $yf>1$ 时随 $yf$ 增大而增大（见图 10.4）。也就是说它会惩罚「非常确信地分对了」的观测，从而**降低**错分观测的相对权重。分类时我们要的是 margin 越大越好的单调递减准则。

<a class="src" href="../esl/ch10-boosting-and-additive-trees.html#s-10-5">原文 §10.5</a>

### 10.3.4 多类与回归的损失函数 {#s-10-3-4}

$K$ 类时 $Y\in\mathcal{G}=\{\mathcal G_1,\dots,\mathcal G_K\}$，$p_k(x)=\Pr(Y=\mathcal G_k\mid x)$。贝叶斯分类规则是

$$
G(x)=\mathcal G_k,\qquad k=\underset{\ell}{\mathrm{argmax}}\ p_\ell(x) \eqno{10.20}
$$

对称的多项 logit（softmax）变换写成

$$
e^{f_k(x)}p_k(x)=\sum_{\ell=1}^{K}e^{f_\ell(x)},\qquad
p_k(x)=\frac{e^{f_k(x)}}{\sum_{\ell=1}^{K}e^{f_\ell(x)}} \eqno{10.21}
$$

> **推导** · (10.21) 自动归一
>
> 对 $k$ 求和：$\sum_ke^{f_k(x)}p_k(x)=\sum_\ell e^{f_\ell(x)}$，右边的求和与 $k$ 无关，所以每个 $p_k(x)$ 都必须等于 $e^{f_k(x)}/\sum_\ell e^{f_\ell(x)}$。这保证 $0\le p_k\le1$ 且 $\sum_kp_k=1$。
>
> **冗余**：$f_k\mapsto f_k+h(x)$ 不改变 (10.21)。原文保留对称性并加约束 $\sum_{k=1}^{K}f_k(x)=0$；传统做法是令 $f_K(x)=0$（对照第 4 章 (4.17)）。

多项 deviance 损失（0/1 指示编码）：

$$
L\big(y,f(x)\big)=-\sum_{k=1}^{K}I\big(y=\mathcal G_k\big)f_k(x)+\log\sum_{\ell=1}^{K}e^{f_\ell(x)} \eqno{10.22}
$$

> **推导** · (10.22) 是多项对数似然的负数
>
> 令 $y_k=I(y=\mathcal G_k)$，则 $\sum_ky_k=1$。多项对数似然为 $\sum_ky_k\log p_k(x)$，用 (10.21) 代入 $\log p_k=f_k-\log\sum_\ell e^{f_\ell}$：
>
> $$\sum_ky_kf_k-\sum_ky_k\log\sum_\ell e^{f_\ell}=\sum_ky_kf_k-\log\sum_\ell e^{f_\ell}$$
>
> 取负即 (10.22) ✓。它对「错得有多离谱」只**线性**惩罚，比指数损失鲁棒。

回归里与平方误差对应的鲁棒损失是绝对误差 $L=|y-f(x)|$，总体解是 $\mathrm{median}(Y\mid x)$；介于两者之间的是 Huber 的 $M$-回归损失

$$
L\big(y,f(x)\big)=\begin{cases}(y-f)^2,&|y-f|\le\delta\\ 2\delta\,|y-f|-\delta^2,&\text{否则}\end{cases} \eqno{10.23}
$$

> **结果** · (10.23) 连续且总体极小元是 $E(Y\mid x)$
>
> 在 $|y-f|=\delta$ 处两支的值都是 $\delta^2$，一阶导都是 $2\delta$，所以 (10.23) 是 $C^1$ 的。它在 $|y-f|\le\delta$ 内是二次（对高斯误差效率接近最小二乘），在 $|y-f|>\delta$ 外是线性（对离群点有界影响，$\partial L/\partial f$ 恒为 $\pm2\delta$）。

<a class="src" href="../esl/ch10-boosting-and-additive-trees.html#s-10-6">原文 §10.6</a>

---

## 10.4 提升树与随机梯度提升 {#s-10-4}

### 10.4.1 树作为基学习函数 {#s-10-4-1}

回归/分类树把特征空间切成 $J$ 个互不相交的矩形 $R_1,\dots,R_J$，每块赋一个常数 $\gamma_j$。写成公式

$$
T(x;\hat\Theta)=\sum_{j=1}^{J}\gamma_j\,I(x\in R_j),\qquad \hat\Theta=\{\hat R_1,\dots,\hat R_J,\gamma_1,\dots,\gamma_J\} \eqno{10.25}
$$

$$
\hat\Theta=\underset{\Theta}{\mathrm{argmin}}\ \sum_{j=1}^{J}\min_{\text{region }R_j}\sum_{i=1}^{N}L\big(y_i,\gamma_j\big) \eqno{10.26}
$$

(10.26) 里 $J$ 通常当元参数看，$R_j$ 的找法是组合优化（原书对分类树用 Gini 指数代替错分率作分裂准则，见第 9 章）：

$$
\tilde\Theta=\underset{\Theta}{\mathrm{argmin}}\ \sum_{i=1}^{N}\tilde L\big(y_i,T(x_i,\Theta)\big) \eqno{10.27}
$$

(10.27) 是为了「优化 $R_j$」而用的**更好优化的替身准则**，找到 $\tilde R_j$ 后再用原准则精修 $\gamma_j$。

**提升树模型**就是这些树之和：

$$
f_M(x)=\sum_{m=1}^{M}T(x;\Theta_m) \eqno{10.28}
$$

### 10.4.2 阶段式提升树的两个子问题 {#s-10-4-2}

把 (10.5) 与 (10.28) 合并，第 $m$ 步要解

$$
\hat\Theta_m=\underset{\Theta}{\mathrm{argmin}}\ \sum_{i=1}^{N}L\big(y_i,f_{m-1}(x_i)+T(x_i;\Theta_m)\big) \eqno{10.29}
$$

给定区域 $R_{jm}$，求常数 $\gamma_{jm}$ 是一维极小化，通常有闭式解：

$$
\hat\gamma_{jm}=\underset{\gamma}{\mathrm{argmin}}\ \sum_{i=1}^{N}L\big(y_i,f_{m-1}(x_i)+\gamma\big) \eqno{10.30}
$$

- **平方误差**：(10.30) 的解是区域 $R_{jm}$ 内**残差 $r_{im}=y_i-f_{m-1}(x_i)$ 的均值**。此时 (10.29) 等价于「对当前残差拟合一棵回归树」，和第 9 章的单棵树一样好解。
- **二分类 + 指数损失**：(10.29) 化简成加权指数准则

$$
\hat\Theta_m=\underset{\Theta}{\mathrm{argmin}}\ \sum_{i=1}^{N}w_i^{(m)}\exp\big[-y_iT(x_i;\Theta_m)\big] \eqno{10.31}
$$

$$
\hat\gamma_{jm}=\log\frac{\sum_{x_i\in R_{jm}}w_i^{(m)}I(y_i=1)}{\sum_{x_i\in R_{jm}}w_i^{(m)}} \eqno{10.32}
$$

> **推导** · (10.32)
>
> 固定区域，把 $T$ 写成常数 $\gamma$。记 $S_+=\sum_{x_i\in R_{jm}}w_iI(y_i=1)$、$S_-=\sum_{x_i\in R_{jm}}w_iI(y_i=-1)$、$S=S_++S_-$。(10.31) 在该区域的目标变成
>
> $$S_+e^{-\gamma}+S_-e^{\gamma}$$
>
> 求导（C1）：$-S_+e^{-\gamma}+S_-e^{\gamma}=0$，即 $S_-e^{\gamma}=S_+e^{-\gamma}$，得
>
> $$e^{2\gamma}=\frac{S_+}{S_-}\quad\Longrightarrow\quad \gamma=\frac12\log\frac{S_+}{S_-}=\frac12\log\frac{p}{1-p},\qquad p=\frac{S_+}{S}$$
>
> 正是 (10.16) 的结论「**$f^\star$ 是 log-odds 的一半**」在区域上的版本，即区域内的**加权 log-odds**（带因子 $\frac12$）。
>
> 顺手核对一下这个极小是唯一的：二阶导 $S_+e^{-\gamma}+S_-e^{\gamma}>0$ ✓
>
> (10.32) 印成 $\log\frac{S_+}{S}$。把它与上面的结果相减：
>
> $$\log\frac{S_+}{S}-\frac12\log\frac{S_+}{S_-}=\frac12\log\frac{S_+(S_++S_-)}{S\cdot S_+}=\frac12\log\frac{S}{S_-}=\frac12\log\frac{p}{1-p}=\gamma$$
>
> **两者相差的量恰好等于 (10.16) 里的 $\frac12\log\frac{p}{1-p}$**，也就是说 (10.32) 等于「极小元 + $\gamma$」。这两种写法只在**参数化**上不同（$f$ 与 $2f$），对「哪个区域」的划分毫无影响（分裂准则只看权重差），但会影响叶节点数值的绝对尺度。习题 10.7 要求推导的正是 $\frac12$ 版本。

但这要求专用的树生长算法；实践上更愿意用 (10.37) 的加权最小二乘回归树近似。

> **坑** · 鲁棒准则没有「简单快速」的提升算法
>
> 绝对误差、Huber 损失 (10.23)、多项 deviance (10.22) 代入 (10.29) 后，(10.30) 仍是简单的一维「位置估计」（绝对误差给中位数），但 (10.29) 里的**树诱导**没有快速算法。下面的梯度提升正是为解决这一点。

### 10.4.3 梯度提升：把提升写成数值优化 {#s-10-4-3}

<a class="src" href="../esl/ch10-boosting-and-additive-trees.html#s-10-10">原文 §10.10</a>

把「参数」取为 $N$ 维向量 $f=\{f(x_1),\dots,f(x_N)\}^\top$，损失

$$
L(f)=\sum_{i=1}^{N}L\big(y_i,f(x_i)\big) \eqno{10.33}
$$

$$
\hat f=\underset{f}{\mathrm{argmin}}\ L(f) \eqno{10.34}
$$

**最速下降**选增量 $h_m=-\rho_mg_m$，其中 $g_m$ 是 $L(f)$ 在 $f=f_{m-1}$ 处的梯度向量，梯度分量（也叫「负梯度」「伪残差」）是

$$
g_i^{(m)}=\left.\frac{\partial L\big(y_i,f(x_i)\big)}{\partial f(x_i)}\right|_{f=f_{m-1}(x_i)}=r_i^{(m)} \eqno{10.35}
$$

> **验证** · 梯度真的是逐点的
>
> $L(f)=\sum_{i=1}^{N}L(y_i,f(x_i))$，$f(x_i)$ 是 $\mathbb R^N$ 里第 $i$ 个坐标，其他坐标不出现在第 $i$ 项里，所以对第 $i$ 个坐标求偏导只有第 $i$ 项有贡献（微分恒等式，预备知识 L4）：
>
> $$\frac{\partial L(f)}{\partial f(x_i)}=\frac{\partial L(y_i,f(x_i))}{\partial f(x_i)}\Big|_{f=f_{m-1}(x_i)}$$
>
> 所以「梯度」是一个 $N$ 维向量，可以逐点写成 (10.35) 的形式。

步长由线搜索给出（沿 $-g_m$ 方向精确最小化）：

$$
\rho_m=\underset{\rho}{\mathrm{argmin}}\ L\big(f_{m-1}-\rho g_m\big),\qquad f_m=f_{m-1}-\rho_mg_m \eqno{10.36}
$$

**梯度提升的做法**：梯度是 $N$ 维向量，**未必来自任何一棵树**。妥协办法是找一棵 $J_m$ 叶节点树 $T(\cdot;\Theta_m)$，使它的预测向量 $-t_m$ 尽量接近 $-g_m$，用平方误差度量「接近」，得到

$$
\tilde\Theta_m=\underset{\Theta}{\mathrm{argmin}}\ \Big(-\sum_{i=1}^{N}g_i^{(m)}T(x_i;\Theta_m)\Big)^2 \eqno{10.37}
$$

> **推导** · (10.37) 为什么能用平方误差
>
> 记 $t_m=\big(T(x_1;\Theta_m),\dots,T(x_N;\Theta_m)\big)^\top$，则被平方的量是 $-t_m^\top g_m$。这是关于 $\Theta_m$ 的二次型：
>
> $$\frac{\partial\psi}{\partial\Theta_m}=-2\big(t_m^\top g_m\big)\cdot\nabla_{\Theta_m}t_m$$
>
> 令零给出 $t_m^\top g_m=0$ 或 $\nabla t_m=0$，两者都不理想。实际做法是**把它当成最小二乘回归**：把 $-g_i^{(m)}$ 当响应、$T(x_i;\Theta_m)$ 当拟合值，在最小二乘意义下最优。这与 10.4.1 节 (10.26)/(10.27) 的「找 $R_j$ 用替身准则」是同一套路。
>
> **假设与代价**：$\tilde R_{jm}\ne\hat R_{jm}$（替身准则的最优区域与原准则不一定相同），但「通常足够接近，可用」。这是梯度提升全部近似的来源。

然后用 (10.30) 给每块赋常数，更新

$$
f_m(x)=f_{m-1}(x)+\sum_{j=1}^{J_m}\hat\gamma_{jm}\,I(x\in R_{jm})
$$

最后输出 $\hat f(x)=f_M(x)$。

> **结果** · 常用损失的伪残差表（Algorithm 10.3 第 2(a) 步）
>
> | 设置 | 损失 | $-\partial L/\partial f(x_i)$ |
> |---|---|---|
> | 回归 | $\big[y_i-f(x_i)\big]^2$ | $y_i-f(x_i)$ |
> | 回归 | $\lvert y_i-f(x_i)\rvert$ | $\mathrm{sign}\big[y_i-f(x_i)\big]$ |
> | 回归 | Huber (10.23) | $y_i-f(x_i)$（$\lvert\cdot\rvert\le\delta_m$），否则 $\delta_m\,\mathrm{sign}\big[y_i-f(x_i)\big]$ |
> | 分类 | deviance (10.22) | 第 $k$ 分量 $I(y_i=\mathcal G_k)-p_k(x_i)$ |
>
> 其中 $\delta_m$ 取 $\lvert y_i-f_{m-1}(x_i)\rvert$ 的 $\alpha$ 分位数。**平方误差时伪残差就是普通残差**，(10.37) 等价于普通最小二乘提升；绝对误差时拟合的是残差的符号。

多分类时每次迭代要长 $K$ 棵树，第 $k$ 棵拟合自己的负梯度：

$$
-g_{ik}^{(m)}=\frac{\partial L\big(y_i,f_1(x_i),\dots,f_K(x_i)\big)}{\partial f_k(x_i)}\Big|_{f=f_{m-1}}=I\big(y_i=\mathcal G_k\big)-p_k(x_i) \eqno{10.38}
$$

> **推导** · (10.38)
>
> 由 (10.22) 与 (10.21)：
>
> $$\frac{\partial L}{\partial f_k}=-y_k+\frac{e^{f_k}}{\sum_\ell e^{f_\ell}}=-y_k+p_k$$
>
> 取负即 $y_k-p_k=I(y=\mathcal G_k)-p_k$。
>
> **为什么 $K=2$ 时只要一棵树**（Exercise 10.10）：约束 $\sum_kf_k=0$ 下 $p_1=1/(1+e^{f_2-f_1})$，只剩一个自由度，故 $K$ 棵树中有一棵是冗余的。

**这条路解决了什么**：只要 $L(y,f(x))$ 可微，$-g_i^{(m)}$ 就总能算出来，而 (10.37) 只是最小二乘回归树诱导——第 9 章有 $O(N\log N)$ 的快速算法。于是**任何**可微损失（Huber、绝对值、多项 deviance）都能得到快速提升算法，同时保留鲁棒性。代价是 (10.37) 的区域一般不等于 (10.29) 的最优区域，只是一阶近似。

---

### 10.4.4 树的规模、正则化与随机子抽样 {#s-10-4-4}

<a class="src" href="../esl/ch10-boosting-and-additive-trees.html#s-10-11">原文 §10.11–10.12</a>

提升要近似的目标函数是

$$
\eta=\underset{\eta}{\mathrm{argmin}}\ E_{X,Y}\big[L\big(Y,f_\eta(X)\big)\big] \eqno{10.39}
$$

它的 ANOVA 展开

$$
\eta(X)=\sum_{j=1}^{p}\eta_j(X_j)+\sum_{j<k}\eta_{jk}(X_j,X_k)+\sum_{j<k<\ell}\eta_{jk\ell}(X_j,X_k,X_\ell)+\cdots \eqno{10.40}
$$

> **结果** · 交互阶数与树大小的对应
>
> $J$ 叶节点的树最多只能表示 $J-1$ 阶交互（每次分裂最多「引入」一个新变量的作用）。因为提升模型是树的加法 (10.28)，这个上限对整体模型同样成立：
>
> - $J=2$（stump）：只有主效应，**不允许任何交互**
> - $J=3$：允许二阶交互
> - 经验上 $4\le J\le8$ 都好用，且结果对这个区间不敏感
>
> 真函数是纯加性时（图 10.2 的例子就是二次单项式之和），$J>2$ 会白白引入方差，导致测试误差升高。

**收缩（shrinkage）**：每棵树乘一个学习率 $0<\nu<1$ 再加进去，

$$
f_m(x)=f_{m-1}(x)+\nu\cdot\sum_{j=1}^{J_m}\hat\gamma_{jm}\,I(x\in R_{jm}) \eqno{10.41}
$$

$\nu$ 控制学习率：$\nu$ 越小，同样训练风险需要越大的 $M$。经验上取 $\nu<0.1$ 再用早停选 $M$ 最好，代价是计算量（正比于 $M$）。

**随机子抽样**：每次迭代只抽 $\eta$ 比例（典型 $\eta=1/2$）的训练观测来长树。这把计算量降到 $\eta$ 倍，且常常更准——与第 8 章 bagging 同源，但这里抽样与加法（而非平均）结合，效果不同于 bagging。

### 10.4.6 例：随机森林与提升的比较 {#s-10-4-6}

第一版 ESL 在这里有一个对比实验（§10.4.1），第二版删掉了，但它把「随机性到底该放在哪」讲得很清楚，值得保留——而且它用的正是 10.1.2 节的 (10.2) 那个例子（判别面是球面 $\sum_jX_j^2>\chi^2_{10}(0.5)$）。第二版 Figure 10.2 里保留下来的数字是：

| 方法 | 测试错分率 |
|---|---|
| 随机猜测 | 50.0% |
| 单个 stump | 45.8% |
| 244 节点的分类树 | 24.7% |
| stump + boosting（400 轮） | 5.8% |

三条策略的对比说明三件事：

1. **平均弱学习器（bagging / 随机森林）能救弱学习器**。stump 单独 45.8%，对 $B$ 个 bootstrap 样本各自长一棵 stump 再平均，可以把错分率降到十几个百分点，接近一棵 244 节点的大树。机制是第 15 章的方差分解：$B$ 个（近似）独立的 stump 平均后，偏差接近单个 stump 的偏差，方差却降到 $1/B$。关键在于随机性**消掉了 stump「切哪一刀」的任意性**——一百个随机切分的平均几乎给出一条稳定的分界。
2. **加法（boosting）通常比平均更强**，而且用的基学习器更弱。它不是把一个粗糙切分重复 $B$ 次，而是**每一步都在修正上一步的错误**（靠 (10.14) 的权重重分配），$B$ 轮之后得到一个由 $B$ 个弱分类器**加性组合**的高阶函数——这正是 (10.3) 的加性展开在起作用。在这个球面判别的问题上，stump + boosting 的 5.8% 远好于一棵 244 节点树（24.7%），原因就在这里。
3. **随机性在提升里是「加速器」而不是「必需」。** 纯 boosting（不抽样）已经能到 5.8%；加上行子抽样（10.4.4 的 $\eta$）与收缩 $\nu$ 后，图 10.11、10.12 显示在 deviance 损失下测试误差下降得更低、保持得更久。原因是：抽样降低单棵树的方差，收缩降低单次更新的步长，两者都让 $f_M$ 更接近总体最优 $\eta$ (10.39)。而 **bagging 里随机性是多样性的唯一来源，去掉它就退化成同一棵树重复 $B$ 次**。

| | bagging / 随机森林 | boosting |
|---|---|---|
| 组合方式 | **平均** $\bar f=\frac1B\sum_bf_b$ | **加法** $f_M=\sum_mT_m$（可带 $\nu$ 缩放） |
| 多样性来源 | 数据重抽样（bootstrap / 子采样） | 数据权重的重分配（(10.14)） |
| 每步是否修正前一步 | 否，各自独立 | 是，专注错分样本 |
| 随机性的角色 | 必需 | 可选（$\nu$、$\eta$ 是加速器） |
| 主要降低 | 方差 | 偏差（在弱学习器之上） |
| 典型元参数 | 树大小 $J$、棵数 $B$ | 树大小 $J$、迭代数 $M$、$\nu$、$\eta$ |

**参数清单**：$J$（树大小）、$M$（迭代数）、$\nu$（学习率）、$\eta$（抽样比例）。实践上先定 $J,\nu,\eta$，把 $M$ 留作主要调节量。

### 10.4.5 变量重要度与部分依赖图 {#s-10-4-5}

单棵 Breiman 树的平方相对重要度

$$
I_\ell^2(T)=\sum_{t\,:\,v(t)=\ell}\hat\iota_t^2 \eqno{10.42}
$$

求和遍历 $J-1$ 个内部节点 $t$，$v(t)$ 是节点 $t$ 的分裂变量，$I(v(t)=\ell)$ 是示性函数，$\hat\iota_t^2$ 是该分裂在平方误差风险上的**估计改进量**。加性树展开下取平均：

$$
\frac{1}{M}\sum_{m=1}^{M}I_\ell^2=\frac{1}{M}\sum_{m=1}^{M}I_\ell^2(T_m) \eqno{10.43}
$$

$K$ 类时每个类一套展开

$$
f_k(x)=\sum_{m=1}^{M}T_{km}(x) \eqno{10.44}
$$

$$
\frac{1}{M}\sum_{m=1}^{M}I_k^2=\frac{1}{M}\sum_{m=1}^{M}I_k^2(T_{km}) \eqno{10.45}
$$

$$
\frac{1}{K}\sum_{k=1}^{K}I_\ell^2=\frac{1}{K}\sum_{k=1}^{K}I_{\ell k}^2 \eqno{10.46}
$$

**平均效应（partial dependence）**：取子向量 $X_{\mathcal S}$（$\mathcal S\subset\{1,\dots,p\}$），$\mathcal C$ 为补集，把 $X_{\mathcal C}$ 上**边缘平均**掉：

$$
\bar f_{\mathcal S}(X_{\mathcal S})=E_{X_{\mathcal C}}\,f(X_{\mathcal S},X_{\mathcal C}) \eqno{10.47}
$$

实践中用训练集上出现过的 $X_{\mathcal C}$ 值做经验平均：

$$
\frac{1}{N}\sum_{i=1}^{N}\bar f_{\mathcal S}(X_{\mathcal S})=\frac{1}{N}\sum_{i=1}^{N}f\big(X_{\mathcal S},x_{i\mathcal C}\big) \eqno{10.48}
$$

（用塔性质 $E[E[f\mid X_{\mathcal S}]]=E[f]$ 把 (10.47) 与经验平均连起来。）

与**条件期望**（最接近的只依赖 $X_{\mathcal S}$ 的最小二乘近似）对照：

$$
\tilde f_{\mathcal S}(X_{\mathcal S})=E\big[f(X_{\mathcal S},X_{\mathcal C})\mid X_{\mathcal S}\big] \eqno{10.49}
$$

**关键区别**：(10.47) 是对 $X_{\mathcal C}$ **取平均**（边缘平均），(10.49) 是**给定** $X_{\mathcal S}$（条件期望）。两者只在 $X_{\mathcal S}$ 与 $X_{\mathcal C}$ 独立时才一致。若效应纯加性

$$
f(X)=h_1(X_{\mathcal S})+h_2(X_{\mathcal C}) \eqno{10.50}
$$

则 (10.47) 给出

$$
E_{X_{\mathcal C}}f=h_1(X_{\mathcal S})+\underbrace{E_{X_{\mathcal C}}h_2(X_{\mathcal C})}_{\text{常数}}
$$

即 $h_1$ 加一个加性常数。若效应纯乘性

$$
f(X)=h_1(X_{\mathcal S})\cdot h_2(X_{\mathcal C}) \eqno{10.51}
$$

则 (10.47) 给出 $h_1(X_{\mathcal S})\cdot E_{X_{\mathcal C}}h_2(X_{\mathcal C})$，即 $h_1$ 乘一个乘性常数。而 (10.49) 在这两种情形下**都**不给出 $h_1$。

> **坑** · (10.49) 会凭空造出效应（Exercise 10.12）
>
> **(a) 纯加性也失败。** 取 $f(X_1,X_2)=X_1+X_2$，$X_1,X_2$ 相关系数 $\rho$、方差 1。则
>
> $$E\big[f\mid X_2\big]=E(X_1\mid X_2)+X_2=\rho X_2+X_2=(1+\rho)X_2$$
>
> **出现了 $X_2$ 的依赖**，而 $h_1(X_{\mathcal S})=X_1$ 里根本没有 $X_2$。
>
> **(b) 零效应也会造出效应。** 取 $f(X_1,X_2)=X_1$（**完全不含 $X_2$**），$(X_1,X_2)$ 是相关系数 $\rho$ 的双高斯，则
>
> $$E\big[f\mid X_2\big]=E(X_1\mid X_2)=\rho X_2$$
>
> 画出来的「部分依赖」是一条斜率为 $\rho$ 的直线，**尽管 $f$ 与 $X_2$ 毫无关系**。
>
> 结论：部分依赖图必须用 (10.47)（边缘平均），不能用 (10.49)（条件期望）。

$K$ 类下 (10.47) 的绘图对象通过

$$
f_k(X)=\log p_k(X)-\log p_\ell(X) \eqno{10.52}
$$

与各类概率相连（等价于 (10.21) 取对数，因为 $\log p_k=f_k-\log\sum e^{f_\ell}$）。

---

## 10.5 示例与多分类提升 {#s-10-5}

### 10.5.1 两个应用 {#s-10-5-1}

加州住房数据用 $J=6$、$\nu=0.1$、Huber 损失，评价指标是平均绝对误差

$$
\mathrm{AAE}=E\big|y-\hat f_M(x)\big| \eqno{10.53}
$$

800 轮后 AAE $=0.31$（最优常数预测器 $\mathrm{median}\{y_i\}$ 为 0.89），$R^2=0.84$。垃圾邮件那一节被建模的量是 log-odds

$$
f(x)=\log\frac{\Pr(\text{spam}\mid x)}{\Pr(\text{email}\mid x)} \eqno{10.24}
$$

用 $J=5$ 的树得测试错分率 4.5%，改用 $J=2$（纯主效应、无交互）得 4.7%，差异不显著，说明可能存在交互——用二维部分依赖图可以看出来（例如 hp 与字符 `!` 之间的强交互：hp 频率很低时 log-odds 随 `!` 上升得更快）。

新西兰一种鱼的出现量分解为

$$
E(Y\mid X)=E\big(Y\mid Y>0,X\big)\cdot\Pr(Y>0\mid X) \eqno{10.54}
$$

前半部分只用 2353 条有渔获的拖网数据拟合，后半部分用逻辑 GBM 拟合 $\Pr(Y>0\mid X)$。

### 10.5.2 多分类指数损失与 Newton 更新 {#s-10-5-2}

Exercise 10.5（Zhu et al., 2005）的编码是

$$
Y_k=\begin{cases}1,&G=\mathcal G_k\\ -\dfrac{1}{K-1},&\text{否则}\end{cases} \eqno{10.55}
$$

$$
L(Y,f)=\exp\Big(-\sum_{k=1}^{K}Y_kf_k\Big) \eqno{10.56}
$$

约束 $\sum_{k=1}^{K}f_k=0$。**先化简指数里的量**：设真类是 $\mathcal G_G$，则

$$
\sum_{k=1}^{K}Y_kf_k=f_G-\frac{1}{K-1}\sum_{k\neq G}f_k=f_G-\frac{1}{K-1}\big(-f_G\big)=\frac{K}{K-1}f_G
$$

（用了零和约束 $\sum_{k\ne G}f_k=-f_G$。）于是

$$
L(Y,f)=\exp\Big(-\frac{K}{K-1}f_{G}\Big)=\exp\Big(-y_G^\top f\Big)
$$

其中 $y_G$ 就是 (10.55) 定义的编码向量。所以这一族损失**指数地惩罚真类 log-odds 的负值**，结构与二类的指数损失完全平行。

> **推导** · (10.56) 的总体极小元（Exercise 10.5(a)）
>
> 设 $\pi_k=\Pr(Y=\mathcal G_k\mid x)$，$\sum_k\pi_k=1$。总体风险
>
> $$R(f)=\sum_k\pi_k\exp\Big(-\frac{K}{K-1}f_k\Big)$$
>
> 加拉格朗日乘子 $\lambda$（预备知识 O1）处理零和约束：
>
> $$\mathcal L(f,\lambda)=\sum_k\pi_ke^{-\frac{K}{K-1}f_k}+\lambda\sum_kf_k$$
>
> 一阶条件（C1，Lagrange 乘子必为零）：
>
> $$-\frac{K}{K-1}\pi_ke^{-\frac{K}{K-1}f_k}+\lambda=0\quad\forall k\ \Longrightarrow\ \pi_ke^{-\frac{K}{K-1}f_k}=\frac{K-1}{K}\lambda\quad(\text{与 }k\ \text{无关})$$
>
> 对 $k$ 求和：$\sum_k\pi_ke^{-\frac{K}{K-1}f_k}=1$，故 $\frac{K-1}{K}\lambda=1$，代回得
>
> $$e^{-\frac{K}{K-1}f_k}=\frac{K-1}{K\pi_k}\quad\Longrightarrow\quad f_k=\frac{K-1}{K}\log\frac{K-1}{K\pi_k}$$
>
> **验证零和**：
>
> $$\sum_kf_k=\frac{K-1}{K}\sum_k\left[\log\frac{K-1}{K}-\log\pi_k\right]=\frac{K-1}{K}\left[K\log\frac{K-1}{K}-\log\prod_k\pi_k\right]$$
>
> **这一般不等于 0！** 也就是说 (10.56) 加上零和约束后，**总体极小元不存在**（除非 $\prod_k\pi_k=\big(\frac{K-1}{K}\big)^K$，即各类均匀）。
>
> 这正是 Zhu et al. (2005) 要引入「不强制零和、只用 softmax 归一化」的版本的原因：$\sum_k\pi_k=1$ 已经由 (10.21) 自动保证，$f$ 有一个整体平移自由度是**无害**的（平移不改变 $p_k$）。
>
> **二阶条件**：$R$ 对每个 $f_k$ 是凸的（$e^{-\cdot}$ 凸），故上述驻点是唯一极小（在无约束版本下）。

**由 (10.56) 得到多分类的权重重分配**：因为 $L(Y,f)=\exp(-y^\top f)$，而 $\sum_kY_kf_k$ 只通过 $y^\top f$ 进入，所以第 $m$ 轮的权重是 $w_i^{(m)}=\exp(-y_{i}^\top f_{m-1}(x_i))$，与二类情形 (10.14) 一模一样，只是「符号函数」换成了指示编码。**这就是多分类提升算法只依赖权重重分配的原因**。

Exercise 10.8 给出「在区域 $R$ 里给每个类加常数 $f_k(x)+\gamma_k$」的一步 Newton 更新。目标（多项负对数似然）在 $\gamma_k$ 上的一阶导是 $y_{ik}-p_{ik}$，二阶导（Hessian 对角元）是 $-p_{ik}(1-p_{ik})$。从 $\gamma_k=0$ 出发做一步近似 Newton：

$$
\gamma_k^{(0)}-\frac{\partial\ell/\partial\gamma_k}{\partial^2\ell/\partial\gamma_k^2}\Big|_{\gamma_k=0}=-\frac{y_{ik}-p_{ik}}{-p_{ik}(1-p_{ik})}=\frac{y_{ik}-p_{ik}}{p_{ik}(1-p_{ik})}
$$

对区域内的 $i$ 加权平均（各观测的一步更新取平均）即得

$$
\gamma_k=\frac{\sum_{x_i\in R}(y_{ik}-p_{ik})}{\sum_{x_i\in R}p_{ik}(1-p_{ik})} \eqno{10.57}
$$

> **结果** · (10.57) 的三个性质
>
> - **分子是「校准残差」**。$p_{ik}$ 是模型预测、$y_{ik}$ 是 0/1 实际值，$y_{ik}-p_{ik}$ 就是第 $i$ 个观测在第 $k$ 类上的残差。**跨类求和为零**：$\sum_k\sum_{x_i\in R}(y_{ik}-p_{ik})=0$，因为对每个 $i$ 都有 $\sum_kp_{ik}=1=\sum_ky_{ik}$。这正是 (10.58) 能把它中心化到零和的原因。
> - **分母是「有效信息量」**。$p_{ik}(1-p_{ik})$ 是多项 deviance 在 $p_{ik}$ 处的曲率（Fisher information）：$p$ 越接近 0 或 1，$p(1-p)$ 越小，同样大小的残差给出的更新越大——牛顿法自然地放大了高置信度观测的修正。
> - **纯区域会给出极端更新**。若区域里所有 $p_{ik}$ 都接近 0 或 1，分母趋近 0，更新会爆掉。所以区域不能太小，(10.58) 的中心化也是必要的稳定化手段。

为了满足和为零的约束，利用对称性做中心化：

$$
\hat\gamma_k=\frac{1}{K-1}\Big(\gamma_k-\frac{1}{K}\sum_{\ell=1}^{K}\gamma_\ell\Big) \eqno{10.58}
$$

> **推导** · (10.58) 的两处系数
>
> **第一处（零和）**：设 $\hat\gamma_k=\gamma_k+c$（对所有 $k$）。则
>
> $$\sum_k\hat\gamma_k=\sum_k\gamma_k+Kc$$
>
> 要它为零就取 $c=-\frac1K\sum_\ell\gamma_\ell$，得 $\tilde\gamma_k=\gamma_k-\frac1K\sum_\ell\gamma_\ell$。
>
> **第二处（$1/(K-1)$）**：再整体乘一个正数，零和性质不变（$\sum_k\frac{1}{K-1}\tilde\gamma_k=\frac{1}{K-1}\cdot0=0$）。这个正数的作用是**把量级拉回二类尺度**：二类情形下 (10.17) 给出 $f^\star=\frac12\log\frac{p}{1-p}$，而多分类一阶条件里的系数是 $\frac{K}{K-1}$（见上面 $\sum_kY_kf_k=\frac{K}{K-1}f_G$）。所以 $\frac{1}{K-1}$ 是同一个系数在反向抵消——它保证 $\hat\gamma_k$ 与 $\frac12\log\frac{p_k}{1-p_k}$ 同量级。
>
> **依赖的假设**：(a) 只用 Hessian 的对角元（忽略类间的强相关），故是**近似** Newton 步；(b) 从 $\gamma_k=0$ 出发，所以只有一阶精度；(c) 区域 $R$ 上的观测数足够多、分母不接近 0。

---

## 10.6 小结与易错点 {#s-10-6}

1. **提升 ≠ bagging**：bagging 对弱学习器**取平均**、各成员独立（bootstrap）、降方差；提升**加权求和**、成员串行、权重由指数损失的加权错分率决定（降偏差）。
2. **AdaBoost = 指数损失 + 阶段式加性建模**。(10.8)–(10.15) 是这一等式的全部内容。
3. **$\alpha_m=2\beta_m=\log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}$**：不是拍脑袋的，由 (10.11) 对 $\beta$ 求导得到；$\tanh\beta_m=1-2\mathrm{err}_m$ 才是那个「加权平均符号」。
4. **指数损失 vs deviance vs 平方误差**：三者总体极小元分别在 (10.16)/(10.17)、(10.18)、(10.19)。前两者相同，平方误差不同且对 margin **非单调**（$yf>1$ 时反而增大）。
5. **梯度提升的核心是 (10.37)**：用回归树最小二乘拟合伪残差 $-g^{(m)}$。这让任何可微损失都能得到快速算法。
6. **$J$ 限制交互阶数，$\nu$ 控制学习率，$M$ 控制过拟合，$\eta$ 提供随机性**：四个元参数，通常先定 $J,\nu,\eta$，只把 $M$ 留作主要调节量。
7. **部分依赖用 (10.47) 而非 (10.49)**：后者会因变量间的相关而凭空造出效应。
8. **两处 $\frac1{K-1}$ 不是随便写的**：多分类的一阶条件里系数是 $\frac{K}{K-1}$（由 (10.55) 的编码决定），(10.57)/(10.58) 的 $\frac{1}{K-1}$ 是在反向抵消它。
9. **常见错写**：$\tanh\beta_m=\mathrm{err}_m$（应为 $1-2\mathrm{err}_m$）、$I(y_if\le0)\le\frac14(1-y_if)^2$（只在额外约束下成立）、$\mathrm{err}_m=\sum_iw_iI(\cdot)$（漏掉 $\sum w_i$ 归一化）。

> **延伸**
>
> 第 15、16 章从方差缩减与随机化角度统一了 bagging 与 boosting。bagging 的方差分解 $\mathrm{Var}(\bar f)=\frac1B\mathrm{Var}(f)+\big(1-\frac1B\big)\sigma^2$（第 8 章）与提升的「慢学习」（小 $\nu$、大 $M$）在极限行为上是一致的：**两者都在用「多算几次、平均掉噪声」换准确度，只是提升平均的是一阶修正而不是独立模型**。Exercise 10.4(d) 里的重叠类分布则解释了为什么 AdaBoost 在噪声标签上会退化。

<a class="src" href="../esl/index.html">返回目录</a>
