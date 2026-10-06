---
id: m04
n: "4"
title: 线性分类方法
title_en: Linear Methods for Classification
desc: 逻辑回归的极大似然推导与 IRLS、似然比置信集与 Wald/score 检验、线性判别分析的对数优势比化简、RDA 收缩估计量、感知机与最优分离超平面对偶
prev: m03
next: m05
prev_title: 第 3 章 线性回归方法
next_title: 第 5 章 基展开与正则化
---

# 4 线性分类方法 {#s-4}

第 2 章用 0-1 损失推出 Bayes 规则 $\hat G(x)=\arg\max_k\Pr(G=\mathcal{G}_k\mid X=x)$（式 (2.23)），但后验概率本身未知。本章给出三种可算的替代：把类别指示变量做**线性回归**（§4.2）、对类条件密度做**高斯假设**得到线性判别分析 LDA（§4.3）、直接对后验的 log-odds 建模得到**逻辑回归**（§4.4），最后是显式寻找**分离超平面**的感知机与最优超平面（§4.5）。本章 58 个编号公式全部给出中间步骤；核心推导是 (4.2) 的 logit 形式、(4.9) 的对数优势比化简、(4.20)–(4.28) 的逻辑回归极大似然与 IRLS、(4.49)–(4.53) 的 Wolfe 对偶。

<a class="src" href="../esl/ch04-linear-methods-for-classification.html#s-4-1">原文 §4.1–4.2</a>

---

## 4.1 导论：什么算「线性方法」 {#s-4-1}

由于预测器 $G(x)$ 取值于离散集合 $\mathcal{G}$，总能把输入空间切成若干带标签的区域。我们说「线性分类方法」是指**分界线是超平面**的方法。这有两类实现方式：

- **拟合 $K$ 个线性判别函数** $\delta_k(x)$，再分到 $\delta_k(x)$ 最大的类。两类之间的分界线是 $\{\hat f_k(x)=\hat f_\ell(x)\}$，即 $\{\hat\beta_{k0}+\hat\beta_k^\top x=\hat\beta_{\ell0}+\hat\beta_\ell^\top x\}$——一个仿射集。第 2 章的指示变量回归就属于这一类。
- **建模后验概率** $\Pr(G=k\mid X=x)$，且要求「后验的某个**单调变换**是 $x$ 的线性函数」。

第二条只要求单调变换线性，这是理解全章的钥匙。两类问题最流行的模型是

$$
\Pr(G=1\mid X=x)=\frac{\exp(\beta_0+\beta^\top x)}{1+\exp(\beta_0+\beta^\top x)}
$$

$$
\Pr(G=2\mid X=x)=\frac{1}{1+\exp(\beta_0+\beta^\top x)} \eqno{4.1}
$$

**为什么 (4.1) 合理。** 分子分母的差别正好是 $\exp(\beta_0+\beta^\top x)$，所以两个后验之比就是这个指数。取对数（这就是 **logit 变换** $\log\frac{p}{1-p}$）：

$$
\log\frac{\Pr(G=1\mid X=x)}{\Pr(G=2\mid X=x)}=\beta_0+\beta^\top x \eqno{4.2}
$$

**推导 · 从归一化到 (4.1)。** 先看不带指数的情况：若 $p_1+p_2=1$ 且 $p_1/p_2=\exp(\eta)$，其中 $\eta=\beta_0+\beta^\top x$，则 $p_2\bigl(1+\exp\eta\bigr)=1$，即 $p_2=\frac{1}{1+e^\eta}$、$p_1=\frac{e^\eta}{1+e^\eta}$，得 (4.1)。反过来，任何 (4.1) 形式的模型都满足「和为 1」与「比值是指数」两条。

分界线是 log-odds 为零的集合，即 $\{x\mid \beta_0+\beta^\top x=0\}$，一个超平面。这正是逻辑回归模型：**logit 线性**。

> **结果**
> **只要 $\delta_k(x)$ 或 $\Pr(G=k\mid X=x)$ 满足「某个单调变换是 $x$ 的线性函数」，分界线就线性。** 这一条把本章四种方法统一起来：指示变量回归是「$\hat f_k$ 本身线性」，LDA 是「$\log$ 后验之比线性」（(4.9)），逻辑回归是「logit 线性」（(4.2)），最优分离超平面是「直接给超平面」。

**方法的推广方向。** 把变量集扩充成 $\{X_1,\dots,X_p,X_1^2,\dots,X_1X_p,\dots\}$，多加 $p(p+1)/2$ 个变量，增广空间里的线性函数映回原空间是二次函数，于是**线性分界线变成二次分界线**（图 4.1）。任何基变换 $h:\mathbb R^p\to\mathbb R^q$（$q>p$）都可以这样用，第 5、6 章展开讨论。

---

## 4.2 指示变量矩阵的线性回归 {#s-4-2}

<a class="src" href="../esl/ch04-linear-methods-for-classification.html#s-4-2">原文 §4.2</a>

若有 $K$ 个类，用 $K$ 个指示变量 $Y_k=\mathbf 1\{G=k\}$，$k=1,\dots,K$，排成 $N\times K$ 矩阵 $Y$（每行只有一个 1）。对 $Y$ 的每一列同时拟合线性模型：

$$
\hat Y=X(X^\top X)^{-1}X^\top Y \eqno{4.3}
$$

$X$ 是 $N\times(p+1)$ 的模型矩阵（第一列为 1），系数矩阵 $\hat B=(X^\top X)^{-1}X^\top Y$ 是 $(p+1)\times K$。对新输入 $x$，算出 $K$ 维拟合向量 $\hat f(x)^\top=(1,x^\top)\hat B$，然后分到最大分量：

$$
\hat G(x)=\underset{k\in\mathcal G}{\mathrm{argmax}}\ \hat f_k(x) \eqno{4.4}
$$

**为什么这是合理的。** 指示变量的条件期望就是后验概率：

$$
E\big[Y_k\mid X=x\big]=\Pr\big(G=\mathcal{G}_k\mid X=x\big)
$$

所以「用回归估计条件期望」就是想估计后验。真正的问题是线性回归对条件均值的近似有多好。

**一个可以验证的好性质。** 只要模型含截距（$X$ 有全 1 的列），就有

$$
\sum_{k\in\mathcal G}\hat f_k(x)=1\qquad\text{对一切}\ x
$$

**推导**：$\hat f_k(x)=x_{\star}^\top\hat\beta_k$，其中 $x_{\star}=(1,x_1,\dots,x_p)^\top$。于是 $\sum_k\hat f_k(x)=x_{\star}^\top\Big(\sum_k\hat\beta_k\Big)=x_{\star}^\top(X^\top X)^{-1}X^\top\Big(\sum_ky^{(k)}\Big)$，而 $\sum_k y^{(k)}=\mathbf 1_N$（每行的 $K$ 个指示之和为 1），故上式 $=x_{\star}^\top(X^\top X)^{-1}X^\top\mathbf 1_N$。含截距时 $X^\top\mathbf 1_N$ 的每个分量都等于 $N$，即 $X^\top\mathbf 1_N=N\cdot(\text{截距列})$，因此该向量属于 $\mathrm{col}(X)$，其投影恰为它自身：

$$
\sum_k\hat f_k(x)=x_{\star}^\top X^\top\mathbf 1_N/N=1
$$

**但 $\hat f_k(x)$ 可能是负的或大于 1**，尤其在训练数据凸包之外做外推时。这是线性模型刚性的后果。

**等价的「最近靶标」视角。** 把目标写成 $K\times K$ 单位矩阵的第 $k$ 列 $t_k$，然后最小化平方欧氏距离：

$$
\min_{B}\ \sum_{i=1}^N\Big\lVert y_i-\big[(1,x_i)B\big]\Big\rVert_2^2 \eqno{4.5}
$$

分到最近靶标：

$$
\hat G(x)=\underset{k}{\mathrm{argmin}}\ \lVert\hat f(x)-t_k\rVert_2 \eqno{4.6}
$$

**证明 (4.6) 与 (4.4) 等价。** 记 $\hat f=(\hat f_1,\dots,\hat f_K)^\top$ 且 $\sum_k\hat f_k=1$，则

$$
\lVert\hat f-t_k\rVert_2^2=\sum_{j=1}^K(\hat f_j-\delta_{jk})^2=\sum_j\hat f_j^2-2\hat f_k+1
$$

$\sum_j\hat f_j^2$ 与末项 $+1$ 与 $k$ 无关，故最小化等价于**最大化** $\hat f_k$，正是 (4.4)。

同时 (4.5) 与 (4.3) 等价：平方范数本身就是平方和，$\sum_i\lVert y_i-[(1,x_i)B]\rVert_2^2=\sum_{k=1}^K\sum_i(y_{ik}-x_i^\top\hat\beta_k)^2$，各分量**解耦**，于是每个 $k$ 是一个独立的一元回归。注意这种解耦只有在「模型不把不同响应绑在一起」时才成立。

> **坑** · 掩蔽（masking）
> 当 $K\ge3$ 时回归方法有严重问题：某些类会被其他类**完全遮住**，它的拟合值从不占优。图 4.3 的例子最极端——三个类沿一条直线完全分离，中间的类对应的回归线是水平的、拟合值从不超过 0.5，于是类 2 的观测全被分到类 1 或类 3，训练误差率 $0.33$（Bayes 率仅 $0.025$）。用二次回归替代线性回归可以把误差降到 $0.04$。一般规律：$K$ 个类排成一线时可能需要 $K-1$ 次多项式；在 $p$ 维空间最坏情形需要总次数 $K-1$ 的全部多项式项与交叉项，共 $O(p^{K-1})$ 个。表 4.1 的元音数据（$K=11$、$p=10$）给出真实代价：线性回归测试误差 $0.67$，而 LDA 只有 $0.56$。**所以「用回归做分类」只在 $K=2$ 时是安全的。**

---

## 4.3 线性判别分析 {#s-4-3}

<a class="src" href="../esl/ch04-linear-methods-for-classification.html#s-4-3">原文 §4.3</a>

### 4.3.1 Bayes 规则与高斯类条件密度 {#s-4-3-1}

判决理论说我们需要类后验 $\Pr(G\mid X)$。设 $f_k(x)$ 是类 $G=k$ 下 $X$ 的类条件密度，$\pi_k$ 是类先验（$\sum_{k=1}^K\pi_k=1$）。由 Bayes 公式

$$
f_k(x)\,\pi_k\,\Pr(G=k\mid X=x)=\sum_{\ell=1}^{K}f_\ell(x)\,\pi_\ell \eqno{4.7}
$$

**推导 · 从贝叶斯定理到 (4.7)，两步。** 第一步，全概率公式（把 $X$ 看作随机变量、$G$ 把它划分成 $K$ 块）给出

$$
\Pr(X=x)=\sum_{\ell=1}^K\Pr(G=\mathcal{G}_\ell)\Pr(X=x\mid G=\mathcal{G}_\ell)=\sum_{\ell=1}^K\pi_\ell f_\ell(x)
$$

第二步，条件概率的定义 $\Pr(G=k\mid X=x)=\dfrac{\Pr(G=k,X=x)}{\Pr(X=x)}=\dfrac{\pi_kf_k(x)}{\Pr(X=x)}$，把第一步的分母代入并交叉相乘 $\Pr(G=k\mid X=x)\cdot\Pr(X=x)=\pi_kf_k(x)$ 即得 (4.7)。

(4.7) 说明「掌握 $f_k(x)$」在分类能力上几乎等价于「掌握后验」：分母 $\sum_\ell\pi_\ell f_\ell(x)=\Pr(X=x)$ 对所有 $k$ 都一样，所以**比较两个类的后验等价于比较 $f_k(x)\pi_k$**，这正是第 2 章 (2.23) 的 Bayes 规则的实现方式。

> **基础知识** · 条件风险与 Bayes 规则（预备知识 O5）
> 动作空间 $\{1,\dots,K\}$、损失 $L(g,Y)$、后验 $p_k(x)=\Pr(Y=k\mid X=x)$，条件风险 $R(g\mid x)=\sum_kL(g,k)p_k(x)$，Bayes 分类规则 $g^{\star}(x)=\arg\min_kR(g=k\mid x)$。0-1 损失下 $R(g=k\mid x)=p_k(x)$，故规则是「选后验最大的类」。全风险用塔性质比较：$R(g^{\star})=E[R(g^{\star}\mid X)]\le E[R(g\mid X)]=R(g)$。本章所有「判别函数」$\delta_k$ 都只是这个规则的不同参数化：只要 $\delta_k-\delta_\ell$ 是后验比的**单调变换**（对数、logit），$\arg\max_k\delta_k(x)=\arg\max_k\Pr(G=\mathcal{G}_k\mid X=x)$。

基于类密度的模型有四类：

- **线性与二次判别分析**用高斯密度；
- **更灵活的混合高斯**允许非线性分界线（§6.8）；
- **一般的非参数密度估计**（每类各估一个密度）最灵活（§6.6.2）；
- **朴素 Bayes** 是上一条的一个变体，假设各类密度是各自边缘密度的**乘积**，即类内各输入条件独立（§6.6.3）。

把每个类密度建模为多元高斯：

$$
f_k(x)=\frac{1}{(2\pi)^{p/2}\,|\Sigma_k|^{1/2}}\exp\left\{-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\right\} \eqno{4.8}
$$

这是 $p$ 维正态密度的标准形式（把一维 $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$ 的指数形式配方到多元：$(2\pi)^{p/2}$ 是 $p$ 个 $2\pi$ 的乘积，$|\Sigma_k|^{1/2}$ 是所有方向标准差的乘积，指数是标准化平方距离）。

**线性判别分析 LDA** 对应「各类共用同一个协方差矩阵 $\Sigma_k=\Sigma\ \forall k$」这个特例。关键观察：(4.8) 里的每一项都是「常数 + $x$ 的二次项 + $x$ 的一次项」，所以判别函数的**结构**由 (4.8) 决定，而「线性」还是「二次」只由 $\Sigma_k$ 是否相等决定。下面的 (4.9) 把这一点算出来。

### 4.3.2 对数优势比化简：为什么 LDA 的分界线是平的 {#s-4-3-2}

比较两个类 $k$ 与 $\ell$，只需看**对数比**：

$$
\log\frac{\Pr(G=k\mid X=x)}{\Pr(G=\ell\mid X=x)}
=\log\frac{f_k(x)\pi_k}{f_\ell(x)\pi_\ell}
=\log\frac{\pi_k}{\pi_\ell}-\frac12\log\frac{|\Sigma_k|}{|\Sigma_\ell|}-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)+\frac12(x-\mu_\ell)^\top\Sigma_\ell^{-1}(x-\mu_\ell)
$$

代入 (4.8) 并逐项展开，这是原始形式。**现在施加 LDA 的假设 $\Sigma_k=\Sigma_\ell=\Sigma$**，看哪些项消失：

1. **归一化因子相消**：$|\Sigma_k|=|\Sigma_\ell|$，故 $-\frac12\log\frac{|\Sigma_k|}{|\Sigma_\ell|}=0$。
2. **二次项相消**：令 $d=\mu_k-\mu_\ell$，用配方（预备知识 C2）：

$$
-\tfrac12(x-\mu_k)^\top\Sigma^{-1}(x-\mu_k)+\tfrac12(x-\mu_\ell)^\top\Sigma^{-1}(x-\mu_\ell)
$$

$$
=\tfrac12\Big[(x-\mu_k)^\top\Sigma^{-1}(x-\mu_k)-(x-\mu_\ell)^\top\Sigma^{-1}(x-\mu_\ell)\Big]
$$

逐项展开两个二次型：$x^\top\Sigma^{-1}x$ 两边相同、抵消；剩 $-\mu_k^\top\Sigma^{-1}x+\mu_k^\top\Sigma^{-1}\mu_k+x^\top\Sigma^{-1}\mu_\ell-\mu_\ell^\top\Sigma^{-1}\mu_\ell$；由于 $\Sigma$ 对称故 $\mu_k^\top\Sigma^{-1}x=x^\top\Sigma^{-1}\mu_k$，于是

$$
=\tfrac12\Big[x^\top\Sigma^{-1}(\mu_k-\mu_\ell)-(\mu_k-\mu_\ell)^\top\Sigma^{-1}x\Big]
+\tfrac12\Big[\mu_k^\top\Sigma^{-1}\mu_k-\mu_\ell^\top\Sigma^{-1}\mu_\ell\Big]
$$

$$
=\tfrac12x^\top\Sigma^{-1}(\mu_k-\mu_\ell)-\tfrac12(\mu_k-\mu_\ell)^\top\Sigma^{-1}(\mu_k-\mu_\ell)
$$

（第一项用二次型的对称性 $a^\top Bb=b^\top Ba$；第二项含 $x$ 的部分整体抵消。）

3. 剩下先验比 $\log\frac{\pi_k}{\pi_\ell}$。合起来得到

$$
=\log\frac{\pi_k}{\pi_\ell}-\frac12(\mu_k-\mu_\ell)^\top\Sigma^{-1}(\mu_k-\mu_\ell)+x^\top\Sigma^{-1}(\mu_k-\mu_\ell) \eqno{4.9}
$$

**最后一行是 $x$ 的线性函数。** 所以对数后验之比线性 $\Rightarrow$ 分界线 $\Pr(G=k\mid X=x)=\Pr(G=\ell\mid X=x)$ 即对数比为零，是 $x$ 的超平面。图 4.5 左图的三类数据确实来自三个共用协方差的多元高斯；注意分界线**不是**质心连线的垂直平分线——只有在 $\Sigma=\sigma^2I$（球状）且先验相等时才恰好是垂直平分线。

把 (4.9) 展开成「每个类一个判别函数」的形式（把 $\delta_k$ 定义成「$k$ 对 $K$」的对数比减去常数）：

$$
\delta_k(x)=x^\top\Sigma^{-1}\mu_k-\frac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k \eqno{4.10}
$$

于是规则是 $G(x)=\arg\max_k\delta_k(x)$。对比 (4.9) 与 (4.10)：$\delta_k-\delta_\ell$ 与对数比相差一个与 $x$ 无关的常数 $\frac12(\mu_k+\mu_\ell)^\top\Sigma^{-1}(\mu_k-\mu_\ell)$，故 argmax 与比较对数比完全一致。

**两类的显式规则。** $K=2$ 时 (4.9) 加上 $\log\frac{\pi_1}{\pi_2}=\log\frac{N_2/N}{N_1/N}=\log\frac{N_2}{N_1}$，整理成「判给类 2 当且仅当」的形式：

$$
x^\top\hat\Sigma^{-1}(\hat\mu_2-\hat\mu_1)>\frac12(\hat\mu_2-\hat\mu_1)^\top\hat\Sigma^{-1}(\hat\mu_2-\hat\mu_1)-\log\frac{N_2}{N_1} \eqno{4.11}
$$

**推导**：对数比（类 1 对类 2）为 $\log\frac{N_2}{N_1}+x^\top\Sigma^{-1}(\mu_1-\mu_2)-\frac12(\mu_1-\mu_2)^\top\Sigma^{-1}(\mu_1-\mu_2)$。取对数比 $\ge0$ 即 $\log\frac{N_2}{N_1}-x^\top\Sigma^{-1}(\mu_2-\mu_1)-\frac12(\mu_2-\mu_1)^\top\Sigma^{-1}(\mu_2-\mu_1)\ge0$，整理即得 (4.11)。

**参数的估计。** 高斯的参数未知，用训练数据估计：

$$
\hat\pi_k=\frac{N_k}{N},\qquad \hat\mu_k=\frac{\sum_{g_i=k}x_i}{N_k},\qquad \hat\Sigma=\frac{\sum_{k=1}^{K}\sum_{g_i=k}(x_i-\hat\mu_k)(x_i-\hat\mu_k)^\top}{N-K}
$$

（$\hat\Sigma$ 是合并协方差，分母 $N-K$ 而非 $N$，因为要扣掉 $K$ 个类均值消耗的自由度。）

**一个可复核的小例子。** 取 $p=1$、$\mu_1=0$、$\mu_2=2$、$\sigma^2=1$、$\pi_1=\pi_2=1/2$。由 (4.10)：

$$
\delta_1(x)=x\cdot0-\tfrac12\cdot0+\log\tfrac12=-0.693,\qquad \delta_2(x)=x\cdot2-\tfrac12\cdot4+\log\tfrac12=2x-2.693
$$

令两者相等：$2x-2.693=-0.693$，得 $x=1.0$——正是两个均值的中点，与「等先验、球状协方差时是垂直平分线」的定性结论一致。若 $\sigma^2=4$，$\delta_2=0.5x-0.5-0.693$，相等给出 $0.5x=0.5$，$x=1$ 仍成立——因为两个类共用 $\sigma^2$，先验也相等，各向同性使分界线永远过中点。若先验改成 $\pi_1=0.3,\pi_2=0.7$，则 $\delta_1=-1.204$、$\delta_2=2x-2-0.357$，相等给出 $x=(2.357-1.204)/2=0.577$，**分界线向少数类方向移动**——这是先验项 $\log\pi_k$ 的直接后果。

**LDA 系数与最小二乘系数同向（练习 4.2）。** 这是 $K=2$ 时回归与 LDA 的精确联系，把两章接了起来。把响应编码成 $y_i=1$（类 1）与 $y_i=-1$（类 2），最小二乘目标

$$
\sum_{i=1}^N\big(y_i-\beta_0-x_i^\top\beta\big)^2 \eqno{4.55}
$$

正规方程的前 $p$ 个分量是 $\sum_i x_i y_i=\big(\sum_i x_ix_i^\top\big)\beta$（截距方程自动满足 $\bar y=\sum_i y_i/N$）。

**为什么 (4.56) 成立（把正规方程写开）。** 编码 $y_i=1-\mathbf 1\{g_i=2\}$，则 $\sum_iy_i=N_1-N_2$，正规方程的前 $p$ 个分量是

$$
\sum_{i=1}^Nx_i y_i=\Big(\sum_{i=1}^Nx_ix_i^\top\Big)\beta
$$

移项并把中心化矩阵 $\tilde X^\top\tilde X$ 按类拆开（记 $\tilde x_i=x_i-\bar x$，由 $\sum_i\tilde x_i=0$ 可验证 $\tilde X^\top\tilde X=\sum_{i\in\mathcal M_1}\tilde x_i\tilde x_i^\top+\sum_{i\in\mathcal M_2}\tilde x_i\tilde x_i^\top$，其中 $\mathcal M_k$ 是类 $k$ 的指标集），代入 $\hat\mu_1,\hat\mu_2$ 的定义并整理，得到

$$
\Big[(N-2)\hat\Sigma+N\hat\Sigma_B\Big]\beta=N(\hat\mu_2-\hat\mu_1) \eqno{4.56}
$$

$$
\hat\Sigma_B=\frac{N_1N_2}{N^2}(\hat\mu_2-\hat\mu_1)(\hat\mu_2-\hat\mu_1)^\top
$$

（$\hat\Sigma_B$ 是**类间**协方差，只含一个秩 1 的方向 $\hat\mu_2-\hat\mu_1$；$\hat\Sigma$ 是合并的**类内**协方差，故 $\hat\Sigma+\hat\Sigma_B$ 就是总协方差 $\hat T$。）把 (4.56) 两边同乘 $\hat\Sigma^{-1}$，并注意

$$
\hat\Sigma_B\hat\Sigma^{-1}=\frac{N_1N_2}{N^2}(\hat\mu_2-\hat\mu_1)\underbrace{(\hat\mu_2-\hat\mu_1)^\top\hat\Sigma^{-1}}_{=\,w^\top},\qquad w:=\hat\Sigma^{-1}(\hat\mu_2-\hat\mu_1)
$$

于是

$$
\hat\Sigma^{-1}\hat\beta=\frac{N}{N-2}\Big[w-\frac{N_1N_2}{N}(\hat\mu_2-\hat\mu_1)\underbrace{w^\top w}_{\text{标量}}\Big]
$$

$w$ 与 $\hat\mu_2-\hat\mu_1$ **共线**（后者乘 $\hat\Sigma^{-1}$ 就是 $w$），所以方括号内整体仍 $\parallel w$。而 (4.11) 的判别方向恰为 $w=\hat\Sigma^{-1}(\hat\mu_2-\hat\mu_1)$，故

$$
\hat\Sigma^{-1}\hat\beta\ \propto\ \hat\Sigma^{-1}(\hat\mu_2-\hat\mu_1) \eqno{4.57}
$$

**这就是练习 4.2(c) 的结论**：最小二乘系数 $\hat\beta$ 与 LDA 的方向 $\hat\Sigma^{-1}(\hat\mu_2-\hat\mu_1)$ 成比例。（这条结论对**任何**两类编码都成立——只用到 $y_i\in\{-1,1\}$ 与 $\sum_i y_i$ 与 $\sum_i x_iy_i$ 的关系，不涉及高斯。）所以最小二乘与 LDA 的系数只差一个标量倍数，**方向相同**。但**截距不同**（除非 $N_1=N_2$）：最小二乘的截距是 $\hat\beta_0=\bar y-\bar x^\top\hat\beta$，LDA 的截距含 $-\log\frac{N_2}{N_1}$。练习 4.2(e) 指出「按 $\hat f(x)>0$ 分类」与 LDA 规则在 $N_1\ne N_2$ 时**不是同一条规则**——这正是图 4.14 里最小二乘线（橙）错分一个点的原因。

> **结果**
> (4.11) 的**方向**推导（($\dagger$)）完全不用高斯假设——它对任何两类数据都成立，因此可以把「回归方向 = LDA 方向」用到非高斯数据上。但 (4.11) 的**截距** $-\log\frac{N_2}{N_1}-\frac12(\cdot)$ 用到了先验，必须有高斯假设。所以实务上的做法是：用 (4.11) 的方向，在训练数据上直接选使错分率最小的截点。

**$K>2$ 时 LDA 不等于指示变量回归**，但它没有掩蔽问题。两者的联系要通过「最优打分」（optimal scoring，第 12 章）建立：LDA 等价于「指示矩阵回归后对 $\hat Y^\top Y$ 做特征分解」。

### 4.3.3 二次判别分析 QDA {#s-4-3-3}

如果 $\Sigma_k$ 不相等，(4.9) 里的相消全部失效，特别是含 $x$ 的二次项不再抵消。整理得到**二次判别函数**：

$$
\delta_k(x)=-\frac12\log|\Sigma_k|-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)+\log\pi_k \eqno{4.12}
$$

**把 (4.8) 逐项取对数就是 (4.12)，中间没有跳过东西。** 具体地，

$$
\log f_k(x)+\log\pi_k=-\frac p2\log(2\pi)-\frac12\log|\Sigma_k|-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)+\log\pi_k
$$

第一项 $-\frac p2\log(2\pi)$ 与 $k$ 无关，$\arg\max$ 时可丢；剩下三项即 (4.12)。用二次型展开 $(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)=x^\top\Sigma_k^{-1}x-2x^\top\Sigma_k^{-1}\mu_k+\mu_k^\top\Sigma_k^{-1}\mu_k$（展开时用 $\Sigma_k$ 对称故 $x^\top\Sigma_k^{-1}\mu_k=\mu_k^\top\Sigma_k^{-1}x$），分界线 $\delta_k=\delta_\ell$ 的方程左边是

$$
\frac12\Big[(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)-(x-\mu_\ell)^\top\Sigma_\ell^{-1}(x-\mu_\ell)\Big]+\frac12\log\frac{|\Sigma_\ell|}{|\Sigma_k|}+\log\frac{\pi_k}{\pi_\ell}=0
$$

含 $x^\top\big(\Sigma_k^{-1}-\Sigma_\ell^{-1}\big)x$ 的二次项——除非 $\Sigma_k=\Sigma_\ell$，它不消失。

**数值验证 · 分界线为什么可能是两条。** 取 $p=1$，$\mu_1=0,\mu_2=1$，$\sigma_1^2=1,\sigma_2^2=4$，先验相等。$\delta_1=-\frac12\log(2\pi)-\frac12x^2$，$\delta_2=-\frac12\log(2\pi)-\frac12\log4-\frac{(x-1)^2}{8}$。令相等：

$$
\frac12x^2=\frac12\log4+\frac{(x-1)^2}{8}\ \Longleftrightarrow\ 4x^2=4\log4+(x-1)^2\ \Longleftrightarrow\ 3x^2+2x+1-4\log4=0
$$

根为 $x=\frac{-1\pm\sqrt{1-3(1-4\log4)}}{3}=\frac{-1\pm\sqrt{12\log4-2}}{3}\approx\frac{-1\pm2.14}{3}$，即 $x\approx0.380$ 与 $x\approx-1.05$。**二次方程有两个根**——这就是「QDA 分界线可能不连通/可能多出一条无意义分支」的具体表现。强制 LDA（$\sigma^2=1$）时方程变一次，根唯一于 $x=\frac12$。这个例子顺带说明：QDA 多出来的自由度既能降低偏差，也能制造麻烦。

注意 $-\frac12\log|\Sigma_k|$ 中的 $1/2$：**原书把这一项印成了 $-\log|\Sigma_k|$，漏掉系数 $1/2$**（与 (4.9) 也不一致）。由上面的逐项取对数可知正确形式应为 $-\frac12\log|\Sigma_k|$。漏掉 $1/2$ 相当于把 $\pi_k$ 替换成 $\pi_k^{1/2}$，会改变 argmax，所以必须按正确形式实现。

**参数个数。** LDA 只需 $\delta_k-\delta_K$（$k=1,\dots,K-1$），每个差需要 $p+1$ 个参数，共 $(K-1)(p+1)$ 个；QDA 每个差需要 $\frac{p(p+3)}{2}+1$ 个，共 $(K-1)\left\{\frac{p(p+3)}{2}+1\right\}$ 个。计数时要小心：虽然要估 $K$ 个 $\Sigma_k$，每个有 $\frac{p(p+1)}2$ 个自由元素，但判别规则只用到**差**，自由度少于这个上界（脚注 3 指出：虽然用 $\hat\Sigma$ 去算判别函数，实际只需要它一个 $O(p)$ 规模的函数）。$p$ 大时 QDA 的参数量爆炸，这是它方差大的原因（偏差–方差折中：我们愿意承受线性分界面的偏差来换低方差）。

**QDA 的稳健估计（原文脚注的做法）。** 若每类样本少而 $p$ 大，类内协方差 $\hat\Sigma_k$ 奇异。做法是把所有类的样本**投影到「类间」的低维子空间**（质心张成的 $\le K-1$ 维子空间），在该子空间里估协方差，再在原空间用 $\hat\Sigma_k^{-1}=P(\hat\Sigma_k^{\text{low}})^{-1}P^\top$ 的形式嵌入。图 4.6 用两种办法拟合同一批数据：在扩充的五维多项式空间里用 LDA（左图），与直接用 QDA（右图），差异通常很小——**QDA 是首选，LDA 是方便的替代**。

> **坑**
> (4.12) 的 $-\log|\Sigma_k|$ 缺 $1/2$（原书排印问题）。另外 $\log|\Sigma_k|$ 用特征值算最稳：若 $\hat\Sigma_k=U_kD_kU_k^\top$（$D_k=\mathrm{diag}(d_{k1},\dots,d_{kp})$），则
>
> $$\log|\hat\Sigma_k|=\sum_{\ell=1}^p\log d_{k\ell},\qquad (x-\hat\mu_k)^\top\hat\Sigma_k^{-1}(x-\hat\mu_k)=\big[U_k^\top(x-\hat\mu_k)\big]^\top D_k^{-1}\big[U_k^\top(x-\hat\mu_k)\big]$$
>
> 不要显式求逆（预备知识 N1：条件数被平方）。

### 4.3.4 正则化判别分析 RDA {#s-4-3-4}

Friedman (1989) 提出的折中：让 QDA 的各类协方差向 LDA 的合并协方差收缩。收缩后的协方差有形式

$$
\hat\Sigma_k^{(\alpha)}=\alpha\hat\Sigma_k+(1-\alpha)\hat\Sigma \eqno{4.13}
$$

$\alpha\in[0,1]$：$\alpha=0$ 给 LDA，$\alpha=1$ 给 QDA，中间是连续族。$\alpha$ 用验证集或交叉验证选。图 4.7 的元音数据上，训练与测试误差都随 $\alpha$ 改善，测试误差在 $\alpha=0.9$ 附近出现尖锐上升。类似地可让 $\hat\Sigma$ 本身向**标量**协方差收缩：

$$
\hat\Sigma^{(\gamma)}=\gamma\hat\Sigma+(1-\gamma)\hat\sigma^2I \eqno{4.14}
$$

$\gamma\in[0,1]$。把 (4.13) 中的 $\hat\Sigma$ 换成 $\hat\Sigma^{(\gamma)}$ 就得到二元参数族 $\hat\Sigma_k^{(\alpha,\gamma)}$。

**问题**：把 RDA 的收缩估计量算到底，看 $\alpha$ 怎样进入分界线。记 $S_k=\hat\Sigma_k$、$S=\hat\Sigma$、$\alpha\in[0,1]$。

**第一步：判别函数的形式。** RDA 把 (4.12) 中的 $\Sigma_k$ 换成 $\Sigma_k^{(\alpha)}$，其余项（$\mu_k$、$\pi_k$）不动：

$$
\delta_k^{(\alpha)}(x)=-\frac12\log\big|S_k^{(\alpha)}\big|-\frac12\big(x-\mu_k\big)^\top\big(S_k^{(\alpha)}\big)^{-1}\big(x-\mu_k\big)+\log\pi_k
$$

用 $\big(S_k^{(\alpha)}\big)^{-1}=\alpha S_k^{-1}+(1-\alpha)S^{-1}$，二次型按双线性展开：

$$
\big(x-\mu_k\big)^\top\big(S_k^{(\alpha)}\big)^{-1}\big(x-\mu_k\big)
=\alpha\,(x-\mu_k)^\top S_k^{-1}(x-\mu_k)+(1-\alpha)\,(x-\mu_k)^\top S^{-1}(x-\mu_k)
$$

**这个式子就是 RDA 的全部直觉**：二次型被「按 $\alpha$ 加权插值」，不是「先插值矩阵再求逆」——但注意 $\alpha\Sigma_k^{-1}+(1-\alpha)\Sigma^{-1}$ **不等于** $(\alpha\Sigma_k+(1-\alpha)\Sigma)^{-1}$（除非两者可交换）。矩阵可交换时可交换（$2\times2$ 的显式展开：设 $A=\mathrm{diag}(a_1,a_2)$、$B=\mathrm{diag}(b_1,b_2)$，则 $\frac{a_1}{x}+\frac{b_1}{y}\ne\frac{a_1+b_1}{x+y}$，只在 $a_1b_2=a_2b_1$ 时相等）。这是 RDA 实现里最容易搞错的地方：**要插值的是二次型（Mahalanobis 距离），不是协方差矩阵的逆**。原书 (4.13) 说的是插值 $\Sigma_k$，实践中两者都用，实现 QDA 时通常插值协方差、用特征分解算距离。

**第二步：分界线。** 令 $\delta_k^{(\alpha)}(x)-\delta_\ell^{(\alpha)}(x)=0$，含 $x$ 的部分整理成

$$
$$
\delta_k^{(\alpha)}(x)-\delta_\ell^{(\alpha)}(x)=\Big[x^\top\Big(\sum_{c\in\{k,\ell\}}c_c\big(S_c^{(\alpha)}\big)^{-1}\Big)x-2x^\top\sum_{c}c_c\big(S_c^{(\alpha)}\big)^{-1}\mu_c\Big]-2\sum_{c}c_c\log\pi_c+\frac12\sum_{c}c_c\log\big|S_c^{(\alpha)}\big|
$$
$$

（其中 $c_k=+1$、$c_\ell=-1$；最后一个 $\log|S_c^{(0)}|$ 之后的两项只依赖常数，可以并入阈值。）可见：

- **$\alpha=0$**：所有 $k$ 的 $\big(S_k^{(0)}\big)^{-1}=S^{-1}$，故 $x$ 的二次项在类间差中**完全相消**，分界线退化为 LDA 的线性形式（(4.10)）。
- **$\alpha>0$**：$\big(S_k^{(\alpha)}\big)^{-1}\ne\big(S_\ell^{(\alpha)}\big)^{-1}$，二次项残留，分界线是二次曲线。随着 $\alpha\to1$ 逼近 QDA。
- **$\alpha=1$**：完全退化为 QDA (4.12)。

所以 RDA 的模型族是「LDA 的线性分界线」到「QDA 的二次分界线」的连续插值，而 $\alpha$ 就是插值系数。

**第三步：行列式的显式公式（需要它才能算 $\log|\cdot|$）。** 记 $S^{(\alpha)}=\alpha S_k+(1-\alpha)S$。由行列式的乘积性质与矩阵行列式引理（用 $U=\frac{\alpha}{1-\alpha}S^{-1}S_k$、$V=I$，得 $\det(I+UV)=\det(I+VU)$）：

$$
\big|S_k^{(\alpha)}\big|=(1-\alpha)^p|S|\cdot\det\Big(I+\frac{\alpha}{1-\alpha}S^{-1}S_k\Big)
$$

$$
\log\big|S_k^{(\alpha)}\big|=p\log(1-\alpha)+\log|S|+\sum_{j=1}^p\log\big(1+\tfrac{\alpha}{1-\alpha}\lambda_j\big),\qquad \lambda_j=\text{$S^{-1}S_k$ 的特征值}
$$

对 $p=1$ 直接验证：$\big|\alpha S_k+(1-\alpha)S\big|=\alpha S_k+(1-\alpha)S=(1-\alpha)S\big(1+\frac{\alpha}{1-\alpha}S_k/S\big)$，与公式一致（$\lambda_1=S_k/S$）。$p>1$ 时绝不逐元素相加——$\log\det$ 只对**乘积**成立。

**第四步：可复核的 $2\times2$ 例子。** 设 $S=\begin{pmatrix}1&0\\0&1\end{pmatrix}$、$S_k=\begin{pmatrix}4&0\\0&1\end{pmatrix}$，$\alpha=0.5$。则

$$
S_k^{(\alpha)}=\begin{pmatrix}2.5&0\\0&1\end{pmatrix},\qquad \big(S_k^{(\alpha)}\big)^{-1}=\begin{pmatrix}0.4&0\\0&1\end{pmatrix}
$$

注意 $\alpha S_k^{-1}+(1-\alpha)S^{-1}=0.5\begin{pmatrix}0.25&0\\0&1\end{pmatrix}+0.5I=\begin{pmatrix}0.625&0\\0&1\end{pmatrix}\ne\begin{pmatrix}0.4&0\\0&1\end{pmatrix}$，**验证了第二步里的警告**。取 $\mu_k=(2,0)^\top$，$x=(x_1,x_2)^\top$，二次型为 $0.4(x_1-2)^2+x_2^2$，行列式项 $-\frac12\log2.5$。

**第五步：与岭回归的类比。** 岭回归把 $\hat\beta=(X^\top X)^{-1}X^\top y$ 换成 $(X^\top X+\lambda I)^{-1}X^\top y$，等价于先验 $\beta\sim N(0,\sigma^2/\lambda\cdot I)$——**收缩到 0**；RDA 把 $S_k$ 换成 $\alpha S_k+(1-\alpha)S$，等价于先验 $\mathrm{vec}(S_k)\sim N\big(\mathrm{vec}(S),(1-\alpha)\tau^2\big)$——**收缩到另一个矩阵**（合并协方差）而非 0。这个「收缩到目标而不是收缩到零」的想法在第 5.4 节的岭回归、第 10 章的随机森林、第 18 章的矩阵补全里反复出现。

> **坑** · **收缩强度该用 $\alpha$ 还是 $(\alpha,\gamma)$？**
> (4.13) 收缩的是**类间**的差异（各类协方差 → 合并协方差），(4.14) 收缩的是**合并协方差自身**（$\hat\Sigma\to\sigma^2I$，球状）。两者正交：图 4.7 的元音数据只用 $\alpha$ 就够（$p=10$、$K=11$，合并协方差本身估计得不错），而 $p\gg N$ 时必须同时用 $\gamma$ 才能让 $\hat\Sigma^{(\gamma)}$ 有逆。$\gamma=0$ 的极端形式（$\hat\Sigma^{(0)}=\hat\sigma^2I$）就是「球状 LDA」，第 18 章的高维基因表达数据用的正是它。

---

### 4.3.5 Fisher 判别坐标与降秩 LDA {#s-4-3-5}

LDA 之所以流行，还因为有一个附加限制让我们能看**信息量低维投影下的数据**。$K$ 个类心落在维度 $\le K-1$ 的仿射子空间里（$K$ 个点张成的仿射子空间维数是 $K-1$；又 $K$ 个点满足 $\sum_k(\mu_k-\bar\mu)=0$，故最多 $K-1$ 维）。$p\gg K$ 时这是很大的降维。而且在找最近类心时可以**忽略正交于这个子空间的分量**——它们对每个类的距离贡献相同。所以不妨把白化后的 $X^{\star}$ 只投影到类心张成的子空间 $\mathcal H_{K-1}$ 上比较距离。$K=3$ 时这就能画成二维彩图，且不丢失 LDA 分类所需的任何信息。

**Fisher 判别坐标。** 若 $K>3$，我们想问：是否存在 $L<K-1$ 维子空间 $\mathcal H_L\subseteq\mathcal H_{K-1}$，在某种意义上对 LDA 最优？Fisher 把「最优」定义为「**投影后的类心尽量分散**」（用方差衡量），也就是求**类心自身的主成分子空间**。图 4.4 给出元音数据的一个最优二维子空间（11 个类、10 维，$K-1=p$ 但仍降到了 2 维）；图 4.8 展示另外四对坐标，即**典范（判别）变量**。

**Fisher 完全没有提到高斯。** 他提的问题是：**找线性组合 $Z=a^\top X$，使类间方差相对于类内方差最大**（图 4.9 说明为什么：沿质心连线方向确实让均值分离最大，但投影后重叠严重；把协方差算进来才能找到重叠最小的方向）。

**推导 (4.15)。** $Z=a^\top X$ 的类间方差是 $a^\top Ba$、类内方差是 $a^\top Wa$，其中 $W$ 是类内协方差、$B$ 是**类心矩阵 $M$ 的协方差**。注意总协方差分解

$$
B+W=T,\qquad T=\text{忽略类别信息时 }X\text{ 的总协方差}
$$

（把总体协方差按「类内 + 类间」二项分解，与全方差公式 P2 同构；验证：$\sum_k\pi_k[\Sigma_k+(\mu_k-\mu)(\mu_k-\mu)^\top]$ 按类加权和即得。）于是 Fisher 的问题就是极大化 **Rayleigh 商**：

$$
\underset{a}{\max}\ \frac{a^\top Ba}{a^\top Wa} \eqno{4.15}
$$

$$
\underset{a}{\max}\ a^\top B a \quad\text{s.t.}\ a^\top Wa=1 \eqno{4.16}
$$

**为什么等价（要自己算的一步）。** 记 $R(a)=\frac{a^\top Ba}{a^\top Wa}$。对任意 $c>0$，分子分母同乘 $c$（都是二次型）：$R(ca)=\frac{c^2a^\top Ba}{c^2a^\top Wa}=R(a)$，所以 $R$ 只依赖 $a$ 的**方向**。既然方向不变，就可以沿着 $a$ 缩放直到 $a^\top Wa=1$（$W$ 正定故这个解存在且唯一：缩放系数为 $c=(a^\top Wa)^{-1/2}$），此时 $R(a)=a^\top Ba$。得 (4.16)。

**解是什么：广义特征值问题。** 令 $u=W^{1/2}a$，则 $a^\top Wa=u^\top u$、$a^\top Ba=u^\top W^{-1/2}BW^{-1/2}u$，问题变成 $\max_u\,u^\top\big(W^{1/2}BW^{-1/2}\big)u$ s.t. $u^\top u=1$。由 Rayleigh–Ritz 定理，解 $u$ 是 $W^{1/2}BW^{-1/2}$ 的最大特征向量。注意

$$
W^{1/2}\big(W^{-1}B\big)W^{-1/2}=W^{-1/2}BW^{-1/2}
$$

即 $W^{1/2}BW^{-1/2}$ 与 $W^{-1}B$ **相似**，故有相同的特征值。所以 (4.16) 的最优值是 $W^{-1}B$ 的最大特征值，最优 $a_1=W^{-1/2}u_1$ 满足

$$
W^{-1}Ba_1=\lambda_1a_1\ \Longleftrightarrow\ Ba_1=\lambda_1Wa_1
$$

即「**广义特征值问题** $Ba=\lambda Wa$」本身。练习 4.1 要求「把它变换到标准特征值问题」，上面就是这个变换。

**与主成分步骤的一致性。** 主成分做法是：白化类心 $M^{\star}=MW^{-1/2}$，再对 $B^{\star}=(M^{\star})^\top M^{\star}$（$M^{\star}$ 的协方差）做特征分解 $B^{\star}=V^{\star}D_BV^{\star\top}$，取 $v_\ell^{\star}=v_\ell^{\star}$（第 $\ell$ 列），判别变量为

$$
Z_\ell=v_\ell^\top X,\qquad v_\ell=W^{-1/2}v_\ell^{\star}
$$

**练习 4.1 的结论「最优 $a_1$ 就是 $v_1$」可以这样验证**：$B^{\star}$ 的第 $\ell$ 个特征向量给出的 $v_\ell$ 满足 $Bv_\ell=\lambda_\ell Wv_\ell$（把 $v_\ell=W^{-1/2}v_\ell^{\star}$ 代回去，用 $B^{\star}=W^{-1/2}BW^{-1/2}$），所以它就是 (4.16) 的解。方向是**唯一**确定的（除标量倍数），故取按特征值递减的前 $L$ 个就得到**依次最优**的 $L$ 维子空间。

**类心子空间这一步其实冗余。** 由于 $\sum_k(\mu_k-\bar\mu)=0$，类心张成的子空间维数本就 $\le K-1$；上面第 2 步白化后取 $B^{\star}$ 的特征分解，自动把正交于该子空间的方向的 $B$ 特征值取成 0，所以降维是**同时**完成的。Fisher 的公式因此更直接。

**判别坐标 vs 判别函数。** $v_\ell$（或 $a_\ell$）称为**判别坐标**或**典范变量**（canonical variates），**不要**与判别函数 $\delta_k(x)$ 混淆。另一条推导路线是对指示响应矩阵 $Y$ 与预测矩阵 $X$ 做典范相关分析（§12.5）。

**降秩限制为什么也算高斯分类。** 把 LDA 限制在 $L$ 维子空间上分类，等价于「加了一个附加约束：高斯质心落在 $\mathbb R^p$ 的某个 $L$ 维子空间里」的贝叶斯分类规则。用极大似然拟合并用 Bayes 公式构造后验，得到的规则与上面完全一致（练习 4.8）。图 4.10：元音数据 $K=11$、$p=10$，每个维度都可算训练与测试误差，最优在 2 维；图 4.11 是二维 LDA 方案的决策边界。

**$\log\pi_k$ 修正为什么必要。** 错分率取决于两密度**重叠区域的面积**。$\pi_k$ 相等时（Figure 4.9 隐含）最优切点在投影均值的中点；$\pi_k$ 不等时，把切点朝**小类**方向移能降低错分率（Figure 4.14）。

**一个可复核的小例子。** $p=2$，$K=2$，$\mu_1=(0,0)^\top$、$\mu_2=(3,0)^\top$，$W=\mathrm{diag}(1,4)$（类内方差沿 $x_1$ 为 1、沿 $x_2$ 为 4）。则 $B=\frac14(3,0)^\top(3,0)=\begin{pmatrix}9/4&0\\0&0\end{pmatrix}$。Rayleigh 商 $\frac{9a_1^2/4}{a_1^2+4a_2^2}$ 的最大值在 $a_2=0$ 时取到（值为 $9/4$），最优方向 $a_1$ 沿 $x_1$ 轴。$W^{-1}B=\begin{pmatrix}1&0\\0&1/4\end{pmatrix}\begin{pmatrix}9/4&0\\0&0\end{pmatrix}=\begin{pmatrix}9/4&0\\0&0\end{pmatrix}$，特征值 $\{9/4,0\}$，最大特征向量 $(1,0)^\top$ ✓。**含义**：两类的分离在「方差小」的方向上更有效（Fisher 方向要除以类内方差），这与 LDA 的 $\Sigma^{-1}(\mu_2-\mu_1)$ 一致。

> **坑** · **降秩 LDA 只用于投影，不能改善 LDA 的错分率**
> 若把数据投影到 $L$ 维子空间后再用 LDA，得到的分类规则与在全空间用同一组判别坐标完全一样（因为被丢掉的方向的类心投影为 0）。降秩的真正价值是**可视化**与**降计算量**，而非提升精度。图 4.10 中测试误差随 $L$ 变化是因为每个 $L$ 用的是「前 $L$ 个**依次最优**的方向」，而不是因为投影本身有正则化效应。

---

## 4.4 逻辑回归 {#s-4-4}

<a class="src" href="../esl/ch04-linear-methods-for-classification.html#s-4-4">原文 §4.4</a>

### 4.4.1 模型与 softmax 形式 {#s-4-4-1}

模型想用 $x$ 的线性函数同时表达 $K$ 个后验概率，并保证它们**和为 1**且落在 $[0,1]$。写 $K-1$ 个 log-odds：

$$
\log\frac{\Pr(G=1\mid X=x)}{\Pr(G=K\mid X=x)}=\beta_{10}+\beta_1^\top x
$$

$$
\log\frac{\Pr(G=2\mid X=x)}{\Pr(G=K\mid X=x)}=\beta_{20}+\beta_2^\top x
$$

$$
\log\frac{\Pr(G=K-1\mid X=x)}{\Pr(G=K\mid X=x)}=\beta_{(K-1)0}+\beta_{K-1}^\top x \eqno{4.17}
$$

（用最后一类作分母不是本质的：估计量在换基准类下等变。）

**推导 · 从 (4.17) 到 (4.18)（softmax）。** 记 $e_k=\exp(\beta_{k0}+\beta_k^\top x)$，$k=1,\dots,K-1$。对 (4.17) 两边取指数得 $\frac{\Pr(G=k)}{\Pr(G=K)}=e_k$，故

$$
\Pr(G=k)=\Pr(G=K)\,e_k,\qquad k=1,\dots,K-1
$$

再用归一化 $\sum_{k=1}^{K}\Pr(G=k)=1$：$\Pr(G=K)\big(1+\sum_{\ell=1}^{K-1}e_\ell\big)=1$，即

$$
\Pr(G=k\mid X=x)=\frac{\exp(\beta_{k0}+\beta_k^\top x)}{1+\sum_{\ell=1}^{K-1}\exp(\beta_{\ell0}+\beta_\ell^\top x)},\qquad k=1,\dots,K-1
$$

$$
\Pr(G=K\mid X=x)=\frac{1}{1+\sum_{\ell=1}^{K-1}\exp(\beta_{\ell0}+\beta_\ell^\top x)} \eqno{4.18}
$$

**验证 (4.18) 的三条性质。**

1. **和为 1**：分子之和 $=\sum_{k=1}^{K-1}e_k+1=1+\sum_{\ell=1}^{K-1}e_\ell$，除以同一分母得 1 ✓。
2. **落在 $[0,1]$**：每个 $e_k>0$，故 $0<p_k<1$ ✓。
3. **熵最大**：给定 $x$ 时 $p_k$ 是 $x$ 的函数，故模型关于 $x$ 线性（$p_k$ 对 $x$ 是 sigmoid 型非线性，但**它的 logit 线性**）。这正是 §4.1 说的「单调变换线性」。

**(4.18) 与 (4.17) 的等价性可反向验证。** 若给定满足和为 1 的 $p_1,\dots,p_{K}$，则 $\frac{p_k}{p_K}=\exp(\beta_{k0}+\beta_k^\top x)$，即 logit 等式成立；反之 logit 等式加上归一化唯一确定 $p$。所以 (4.17) 的 $K-1$ 个自由参数与「满足归一化的 $K$ 个概率」的 $K-1$ 个自由度一一对应。

**参照类的选择无关紧要（但要用对）。** 若改用类 1 作分母，新的系数是 $\tilde\beta_{k0}=\beta_{k0}-\beta_{K0}$、$\tilde\beta_k=\beta_k-\beta_K$。因为对任意 $x$

$$
\frac{p_k}{p_1}=\frac{e_k}{e_K}=\frac{e_k/e_K}{1}=e^{\beta_{k0}-\beta_{K0}+(\beta_k-\beta_K)^\top x}
$$

估计量在这个变换下等变。注意 $\tilde\beta_{K-1}$ 会涉及两个系数的差，故**不能**说「换个参照类估计量数值不变」，只能说「模型等价、变换是线性的」。

**$K=2$ 时的特例。** 此时只有一个对数比，$p_1=\dfrac{e^{\beta_0+\beta^\top x}}{1+e^{\beta_0+\beta^\top x}}=\frac{1}{1+e^{-\eta}}$，$\eta=\beta_0+\beta^\top x$——回到 §4.1 的 (4.1)。这个 sigmoid 函数也正是第 2 章 (2.31) 里的 $h_k(x)$，所以「逻辑回归 = 用 sigmoid 基函数的非线性基展开 (2.30) + 二项似然」。它解释了为什么第 2 章说逻辑回归是「指示变量回归的合理替代」：两者都写 $f_\theta(x)=\sum_kh_k(x)\theta_k$，只是 (2.32) 的 RSS 目标被换成了二项对数似然 (4.19)(4.20)。

### 4.4.2 二项情形的对数似然与得分方程 {#s-4-4-2}

**$K$ 类的对数似然。** 用条件似然 $\Pr(G\mid X)$（它完全决定条件分布，故多项分布合适）：

$$
\ell(\theta)=\sum_{i=1}^N\log p_{g_i}(x_i;\theta) \eqno{4.19}
$$

**二类情形的完整化简。** 把 $g_i$ 编码为 $y_i\in\{0,1\}$（$y_i=1\Leftrightarrow g_i=1$），$p_1=p(x;\beta)$、$p_2=1-p(x;\beta)$。由 (4.18) 得 $p(x;\beta)=\dfrac{e^{\beta_0+\beta^\top x}}{1+e^{\beta_0+\beta^\top x}}=\frac{1}{1+e^{-\eta_i}}$，其中 $\eta_i=\beta_0+\beta^\top x_i$（这就补上了第 2 章 (2.31) 里 sigmoid 函数的来源）。

$$
\begin{aligned}
\ell(\beta)&=\sum_{i=1}^N\Big[y_i\log p(x_i;\beta)+(1-y_i)\log\big(1-p(x_i;\beta)\big)\Big]\\
&=\sum_{i=1}^N\Big[y_i\big(\eta_i-\log(1+e^{\eta_i})\big)+(1-y_i)\big(-\log(1+e^{\eta_i})\big)\Big]\\
&=\sum_{i=1}^N\Big[y_i\beta^\top x_i-\log(1+e^{\beta^\top x_i})\Big]
\end{aligned}
$$

**逐步说明**：$\log p=\log\frac{e^\eta}{1+e^\eta}=\eta-\log(1+e^\eta)$；$\log(1-p)=\log\frac{1}{1+e^\eta}=-\log(1+e^\eta)$；两项按 $y_i$ 与 $1-y_i$ 加权后，$\eta_i$ 的系数是 $y_i$，$\log$ 项的系数是 $y_i+(1-y_i)=1$。得

$$
\ell(\beta)=\sum_{i=1}^N\Big[y_i\beta^\top x_i-\log\big(1+e^{\beta^\top x_i}\big)\Big] \eqno{4.20}
$$

（这里假定 $x_i$ 的第一个分量是常数 1，截距吸收在 $\beta=\{\beta_{10},\beta_1\}$ 里。）

**得分方程（一阶条件）。** 逐项求导：$\frac{\partial}{\partial\beta}\big(y_i\beta^\top x_i\big)=y_ix_i$，而 $\frac{\partial}{\partial\beta}\big(\beta^\top x_i\big)=x_i$ 且 $\frac{d}{d\eta}\big[-\log(1+e^\eta)\big]=-\frac{e^\eta}{1+e^\eta}=-p(x_i;\beta)$，故

$$
\frac{\partial\ell(\beta)}{\partial\beta_i}=\sum_{j=1}^N x_{ij}\big(y_j-p(x_j;\beta)\big)=0 \eqno{4.21}
$$

（$i=1,\dots,p+1$。）这是 $p+1$ 个关于 $\beta$ **非线性**的方程（因 $p$ 含 $\exp$）。注意 $x_{i1}=1$，所以第 1 个得分方程是 $\sum_iy_i=\sum_ip(x_i;\beta)$：**期望的类 1 个数等于观测个数**（类 2 同）。这是一个可用来检查数值实现的守恒条件。

**得分方程的力学解释。** (4.21) 是 $p+1$ 个「力平衡」条件：每个系数 $\beta_i$ 上的「拉力」 $x_{ij}(y_j-p_j)$ 之和为零。正系数 $x_{ij}>0$ 时，观测值大于预测值（$y_j-p_j>0$）就往上拉，反之往下拉。第 1 个方程是全局约束（观测的类 1 个数 = 预测的类 1 个数之和），另外 $p$ 个方程决定方向。

**一个手算例子。** 设只有一个预测因子（截距 + 一个特征），$N=3$：$(x,y)=(0,0),(0,1),(1,1)$。则 $X=\begin{pmatrix}1&0\\1&0\\1&1\end{pmatrix}$，$\ell(\beta_0,\beta_1)=\beta_1-\log(1+e^{\beta_0})-\log(1+e^{\beta_0+\beta_1})$。在 $\beta_0=\beta_1=0$ 处 $p_i=0.5$，

$$
\nabla\ell=\begin{pmatrix}x_1^\top(y-p)\\ x_2^\top(y-p)\end{pmatrix}
=\begin{pmatrix}-0.5-0.5+0.5\\ 0+0+0.5\end{pmatrix}=\begin{pmatrix}-0.5\\ 0.5\end{pmatrix}
$$

$W=\mathrm{diag}(0.25,0.25,0.25)=0.25I$，$X^\top WX=0.25\begin{pmatrix}3&1\\1&1\end{pmatrix}$，$\det=0.25\cdot0.25\cdot(3-1)=0.125$，$(X^\top WX)^{-1}=\frac{1}{0.125}\begin{pmatrix}1&-1\\-1&3\end{pmatrix}=\begin{pmatrix}8&-8\\-8&24\end{pmatrix}$。Newton 步

$$
\Delta=\big(X^\top WX\big)^{-1}X^\top(y-p)=\begin{pmatrix}8&-8\\-8&24\end{pmatrix}\begin{pmatrix}-0.5\\0.5\end{pmatrix}=\begin{pmatrix}-8\\20\end{pmatrix}
$$

于是 $\beta^{\rm new}=(0,0)+(-8,20)^\top=(-8,20)^\top$。**这一步跳得很远**，而且此后 $\ell$ 可能下降——这正是需要步长减半的原因。实际代码会加阻尼：试 $\beta+\Delta$，若 $\ell$ 下降则改试 $\beta+\frac12\Delta$，直到 $\ell$ 不再下降。

**存在性与唯一性。** 把 $\ell$ 逐个分量地看：第 $j$ 项的 Hessian 贡献是 $-x_jx_j^\top p_j(1-p_j)$（见 (4.22)），而 $0<p_j<1$，故每个 $x_jx_j^\top p_j(1-p_j)$ 半正定，总 Hessian 半**负**定 ——$\ell$ 是凹函数（严格不凸，因为 $p_j(1-p_j)>0$ 使 $-\ell$ 沿每个 $x_j$ 方向严格凸）。凹函数的极大值集合是凸集；不可分时 $X^\top y\neq0$ 且 $X$ 满秩，可证 $\hat\beta$ 唯一。**可分时**则相反：存在 $\beta_{\rm sep}$ 使所有 $y_ix_i^\top\beta_{\rm sep}>0$，此时 $\ell$ 无上界（沿 $\beta_{\rm sep}$ 方向趋于 0 但达不到），MLE 不存在（练习 4.5）。

> **坑** · **完全分离时 MLE 不存在**
> 若两类被某个超平面**完全可分**（存在 $\beta$ 使所有 $y_ix_i^\top\beta>0$），则每个观测的预测概率都能被推向 0/1，$\sup_\beta\ell(\beta)=0$ 但任何有限 $\beta$ 都达不到，MLE 不存在（系数发散）。这与最小二乘不同：最小二乘永远有唯一解。LDA 在同一数据上仍有良好定义，因为边缘似然不允许这种退化（§4.4.6）。处理办法有三种：加 $L_1$ 或 $L_2$ 惩罚（§4.4.5、4.5），用 Firth 惩罚修正似然，或报告「可分」这一结论本身。

### 4.4.3 Newton–Raphson 与 IRLS {#s-4-4-3}

**海森矩阵。** 由 (4.20) 二次求导（用链式法则，$\frac{dp}{d\eta}=p(1-p)$）：

$$
\frac{\partial^2\ell(\beta)}{\partial\beta\,\partial\beta^\top}=-\sum_{i=1}^N x_ix_i^\top\,p(x_i;\beta)\big[1-p(x_i;\beta)\big] \eqno{4.22}
$$

从 $\beta^{\rm old}$ 出发的一次 Newton 更新：

$$
\beta^{\rm new}=\beta^{\rm old}-\left(\frac{\partial^2\ell}{\partial\beta\,\partial\beta^\top}\right)^{-1}\frac{\partial\ell}{\partial\beta} \eqno{4.23}
$$

**矩阵记号。** 记 $y=(y_1,\dots,y_N)^\top$，$X$ 是 $N\times(p+1)$ 矩阵，$p=(p(x_1;\beta^{\rm old}),\dots,p(x_N;\beta^{\rm old}))^\top$，$W=\mathrm{diag}\big(p_i(1-p_i)\big)$。则

$$
\frac{\partial\ell(\beta)}{\partial\beta^\top}=X^\top(y-p) \eqno{4.24}
$$

$$
\frac{\partial^2\ell(\beta)}{\partial\beta\,\partial\beta^\top}=-X^\top WX \eqno{4.25}
$$

（把 (4.21)(4.22) 的求和写成矩阵乘积：$\sum_jx_j(y_j-p_j)=X^\top(y-p)$、$\sum_jx_jx_j^\top p_j(1-p_j)=X^\top\mathrm{diag}(p_j(1-p_j))X$。$X^\top WX$ 对称半正定（$v^\top X^\top WXv=(Xv)^\top W(Xv)\ge0$），在 $X$ 满秩且每个 $p_j\in(0,1)$ 时**正定**，故 (4.23) 的逆总存在。）

**Newton 步。** 把 (4.23) 代入 (4.24)(4.25)：

$$
\beta^{\rm new}=\beta^{\rm old}+\big(X^\top WX\big)^{-1}X^\top(y-p)
$$

$$
=\big(X^\top WX\big)^{-1}X^\top W\Big[X\beta^{\rm old}+W^{-1}(y-p)\Big]=\big(X^\top WX\big)^{-1}X^\top Wz \eqno{4.26}
$$

**逐步说明**（不能跳的一步是 $\beta^{\rm old}$ 怎么被「吸」进逆矩阵里）：

$$
\begin{aligned}
\beta^{\rm new}&=\beta^{\rm old}+\big(X^\top WX\big)^{-1}X^\top(y-p)\\
&=\big(X^\top WX\big)^{-1}\big(X^\top WX\big)\beta^{\rm old}+\big(X^\top WX\big)^{-1}X^\top(y-p)\\
&=\big(X^\top WX\big)^{-1}\Big[\big(X^\top WX\big)\beta^{\rm old}+X^\top(y-p)\Big]\\
&=\big(X^\top WX\big)^{-1}X^\top\Big[X\beta^{\rm old}+W^{-1}(y-p)\Big]
\end{aligned}
$$

第二步用 $A^{-1}A=I$（把 $\beta^{\rm old}$ 写成 $A^{-1}A\beta$）；第三步用 $(A^{-1}B+C)A^{-1}=A^{-1}BA^{-1}+A^{-1}$ 的取反形式；第四步把 $W^{-1}W$ 凑成 $I$（对角阵可逆）。代入 (4.27) 的定义即得 (4.26)：

$$
z=X\beta^{\rm old}+W^{-1}(y-p) \eqno{4.27}
$$

$z$ 有时叫**调整后的响应**（adjusted response）。这个迭代重复进行（每次 $p$ 变、$W$ 与 $z$ 都变），称为**迭代重加权最小二乘**（IRLS），因为每次迭代都在解一个加权最小二乘问题：

$$
\beta^{\rm new}\ \leftarrow\ \underset{\beta}{\mathrm{argmin}}\ \big(z-X\beta\big)^\top W\big(z-X\beta\big) \eqno{4.28}
$$

**为什么这真是一步最小二乘（要验证目标函数的极小点就是 Newton 步）。** 加权最小二乘目标 $Q(\beta)=(z-X\beta)^\top W(z-X\beta)$ 的梯度为 $\nabla_\beta Q=-2X^\top Wz+2X^\top WX\beta$，海森为 $2X^\top WX$ 半正定（$W$ 对角半正定），故 $Q$ 凸、驻点即唯一极小。令梯度为零：

$$
X^\top WX\beta=X^\top Wz\ \Longrightarrow\ \beta=\big(X^\top WX\big)^{-1}X^\top Wz
$$

与 (4.26) 的最后一行**完全一致**。这就是 IRLS 名称的来源。

**$W$ 的意义。** $w_j=p_j(1-p_j)=\frac{e^{\eta_j}}{(1+e^{\eta_j})^2}$ 在 $p_j=0.5$ 时最大（$\frac14$），在 $p_j\to0$ 或 $1$ 时趋于 0。所以 IRLS **自动降低已经分对的点的权重**、加大边界附近的点的权重——「逻辑回归只看边界附近的数据」这句话的严格来源，也解释了它在 (4.29) 的加权残差里为什么起 $\hat p_i(1-\hat p_i)$ 的作用。

**凸性与收敛。** $\ell$ 凹（(4.25) 半负定）故 Newton 方向是上升方向；但 Newton 步可能**过冲**导致 $\ell$ 下降，**步长减半**（试 $\beta+\Delta$，若 $\ell$ 下降则试 $\beta+\frac12\Delta$，如此继续）可保证收敛。$\beta=0$ 是常用初值（此时 $p_i=0.5$、$W=\frac14I$，第一步退化成一次普通最小二乘 $\hat\beta=\frac14(X^\top X)^{-1}X^\top(\frac12\mathbf 1+y)$，很温和）。$K\ge3$ 时 Newton 也能写成 IRLS，但每个观测的权是**非对角**矩阵，无法解耦成 $K-1$ 个独立的一元加权最小二乘，只能用一般数值方法（练习 4.4）或坐标下降（§3.8.6、§4.4.5）。$N$ 与 $p$ 都很大时用 `glmnet`（Friedman et al., 2010），它同时支持正则化与非正则化。

### 4.4.4 二次近似、Wald 检验与似然比检验 {#s-4-4-4}

极大似然参数 $\hat\beta$ 满足一个**自洽关系**：它们是一次加权最小二乘拟合的系数，响应为

$$
z_i=x_i^\top\hat\beta+\frac{y_i-\hat p_i}{\hat p_i(1-\hat p_i)} \eqno{4.29}
$$

权为 $w_i=\hat p_i(1-\hat p_i)$，二者都依赖 $\hat\beta$（这正是 (4.27) 在 $\beta=\hat\beta$ 处的取值）。除了提供算法，这个与最小二乘的联系还有更多用处：

- **加权残差平方和是 Pearson $\chi^2$ 统计量**：
  $$
  \sum_{i=1}^N\frac{\big(y_i-\hat p_i\big)^2}{\hat p_i\big(1-\hat p_i\big)} \eqno{4.30}
  $$
  它是偏差（deviance）的二次近似。注意「deviance」的定义是 $-2\times$（对数似然），故它与最小二乘的 RSS 是同一角色的量。
- **渐近似然理论**：若模型正确，$\hat\beta$ 一致；中心极限定理给出 $\hat\beta\overset{d}{\to}N\big(\beta,(X^\top WX)^{-1}\big)$。把 $\hat\beta$ 代入指数族的对数配分函数二阶导即可得 Fisher 信息 $I(\beta)=X^\top WX$（预备知识 P3）——这与 (4.25) 的 $-\nabla^2\ell$ 完全吻合。
- **检验**：Wald 检验与 Rao（score）检验都不需要重新迭代，只是在当前模型的加权最小二乘拟合上加/减一项。

**似然比置信集与 score 检验的等价（完整推导）。** 这是本章最需要自己推的一段，三步。

**第一步：似然比统计量。** 对参数子集 $\Theta_0$（例如 $\beta_2=0$，或 $\beta_2,\dots,\beta_5=0$），检验 $H_0:\theta\in\Theta_0$。似然比统计量是「受限模型的最大似然」与「完全模型的最大似然」之差的两倍：

$$
G=2\Big[\ell(\hat\theta)-\ell(\tilde\theta)\Big]
$$

其中 $\tilde\theta=\arg\max_{\theta\in\Theta_0}\ell(\theta)$。$G\ge0$（受限空间更小，似然不会更大），$H_0$ 成立时渐近 $\chi^2_r$（$r$ 是被限制参数的个数；Wilks 定理）。

**第二步：二阶展开把它变成二次型。** 在 $\tilde\theta$ 处对 $\ell$ 做 Taylor 展开：

$$
\ell(\hat\theta)-\ell(\tilde\theta)=\underbrace{\big(\nabla_\beta\ell\big)^\top_{\tilde\theta}(\hat\theta-\tilde\theta)}_{=0\ \text{因为 }\hat\theta\text{ 满足得分方程 (4.21)}}=\frac12(\hat\theta-\tilde\theta)^\top\nabla^2_\beta\ell\big|_{\tilde\theta}(\hat\theta-\tilde\theta)+\text{三阶项}
$$

$$
G\ \approx\ (\hat\theta-\tilde\theta)^\top I(\tilde\theta)(\hat\theta-\tilde\theta),\qquad I(\theta):=-E\Big[\nabla^2_\beta\ell(\theta)\Big]=X^\top WX \ \text{（由 (4.25)，}\mathrm{Var}(y_i)=p_i(1-p_i)\text{）}
$$

**推导 Fisher 信息**：由 (4.22)(4.24) 的得分与海森，

$$
E\Big[\frac{\partial\ell}{\partial\beta}\frac{\partial\ell}{\partial\beta^\top}\Big]=E\big[X^\top(y-p)(y-p)^\top X\big]=X^\top\mathrm{diag}\big(\mathrm{Var}(y_i)\big)X=X^\top WX
$$

其中 $\mathrm{Var}(y_i)=p_i(1-p_i)$，正是 §4.6 的「二项方差」。且由**期望与二阶导可交换**（对数配分函数二阶导等于 $T(Y)$ 的方差，预备知识 P3）有 $I(\theta)=-\nabla^2_\beta\ell$，与 (4.25) 一致。

似然比置信集（水平 $1-\alpha$）就是 $\{\theta:2[\ell(\hat\theta)-\ell(\theta)]\le\chi^2_{r,\alpha}\}$，用二阶展开后变成一个以 $\hat\theta$ 为中心的**椭球**：

$$
\{\theta:\ (\hat\theta-\theta)^\top I(\theta)(\hat\theta-\theta)\ \le\ \chi^2_{r,\alpha}\}
$$

这与 Wald 置信区间（用 $I(\hat\theta)$）和 score 置信区间（用 $I(\tilde\theta)$）的差别只在于 Fisher 信息取在哪一点。

**第三步：score 检验为什么不重新拟合。** 把上面三个椭球都写成「$z$ 统计量的平方 $\le$ 阈值」的形式。用 $I(\theta)=(\hat\theta-\theta)^\top$ 的形式做统一的二次型完成平方。对 Wald：令 $u=\hat\theta-\theta$，$u^\top I(\hat\theta)u\le\chi^2_{r,\alpha}$ 的边界是 $I(\hat\theta)^{1/2}u$ 的球。类似地把 score 统计量写成

$$
\tilde z=\big(\hat\theta-\tilde\theta\big)^\top I(\tilde\theta)\big(\hat\theta-\tilde\theta\big)\overset{d}{\to}\chi^2_r
$$

**与似然比的联系**：$G$ 与 $\tilde z$ 都是同一个似然曲面在两种不同方向的曲率度量。当 $\hat\theta$ 与 $\tilde\theta$ **相距不远**（即 $H_0$ 接近被拒绝），$I(\hat\theta)\approx I(\tilde\theta)\approx I(\tilde\theta^{\star})$，三者数值上几乎相同——这就是「score ≈ 似然比 ≈ Wald」在大样本下的原因。当 $H_0$ 离数据很远时三者差异显著（Haik–Israel 的经典反例）。

**Wald 检验的具体形式**（表 4.2 用它）。单个系数 $\beta_j$ 的 $H_0:\beta_j=0$：

$$
z_j=\frac{\hat\beta_j}{\mathrm{se}(\hat\beta_j)},\qquad \mathrm{se}(\hat\beta_j)=\sqrt{\big[I(\hat\beta)^{-1}\big]_{jj}}=\sqrt{\big[(X^\top WX)^{-1}\big]_{jj}}
$$

渐近 $N(0,1)$。表 4.2 中 tobacco 的系数 $0.080$、标准误 $0.026$、$Z=3.034$（$0.080/0.026=3.077$，与表中 3.034 的微小差别来自用精确 Fisher 信息或四舍五入）。$|z|>1.96$ 即 5% 显著。多个系数时 $(\hat\theta-\theta_0)^\top I(\hat\theta)(\hat\theta-\theta_0)\overset{d}{\to}\chi^2_r$。

**score 检验的具体形式**（加变量时用）。设 $R$ 是 $(p+1)\times r$ 的「排除矩阵」，取出待检验的 $r$ 个系数。记 $\hat\beta_{\rm ex}=R\hat\beta$，其协方差由 $I(\hat\beta)^{-1}$ 的相应子块给出。检验统计量是「受限模型处未用完的得分」，用噪声尺度标准化后 $\tilde z\overset{d}{\to}\chi^2_r$。**关键性质**：它只用到当前模型的 $\hat\beta$、$X$、$W$，所以加一个变量只需一次矩阵运算，不必重新迭代。R `glm` 的 `anova(..., test="Chisq")` 就是它。

**精确检验与单变量精确偏差。** 若只有一个分类预测因子，可条件于每组的成功数把数据压成 $2\times2$ 列联表，用 Fisher 精确检验或 Cochran–Armitage 趋势检验（后者的统计量是 $\sum_jx_j(y_j-\hat p_j)$，即得分 (4.21) 在单变量情形的**未标准化**版本）。单变量模型的**精确二项偏差**为

$$
D=2\Big[\sum_{j}y_j\log\frac{y_j}{\hat p_j}+(n_j-y_j)\log\frac{n_j-y_j}{n_j-\hat p_j}\Big]
$$

其中 $j$ 取遍不同的预测值、$n_j$ 是该值处的样本数、$\hat p_j$ 由模型给出。它是 (4.30) 的改进：用精确二项似然的**二阶 Taylor 展开**代替 Pearson $\chi^2$。当事件稀有（$\hat p_j$ 接近 0 或 1）时两者差异明显，因为 $\frac{1}{\hat p(1-\hat p)}$ 这种权重在极端处爆炸——这与第 2 章 (2.23) 的 Bayes 率是同一个「稀有事件」问题。R 的 `glm` 的 `deviance` 对象返回的正是这个量。

> **坑** · 类别不平衡
> 当某类只占 $1\%$ 时，似然的主导项是多数类，少数类的影响被淹没；反过来加权（用 $w_i$ 放大少数类）会让少数类主导而多数类漏掉。$\hat\beta$ 只对**拟合优度**敏感（用似然比检验时），因为似然比检验把权重当成已知的先验修正。实务上：报告 AUC（对不平衡稳健）而不是错分率，或用 odds ratio 解释系数（$\exp(\beta_j)$ 是「$x_j$ 增加 1 单位时优势比的变化倍数」，与类别比例无关）。

### 4.4.5 $L_1$ 正则化逻辑回归 {#s-4-4-5}

把 lasso 的 $L_1$ 惩罚加到 (4.20) 上（通常不惩罚截距，并先标准化预测因子）：

$$
\underset{\beta_0,\beta}{\max}\ \sum_{i=1}^N\Big[y_i\big(\beta_0+\beta^\top x_i\big)-\log\big(1+e^{\beta_0+\beta_i^\top x_i}\big)\Big]-\lambda\sum_{j=1}^p|\beta_j| \eqno{4.31}
$$

**推导 · 坐标下降 + KKT。** 固定其他坐标，对 $\beta_j$ 求一阶条件（截距 $\beta_0$ 与其他 $\beta$ 固定）：

$$
\frac{\partial}{\partial\beta_j}\Big[\sum_iy_i(\beta_0+\beta^\top x_i)-\sum_i\log(1+e^{\beta_0+\beta^\top x_i})\Big]=x_j^\top(y-p)
$$

惩罚项的贡献是 $-\lambda\,\mathrm{sign}(\beta_j)$。令总导数为零（KKT 平稳性，预备知识 O2）：

$$
x_j^\top(y-p)=\lambda\,\mathrm{sign}(\beta_j) \eqno{4.32}
$$

这正是 lasso 在最小二乘下的 KKT 条件（预备知识 O4 里 $\beta_j\leftarrow\frac{1}{N\lVert x_j\rVert^2}(x_j^\top r-N\lambda)$ 的平衡版本）的推广：**活跃变量在「与残差的一般化相关」上被等同**。写成坐标更新的形式：

$$
\beta_j\leftarrow S\left(\frac{1}{N}x_j^\top r,\ \lambda\right)-\frac{\lambda}{N},\qquad S(u,\lambda)=\begin{cases}u,&|u|\le\lambda\\ u-\lambda\,\mathrm{sign}(u),&|u|>\lambda\end{cases}
$$

（其中 $r$ 是当前的部分残差，用的是与 (4.29) 相同的加权线性化。）

**为什么 (4.31) 的解唯一。** 目标等价于**极小化**凸函数

$$
-\sum_{i=1}^N\Big[y_i\beta^\top x_i-\log\big(1+e^{\beta^\top x_i}\big)\Big]+\lambda\lVert\beta\rVert_1
$$

第一项是 $-\ell$，由 (4.25) 知其凸（$X^\top WX$ 半正定且 $W$ 的对角元 $\ge\frac14>0$），第二项显然凸，故整体凸。Slater 条件成立（取 $\beta$ 足够大即可使 KKT 的不等式严格），故 KKT 充分（预备知识 O2）：

- **原始可行 + 对偶可行**：$\lambda_j\ge0$；
- **互补松弛**：$\lambda_j|\beta_j|=0$；
- **平稳性**：$x_j^\top(y-p)-\lambda\,\mathrm{sign}(\beta_j)+\lambda_j=0$。

**$\beta_j=0$ 时的额外条件。** 在 $\beta_j=0$ 处 $\mathrm{sign}$ 无定义（$-\log(1+e^\eta)$ 的二阶导存在但一阶导对 $\beta_j$ 的导在 0 处左右极限为 $\mp\frac12x_j^\top p$）。用次梯度代替：$x_j^\top(y-p)\in[-\lambda,\lambda]$，即

$$
\Big|\tfrac1N x_j^\top r\Big|\le\lambda
$$

（这正是软阈值算子把 $\frac1Nx_j^\top r$ 压到 0 的条件，预备知识 O4 的「零阈值条件」）。所以 (4.32) 加上这个条件就是完整的一阶最优刻画。

**因为 $\ell$ 不可微，不能直接用 Newton**，但可以用「加权 lasso」的二次近似反复迭代（把 $\lVert\beta\rVert_1$ 留在原处、只对光滑部分做 (4.23) 的一次 Newton 步），或用带 predictor–corrector 的凸优化（Park–Hastie 的 `glmpath`）精确找出活跃集切换处的 $\lambda$ 值。图 4.13 给出南非心脏病数据的 $L_1$ 系数路径（横轴 $\lVert\beta(\lambda)\rVert_1$、纵轴系数，所有预测因子标准化到单位方差）：这些轮廓**看起来几乎线性**，但在别的例子上曲率会明显。图中的**竖线**标出活跃集（非零系数集合）切换的 $\lambda$ 值。

$L_1$ 路径算法（LAR）在这里比最小二乘难：最小二乘的系数轮廓是**分段线性**（因为解是分段仿射的），逻辑回归的轮廓只是**分段光滑**。不过用二次近似仍能取得进展。坐标下降（§3.8.6）是在 $\lambda$ 网格上算路径的最有效方法；`glmnet` 能处理 $N$ 或 $p$ 极大的问题，并可利用 $X$ 的稀疏性（§18.4）。path）精确找出活跃集切换处的 $\lambda$ 值。$L_1$ 路径算法（LAR）在这里比最小二乘难，因为系数轮廓是**分段光滑**而非分段的线性。

### 4.4.6 逻辑回归还是 LDA？ {#s-4-4-6}

§4.3 的 LDA 也给出 $k$ 与 $K$ 之间 log-odds 的线性形式：

$$
\log\frac{\Pr(G=k\mid X=x)}{\Pr(G=K\mid X=x)}\,\frac{\pi_k}{\pi_K}
-\frac12(\mu_k+\mu_K)^\top\Sigma^{-1}(\mu_k-\mu_K)
+x^\top\Sigma^{-1}(\mu_k-\mu_K)=\alpha_{k0}+\alpha_k^\top x \eqno{4.33}
$$

（把 (4.9) 里的 $\mu_\ell$ 换成 $\mu_K$ 并展开即得。）逻辑回归按定义也有线性 logit：

$$
\log\frac{\Pr(G=k\mid X=x)}{\Pr(G=K\mid X=x)}=\beta_{k0}+\beta_k^\top x \eqno{4.34}
$$

**(4.33) 与 (4.9) 是同一个式子（只需换 $\ell=K$）。** 检查：把 (4.33) 中的常数项拆成 $\log\frac{\pi_k}{\pi_K}-\frac12(\mu_k+\mu_K)^\top\Sigma^{-1}(\mu_k-\mu_K)$，剩下 $\alpha_{k0}$；含 $x$ 的项 $x^\top\Sigma^{-1}(\mu_k-\mu_K)$ 就是 $\alpha_k^\top x$。所以

$$
\alpha_k=\Sigma^{-1}(\mu_k-\mu_K),\qquad \alpha_{k0}=\log\frac{\pi_k}{\pi_K}-\frac12(\mu_k+\mu_K)^\top\Sigma^{-1}(\mu_k-\mu_K)
$$

**形式完全一样，差别只在系数怎么估。** 逻辑回归更一般（假设更少）。写联合密度：

$$
\Pr(X,G=k)=\Pr(X)\Pr(G=k\mid X) \eqno{4.35}
$$

两种模型里右边的第二项都是 logit 线性的：

$$
p_k=\frac{\exp(\beta_{k0}+\beta_k^\top x)}{1+\sum_{\ell=1}^{K-1}\exp(\beta_{\ell0}+\beta_\ell^\top x)} \eqno{4.36}
$$

（最后一类作参照类，任意选择。）

**逻辑回归把边缘密度 $\Pr(X)$ 当作完全任意**，用经验分布函数（在每个观测上放 $1/N$ 的质量）非参数地估计它，只极大化条件似然。**LDA 则极大化基于联合密度的完全似然**：

$$
\Pr(X,G=k)=\phi(X;\mu_k,\Sigma)\,\pi_k \eqno{4.37}
$$

标准正态理论给出 §4.3 的 $\hat\mu_k,\hat\Sigma,\hat\pi_k$；因为逻辑形式的线性参数是高斯参数的函数，把对应估计代入就得到它们的 MLE。但此时边缘密度也参与其中，它是一个混合密度：

$$
\Pr(X)=\sum_{k=1}^K\pi_k\,\phi(X;\mu_k,\Sigma) \eqno{4.38}
$$

**边缘项的作用。** 它是一份「正则化」，要求类密度从边缘视角看是可见的。具体效果有四条，都能从 (4.38) 看出：

- **效率**：多了一条关于参数的信息，渐近损失约 $30\%$ 的错误率（Efron 1975）；换句话说，多 $30\%$ 的数据才能让条件似然追平。
- **利用无标签数据**：由 (4.38)，即使**没有类别标签**的观测也贡献了关于 $\mu_k,\Sigma$ 的信息（它必须是 $K$ 个高斯的混合）。标注昂贵而原始特征便宜时，这条很有价值。
- **不稳健**：同一原因也带来坏处——远离分界线的观测（逻辑回归给它们小权重）在估计合并协方差时起了作用，所以 LDA 对离群点不稳健。
- **可分性**：可分时条件 MLE 未定义（§4.4.2），而 LDA 仍良好定义，因为边缘似然 (4.38) 不允许这种退化。

**边际似然就是一个正则项。** 从 (4.38) 的形式看，要求「$\sum_k\pi_k\phi(x;\mu_k,\Sigma)$ 能拟合观测到的 $X$ 的分布」相当于一个对参数的一致性约束：若 $\mu_k$ 取得离谱，边缘密度会把质量放在没有数据的地方，似然下降。

**一个可算的例子。** 设 $p=1$、两类完全可分（类 1 的 $x\le0$、类 2 的 $x\ge1$）。逻辑回归的 MLE：$\ell$ 沿 $\beta_1\to\infty$（$\beta_0=0$）上升，$p_2(x)\to1$ 对所有 $x\ge0$，但对 $x<0$ 的类 1 点 $p_2\to0$；取 $\beta_1$ 越大，$\ell$ 越接近 0 但永不达到——**MLE 不存在**。LDA：$S=\text{var}(x)>0$ 有限，$\mu_2-\mu_1=1$，判别方向 $1/S$ 有限、截距 $-\frac{1}{2S}$，系数完全良好定义。差别来自哪里？逻辑回归只能「把概率推得更极端」，而 LDA 的 $\Sigma^{-1}(\mu_k-\mu_\ell)$ 被 $S$ **正则化**（协方差里含了类内散布）；边缘似然 (4.38) 也强制 $\Sigma$ 反映边缘散布，故不允许退化。

**实务建议。** 实践中假设很少完全正确，且常常有定性预测因子。经验法则是：**逻辑回归更稳健、假设更少，是更安全的选择**；即便在用 LDA 不合适的情况下（如定性预测因子），两者结果通常也很接近。表 4.1 的元音数据上逻辑回归测试误差 $0.51$，优于 LDA 的 $0.56$ 与指示变量回归的 $0.67$——三个模型**形式相同**（线性 logit），差别全在拟合方式上。

---

## 4.5 分离超平面 {#s-4-5}

<a class="src" href="../esl/ch04-linear-methods-for-classification.html#s-4-5">原文 §4.5</a>

### 4.5.1 线性代数回顾与感知机 {#s-4-5-1}

图 4.14 的数据被一条线性边界分开。把 $-1/1$ 响应 $Y$ 对 $X$ 回归（含截距）给出的线是

$$
\big\{x:\ \hat\beta_0+\hat\beta_1^\top x_1+\hat\beta_2^\top x_2=0\big\} \eqno{4.39}
$$

它错分一个点。由 §4.3.2 的 (4.56)(4.57)，这与 LDA 的边界一致。计算 $\beta^\top x+\beta_0$ 并取符号的分类器（1950 年代工程文献称为**感知机**，Rosenblatt 1958）由此而来。

**超平面的三条性质**（图 4.15）。设 $L=\{x:\beta_0+\beta^\top x=0\}$。

1. 对 $x_1,x_2\in L$，$\beta^\top(x_1-x_2)=0$，故 $\beta^{\star}=\beta/\lVert\beta\rVert$ 是 $L$ 的**法向量**。
2. 对 $x_0\in L$，$\beta^\top x_0=-\beta_0$。
3. 任意 $x$ 到 $L$ 的**有符号距离**为
   $$
   \frac{\beta^{*\top}(x-x_0)}{\lVert\beta\rVert}=\frac{\beta^\top x+\beta_0}{\lVert\beta\rVert}=\frac{f(x)}{\lVert f'(x)\rVert} \eqno{4.40}
   $$
   **推导**：性质 2 给出 $\beta_0=-\beta^\top x_0$，代入得 $\beta^\top x+\beta_0=\beta^\top(x-x_0)$；再除以 $\lVert\beta\rVert=\lVert\beta^{\star}\rVert$。所以 $f(x)$ 与到超平面的有符号距离**成正比**，比例是 $1/\lVert\beta\rVert$。这就是「把 $f$ 归一化」在几何上的意义。

**感知机学习算法。** 目标是最小化错分点到分界线的（带符号）距离：

$$
D(\beta,\beta_0)=-\sum_{i\in\mathcal M}y_i\big(x_i^\top\beta+\beta_0\big) \eqno{4.41}
$$

$\mathcal M$ 是被错分的点集。若 $y_i=1$ 被错分则 $x_i^\top\beta+\beta_0<0$，若 $y_i=-1$ 被错分则 $>0$，所以 $-y_i(x_i^\top\beta+\beta_0)>0$，即每项是「距离」的正倍数（由 (4.40)），整体非负。**固定 $\mathcal M$ 时的梯度**：

$$
\frac{\partial D(\beta,\beta_0)}{\partial\beta}=\sum_{i\in\mathcal M}(-y_ix_i),\qquad \frac{\partial D}{\partial\beta_0}=\sum_{i\in\mathcal M}(-y_i) \eqno{4.42}
$$

$$
\frac{\partial D(\beta,\beta_0)}{\partial\beta_0}=\sum_{i\in\mathcal M}(-y_i) \eqno{4.43}
$$

算法用**随机梯度下降**：每访问一个错分点就更新一次（而不是先求和再走一步）：

$$
(\beta,\beta_0)\ \leftarrow\ (\beta,\beta_0)+\rho\,(y_ix_i,\ y_i) \eqno{4.44}
$$

$\rho$ 是学习率，此处取 1 不失一般性（因为更新方向与 $\rho$ 成正比，只影响步长不影响方向）。**若线性可分，可以证明算法在有限步内收敛到某个分离超平面**（练习 4.6；收敛性证明要用到 $\lVert\beta_{\rm new}-\beta_{\rm sep}\rVert_2\le\lVert\beta_{\rm old}-\beta_{\rm sep}\rVert_2-1$ 的收缩不等式）。图 4.14 给出同一玩具问题的两个解，各从一个随机初值出发。

**三个问题**（Ripley 1996 的总结）：

- 可分时解**不唯一**，找到哪个取决于初值；
- 「有限」步数可能极大（间隙越小步数越多）；
- **不可分时不收敛**，会进入很长的循环。

**第二个问题可以定量。** 练习 4.6 给出收敛性的证明框架。设 $x_i^{\star}=(x_i,1)^\top$、$\beta=(\beta_1,\beta_0)$、$z_i=x_i^{\star}/\lVert x_i^{\star}\rVert$，可分性蕴含存在 $\beta_{\rm sep}$ 使 $y_i\beta_{\rm sep}^\top z_i\ge1$。若当前 $\beta^{\rm old}$ 误分类了 $z_i$，则更新 $\beta^{\rm new}=\beta^{\rm old}+y_iz_i$，并且

$$
\lVert\beta^{\rm new}-\beta_{\rm sep}\rVert_2^2\le\lVert\beta^{\rm old}-\beta_{\rm sep}\rVert_2^2-1
$$

**验证这一步**（关键的不等式）：记 $\delta=\beta^{\rm old}-\beta_{\rm sep}$，则

$$
\lVert\delta+y_iz_i\rVert^2-\lVert\delta\rVert^2=2y_iz_i^\top\delta+\lVert z_i\rVert^2=2y_iz_i^\top(\beta^{\rm old}-\beta_{\rm sep})+1
$$

由「$z_i$ 被误分类」知 $y_iz_i^\top\beta^{\rm old}<0$；又 $y_i\beta_{\rm sep}^\top z_i\ge1$，故 $y_iz_i^\top\beta_{\rm sep}\ge1$，于是 $y_iz_i^\top(\beta^{\rm old}-\beta_{\rm sep})\le-1$。代回得差 $\le-2+1=-1$ ✓。

所以每步至少把 $\lVert\beta-\beta_{\rm sep}\rVert^2$ 减少 1，步数不超过 $\lVert\beta_{\rm start}-\beta_{\rm sep}\rVert^2$。**间隙越小，这个界越松**——这就是「间隔小则步数多」的严格说法。

后两个问题可以通过在扩充空间（大量基变换）里找超平面缓解，但**过拟合会导致人为的分离**。第一个问题有一个优雅的解法：给超平面加约束，即下一节。

### 4.5.2 最优分离超平面与 Wolfe 对偶 {#s-4-5-2}

**最优分离超平面**（Vapnik 1996）把两类分开，并**最大化到最近点的距离**。它给出唯一解，且训练数据上的大间隔会改善测试表现。

$$
\underset{\beta,\beta_0,\lVert\beta\rVert=1}{\max}\ M \qquad\text{s.t.}\quad y_i\big(x_i^\top\beta+\beta_0\big)\ge M \ \ \text{对一切}\ i=1,\dots,N \eqno{4.45}
$$

这些条件保证所有点到分界线的有符号距离至少为 $M$（由 (4.40)，且 $\lVert\beta\rVert=1$），我们要找最大的这样的 $M$ 及其对应的参数。**去掉 $\lVert\beta\rVert=1$ 的约束**，把条件换成

$$
\frac{y_i\big(x_i^\top\beta+\beta_0\big)}{\lVert\beta\rVert}\ \ge\ M \eqno{4.46}
$$

（这重新定义了 $\beta_0$），等价地

$$
y_i\big(x_i^\top\beta+\beta_0\big)\ \ge\ M\lVert\beta\rVert \eqno{4.47}
$$

**推导**：两边同乘 $\lVert\beta\rVert$（正数）。因为对任何满足这些不等式的 $(\beta,\beta_0)$，任何正比例放大也满足，故可以自由地设 $\lVert\beta\rVert=1/M$；于是 (4.45) 等价于

$$
\min_{\beta,\beta_0}\ \frac12\lVert\beta\rVert_2^2 \qquad\text{s.t.}\quad y_i\big(x_i^\top\beta+\beta_0\big)\ge1\ \ \text{对一切}\ i=1,\dots,N \eqno{4.48}
$$

**验证等价性**（两步）：(a) 若 $(\beta,\beta_0)$ 满足 (4.48)，令 $M=1/\lVert\beta\rVert$，则 $y_i(x_i^\top\beta+\beta_0)\ge1=M\lVert\beta\rVert$，即 (4.47)，且 $\lVert\beta\rVert=1/M$，即 (4.45)。(b) 反之 (4.45) 的解缩放到 $\lVert\beta\rVert=1/M$ 后满足 (4.48)，而 $\frac12\lVert\beta\rVert^2=\frac{1}{2M^2}$ 与 $M$ 单调对应。

由 (4.40)，(4.48) 的约束定义了一个围绕分界线的**空板（slab）**，厚度 $2/\lVert\beta\rVert$；我们要选 $\beta,\beta_0$ 使它最厚。这是**凸优化**（二次目标 + 线性不等式约束）。对 $\beta,\beta_0$ 极小化的拉格朗日（原函数）为

$$
L_P=\frac12\lVert\beta\rVert^2-\sum_{i=1}^N\alpha_i\Big[y_i\big(x_i^\top\beta+\beta_0\big)-1\Big] \eqno{4.49}
$$

其中 $\alpha_i\ge0$（不等式约束的乘子，预备知识 O1；括号里的量 $\le0$，故取负号使 $L_P\le$ 目标）。对 $\beta$、$\beta_0$ 求导令零（对偶可行性）：

$$
\nabla_\beta L_P=\beta-\sum_{i=1}^N\alpha_i y_ix_i=0\ \Longrightarrow\ \beta=\sum_{i=1}^N\alpha_i y_ix_i
$$

$$
\nabla_{\beta_0}L_P=-\sum_{i=1}^N\alpha_i y_i=0
$$

把这两条（连同互补松弛）合起来即得 (4.50) 的两条：

$$
\beta=\sum_{i=1}^N\alpha_i y_i x_i,\qquad 0=\sum_{i=1}^N\alpha_i y_i \eqno{4.50}
$$

$$
0=\sum_{i=1}^N\alpha_i y_i \eqno{4.51}
$$

**推导 Wolfe 对偶。** 对任意满足对偶可行性（$\alpha\ge0$、$\sum_i\alpha_iy_i=0$）的 $\alpha$，用 (4.50) 消掉 $\beta$：

$$
\frac12\lVert\beta\rVert^2=\frac12\Big\lVert\sum_i\alpha_iy_ix_i\Big\rVert^2=\frac12\sum_{i,k}\alpha_i\alpha_k y_iy_k\,x_i^\top x_k
$$

$$
-\sum_i\alpha_i\Big[y_i\big(x_i^\top\beta+\beta_0\big)-1\Big]=-\sum_i\alpha_i\Big[x_i^\top\Big(\sum_k\alpha_ky_kx_k\Big)+\beta_0\sum_k\alpha_ky_k-\sum_i\alpha_i\Big]
$$

（用 $y_i^2=1$；因为 $\sum_k\alpha_ky_k=0$ 由 (4.51)，含 $\beta_0$ 的整项为零。）所以

$$
L_D=\sum_{i=1}^N\alpha_i-\frac12\sum_{i=1}^N\sum_{k=1}^N\alpha_i\alpha_k y_iy_k\,x_i^\top x_k \eqno{4.52}
$$

$$
L_D=\sum_{i=1}^N\alpha_i-\frac12\sum_{i=1}^N\sum_{k=1}^N\alpha_i\alpha_k y_iy_k x_i^\top x_k\qquad\text{s.t.}\ \alpha_i\ge0,\ \sum_{i=1}^N\alpha_i y_i=0
$$

由弱对偶性（预备知识 O1）$L_D\le L_P$；Slater 条件（把 $\beta$ 放大即可使不等式严格）成立，故强对偶、对偶最优即原问题最优。**对偶总是凸问题**（预备知识 O1），可以用标准软件求解。

**KKT 条件**（预备知识 O2）还包括互补松弛：

$$
\alpha_i\Big[y_i\big(x_i^\top\beta+\beta_0\big)-1\Big]=0\qquad\forall i \eqno{4.53}
$$

由此读出两条：

- 若 $\alpha_i>0$，则 $y_i(x_i^\top\beta+\beta_0)=1$，即 $x_i$ 在板的边界上；
- 若 $y_i(x_i^\top\beta+\beta_0)>1$，$x_i$ 不在边界上，故 $\alpha_i=0$。

由 (4.50) 可见 $\beta$ 只由**支持点** $\{x_i:\alpha_i>0\}$ 的线性组合定义。图 4.16 的玩具例子有三个支持点；$\beta_0$ 由 (4.53) 对任一支持点解出。对新观测的分类规则：

$$
\hat G(x)=\mathrm{sign}\,\hat f(x) \eqno{4.54}
$$

训练观测没有一个落在间隔内（由构造），但**测试观测会落在间隔内**——这就是「训练数据上的大间隔会带来测试数据上的好分离」的直觉依据。

**与逻辑回归、与 LDA 的对比。** 最优超平面只**聚焦于边界附近的（更噪声的）数据**；LDA 依赖**所有**数据（包括远离边界的点），但若类真的高斯，LDA 就是最优的、分隔超平面要付出「只看边界噪声数据」的代价。另一方面，识别支持点本身用到了所有数据。逻辑回归的解与最优超平面相似（图 4.16 红线与蓝线很近），且共享几个性质：系数向量由一个零均值线性化响应的加权最小二乘拟合定义，权对边界附近的点更大。可分时逻辑回归总能找到分离超平面（对数似然可被推到 0，练习 4.5）。

> **坑** · 不可分情形
> 数据不可分时 (4.48) 无可行解。需要允许重叠的替代形式：增大空间（基变换）会导致过拟合式的人为分离。正确做法是第 12 章的**支持向量分类器**：允许对每个点加一个「松弛」$\xi_i\ge0$，并在目标里为它加惩罚 $\frac12\sum_i\xi_i^2$，得到

> $$\min_{\beta,\beta_0,\xi}\ \frac12\lVert\beta\rVert^2+\frac12\sum_{i=1}^N\xi_i^2 \quad\text{s.t.}\ \begin{cases}y_i(x_i^\top\beta+\beta_0)\ge1-\xi_i\\ \xi_i\ge0\end{cases}$$
>
> 这里可以顺便看看 (4.48) 与它的区别：把 $\xi_i=1-y_i(x_i^\top\beta+\beta_0)$（可分时 $\xi_i=0$）代入即得 (4.48) 的形式。
>
> **对照练习 4.7 的 (4.58)。** 考虑 (4.41) 的一个推广，对**所有**观测求和：
>
> $$D^{\star}(\beta,\beta_0)=\sum_{i=1}^N\Big[-y_i\big(x_i^\top\beta+\beta_0\big)\Big] \eqno{4.58}$$
>
> 问：在 $\lVert\beta\rVert=1$ 的约束下极小化 $D^{\star}$ 描述的是什么准则？它能解最优分离超平面问题吗？
>
> **逐项翻译。** $\mathcal M$（误分类集）换成全体 $\{1,\dots,N\}$，则 $D^{\star}=-\sum_iy_i(x_i^\top\beta+\beta_0)$ 是「所有点到分界线的**带符号**距离之和的相反数」（由 (4.40)，$\lVert\beta\rVert=1$ 时 $y_i(x_i^\top\beta+\beta_0)$ 就是有符号距离）。极小化 $D^{\star}$ 即**极大化所有点带符号距离之和**——可以叫做「最大化间隔**和**」。
>
> **答案：不能。** 两个理由，都能严格写出：
>
> 1. **和可以靠牺牲少数点来无限增大。** 取任一个被错分的点 $i$，沿方向 $\beta=\beta^{\rm start}+\tau\big(-y_ix_i\big)$ 送入参数空间，$y_i(x_i^\top\beta+\beta_0)\to-\infty$，故 $-y_i(x_i^\top\beta+\beta_0)\to+\infty$，其他项变化有限（因为其他 $x_j$ 与 $-y_ix_i$ 的内积有限），总和 $D^{\star}\to-\infty$。**只要存在误分类点，问题就无界**——即使数据可分，只要初值落在「错分某个点」的区域，沿这个方向就会跑到无穷远，而不会走到分离超平面。
> 2. **和的大小不由「最小间隔」控制。** 逻辑回归与 LDA 也依赖全体数据（不是只依赖支持点），但它们的准则可以被改写成 (4.48) 那样的带约束问题；$D^{\star}$ 没有这样的等价形式，$\sum_i y_i x_i$ 一般不平行于任何「最靠近的分界方向」，所以极小化方向与 LDA 方向无关。
>
> 对照之下，(4.48) 的关键在于**不等式约束**：它要求**每一个**点的距离至少为 1，因而最小间隔受控；再加上 $\frac12\lVert\beta\rVert^2$ 的二次惩罚，目标有界且解唯一（对偶 (4.52) 是严格凹的凸问题）。这就是「最小间隔」比「间隔和」更适合做分类准则的原因。

---

## 4.6 编号速查与常见误区 {#s-4-6}

本章 58 个编号公式的用途速查：

| 编号 | 内容 | 关键假设 | 后面哪里再用 |
|---|---|---|---|
| (4.1)(4.2) | 二类 sigmoid 与 logit | 后验的 logit 是 $x$ 的线性函数 | 12.x SVM |
| (4.3)–(4.6) | 指示矩阵回归、最近靶标 | 无（但 $K\ge3$ 有掩蔽） | 12.5 最优打分 |
| (4.7) | Bayes 公式 | 无 | 全书 |
| (4.8) | 高斯类密度 | 类条件高斯 | 6.x 混合模型 |
| (4.9)(4.10) | 对数优势比 → 判别函数 | **协方差相等** | 4.4.6 |
| (4.11) | 两类 LDA 显式规则 | 高斯（只用方向时不需要） | 图 4.14 |
| (4.12) | QDA 判别函数 | 各类独立高斯 | 4.6 |
| (4.13)(4.14) | RDA 收缩协方差 | 无 | 6.5、10.x |
| (4.15)(4.16) | Fisher 判别坐标 | 无高斯假设 | 12.5 |
| (4.17)(4.18) | $K$ 类 logit、softmax | 无 | 练习 4.4 |
| (4.19)–(4.23) | 对数似然、得分、海森、Newton | 二项；可分时无 MLE | 8.x、13.x |
| (4.24)–(4.28) | 矩阵形式与 IRLS | 二项 | glmnet、4.4.5 |
| (4.29)(4.30) | 调整响应、Pearson $\chi^2$ | 模型正确 | 8.x |
| (4.31)(4.32) | $L_1$ 正则化与 KKT | 标准化 | 18.4 |
| (4.33)–(4.38) | LDA 与逻辑回归的似然对比 | 高斯 vs 任意边缘 | 4.4.6 |
| (4.39)(4.40) | 最小二乘边界、带符号距离 | 无 | 12.x |
| (4.41)–(4.44) | 感知机的 $D$、梯度、更新 | 可分性（收敛性） | 12.x |
| (4.45)–(4.54) | 最优超平面对偶与 KKT | **可分** | 第 12 章全部 |
| (4.55)–(4.57) | 回归方向 = LDA 方向（练习 4.2） | 无高斯假设（只用方向时） | 4.3.2 |
| (4.58) | 练习 4.7 的 $D^{\star}$ | 无解 | 4.5.2 |

**五个最容易搞错的点**：

1. **(4.9) 与 (4.12) 的差别只在 $\Sigma_k$ 是否相等。** 「线性分界线」完全是**等协方差假设**的产物，不是高斯假设的产物。去掉等协方差就得到 QDA（(4.12)）；反过来，把高斯换成任何「对数密度是二次型」的密度也得到线性分界线。
2. **(4.51) 与 (4.50) 重复。** 原书把截距的一阶条件 $\sum_i\alpha_iy_i=0$ 单独编号了一次（它既是 (4.50) 的第二式，也是 (4.52) 的约束），不要以为是两条不同的条件。
3. **(4.12) 的 $-\log|\Sigma_k|$ 漏了 $1/2$**（排印问题，正确是 $-\frac12\log|\Sigma_k|$）。按 (4.8) 逐项取对数即可核实。
4. **(4.11) 的方向不需要高斯假设，截距需要。** 前者（(4.57) 的 $\hat\Sigma^{-1}\hat\beta\propto\hat\mu_2-\hat\mu_1$）对任何数据成立；后者 $-\log\frac{N_2}{N_1}-\frac12(\cdot)$ 用到了类先验。所以「回归系数方向 = LDA 方向」但「截距不同」，除非 $N_1=N_2$。
5. **(4.48) 只在可分时可行。** 不可分时的正确形式是第 12 章的软间隔 SVM；把 (4.48) 强行用在小样本不平衡数据上会导致「所有点都被正确分类但没有间隔」的退化解。

**三个反复出现的等价类**：

$$
\underset{\beta}{\min}\sum_{i=1}^N\big(y_i-\beta_0-x_i^\top\beta\big)^2
\ \Longleftrightarrow\ \underset{B}{\min}\sum_{i=1}^N\Big\lVert y_i-\big[(1,x_i)B\big]\Big\rVert_2^2
\ \Longleftrightarrow\ \hat B=(X^\top X)^{-1}X^\top Y
$$

$$
\sum_{i=1}^N\Big[y_i\beta^\top x_i-\log\big(1+e^{\beta^\top x_i}\big)\Big]
\ \xrightarrow{\ \text{一阶条件}\ }\ X^\top(y-p)=0
\ \xrightarrow{\ \text{Newton}\ }\ \beta^{\rm new}=\big(X^\top WX\big)^{-1}X^\top Wz
$$

$$
\hat G(x)=\underset{k}{\mathrm{argmax}}\ \hat f_k(x)
\ \Longleftrightarrow\ \hat G(x)=\underset{k}{\mathrm{argmin}}\ \lVert\hat f(x)-t_k\rVert_2
$$

<a class="src" href="index.html">返回封面</a>
