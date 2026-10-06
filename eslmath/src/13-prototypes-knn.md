---
id: m13
n: "13"
title: 原型方法与最近邻
title_en: Prototype Methods and Nearest Neighbors
desc: 类均值最近邻与 LDA 的等价、LVQ 原型更新、Cover–Hart 一致性定理与风险界、Stone 定理、中位半径与 DANN 自适应近邻
prev: m12
next: m14
prev_title: 第 12 章 支持向量机与柔性判别
next_title: 第 14 章 无监督学习
---

# 13 原型方法与最近邻 {#s-13}

第 11、12 章的神经网络和支持向量机都做同一件事：先猜一个函数族（$f\in\mathcal F$），再用训练数据在这个函数族里搜索最优的 $\hat f$。本章的方法不走这条路。它们既不假设 $Y=f(X)+\varepsilon$，也不去估计条件均值 $\eta(x)=E[Y\mid X=x]$（见预备知识 P2），而是把「已经见过的样本」本身当成知识库。

这带来两种截然不同的哲学。一种是**记忆**：训练集原封不动地留着，来了新点就在旧点里检索，1-近邻与 $k$-近邻是它的极端形式；另一种是**压缩**：训练集被换成少数几个带标签的「原型」，K-均值、学习向量量化与简单贝叶斯属于这一类。两者的共同点是结构极弱——正因为弱，它们在边界极不规则的问题上经常打败那些参数丰富的模型；它们的代价是没法告诉你「变量与响应之间到底是什么关系」。

本章的公式集中在三处：最近邻误差相对贝叶斯误差的定量上界（Cover–Hart 定理，含 (13.2)–(13.5)、(13.12)）、高维下最近邻半径的定量刻画 (13.7)，以及判别自适应度量与全局降维背后的两个矩阵分解 (13.8)–(13.11)。

## 13.1 引言 {#s-13-1}

先把两种思路的形式写清楚。特征必须先标准化，否则欧氏距离会被量纲最大的那个变量主宰：

$$
z_j=\frac{x_j-\bar x_j}{s_j},\qquad j=1,\dots,p
$$

> **基础知识** · 0-1 损失下的风险
> 判别规则 $\hat g$ 的风险是 $\mathrm{Risk}(\hat g)=E\big[\mathbb{I}\big(\hat g(X_0)\ne Y_0\big)\big]$。
> 给定 $X_0=x$，若记 $p_k(x)=\Pr(Y=k\mid X=x)$、$\sum_{k=1}^K p_k(x)=1$，则
> $$\mathrm{Risk}(\hat g)=\int \big(1-p_{\hat g(x)}(x)\big)\,dP_X(x)$$
> 最优规则取 $\hat g^\star(x)=\arg\max_k p_k(x)$，即**判决理论**（见预备知识 O5）。

由全期望公式（预备知识 P2）立刻得到「超出贝叶斯的部分」：

$$
\mathrm{Risk}(\hat g)-\mathrm{Risk}(g^*)=E\Big[\big(1-p_{\hat g(X_0)}(X_0)\big)-\big(1-p_{k^*(X_0)}(X_0)\big)\Big],\qquad k^*(x)=\arg\max_k p_k(x)
$$

> **结果**
> 风险差被限制在 $[0,1]$ 内，而且是**逐点**的：某个点上 $p_k$ 离最大值有多远，就决定了在那里做错的最坏代价有多大。这解释了为什么最近邻这类完全不做平滑的规则不会「灾难性地」错——它在每个点上都至少守着该点的次大类概率。

记忆与压缩的分工可以这样对照：

- 记忆：存 $N$ 个 $(x_i, g_i)$，预测时算 $d(x_0,x_i)$ 并取前 $k$ 个；存储 $O(Np)$，每个查询点 $O(Np)$。
- 压缩：存 $KR$ 个原型 $(m_j, \text{标签}_j)$，预测时取最近的一个；存储 $O(KRp)$，每个查询点 $O(KRp)$。

> **坑**
> 最近邻在**高维**里会退化的定量原因在 §13.4 的 (13.7)：查询点周围 $N$ 个均匀点的中位距离随 $p$ 增大迅速逼近立方体的边长，此时「最近」的判别信息基本被稀释掉。1-近邻分类的偏差常常低、方差很高；$k$ 近邻的偏差–方差权衡在低维分类上很不利（因为分类损失有界，不像平方损失那样方差随 $k$ 爆炸），但在高维回归上就吃亏了。

<a class="src" href="../esl/ch13-prototype-methods-and-nearest-neighbors.html#s-13">原文 §13.1</a>

## 13.2 原型方法 {#s-13-2}

训练数据是 $(x_1,g_1),\dots,(x_N,g_N)$，$g_i\in\{1,\dots,K\}$。一个原型方法用一组点 $m_1,\dots,m_{KR}$（每个带一个类标签）表示训练数据，判别规则是

$$
\hat g(x)=k^*(x)=\arg\min_{1\le j\le KR}\big\lVert x-m_j\big\rVert_2
$$

除了 1-近邻（此时原型就是训练样本本身），原型通常**不是**训练样本。原始的版权见 <a class="src" href="../esl/ch13-prototype-methods-and-nearest-neighbors.html#s-13-2">原文 §13.2</a>。

### 13.2.1 K-均值聚类 {#s-13-2-1}

先用无标签数据找簇心。给定簇数 $R$，优化的目标是「簇内平方和」：

$$
L\big(c,m\big)=\sum_{i=1}^{N}\sum_{r=1}^{R}\mathbb{I}\big(c(i)=r\big)\big\lVert x_i-m_r\big\rVert_2^2
$$

算法在两个坐标块上交替做**精确极小化**：

1. 固定中心、分配样本：把 $\lVert x_i-m_r\rVert_2^2=\lVert x_i\rVert_2^2-2x_i^\top m_r+\lVert m_r\rVert_2^2$ 展开，$\lVert x_i\rVert_2^2$ 与 $r$ 无关可丢掉，于是 $c(i)=\arg\min_r -2x_i^\top m_r+\lVert m_r\rVert_2^2$，即「离哪个中心最近」。
2. 固定分配、求中心：目标分离成 $R$ 个互不相干的子问题，第 $r$ 个是 $\min_{m_r}\sum_{i: c(i)=r}\lVert x_i-m_r\rVert_2^2$。求导 $\partial/\partial m_r=2\sum_{i: c(i)=r}(m_r-x_i)=0$，得 $m_r=\bar x_{c(i)}$（簇内均值）；若某簇为空（$n_r=0$），该簇的均值未定义，这就是「死簇」的来源。

因为每一步都在精确极小化某个坐标块，$L$ 单调不增；而 $L$ 只取有限多个值（分配方案有限），所以算法在有限步内停住（除非出现并列最小值）。

> **推导** · 簇内散度与簇间散度的分解
> 记 $\bar x=N^{-1}\sum_{i=1}^N x_i$，$n_r=\#\{i: c(i)=r\}$。把 $x_i-\bar x$ 拆成 $(x_i-\bar x_{c(i)})+(\bar x_{c(i)}-\bar x)$，平方后交叉项
> $$2\sum_{i: c(i)=r}(x_i-\bar x_{c(i)})^\top(\bar x_{c(i)}-\bar x)=2(\bar x_{c(i)}-\bar x)^\top\sum_{i: c(i)=r}(x_i-\bar x_{c(i)})=0$$
> 所以总平方和可以精确分解成
> $$\sum_{i=1}^{N}\big\lVert x_i-\bar x\big\rVert_2^2=\underbrace{\sum_{r=1}^{R}\sum_{i: c(i)=r}\big\lVert x_i-\bar x_{c(i)}\big\rVert_2^2}_{W}+\underbrace{\sum_{r=1}^{R}n_r\big\lVert \bar x_{c(i)}-\bar x\big\rVert_2^2}_{B}$$
> 这就是 LDA 里 $W$（组内）与 $B$（组间）的同一套矩阵，§13.4 的 DANN 度量正是用它搭出来的。

用 K-均值做分类的步骤是：**在每个类内单独**跑 K--means（每类 $R$ 个中心），给 $K\times R$ 个中心打上该类标签，再把新点归给最近中心的类。缺点很明确：每个类的中心位置只由本类数据决定，别的类没有发言权，于是中心容易堆到类边界上，边界附近的点被错分。

### 13.2.2 学习向量量化 {#s-13-2-2}

LVQ（Kohonen, 1989）把 K-均值改成**在线**算法，一次只处理一个样本，并且让别的类的原型也参与表态：

$$
m_j^{(k)}\ \leftarrow\ \begin{cases} m_j^{(k)}+\epsilon\,\big(x_i-m_j^{(k)}\big), & g_i=k,\\[4pt] m_j^{(k)}-\epsilon\,\big(x_i-m_j^{(k)}\big), & g_i\ne k \end{cases}
$$

其中 $j=k(x_i)$ 是离 $x_i$ 最近的原型下标，$\epsilon$ 是学习率，第 $t$ 步按随机逼近的节奏把 $\epsilon$ 降到 0（预备知识 O4）。

> **推导** · 两条规则都是同一个二次型损失的随机梯度步
> 取「向量量化失真」准则 $L(m)=\sum_{i=1}^{N}\big\lVert x_i-m_{k(x_i)}\big\rVert_2^2$（其中 $k(x_i)=\arg\min_j\lVert x_i-m_j\rVert_2$ 视作常数分配）。对它关于**当前激活的**那个原型求导：$\nabla_{m_j}L=2(m_{k(x_i)}-x_i)$。取步长 $\epsilon/2$ 的下降步得 $m\leftarrow m-\frac{\epsilon}{2}\cdot 2(m-x_i)=m+\epsilon(x_i-m)$，正是「同类则吸引」规则。
> 因此 LVQ1 的第一条规则就是 K-均值准则的一次 SGD；第二条规则只是把同一个梯度步**取反**，即在错分样本上做一步上升，强行让错误原型远离。

再看判别边界。把两类的边界写成「等距超平面」，定义带符号余量 $s(x)=\tfrac12\big(\lVert x-m_1\rVert_2^2-\lVert x-m_2\rVert_2^2\big)$（$s>0$ 表示 $x$ 离 $m_2$ 更近）。设 $v=x_i-m_2$，错误类原型做排斥步 $m_2'=m_2-\epsilon v$，则

$$
s'(x_i)=s(x_i)-\epsilon\,x_i^\top v+\epsilon\,m_2^\top v-\frac{\epsilon^2}{2}\big\lVert v\big\rVert_2^2=s(x_i)-\Big(\epsilon+\frac{\epsilon^2}{2}\Big)\big\lVert x_i-m_2\big\rVert_2^2
$$

对「正确类原型吸引」的情形，同理（令 $u=x_i-m_1$）得 $s'(x_i)=s(x_i)-(\epsilon-\epsilon^2/2)\lVert x_i-m_1\rVert_2^2$。两条规则都让余量按**距离平方**的速度减小：$\epsilon\in(0,1)$ 时两个系数都为正，所以 LVQ1 的每一步都在几何上把边界往「让 $x_i$ 归对类」的方向挪，挪动的幅度与它离原型的远近成正比——这就是「正确类吸引、错误类排斥」。

LVQ2 修的正是 LVQ1 的一个漏洞：当最近的两个原型分属不同的类（最近的是正确类、其次是错误类）时，两个原型同时动、方向相反：

$$
m_j^{(k)}\leftarrow m_j^{(k)}+\epsilon\big(x_i-m_j^{(k)}\big),\qquad m_\ell^{(k')}\leftarrow m_\ell^{(k')}-\epsilon\big(x_i-m_\ell^{(k')}\big)
$$

上两式的余量分析说明这两次更新**同向叠加**，边界移动速度翻倍，同时正确类原型向类内拉、错误类原型被推离，两类都不会因为互相避让而落进对方的区域。LVQ3 及各种「软」版本再把两个步长拆成不同的权重。

> **坑**
> LVQ 是**由算法定义**而不是由某个固定准则定义的（原文原话），所以它的收敛性、样本复杂度都没有干净的结论；$\epsilon$ 不递减到 0 会让原型永远抖动。另外，如果某个原型从未被选中，它就永远不动，成为死原型。

<a class="src" href="../esl/ch13-prototype-methods-and-nearest-neighbors.html#s-13-2-2">原文 §13.2.2</a>

### 13.2.3 高斯混合 {#s-13-2-3}

把每个成分高斯看成「带协方差的原型」，原型方法就与 K-均值、LVQ 统一了。设

$$
p(x)=\sum_{k=1}^{K}\pi_k\,\varphi_p\big(x;\ \mu_k,\ \sigma_k^2 I\big),\qquad \varphi_p(x;\mu,\sigma^2I)=(2\pi\sigma^2)^{-p/2}\exp\Big(-\frac{\lVert x-\mu\rVert_2^2}{2\sigma^2}\Big)
$$

EM 的 E 步给出的责任度（软分配）是

$$
r_{ik}=\frac{\pi_k\,\varphi_p(x_i;\mu_k,\sigma_k^2I)}{\sum_{\ell=1}^{K}\pi_\ell\,\varphi_p(x_i;\mu_\ell,\sigma_\ell^2I)}
$$

M 步把每个样本按权重 $r_{ik}$ 贡献给**每个**簇（对比 K-均值只贡献给一个簇），于是 $\mu_k=\frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}}$。这就是「软聚类 vs 硬聚类」的全部差别。

> **推导** · 固定方差时高斯混合的判别面就是 LDA
> 两类、等方差 $\sigma^2I$ 时，把后验比取对数：
> $$\log\frac{\Pr(Y=1\mid X=x)}{\Pr(Y=2\mid X=x)}=\log\frac{\pi_1}{\pi_2}-\frac{1}{2\sigma^2}\Big(\big\lVert x-\mu_1\big\rVert_2^2-\big\lVert x-\mu_2\big\rVert_2^2\Big)$$
> 记 $m=\tfrac12(\mu_1+\mu_2)$、$d=\mu_1-\mu_2$，则 $\mu_1=m+\tfrac12 d$、$\mu_2=m-\tfrac12 d$，于是
> $$\big\lVert x-\mu_1\big\rVert_2^2-\big\lVert x-\mu_2\big\rVert_2^2=\big\lVert (x-m)-\tfrac12 d\big\rVert_2^2-\big\lVert (x-m)+\tfrac12 d\big\rVert_2^2=-2(x-m)^\top d$$
> 代入得 $\log\frac{\pi_1}{\pi_2}+\frac{(x-m)^\top(\mu_1-\mu_2)}{\sigma^2}$。判别边界即超平面
> $$x^\top(\mu_1-\mu_2)=\sigma^2\log\frac{\pi_1}{\pi_2}-\frac{\big\lVert\mu_1\big\rVert_2^2-\big\lVert\mu_2\big\rVert_2^2}{2}$$
> 与第 4 章 LDA 的边界式逐字相同。同理，欧氏距离下「离哪个类中心近就归哪类」$\arg\min_k\lVert x-m_k\rVert_2^2$ 在共享协方差时也给出同一条边界（两边只差一个正定二次项的公共部分），这就是「共享协方差下最近类均值分类 $\equiv$ LDA」的确切含义。

> **结果** · $\sigma\to 0$ 时高斯混合退化为 K-均值
> 责任度里 $\log\varphi_p$ 的对比项是 $-\lVert x_i-\mu_k\rVert_2^2/(2\sigma^2)$。若 $\mu_{k^\star}$ 是唯一最近中心，则分子指数为 $0$ 而分母中其他项指数发散到 $-\infty$，故 $r_{ik^\star}\to 1$、$r_{ik}\to 0\ (k\ne k^\star)$；M 步的加权均值随之退化为 $\mu_{k^\star}=\bar x_{\{i: k(x_i)=k^\star\}}$。这就是习题 13.1 的结论，也是 K-均值与 EM 关系的精确说法。

<a class="src" href="../esl/ch13-prototype-methods-and-nearest-neighbors.html#s-13-2-3">原文 §13.2.3</a>

## 13.3 k 近邻分类器 {#s-13-3}

$k$-近邻是**基于记忆**的分类器：不需要拟合任何模型。给定查询点 $x_0$，先算它到全部训练点的欧氏距离并升序排列

$$
d_{(i)}=\big\lVert x_{(i)}-x_0\big\rVert_2,\qquad i=1,\dots,N
\eqno{13.1}
$$

再在前 $k$ 个邻居里按多数票决定标签（并列时随机打破）。$k=1$ 时每个训练点就是一个原型，所以 1-近邻是原型方法的特例。典型做法仍是先把每个特征标准化到训练样本上的均值 0、方差 1。

### 13.3.1 Cover–Hart 定理 {#s-13-3-1}

原文把这一段放在 §13.3 的正文里。设 $k^\star=k^\star(x)=\arg\max_k p_k(x)$。在点 $x$ 处贝叶斯规则的错误率是

$$
E^\star(x)=1-p_{k^\star}(x)
\eqno{13.2}
$$

1-近邻的错误率则可以算出来。渐近地假设查询点与最近的训练点重合（偏差为零），设训练点标签 $Y_1$ 与测试点标签 $Y_2$ 在 $p_k=p_k(x)$ 下**独立**同分布，规则预测 $Y_1$，于是

$$
\mathrm{err}_{\text{1-NN}}(x)=\Pr(Y_1\ne Y_2)=\sum_{k=1}^{K}\Pr(Y_1=k)\Pr(Y_2\ne k)=\sum_{k=1}^{K}p_k(x)\big(1-p_k(x)\big)
\eqno{13.3}
$$

> **推导** · (13.3) 不小于贝叶斯错误率 (13.4)
> 记 $s=p_{k^\star}$，因为 $p_k\le s$ 对每个 $k$ 成立，$p_k^2\le s\,p_k$，求和得 $\sum_{k=1}^{K}p_k^2\le s\sum_k p_k=s$。代回：
> $$\sum_{k=1}^{K}p_k(1-p_k)=1-\sum_{k=1}^{K}p_k^2\ \ge\ 1-s=1-p_{k^\star}(x)=E^\star(x)$$
> 也就是
> $$\sum_{k=1}^{K}p_k(x)\big(1-p_k(x)\big)\ \ge\ 1-p_{k^\star}(x)
> \eqno{13.4}$$

上界的方向要用柯西–施瓦茨不等式（预备知识 L1）。剩余的 $K-1$ 个分量和为 $1-s$，故

$$
\sum_{k\ne k^\star}p_k^2\ \ge\ \frac{\big(\sum_{k\ne k^\star}p_k\big)^2}{K-1}=\frac{(1-s)^2}{K-1}
$$

于是 $\sum_{k=1}^K p_k^2\ge s^2+(1-s)^2/(K-1)$，代入 $\sum_k p_k(1-p_k)=1-\sum_k p_k^2$ 并令 $r=1-s$：

$$
1-s^2-\frac{r^2}{K-1}=1-(1-r^2)-\frac{r^2}{K-1}=2r-r^2-\frac{r^2}{K-1}=2r-\frac{K}{K-1}r^2\le 2r-r^2
$$

丢掉 $-\frac{r^2}{K-1}$ 这一非正项即得原文的 (13.5)：

$$
\sum_{k=1}^{K}p_k(x)\big(1-p_k(x)\big)\ \le\ 2\big(1-p_{k^\star}(x)\big)-\big(1-p_{k^\star}(x)\big)^2
\eqno{13.5}
$$

若保留被丢掉的项，把上式在 $x$ 上取平均（用全期望公式，预备知识 P2），再用 Jensen 不等式 $\mathrm{E}\big[E^\star(x)^2\big]\ge\big(\mathrm{E}^\star\big)^2$（括号内对 $r=1-s\ge 0$ 单调），就得到习题 13.3 里的 Cover–Hart 上界：1-近邻误差率在 $L_1$ 意义下收敛到 $E_1$，且

$$
E_1\ \le\ E^\star\Big(2-E^\star\frac{K}{K-1}\Big)\ \le\ 2E^\star
\eqno{13.12}
$$

> **结果**
> 1-近邻的错误率最多是贝叶斯错误率的两倍。两类时 (13.5) 取等号：$p_1+p_2=1$ 使 $\sum_k p_k(1-p_k)=2p_1(1-p_1)=2E^\star(1-E^\star)$，而 (13.12) 正好给出 $E_1\le E^\star(2-2E^\star)$，两者一致。所以若 1-近邻实测错 10%，贝叶斯错分率至少是 5%。
>
> **坑**
> 「两倍」这句话完全依赖**偏差为零**这个渐近假设（维度固定、训练数据把空间填满）。真实问题里偏差可能很大，这也是 §13.4 自适应度量的动机；此外它是**最坏情况**上界，实际差距往往小得多。

### 13.3.2 一致性与两个模拟问题 {#s-13-3-2}

$k$-近邻的贝叶斯一致性可以用三条条件写清楚（Stone 定理在可测空间上的版本）：设 $k=k_N$，并记 $N_k(x)=\#\{i: g_i=k,\ x_i=x\}$-类计数，则当

1. $k=k_N\to\infty$（保证多数票能压过并列）；
2. $k_N/N\to 0$（保证邻域的相对体积收缩到 0）；
3. 边界集 $A_\varepsilon=\{x:\ p_k(x)\ge p_{k^\star}(x)-\varepsilon\ \text{对某个 }k\}$ 的测度随 $\varepsilon\to 0$ 趋于 0；

三条同时成立时，$\hat g_N(x)=\arg\max_k N_k$ 一致收敛到贝叶斯规则。

> **推导** · 三条条件怎么用
> 固定 $x_0$ 与 $\varepsilon>0$，记 $A_\varepsilon=\{x:\ p_k(x)\ge p_{k^\star}(x)-\varepsilon\ \text{对某个 }k\}$。指示函数 $\{x:\mathbb{1}(X_i\in A)\}$ 是 VC 类，故 Glivenko–Cantelli 定理（预备知识 P4）给出
> $$\Pr\Big(\sup_{x}\Big|\tfrac{N_k(x)}{N}-p_k(x)\Big|>\varepsilon\Big)\le 2K e^{-2N\varepsilon^2}\xrightarrow[N\to\infty]{}0$$
> 条件 3 保证 $x_0$ 的某个 $\delta$-邻域 $B_\delta$ 含有正测度的 $A_\varepsilon$ 成分（否则 $B_\delta$ 整体落在「每个 $p_k$ 都比 $p_{k^\star}$ 小 $\varepsilon$」的集合里，与 $p_{k^\star}$ 连续、且条件 3 说这集合测度趋于 0 矛盾）。在上式的事件上，$A_\varepsilon\cap B_\delta$ 的每个点都被样本覆盖，而覆盖它的类满足 $p_k\ge p_{k^\star}(x_0)-\varepsilon$。再由条件 2 取 $\delta$ 足够大使 $\pi_{k^\star}\Pr(X\in B_\delta)>k_N/N$，$B_\delta$ 内的样本数超过 $k_N$，于是投票给 $p_k\ge p_{k^\star}-\varepsilon$ 的类的票数至少是 $(k_N/N)\big(p_{k^\star}(x_0)-\varepsilon\big)$，严格压过任何 $p_k<p_{k^\star}(x_0)-\varepsilon$ 的类；条件 1 保证不出现并列。故 $\hat g_N(x_0)=k^\star(x_0)$ 的概率趋于 1，再令 $\varepsilon,\delta\to 0$ 即得一致性。

原文用来对比方法的模拟问题是：10 个特征各自独立 $\mathrm{Unif}[0,1]$，两类目标为「简单」问题 $Y=\mathbb{I}(X_1>1/2)$ 与「困难」问题

$$
Y=\mathbb{I}\Big(\operatorname{sign}\Big(\prod_{j=1}^{3}X_j\Big)>0\Big)
\eqno{13.6}
$$

两者的贝叶斯错分率都是 0。困难问题里两个类在前 3 个坐标张成的超立方体里形成棋盘格，判别面高度不规则，正是压缩型方法的用武之地；结论是 K-均值与 LVQ 表现几乎一样，在简单问题上胜过最近邻、在困难问题上与「调好的」最近邻相当，而且最优的 $k$ 完全是情境相关的（简单问题上 25-近邻比 1-近邻好 70%，困难问题上 1-近邻反而最好）。

### 13.3.3 不变度量与切距离 {#s-13-3-3}

手写数字识别里，同一个「3」旋转 $7.5^\circ$ 后在 $\mathbb R^{256}$ 中的欧氏距离可以很远。把每个训练图像的所有旋转版本看成 $\mathbb R^{256}$ 中的一条**不变流形**（原书里共 7 维：两个平移、两个缩放、一个旋转、错切、笔画粗细），两条流形之间的最短欧氏距离就是不变度量：

$$
d_I\big(x_i,x_j\big)=\min_{a\in\mathbb R^{7},\,b\in\mathbb R^{7}}\left\lVert x_i-a(x_i)-x_j+b(x_j)\right\rVert_2
$$

其中 $a(x)$ 是流形在 $x$ 处的切向量。$a$ 在 $x$ 处的取法：对图像函数 $F(x)$ 沿参数方向求偏导即可（链式法则，练习 13.4 把它写成 $F(c+x_0+A(x-x_0))$ 的四参数分解）。但完整不变度量有两个毛病：算不动；而且它允许大变换，旋转 $180^\circ$ 后「6」和「9」会被判为相近。**切距离**用两条切线代替两条流形：

$$
d_T\big(x_i,x_j\big)=\min_{a\in\mathbb R^{7},\,b\in\mathbb R^{7}}\left\lVert x_i-a(x_i)-x_j+b(x_j)\right\rVert_2
$$

两个式子的形式完全一样，唯一区别是 $a,b$ 只在**小范围**内取值（沿不变流形取满范围时，$180^\circ$ 旋转会把「6」变成「9」；而切线只取小参数，$a,b$ 大时切线已经不像原来的数字了）。

这就是「局部线性化不变流形」的全部内容，误差是二阶的（小参数时 $\lVert \gamma_i(a)-\gamma_i(a^\star)\rVert=O(\lVert a-a^\star\rVert^2)$）。在 7291 训练、2007 测试的 USPS 手写邮编数据上，1-近邻/欧氏距离错 0.055，1-近邻/**切距离**错 0.026，几乎达到人眼的水平。工程上的替代方案是「hints」：把每个训练图像的若干旋转版本直接加进训练集再跑普通最近邻；不变空间小时这很有效，7 维时就失效了。

<a class="src" href="../esl/ch13-prototype-methods-and-nearest-neighbors.html#s-13-3-3">原文 §13.3.3</a>

## 13.4 自适应最近邻方法 {#s-13-4}

### 13.4.1 中位数半径与判别自适应度量 DANN {#s-13-4-1}

高维里最近邻「很远」不是修辞，可以算。设 $N$ 个点均匀分布在单位立方体 $[-\tfrac12,\tfrac12]^p$ 内，查询点取在原点，$R$ 是 1-近邻半径。半径 $r$ 的球完全落在立方体内（$r\le\tfrac12$），球体积 $v_p r^p$，其中 $v_p=\pi^{p/2}/\Gamma(p/2+1)$ 是 $p$ 维单位球体积。由「球内没有点」的概率得

$$
\Pr(R>r)=\Big(1-v_p r^p\Big)^{N}
$$

中位数半径满足 $\Pr(R>r_{1/2})=\tfrac12$，于是 $1-v_p r^p=2^{-1/N}$、$v_p r^p=1-2^{-1/N}$，即

$$
\operatorname{median}(R)=v_p^{-1/p}\Big(1-\frac{1}{2^{1/N}}\Big)^{1/p}
\eqno{13.7}
$$

> **数值** · 三个典型值
> $N=100$、$p=1$：$v_1=2$，$0.5(1-2^{-0.01})=3.5\times10^{-3}$；$p=2$：$v_2=\pi$，$\sqrt{(1-2^{-0.01})/\pi}=4.7\times10^{-2}$，与 $0.5/\sqrt{N}$ 同阶；$p=10$、$N=1000$：$v_{10}=2.55$，$0.9106\times(1-2^{-0.001})^{0.1}=0.44$，已经贴近立方体的边长 $0.5$。这正是原文「中位半径迅速逼近 0.5」那句话的数值含义。

> **坑**
> 三点。
>
> 其一，**公式要求「球在立方体内」**，即结果不能超过 $\tfrac12$。$p=10,N=100$ 时算得 $0.554>0.5$，此时公式失效——几何上最近邻距离已被立方体的对角线限死，$N$ 再大也不会更远。这就是 §2.4.1「维数灾难」最精确的版本。
>
> 其二，**原书排印的 (13.7) 里那个 $1^{1/N}$ 不是笔误里随便写的孤例，而是一个记号**。它在 §2.4.1 的语境里表示「半径为 1 的球至少含一个点的概率」
>
> $$\Pr(R<1)=1-\big(1-v_p\big)^{N}\ \xrightarrow{\,N\to\infty\,}\ 1-\mathrm{e}^{-N(1-v_p)}\ \xrightarrow[\text{球体积固定}]{}\ 1$$
>
> 也就是说 $1^{1/N}\approx1$ 是「这个球非空」的概率，指数上的 $1/N$ 只是为了把记号凑成 $N$ 次幂的形式。于是「中位数球」的体积分数被原书写成 $\tfrac{1^{1/N}}{2}$——取 $\Pr(R<r)=\tfrac12$。
>
> 其三，**但排印确实有一处不自洽**：按这个读法，体积分数应当出现在**加号**的位置，而原书印的是减号，即 $\big(1-\tfrac{1^{1/N}}{2}\big)^{1/p}$。对照上面的生存函数推导，中位半径应由 $\Pr(R>r)=\tfrac12$ 得到 $v_pr^p=1-2^{-1/N}$，所以正确形式是
>
> $$\mathrm{median}(R)=v_p^{-1/p}\big(1-2^{-1/N}\big)^{1/p}$$
>
> 两者只在 $N=1$ 时重合（此时 $1^{1/1}/2=1/2=2^{-1}$），$N>1$ 时原书的减号使结果与 $N$ 几乎无关（$1^{1/N}\to1$ 给出恒为 $\tfrac12$），这与「中位半径随 $N$ 增大而收缩」明显矛盾。本页采用由生存函数推出的形式，并把原书写法记在这里备查。

动机：近邻分类隐含假设「类概率在邻域内近似常数」，但若概率只沿某个方向变化（如 Figure 13.13 里只有水平方向变），就该把邻域沿垂直方向拉长——偏差下降、方差不变。于是要**自适应地改度量**。

在每个查询点 $x_0$ 处取 50 个近邻，用它们估计局部判别模型，把度量定成

$$
D\big(x,x_0\big)=\big(x-x_0\big)^\top\Sigma\big(x-x_0\big)
\eqno{13.8}
$$

$$
\Sigma=W^{-1/2}\Big[W^{-1/2}BW^{-1/2}+\epsilon I\Big]W^{-1/2}=W^{-1/2}\big[B^\ast+\epsilon I\big]W^{-1/2}
\eqno{13.9}
$$

其中 $W=\sum_{k=1}^{K}\pi_k W_k$ 是合并的组内协方差，$B=\sum_{k=1}^{K}\pi_k(\bar x_k-\bar x)(\bar x_k-\bar x)^\top$ 是组间协方差，二者都用 $x_0$ 附近那 50 个点估计。

> **推导** · (13.9) 到底做了什么
> 第一步是**白化**。令 $y=W^{-1/2}x$、$u=W^{-1/2}(x-x_0)$，并记 $a_k=W^{-1/2}(\bar x_k-\bar x)$，则
> $$B^\ast=W^{-1/2}BW^{-1/2}=\sum_{k=1}^{K}\pi_k a_k a_k^\top,\qquad \sum_{k=1}^{K}\pi_k a_k=0$$
> 于是 (13.9) 等价于
> $$D\big(x,x_0\big)=u^\top B^\ast u+\epsilon\,\big\lVert u\big\rVert_2^2=\sum_{k=1}^{K}\pi_k\big(u^\top a_k\big)^2+\epsilon\sum_{j=1}^{p}u_j^2$$
> 第一项只依赖 $u$ 在 $\{a_k\}$ 张成的子空间上的分量——而这至多是 $K-1$ 维（因为 $\sum_k\pi_k a_k=0$）；第二项是 Mahalanobis 距离的平方。所以几何上：先按 $W$ 把数据球化，再把邻域沿「局部类均值有差异」的 $K-1$ 个方向拉长，其余方向保持球形。
>
> 第二步是 (13.8) 真的是一个度量。取 $\epsilon>0$，有 $W^{1/2}\Sigma W^{1/2}=B^\ast+\epsilon I\succ 0$，由**合同正定**（预备知识 L1）得 $\Sigma\succ 0$。记 $T=\Sigma^{1/2}$，则 $D(x,x_0)=\lVert T(x-x_0)\rVert_2^2$，即 $D$ 是某个内积下的平方距离，于是非负、对称、三角不等式（$\lVert Tu+Tv\rVert_2\le\lVert Tu\rVert_2+\lVert Tv\rVert_2$，再平方）自动成立。取 $\epsilon=0$ 时 $\operatorname{rank}B^\ast\le K-1$，$D$ 退化为**伪度量**：垂直于那些差异方向的位移距离为 0，邻域会变成无限长的条带。$\epsilon$ 的作用正是把条带圆成椭球，$\epsilon=1$ 通常够用。

10 维「同心球」模拟（类 1 限制在平方半径 22.4 与 40 之间、类 2 无限制，各 250 样本）上，DANN 的 5-近邻显著优于 LVQ 与普通 5-近邻：那里判别方向在特征空间里不断变化、所有变量都在某处有用，正是「局部降维」的价值。

> **坑**
> $W^{-1/2}$ 要求 $W$ 正定。若局部 50 个点落在比 50 更高的维数里（例如 $p>50$），$W$ 必然奇异；实际实现里要对 $W$ 做收缩或取伪逆。另外 DANN 是**逐查询点**换度量的，测试一个样本要做 $N$ 次局部 50-近邻搜索与分解，代价远高于普通近邻。

### 13.4.2 全局降维 {#s-13-4-2}

DANN 是逐点的局部降维；很多问题还可以做一次**全局**降维：在原特征空间的最优子空间里跑近邻。在每个训练点 $x_i$ 处算出局部组间质心平方和矩阵 $B_i$，再平均：

$$
\bar B=\frac{1}{N}\sum_{i=1}^{N}B_i
\eqno{13.10}
$$

设 $\bar B=\sum_{\ell=1}^{p}\theta_\ell e_\ell e_\ell^\top$、$\theta_1\ge\theta_2\ge\dots\ge\theta_p\ge0$，那么最优子空间由 $e_1,\dots,e_L$ 张成，最优的秩 $L$ 近似是 $\bar B_{[L]}=\sum_{\ell=1}^{L}\theta_\ell e_\ell e_\ell^\top$。

> **推导** · 为什么 (13.11) 的解恰好是这个近似
> (13.11) 是
> $$\min_{\operatorname{rank}(M)=L}\sum_{i=1}^{N}\operatorname{trace}\Big[\big(B_i-M\big)^2\Big] \eqno{13.11}$$
>
> 1. 把 $M$ 参数化。任何秩 $L$ 的对称半正定矩阵都能写成 $M=AA^\top$，$A$ 是 $p\times L$ 且 $A^\top A=I_L$（对列做 Gram–Schmidt，预备知识 L2/L3）。故上式可写为对 $A$ 优化。
> 2. 展开。$M=AA^\top$ 且 $A^\top A=I_L$ 时，用到 $\operatorname{trace}(PQ)=\operatorname{trace}(QP)$ 以及 $(AA^\top)(AA^\top)=(A^\top A)^2=I_L$，得
>
> $$
> \operatorname{trace}\Big[(B_i-AA^\top)^2\Big]=\operatorname{trace}(B_i^2)-2\operatorname{trace}\big[A^\top B_iA\big]+L
> $$
>
> 3. 对 $i$ 求和。用迹的线性性与 (13.10)：$\sum_{i=1}^{N}\operatorname{trace}(A^\top B_iA)=\operatorname{trace}\big[A^\top\big(\sum_{i=1}^{N}B_i\big)A\big]=N\operatorname{trace}\big[A^\top\bar BA\big]$。目标函数变成 $c_0-2N\operatorname{trace}\big[A^\top\bar BA\big]+NL$（$c_0=\sum_i\operatorname{trace}(B_i^2)$），所以**只需最大化** $\operatorname{trace}\big[A^\top\bar BA\big]$。
> 4. 求解。取 $C=[e_1,\dots,e_L]$，写成 $A=CZ$、$Z^\top Z=I_L$。代入特征分解得
>
> $$
> \operatorname{trace}\big[A^\top\bar BA\big]=\operatorname{trace}\Big[Z^\top\operatorname{diag}(\theta_1,\dots,\theta_L)Z\Big]=\sum_{\ell=1}^{L}\theta_\ell\sum_{m=1}^{L}z_{\ell m}^2\le\sum_{\ell=1}^{L}\theta_\ell
> $$
>
> 最后一个不等式用了 $Z^\top Z=I_L\Rightarrow ZZ^\top=I_L$（$Z$ 是方阵），即 $\sum_{m=1}^{L}z_{\ell m}^2=1$。等号当且仅当 $Z$ 是正交阵，从而 $M=AA^\top=\sum_{\ell=1}^{L}\theta_\ell e_\ell e_\ell^\top=\bar B_{[L]}$。
> 5. 最小值代回去：$c_0-N\Big(2\sum_{\ell=1}^{L}\theta_\ell-L\Big)$，注意它只通过 $\bar B$ 的前 $L$ 个特征对依赖数据。
>
> 顺带说明 $\bar B_{[L]}$ 确实是 $\bar B$ 的最佳秩 $L$ 逼近：$\operatorname{trace}\big[(\bar B-\bar B_{[L]})^2\big]=\sum_{\ell>L}\theta_\ell^2$ 显然最小（Eckart–Young，预备知识 L3）。

> **结果**
> 这也顺带解释了「加权最小二乘拟合一族子空间」的含义：$B_i$ 既记下了局部判别子空间，又记下了该方向上的判别强度。在「两个嵌套球 + 6 个噪声维」的例子里，$\bar B$ 有 4 个大特征值、6 个接近 0，把数据投影到前 4 维再做近邻即可；Figure 13.8 里 STATLOG 卫星图像任务中名为 DANN 的方法就是在全局约化子空间上跑 5-近邻。

<a class="src" href="../esl/ch13-prototype-methods-and-nearest-neighbors.html#s-13-4">原文 §13.4</a>

## 13.5 计算考虑 {#s-13-5}

最近邻的负担有两块：找邻居和存训练集。每个查询点找邻居要 $Np$ 次运算（朴素实现还要排序，$O(N\log N)$），距离矩阵可以一次性算完并缓存；度量一旦变成 $\Sigma(x-x_0)$（DANN），运算量变成每个查询点一次 $O(p^2)$ 的二次型加一次分解，比 $Np$ 还贵——这也是切距离分类器后来被神经网络模仿的动因（Hastie & Simard, 1998）。

减少存储的办法是**编辑**与**压缩**：从训练集里挑一个子集，其余丢弃。直觉上应该保留判别边界附近、且在边界正确一侧的点。

- multi-edit（Devijver & Kittler, 1982）：循环地把数据切成训练/测试，用训练集上的近邻规则去预测测试集，**删掉被分错**的测试点，重复若干轮。
- condensing（Hart, 1968）：从单个随机样本开始，逐个处理后续数据，只有当它被「当前训练集上的近邻规则」分错时才把它收进训练集。

> **延伸** · 原型方法与核
> $k$-近邻其实是一个**硬核**估计。把 $k$ 换成核权重就得到 Nadaraya–Watson 估计：
> $$\hat p_k(x)=\frac{\sum_{i=1}^{N}K\big(d(x,x_i)/h\big)\,\mathbb{I}(g_i=k)}{\sum_{i=1}^{N}K\big(d(x,x_i)/h\big)},\qquad \hat g(x)=\arg\max_k \hat p_k(x)$$
> $K=\mathbb{1}_{\{t<1\}}$ 时回到近邻（球形邻域），$K(t)=(1-t^2)_{+}$（Epanechnikov）或高斯核时就是第 6 章的核平滑。压缩则是把求和范围从 $N$ 个训练点换到 $KR$ 个原型：K-均值、LVQ、GMM 都是「先算权重、再查表」的核方法的固化版本。
>
> 另一个极端是简单贝叶斯，它把判别信息压成**逐特征的对数似然比之和**。在条件独立假设 $p_k(x)=\prod_{j=1}^{p}p_{kj}(x_j)$ 下
> $$\log\frac{\Pr(Y=C_k\mid X=x)}{\Pr(Y=C_{k'}\mid X=x)}=\log\frac{\pi_k}{\pi_{k'}}+\sum_{j=1}^{p}\log\frac{p_{kj}(x_j)}{p_{k'j}(x_j)}$$
> 每一项只依赖一个坐标 $x_j$，是 $[0,1]$ 上的一个一维函数——高维之所以还能work，是因为每个一维函数都能用 $N$ 个样本估得相当准。若 $x_j$ 取高斯，后验比里就出现 $\log p_{kj}(x_j)=-\frac12\log(2\pi\sigma_{kj}^2)-\frac{(x_j-\mu_{kj})^2}{2\sigma_{kj}^2}$，即每个特征贡献一个二次代价，合起来正是**对角协方差的 QDA**。

> **坑**
> 最近邻与简单贝叶斯在高维上失效的机制不同，但都与 (13.7) 有关：最近邻的邻域半径被推到立方体边长，样本间距离趋于「一样远」，最近的那一个几乎随机；简单贝叶斯则要把 $p$ 个特征的条件分布都估准，参数个数随 $p$ 线性增长，且它假定的条件独立在高维下被数据违背。另外别忘了标准化：不做标准化，欧氏距离会被量纲最大的特征主宰。

<a class="src" href="../esl/ch13-prototype-methods-and-nearest-neighbors.html#s-13-5">原文 §13.5</a>