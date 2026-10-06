---
id: m08
n: "8"
title: 模型推断与平均
title_en: Model Inference and Averaging
desc: 自助法与极大似然的对应（参数化自助恰好等于最小二乘）、信息矩阵与 $\chi^2$ 近似的推导、高斯共轭下的贝叶斯线性模型后验、Dirichlet–多项对应、EM 的单调提升、Gibbs 采样、bagging 永不增大均方误差、贝叶斯模型平均与 stacking
prev: m07
next: m09
prev_title: 第 7 章 模型评估与选择
next_title: 第 9 章 加性模型、树与相关方法
---

# 8 模型推断与平均 {#s-8}

前面几章的拟合方法几乎都可以归结为**最小化平方和**（回归）或**最小化交叉熵**（分类），这两者都是**极大似然**的特例。本章把这条线索拉直：先完整复习极大似然推断的机器（score、信息矩阵、$\chi^2$ 近似），说明第 7 章的自助法其实就是「不用手算公式的极大似然」；再把贝叶斯方法讲清楚，并用 Dirichlet–多项这一对精确对应证明**自助分布就是非参数意义下的后验分布**；然后是本章两个计算工具——EM 算法与 Gibbs 采样；最后是本章的建模思想——**平均**（bagging、贝叶斯模型平均、stacking、bumping）。

这一章的核心观点只有一句：**后验均值比后验众数更能降低均方误差，而自助法的 bootstrap 均值就是后验均值的近似。** 这句话把第 7 章的自助法与本章的 bagging 缝在了一起。

---

## 8.1 引言 {#s-8-1}

推断有三条并行的路线，它们回答同一个问题——「$\hat\theta$ 的不确定性有多大」：

1. **极大似然（频率学派）**：给出解析公式（信息矩阵的逆）或用抽样方法模拟（自助法）。
2. **贝叶斯**：给出后验分布 $\mathrm{Pr}(\theta\mid Z)$，从中抽样或求矩。
3. **直接抽样**：MCMC。

本章的主线是说明这三者其实高度一致。8.2 节用一个平滑例子把「最小二乘 = 极大似然 = 参数化自助法」三方对齐；8.3 节给出贝叶斯版本的闭式解；8.4 节证明 $\tau\to\infty$ 时后验与自助分布重合，并在离散情形（Dirichlet–多项）给出**精确**对应。8.5 节的 EM 与 8.6 节的 Gibbs 采样是「参数空间的随机搜索」与「后验抽样的计算工具」，8.7 节起则把它们从「推断」转向「预测改进」。

> **结果** · 记住这条等式，后面所有结论都是它的推论：
>
> $$\text{自助分布}\ \approx\ \text{非信息先验下的后验分布}$$
>
> 第 7 章 (7.51) 的 bagging（那里叫 bootstrap 平均）＝ (8.51)，是后验均值；$\hat f(x)$ 本身是后验众数（极大似然）。

**为什么把推断与平均放在同一章**。上一章我们学会了「误差有多大」，但没回答「能不能改得更好」。本章的回答是：**把不确定性本身当成方法来用**。既然自助分布近似后验分布，那么对参数的每个可能取值都拟合一个模型再平均（bagging），就等于用后验平均代替点估计；而平方损失下平均不增大误差，所以这是「免费」的改进。8.8 节把这件事推广到「不同模型之间的平均」，第 15 章的随机森林则是「在 bagging 之上再降低树间相关性」的最终形态。

**阅读路线建议**。只想学统计推断的读者走 8.2 → 8.3 → 8.4 → 8.6；只想学模型的读者走 8.7 → 8.8 → 8.9。但 8.4 节是理解 bagging 的理论前提，8.5 节的 EM 在第 14 章（无监督学习）还会再次出现。8.10 节把习题里三段最关键的推导（KL 非负、MM 算法、线性拟合做 bagging 无效）单独拎出来。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-1">原文 §8.1</a>

---

## 8.2 Bootstrap 与极大似然 {#s-8-2}

### 8.2.1 一个平滑例子 {#s-8-2-1}

**问题**：先把最小二乘的三个结论写清楚，后面 8.2.2 节再换成极大似然。

设训练数据 $Z=\{z_1,\dots,z_N\}$，$z_i=(x_i,y_i)$，$x_i$ 一维。取三个节点的三次样条，样条空间是 7 维线性空间，用 B 样条基表示：

$$
\mu(x)=\sum_{j=1}^{7}\beta_jh_j(x)\eqno{8.1}
$$

可以把 $\mu(x)$ 理解为条件均值 $E(Y\mid X=x)$。记 $H$ 为 $N\times 7$ 矩阵，$H_{ij}=h_j(x_i)$，$H_{ij}$ 是第 $i$ 个观测在第 $j$ 个基函数上的取值。最小化训练平方和即得

$$
\hat\beta=(H^\top H)^{-1}H^\top y\eqno{8.2}
$$

（预备知识 L2 的投影公式，$\hat\mu(x)=h(x)^\top\hat\beta$，$h(x)^\top=(h_1(x),\dots,h_7(x))$。）由预备知识 L5 的线性变换律与 L4 的二次型公式，

$$
\mathrm{Var}\bigl(\hat\beta\bigr)=(H^\top H)^{-1}H^\top(\sigma^2I_N)H(H^\top H)^{-1}=(H^\top H)^{-1}\hat\sigma^2\eqno{8.3}
$$

（用了 $(H^\top H)^{-1}$ 对称，把中间两个 $H^\top,H$ 夹成 $H^\top H$ 再消掉。）$\hat\sigma^2=\frac1N\sum_{i=1}^N\bigl(y_i-\hat\mu(x_i)\bigr)^2$。预测 $\hat\mu(x)=h(x)^\top\hat\beta$ 是 $\hat\beta$ 的线性组合，故

$$
\mathrm{se}\bigl[\hat\mu(x)\bigr]=\Bigl[h(x)^\top(H^\top H)^{-1}h(x)\Bigr]^{1/2}\hat\sigma\eqno{8.4}
$$

用 $\hat\mu(x)\pm1.96\cdot\mathrm{se}[\hat\mu(x)]$ 画出图 8.2 右上角的**点态 95% 置信带**（1.96 是标准正态的 97.5% 分位，$100-2\times2.5\%=95\%$）。

**自助法的两种版本。**

- **非参数自助法**：从训练数据有放回抽 $B$ 个大小为 $N$ 的数据集，每个样本单位是**数据对** $(x_i,y_i)$。在每个 $Z^\star$ 上重新拟合样条，取 $B=200$，在每个 $x$ 上取第 $0.025B$ 与第 $0.975B$ 个顺序统计量，得到图 8.2 右下角的百分位置信带。它比解析带略宽，尤其在端点处。
**参数化自助法**：假定高斯误差

$$
Y=\mu(X)+\varepsilon;\quad \varepsilon\sim N(0,\sigma^2),\qquad \mu(x)=\sum_{j=1}^{7}\beta_jh_j(x)\eqno{8.5}
$$

（(8.5) 与 (8.1) 是同一个均值模型，(8.5) 只是多了一层噪声假设。）从拟合值出发**加噪声**生成新响应：

$$
y_i^{\star}=\hat\mu(x_i)+\varepsilon_i^{\star};\quad \varepsilon_i^{\star}\sim N\bigl(0,\hat\sigma^2\bigr);\quad i=1,2,\dots,N\eqno{8.6}
$$

**核心推导：为什么参数化自助法恰好复现最小二乘。** 从 bootstrap 样本得到的拟合是

$$
\hat\mu^\star(x)=h(x)^\top(H^\top H)^{-1}H^\top\boldsymbol y^{\star}
$$

代入 (8.6)：

$$
\hat\mu^\star(x)=\hat\mu(x)+h(x)^\top(H^\top H)^{-1}H^\top\boldsymbol\varepsilon^{\star}
$$

其中 $\hat\mu(x)=h(x)^\top(H^\top H)^{-1}H^\top\boldsymbol y$ 是最小二乘估计。注意 $\boldsymbol y$ 与噪声向量 $\boldsymbol\varepsilon^{\star}$ **相互独立**（$H$ 不含 $y$，$\hat\sigma^2$ 当常数），所以增量项是零均值高斯：

$$
\boldsymbol\varepsilon^{\star}\sim N\bigl(\boldsymbol 0,\hat\sigma^2I_N\bigr)\ \Rightarrow\ h(x)^\top(H^\top H)^{-1}H^\top\boldsymbol\varepsilon^{\star}\sim N\Bigl(\boldsymbol 0,\ \hat\sigma^2\,h(x)^\top(H^\top H)^{-1}\underbrace{H^\top H(H^\top H)^{-1}}_{I_7}h(x)\Bigr)
$$

（中间那一坨用 $(H^\top H)$ 对称而消掉。）故

$$
\hat\mu^{\star}(x)\sim N\Bigl(\hat\mu(x),\ h(x)^\top(H^\top H)^{-1}h(x)\hat\sigma^2\Bigr)\eqno{8.7}
$$

**均值是最小二乘估计，标准差与 (8.4) 的近似公式完全一样。** 这就是「bootstrap 是极大似然的计算机实现」的第一个证据，而且它比解析公式更强：即使没有 $\sigma^2$ 的解析公式，只要能抽样，就能拿到分布。

> **结果** · 置信带的两种算法何时不同？解析带来自**渐近正态近似**（Fisher 信息矩阵逆）；自助带来自**真实的抽样分布**。样本少时后者更可信。

**这个推导的结构值得记下来**，它其实是本章的通用套路：

1. 把 bootstrap 样本写成「点估计 + 零均值随机项」（**注意 $H$ 只含 $x$，不依赖 $y$**，所以点估计部分与噪声独立）；
2. 对零均值随机项套「高斯线性变换律」（预备知识 L5）：$Aa\sim N(\boldsymbol 0,A\Sigma A^\top)$；
3. 中间出现的 $A^\top\Sigma A$ 在这里恰好能化简成 $H^\top H$ 消掉，剩下的正好是信息矩阵的逆。

$2\hat\sigma^2$ 这个因子（来自 $H^\top H/2\sigma^2$，见 (8.22)）是全章自始至终的同一个数：解析带用 $\hat\sigma^2$，自助带用 $\hat\sigma^2$，后验用 $\sigma^2$，三者在 $\hat\sigma\to\sigma$ 时合流。

> **坑** · 这里假定 $\hat\sigma^2$ 当常数处理。若在每次自助抽样里重新估计 $\hat\sigma^{\star 2}$，则 $\hat\mu^\star$ 的分布不再是单个高斯（是 $\sigma$ 与 $\beta$ 的混合），(8.7) 精确性失效（见习题 8.4 的注意事项）。

### 8.2.2 极大似然推断 {#s-8-2-2}

**参数化模型。** 记

$$
z_i\sim g_\theta(z)\eqno{8.8}
$$

$\theta$ 是未知参数。例如正态模型取

$$
\theta=(\mu,\sigma^2)\eqno{8.9}
$$

密度是

$$
g_\theta(z)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\Bigl(-\frac{(z-\mu)^2}{2\sigma^2}\Bigr)\eqno{8.10}
$$

（正态密度的公式：$g(z)=\frac1\sigma\varphi\bigl(\frac{z-\mu}{\sigma}\bigr)$，$\varphi$ 是标准正态密度。）

**似然与对数似然。** 似然是数据在模型下的「概率」：

$$
L(\theta;Z)=\prod_{i=1}^{N}g_\theta(z_i)\eqno{8.11}
$$

似然只定义到正倍数（这里取 1）。把 $Z$ 固定、看作 $\theta$ 的函数，取对数：

$$
\ell(\theta;Z)=\sum_{i=1}^{N}\ell(\theta;z_i)=\sum_{i=1}^{N}\log g_\theta(z_i)\eqno{8.12}
$$

$\ell(\theta;z_i)$ 叫对数似然分量。极大似然取 $\hat\theta=\arg\max_\theta\ell(\theta;Z)$。

**score 与信息矩阵。** 定义 score（对数似然的梯度）

$$
\sum_{i=1}^{N}\dot\ell(\theta;Z)=\dot\ell(\theta;z_i),\qquad \dot\ell(\theta;z_i)=\frac{\partial\ell(\theta;z_i)}{\partial\theta}\eqno{8.13}
$$

（用 $\dot\ell$ 表示沿参数方向的导数，与预备知识 L4 一致。）若极大值在参数空间内部，则 $\dot\ell(\hat\theta;Z)=0$——这是解极大似然的方程。**信息矩阵**定义为对数似然的海森负值

$$
I(\theta)=-\sum_{i=1}^{N}\frac{\partial^2\ell(\theta;z_i)}{\partial\theta\partial\theta^\top}\eqno{8.14}
$$

$I(\hat\theta)$ 叫**观测信息**；**Fisher（期望）信息**是

$$
i(\theta)=E_\theta\bigl[I(\theta)\bigr]\eqno{8.15}
$$

记 $\theta_0$ 为真值。**核心定理**：

$$
\hat\theta\ \xrightarrow{\ d\ }\ N\bigl(\theta_0,\ i(\theta_0)^{-1}\bigr)\qquad (N\to\infty)\eqno{8.16}
$$

所以实际用

$$
N\bigl(\hat\theta,\ i(\hat\theta)^{-1}\bigr)\ \text{或}\ N\bigl(\hat\theta,\ I(\hat\theta)^{-1}\bigr)\eqno{8.17}
$$

逼近。参数 $\theta_j$ 的标准误是 $i(\hat\theta)_{jj}^{-1/2}$ 或 $I(\hat\theta)_{jj}^{-1/2}$（(8.18)）：

$$
\mathrm{se}(\hat\theta_j)\approx\sqrt{i(\hat\theta)_{jj}^{-1}}\ \text{或}\ \sqrt{I(\hat\theta)_{jj}^{-1}}\eqno{8.18}
$$

置信点形如 $\hat\theta_j\mp z_{1-\alpha}\sqrt{i(\hat\theta)_{jj}^{-1}}$。

**推导 (8.16)（$p$ 维、Taylor 展开法）。** 记 $J=N^{-1}\dot\ell(\theta_0;Z)$ 为平均 score。$E_{\theta_0}[J]=0$（对数似然的期望对参数求导为零，积分分部），$\mathrm{Var}_{\theta_0}(J)=N^{-1}i(\theta_0)$。按 CLT（预备知识 P1），$\sqrt N\,J\xrightarrow{d}N(\boldsymbol 0,i(\theta_0))$。接着在 $\theta_0$ 处对 $\theta\mapsto N^{-1}\dot\ell(\theta;Z)$ 作一阶 Taylor 展开（$\hat\theta$ 是一致估计，$E|\hat\theta-\theta_0|=O(N^{-1/2})$）：

$$
N^{-1}\dot\ell(\hat\theta;Z)=N^{-1}\dot\ell(\theta_0;Z)+N^{-1}\Bigl[\frac{\partial^2\ell}{\partial\theta\partial\theta^\top}(\theta_0)\Bigr](\hat\theta-\theta_0)+o_p\lVert\hat\theta-\theta_0\rVert
$$

方括号里是 $-I(\theta_0)$。因为 $\dot\ell(\hat\theta;Z)=\boldsymbol 0$，移项得

$$
I(\theta_0)(\hat\theta-\theta_0)=\dot\ell(\theta_0;Z)+o_p(1)\ \Rightarrow\ \hat\theta-\theta_0=I(\theta_0)^{-1}N\,J+o_p(N^{-1/2})
$$

故 $\sqrt N(\hat\theta-\theta_0)\xrightarrow{d}N(\boldsymbol 0,i(\theta_0)^{-1})$，即 (8.16)。**依赖的正则性条件**（删掉任何一条结论都不成立）：极大似然解存在且唯一；参数空间内部极大；$E_{\theta_0}|J_i|<\infty$ 且二阶矩有限；$\hat\theta$ 一致；$i(\theta_0)$ 正定。

**更精确的区间：$\chi^2$ 近似。** 比正态近似更好的是二次型极限（(8.19)）：

$$
2\bigl[\ell(\hat\theta)-\ell(\theta_0)\bigr]\ \sim\ \chi_p^2\eqno{8.19}
$$

**推导。** 在 $\hat\theta$ 处对 $\ell$ 作二阶展开：

$$
\ell(\theta_0)=\ell(\hat\theta)+\dot\ell(\hat\theta)^\top(\theta_0-\hat\theta)+\frac12(\theta_0-\hat\theta)^\top I(\hat\theta)(\theta_0-\hat\theta)+o_p\lVert\hat\theta-\theta_0\rVert^2
$$

$\dot\ell(\hat\theta)=\boldsymbol 0$，故 $\ell(\hat\theta)-\ell(\theta_0)\approx\frac12(\hat\theta-\theta_0)^\top I(\hat\theta)(\hat\theta-\theta_0)$。由 (8.16)，$\hat\theta-\theta_0\approx N(\boldsymbol 0, i(\theta_0)^{-1})$，取 $\theta_0$ 处求值，由预备知识 P4 的高斯二次型定理：

$$
(\hat\theta-\theta_0)^\top I(\theta_0)(\hat\theta-\theta_0)\sim \sum_{k=1}^{p}\lambda_k\chi^2_1\ \text{的加权形式},\qquad \lambda_k=\text{\(I(\theta_0)\) 的特征值}
$$

加权 $\chi^2$ 的求和定理（预备知识 P4）给出它等于 $\sigma^2\chi^2_{\sum\lambda_k}$，这里 $\sum\lambda_k=\operatorname{trace}(I(\theta_0))=p$。故 $2[\ell(\hat\theta)-\ell(\theta_0)]\sim\chi^2_p$。**由此得到 $1-2\alpha$ 置信区间**：$\{\theta_0: 2[\ell(\hat\theta)-\ell(\theta_0)]\le\chi^2_p(1-2\alpha)\}$，即「对数似然下降不超过 $\frac12\chi^2_p(1-2\alpha)$ 的所有 $\theta_0$」，即似然区间（似然轮廓的「$1-2\alpha$ 等高线」）。

**为什么 $\operatorname{trace}(I(\theta_0))=p$（这一步最容易被糊过去）**。Fisher 信息有个漂亮的不变量。由 (8.15) 与正则性条件，

$$
\operatorname{trace}\bigl(i(\theta_0)\bigr)=-E_{\theta_0}\left[\operatorname{trace}\left(\frac{\partial^2\ell(\theta;Z)}{\partial\theta\partial\theta^\top}\right)\right]=-\sum_{j=1}^{p}E_{\theta_0}\left[\frac{\partial^2\ell}{\partial\theta_j^2}\right]=\sum_{j=1}^{p}\mathrm{Var}_{\theta_0}\bigl(\dot\ell(\theta_j)\bigr)=\sum_{j=1}^{p}1=p
$$

第一步把 $I(\hat\theta)$ 换成 $I(\theta_0)$ 只差 $O_p(N^{-1})$；第二步用迹的线性与 $i(\theta_0)$ 的定义；第三步用了**「导数—二阶导数」恒等式** $E_{\theta_0}\bigl[\partial^2\ell/\partial\theta_j^2\bigr]=-E_{\theta_0}\bigl[(\partial\ell/\partial\theta_j)^2\bigr]$（对密度关于 $\theta_j$ 求一次导、并假设矩条件允许求导与积分交换）。于是**每个 score 的方差恰为 1**——这就是 Fisher 信息的「归一化」性质。**所以 $\chi^2$ 的自由度恰是参数个数，没有一个与 $p$ 无关的常数混进来。** 这也解释了为什么 (8.19) 可以写成 $\chi_p^2$ 而不是加权 $\chi^2$ 之和——加权求和里的权重之和**恰好**是 $p$。

**两个近似的相对精度**。(8.16) 的正态近似是一阶的，其覆盖误差是 $O(N^{-1})$；(8.19) 的 $\chi^2$ 近似多带一项二阶信息，实践中在 $p$ 较大时更准。代价是它只给出**集合**（似然区间）而不是单点或逐坐标区间，且区间形状由对数似然的曲率决定，不保证各坐标能同时被 $1-2\alpha$ 覆盖。

> **坑** · (8.17) 里 $i(\hat\theta)$ 与 $I(\hat\theta)$ **必须逐元素取倒数**再开方，不是「取逆后再看对角元」。这两个量差别是 $O_p(1/N)$，样本少时用观测信息 $I(\hat\theta)$ 更稳。

**回到平滑例子。** 参数 $\theta=(\beta,\sigma^2)$，对数似然

$$
\ell(\theta)=\sum_{i=1}^{N}\Bigl[-\frac12\log 2\pi-\frac12\log\sigma^2-\frac{\bigl(y_i-h(x_i)^\top\beta\bigr)^2}{2\sigma^2}\Bigr]=-\frac N2\log 2\pi-\frac N2\log\sigma^2-\frac1{2\sigma^2}\sum_{i=1}^{N}\bigl(y_i-h(x_i)^\top\beta\bigr)^2\eqno{8.20}
$$

**推导 MLE。** 两个一阶条件：

1. $\partial\ell/\partial\beta=\frac1{\sigma^2}H^\top(y-H\beta)=\boldsymbol 0$。右乘 $H^\top$ 可逆化得 $\hat\beta=(H^\top H)^{-1}H^\top y$（与 (8.2) 相同）。
2. $\partial\ell/\partial\sigma^2=-\frac N{2\sigma^2}+\frac1{2\sigma^4}\sum_i\bigl(y_i-h(x_i)^\top\hat\beta\bigr)^2=\boldsymbol 0\Rightarrow\hat\sigma^2=\frac1N\sum_i\bigl(y_i-\hat\mu(x_i)\bigr)^2$。

$$
\hat\beta=(H^\top H)^{-1}H^\top y,\qquad \hat\sigma^2=\frac1N\sum_{i=1}^{N}\bigl(y_i-\hat\mu(x_i)\bigr)^2\eqno{8.21}
$$

**信息矩阵。** 对 $\theta=(\beta,\sigma^2)$，$I(\theta)$ 是块对角的，$\beta$ 块为

$$
I(\beta)=-\frac{\partial^2\ell}{\partial\beta\partial\beta^\top}=\frac1{\sigma^2}\sum_{i=1}^{N}h(x_i)h(x_i)^\top=\frac{H^\top H}{\sigma^2}\eqno{8.22}
$$

（$\sum_ih(x_i)h(x_i)^\top=H^\top H$ 就是 $\sum_j\sum_iH_{ij}^2$。）于是 $\mathrm{cov}(\hat\beta)=I(\beta)^{-1}=(H^\top H)^{-1}\sigma^2$，与最小二乘的 (8.3) 一致。**这就是「参数化自助法 = 最小二乘」的解析版本。**

### 8.2.3 Bootstrap 与极大似然的比较 {#s-8-2-3}

自助法的优势是：**在没有解析公式的地方也能算出极大似然标准误**。举例：平滑例子里若节点的个数与位置由交叉验证**自适应**决定（记为 $\lambda$），标准误与置信带理应把 $\hat\lambda$ 的选择也算进去，但解析上做不到。用自助法：对每个 bootstrap 样本**重新用 CV 选节点**，再做样条平滑，取分位数——得到的曲线同时包含了噪声波动与 $\hat\lambda$ 的波动。在这个例子里两者差别不大，但在自适应更多的场合，这个效应不可忽略。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-2">原文 §8.2</a>

---

## 8.3 贝叶斯方法 {#s-8-3}

### 8.3.1 后验与预测分布 {#s-8-3-1}

给定抽样模型 $\mathrm{Pr}(Z\mid\theta)$ 与参数先验 $\mathrm{Pr}(\theta)$，**后验分布**

$$
\mathrm{Pr}(\theta\mid Z)=\frac{\mathrm{Pr}(Z\mid\theta)\cdot\mathrm{Pr}(\theta)}{\int \mathrm{Pr}(Z\mid\theta)\cdot\mathrm{Pr}(\theta)\,d\theta}\eqno{8.23}
$$

代表看到数据后对 $\theta$ 的更新认识。未来的观测用**预测分布**

$$
\mathrm{Pr}(z_{\mathrm{new}}\mid Z)=\int \mathrm{Pr}(z_{\mathrm{new}}\mid\theta)\cdot\mathrm{Pr}(\theta\mid Z)\,d\theta\eqno{8.24}
$$

**与频率学派的关键差别**：极大似然用 $\mathrm{Pr}(z_{\mathrm{new}}\mid\hat\theta)$ 预测，**不反映 $\theta$ 估计本身的不确定性**；(8.24) 则把它积分掉了。

**三种「预测」的对照**（这是本章最容易被忽略但最重要的一处差别）：

| 做法 | 公式 | 反映参数不确定性？ |
|---|---|---|
| 极大似然（plug-in） | $\mathrm{Pr}(z_{\mathrm{new}}\mid\hat\theta)$ | 否 |
| 贝叶斯后验均值（拟合） | $\int\theta\,\mathrm{Pr}(\theta\mid Z)d\theta$ | 是 |
| 贝叶斯预测分布 | (8.24) | 是，且含观测噪声 |

用第 7 章的记号：$\mathrm{Err}_{\mathcal T}$ 估的是「$\theta$ 已知时」的新观测误差；把 $\theta$ 也积分掉，得到的是「$\theta$ 未知时」的总误差。后者必然更大，差额正是「参数不确定性贡献的那部分 MSE」。**8.2 节的 (8.3) 与 8.3 节的 (8.27) 把这两种量分别算出来，读者可以自己相减核对。**

### 8.3.2 高斯线性模型的后验（核心推导）{#s-8-3-2}

回到平滑例子。假定 $\sigma^2$ 已知，$x_1,\dots,x_N$ 固定（随机性只来自 $y$）。对函数空间的先验不好直接写，但**因为 $\mu(x)=\sum_j\beta_jh_j(x)$，只需给系数 $\beta$ 一个先验**，就隐式定义了函数的先验。取零中心高斯先验

$$
\beta\sim N\bigl(0,\tau\Sigma\bigr)\eqno{8.25}
$$

$\Sigma$ 是先验相关矩阵，$\tau$ 控制先验方差。由此得到 $\mu$ 的隐式高斯过程先验，其协方差核

$$
K(x,x')=\mathrm{cov}\bigl[\mu(x),\mu(x')\bigr]=\tau\cdot h(x)^\top\Sigma h(x')\eqno{8.26}
$$

**推导（高斯共轭）。** 联合密度中与 $\theta$ 有关的部分：

$$
p(\beta\mid\boldsymbol y)\propto\underbrace{\exp\Bigl(-\frac1{2\sigma^2}\lVert\boldsymbol y-H\beta\rVert^2\Bigr)}_{\text{似然}}\cdot\underbrace{\exp\Bigl(-\frac1{2\tau}\beta^\top\Sigma^{-1}\beta\Bigr)}_{\text{先验}}
$$

把似然里的平方展开 $\lVert\boldsymbol y-H\beta\rVert^2=\boldsymbol y^\top\boldsymbol y-2\boldsymbol y^\top H\beta+\beta^\top H^\top H\beta$，与先验合并，得

$$
p(\beta\mid\boldsymbol y)\propto\exp\Bigl\{-\frac12\Bigl[\beta^\top\Bigl(\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\Bigr)\beta-2\boldsymbol y^\top\frac H{\sigma^2}\beta+\text{const}\Bigr]\Bigr\}
$$

括号内是**标准多元正态的二次型形式**：对一般正定矩阵 $A$ 与向量 $b$，$\exp\{-\frac12(\beta^\top A\beta-2\beta^\top b+c)\}$ 是均值为 $A^{-1}b$、精度为 $A$ 的高斯核。故后验也是高斯：

$$
E(\beta\mid Z)=\Bigl(\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\Bigr)^{-1}\frac{H^\top\boldsymbol y}{\sigma^2}\eqno{8.27}
$$

$$
\mathrm{cov}(\beta\mid Z)=\Bigl(\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\Bigr)^{-1}
$$

**别名的等价写法**（书里在 (8.27) 附近还给了 $\sigma^2$ 与 $\tau$ 分离的版本，读者常被 $\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}$ 与 $\frac{H^\top H+\sigma^2\Sigma^{-1}/\tau}{\sigma^2}$ 弄混）：

$$
\Bigl(\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\Bigr)^{-1}=\sigma^2\Bigl(H^\top H+\frac{\sigma^2}{\tau}\Sigma^{-1}\Bigr)^{-1},\qquad E(\beta\mid Z)=\Bigl(H^\top H+\frac{\sigma^2}{\tau}\Sigma^{-1}\Bigr)^{-1}H^\top\boldsymbol y
$$

后者的 $\mu$ 版本由 $\mu(x)=h(x)^\top\beta$ 线性变换得到：

$$
E\bigl(\mu(x)\mid Z\bigr)=h(x)^\top\Bigl(\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\Bigr)^{-1}\frac{H^\top\boldsymbol y}{\sigma^2}\eqno{8.28}
$$

$$
\mathrm{cov}\bigl[\mu(x),\mu(x')\mid Z\bigr]=\tau\cdot h(x)^\top\Bigl(\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\Bigr)^{-1}\Sigma^{-1}h(x')
$$

（协方差用线性变换律 $\mathrm{cov}(A\beta)=A\,\mathrm{cov}(\beta)A^\top$，取 $A=h(x)^\top$；交叉协方差再用 $\mathrm{cov}[\mu(x),\mu(x')]=h(x)^\top\mathrm{cov}(\beta)h(x')$。注意 $\tau$ 从 $\Sigma^{-1}$ 里搬到了外面，因为 $\bigl(\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\bigr)^{-1}=\sigma^2(\cdot)^{-1}$ 且 $\tau\,(\cdot)^{-1}$ 中 $\Sigma^{-1}$ 只出现一次。）

**三个一致性检查，每一个都值得读者自己算一遍：**

1. **$\tau\to\infty$（非信息先验）**：$\Sigma^{-1}/\tau\to\boldsymbol 0$，于是 $E(\beta\mid Z)\to(H^\top H)^{-1}H^\top y=\hat\beta$（(8.2)），$\mathrm{cov}(\beta\mid Z)\to\sigma^2(H^\top H)^{-1}$（(8.3)）。**后验 = 极大似然 = 参数化自助分布。** 这正是图 8.4 右panel（$\tau=1000$）与图 8.2 左下（bootstrap）看起来一样的原因。
2. **$\Sigma=I$**：$E(\beta\mid Z)=(H^\top H+\frac{\sigma^2}{\tau}I)^{-1}H^\top y$，这正是惩罚参数为 $\lambda=\sigma^2/\tau$ 的**岭回归**解（第 3 章配方恒等式 (C2) 的直接结果）。**贝叶斯先验方差 $\tau$ 越大，后验越平坦，岭惩罚越小。**
3. **先验怎么选**：图 8.3 取 $\Sigma=I$ 画了十条先验曲线——先验函数 $\mu$ 是「以零为中心、在 7 维 B 样条空间里任意的平滑曲线」。$\tau=1$ 时后验曲线（图 8.4 左）明显比 bootstrap 更平滑，因为先验把更多权重压给了光滑性；$\tau=1000$ 时两者几乎一样。

**后验宽度为什么总是 $\le$ 最小二乘宽度**。逐方向比较 (8.27) 的协方差与 (8.3)：先验精度 $\Sigma^{-1}/\tau$ 半正定，故后验精度矩阵 $\frac{H^\top H}{\sigma^2}+\frac{\Sigma^{-1}}{\tau}\succeq\frac{H^\top H}{\sigma^2}$，对正定对称矩阵 $A\succeq B\Rightarrow A^{-1}\preceq B^{-1}$，于是对任意方向 $v$ 有

$$
v^\top\mathrm{cov}(\beta\mid Z)v\ \le\ v^\top\sigma^2(H^\top H)^{-1}v
$$

**贝叶斯不确定性只会更小，不会更大**——因为它把「模型可能错」也算进了加权，而不是假装 $\hat\beta=\beta$。这是贝叶斯区间通常比频率学派区间窄、也更容易覆盖不足的原因。

**$\Sigma\neq I$ 什么时候必要**。$\Sigma=I$ 意味着「各 B 样条系数互不相关」，先验把函数看成 7 个独立「基块」相加。若基函数很多，仅靠 B 样条的低维性不足以保证光滑，就要通过约束 $\Sigma$ 额外施加光滑性——**这正是平滑样条的情形**（第 5 章 5.8.1 节：在系数上放一个与二阶差分算子共轭的先验）。此时 (8.28) 的后验收缩不是等向的，而是沿「粗糙方向」收缩更多。

> **坑** · 这里有两处**从贝叶斯角度不规整**的做法：对 $\sigma^2$ 用了常数（非信息）先验，还在后验里直接取 $\sigma^2=\hat\sigma^2$。标准的贝叶斯做法是对 $\sigma$ 也放先验（通常 $g(\sigma)\propto1/\sigma$），求 $(\mu(x),\sigma)$ 的联合后验再**积分掉** $\sigma$，而不是取后验众数（MAP）。书里明确指出这一点。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-3">原文 §8.3</a>

---

## 8.4 Bootstrap 与贝叶斯推断的关系 {#s-8-4}

### 8.4.1 一个能算出闭式的例子 {#s-8-4-1}

从最简单的情形起步：单个观测

$$
z\sim N(\theta,1)\eqno{8.29}
$$

取先验 $\theta\sim N\bigl(0,\tau\bigr)$（这里 $\tau$ 是方差不是标准差），共轭性给出后验

$$
\theta\mid z\sim N\Bigl(\frac{z}{1+\frac1\tau},\ \frac{1}{1+\frac1\tau}\Bigr)\eqno{8.30}
$$

**验证一遍**：先验精度 $\frac1\tau$，似然精度 $1$，后验精度是两者之和 $\frac1\tau+1$，后验方差是精度倒数 $\frac1{1/\tau+1}=\frac{1}{1+1/\tau}$；后验均值是「精度加权平均」$\frac{(1/\tau)\cdot0+1\cdot z}{1/\tau+1}=\frac{z}{1+1/\tau}$ ✓。$\tau$ 越大，先验越弱，后验越集中在 $\hat\theta=z$（极大似然）附近。$\tau\to\infty$ 得

$$
\theta\mid z\sim N(z,1)\eqno{8.31}
$$

**这恰好就是参数化自助分布**：从极大似然估计的密度 $N(z,1)$ 抽 $z^\star$，则 $z^\star$ 的分布是 $N(z,1)$。

**让这个对应成立的三个要素**（书里明确列出，值得逐条理解）：

1. **对 $\theta$ 用非信息先验**；
2. **对数似然 $\ell(\theta;Z)$ 只通过极大似然估计 $\hat\theta$ 依赖数据**，故可写成 $\ell(\theta;\hat\theta)$；
3. **对数似然在 $\theta$ 与 $\hat\theta$ 之间对称**：$\ell(\theta;\hat\theta)=\ell(\hat\theta;\theta)+\text{const}$。

要素 2、3 本质上只有高斯分布满足（对称性来自二次型），但对多项分布**近似**成立，于是引出下一节的精确离散对应。

**为什么「对称性」这么重要**：把 $\ell(\theta;\hat\theta)$ 与 $\ell(\hat\theta;\theta)$ 对调，本质上是在说「$Z$ 与 $\theta_0$ 在似然函数里的地位是对称的」。对高斯，对数似然是

$$
\ell(\theta;Z)=-\frac{N}2\log2\pi-\frac N2\log\sigma^2-\frac1{2\sigma^2}\lVert\boldsymbol y-H\theta\rVert^2
$$

而 $H\theta$ 与 $\boldsymbol y-H\theta$ 只差符号与残差范数的常数倍，故 $\ell$ 在 $\theta$ 与 $(H^\top H)^{-1}H^\top\boldsymbol y$ 之间近似对称。这个对称性保证了「用 $\hat\theta$ 处拟合出的密度去生成自助样本」等价于「从后验里抽一个 $\theta$ 再生成数据」——**这就是参数化自助法为什么等于非信息先验下的贝叶斯后验**。二元情形的显式计算（(8.30) → (8.31)）已经把这件事算到底：$\tau\to\infty$ 时后验均值 $\to z$，方差 $\to1$，正是 $N(z,1)$。

### 8.4.2 Dirichlet–多项对应（精确结果）{#s-8-4-2}

设离散样本空间有 $L$ 个类别，类别 $j$ 的概率是 $w_j$，观测频率是 $\hat w_j$，估计量是 $S(\hat w)$。取对称 Dirichlet 先验

$$
w\sim\mathrm{Di}_{L}(a_1)\quad\Longleftrightarrow\quad p(w)\ \propto\ \prod_{\ell=1}^{L}w_\ell^{a_1-1}\eqno{8.32}
$$

**推导后验（二项数据的共轭性）。** 把似然 $\prod_\ell\hat w_\ell^{n_\ell}$（$\sum_\ell n_\ell=N$）与先验核相乘：

$$
p(w\mid\hat w)\propto\prod_{\ell=1}^{L}w_\ell^{a_1-1}\cdot\prod_{\ell=1}^{L}\hat w_\ell^{n_\ell}
=\prod_{\ell=1}^{L}w_\ell^{a_1+N\hat w_\ell-1}\cdot\prod_{\ell=1}^{L}\hat w_\ell^{n_\ell}
$$

后两项只依赖**数据**，与 $w$ 无关，故归入常数并归一化，得到

$$
w\sim\mathrm{Di}_{L}\bigl(a_1+N\hat w\bigr)\eqno{8.33}
$$

令 $a\to0$（非信息先验），$a_1+N\hat w_\ell\to N\hat w_\ell$：

$$
w\sim\mathrm{Di}_{L}\bigl(N\hat w\bigr)\eqno{8.34}
$$

**bootstrap 分布就是多项分布。** 有放回抽 $N$ 次，类别 $\ell$ 出现 $N\hat w_\ell^{\star}$ 次，故

$$
N\hat w^{\star}\sim\mathrm{Mult}\bigl(N,\hat w\bigr)\eqno{8.35}
$$

即概率质量函数

$$
\Pr\bigl(N\hat w^{\star}_1,\dots,N\hat w^{\star}_L\bigr)=\frac{N!}{\prod_\ell(N\hat w_\ell^{\star})!}\prod_\ell\hat w_\ell^{N\hat w_\ell^{\star}}
$$


**现在做关键比较：$\mathrm{Mult}(N,\hat w)$ 与 $\mathrm{Di}(N\hat w)$ 的前两阶矩。** 对多项分布：

$$
E\bigl[N\hat w_\ell^{\star}\bigr]=N\hat w_\ell,\qquad \mathrm{Cov}\bigl(N\hat w_\ell^{\star},N\hat w_m^{\star}\bigr)=N\hat w_\ell\delta_{\ell m}-N^2\hat w_\ell\hat w_m
$$

即自助权重协方差矩阵的第 $(\ell,m)$ 元素是 $\frac1N\hat w_\ell\delta_{\ell m}-\hat w_\ell\hat w_m$。

对 Dirichlet：先用矩母函数 $M(t)=\prod_\ell E[e^{t_\ell w_\ell}]$，由 $\mathrm{Di}_L(\boldsymbol a)$ 的 $E[w_\ell]=a_\ell/\sum_ja_j$、$\mathrm{Cov}(w_\ell,w_m)=\frac{a_\ell\delta_{\ell m}}{\bigl(\sum_ja_j\bigr)^2}-\frac{a_\ell a_m}{\bigl(\sum_ja_j\bigr)^2}$（$\mathrm{Di}(N\hat w)$ 的 $\sum_j a_j=N$）：

$$
E[w_\ell]=\hat w_\ell,\qquad \mathrm{Cov}(w_\ell,w_m)=\frac{\hat w_\ell\delta_{\ell m}}{N}-\hat w_\ell\hat w_m
$$

**与多项分布完全一样。** 于是 $S(\hat w^{\star})$ 与 $S(w)$ 的分布**同均值、同协方差、同样本空间**，故对光滑的 $S$ 分布极相近。

**把这一步算得更细一点（多项权重的二阶展开）**。用「多项抽样 = 独立伯努利」的表示，$w_\ell^{\star}=n_\ell/N$，其中每个 $N$ 次抽样独立地以概率 $\hat w_\ell$ 落入类别 $\ell$。于是单项 $w_\ell^{\star}$ 的矩由 $n_\ell\sim\mathrm{Bin}(N,\hat w_\ell)$ 给出：

$$
E[w_\ell^{\star}]=\hat w_\ell,\qquad \mathrm{Var}(w_\ell^{\star})=\frac{1}{N^2}\mathrm{Var}(n_\ell)=\frac{1}{N^2}\cdot N\hat w_\ell(1-\hat w_\ell)=\frac{\hat w_\ell(1-\hat w_\ell)}{N}
$$

交叉协方差来自「固定次数 $N$」这个约束：$\sum_\ell n_\ell=N$ 恒成立，故 $\sum_\ell\mathrm{Cov}(n_\ell,n_m)=0$ 对任意 $m$ 成立。配合 $\mathrm{Cov}(n_\ell,n_m)=-N\hat w_\ell\hat w_m\ (m\ne\ell)$，直接读出

$$
\mathrm{Cov}(w_\ell^{\star},w_m^{\star})=-\frac1N\hat w_\ell\hat w_m\quad (\ell\ne m)
$$
**与 Dirichlet 那一行逐字相同。**

这正是第 7 章 7.11.2 节第五步那条一般结论（用多项权重 $\mathrm{Var}(w_a)=\frac{N-1}{N^3}$、$\mathrm{Cov}(w_a,w_b)=-\frac1{N^3}$）在多项情形下的特例；那里的两个量符号相反，正是这里「$\sum w=1$ 的约束」的效果。

> **结果** · 自助分布是一个**近似的、非参数的、非信息性的后验分布**。它不需要正式指定先验、不需要从后验采样，纯粹靠扰动数据得到——「穷人的贝叶斯后验」。**通过扰动数据，自助法逼近了扰动参数的贝叶斯效应**，而且通常简单得多。

**为什么「同均值、同协方差」就够了**。对光滑的 $S$ 由链式法则与二阶 Taylor，

$$
S(w^{\star})-S(w)\approx \nabla S(w)^\top(w^{\star}-w)
$$

故 $\mathrm{Var}(S(w^{\star}))\approx\nabla S^\top\mathrm{Cov}(w)\nabla S$；又 $\mathbb{E}[S(w^{\star})]\approx S(\mathbb{E}[w])+\frac12\nabla^2S\cdot\mathrm{tr}\,\mathrm{Cov}(w)$，故偏差由协方差矩阵的迹决定。**均值与协方差都相同 ⇒ 这两个量都相同**，这是自助逼近后验的定量依据。反过来也能说明自助法什么时候会失灵：**$S$ 不光滑时（如中位数、$\max$、非连续指标）Taylor 展开失效，自助分布与后验的差距会变大**，此时应考虑 BCa 等修正区间（见第 7 章 7.11.3 节）。

**对称 Dirichlet 先验 $a_1$ 的作用**。$a_1>1$ 是「伪计数」，把每个类别先验地给了 $a_1$ 个观察的权重：后验参数变成 $a_1+N\hat w_\ell$（(8.33)）。取 $a_1=1$ 相当于 Haldane 伪计数（先验只支持平坦密度，$E[w_\ell]=1/L$）；取 $a_1=1/2$ 是 Jeffreys 先验（非信息）；取 $a_1\to0$（(8.34)）是 Jeffrey 式非信息先验的极限，也是自助法对应的那个。$a_1\to0$ 时 Dirichlet 密度在单纯形边界处发散（因为指数 $N\hat w_\ell-1$ 在 $\hat w_\ell=0$ 时为 $-1$），故 $a_1=0$ 不是正常先验，只能作为极限。

这条结论直接支撑 8.7 节的 bagging：**bagging 均值 = 近似后验均值**，而 (8.25)(8.27) 的 $\tau\to\infty$ 情形说明贝叶斯后验均值在 $\tau$ 大时也趋于自助分布。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-4">原文 §8.4</a>

---

## 8.5 EM 算法 {#s-8-5}

EM 解决的是这样一类问题：**直接极大化对数似然数值上很难，但补上未观测（潜在）数据后变容易**。补数据的这一步叫数据增广（data augmentation）。

### 8.5.1 二分量混合模型 {#s-8-5-1}

**生成模型。** 双峰的密度不适合单个高斯，故设两个高斯分量

$$
Y_1\sim N(\mu_1,\sigma_1^2),\qquad Y_2\sim N(\mu_2,\sigma_2^2)\eqno{8.36}
$$

$$
Y=(1-\Delta)\cdot Y_1+\Delta\cdot Y_2,\qquad \Delta\in\{0,1\},\ \mathrm{Pr}(\Delta=1)=\pi
$$

（$\Delta$ 未观测）。记 $\phi_\theta(x)$ 为参数 $\theta=(\mu,\sigma^2)$ 的正态密度，则 $Y$ 的密度是混合

$$
g_Y(y)=(1-\pi)\phi_{\theta_1}(y)+\pi\phi_{\theta_2}(y)\eqno{8.37}
$$

**参数**：

$$
\theta=(\pi,\theta_1,\theta_2)=(\pi,\mu_1,\sigma_1^2,\mu_2,\sigma_2^2)\eqno{8.38}
$$

**对数似然**：

$$
\ell(\theta;Z)=\sum_{i=1}^{N}\log\bigl[(1-\pi)\phi_{\theta_1}(y_i)+\pi\phi_{\theta_2}(y_i)\bigr]\eqno{8.39}
$$

**为什么难**：对数里有一堆和，$\partial^2\ell/\partial\theta_k^2$ 一般不含 $\theta_k$，一阶条件无法解耦（这就是「混合模型的极大似然没有闭式解」的原因）。

**若知道 $\Delta$ 就简单了**。完整数据对数似然

$$
\ell_0(\theta;Z,\Delta)=\sum_{i=1}^{N}\bigl[(1-\Delta_i)\log\phi_{\theta_1}(y_i)+\Delta_i\log\phi_{\theta_2}(y_i)\bigr]+\sum_{i=1}^{N}\bigl[(1-\Delta_i)\log(1-\pi)+\Delta_i\log\pi\bigr]\eqno{8.40}
$$

此时 $\mu_1,\sigma_1^2$ 就是 $\{\Delta_i=0\}$ 那批数据的样本均值与方差，$\mu_2,\sigma_2^2$ 类似，$\hat\pi=\frac1N\sum_i\Delta_i$。

**EM 的两步。** 因为 $\Delta$ 未知，就用它的**期望值**（responsibility）代替：

$$
\gamma_i(\theta)=E(\Delta_i\mid\theta,Z)=\mathrm{Pr}(\Delta_i=1\mid\theta,Z)\eqno{8.41}
$$

**E 步**（软分配，用当前参数按密度比分配责任）：

$$
\hat\gamma_i=\frac{\hat\pi\phi_{\hat\theta_2}(y_i)}{(1-\hat\pi)\phi_{\hat\theta_1}(y_i)+\hat\pi\phi_{\hat\theta_2}(y_i)},\qquad i=1,2,\dots,N\eqno{8.42}
$$

（分母是 (8.37) 的混合密度，分子是「来自分量 2」的联合密度。）
- **M 步**（加权极大似然）：$\hat\mu_1=\frac{\sum_i(1-\hat\gamma_i)y_i}{\sum_i(1-\hat\gamma_i)}$，$\hat\sigma_1^2=\frac{\sum_i(1-\hat\gamma_i)(y_i-\hat\mu_1)^2}{\sum_i(1-\hat\gamma_i)}$，$\hat\mu_2,\hat\sigma_2^2$ 用 $\hat\gamma_i$ 同理，$\hat\pi=\frac1N\sum_i\hat\gamma_i$。
- 迭代 2、3 直至收敛。

**初始化**。取两个随机的 $y_i$ 作为 $\hat\mu_1,\hat\mu_2$；两个方差都设为总样本方差 $\frac1N\sum_i(y_i-\bar y)^2$；$\hat\pi$ 从 0.5 开始。

**这个例子里收敛到什么**。表 8.2：迭代 1、5、10、15、20 次时 $\hat\pi$ 依次为 0.485、0.493、0.523、0.544、0.546，最终

$$
\hat\mu_1=4.62,\quad \hat\sigma_1^2=0.87,\quad \hat\mu_2=1.06,\quad \hat\sigma_2^2=0.77,\quad \hat\pi=0.546
$$

**重要警告**：似然的真正极大出现在「把无限高的尖峰放在某个数据点上」（$\hat\mu_1=y_i,\hat\sigma_1^2=0$），似然无穷大但毫无用处。所以我们找的是 $\hat\sigma_1^2,\hat\sigma_2^2>0$ 下的**局部**极大；而且这样的局部极大还可能不止一个。书里的做法是：所有初始值都取 $\hat\sigma_k^2>0.5$，跑很多次，取极大似然最高的那次。

> **坑** · 混合模型里的 $\Delta_i$ 不是「标签」，而是**潜在变量**；E 步给出的是后验责任（软分配），不是硬分类。与决策树的叶节点类别比例相比，$\hat\gamma_i$ 与后者是同一角色的东西。

### 8.5.2 一般形式的 EM：为什么它不降低似然 {#s-8-5-2}

**问题**：(8.47) 说 EM 迭代从不降低对数似然。推导它。

**设定。** 观测数据 $Z$，对数似然 $\ell(\theta;Z)$。潜在数据 $Z^m$，完整数据 $T=(Z,Z^m)$，完整数据对数似然 $\ell_0(\theta;T)$。混合问题里 $(Z,Z^m)=(y,\Delta)$，$\ell_0$ 就是 (8.40)。

**第一步：联合分布分解。** 贝叶斯公式：

$$
\mathrm{Pr}(Z^m,Z\mid\theta')=\mathrm{Pr}(Z\mid\theta')\mathrm{Pr}(Z^m\mid Z,\theta')\eqno{8.44}
$$

两边除以 $\mathrm{Pr}(Z\mid\theta)$ 并用 $\mathrm{Pr}(T\mid\theta')=\mathrm{Pr}(Z^m,Z\mid\theta')$：

$$
\frac{\mathrm{Pr}(T\mid\theta')}{\mathrm{Pr}(Z\mid\theta)}=\mathrm{Pr}(Z^m\mid Z,\theta')\eqno{8.45}
$$

（$\mathrm{Pr}(Z\mid\theta)=\mathrm{Pr}(T\mid\theta)/\mathrm{Pr}(Z^m\mid Z,\theta)$。）

**第二步：取对数，并对 $T$ 在「参数为 $\theta$」的条件下求条件期望。**

$$
\ell(\theta';Z)=\ell_0(\theta';T)-\ell_1(\theta';Z^m\mid Z),\qquad \ell_1:=\log\mathrm{Pr}(Z^m\mid Z,\theta')
$$

$$
\ell(\theta';Z)=E\bigl[\ell_0(\theta';T)\mid Z,\theta\bigr]-E\bigl[\ell_1(\theta';Z^m\mid Z)\mid Z,\theta\bigr]
$$

**第三步：定义 $Q$ 与 $R$。** E 步只算第一项（M 步只需它，且它对 $\theta'$ 是二次的）：

$$
Q(\theta',\hat\theta^{(j)})=E\bigl[\ell_0(\theta';T)\mid Z,\hat\theta^{(j)}\bigr]\eqno{8.43}
$$

（注意它是**哑变量** $\theta'$ 的函数，$\hat\theta^{(j)}$ 是当前估计。）于是

$$
\ell(\theta';Z)\ \equiv\ Q(\theta',\theta)-R(\theta',\theta)\eqno{8.46}
$$

其中 $R(\theta',\theta):=E[\log\mathrm{Pr}(Z^m\mid Z,\theta')\mid Z,\theta]$。

**第四步：两个不等式。**

- **$Q(\theta',\theta)\le Q(\theta,\theta)$**：对固定的 $\theta$，把 $Q$ 看成 $\theta'$ 的函数，$\theta$ 是它的一个合法取值（$\ell_0(\theta;T)$ 的期望在 $\theta'=\theta$ 处）。
- **$R(\theta',\theta)\le R(\theta,\theta)$**：$R(\theta',\theta)$ 是密度 $\mathrm{Pr}(Z^m\mid Z,\theta)$ 关于**对数** $\log\mathrm{Pr}(Z^m\mid Z,\theta')$ 的期望。由 Jensen 不等式（预备知识 C2，$\log$ 是凹函数）
$$
E\bigl[\log\mathrm{Pr}(Z^m\mid Z,\theta')\mid Z,\theta\bigr]\ \le\ \log E\bigl[\mathrm{Pr}(Z^m\mid Z,\theta')\mid Z,\theta\bigr]=\log 1=0
$$

而 $R(\theta,\theta)=E[\log\mathrm{Pr}(Z^m\mid Z,\theta)\mid Z,\theta]=\log1=0$，故 $R(\theta',\theta)\le R(\theta,\theta)$，且**等号当且仅当 $\theta'=\theta$**（这就是习题 8.1 要证的 KL 非负性，见 (8.61)）。

两个不等式其实是同一个不等式：交换 $\theta$ 与 $\theta'$ 后相加（预备知识 P2 的全期望口径），

$$
\frac12\bigl[R(\theta',\theta)-R(\theta,\theta)\bigr]+\frac12\bigl[R(\theta,\theta')-R(\theta',\theta)\bigr]=-\mathrm{KL}\Bigl(\mathrm{Pr}(Z^m\mid Z,\theta)\,\Big\|\,\mathrm{Pr}(Z^m\mid Z,\theta')\Bigr)\ \le0
$$

两项对称相加给出非正的常数，故**两项各自都 $\le0$**。于是 $R(\theta',\theta)-R(\theta,\theta)\le-\mathrm{KL}\le0$，即**似然提升精确等于「$Q$ 的提升」加上「两个条件分布之间的 KL」，两个非负项**（合并形式见 8.10.1 节）。这是 EM 单调性论证里唯一真正的不等式来源，其余都是恒等变形。

**第五步：作差。** 若 $\theta'$ 是 $Q(\theta',\theta)$ 的极大点，

$$
\ell(\theta';Z)-\ell(\theta;Z)=\bigl[Q(\theta',\theta)-Q(\theta,\theta)\bigr]-\bigl[R(\theta',\theta)-R(\theta,\theta)\bigr]\ \ge 0\ \ -\ 0\eqno{8.47}
$$

因为括号里的 $Q$ 差非负、$R$ 差非正，减去非正等于加上非负。**EM 迭代从不降低对数似然。**

**混合例子里可以逐项验一遍这个不等式。** 在混合模型中 $\theta'$ 与 $\theta$ 只差在 $(\pi,\mu_1,\sigma_1^2,\mu_2,\sigma_2^2)$ 的取值上，$R(\theta',\theta)=E[\log\mathrm{Pr}(\Delta\mid y,\theta')\mid\theta]$ 是用 $\theta$ 的责任作权重对 $\theta'$ 的混合对数密度求的期望。把 (8.46) 写开后：

$$
\ell(\theta';Z)=\sum_i\log\bigl[(1-\pi')\phi_{\theta_1'}(y_i)+\pi'\phi_{\theta_2'}(y_i)\bigr]
$$

$$
Q(\theta',\theta)=\sum_i\bigl[(1-\hat\gamma_i)\log\phi_{\theta_1'}(y_i)+\hat\gamma_i\log\phi_{\theta_2'}(y_i)\bigr]+\sum_i\bigl[(1-\hat\gamma_i)\log(1-\pi')+\hat\gamma_i\log\pi'\bigr]
$$

（就是 (8.40) 里把 $\Delta_i$ 换成 $\hat\gamma_i$，书里也这样说。）而 $R$ 由 (8.46) 的定义自动给出。三者相减回到 $\ell$，所以整个证明是**恒等式**，不等号只来自 $\theta'\mapsto Q(\theta',\theta)$ 与 $\theta'\mapsto R(\theta',\theta)$ 的极大/极小性质——这两个性质合起来正是 Jensen 与 $Q$ 的凹性。

**推论（GEM）**：M 步不必求到极大，只需找到使 $Q(\theta',\hat\theta^{(j)})>Q(\hat\theta^{(j)},\hat\theta^{(j)})$ 的 $\theta'$，上升论证照样成立——这叫广义 EM（GEM）。

### 8.5.3 EM 作为最大化–最大化过程 {#s-8-5-3}

换个角度看同一件事。定义函数

$$
F(\theta',\tilde P)=E_{\tilde P}\bigl[\ell_0(\theta';T)\bigr]-E_{\tilde P}\bigl[\log\tilde P(Z^m)\bigr]\eqno{8.48}
$$

其中 $\tilde P(Z^m)$ 是潜在数据上的**任意**分布（混合例中就是责任概率集合 $\gamma_i$）。

**为什么 $F$ 在 $\tilde P(Z^m)=\mathrm{Pr}(Z^m\mid Z,\theta')$ 处等于观测对数似然**。此时

$$
F(\theta',\tilde P)=E\bigl[\ell_0(\theta';T)\mid Z,\theta'\bigr]-E\bigl[\log\mathrm{Pr}(Z^m\mid Z,\theta')\mid Z,\theta'\bigr]
$$

由 (8.45)，$\mathrm{Pr}(T\mid\theta')=\mathrm{Pr}(Z\mid\theta')\mathrm{Pr}(Z^m\mid Z,\theta')$，故第二项 $=\log\mathrm{Pr}(T\mid\theta')-\log\mathrm{Pr}(Z\mid\theta')=\ell_0(\theta';T)-\ell(\theta';Z)$，两项相消得 $F=\ell(\theta';Z)$（即 (8.46) 的特例 $\theta'=\theta$）。

**EM 是 $F$ 的联合极大化**：固定 $\theta'$ 对 $\tilde P$ 极大、再固定 $\tilde P$ 对 $\theta'$ 极大。对 $\tilde P$ 的极大点是

$$
\tilde P(Z^m)=\mathrm{Pr}(Z^m\mid Z,\theta')\eqno{8.49}
$$

**推导 (8.49)（拉格朗日乘子，预备知识 O1）**。最大化 $E_{\tilde P}[\ell_0]-\sum_{z^m}\tilde P(z^m)\log\tilde P(z^m)$，约束 $\tilde P(z^m)\ge0$、$\sum_{z^m}\tilde P(z^m)=1$：

$$
\frac{\partial}{\partial\tilde P(z^m)}\Bigl[\cdots+\lambda\Bigl(\sum\tilde P-1\Bigr)\Bigr]=q(z^m)-\bigl(\log\tilde P(z^m)+1\bigr)+\lambda=0
$$

其中 $q(z^m):=E[\ell_0(\theta';Z,z^m)\mid Z]$。故 $\tilde P(z^m)=\exp(q(z^m)-1+\lambda)\propto\exp(q(z^m))$，$\tilde P$ 与 $q$ 单调 ⇒ $\tilde P$ 与 $q$ 同序。而由 (8.45)，$\exp(q(z^m))\propto\mathrm{Pr}(Z^m\mid Z,\theta')$，故极大点就是条件分布 (8.49)。**这正是 E 步算的东西**（混合例里是 (8.42)）。

对 $\theta'$ 极大时第二项不含 $\theta'$，故只需极大 $Q(\theta',\tilde P)=E_{\tilde P}[\ell_0(\theta';T)]$，这正是 M 步。

**读图 8.7**：横轴-纵轴平面画出增广后的 $F(\theta',\tilde P)$ 的等高线。红线是观测对数似然轮廓——即对每个 $\theta'$ 极大化 $F$ 得到的 profile。EM 就是沿「先沿 $\tilde P$ 方向上坡（E 步）、再沿 $\theta'$ 方向上坡（M 步）」交替爬坡，必然单调上升。**这也是为什么 EM 可能停在局部极大**（图 8.6 里似然曲线平台化）。这个视角还给出替代算法：不必一次把所有潜在数据参数都极大化，可以逐个极大、交替进行。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-5">原文 §8.5</a>

---

## 8.6 MCMC：Gibbs 采样 {#s-8-6}

有了贝叶斯模型后想从后验抽样，通常很难计算。MCMC 是通用办法，**Gibbs 采样**是其中最简单的一种。

**设定与算法**。有随机变量 $U_1,\dots,U_K$，想从它们的联合分布抽样。假设直接从联合分布抽样困难，但从条件分布 $\mathrm{Pr}(U_k\mid U_1,\dots,U_{k-1},U_{k+1},\dots,U_K)$ 抽样容易。Gibbs 采样（算法 8.3）交替地从这 $K$ 个条件分布抽样：当过程稳定后，$(U_1^{(t)},\dots,U_K^{(t)})$ 就是联合分布的一个样本。

**为什么「稳定后就是联合分布的样本」**：每次扫过一轮，各 $U_k$ 的边缘分布都不变（因为扫完一轮等价于把当前边缘分布当作条件分布的边缘），而唯一同时满足全部 $K$ 个边缘不变与相容的分布就是联合分布。这就是「Markov chain Monte Carlo」名字的由来：链的平稳分布（stationary distribution）正是目标联合分布。

**为什么需要 burn-in**：链的样本显然不独立；前若干轮还没到平稳，**估计边缘密度时要丢掉前 $m$ 轮**：

$$
\widehat{\mathrm{Pr}}(U_k(u))=\frac{1}{M-m+1}\sum_{t=m}^{M}\mathrm{Pr}\bigl(u\mid U_\ell,\ \ell\ne k\bigr)\eqno{8.50}
$$

**为什么链的平稳分布就是联合分布（更仔细的论证）**。设联合分布的密度为 $p(u_1,\dots,u_K)$，条件密度为 $p_k(u_k\mid u_{-k})=p(u_k\mid u_{-k})$。**一次扫描（sweep）的不变性与不变性条件**：

$$
\int p(u_1)\cdots p(u_K)\prod_{k=1}^{K}p_k\bigl(\tilde u_k\mid \tilde u_1,\dots,\widehat{\tilde u_k},\dots,\tilde u_K\bigr)\,d\tilde u\ =p(u)
$$

对 $K=2$ 直接验证：$\int p(u_1)p(u_2)p(\tilde u_2\mid\tilde u_1)p(u_1\mid\tilde u_2)\,d\tilde u_2=\int p(u_1)p(\tilde u_2\mid\tilde u_1)\,d\tilde u_2=p(u_1)p(u_2)$，其中第二个等号用了 $p(u_1,\tilde u_2)=p(\tilde u_2)p(u_1\mid\tilde u_2)$（贝叶斯公式）。一般情形按条件独立逐个积分即得（这是 Gibbs 采样正确性的标准证明）。既然 $p$ 本身是不变的平稳分布，且链各态可通，就得到链收敛到 $p$。

**推导 (8.50)（全概率公式，习题 8.3）**。设平稳后的边缘密度是 $g_k(u):=\mathrm{Pr}(U_k=u)$，条件密度是 $g_{k\mid-\!(k)}(u\mid u_{-\!(k)})$。用

$$
\mathrm{Pr}(A)=\int \mathrm{Pr}(A\mid B)\,d\bigl(\mathrm{Pr}(B)\bigr)
$$

取 $A=\{U_k=u\}$、$B=U_{-\!(k)}=U_1,\dots$（去掉第 $k$ 个）：

$$
g_k(u)=\int g_{k\mid-\!(k)}(u\mid u_{-\!(k)})\,d\bigl(\mathrm{Pr}(U_{-\!(k)})\bigr)
=\sum_{t=m}^{M}g_{k\mid-\!(k)}\bigl(u\mid U_1^{(t)},\dots,\widehat{U_k^{(t)}},\dots,U_K^{(t)}\bigr)\cdot\frac{1}{M-m+1}
$$

这就是 (8.50)。**关键**：不需要条件密度的**显式表达式**，只需要能从中抽样；但若表达式可得，直接用 (8.50) 估计 $U_k$ 的边缘密度比用样本做核密度估计更好。

**Gibbs 与 EM 的对应**。取参数 $(\theta,Z^m)$，把 EM 的潜在数据当成 Gibbs 采样器的另一个「参数」。以混合模型为例（算法 8.4）：固定 $\sigma_1^2,\sigma_2^2,\pi$，只让 $\mu_1,\mu_2$ 未知，则

$$
\Delta_i^{(t)}\ \text{从}\ \mathrm{Pr}(\Delta_i\mid\theta^{(t)},Z)\ \text{抽样}\qquad\text{（替代 E 步：算责任）}
$$

$$
\mu_1^{(t)}\sim N\bigl(\hat\mu_1, \hat\sigma_1^2/\textstyle\sum_i(1-\Delta_i^{(t)})\bigr),\quad \hat\mu_1=\frac{\sum_i(1-\Delta_i^{(t)})y_i}{\sum_i(1-\Delta_i^{(t)})}\qquad\text{（替代 M 步：算加权均值）}
$$

$\mu_2$ 类似。**唯一区别是「最大化」换成「抽样」**：EM 取条件期望 $E(\Delta_i\mid\theta,Z)$，Gibbs 从该条件分布抽一个 $\Delta_i$。图 8.8 显示 200 次迭代中 $\mu_1,\mu_2$ 与 $\sum_i\Delta_i/N$ 很快稳定，且**均匀分布在极大似然估计两侧**——这正是后验的样子。

> **坑** · 先验不能是不当（improper）的。书里特别指出：若对 $\sigma_1^2,\sigma_2^2,\pi$ 用不恰当的先验，后验会退化，混合权重全落在一个分量上。要给它们设**正常**先验。

其他后验抽样方法（如 Metropolis–Hastings）不需要条件结构，代价是混合更慢。相关应用见书末参考文献（Spiegelhalter 等 1996）。

**EM 与 Gibbs 的对照表**（读这两节时最好并排放）：

| | EM | Gibbs |
|---|---|---|
| 对象 | 求 $\hat\theta=\arg\max_\theta\ell(\theta;Z)$ | 从 $\mathrm{Pr}(\theta,Z^m\mid Z)$ 抽样 |
| 潜在数据的用法 | 取条件期望 $E(Z^m\mid Z,\theta)$ | 从条件分布 $\mathrm{Pr}(Z^m\mid Z,\theta)$ **抽一个样本** |
| 单调性保证 | (8.47) 似然单调上升 | 无（按构造平稳分布即 $p$） |
| 输出 | 一个点（局部极大） | 一串样本（可算任意泛函的后验） |
| 代价 | 快，但只能给点估计 | 慢，但给完整后验 |

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-6">原文 §8.6</a>

---

## 8.7 Bagging {#s-8-7}

### 8.7.1 定义 {#s-8-7-1}

**问题**：既然 bootstrap 分布近似后验分布，能否用**扰动数据再平均**来改进预测？

对每个 bootstrap 样本 $Z^{\star b}$（$b=1,\dots,B$）拟合模型，得预测 $\hat f^{\star b}(x)$。**Bagging（bootstrap aggregation）** 定义为

$$
\hat f_{\mathrm{bag}}(x)=\frac1B\sum_{b=1}^{B}\hat f^{\star b}(x)\eqno{8.51}
$$

记 $\hat{\mathcal P}$ 为把概率 $1/N$ 均分在每个 $(x_i,y_i)$ 上的经验分布。严格的「真」bagging 估计是 $E_{\hat{\mathcal P}}\hat f^{\star}(x)$，(8.51) 是它的 Monte Carlo 估计，$B\to\infty$ 时收敛。

**$\hat f_{\mathrm{bag}}(x)$ 与 $\hat f(x)$ 只在后者是数据的非线性/自适应函数时才不同**：见下面的「为什么 bagging 对线性拟合没用」一段。回归树则不同：每棵 bootstrap 树会用不同特征、不同切点、可能有不同数量的终端节点，所以平均有意义。

**为什么 bagging 对线性拟合没用（习题 8.4 的证明）**。样条平滑是**响应的线性函数**：$\hat\mu(x)=h(x)^\top(H^\top H)^{-1}H^\top\boldsymbol y$。由 8.2.1 节 (8.7)，增量 $\hat\mu^{\star}(x)-\hat\mu(x)$ 是 $h(x)^\top(H^\top H)^{-1}H^\top$ 作用在零均值噪声上的结果，即**零均值**高斯，故

$$
E_{\mathcal P}\bigl[\hat\mu^{\star}(x)\bigr]=\hat\mu(x)+\underbrace{E\bigl[h(x)^\top(H^\top H)^{-1}H^\top\boldsymbol\varepsilon^{\star}\bigr]}_{=0}=\hat\mu(x)
$$

$$
\hat f_{\mathrm{bag}}(x)=\frac1B\sum_{b=1}^{B}\hat\mu^{\star b}(x)\ \xrightarrow[B\to\infty]{}\ E_{\mathcal P}\bigl[\hat\mu^{\star}(x)\bigr]=\hat\mu(x)
$$

**所以 bagging 线性拟合等于什么都没做。** 用非参数自助法（重抽数据对）时结论近似成立。**这个例子的价值在于它划出了 bagging 的适用边界**：必须先有方差可降——若 $\hat f$ 对数据已经「连续」，bagging 就没有用武之地。

**分类情形**：树输出 $\hat G(x)\in\{1,\dots,K\}$。引入**指示向量** $\hat f(x)$（一个 1、其余 $K-1$ 个 0），$\hat G(x)=\arg\max_k\hat f_k(x)$。则 $\hat f_{\mathrm{bag}}(x)$ 是 $K$ 维向量 $[p_1(x),\dots,p_K(x)]$，$p_k(x)$ 是预测为类 $k$ 的树的比例；bagged 分类器取 $\hat G_{\mathrm{bag}}(x)=\arg\max_k\hat f_{\mathrm{bag},k}(x)$。

> **坑** · **不要把投票比例 $p_k(x)$ 当作类概率的估计**。例：真概率 0.75，每棵树都准确预测 1，则 $p_1(x)=1$，与真值差得远。更好的做法是直接平均树的**类别概率估计**（树就是终端节点里的类别比例），这样既改进概率估计，也降低 bagged 分类器的方差（尤其 $B$ 小时，图 8.10）。

### 8.7.2 核心推导：bagging 永不增加均方误差 {#s-8-7-2}

**问题**：(8.52) 说「总体平均永不增加均方误差」，逐项推出来。

**设定。** 训练观测 $(x_i,y_i)$ 独立来自分布 $\mathcal P$。固定输入 $x$，考虑理想聚合估计 $f_{\mathrm{ag}}(x)=E_{\mathcal P}\bigl[\hat f^{\star}(x)\bigr]$——从**真实总体** $\mathcal P$ 而不是数据抽 bootstrap 样本。$f_{\mathrm{ag}}$ 无法实际计算，但便于分析。

**第一步：加减 $f_{\mathrm{ag}}$。**

$$
E_{\mathcal P}\bigl[\bigl(Y-\hat f^{\star}(x)\bigr)^2\bigr]=E_{\mathcal P}\bigl[\bigl(Y-f_{\mathrm{ag}}(x)+f_{\mathrm{ag}}(x)-\hat f^{\star}(x)\bigr)^2\bigr]
$$

**第二步：展开，交叉项为零。** 记 $U=Y-f_{\mathrm{ag}}(x)$，$V=f_{\mathrm{ag}}(x)-\hat f^{\star}(x)$，则

$$
E[U^2]+E[V^2]+2E[UV]
$$

$E[UV]=E\bigl[E[UV\mid Z^{\star}]\bigr]$。给定 bootstrap 样本 $Z^{\star}$，$Y$ 与之独立，且

$$
E_{\mathcal P}\bigl[V\mid Z^{\star}\bigr]=f_{\mathrm{ag}}(x)-E_{\mathcal P}\bigl[\hat f^{\star}(x)\mid Z^{\star}\bigr]=0
$$

（对线性拟合这**精确**成立，因为 $\hat f^{\star}(x)$ 是响应的线性函数、条件期望可穿过线性；一般估计量只是近似。）故 $E[UV]=E\bigl[U\cdot E[V\mid Z^{\star}]\bigr]=E[U\cdot0]=0$。

**第三步：结果。**

$$
E_{\mathcal P}\bigl[\bigl(Y-\hat f^{\star}(x)\bigr)^2\bigr]=E_{\mathcal P}\bigl[\bigl(Y-f_{\mathrm{ag}}(x)\bigr)^2\bigr]+E_{\mathcal P}\bigl[\bigl(f_{\mathrm{ag}}(x)-\hat f^{\star}(x)\bigr)^2\bigr]\ \ge\ E_{\mathcal P}\bigl[\bigl(Y-f_{\mathrm{ag}}(x)\bigr)^2\bigr]\eqno{8.52}
$$

> **结果** · 右边多出来的第二项就是 $\hat f^{\star}(x)$ 围绕 $f_{\mathrm{ag}}(x)$ 的方差。**所以总体聚合永不增加均方误差。** 由此推断：从训练数据抽的 bagging 也常会降低均方误差（两者只差「$\mathcal P$ 换成 $\hat{\mathcal P}$」）。

**把它写回「偏差–方差」形式，看清哪一项在动。** 回归模型是 $Y=f(x)+\varepsilon$，$E[\varepsilon]=0$、$\mathrm{Var}(\varepsilon)=\sigma_\varepsilon^2$，故对任何只依赖 $x$ 的估计 $\hat g(x)$（bootstrap 样本只含 $x_i$，故 $\hat g$ 与 $\varepsilon$ 独立）：

$$
E\bigl[(Y-\hat g(x))^2\bigr]=\sigma_\varepsilon^2+\bigl(f(x)-E[\hat g(x)]\bigr)^2+\mathrm{Var}\bigl(\hat g(x)\bigr)
$$

把 $g$ 取成 $\hat f^{\star}$：这是第 7 章的 (7.9)。把 $g$ 取成常数 $f_{\mathrm{ag}}$：方差项消失。于是两式相减，差恰为 $E[(\hat f^{\star}(x)-f_{\mathrm{ag}}(x))^2]$，也就是 (8.52) 里的第二项。**结论：聚合把 $\mathrm{Var}(\hat f^{\star})$ 换成了 0**。

**有限 $B$ 时还剩多少方差**。$\hat f_{\mathrm{bag}}=\frac1B\sum_{b=1}^B\hat f^{\star b}$，由预备知识 L5 的二次型方差公式：

$$
\mathrm{Var}\bigl(\hat f_{\mathrm{bag}}(x)\bigr)=\frac1{B^2}\sum_{b=1}^{B}\mathrm{Var}\bigl(\hat f^{\star b}(x)\bigr)+\frac2{B^2}\sum_{b<c}\mathrm{Cov}\bigl(\hat f^{\star b}(x),\hat f^{\star c}(x)\bigr)
$$

**关键：协方差项不会随 $B$ 消失**——两个 bootstrap 样本平均重叠约 63.2%（第 7 章 (7.55)），$\hat f^{\star b}$ 与 $\hat f^{\star c}$ 强相关。故总方差趋于一个**正的下界**，这解释了「bagging 降低但不消除方差」以及「$B$ 取 100 或 200 就够，再多没收益」。对比第 15 章随机森林的做法：它专门去削减这个协方差项（每次分裂只在随机抽取的特征子集里选最优），这才是超越 bagging 的关键。

**关于「偏差不变」的严格说法**。教科书常说「平均降方差、不动偏差」。前半句由上式严格成立；后半句严格成立**需要**

$$
E_{\hat{\mathcal P}}\bigl[\hat f^{\star}(x)\bigr]=E_{\mathcal P}\bigl[\hat f^{\star}(x)\bigr]
$$

这对样条这类线性方法并不成立——习题 8.4 恰好证明了 $\hat f_{\mathrm{bag}}(x)\to\hat f(x)$，即 bagging 线性拟合等于什么都没做。所以「不动偏差」这个论断是**对树这类不稳定方法经验上成立**，不是普遍定理。理解这一点很重要：它解释了为什么 bagging 对岭回归/样条无用、对树有用。

### 8.7.3 图 8.9–8.12 的实例与限制 {#s-8-7-3}

**实例**。$N=30$，两类，$p=5$ 个标准高斯特征（两两相关 0.95），$\mathrm{Pr}(Y=1\mid x_1\le0.5)=0.2$、$\mathrm{Pr}(Y=1\mid x_1>0.5)=0.8$，Bayes 误差 0.2；另取 2000 个测试点。不剪枝地拟合分类树，并拟合 200 个 bootstrap 树。图 8.9 显示这些树**各不相同**（切分特征、切点、终端节点数都不同）；图 8.10 显示单棵树的测试误差很大而 bagged 树的误差低得多。原因是这个例子里树因特征间的高度相关而方差极大，bagging 把方差磨平。

**0–1 损失下不成立**。上面的论证依赖平方误差损失的可加性；0–1 损失下没有这种可加性，所以「bagging 一个好分类器可能变好，bagging 一个坏分类器可能变坏」。反例：$Y\equiv1$，分类器 $\hat G(x)$ 以概率 0.4 预测 1、以概率 0.6 预测 0，则 $\hat G$ 的错误率是 0.6，而 bagged 分类器（众数投票）的错误率是 **1.0**。

**共识投票的正确版本**。设两分类，Bayes 决策 $G(x)=1$，每个弱学习器 $\hat G^{\star b}_b$ 错误率 $e_b=e<0.5$，且**相互独立**。记 $S_1(x)=\sum_{b=1}^B\hat G^{\star b}_b(x)$ 为投 1 的票数，则 $S_1(x)\sim\mathrm{Bin}(B,1-e)$，且

$$
\mathrm{Pr}\bigl(S_1>B/2\bigr)=1-\mathrm{Pr}\bigl(\mathrm{Bin}(B,1-e)\le B/2\bigr)\to1\qquad(B\to\infty)
$$

（CLT 或大数律，预备知识 P1。）这就是「群众的智慧」（Surowiecki 2004）。**关键前提是「独立」，而 bagged 树并不独立**——这正是第 15 章随机森林要改进的地方（降低树间相关）。图 8.11 的模拟：50 名评委、10 个奖项、每项 4 个候选，只有 15 人有知识；若有知识者选对概率 0.5，共识的表现约为个人的两倍。

**bagging 什么时候没用**。图 8.12：100 个数据点、两个特征、两类，真边界是对角线 $x_1+x_2=1$，但规则只有「沿 $x_1$ 或 $x_2$ 的单次切分」。bagging 50 次后得到的边界（左panel蓝线）很差——单切分规则在训练数据上找到的切点接近 0（变量取值范围的中点），在远离中心处贡献很小。改用平均概率也没用：bagging 估计的是**单切分规则的期望类别概率**，而这个期望概率**在任何单次实现里都不可实现**（就像「一个女人不能有 2.4 个孩子」）。所以 bagging 稍微扩大了基学习器的模型空间，但需要更大的扩大时就不够了——那正是**提升（boosting，第 10 章）**的用途。

**代价**：bagging 会抹掉模型里的简单结构。bagged 树不再是树，模型解释性丧失。而最受益于 bagging 的不稳定模型（树）之所以不稳定，正是因为强调了可解释性——**bagging 把可解释性换成了预测精度**。最近邻这类稳定过程做 bagging 几乎没变化。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-7">原文 §8.7</a>

---

## 8.8 模型平均与 stacking {#s-8-8}

### 8.8.1 贝叶斯模型平均 {#s-8-8-1}

8.4 节把自助值看作参数的后验值，于是 $\hat f_{\mathrm{bag}}(x)$（(8.51)）就是**近似的后验贝叶斯均值**，而训练估计 $\hat f(x)$ 对应**后验众数**。既然后验均值（不是众数）最小化平方误差损失，bagging 能降均方误差就毫不奇怪。

推广：有一组候选模型 $\mathcal M_m$（$m=1,\dots,M$），关心某个量 $\zeta$（例如某个固定 $x$ 处的预测）。其**后验分布**是混合

$$
\mathrm{Pr}(\zeta\mid Z)=\sum_{m=1}^{M}\mathrm{Pr}(\zeta\mid\mathcal M_m,Z)\mathrm{Pr}(\mathcal M_m\mid Z)\eqno{8.53}
$$

**后验均值**

$$
E(\zeta\mid Z)=\sum_{m=1}^{M}E(\zeta\mid\mathcal M_m,Z)\mathrm{Pr}(\mathcal M_m\mid Z)\eqno{8.54}
$$

即以模型后验概率为权重的加权平均。由此得到若干具体策略：

- **委员会方法（committee）**：等权平均，相当于给每个模型相同概率。Bagging 属于这一类（等权，因为 $\hat f_{\mathrm{bag}}$ 是简单平均）。
- **BIC 权重**：由第 7 章 7.7 节的 (7.41)，$\mathrm{Pr}(\mathcal M_m\mid Z)\approx e^{-\mathrm{BIC}_m/2}/\sum_\ell e^{-\mathrm{BIC}_\ell/2}$。适用于「同一参数化模型、不同参数取值」的情形（线性回归的各个子集）。
**完整的贝叶斯配方**是

$$
\mathrm{Pr}(\mathcal M_m\mid Z)\propto\mathrm{Pr}(\mathcal M_m)\cdot\mathrm{Pr}(Z\mid\mathcal M_m)\propto\mathrm{Pr}(\mathcal M_m)\cdot\int\mathrm{Pr}(Z\mid\theta_m,\mathcal M_m)\mathrm{Pr}(\theta_m\mid\mathcal M_m)d\theta_m\eqno{8.55}
$$

理论上可以指定 $\mathrm{Pr}(\theta_m\mid\mathcal M_m)$ 并数值积分算后验概率。**但作者说不曾见到有说服力的证据表明这比简单的 BIC 近似更值得付出努力。**

### 8.8.2 频率学派版本的平均：最优权重与它为什么有效 {#s-8-8-2}

**问题**：(8.57) 的权重怎么算出来？为什么组合模型一定不劣？

**第一步：写出优化问题。** 给定预测 $\hat f_1(x),\dots,\hat f_M(x)$，平方误差损失下找权重 $w=(w_1,\dots,w_M)^\top$ 使

$$
\hat w=\arg\min_{w}\ E_{\mathcal P}\bigl[Y-\sum_{m=1}^{M}w_m\hat f_m(x)\bigr]^2\eqno{8.56}
$$

$x$ 固定，$Z$ 里的 $N$ 个观测与 $Y$ 都按 $\mathcal P$ 分布。

**第二步：一阶条件。** 记 $\hat F(x)^\top\equiv[\hat f_1(x),\dots,\hat f_M(x)]$，目标 $g(w)=E[(Y-\hat F^\top w)^2]=E[Y^2]-2w^\top E[\hat F Y]+w^\top E[\hat FF^\top]w$。由预备知识 L4 的矩阵微分恒等式（取 $G=-\hat F$，$A=\hat F^\top$）：

$$
\nabla_w g=-2E[\hat F Y]+2E[\hat FF^\top]w=\boldsymbol 0
$$

解出（假定 $E[\hat FF^\top]$ 可逆）：

$$
\hat w=E_{\mathcal P}\bigl[\hat F(x)\hat F(x)^\top\bigr]^{-1}E_{\mathcal P}\bigl[\hat F(x)Y\bigr]\eqno{8.57}
$$

**这就是把 $Y$ 对 $[\hat f_1(x),\dots,\hat f_M(x)]$ 做总体线性回归。** 顺带验证二阶条件：$g$ 是 $w$ 的严格凸二次函数（$E[\hat FF^\top]$ 半正定且通常正定），故 (8.57) 是唯一全局极小。

**第三步：为什么组合不劣（关键）。** 记 $\mathcal F=\mathrm{span}\{\hat f_1,\dots,\hat f_M\}$，把 $Y$ 投影到 $\mathcal F$ 上（这正是 (8.57)）。由投影的正交性，残差 $Y-\hat F^\top\hat w$ 与 $\mathcal F$ 垂直，故

$$
E_{\mathcal P}\bigl[\bigl(Y-\sum_{m=1}^{M}\hat w_m\hat f_m(x)\bigr)^2\bigr]\ \le\ E_{\mathcal P}\bigl[\bigl(Y-\hat f_m(x)\bigr)^2\bigr]\qquad \forall m\eqno{8.58}
$$

**具体证明**：任何单个 $\hat f_m(x)$ 都属于 $\mathcal F$，所以 $Y-\hat f_m(x)=(Y-\hat F^\top\hat w)+(\hat F^\top\hat w-\hat f_m(x))$，第二项在 $\mathcal F$ 里，两项正交（预备知识 L2），故

$$
E[(Y-\hat f_m)^2]=E[(Y-\hat F^\top\hat w)^2]+E[(\hat F^\top\hat w-\hat f_m)^2]\ \ge\ E[(Y-\hat F^\top\hat w)^2]
$$

**这一步是纯粹的 Pythagoras 恒等式，与分布无关，恒成立。** 这与 bagging 的 (8.52) 是同一族结论（那里用「交叉项为零」，这里用「正交性」）——两者的共同根源都是预备知识 L2 的正交投影定理。

**再往下一步：什么时候能取到等号**。等号当且仅当 $\hat f_m\in\mathcal F$（自动成立）**且** $\hat F^\top\hat w=\hat f_m$。也就是说，只有当最优权重恰好把全部权重放在第 $m$ 个模型上（其余为零）时才等号。所以严格结论是：**组合后的误差不超过任何一个单独模型**，而组合的真好处在于它**同时**逼近所有模型——$E[(\hat F^\top\hat w-\hat f_m)^2]$ 这一项体现了「为了兼顾其他模型而付出的代价」。

**与 (8.52) 的对比**。(8.52) 里聚合器的权重被限制成**全等**（$1/B$），故它保留一点偏差但方差大降；(8.57) 允许**任意**权重，故在总体层面严格最优，但必须能估出总体矩 $\mathcal P$ 才能用。bagging 相当于「权重只能取 $1/B$ 这一组」的特例，stacking (8.59) 则是「用留一预测去估这组权重」的可行版本。

> **坑** · (8.58) 只在**总体层面**成立。实际用训练集做 (8.57) 的线性回归并不总是成功：若 $\hat f_m(x)$（$m=1,\dots,M$）是 $M$ 个输入中规模为 $m$ 的最佳子集预测，线性回归会把**全部权重压在最大模型上**（$\hat w_M=1$，其余为 0）。原因是**没有把复杂度放到同样的 footing 上**（本例中复杂度是输入个数 $m$）。这正是 8.8.3 节 stacking 要修正的地方。

### 8.8.3 Stacking {#s-8-8-3}

Stacking 就是修正上一段那个问题。记 $\hat f_{m-i}(x)$ 为模型 $m$ 在**删去第 $i$ 个训练观测**后、在 $x$ 处的预测。权重由 $y_i$ 对 $\hat f_{m-i}(x_i)$ 的最小二乘回归给出：

$$
\hat w_{\mathrm{st}}=\arg\min_{w}\ \sum_{i=1}^{N}\Bigl[y_i-\sum_{m=1}^{M}w_m\hat f_{m-i}(x_i)\Bigr]^2\eqno{8.59}
$$

最终预测是 $\sum_m\hat w_{\mathrm{st},m}\hat f_m(x)$。

**为什么用交叉验证的预测就能避免复杂度偏袒**：$\hat f_{m-i}(x_i)$ 是模型 $m$ 在没看过第 $i$ 个点时给出的预测，所以「模型 $m$ 在训练集上表现好」这件事部分来自模型自己的复杂度，而 $\hat f_{m-i}(x_i)$ 扣掉了这部分；所有模型因此被放在同样的 footing 上。

**改进**：把权重限制为非负且和为 1。这在「权重即后验模型概率」(8.54) 的解释下很自然，且把问题变成可解的二次规划（约束集是单纯形）。

**与 LOO 模型选择的精确关系**：把 (8.59) 的可行域限制成「$\sum_mw_m=1$ 且 $w_m\in\{0,1\}$」，就退化为「选 CV 误差最小的 $\hat m$」。区别是：**stacking 不是选一个，而是按估计出的最优权重组合**——通常预测更好，但可解释性更差。

**stacking 的一般化**：(8.59) 里的「线性回归」可以换成任何学习方法；权重还可以依赖输入位置 $x$。这样学习方法就被「叠」在彼此之上，称为 stacked generalization（Wolpert 1992；Breiman 1996b）。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-8">原文 §8.8</a>

---

## 8.9 随机搜索：bumping {#s-8-9}

Bumping 不做平均、不组合模型，而是用 bootstrap 在模型空间里**随机游走**，找更好的单个模型——当拟合方法有很多局部极小、或容易卡在糟糕解上时有用。

具体做法与 bagging 一样抽 $Z^{\star 1},\dots,Z^{\star B}$ 并各自拟合，得 $\hat f^{\star b}(x)$，$b=1,\dots,B$。**但不是平均，而是选训练集误差最小的那一个。** 平方误差下，选中的 $b$ 满足

$$
\hat b=\arg\min_{b}\ \sum_{i=1}^{N}\Bigl[y_i-\hat f^{\star b}(x_i)\Bigr]^2\eqno{8.60}
$$

最终模型是 $\hat f^{\star\hat b}(x)$。按惯例**把原始训练样本也算作一个 bootstrap 样本**，使方法可以选择原模型（若它的训练误差最低）。

**XOR 例子**（图 8.13）：两个类、两个特征、纯交互。理论上在 $x_1=0$ 处切一刀、再在每层 $x_2=0$ 处切一刀可以完美分类。但 CART（第 9 章）贪心且短视：在 $x_1$ 或 $x_2$ 上找最好的第一刀，由于数据平衡，任何第一刀看起来都没用，于是算法**几乎随机地**在顶层生成一个切分（左图）。bumping 通过 bootstrap 打破类别平衡，20 个 bootstrap 样本里**总会有一个**第一刀接近 $x_1=0$ 或 $x_2=0$，右图给出了近似最优的切分。

**两个注意点**：

1. **bumping 用训练集比较不同模型，所以各模型的复杂度必须大致相同。** 对树来说，意味着每个 bootstrap 样本上都长出**同样多终端节点**的树——因为训练误差天然偏爱更复杂的模型，不锁定复杂度就会选中最深的那棵。
2. bumping 也能用于「原始准则难优化」的问题：在一个**更方便的**代理准则上优化 bootstrap 样本，再用原始准则在训练集上挑模型。

**bumping 与 bagging 的对照**：

| | Bagging (8.51) | Bumping (8.60) |
|---|---|---|
| bootstrap 样本的用法 | 全部用来平均 | 全部用来挑选，只取最好那个 |
| 输出 | 复合模型（不再是原来的模型类） | 单个原模型 |
| 可解释性 | 差 | 好 |
| 方差 | 大幅下降 | 不一定下降（只取一个） |
| 适用 | 拟合法不稳定（树） | 拟合法容易卡在糟糕的局部解 |

**为什么 bumping 对 CART 在 XOR 上的问题有效而 bagging 无效**：bagging 只能把「单次随机顶层切分」的期望学出来——而单次切分族里**根本没有**能分开对角边界的那一刀（无论沿 $x_1$ 还是 $x_2$ 切都只能切出一条轴平行线）。bumping 则只需要**一次**运气好的抽样，就直接拿到了接近最优的树。**平均一个不含正确答案的模型族，永远得不到正确答案；挑选则可以碰运气碰对。**

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8-9">原文 §8.9</a>

---

## 8.10 习题中的关键推导 {#s-8-10}

### 8.10.1 KL 散度与 EM 的 Jensen 步骤（习题 8.1）{#s-8-10-1}

**问题**：证明 (8.47) 用到的 $R(\theta,\theta)\ge R(\theta',\theta)$。

由 (8.46)，$R(\theta',\theta)-R(\theta,\theta)=E\bigl[\log\frac{r(Y)}{q(Y)}\bigr]$，其中 $r(y)=\mathrm{Pr}(Z^m=y\mid Z,\theta')$、$q(y)=\mathrm{Pr}(Z^m=y\mid Z,\theta)$ 是两个密度：

$$
E\Bigl[\log\frac{r(Y)}{q(Y)}\Bigr]\eqno{8.61}
$$

**证明。** $Y\sim q$。取 $\varphi(x)=\log(x)$（凹，预备知识 C2），$X=\frac{r(Y)}{q(Y)}$。由 Jensen 不等式

$$
E\left[\log\frac{r(Y)}{q(Y)}\right]\ \le\ \log E\left[\frac{r(Y)}{q(Y)}\right]=\log\int \frac{r(y)}{q(y)}q(y)\,dy=\log 1=0
$$

**等号条件**：Jensen 取等当且仅当 $X$ 几乎处处为常数，即 $\frac{r(y)}{q(y)}=c$ 对所有 $y$（连续情形）；两侧积分为 1，故 $c=1$，即 $r=q$，$\theta'=\theta$。这正是 (8.47) 的等号条件。

于是 $R(\theta',\theta)-R(\theta,\theta)\le0$，代入 (8.47) 的分解得 $\ell(\theta';Z)-\ell(\theta;Z)\ge0$。

**推广：这就是 KL 散度非负性。** 定义 $\mathrm{KL}(r\|q)=E_q\bigl[\log\frac{r}{q}\bigr]$，非负性由同一条 Jensen 得到。本章用到三条性质：

- **$\mathrm{KL}(r\|q)\ge0$，等号当且仅当 $r=q$**（刚证）。这给出 EM 的**收敛性**：$\ell$ 单调且有上界，每步提升量 $\ge0$ 且趋于 0。
- **$\mathrm{KL}$ 在「$r$ 固定、$\log q$ 变」时对 $\log q$ 严格凸**，故 $\tilde P(z^m)=\mathrm{Pr}(z^m\mid Z,\theta')$ 是 8.5.3 节 (8.49) 的唯一极大点——这是习题 8.2 的另一种证法，比拉格朗日乘子更短。
- **非对称**：$\mathrm{KL}(r\|q)\ne\mathrm{KL}(q\|r)$（反向要用 $\varphi(t)=-\log t$ 才是凹的）。所以 8.5.2 节那两项 KL 的方向不能随意交换，写成 $\mathrm{KL}(\theta'\|\theta)+\mathrm{KL}(\theta\|\theta')$ 是错的。

**为什么必须用 $\log$ 而不是别的 $\varphi$**：$\log$ 的可加性让似然提升能被精确分解成「$Q$ 的提升」+「KL」。若把 $\varphi$ 取成 $t\log t$（直接用 KL 散度本身），分解里会多出 $\frac12(\mathrm{KL}(r\|q)-\mathrm{KL}(q\|r))$ 这类无法定号的项，单调性反而看不清楚。

> **结果** · **KL 散度非负**是本章唯一被反复用到的不等式。它也是 8.5.3 节最大–最大化论证的引擎：$-\mathrm{KL}\bigl(\mathrm{Pr}(Z^m\mid Z,\theta)\,\|\,\mathrm{Pr}(Z^m\mid Z,\theta')\bigr)$ 恰好是 $R(\theta,\theta)-R(\theta',\theta)$，于是 (8.47) 可以改写成

$$
\ell(\theta';Z)-\ell(\theta;Z)=\bigl[Q(\theta',\theta)-Q(\theta,\theta)\bigr]+\mathrm{KL}\Bigl(\mathrm{Pr}(Z^m\mid Z,\theta)\,\Big\|\,\mathrm{Pr}(Z^m\mid Z,\theta')\Bigr)\ \ge0
$$

**似然提升 = Q 函数的提升 + 两个条件分布之间的 KL。** 这是全章最漂亮的一行：EM 每一步的收益被精确地分解成「在 $Q$ 上走得更远」与「让后验责任更集中」两部分，两部分都非负。等号成立当且仅当 $\theta'=\theta$，所以**似然严格单调的步子里，$\hat\theta$ 一定在变**。第 14 章讨论 EM 的收敛速度时用的正是这个分解（分解速率与信息增量 $I(\theta';\theta)$ 成正比）。

### 8.10.2 EM 作为 MM 算法（习题 8.7）{#s-8-10-2}

**定义（minorize，弱化）**：函数 $g(x,y)$ 弱化 $f(x)$，若

$$
g(x,y)\ \le\ f(x),\qquad g(x,x)=f(x)\eqno{8.62}
$$

（对定义域内所有 $x,y$。）第二个条件是关键的**可触及性**：$y=x$ 时能取到 $f$，否则弱化再紧也没用。由于 $g(x,x)=f(x)$，更新规则

$$
x_{s+1}=\arg\max_x\ g(x,x_s)\eqno{8.63}
$$

使 $f$ 单调不减，因为 $f(x_{s+1})\ \ge\ g(x_{s+1},x_s)\ \ge\ g(x_s,x_s)=f(x_s)$（第一步用 (8.62) 的不等式方向，第二步用 $x_{s+1}$ 的最优性）。这就叫 **MM 算法**（Minorize–Maximize；极小化问题对应 Majorize–Minimize）。

**EM 是 MM 算法的一个例子**。取

$$
g(\theta';\theta)=Q(\theta',\theta)+\log\mathrm{Pr}(Z\mid\theta)-Q(\theta,\theta)
$$

它只含 $\theta$ 不含 $\theta'$，故 (8.63) 的 $\arg\max_x$ 就是 M 步的 $\arg\max_{\theta'}Q(\theta',\theta)$。检查 (8.62)：

$$
g(\theta';\theta)-\ell(\theta';Z)=Q(\theta',\theta)-Q(\theta,\theta)-R(\theta',\theta)+R(\theta,\theta)=-\bigl[R(\theta',\theta)-R(\theta,\theta)\bigr]=\mathrm{KL}\Bigl(\mathrm{Pr}(Z^m\mid Z,\theta)\Big\|\mathrm{Pr}(Z^m\mid Z,\theta')\Bigr)\ \ge0
$$

（最后一步用 8.10.1 节。）且 $g(\theta;\theta)=\ell(\theta;Z)$（可触及性，因为 $R(\theta,\theta)=0$）。故 $g\le\ell$ 处处成立、且在 $\theta'=\theta$ 取等，**EM 严格是 MM 算法**，单调性由 (8.63) 自动保证。

### 8.10.3 其余练习 {#s-8-10-3}

- **习题 8.2**：(8.49) 的拉格朗日推导已在 8.5.3 节给出。约束 $\tilde P\ge0$、$\sum\tilde P=1$ 下最大化 (8.48)，解为 $\tilde P(Z^m)=\mathrm{Pr}(Z^m\mid Z,\theta')$。
- **习题 8.3**：(8.50) 的全概率公式推导已在 8.6 节给出。
- **习题 8.4**：对 B 样条平滑用参数化自助法 bagging，证明 $\hat f_{\mathrm{bag}}(x)\to\hat f(x)$。**推导**：8.2.1 节已证 $\hat\mu^\star(x)\sim N\bigl(\hat\mu(x),\,h(x)^\top(H^\top H)^{-1}h(x)\hat\sigma^2\bigr)$，即 $\hat\mu^\star(x)-\hat\mu(x)$ 是零均值高斯，于是
$$
E_{\mathcal P}\bigl[\hat\mu^\star(x)\bigr]=\hat\mu(x)\ \Rightarrow\ \hat f_{\mathrm{bag}}(x)=\frac1B\sum_{b=1}^B\hat\mu^{\star b}(x)\xrightarrow{B\to\infty}E_{\mathcal P}[\hat\mu^\star(x)]=\hat\mu(x)
$$
**注意** $\hat\sigma^2$ 在各次抽样中当常数处理（用的是 $\hat\sigma^2$ 而非重估），这是保证等式成立的关键；若每次重估 $\hat\sigma^{\star 2}$，结论仍近似成立但不再精确。
- **习题 8.5、8.6**：损失函数的多分类推广（见第 10 章图 10.4）；以及骨密度数据的平滑例子——分别用 (a) CV 选平滑程度做自助百分位带、(b) (8.28) 的贝叶斯后验带、(c) 100 次自助复制，比较三者。

<a class="src" href="../esl/ch08-model-inference-and-averaging.html#s-8">原文习题</a>