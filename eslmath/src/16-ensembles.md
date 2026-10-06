---
id: m16
n: "16"
title: 集成学习
title_en: Ensemble Learning
desc: 提升的正则化路径视角：惩罚回归 (16.1)–(16.4)、单调 lasso 路径与 L1 间隔 (16.6)(16.7)、学习集成与 ISLE (16.8)–(16.13)、规则集成 (16.14)(16.15)
prev: m15
next: m17
prev_title: 第 15 章 随机森林
next_title: 第 17 章 无向图模型
---

# 16 集成学习 {#s-16}

> 本章不重复第 8 章的 bagging、第 10 章的 boosting、第 15 章的随机森林，而是给出统一视角：所有集成方法都可以看成「在一棵巨大的基函数字典 $\mathcal{T}$ 上做正则化拟合」。在这个视角下，boosting 是沿 $L_1$ 弧长单调行进的正则化路径，随机森林是等权平均，Friedman 与 Popescu 的 ISLE 则是「重要性采样 + lasso 后处理」的混合体。本章的核心结论是：集成的成败不取决于成分模型的强弱，而取决于集合的**宽度** $\sigma$（式 (16.12)）与字典的**冗余度**。

<a class="src" href="../esl/ch16-ensemble-learning.html#s-16-1">原文 §16.1</a>

## 16.1 引言：纠错输出码 {#s-16-1}

集成学习可以拆成两个任务：**从训练数据里造出一个成分学习器的总体**，然后**把它们组合成复合预测器**。第 8 章的 bagging 用重抽样造总体，第 15 章的随机森林再加上随机选分裂变量，第 10 章的 boosting 则让总体随迭代演化并用加权投票。本章把这一切统一到「基函数字典上的正则化」这一形式。

最早的多类集成方法之一是 **纠错输出码**（error-correcting output codes, ECOC；Dietterich & Bakiri, 1995）。以 10 类手写数字为例：取一个 $10\times L$ 的 0–1 编码矩阵 $C$（表 16.1，$L=15$），**每一列定义一个把 10 类合并成两类的二分类问题**。

> **推导** · 编码矩阵的构造与解码
>
> 1. 对每个列 $\ell=1,\dots,L$，用 $C_{\cdot\ell}$ 诱导的标签 $z_{i\ell}=C_{k(i)\ell}$ 训练一个二分类器，输出 $\hat p_\ell(x)\in[0,1]$（正类的预测概率）。
> 2. 第 $k$ 行的 $C_{k\cdot}$ 是第 $k$ 类的二进制码字。设计原则是**任意两行之间的 Hamming 距离要大**，使得解码时若少数几位解码错误，仍能选中正确类别。
> 3. 解码量是第 $k$ 行与「预测的 $L$ 位码字」之间的逐位差之和，即下式 (16.5)；取 $\delta_k(x)$ 最小的那个 $k$ 作为预测类别。

$$
\delta_k(x) = \sum_{\ell=1}^{L}\big\lvert C_{k\ell}-\hat p_\ell(x)\big\rvert \eqno{16.5}
$$

> **结果** · 为什么 (16.5) 有效
>
> 每个 $\ell$ 上，若二分类器准确，则 $\hat p_\ell(x)\approx C_{k\ell}$，该位贡献 $\approx 0$；错的位贡献 $\approx 1$。因此 $\delta_k(x)$ 就是「第 $k$ 类的码字与预测码字不一致的位数」的软化版本（用概率代替 0–1 硬判决），于是 $\delta_k$ 的最小值对应于**到预测码字 Hamming 距离最小**的类。
>
> 表 16.1 中 $L=15$ 而 10 类只需 $\lceil\log_2 10\rceil=4$ 位，多出的 $15-4$ 位是**冗余纠错位**；该码的最小 Hamming 距离为 $d_{\min}=7$。粗略地说，若 $L$ 位中每一位出错概率为 $p$，判对需要「错的位数 $<d_{\min}/2$」。

> **坑**
>
> - 原文同时指出，指示响应编码（indicator response coding，第 4.2 节）本身也是冗余的：它的 $d_{\min}=2$。
> - James & Hastie (1998) 的分析指出 ECOC 的主要收益其实来自**方差缩减**而非纠错：不同的编码列诱导出不同的树，解码步骤 (16.5) 在效果上等价于一次平均。所以不要把 ECOC 当作「纠错」理论来用，它在实践中更接近随机森林。

## 16.2 提升与正则化路径 {#s-16-2}

<a class="src" href="../esl/ch16-ensemble-learning.html#s-16-2">原文 §16.2</a>

第 1 版 10.12.2 节曾指出：梯度提升产生的模型序列，与高维特征空间里的正则化拟合高度相似。这一节把这个类比做实。

### 16.2.1 惩罚回归 {#s-16-2-1}

设 $\mathcal{T}=\{T_k\}_{k=1}^K$ 是所有能在训练数据上实现的 $J$ 端结点回归树构成的字典（$K=\mathrm{card}(\mathcal{T})$，通常是天文数字），把它们当作 $\mathbb{R}^p$ 中的基函数，线性模型为

$$
f(x)=\sum_{k=1}^{K}\alpha_k T_k(x) \eqno{16.1}
$$

> **推导** · 平方误差损失展开成二次型
>
> 记设计矩阵 $Z\in\mathbb{R}^{N\times K}$，$Z_{ik}=T_k(x_i)$，残差 $r=y-Z\alpha\in\mathbb{R}^N$。用 $\lVert r\rVert_2^2=(y-Z\alpha)^\top(y-Z\alpha)$ 展开：
>
> $$\lVert y-Z\alpha\rVert_2^2 = y^\top y - 2\alpha^\top Z^\top y + \alpha^\top (Z^\top Z)\,\alpha$$
>
> 三项依次是常数项、一次项、二次项（$Z^\top Z$ 对称，故交叉项无剩余）。这说明「损失」是 $\alpha$ 的二次凸函数，加上一项正则 $\lambda J(\alpha)$ 后仍是凸函数：

$$
\hat\alpha(\lambda)=\arg\min_{\alpha}\ \sum_{i=1}^{N}\Big(y_i-\sum_{k=1}^{K}\alpha_k T_k(x_i)\Big)^2+\lambda\, J(\alpha) \eqno{16.2}
$$

$J(\alpha)$ 一般是「惩罚大系数」的函数，两个最常用的例子是

$$
J(\alpha)=\sum_{k=1}^{K}\alpha_k^2 \quad(\text{ridge 回归}) \eqno{16.3}
$$

$$
J(\alpha)=\sum_{k=1}^{K}\lvert\alpha_k\rvert \quad(\text{lasso}) \eqno{16.4}
$$

> **推导** · 两种惩罚解的闭式对比
>
> **ridge**：把 (16.3) 代入 (16.2)，梯度为零给出正规方程 $\big(Z^\top Z+\lambda I_K\big)\hat\alpha=Z^\top y$，即
>
> $$\hat\alpha(\lambda)=\big(Z^\top Z+\lambda I_K\big)^{-1}Z^\top y$$
>
> **lasso**：损失不可微，改用次梯度。记残差 $r=y-Z\hat\alpha$，活动集 $S=\{k:\hat\alpha_k\neq 0\}$，KKT 条件（见预备知识 O2）给出
>
> $$Z_S^\top\big(y-Z_S\hat\alpha_S\big)=\lambda\,\mathrm{sgn}\big(\hat\alpha_S\big),\qquad Z_{S^c}^\top\big(y-Z\hat\alpha\big)\in\lambda\,[-1,1]$$
>
> 第一式在 $S$ 上仍是**最小二乘解**；第二式说明被置零的变量其残差相关性与 $\lambda$ 同量级。对 ridge 与 lasso 都成立。
>
> **两者的收缩算子**（把单个系数写出来）：
>
> $$ \begin{cases}
> \hat\alpha^{\text{ridge}}_k = \dfrac{1}{1+\lambda\sigma^2/K}\ \hat\alpha^{\text{LS}}_k & \text{在正交基下逐坐标成立}\\[6pt]
> \hat\alpha^{\text{lasso}}_k = \mathrm{sgn}\big(\hat\alpha^{\text{LS}}_k\big)\max\Big(0,\ \big\lvert\hat\alpha^{\text{LS}}_k\big\rvert-\tfrac{\lambda}{2\sigma^2}\Big) & \text{软阈值}
> \end{cases}$$
>
> 也就是说 ridge 是**线性收缩**（乘一个 $<1$ 的因子，所有系数都保留一点），lasso 是**软阈值**（小的直接归零，大的减去一个常数）。

> **结果** · 算法 16.1：前向分步线性回归
>
> $K$ 太大，$Z^\top Z$ 是 $K\times K$ 矩阵，(16.2) 直接求解不可行。但若各基函数近似互不相关，可以「一次只动一个坐标」：
>
> 1. 初始化 $\check\alpha_k=0$，$k=1,\dots,K$；取很小的 $\varepsilon>0$ 与很大的 $M$。
> 2. 对 $m=1,\dots,M$：
>    - (a) 选使残差下降最快的基函数：$(\beta^\star,k^\star)=\arg\min_{\beta,k}\sum_{i=1}^N\big[y_i-\sum_{l}\check\alpha_l T_l(x_i)-\beta T_k(x_i)\big]^2$，此时 $\beta^\star=\frac{Z_k^\top r}{\lVert Z_k\rVert_2^2}$，$r=y-Z\check\alpha$；
>    - (b) $\check\alpha_{k^\star}\leftarrow\check\alpha_{k^\star}+\varepsilon\cdot\mathrm{sign}(\beta^\star)$。
> 3. 输出 $f_M(x)=\sum_{k=1}^K\check\alpha_k T_k(x)$。
>
> **与 lasso 的定量关系**：若 $T_k$ 互不相关，则在 $\varepsilon\downarrow 0$、$M\uparrow\infty$ 且 $M\varepsilon\to t$ 时，算法 16.1 与 $L_1$ 范数界 $t=\sum_k\lvert\alpha_k\rvert$ 的 lasso 解**逐点相同**。步骤 2(b) 的「每次只加 $\varepsilon$」正是在走 $L_1$ 弧长（见 §16.5 的 (16.16)）。

> **坑**
>
> - $K>N$ 时「最小二乘值」不唯一（脚注 2），原文取其中 $L_1$ 范数最小者，而这恰是 lasso 解。
> - 上述「完全相同」的结论只要求 $\hat\alpha_k(\lambda)$ 都是 $\lambda$ 的**单调函数**，不要求严格互不相关；当变量高度相关时 $\hat\alpha_k(\lambda)$ 非单调，lasso 与算法 16.1 的解集就不同（16.2.3 节图 16.3 的实验正是这个情形）。
> - $\varepsilon=1$（不收缩）对应前向分步回归，其激进版本是子集选择 $J(\alpha)=\sum_k\lvert\alpha_k\rvert^0$，在中等密度下容易「贪心得过头」（Copas, 1983）。

### 16.2.2 「押注稀疏性」原理 {#s-16-2-2}

$L_1$ 惩罚与 $L_2$ 惩罚的差别在稀疏情形下被放大。作者的建议是：*「使用在稀疏问题上表现好的方法，因为在稠密问题上没有方法表现好。」*

用一个说明性例子（Friedman et al., 2004）：$N=10000$ 个观测，模型是 $10^6$ 棵树的线性组合。若真实系数服从高斯分布，贝叶斯意义下的最优预测器是 ridge（第 3 章练习 3.6），应当用 $L_2$；若只有约 1000 个系数非零，则 lasso 的 $L_1$ 更好。注意稠密情形下**两种方法都不好**——要估这么多非零系数而数据太少，维度灾难在收费。

> **推导** · 为什么 $L_1$ 偏好稀疏：把 KKT 的对偶形式写出来
>
> ridge 与 lasso 都可写成带线性约束的形式（第 3 章）：以 $\lVert y-Z\alpha\rVert_2\le t$ 为原始约束，则
>
> $$\begin{cases} J(\alpha)=\tfrac{1}{2}\lVert\alpha\rVert_2^2 &\Rightarrow\ \text{球约束，系数沿径向连续收缩，永不为零}\\[4pt] J(\alpha)=\lVert\alpha\rVert_1 &\Rightarrow\ \text{菱形约束，与 $\ell_1$ 球的面相交的仿射方向必落在坐标轴上} \end{cases}$$
>
> 这不是直觉：$\ell_1$ 球 $\sum_k\lvert\alpha_k\rvert\le t$ 的面由 $2^K$ 个「面片」组成，第 $q$ 个面片是 $\{\alpha:\ \alpha_k=s_k u_k,\ u\ge0,\ \sum_k u_k=t\}$。最小二乘损失沿**固定方向** $\hat\alpha-\alpha$ 线性下降（因为损失是二次的，沿直线的导数是线性的），所以最陡下降方向落在哪个面片，就决定了哪些坐标被激活。因此最优解必有 $\alpha$ 落在这 $2^K$ 个面片之一上，即**至少 $K-1$ 个坐标为 0**。而 $\ell_2$ 球的面是光滑的球面，没有这种「轴对齐」的效应。

> **延伸** · 三条限定
>
> - 稀疏/稠密是相对于**未知的真目标函数**与所选字典 $\mathcal{T}$ 而言的。
> - 稀疏度还相对于**训练样本量**与**信噪比 NSR**：样本越多、NSR 越小，能被识别为非零的系数越多。
> - 字典大小也有影响：字典越大，表示可能越稀疏，但搜索更难、方差更高。

图 16.2 用模拟验证了这一点：$n=50$，$p=300$ 高斯预测变量，上排 300 个系数全非零、中排 10 个、下排 30 个；五种 NSR。$L_2$ 惩罚处处表现差，$L_1$ 只在它能赢的两处（稀疏情形）表现合理，而且 NSR 越大越差。

### 16.2.3 正则化路径、过拟合与间隔 {#s-16-2-3}

人们常说 boosting「不过拟合」，更准确的说法是「**过拟合得很慢**」。

<a class="src" href="../esl/ch16-ensemble-learning.html#s-16-2-3">原文 §16.2.3</a>

Hastie et al. (2007) 证明 $\mathrm{FS}_0$（步长 $\varepsilon\to 0$ 的极限前向分步）拟合的是 lasso 的一个**单调版本**。做法是引入增广字典 $\mathcal{T}_a=\mathcal{T}\cup\{-T\}$，并把系数限制为非负 $\alpha_k\ge 0$；此时 lasso 路径本身为正，而 $\mathrm{FS}_0$ 路径单调不减。记 $\ell$ 为路径 $\alpha(\ell)$ 的 $L_1$ 弧长，则这条单调 lasso 路径由一个微分方程刻画，初值 $\alpha(0)=0$：

$$
\frac{\partial \alpha_m(\ell)}{\partial\ell}=\rho_m^{ml}\big(\alpha(\ell)\big) \eqno{16.6}
$$

$\rho^{ml}$ 是「每单位 $L_1$ 弧长使损失按最优二次速率下降」的单调 lasso 移动方向（速度向量）。由于对一切 $k,\ell$ 都有 $\rho^{ml}_k(\alpha(\ell))\ge 0$，解路径必单调。

> **推导** · (16.6) 的速度向量就是「单位 $L_1$ 球面上的最速下降方向」
>
> 记风险 $R(\alpha)=\sum_{i=1}^N L(y_i,\sum_k\alpha_kT_k(x_i))$，梯度 $g=\nabla R(\alpha)$。定义速度为「在 $d$ 上单位 $L_1$ 弧长内使 $R$ 下降最多」的方向，即解
>
> $$\max_{d}\ \big(-g^\top d\big)\quad\text{s.t.}\quad \lVert d\rVert_1=1,\ d\ge 0$$
>
> 写成极小化并引入拉格朗日乘子 $\mu\ge 0$（$\|d\|_1=1$）与不等式乘子 $\nu\ge0$（$d\ge0$）：
>
> $$\mathcal{L}(d,\mu,\nu)=-g^\top d+\mu\big(\lVert d\rVert_1-1\big)-\nu^\top d$$
>
> 平稳条件逐分量给出 $-g_k+\mu\,\mathrm{sgn}(d_k)-\nu_k=0$，互补松弛 $\nu_kd_k=0$。分两种情形：
>
> - 若 $d_k>0$，则互补松弛给 $\nu_k=0$，故 $g_k=\mu$；
> - 若 $d_k=0$，则 $-g_k-\nu_k=0$，$\nu_k=-g_k\ge0$，故 $g_k\le 0$。
>
> 取 $\mu=\max_k g_k$（不下降方向上的最大梯度），得
>
> $$\rho^{ml}_k(\alpha)=\begin{cases}1 & \text{若 } k\in\arg\max_j g_j(\alpha)\\ 0 & \text{否则}\end{cases}\ \ge 0$$
>
> 平方误差 $R=\tfrac12\lVert y-Z\alpha\rVert_2^2$ 时 $g=-Z^\top(y-Z\alpha)=-Z^\top r$，即「与当前残差最相关的那个基函数」，与提升算法的选树规则完全一致。

> **结果** · 对偶的普通 lasso
>
> lasso 的路径也可以写成 (16.6) 形式，区别在于移动方向是「每单位 $L_1$ **范数**下降最多」。此时方向向量可以有正有负，故 lasso 路径**不必**单调——这正是图 16.3 中 lasso 路径在后期剧烈波动的原因（该模拟用 1000 个变量、块内相关系数 $\rho=0.95$、$n=60$，lasso 吃了多重共线性的亏）。
>
> 增广字典 + 非负约束是自然的：它同时消除了「$T$ 与 $-T$ 共线」造成的符号二义性，并且与树提升一致——我们总能找到与当前残差**正相关**的树。

**归一化 $L_1$ 间隔。** 有观点认为 boosting 表现好是因为它像第 4、12 章的 SVM 一样具有最大间隔性质。定义 $f(x)=\sum_k\alpha_kT_k(x)$ 的归一化 $L_1$ 间隔为

$$
m(f)=\min_{i}\ \frac{y_i f(x_i)}{\sum_{k=1}^{K}\big\lvert\alpha_k\rvert} \eqno{16.7}
$$

> **推导** · (16.7) 的分母为什么是 $\sum_k\lvert\alpha_k\rvert$
>
> 间隔要「归一化」，就得先规定模型的尺度。SVM 用 $L_2$ 范数 $\|f\|_2=\sqrt{\sum_i f(x_i)^2}$ 归一化，因为 $\|f\|_2^2=\alpha^\top Z^\top Z\alpha$ 对 $\alpha$ 是二次型、度量的是系数向量的欧氏长度。但基函数字典里常含「成对」或尺度悬殊的元素，$\alpha^\top Z^\top Z\alpha$ 无法反映模型在数据上的真实大小，于是改用无穷范数 $\|f\|_\infty=\max_i\big\lvert f(x_i)\big\rvert$。由三角不等式，
>
> $$\lVert f\rVert_\infty=\max_i\Big\lvert\sum_k\alpha_kT_k(x_i)\Big\rvert\le\max_i\sum_k\lvert\alpha_k\rvert\,\lvert T_k(x_i)\rvert\le\Big(\sum_k\lvert\alpha_k\rvert\Big)\max_{i,k}\lvert T_k(x_i)\rvert$$
>
> 若字典里的树都已归一化到 $\max\lvert T_k\rvert\le 1$，则 $\lVert f\rVert_\infty\le\lVert\alpha\rVert_1$。取 $\lVert\alpha\rVert_1=1$ 归一化，就把间隔写成 (16.7)。

> **结果** · 三条关于间隔的定理
>
> - Schapire et al. (1998)：数据可分时，AdaBoost 每轮都使 $m(f)$ 增大，收敛到一个「对称间隔」解。
> - Rätsch & Warmuth (2002)：带收缩的 AdaBoost **渐近**收敛到 $L_1$-最大间隔解。
> - Rosset et al. (2004a)：对 (16.2) 型的一般损失，当 $\lambda\downarrow0$ 时，特定损失（AdaBoost 的指数损失、二项偏差）的解收敛到最大间隔构型。
>
> 归纳成一句：**提升得到的分类器序列，是通向最大间隔解的一条 $L_1$-正则单调路径。**

> **坑**
>
> 路径的「最大间隔端」可能是一个极差的过拟合解（图 16.5 的 mixture 数据：$m(f)$ 在 10000 棵树后趋稳，但测试误差在 240 棵树时最小，再跑就灾难性地过拟合）。早停 = 在路径上选一个点，必须靠验证集。

## 16.3 学习集成 {#s-16-3}

<a class="src" href="../esl/ch16-ensemble-learning.html#s-16-3">原文 §16.3</a>

把前面的洞察变成可操作的两阶段方案。模型仍是

$$
f(x)=\alpha_0+\sum_{k=1}^{M}\alpha_k T_k(x) \eqno{16.8}
$$

其中 $\mathcal{T}=\{T_k\}$ 是基函数字典。对梯度提升与随机森林，$\lvert\mathcal{T}\rvert$ 极大，最终模型常常包含成千上万棵树；而 16.2 节已论证，带收缩的梯度提升是在这个树空间里拟合一条 $L_1$-正则单调路径。

Friedman & Popescu (2003) 提出的混合方案把它拆成两步：

- 从训练数据里**诱导出一个有限的字典** $\mathcal{T}_L=\{T_1(x),\dots,T_M(x)\}$（可以直接取梯度提升或随机森林产出的全部树）；
- 在这个有限字典上**拟合一条 lasso 路径**，得到一族函数 $f_\lambda(x)$：

$$
\alpha(\lambda)=\arg\min_{\alpha}\ \sum_{i=1}^{N}L\Big[y_i,\ \alpha_0+\sum_{m=1}^{M}\alpha_m T_m(x_i)\Big]+\lambda\sum_{m=1}^{M}\big\lvert\alpha_m\big\rvert \eqno{16.9}
$$

> **结果** · 后处理的三点好处
>
> - **稀疏化**：朴素后处理通常把上千棵树压到约 40 棵，节省预测时的计算与存储。图 16.6（spam 数据，1000 棵深度 $m=7$ 的树）：lasso 后处理给出适度改进，后处理后的性能与梯度提升持平。
> - **去相关**（下一节）：如果原始集成里的树高度相关，$\lambda$ 惩罚无法把它们分开，后处理就没用。
> - **可移植**：改用「5% 无放回子样本 + 约 6 个端结点的浅树」后，训练成本降低约 100 倍，后处理带来的改进更显著（但绝对性能略逊于深度树版本）。

> **推导** · (16.9) 与 (16.2) 的关系
>
> 记 $W\in\mathbb{R}^{N\times M}$ 为诱导字典的设计矩阵（$W_{im}=T_m(x_i)$），$L$ 为损失。(16.9) 就是「$W$ 上的 (16.2)，$J(\alpha)=\lVert\alpha\rVert_1$，但把平方误差换成一般损失 $L$」。$M$ 有限时它可以用标准的坐标下降（预备知识 O4）求解，其 KKT 条件为
>
> $$W_m^\top\big[\nabla L\big]_0=\lambda\,\mathrm{sgn}(\alpha_m)\ \ \text{若}\ \alpha_m\neq0;\qquad \big\lvert W_m^\top[\nabla L]_0\big\rvert\le\lambda\ \ \text{若}\ \alpha_m=0$$
>
> 若 $M>N$ 则 $W^\top W$ 奇异，但 (16.9) 的解仍由 KKT 条件唯一刻画（lasso 在凸问题上解唯一，见预备知识 O2）。

### 16.3.1 学到一个好的集成 {#s-16-3-1}

不是所有 $\mathcal{T}_L$ 都适合后处理。我们希望基函数集合**在该用的地方覆盖得好**，且**彼此足够不同**。Friedman & Popescu 从数值积分与重要性采样得到了一套设计准则。

把未知函数看成一个积分：

$$
f(x)=\int_{\Gamma}\beta(\gamma)\,b(x;\gamma)\,d\gamma \eqno{16.10}
$$

$\gamma\in\Gamma$ 索引基函数；树的情形下 $\gamma$ 索引分裂变量、分裂点以及端结点取值。**数值求积**＝找 $M$ 个求值点 $\gamma_m\in\Gamma$ 与权重 $\alpha_m$，使 $f_M(x)=\alpha_0+\sum_{m=1}^M\alpha_m b(x;\gamma_m)$ 在 $x$ 的定义域上逼近 $f(x)$；**重要性采样**＝随机抽 $\gamma$，但对 $\Gamma$ 中更相关的区域给更大权重。

> **推导** · 宽度 $\sigma$：随机化引入的代价
>
> 先定义「单个基函数的质量」——只用一棵树时，最好的一棵是全局极小者 $\gamma^\star=\arg\min_{\gamma\in\Gamma}Q(\gamma)$，其中
>
> $$Q(\gamma)=\min_{c_0,c_1}\ \sum_{i=1}^{N}L\Big[y_i,\ c_0+c_1 b(x_i;\gamma)\Big] \eqno{16.11}$$
>
> $c_0,c_1$ 把 $b$ 缩放平移后接到 $f_0$ 上。引入随机化必然得到更差的 $Q$，即 $Q(\gamma)\ge Q(\gamma^\star)$。作者用**特征宽度** $\sigma$ 刻画随机采样方案 $\mathcal{S}$ 的代价：
>
> $$\sigma=E_{\mathcal{S}}\big[Q(\gamma)-Q(\gamma^\star)\big] \eqno{16.12}$$
>
> $E_{\mathcal{S}}$ 是对采样方案 $\mathcal{S}$ 的分布取期望（这一步依赖「训练数据固定、随机性只来自抽样」）。
>
> **$\sigma$ 的两个失效方向**（原文列表）：
>
> - $\sigma$ 太窄：说明太多 $b(x;\gamma_m)$ 长得一样，彼此与 $b(x;\gamma^\star)$ 相似 → 后处理挑不出东西；
> - $\sigma$ 太宽：说明 $b(x;\gamma_m)$ 分散，但可能大量是无关样本 → 字典污染，信噪比下降。
>
> **配套推导（无编号，对应第 8、15 章）**：设成分估计量 $\hat f_m(x)$ 在随机性下**独立同分布**，记
>
> $$\sigma^2=E\big[(\hat f(x)-f(x))^2\big],\qquad \rho=\mathrm{Corr}\big(\hat f_1(x),\hat f_2(x)\big)$$
>
> 平均 $\bar f(x)=\frac1M\sum_{m=1}^M\hat f_m(x)$ 的方差由二次型方差公式（预备知识 L5）给出：逐项展开
>
> $$\mathrm{Var}\big[\bar f(x)\big]=\frac1{M^2}\sum_{m,m'}\mathrm{Cov}\big(\hat f_m,\hat f_{m'}\big)=\frac1{M^2}\Big[\sum_{m=1}^M\mathrm{Var}(\hat f_m)+\sum_{m\neq m'}\mathrm{Cov}(\hat f_m,\hat f_{m'})\Big]$$
>
> 第一组共 $M$ 项、每项 $\sigma^2$；第二组共 $M(M-1)$ 项、每项 $\rho\sigma^2$，故
>
> $$\mathrm{Var}\big[\bar f(x)\big]=\frac{M\sigma^2+M(M-1)\rho\sigma^2}{M^2}=\frac{\sigma^2}{M}\big[1+(M-1)\rho\big]$$
>
> 三个边界：$\rho=0$（独立）时方差是原来的 $1/M$；$\rho\to1$ 时方差回到 $\sigma^2$，平均**完全无效**；$\rho<0$（反相关）时比 $1/M$ 降得更快。随机森林降低树间相关（第 15 章的 OOB 误差正是 $\rho>0$ 时这一项的直接体现），子采样提升 $\sigma$，两者都在压 $\rho$ 而不是压 $\sigma$。

**算法 16.2（ISLE 集成生成）。** 用子采样引入随机性：

1. $f_0(x)=\arg\min_{c}\sum_{i=1}^N L(y_i,c)$；
2. 对 $m=1,\dots,M$：
   - (a) $\gamma_m=\arg\min_{\gamma}\sum_{i\in S_m(\eta)}L\big[y_i, f_{m-1}(x_i)+b(x_i;\gamma)\big]$；
   - (b) $f_m(x)=f_{m-1}(x)+\nu b(x;\gamma_m)$；
3. $\mathcal{T}_{\text{ISLE}}=\{b(x;\gamma_1),\dots,b(x;\gamma_M)\}$。

其中 $S_m(\eta)$ 是训练样本中大小约 $N\cdot\eta$ 的子样本（$\eta\in(0,1]$，通常**无放回**）。模拟建议取 $\eta\le\tfrac12$，$N$ 大时取 $\eta\sim 1/\sqrt{N}$。减小 $\eta$ 增大随机性，因而增大 $\sigma$。参数 $\nu\in[0,1]$ 引入**记忆**：$\nu$ 越大，算法越会回避与先前找到的 $b(x;\gamma)$ 相似的基函数。

> **结果** · 已知随机化方案都是算法 16.2 的特例
>
> - **bagging**：$\eta=1$ 但**有放回**，$\nu=0$。Friedman & Hall (2007) 论证 $\eta=\tfrac12$ 无放回与 $\eta=1$ 有放回等价，前者效率高得多。
> - **随机森林**：类似，额外通过选分裂变量引入随机性。取 $\eta<\tfrac12$ 与随机森林中减小 $m$ 效果类似，但**没有**第 15.4.2 节讨论的偏差问题。
> - **带收缩的梯度提升**（10.41）：$\eta=1$，但通常 $\sigma$ 不够宽。
> - **随机梯度提升**（Friedman, 1999）：完全按此配方，推荐 $\nu=0.1$、$\eta\le\tfrac12$；两阶段合称 **ISLE**（importance sampled learning ensemble）。

图 16.8 的回归模拟用 (16.13) 作为真函数：

$$
f(X)=10\cdot\prod_{j=1}^{5}e^{-2X_j^{2}}+\sum_{j=6}^{35}X_j \eqno{16.13}
$$

$X\sim U[0,1]^{100}$（后 65 个是噪声变量），$Y=f(X)+\varepsilon$，$\varepsilon\sim N(0,\sigma^2)$，$\sigma=1.3$。

> **数值** · 核对 (16.13) 的信噪比
>
> 令 $g(X)=e^{-2X^2}$，$X\sim U[0,1]$，则
>
> $$E[g]=\int_0^1 e^{-2x^2}\mathrm{d}x=\frac{\sqrt{\pi}}{2\sqrt2}\,\mathrm{erf}\!\big(\sqrt2\big)\approx0.598,\qquad E[g^2]=\int_0^1 e^{-4x^2}\mathrm{d}x=\frac{\sqrt\pi}{4}\,\mathrm{erf}(2)\approx0.441$$
>
> 因 $X_1,\dots,X_5$ 独立，$E\big[\prod_{j=1}^5g(X_j)\big]=0.598^5\approx0.077$，所以第一项均值 $\approx0.77$；$\sum_{j=6}^{35}X_j$ 均值 $30\cdot\tfrac12=15$、方差 $30\cdot\tfrac1{12}=2.5$。合起来 $E[f]\approx15.8$。第一项的方差 $\approx100\big(0.441^5-0.077^2\big)\approx1.1$，故 $\mathrm{Var}(f)\approx2.5+1.1\approx3.6$，标准差 $\approx1.9$。于是信噪比 $\approx1.9/1.3\approx1.5$，噪声信号比 $\mathrm{Var}(\varepsilon\mid\eta(X))/\mathrm{Var}(\eta(X))\approx1.69/3.6\approx0.47$，与原文「约 2」的量级一致（原文的精确取值依赖于是否先中心化）。
>
> 结果：**子采样 GBM（浅蓝）胜过全量 GBM（橙）**，lasso 后处理版误差相当；随机森林及其后处理版都落在后面。

### 16.3.2 规则集成 {#s-16-3-2}

<a class="src" href="../esl/ch16-ensemble-learning.html#s-16-3-2">原文 §16.3.2</a>

把「树 → 规则」的变换插进集成里（Friedman & Popescu, 2003）。图 16.9 是一棵带编号结点的小树，可以从中导出下列规则（$I(\cdot)$ 为示性函数，$S,M,L\in\{1,2,3\}$ 是 $X_3$ 的三个取值）：

$$
\begin{aligned} R_1(X)&=I(X_1<2.1) \\ R_2(X)&=I(X_1\ge 2.1) \\ R_3(X)&=I(X_1\ge 2.1)\cdot I(X_3\in\{S\}) \\ R_4(X)&=I(X_1\ge 2.1)\cdot I(X_3\in\{M,L\}) \\ R_5(X)&=I(X_1\ge 2.1)\cdot I(X_3\in\{S\})\cdot I(X_7<4.5) \\ R_6(X)&=I(X_1\ge 2.1)\cdot I(X_3\in\{S\})\cdot I(X_7\ge 4.5) \end{aligned} \eqno{16.14}
$$

> **推导** · 练习 16.3：规则 1、4、5、6 精确重构这棵树
>
> 设树的四个端结点取值为 $a$（$X_1<2.1$）、$b$（$X_1\ge2.1$ 且 $X_3\in\{M,L\}$）、$c$ 与 $d$（$X_1\ge2.1,\ X_3\in\{S\}$，由 $X_7$ 继续二分）。考察
>
> $$\hat f(X)=a\,R_1(X)+b\,R_2(X)+c\big[R_5(X)+R_6(X)\big]+d\,R_4(X)$$
>
> 逐区域代入（互斥且完备）：
>
> $$\begin{cases} X_1<2.1: & R_1=1,\ \text{其余}=0 &\Rightarrow \hat f=a\\ X_1\ge2.1,\ X_3\in\{S\}: & R_2=1,\ R_5+R_6=1,\ R_4=0 &\Rightarrow \hat f=b+c\\ X_1\ge2.1,\ X_3\in\{M,L\}: & R_2=1,\ R_4=1 &\Rightarrow \hat f=b+d \end{cases}$$
>
> 三个区域恰好覆盖 $\mathbb{R}^p$，逐区域与树的取值相同，故 $\hat f$ 就是这棵树。
>
> **线性相关**：$R_3=R_5+R_6$（因为 $I(X_7<4.5)+I(X_7\ge4.5)=1$）且 $R_2=R_3+R_4$，两个等式在全部 $x$ 上成立。所以 6 条规则线性相关，**是这棵树的过完备基**。用 lasso 时这些冗余会被 $L_1$ 惩罚自动压缩掉；用最小二乘则要靠数值秩截断处理（预备知识 L3）。

对集成里的每棵树 $T_m$ 造出它的「小规则集成」$\mathcal{T}_{\text{RULE}}^m$，再把所有树的小集成合并：

$$
\mathcal{T}_{\text{RULE}}=\bigcup_{m=1}^{M}\mathcal{T}_{\text{RULE}}^{m} \eqno{16.15}
$$

之后按任何其它集成的方式处理：或平均，或用 lasso 之类的正则化方法后处理。

> **结果** · 规则集成的三点优势
>
> - **扩大模型空间**：字典从 $M$ 个函数变成 $O(M\cdot J)$ 个函数（$J$ 为树的结点数），可以带来性能提升。
> - **更易解释**：规则比树更接近人读得懂的形式，有希望化简出真正的稀疏模型。
> - **可自然增广**：常把每个变量 $X_j$ 单独作为一个规则加进 $\mathcal{T}_{\text{RULE}}$，使集成也能很好地拟合线性函数（补上前向分步不擅长的方向）。

> **坑**
>
> 图 16.10 中规则集成在 20 次重复上的均方误差与图 16.8 的最优结果「接近但不完全可比」：规则版本用**交叉验证**选最终模型，而图 16.8 报的是测试集上的最优路径点。两张图的数字不能直接比大小。

## 16.4 本章小结与文献注 {#s-bibliographic-notes-15}

<a class="src" href="../esl/ch16-ensemble-learning.html#s-bibliographic-notes-15">原文 文献注</a>

把各方法放回同一张表里比较，判据是「字典 + 惩罚 + 组合方式」三要素：

| 方法 | 总体怎么来 | 组合方式 | 等价的最优化问题 |
|---|---|---|---|
| bagging | 重抽样，$\eta=1$ 有放回 | 等权平均 | 无（方差缩减） |
| 随机森林 | 随机选分裂变量 + bagging | 等权平均 + 多数投票 | 无（方差缩减） |
| 前向分步 (16.1) | 逐次选最相关基函数 | 系数的 $L_1$ 球约束 | (16.2) + (16.4) 的 $\varepsilon\to0$ 极限 |
| 梯度提升 + 收缩 | 同上（贪心树诱导） | 同上 | 同上，单调版本 |
| lasso 后处理 (16.9) | 先造大集成 | 事后稀疏化 | (16.9) |
| ISLE (算法 16.2) | 子采样 $\eta$ + 记忆 $\nu$ | lasso 后处理 | (16.9) |
| 规则集成 (16.15) | 树 → 规则的过完备字典 | lasso 后处理 | (16.9)，字典换成规则 |

> **结果** · 一句话总结本章
>
> 集成的性能不来自「成分模型更强」，而来自两件事：**成分之间的相关被压低**（对应 §16.3.1 配套推导里的 $\rho$，以及宽度 $\sigma$ 不能太窄），以及**后处理能在字典里挑出少数几个**（要求字典不过完备也不欠完备）。第 15 章的随机森林在第一件事上做得最好（$\rho$ 低），第 10 章的提升在第二件事上做得最好（沿 $L_1$ 路径单调、系数稀疏）；ISLE 与规则集成试图同时做到两件。

> **坑**
>
> - 「$\eta$ 越大越好」是错的。图 16.8 中**子采样** GBM 反而胜过全量 GBM——说明 $\eta$ 需要调，$\sigma$ 太宽同样有害。
> - ECOC（§16.1）容易被误当成纠错理论；James & Hastie (1998) 的实验说明它主要在方差缩减层面起作用。
> - 「相关变量里应该让 lasso 走更长」没有依据。16.2.3 节的模拟说明变量高度相关时 $\hat\alpha_k(\lambda)$ 非单调，lasso 与前向分步的解集分岔，此时**前向分步路径更平滑、过拟合更晚**（图 16.3、16.4）。

> **延伸** · 文献注要点
>
> - Dietterich (2000a) 有树集成方法的综述。
> - 神经网络（第 11 章）其实更配得上「集成」这个名字：它同时学基函数（隐藏单元）和它们的组合权重。
> - SVM（第 12 章）也是一种集成：在高维特征空间里做 $L_2$-正则拟合，靠「核技巧」避免显式搜索基函数。
> - 「混合不同类型分类器」这一类做法有大量 ad-hoc 方案；Kittler et al. (1998) 给出了有原则的处理。
> - 商业软件 C5.0（Quinlan, 2004）与本章的 `Rulefit` 目标相近。

## 16.5 练习与补充 {#s-exercises-15}

<a class="src" href="../esl/ch16-ensemble-learning.html#s-exercises-15">原文 练习</a>

**练习 16.1**（块相关模拟数据的构造）。§16.2.3 节的模拟设定是：1000 个高斯变量分成 50 个块，每块 20 个变量块内相关系数 $\rho=0.95$、块间为 0；真模型的 50 个非零系数每个块里挑一个，系数取标准高斯；噪声为高斯，噪声信号比 0.72；$n=60$。

> **推导** · 相关系数矩阵的构造
>
> 用**一阶自回归（AR(1)）** 设每个块 $b$ 的相关系数矩阵为 $\Sigma_b$，$\big[\Sigma_b\big]_{uv}=\rho^{|u-v|}$（$u,v=1,\dots,20$）。这个 Toeplitz 矩阵是半正定的：它等于
>
> $$\Sigma_b = \rho^{|u-v|} \iff \text{Yule–Walker:}\ (1-\rho^2)\big[\Sigma_b^{-1}\big]_{uv}=1\ \text{若}\ u=v,\ -1\ \text{若}\ |u-v|=1,\ 0\ \text{否则}$$
>
> 具体地令 $\varepsilon_u$ 为独立标准高斯，取 $Z_1=\sqrt{1-\rho^2}\,\varepsilon_1$、$Z_{u}=\rho Z_{u-1}+\sqrt{1-\rho^2}\,\varepsilon_u$，则 $\mathrm{Corr}(Z_u,Z_v)=\rho^{|u-v|}$。最后把 50 个块的标准正交化（如 Cholesky 或特征分解）拼成 $1000\times1000$ 的 $\Sigma$，再令 $X=\Sigma^{1/2}z$，$z\sim N(0,I_{1000})$，即得所需设计矩阵。这样块内共线、块间独立，模拟的是「树与树之间高度相关」的字典，正是 §16.2.3 想考察的情形。

**练习 16.2**（$L_1$ 弧长）。设 $\alpha(t)\in\mathbb{R}^p$ 分段可微且连续，$\alpha(0)=0$，其从 0 到 $t$ 的 $L_1$ 弧长定义为

$$
\Lambda(t)=\int_{0}^{t}\big\lvert\dot\alpha(u)\big\rVert_{1}\,\mathrm{d}u \eqno{16.16}
$$

证明 $\Lambda(t)\ge\lVert\alpha(t)\rVert_1$，且等号成立当且仅当 $\alpha(t)$ 的每个分量都单调。

> **推导** · 三角不等式与等号条件
>
> 由微积分基本定理（逐分量）$\alpha(t)-\alpha(0)=\int_0^t\dot\alpha(u)\,\mathrm{d}u$，而 $\alpha(0)=0$。对积分取 $L_1$ 范数，用「积分的三角不等式」（$\int\lVert a\rVert_1\le\int\lVert a\rVert_1$ 对任意可积向量函数 $a$ 成立）：
>
> $$\big\lVert\alpha(t)\big\rVert_1=\Big\lVert\int_0^t\dot\alpha(u)\,\mathrm{d}u\Big\rVert_1=\int_0^t\Big\lVert\int_0^u\dot\alpha(s)\,\mathrm{d}s\Big\rVert_1\mathrm{d}u\le\int_0^t\Big(\int_0^u\big\lVert\dot\alpha(s)\big\rVert_1\mathrm{d}s\Big)\mathrm{d}u$$
>
> **换序积分**：区域 $\{0\le s\le u\le t\}$ 上被积函数可分离为 $\lVert\dot\alpha(s)\rVert_1$，
>
> $$\int_0^t\!\int_0^u\lVert\dot\alpha(s)\rVert_1\,\mathrm{d}s\,\mathrm{d}u=\int_0^t(t-s)\,\big\lVert\dot\alpha(s)\big\rVert_1\mathrm{d}s\le\Big(\int_0^tt\,\mathrm{d}s\Big)\int_0^t\lVert\dot\alpha(s)\rVert_1\mathrm{d}s=t\,\Lambda(t)$$
>
> 而 $\lVert\alpha(t)\rVert_1\le t\,\Lambda(t)$。**等号条件**：上面两步取等要求 (i) $\int_0^t\dot\alpha\,\mathrm{d}u$ 的三角不等式取等，即 $\dot\alpha(u)$ 在 $u$ 上几乎处处落在同一闭象限内；(ii) 积分换序后取等要求 $\lVert\dot\alpha(s)\rVert_1$ 集中在 $s=t$ 处。结合 (i)，(ii) 在非退化情形下等价于 $\dot\alpha(u)$ 几乎处处**只在一个坐标上非零**且符号不变。综合起来，等号成立当且仅当存在固定的符号向量 $s\in\{-1,+1\}^p$ 使 $s_k\dot\alpha_k(u)\ge0$ 对一切 $k$ 与几乎所有 $u$ 成立，即每个 $\alpha_k$ 都是 $t$ 的单调函数。
>
> 这正是 §16.2.3 的结论：**单调 lasso 路径（系数非负、单调不减）沿 $L_1$ 弧长参数化时，每一步的移动量恰好等于弧长的增量**，所以弧长就是「路径上走了多远」这一自然度量；而普通 lasso 路径允许系数反复变号，其 $L_1$ 弧长严格大于 $\lVert\alpha(t)\rVert_1$，即同样的系数终点要绕更远的路。

**练习 16.3**（规则与树等价）。证明用规则 $R_1,R_4,R_5,R_6$ 拟合线性回归与用对应的回归树拟合给出同一个拟合；对分类，若拟合 logistic 回归也成立。证明见 §16.3.2 的推导（等号在三个互斥区域上逐一验证）；logistic 的情形只需注意 logistic 回归在系数上是严格单调的函数族，故若两组基函数在数据上张成相同的线性空间、最小二乘系数对应相同，则拟合值也相同。

**练习 16.4**（复现图 16.2）。$n=50$，$p=300$ 独立高斯预测变量；上排 300 个系数均非零、中排 10 个、下排 30 个（系数取标准高斯）；回归时给线性预测子加标准高斯噪声，分类时经 inverse-logit 变换生成二元响应；用缩放 $\eta(X)=X^\top\beta$ 造出五种噪声信号比；两条系数路径各取 50 个 $\lambda$（对应 1 到 50 的自由度）；在无限（高斯）或 5000（二元）大小的测试集上选最优 $\lambda$，报告 20 次重复的箱线图。

> **数值** · 该模拟中 NSR 的定义与取值
>
> 回归时 $\mathrm{Var}(Y\mid\eta(X))=\sigma^2$ 且 $\mathrm{Var}(\eta(X))=\lVert\beta\rVert_2^2$，故 $\mathrm{NSR}=\sigma^2/\lVert\beta\rVert_2^2$；分类时需换成二项偏差下的对应量。原文结论：$L_2$ 惩罚处处表现差，$L_1$ 只在稀疏的两行表现合理，且 NSR 越大、模型越稠密，效果越差（分类的差异比回归小）。