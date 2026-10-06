---
id: m15
n: "15"
title: 随机森林
title_en: Random Forests
desc: 树平均与 bagging 的方差精确公式、变量重要性与 OOB 误差、特征不纯度的重要性别名
prev: m14
next: m16
prev_title: 第 14 章 无监督学习
next_title: 第 16 章 集成学习
---

# 15 随机森林 {#s-15}

第 8 章的 bagging 和第 10 章的 boosting 都属于「委员会方法」：把一堆弱预测器拼成一个强预测器。它们的差别在**委员会怎么长**——bagging 的成员各自独立地从自助样本上长、然后等权平均，成员之间同分布（i.d.）所以偏差不变，全部收益来自降方差；boosting 的成员自适应地长大、最后加权投票，成员之间不同分布，所以它能同时降偏差。

本章的随机森林（Breiman, 2001）是 bagging 的一次实质性改造：除了对**数据**做自助抽样，还在每次分裂时对**变量**做随机抽样，从而「去相关」（de-correlate）一批树。整章的分析主线只有一条——把「平均一堆有噪声的树」的均方误差精确地写成三项：树之间的相关系数、树数、以及偏差。原文见 <a class="src" href="../esl/ch15-random-forests.html#s-15-4">原文 §15.4</a>。

## 15.1 回顾与动机 {#s-15-1}

树是 bagging 的理想候选，原因有两条，且都可量化：

1. **低偏差**。一棵长到底的回归树可以拟合任意复杂的交互结构，只要叶子节点足够纯，它的近似误差就很小。
2. **高方差**。树是「不稳定」估计量：训练集里换一个观测，整棵树的结构可能大改，于是单棵树的预测方差 $\sigma^2(x)$ 很大。

把 $B$ 棵这样的树在点 $x$ 上平均（bagging 记作 $f^{*}_b$，即第 $b$ 个自助样本上长出的树）：

$$
\hat f_{\mathrm{bag}}(x)=\frac1B\sum_{b=1}^Bf^{*}_b(x),\qquad
\hat f_{\mathrm{bag}}(x)=f(x)+O_p\Big(\frac{\sigma(x)}{\sqrt B}\Big)
$$

第二式是大数律的直接结论（预备知识 P1）：只要 $\sigma(x)$ 有限，收敛速度就是 $B^{-1/2}$。

若这些树完全独立，方差被除以 $B$；若有正相关 $\rho>0$，就压不下去。这就把本章的问题说清楚了：**平均的收益上限由树与树之间的相关结构决定**。

$$
\mathrm{Var}\Big[\frac1B\sum_{b=1}^B T_b(x)\Big]\ge \frac{\sigma^2(x)}{B}
$$

随机森林的做法是在不把 $\sigma^2$ 抬得太高的前提下把 $\rho$ 压到 0.05 这样的量级，于是方差的地板 $\rho\sigma^2$ 只剩单棵树的 5%。本章逐条把这个「地板」算清楚：§15.2 给一般公式，§15.4 给出 $\rho(x)$ 与 $\sigma^2(x)$ 的严格定义和可解释的分解，§15.3 讲 OOB 误差与变量重要性这两个「免费」副产品。

## 15.2 bagging 的偏差–方差分解 {#s-15-2}

<a class="src" href="../esl/ch15-random-forests.html#s-15-2">原文 §15.2</a>

这一节把「平均 $B$ 棵树」的均方误差完整算出来。结论先摆出来：误差 = 树间相关造成的地板 + 平均带来的下降 + 与树数无关的偏差项。

### 15.2.1 单棵树的均方误差 {#s-15-2-1}

先做一次平方展开。对随机变量 $U$ 与常数 $f$：

1. 展开：$E[(U-f)^2]=E[U^2]-2f\,E[U]+f^2$。
2. 代入 $E[U^2]=E[(U-E[U])^2]+(E[U])^2=\mathrm{Var}(U)+(E[U])^2$（见预备知识 P2）。
3. 整理：$E[U^2]-f^2=\mathrm{Var}(U)+(E[U])^2-f^2=\mathrm{Var}(U)+(E[U]-f)^2$。

于是对单棵 bootstrap 树 $T_b$ 有

$$
E\big[(T_b(x)-f(x))^2\big]=\mathrm{Var}\big[T_b(x)\big]+\big(E[T_b(x)]-f(x)\big)^2
$$

> **基础知识** · 方差分解与协方差（P2、L5）
> $$
> E[(U-f)^2]=\mathrm{Var}(U)+\big(E[U]-f\big)^2,\qquad
> \mathrm{Var}\Big[\sum_{i}a_iU_i\Big]=\sum_{i}\sum_{j}a_ia_j\mathrm{Cov}(U_i,U_j)
> $$
> 以及 $\mathrm{Cov}(U,V)=\mathrm{Corr}(U,V)\sigma_U\sigma_V$；特别地 $B$ 个 i.i.d. 随机变量的平均方差为 $\sigma^2/B$。

### 15.2.2 $B$ 棵树的平均与方差的展开 {#s-15-2-2}

随机森林的回归预测就是算术平均（分类则是多数投票）：

$$
\hat f_{\mathrm{rf}}(x)=\frac1B\sum_{b=1}^B T\big(x;\Theta_b\big) \eqno{15.2}
$$

其中 $\Theta_b$ 是第 $b$ 棵树全部节点上的分裂变量、切点与叶子取值打包成的向量（$B\times(p+1)$ 量级）；它刻画的是「树长什么样」，而 $T(x;\Theta)$ 是「给定树形后对 $x$ 的预测」。见 <a class="src" href="../esl/ch15-random-forests.html#s-15-2">原文 §15.2</a> 算法 15.1。

> **推导** · $\hat f_{\mathrm{rf}}$ 的方差（L5 的双重求和）
> **第 1 步**：把平均写成线性组合，$a_b=1/B$，于是
> $$
> \mathrm{Var}\Big[\frac1B\sum_{b=1}^B T_b\Big]=\frac{1}{B^2}\sum_{b=1}^B\sum_{b'=1}^B \mathrm{Cov}(T_b,T_{b'})
> $$
> **第 2 步**：对角线项 $b=b'$ 共 $B$ 个，每个是 $\mathrm{Var}(T_b)=\sigma^2$。
>
> **第 3 步**：非对角项 $b\ne b'$ 共 $B^2-B=B(B-1)$ 个，每个是 $\mathrm{Cov}(T_b,T_{b'})=\mathrm{Corr}(T_b,T_{b'})\sigma_b\sigma_{b'}=\rho\sigma^2$（同分布，所以 $\sigma_b=\sigma_{b'}=\sigma$；等相关假设）。
>
> **第 4 步**：两项分开求和，
> $$
> \frac{1}{B^2}\underbrace{\sum_{b=1}^B\sigma^2}_{B\,\text{项}}+\frac{1}{B^2}\underbrace{\sum_{b\ne b'}\rho\sigma^2}_{B(B-1)\,\text{项}}=\frac{B\sigma^2+B(B-1)\rho\sigma^2}{B^2}
> $$
> **第 5 步**：分子提出 $B\sigma^2(1+(B-1)\rho)$，除以 $B^2$ 得 $\frac{\sigma^2}{B}\big(1+(B-1)\rho\big)$。
> **第 6 步**：把 $\frac{1+(B-1)\rho}{B}=\frac{\rho}{1}+\frac{1-\rho}{B}$ 拆开（因为 $1+(B-1)\rho=B\rho+(1-\rho)$）。
>
> $$
> \mathrm{Var}\Big[\frac1B\sum_{b=1}^B T_b\Big]=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2 \eqno{15.1}
> $$

> **结果** · (15.1) 的三项读法
> - $\rho\sigma^2$ 是**地板**：$B\to\infty$ 时它不消失。它等于把「所有树都错了，而且错得一样」的误差原样留下。
> - $\frac{1-\rho}{B}\sigma^2$ 是**平均的收益**：从 $\sigma^2$（$B=1$）降到地板，中间按 $\frac{1-\rho}{B}$ 的速度。取 $B=1/(1-\rho)$ 就能把这个项压到与地板同量级。
> - $\rho=1$ 时两项都还原成 $\sigma^2$（$B$ 棵树完全相同，平均毫无用处）；$\rho=0$ 时还原成 $\sigma^2/B$。

回到平方展开，把 (15.1) 里的方差代进去。**关键一步**：因为 $T_1,\dots,T_B$ 同分布（i.d.），交叉项的期望守恒，

$$
E\Big[\frac1B\sum_{b=1}^B T_b\Big]=\frac1B\sum_{b=1}^B E[T_b]=E[T_b],
$$

即**森林的偏差与单棵树完全一样**，bagging / 随机森林不改变偏差。于是均方误差

$$
E\big[(\hat f_{\mathrm{rf}}(x)-f(x))^2\big]=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2+\big(\mathrm{bias}_{\mathrm{rf}}(x)\big)^2
$$

> **坑** · 三个被省略的前提
> - 等相关假设：$\rho$ 是所有 $B(B-1)/2$ 对的共同相关系数。真实森林里 $\rho$ 依赖 $x$（见 (15.6)），所以这条式子只对「$x$ 固定、抽样重复」的极限情形成立。
> - 「偏差不增」依赖各树同分布。若各树在不同数据上长（如 boosting），$\frac1B\sum_bE[T_b]$ 不等于 $E[T_1]$，偏差项就要重新算。
> - **$\rho<0$ 时公式「看起来失效」**（习题 15.1）。诊断：$B\times B$ 的等相关阵 $\big[(1-\rho)I+\rho J\big]$ 要半正定就得 $\rho\ge-1/(B-1)$。固定一个负的 $\rho$ 在 $B\to\infty$ 时不可能存在；负相关必须让 $\rho$ 随 $B$ 一起放大，且「$B$ 个和为零」的约束让互斥投票（$\rho=-1$）最多只能有 2 棵树。

<a class="src" href="../esl/ch15-random-forests.html#s-15-2">原文 §15.2</a> 的图 15.1 也印证了这点：spam 数据上 bagging 的误分类率 5.4%，加入随机变量抽样后随机森林降到 4.88%，而梯度提升是 4.5%。

### 15.2.3 例子的计算 {#s-15-2-3}

取一组具体数字：单棵树在 $x$ 处的方差 $\sigma^2=25$（标准差 5），树间平均相关 $\rho=0.05$（随机森林的典型值），森林相对真值的偏差平方 $\mathrm{bias}^2=4$（标准差 2）。用 (15.1) 逐项算：

$$
\mathrm{Var}\big[\hat f_{\mathrm{rf}}(x)\big]=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2=0.05\cdot 25+\frac{0.95}{B}\cdot 25=1.25+\frac{23.75}{B}
$$

$$
\begin{aligned}
B=1&:\ 1.25+23.75=25.00\\ B=10&:\ 1.25+2.375=3.625\\ B=100&:\ 1.25+0.2375=1.4875\\ B=1000&:\ 1.25+0.02375=1.27375
\end{aligned}
$$

逐项列表：

| $B$ | 地板 $\rho\sigma^2$ | 下降项 $(1-\rho)\sigma^2/B$ | 方差合计 | 均方误差（$+$ 偏差平方 4） |
|---|---|---|---|---|
| 1 | 1.25 | 23.75 | 25.000 | 29.00 |
| 10 | 1.25 | 2.375 | 3.625 | 7.63 |
| 100 | 1.25 | 0.2375 | 1.4875 | 5.49 |
| 1000 | 1.25 | 0.02375 | 1.27375 | 5.27 |

对应的均方误差是上面加上偏差平方 4：**29.00 → 7.63 → 5.49 → 5.27**。三点观察：

1. **收敛很快**：$B=100$ 时下降项只剩 0.24，占地板的 19%；再加树几乎白费。原文图 15.4（spam 的 oob 误差）与图 15.3（California 房价，200 棵树就稳定）都印证这个数。
2. **偏差项是硬约束**：均方误差的极限是 $1.25+4=5.25$。如果 $\sigma^2=0$（完全不 bootstrap 的确定性树），方差为 0，误差就是偏差——**降方差救不了偏差**，这正是 §15.4 要给偏差单独写一条 (15.10) 的原因。
3. **相关性比树数更值钱**：若 $\rho=0.3$，同样 $\sigma^2=25$、$B=100$，方差 $=0.3\cdot25+0.7\cdot25/100=7.5+0.175=7.675$，均方误差 11.68，比 $\rho=0.05$ 的 5.49 差一倍多。**结论：先把 $\rho$ 压到 0.05，再把 $B$ 加大**；反过来做收益极小。

> **坑** · 偏差项里的 $B$ 从哪来
> 森林的偏差 $\mathrm{bias}_{\mathrm{rf}}(x)=E[\hat f_{\mathrm{rf}}(x)]-f(x)$ 不含 $B$。有些教材把公式写成 $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2+B\big(\mathrm{bias}_{\mathrm{rf}}\big)^2$ 之类，那是把「$B$ 棵**不**同分布的树各自有偏差再平均」算错了；只有在偏差随树数累积（如 boosting 的加性模型 $\sum_{b=1}^B\eta T_b$）时才出现 $B$ 的因子。

## 15.3 随机森林的细节：OOB 与变量重要性 {#s-15-3}

<a class="src" href="../esl/ch15-random-forests.html#s-15-3">原文 §15.3</a>

### 15.3.1 算法 15.1 与 $\rho$ 的来源 {#s-15-3-1}

对 $b=1,\dots,B$ 重复：抽一个大小为 $N$ 的自助样本 $\mathcal{Z}_b^{*}$，用**递归分裂**的方式在上面长一棵树，每次分裂前从 $p$ 个变量里**等概率随机抽 $m$ 个**，只在这 $m$ 个里挑最好的变量与切点；叶子小到 $n_{\min}$ 就停。回归用 (15.2) 平均，分类用多数投票。默认 $m$：分类 $\lfloor\sqrt{p}\rfloor$、最小节点 1；回归 $\lfloor p/3\rfloor$、最小节点 5。

> **基础知识** · 有放回抽样（N1、P1）
> $N$ 个点里有放回抽 $N$ 次；单个点被抽中的概率 $1-\big(\frac{N-1}{N}\big)^N$；样本均值 $\bar X^{*}$ 的期望仍是总体均值，但方差是原来的 $N/(N-1)$ 倍——这正是 bagging 对线性统计量**完全无用**的原因（习题 15.4：两个自助样本均值的相关系数 $\approx 50\%$）。

随机化变量为什么能降 $\rho$？极端情形说清楚：

- **不随机**（$m=p$）：每个节点都在全部 $p$ 个变量里找最优切点。由于各节点用的训练数据同分布，这个「最优切点」的位置也是同分布的，两棵树选的分裂变量、切点几乎必然重合，于是 $\rho(x)\to 1$，(15.1) 里的 $\rho\sigma^2$ 项吃掉全部方差。
- **随机抽 $m$ 个**：即使最优的那个变量存在，它也有 $(1-\frac{m-1}{p})^{k}$（$k$ 为节点数）的概率在某个节点上根本没被抽中，于是两棵树在同一点上「各说各话」的概率变大，$\rho$ 下降。

> **结果** · $m$ 与 $\rho$ 的量化（原文图 15.9）
> 每次分裂抽中某个相关变量（$p$ 个变量里 $J$ 个相关）的概率是超几何式的：
> $$
> \Pr\big\{\text{某相关变量被选中}\big\}=1-\frac{\binom{p-J}{m}}{\binom{p}{m}},\qquad
> \frac{\binom{p-J}{m}}{\binom{p}{m}}=\prod_{i=0}^{m-1}\frac{p-J-i}{p-i}
> $$
> 两个组合数相除时阶乘逐个约掉，剩下这个连乘。原文图 15.7 的例子 $p=106$、$J=6$、$m\approx\sqrt{106}\approx 10$：乘积 $=\frac{100}{106}\cdot\frac{99}{105}\cdot\frac{98}{104}\cdot\ldots\cdot\frac{91}{97}\approx 0.543$，故概率 $\approx 0.457\approx0.46$，与原文一致。这个数一旦变小，「相关变量被漏掉」就频繁发生，$\rho$ 虽然也小，但偏差暴涨，误差反而变差。
>
> **坑** · 减小 $m$ 是双刃剑
> (15.1) 里 $m$ 同时出现在两处：$m\downarrow\Rightarrow\rho\downarrow$（好），但也使单棵树更弱、$\sigma^2$ 与偏差上升（坏）。原文图 15.10 用全方差公式把两者分开（见 (15.9)），结论是存在一个经典偏差–方差最优的 $m$。California 房价数据上 $m=6$ 远好于默认的 $\lfloor 8/3\rfloor=2$。

### 15.3.2 OOB 样本与 OOB 误差 {#s-15-3-2}

**定义**：对第 $i$ 个训练观测 $z_i=(x_i,y_i)$，只用那些「$\mathcal{Z}_b^{*}$ 不含 $z_i$」的树来构造预测器：

$$
\hat f_{\mathrm{rf}}^{\,(-i)}(x)=\frac{1}{K_i}\sum_{b\in\mathcal{B}_{\mathrm{oob}}(i)}T_b(x),\qquad K_i=\big|\mathcal{B}_{\mathrm{oob}}(i)\big|
$$

**$K_i$ 有多大**。$z_i$ 一次没被抽中的概率：每次抽中的概率是 $1/N$（$N$ 个点里有 $z_i$ 就抽到它），$N$ 次独立抽样全落空：

$$
\Pr\big\{z_i\notin\mathcal{Z}_b^{*}\big\}=\Big(1-\frac1N\Big)^{N}\xrightarrow[B\to\infty]{}e^{-1}\approx 0.3679
$$

（用 $\lim_{N\to\infty}(1-\frac1N)^N=e^{-1}$，$\ln$ 展开：$N\ln(1-\frac1N)=-\frac1N-\frac{1}{2N^2}-\ldots\to-1$。）所以期望有 $0.368N$ 棵树可以投票；$K_i\sim\mathrm{Binomial}\big(B,(1-\frac1N)^N\big)\approx\mathrm{Binomial}(B,0.368)$，$B=2500$ 时标准差只有 $\sqrt{2500\cdot0.368\cdot0.632}\approx 24$，非常稳。这就是「随机森林不用交叉验证就能自己报告误差」的原因。

> **推导** · OOB 误差是测试误差的无偏估计（习题 15.2）
> **第 1 步（条件化在训练集上）**：固定 $\mathcal{Z}=\{z_1,\dots,z_N\}$。$\mathcal{B}_{\mathrm{oob}}(i)$ 由「哪些自助样本漏掉了 $z_i$」决定，这部分与树的生长无关，只依赖自助抽样的随机数。
>
> **第 2 步**：给定 $i$ 被漏掉 $K$ 次这个事件，那 $K$ 个自助样本是从**不含 $z_i$ 的 $N-1$ 个点的经验分布**里独立、有放回抽出来的（因为「漏掉 $z_i$」这一事件下，其余 $N-1$ 个点仍等概率）。因此这 $K$ 棵树是 i.i.d. 的，其预测 $T_b(x_i;\Theta_b)$ 相互独立，均值方差 $\sigma_{-i}^2(x_i)/K$。
>
> **第 3 步**：由 (15.1) 的推导，$K\to\infty$ 时
> $$
> \hat f_{\mathrm{rf}}^{\,(-i)}(x_i)\xrightarrow{\;K\to\infty\;}E_{\Theta\mid\mathcal{Z}_{-i}}T\big(x_i;\Theta(\mathcal{Z}_{-i})\big)
> $$
> 右端正是「用 $\mathcal{Z}\setminus\{z_i\}$ 这 $N-1$ 个点训练一个无限棵树的随机森林」的预测值。
>
> **第 4 步**：对 $i$ 求平均，$\frac1N\sum_{i=1}^N\big(y_i-\hat f^{\,(-i)}(x_i)\big)^2$ 就是 **$N$ 折交叉验证（留一法）误差**的估计，因为每一折都恰好吃掉了 $z_i$。而无限棵树的误差与测试误差只差 $\frac{1-\rho}{B'}\sigma^2$，随 $B'\to\infty$ 消失。
>
> **结论**：$B\to\infty$ 时 OOB 误差 $\to$ 留一交叉验证误差 $\to$ 测试误差。原文说「几乎完全相同」；图 15.4（spam）里 Wilcoxon 检验的均值差 $p$-值 $=0.007$，非常接近但不完全相同——差额就来自第 3 步里那个还没消掉的 $\frac{1-\rho}{K_i}\sigma^2$。

> **坑** · OOB 的前提
> - 需要真自助抽样。若像某些实现那样用「子采样」（不放回抽 $N/2$），每个点约有一半概率落选，$K_i\approx B/2$，但**各树之间的相关会上升**，所以不能把 OOB 当成免费午餐照搬到子采样版本（Friedman & Hall, 2007 的替代方案）。
> - 单次拟合的树数必须够大。$B=50$ 时 $\sqrt{50\cdot0.368\cdot0.632}\approx3.4$，$K_i$ 波动可达 $\pm10$，OOB 误差会明显抖。
> - OOB 只在**不调参**的前提下是测试误差的无偏估计；一旦用 OOB 去选 $m$、$n_{\min}$，就有选择偏差，需要另做交叉验证。

### 15.3.3 变量重要性 {#s-15-3-3}

两种度量（对应原文图 15.5 的左、右两幅）。

**（一）分裂不纯度下降**。回归时判据取节点残差平方和，分类时取 Gini 指数 $I(m)=\frac{N(m)}{N}p(m)\big[1-p(m)\big]$。对变量 $j$：

$$
\mathrm{VI}_{\mathrm{imp}}(j)=\frac1B\sum_{b=1}^{B}\ \sum_{m\,:\,m\ \text{在}\ X_j\ \text{上分裂}}\Big[\mathcal{C}(m_0)-\mathcal{C}(m_1)-\mathcal{C}(m_2)\Big]
$$

推导就是「累加每一次分裂带来的判据下降」：节点 $m$ 被 $X_j$ 以切点 $t$ 分成 $m_1,m_2$ 后，$\mathcal{C}(m_0)-\mathcal{C}(m_1)-\mathcal{C}(m_2)\ge0$（纯节点不再分裂，这项就是 0），把整棵树里所有用 $X_j$ 的节点的下降加起来，就是 $X_j$ 在这棵树里的总贡献；对 $B$ 棵树取平均即得。这个量纲是判据单位，随 $B$ 变化，所以必须除以 $B$ 才可比。

**（二）OOB 置换型重要性**。长第 $b$ 棵树时，把它的 OOB 样本沿树往下送，预测；然后**只在 OOB 样本里**随机打乱 $X_j$ 的取值，再预测一次，精度下降量：

$$
\mathrm{VI}_{\mathrm{perm}}(j)=\frac1B\sum_{b=1}^{B}\Big[A_o\big(T_b\big)-A_o\big(T_b^{\,\pi(j)}\big)\Big],\qquad A_o=\text{OOB 精度}
$$

其中 $T_b^{\,\pi(j)}$ 表示「只用 $T_b$ 预测，但输入的 $X_j$ 被置换」。这在**线性模型里有精确对应**：把线性模型的变量置换掉，测试均方误差的期望上升 $2\hat\beta_j^2$（见 (15.13)），所以这种重要性衡量的是「这个变量在自己模型里贡献了多少预测精度」。

> **坑** · 置换重要性不等于「去掉变量」的效果
> 若把变量 $j$ 从模型里**删掉并重训**，其它变量可以充当代理（surrogate）来补偿；置换只是「把它的效果置零」，不让别人补位。原文明说这不是「若该变量不可用时预测会怎样」。

**变量重要性的偏差与条件重要性**。纯度型重要性有系统偏差：强变量在每个节点几乎总能赢，被选走之后，留给弱变量的「可用下降」就不存在了，于是弱变量被**低估**（一个与 $y$ 完全无关的变量只要被抽到、又被一个偶然有利的切点切分，也有非零的 $\mathrm{VI}$）；反过来，若 $J$ 个相关变量高度相关，随机抽样会让其中一个反复独占分裂点，其余的接近 0，这就是重要性矩阵的成团高估。Xu & Raghavan (2002) 把这个问题叫做**高维偏差**。

修正办法是**条件重要性**：先把 $X_j$ 按其自身分箱（如四分位），只在**同一个箱内**打乱取值，再算精度下降：

$$
\mathrm{VI}_{\mathrm{cond}}(j)=\frac1B\sum_{b=1}^{B}\Big[A_o(T_b)-A_o\big(T_b^{\,\pi(j)\mid\mathrm{bin}}\big)\Big]
$$

这样 $X_j$ 与其它变量的联合结构被打乱，但 $X_j$ 自身的边缘分布（以及与 $y$ 的主效应）保留，重要性不会被「抢走效应」压低。

> **延伸** · 邻近矩阵（proximity）
> 长森林时累加一个 $N\times N$ 矩阵：每棵树里若 $z_i,z_j$ 落在**同一个终端节点**（且二者都是该树的 OOB），就 $P(i,j)\mathrel{+}=1$，
> $$
> P(i,j)=\frac1B\sum_{b=1}^{B}\mathbf{1}\big\{T_b(z_i)\ \text{与}\ T_b(z_j)\ \text{同一叶子}\big\}
> $$
> 再用多维尺度法（§14.8）降到二维作图。原文图 15.6 指出：纯类区域的点落在星形图的臂端（纯叶子不再分裂，同类邻点极易同桶），靠近边界的点落在中心；不同类的近邻**有时**同叶、有时不，所以星形结构在任何数据上都差不多，这削弱了它的实用性。

### 15.3.4 「随机森林不会过拟合」到底对不对 {#s-15-3-4}

令 $B\to\infty$，(15.2) 依大数律（$T_b$ 同分布、有限二阶矩）收敛到一个极限：

$$
\lim_{B\to\infty}\hat f_{\mathrm{rf}}(x)=E_{\Theta}T(x;\Theta)\eqno{15.3}
$$

其中 $\Theta$ 的分布**条件于训练数据** $\mathcal{Z}$：把「抽自助样本」和「每次抽变量」这两层随机性都积分掉了。所以严格说 (15.3) 是「在 $\mathcal{Z}$ 上不再变」的极限——增加 $B$ 本身不会过拟合（序列 $f_B$ 一致收敛到一个固定函数）。

> **坑** · 极限本身可以过拟合
> 那个极限是 $B$ 棵**长到底**的树的平均，等于一个过丰富的模型：树越多深、节点越小，模型越能记住训练集。原书指出 Segal (2004) 通过控制树深能小幅改善性能；作者的经验是用长到底的树代价不大，还少一个要调的参数。图 15.8（12 个变量里 2 个加性）显示控制深度收益有限，而**分类问题对这种过拟合几乎不敏感**（$0$–$1$ 损失对训练响应的过拟合不敏感，见 §7.3.1），所以图 15.8 的现象在分类上基本看不到。

## 15.4 随机森林的分析 {#s-15-4}

<a class="src" href="../esl/ch15-random-forests.html#s-15-4">原文 §15.4</a>

下面固定一个目标点 $x$，假定回归 + 平方损失，真条件均值记为 $\mu(x)=E[Y\mid X=x]$。**两层随机性**要分清：

- 外层：训练集抽样 $\mathcal{Z}\sim P$，每个 $\mathcal{Z}$ 上得到 $B$ 棵树；
- 内层：在给定的 $\mathcal{Z}$ 上，自助抽样与每次抽变量这两层随机性，用 $\Theta\mid\mathcal{Z}$ 表示。

### 15.4.1 极限森林与 (15.1) 的推广 {#s-15-4-1}

$B\to\infty$ 时森林收敛到一个**只依赖 $\mathcal{Z}$** 的函数：把内层随机性全部平均掉（(15.3) 里的 $E_\Theta$ 再对 $\mathcal{Z}$ 取条件期望）：

$$
\hat f_{\mathrm{rf}}(x)=E_{\Theta\mid\mathcal{Z}}T\big(x;\Theta(\mathcal{Z})\big)\eqno{15.4}
$$

把 (15.1) 用在「$B$ 个从 $(\mathcal{Z},\Theta)$ 联合分布独立抽取的树」上，就得到有限 $B$ 时的森林方差。把 $\sigma\to\sigma(x)$、$\rho\to\rho(x)$ 并写出完整三项：

$$
\mathrm{Var}\big[\hat f_{\mathrm{rf}}(x)\big]=\rho(x)\sigma^2(x)+\frac{1-\rho(x)}{B}\sigma^2(x)\eqno{15.5}
$$

> **推导** · 为什么 (15.1) 可以直接搬过来
> **第 1 步**：$T_1,\dots,T_B$ i.i.d.（每棵树用独立的自助抽样与独立的抽变量序列），所以 $\frac1B\sum_bT_b$ 的方差按 (15.1) 的一般形式等于 $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$，其中 $\sigma^2=\mathrm{Var}(T_b)$、$\rho=\mathrm{Corr}(T_b,T_{b'})$。
> **第 2 步**：无偏性给出 $E[\frac1B\sum_bT_b]=E[T_b]$，故该方差同时等于 $\mathrm{Var}(\hat f_{\mathrm{rf}})$（对 $\mathcal{Z}$ 与 $\Theta$ 求）。
> **第 3 步**：$\mathrm{Var}(T_b)$ 里同时含内层与外层随机性，故记 $\sigma^2(x)$；$\mathrm{Corr}(T_b,T_{b'})$ 同理记 $\rho(x)$。代回即 (15.5)。
>
> **注意**：$\rho(x)$ 与 $\sigma^2(x)$ 都**依赖 $x$**，所以 (15.5) 不是一个全空间的等式，而是逐点成立；这正是原书要专门警告 $\rho(x)$ 不能与「给定数据下 $B$ 棵已拟合树的平均相关」混淆的原因——后者是 $N$ 维向量之间的相关，与平均过程中的 $\rho(x)$ 不是一回事。

两者的严格定义：

$$
\rho(x)=\mathrm{Corr}\Big[T\big(x;\Theta_1(\mathcal{Z})\big),\ T\big(x;\Theta_2(\mathcal{Z})\big)\Big]\eqno{15.6}
$$

$$
\sigma^2(x)=\mathrm{Var}_{\Theta,\mathcal{Z}}\Big[T\big(x;\Theta(\mathcal{Z})\big)\Big]\eqno{15.7}
$$

这里 $\Theta_1,\Theta_2$ 是「在**随机抽到的** $\mathcal{Z}$ 上」长出的一对独立的随机森林树，所以 (15.6) 是对 $Z$ 的抽样分布与 $\Theta$ 的抽样分布共同诱导的**无条件**相关。

> **坑** · 条件相关恒为零
> 固定 $\mathcal{Z}$ 后，两棵树的 bootstrap 与抽变量序列 i.i.d. 独立，于是
> $$
> \mathrm{Cov}\big(T(x;\Theta_1),T(x;\Theta_2)\mid\mathcal{Z}\big)=E\big[T(x;\Theta_1)T(x;\Theta_2)\mid\mathcal{Z}\big]-E[T(x;\Theta_1)\mid\mathcal{Z}]^2=E_{\Theta\mid\mathcal{Z}}[T(x;\Theta)]^2-\big(E_{\Theta\mid\mathcal{Z}}T(x;\Theta)\big)^2=0
> $$
> （内层交叉期望用独立性裂开成乘积，再用塔性质）。**整棵森林的 $\rho(x)>0$ 全部来自 $\mathcal{Z}$ 的抽样波动**——训练集只换一点点，两棵树就会选出同样的变量。这条结论在 §15.5 变成 (15.12) 的公式。

### 15.4.2 (15.9)：总方差 = 森林的抽样方差 + 组内方差 {#s-15-4-2}

原文用一个具体的仿射模拟模型（图 15.7/15.9/15.10 的底板）来展示 $m$ 的作用：

$$
Y=\sum_{j=1}^{p}\beta_jX_j+\varepsilon,\qquad X_1,\dots,X_p,\varepsilon\ \text{皆 i.i.d.}\ N(0,1)\eqno{15.8}
$$

（用 500 个大小为 100 的训练集、600 个测试点。注意它是**线性**的，所以作者特别指出：正因为真模型线性，岭回归在这个例子上更优。）

关键是把 (15.7) 的 $\sigma^2(x)$ 拆成两块。直接用塔性质（见预备知识 P2）：

$$
\mathrm{Var}_{\Theta,\mathcal{Z}}\big[T(x;\Theta(\mathcal{Z}))\big]=E_{\mathcal{Z}}\Big[E_{\Theta\mid\mathcal{Z}}\big[T^2\mid\mathcal{Z}\big]\Big]-E_{\Theta,\mathcal{Z}}\big[T\big]^2
$$

第一项里 $E_\Theta[T^2\mid\mathcal{Z}]=\mathrm{Var}_{\Theta\mid\mathcal{Z}}[T\mid\mathcal{Z}]+\big(E_{\Theta\mid\mathcal{Z}}T\big)^2$，代入 (15.4) 得 $\mathrm{Var}_{\Theta\mid\mathcal{Z}}[T\mid\mathcal{Z}]+\hat f_{\mathrm{rf}}(x)^2$。逐项整理：

$$
\mathrm{Var}_{\Theta,\mathcal{Z}}\big[T(x;\Theta(\mathcal{Z}))\big]=\underbrace{E_{\mathcal{Z}}\mathrm{Var}_{\Theta\mid\mathcal{Z}}\big[T(x;\Theta(\mathcal{Z}))\big]}_{\text{组内方差（within-}Z\text{，随机化造成）}}+\underbrace{\mathrm{Var}_{\mathcal{Z}}\big[\hat f_{\mathrm{rf}}(x)\big]}_{\text{整个森林的抽样方差}}\eqno{15.9}
$$

（两个 underbrace 的说明文字只是记号；原书写作 `Total Variance = $\mathrm{Var}_{\mathcal{Z}}\hat f_{\mathrm{rf}}(x)$ + within-$Z$ Variance`。）

即原文那句 `总方差 = $\mathrm{Var}_{\mathcal{Z}}\hat f_{\mathrm{rf}}(x)$ + 组内方差`。**这就是 (15.5) 与 (15.7) 的联立方程**：把 $\sigma^2(x)$ 从 (15.7) 换成 (15.9) 的右边，再代进 (15.5)，可以在只知其中两个量时解出第三个。

> **结果** · 图 15.9/15.10 的读法
> - $m\downarrow\Rightarrow\rho(x)\downarrow$：两棵树若分裂变量不同，预测就不同。
> - $m\downarrow\Rightarrow$ 组内方差**上升**：随机化让每棵树更野。
> - $m$ 在很大范围内变化时，**总**方差 $\sigma^2(x)$ 几乎不动（图 15.10 左图那条水平线），即「更野」的树主要是把方差从「跨数据集」挪到「组内」，总量守恒。
> - 于是在 (15.5) 里 $\rho\sigma^2$ 这项随 $\rho$ 一起下降，**森林的方差被显著压低**：第 4 节的例子 $\sigma^2=25,\rho=0.05,B=200$ 时森林方差 $=1.25+23.75/200=1.369$，是单棵树的 5.5%。

### 15.4.3 偏差 {#s-15-4-3}

偏差沿用第 2 节那条「$B$ 个 i.i.d. 平均不改变期望」的论证，外层对 $\mathcal{Z}$ 再取一次：

$$
\mathrm{Bias}(x)=\mu(x)-E_{\mathcal{Z}}E_{\Theta\mid\mathcal{Z}}\big[T(x;\Theta(\mathcal{Z}))\big]\eqno{15.10}
$$

> **坑** · 三层「$E$」别搞混
> - (15.4) 里的 $E_{\Theta\mid\mathcal{Z}}$ 已经把内层随机性（bootstrap + 抽变量）平均掉了，**不含** $\mathcal{Z}$ 的波动；
> - (15.10) 才再对 $\mathcal{Z}$ 求一次，$\mathrm{Bias}(x)$ 完全由「用一份有限样本训练出长到底的树」这件事造成，与 $B$ 无关；
> - 换一个 $\mathcal{Z}$，偏差就变，所以偏差的**平方**才可估（原文用 500 次重复模拟来估 $\mathrm{Bias}^2$）。
>
> 原文还指出：$|\mathrm{Bias}(x)|$ 通常**大于**在同一个 $\mathcal{Z}$ 上长到底但不分层随机化的树的偏差——随机化与候选子集都限制了每棵树能看到的信号。因此 (15.4) 情形下随机森林的 $\mathrm{Bias}^2$ 高于单棵满树（图 15.10 右图的水平线），**bagging/随机森林的全部收益来自降方差**。

**$m$ 的偏差–方差权衡**。图 15.10 右图同时画了 MSE、$\mathrm{Bias}^2$、$\mathrm{Var}$：$m$ 减小时 $\mathrm{Bias}^2$ 单调上升、$\mathrm{Var}$ 单调下降，交点给出最优 $m$。作者把这类行为与**岭回归**类比：岭回归把强相关变量一起向零收缩，让每个相关变量都以（被削弱的）方式发声；小 $m$ 的随机森林做的是类似的平均——每个相关变量轮流当主角，集成平均又压低任何单个变量的贡献。

> **延伸** · 随机森林是一种自适应近邻
> 每棵树都长到底，所以某个固定 $\Theta^{*}$ 的 $T(x;\Theta^{*}(\mathcal{Z}))$ 就是**某个训练样本的响应值**；生长算法为它找了一条「最优」路径，用的是它手头最有信息量的预测变量。最终的森林预测是这些训练响应的加权平均，权重随「与 $x$ 同叶的次数」而定——这正是一个以近邻为原点的核。于是随机森林可以看成 $k$-近邻的加权版（图 15.11 把两者的判决边界对比在同一份 mixture 数据上）。

## 15.5 习题中的三个公式 {#s-15-5}

<a class="src" href="../esl/ch15-random-forests.html#s-15">原文 §15.5 习题</a>

### (15.11) Mease–Wyner 模型与 Bayes 误差 {#s-15-5-1}

习题 15.3 的模型：$X\sim U[0,1]^p$，$J\le p$ 是预先给定的**偶数**，$0\le q\le\frac12$：

$$
\Pr(Y=1\mid X)=q+(1-2q)\cdot I\Big\{W>\tfrac{J}{2}\Big\},\qquad
W=\sum_{j=1}^{J}I\big\{X_j>\tfrac12\big\}\eqno{15.11}
$$

（$I\{\cdot\}$ 是指示函数：$X_j$ 落在 $(1/2,1]$ 算 1 票。）

**概率曲面**：前 $J$ 个变量「投票」，多数票为 1 时 $\Pr(Y=1)$ 从 $q$ 跳到 $q+(1-2q)=1-q$，中间是一个宽度 $2q$ 的「平台」；其余 $p-J$ 个变量完全不出现，所以它们**增加维度但不改边界**。

**Bayes 误差**。每个 $X_j$ 以概率 $1/2$ 落在 $(1/2,1]$，所以 $W\sim\mathrm{Binomial}(J,\frac12)$；$J$ 为偶数时票数关于 $J/2$ 对称，于是

$$
\pi:=\Pr\big(W>J/2\big)=\frac{1-\Pr(W=J/2)}{2}<\frac12
$$

例如 $J=2$ 时 $\Pr(W=1)=\frac12$，故 $\pi=\frac{1-\frac12}{2}=\frac14=\Pr(W=2)$ ✓；$J=6$ 时 $\Pr(W=3)=\frac{20}{64}=0.3125$，故 $\pi=0.34375$。

最优规则是 Bayes 规则：$A$ 上判 1，$\bar A$ 上判 0（因为 $1-2q\ge0$，$\Pr(Y=1\mid X)$ 在 $A$ 上更大），误差

$$
\pi\cdot q+(1-\pi)(1-q)
$$

与「恒判多数类」的误差比较：后者为 $\min\{\Pr(Y=1),\Pr(Y=0)\}$，而 $\Pr(Y=1)=q+(1-2q)\pi$。把前者减去 $\Pr(Y=0)=1-q-(1-2q)\pi$：

$$
\pi q+(1-\pi)(1-q)-\big[1-q-(1-2q)\pi\big]=\pi q+1-\pi-q+\pi q-1+q+(1-2q)\pi=0
$$

两者**完全相等**，所以 Bayes 误差 $=1-q-(1-2q)\pi$，且恒判多数类也是最优——这是模型「信噪比低」的表现。

> **结果** · 对随机森林的意义
> $p$ 个变量里只有 $J$ 个相关，噪声变量越多，随机森林每次分裂抽中相关变量的概率越低（§15.3 的 $1-\binom{p-J}{m}{\binom{p}{m}}$），因此它在这个模型上退化；boosting 用贪心分裂不受此限。

### (15.12) 树对相关性的精确表达式 {#s-15-5-2}

习题 15.5 要求把 $\rho(x)$ 写出来：

$$
\rho(x)=\frac{\mathrm{Var}_{\mathcal{Z}}\big[E_{\Theta\mid\mathcal{Z}}T(x;\Theta(\mathcal{Z}))\big]}{\mathrm{Var}_{\mathcal{Z}}\big[E_{\Theta\mid\mathcal{Z}}T(x;\Theta(\mathcal{Z}))\big]+E_{\mathcal{Z}}\mathrm{Var}_{\Theta\mid\mathcal{Z}}\big[T(x;\Theta(\mathcal{Z}))\big]}\eqno{15.12}
$$

> **推导**
> **第 1 步**：记 $m(\mathcal{Z})=E_{\Theta\mid\mathcal{Z}}[T(x;\Theta(\mathcal{Z}))]=\hat f_{\mathrm{rf}}(x)$（用 (15.4)），并令 $e=T(x;\Theta(\mathcal{Z}))-m(\mathcal{Z})$。于是 $E[e\mid\mathcal{Z}]=0$，且 $T=m+e$。
> **第 2 步**：对两棵独立树（内层随机性独立），$\mathrm{Cov}(T_1,T_2)=E[T_1T_2]-\big(E[m]\big)^2$。
> **第 3 步**：用塔性质 $E[T_1T_2]=E_{\mathcal{Z}}\big[E_{\Theta\mid\mathcal{Z}}[T_1T_2]\big]$；再在内层用独立性 $T_1=m+e_1,\ T_2=m+e_2$，得 $E[T_1T_2\mid\mathcal{Z}]=m^2+E[e_1e_2\mid\mathcal{Z}]=m^2+0$（条件均值零 ⇒ 交叉期望为零）。
> **第 4 步**：于是 $\mathrm{Cov}(T_1,T_2)=E_{\mathcal{Z}}[m(\mathcal{Z})^2]-\big(E_{\mathcal{Z}}m\big)^2=\mathrm{Var}_{\mathcal{Z}}\big[m(\mathcal{Z})\big]$。
> **第 5 步**：分母用 $\sigma^2(x)=\mathrm{Var}_{\Theta,\mathcal{Z}}[T]$，再用 (15.9) 拆成 $\mathrm{Var}_{\mathcal{Z}}[m]+E_{\mathcal{Z}}\mathrm{Var}_{\Theta\mid\mathcal{Z}}[T]$。两式相除即 (15.12)。

> **结果** · (15.12) 的解读
> 分子是**组间**（跨训练集）方差，分母是总方差，所以 $\rho(x)$ 读作「单棵树预测的波动里，有多大比例来自换了训练集」。结合 §15.4 的例子：$\sigma^2=25$、组内方差 $23.75$、$B=200$ 时森林方差 $1.369$，由 (15.5) 反解 $\rho(x)=\frac{1.369-25/200}{25}\approx0.0548$，与 (15.12) 给的 $25-23.75=1.25$ 对得上。

### (15.13) 线性模型里的置换重要性 {#s-15-5-3}

习题 15.7：$N$ 个观测、$p$ 个变量，全部标准化（$\bar x_j=0$，$\frac1N\sum_i x_{ij}^2=1$），$\hat\beta$ 是 OLS 解；$\mathrm{RSS}=\frac1N\sum_i r_i^2$ 是**训练**均方残差，$\mathrm{RSS}_j^{*}$ 用**同一个** $\hat\beta$ 计算，但先把第 $j$ 列的 $N$ 个取值随机置换。结论：

$$
E_{\mathcal{P}}\big[\mathrm{RSS}_j^{*}-\mathrm{RSS}\big]=2\hat\beta_j^2\eqno{15.13}
$$

> **推导**（$\mathcal{P}$ 是置换分布，见预备知识 P1 的期望定义）
> **第 1 步**：把残差写成 $r_i=y_i-\sum_{k=1}^{p}\hat\beta_kx_{ik}$，置换后的「残差」是
> $$
> y_i-\sum_{k}\hat\beta_kx_{ik}^{(j)}=r_i+\hat\beta_j\big(x_{ij}-x_{ij}^{(j)}\big),\qquad
> \mathrm{RSS}_j^{*}=\frac1N\sum_i\Big[r_i+\hat\beta_j\big(x_{ij}-x_{ij}^{(j)}\big)\Big]^2
> $$
> **第 2 步**：展开成三项——$\mathrm{RSS}+\frac{2\hat\beta_j}{N}\sum_ir_i\big(x_{ij}-x_{ij}^{(j)}\big)+\frac{\hat\beta_j^2}{N}\sum_i\big(x_{ij}-x_{ij}^{(j)}\big)^2$。
> **第 3 步（交叉项为零）**：$E_{\mathcal{P}}[x_{ij}^{(j)}]=\bar x_j=0$，所以 $E_{\mathcal{P}}\sum_ir_ix_{ij}^{(j)}=\sum_ir_i\bar x_j=0$；而 $\sum_ir_ix_{ij}=0$ 来自 OLS 正则方程（$\hat\beta$ 拟合值与残差正交，见预备知识 L2）。
> **第 4 步（平方项）**：记 $x_j^{(j)}$ 为第 $j$ 列的一个置换。逐项算
> $$
> E_{\mathcal{P}}\big[(x_{ij}-x_{ij}^{(j)})^2\big]=E[x_{ij}^2]-2E\big[x_{ij}x_{ij}^{(j)}\big]+E\big[(x_{ij}^{(j)})^2\big]
> $$
> 末项 $=E[x_{ij}^2]=1$（置换不改列的元素）。中间项：给定第 $j$ 列的一个置换 $\pi$，$\frac1N\sum_kx_{ij}x_{\pi(j),j}$，对 $\pi$ 取期望得 $\frac1N\sum_kx_{ij}\cdot\bar x_j=\bar x_j^2=0$。
> **第 5 步**：故 $E[(x_{ij}-x_{ij}^{(j)})^2]=1-0+1=2$，$\frac1N\sum_i$ 求和后仍是 2，得 $E[\mathrm{RSS}_j^{*}-\mathrm{RSS}]=\hat\beta_j^2\cdot 2$。

> **坑** · 两个「大约」
> - 结论依赖**同分布的置换**，所以 $\bar x_j=0$、$\frac1N\sum x_{ij}^2=1$ 两个标准化都要做；少一个标准化就得不到系数 2。
> - $\mathrm{RSS}$ 是**训练**残差，训练集上置换的好处被其它样本「借走」了；用独立测试集评估时结论更接近精确（原书也这么说）。这也是为什么随机森林的置换重要性用 OOB 样本算，而不是全体训练样本。
