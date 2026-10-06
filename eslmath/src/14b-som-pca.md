## 14.4 自组织映射 {#s-14-4}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-4">原文 §14.4</a>

自组织映射（self-organizing map, SOM）在算法层面只做一件事：**给格网上的每个原型安排一个整数坐标，然后让「距离近」既指特征空间里的近，也指格网上的近**。形式上它是 $K$-means 的一个约束版本——原型被限制成一张一维或二维的网格，网格的邻接关系由一套与数据无关的整数坐标 $\ell_j\in\mathcal{Q}_1\times\mathcal{Q}_2$ 规定，$K=q_1q_2$。因为网格只有一两个自由度，这套原型在 $\mathbb{R}^p$ 里就形成了一个「被弯折的流形」，高维观测可以投到这张低维坐标表上，而投下去的顺序是有意义的。

### 14.4.1 从 $K$-means 到 SOM：只差一步 {#s-14-4-1}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-3-6">原文 §14.3.6</a> <a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-3">原文 §14.3.10</a>

先把 $K$-means 的目标函数摆出来，后面所有的推导都是它的变体。给定 $K$ 个原型 $m_1,\dots,m_K\in\mathbb{R}^p$，每个观测被指派给欧氏距离最近的原型 $c(i)$：

$$
\mathrm{RSS}(m_1,\dots,m_K)=\sum_{i=1}^{N}\lVert x_i-m_{c(i)}\rVert_2^2,\qquad c(i)=\arg\min_{k}\lVert x_i-m_k\rVert_2
$$

$L_i$ 型的向量量化（§14.3.10）用的就是这个目标：训练好一个码本（codebook）之后，把 $p$ 维向量用最近的码字近似。所以 SOM 本质上是「带拓扑约束的向量量化」。

$K$-means 的 Lloyd 算法之所以只剩两步，是因为在**分区固定**时目标函数对原型是可分离的二次函数。把观测按 $c(i)$ 分成簇 $C_k=\{i:c(i)=k\}$，记 $N_k=|C_k|$、$\bar x_{C_k}=N_k^{-1}\sum_{i\in C_k}x_i$，把范数平方展开：

$$
\mathrm{RSS}=\sum_{k=1}^{K}\sum_{i\in C_k}\big(\lVert x_i\rVert_2^2-2x_i^\top m_k+\lVert m_k\rVert_2^2\big)=\mathrm{const}-2\sum_{k=1}^{K}N_k\,\bar x_{C_k}^\top m_k+\sum_{k=1}^{K}N_k\lVert m_k\rVert_2^2
$$

$\mathrm{const}=\sum_i\lVert x_i\rVert_2^2$ 与 $m$ 无关。对单个 $m_k$ 求梯度并置零（用预备知识 L4 的 $\nabla_x\lVert x-m\rVert_2^2=2(x-m)$）：

$$
\nabla_{m_k}\mathrm{RSS}=-2N_k\bar x_{C_k}+2N_k m_k=0\quad\Longrightarrow\quad m_k=\bar x_{C_k}
$$

于是 $K$-means = **指派（最近原型）** + **更新（簇均值）** 反复交替；二次型在 $m$ 上凸（$H=2\,\mathrm{diag}(N_1,\dots,N_K)\succeq0$，见预备知识 L1），所以这个步骤是精确的全局最小，不是近似。

SOM 换掉的只有第二步，而且换得很轻：**更新时不只动胜出的那个原型，而是动它在格网上的邻居**。

### 14.4.2 SOM 的目标函数与在线更新 {#s-14-4-2}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-4">原文 §14.4</a>

记 $j^\star=c(i)$ 为 $x_i$ 的胜出原型，$h(r)$ 是邻域函数（neighborhood function），$h(0)=1$ 且关于 $r$ 单调不增，最常用的两个选择是高斯核 $h(r)=\exp\big(-r^2/(2\sigma_t^2)\big)$ 与倒角核 $h(r)=(1-r/1)^{\beta}$ 型的矩形核。于是「带权 $K$-means」目标函数是

$$
\mathrm{RSS}_{\mathrm{SOM}}=\sum_{i=1}^{N}\sum_{j=1}^{K}w_{j^\star,j}\big(i\big)\,\lVert x_i-m_j\rVert_2^2,\qquad w_{j^\star,j}(i)=h\big(\lVert \ell_{j^\star}-\ell_j\rVert\big)
$$

$w$ 只依赖整数坐标，与特征空间无关——这就是「拓扑约束」的数学内容。在线（逐点）版本只用一个观测 $x_i$ 做随机梯度：把目标函数限制到所有 $c(t)=j^\star$ 的项上，其对 $m_k$ 的（单样本）梯度是 $2w_{j^\star,k}(x_i-m_k)$，梯度下降一步 $m_k\leftarrow m_k-\tfrac{\alpha}{2}\nabla$ 得

$$
m_k\leftarrow m_k+\alpha\,h\big(\lVert \ell_{j^\star}-\ell_k\rVert\big)(x_i-m_k)
$$

邻域很窄、$h$ 取矩形核（$h=1$ 当 $\ell_k$ 是 $\ell_{j^\star}$ 的邻居，否则 0）时，退化成原书最简版本

$$
m_k\leftarrow m_k+\alpha(x_i-m_k),\qquad k\in\mathcal{N}(j^\star)
$$

$$\eqno{14.46}$$

更精细的版本让更新强度随**格网坐标上的距离**衰减（原文 (14.47)），$t$ 时刻邻域半径 $\sigma_t$ 同步收缩：

$$
m_k\leftarrow m_k+\alpha_t\,h_t\big(\lVert \ell_{j^\star}-\ell_k\rVert\big)(x_i-m_k)
$$

$$\eqno{14.47}$$

两个超参数都随 $t$ 下降：$\alpha$ 从 $1.0$ 线性降到 $0$，半径（阈值）$r$ 从 $R$ 线性降到 1（这样每个邻域最终只剩自己）。原书半球面例子取 $5\times5$ 格网、$R=2$（即初始每个邻域大约装下三分之一的原型）、40 遍数据共 3600 次迭代，$\alpha$ 与 $r$ 在这 3600 步内线性下降。

> **坑** · 「邻居」的距离是在整数坐标空间 $\mathcal{Q}_1\times\mathcal{Q}_2$ 里量的，不是在 $\mathbb{R}^p$ 里量的。两个原型即使在特征空间里贴得很近，只要格网上隔得远就不会互相拉动；反之亦然。忘记这一点就会以为 SOM 只是「多原型版的 $K$-means」。

### 14.4.3 邻域函数与半径为什么必须衰减 {#s-14-4-3}

先做一步「期望方向」的检查：把 $x_i$ 看成按经验分布抽取的一次样本、且 $c(i)=k$。所有被指派到 $k$ 的观测求和后噪声相消：

$$
E\big[x_i-m_k\big]=E[x_i]-m_k=\mu-m_k,\qquad \sum_{i:c(i)=k}(x_i-m_k)=0
$$

也就是说这一步**平均而言把 $m_k$ 朝总体均值拉**。但这只是平均，噪声还在；噪声会不会把 $m_k$ 搅得不收敛，取决于 $\alpha_t$ 的取法。把一次迭代写成线性递推，$\varepsilon_t$ 是零均值、方差有界的加性噪声（来自 $x_t-\mu$ 的偏离）：

$$
m_{t+1}=(1-\alpha_t)m_t+\alpha_t\big(\mu+\varepsilon_t\big)
$$

迭代展开（T 次迭代）：

$$
m_T=\mu+m_0\prod_{t=0}^{T-1}(1-\alpha_t)-\sum_{s=0}^{T-1}\alpha_s\,\varepsilon_s\prod_{t=s+1}^{T-1}(1-\alpha_t)
$$

分两部分看。

1. **初始记忆**：$\prod_{t<T}(1-\alpha_t)\to0\iff\sum_{t}\alpha_t=\infty$。因为 $\log(1-\alpha)\le-\alpha$，$\prod_{t<T}(1-\alpha_t)\le\exp\big(-\sum_{t<T}\alpha_t\big)\to0$；反之若 $\sum\alpha_t<\infty$ 则乘积收敛到正数，$m_0$ 永远擦不掉。若 $\alpha_t\equiv\alpha_0>0$ 常数，乘积是 $(1-\alpha_0)^T\to0$，这一条满足。
2. **噪声**：噪声项的方差（噪声独立时各项独立）为 $\mathrm{Var}(\varepsilon)\sum_s\alpha_s^2\prod_{t>s}(1-\alpha_t)^2$。每个因子 $\prod_{t>s}(1-\alpha_t)\to0$（对固定 $s$，因为尾和仍发散），故该项 $\le\mathrm{Var}(\varepsilon)\sum_s\alpha_s^2$。要让方差趋于 0，必须 $\sum_t\alpha_t^2<\infty$。

于是**Robbins–Monro 条件**（见预备知识 O4 随机梯度）正好卡在这两条上：

> **结果** · 收敛的充要（几乎处处）条件
>
> $$\sum_{t=1}^{\infty}\alpha_t=\infty,\qquad \sum_{t=1}^{\infty}\alpha_t^2<\infty$$
>
> 典型取法 $\alpha_t=a/t$（$a>0$）：由积分判别法 $\sum_{t\le T}a/t\ge a\log T\to\infty$，而 $\sum_t a^2/t^2=a^2\pi^2/6<\infty$。这解释了原文为什么强调 $\alpha$「递减到 0」而不是固定一个小常数。

> **坑** · 若 $\alpha_t\equiv\alpha_0>0$，条件一满足、条件二不满足：$m_t$ 会绕着 $\mu$ 做半径约 $\sqrt{\alpha_0/(2-\alpha_0)}\cdot\mathrm{sd}$ 的随机游走，算法**永远不停**，每个点得到的都是噪声平均过的原型。同理，若邻域半径 $r$ 一直很大，所有原型被同一批数据反复拉动，格网就退化成 $K$ 个几乎重合的点——拓扑结构根本没建立。所以 $\alpha$ 与 $r$ 必须同步收缩到最小格网尺度。

半径收缩还带一个几何效应：早期大邻域相当于对整张地图做「粗粒度预排序」（先分大区），后期小邻域只做局部微调。这是 SOM 能同时得到粗结构和细结构的原因，也是它比纯 $K$-means 多出来的信息。

### 14.4.4 批量版本 (14.48) 与 $K$-means 的恢复 {#s-14-4-4}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-4">原文 §14.4</a>

在线更新随机性大，原书随后给出批量（batch）版本：一次性扫完全部数据，把所有点按 $c(i)$ 指派，再统一更新每个原型。目标函数仍是 (14.4.2) 的加权 RSS，只是现在对**所有** $i$ 求驻点：

$$
\nabla_{m_j}\mathrm{RSS}_{\mathrm{SOM}}=-2\sum_{i=1}^{N}w_{j^\star(i),j}\,(x_i-m_j)=0
$$

记 $w_{kj}=\mathbb{1}\{c(k)=j\}h(\lVert\ell_k-\ell_j\rVert)$，把 $j$ 的邻域内的点求和，$m_j$ 是它们的**加权重心**：

$$
m_j=\frac{\sum_{k=1}^{N}w_{kj}\,x_k}{\sum_{k=1}^{N}w_{kj}}
$$

$$\eqno{14.48}$$

- 权重取矩形（$w_{kj}=1$ 当 $\ell_k$ 与 $\ell_j$ 相邻，否则 0），分母是邻域内点数。
- 权重随 $\lVert\ell_k-\ell_j\rVert$ 平滑下降，分母是邻域权重和。
- 若邻域小到只含 $m_k$ 自己且权重取矩形，则分母 $=\{c(k)=j\}$ 的点数、分子是该簇均值，(14.48) **逐字变成 $K$-means 的更新步**。此时每个观测只更新胜出原型，与 (14.46) 一致。

> **结果** · SOM $\supset$ $K$-means
>
> $$\text{邻域}\to\{\text{自己}\},\ h\to\text{矩形}\ \Longrightarrow\ (14.46),(14.48)\ \text{都退化为 }K\text{-means}$$
>
> 反过来不成立：$K$-means 的原型顺序没有任何约束，重启一次就可能给出完全不同的排列。SOM 的格网坐标 $\ell_j$ 是**固定**的，它给出的是 $m_j$ 到 $\ell_j$ 的一个固定同态，因而在原型集合相同的两次运行之间也可以对齐。这就是「保序性」：格网上相邻的两个结点，其负责的观测在特征空间里也相近（在实践中成立，不是恒等式）。

两个方法的差别就是**误差代价**：$K$-means 的原型不受约束，$\mathrm{RSS}$ 一定不高于 SOM 的。正因为如此，评估拓扑约束是否合理的方式是把两者的重构误差 $\sum_i\lVert x_i-m_{c(i)}\rVert_2^2$ 都算出来对比——原书对半球的 90 个点用 25 个原型，SOM 的误差从初始值大幅下降并接近 $K$-means 的水平线，说明二维约束在这份数据上没付出多少代价。

> **坑** · SOM 投影**丢掉**了距离信息。格网上两个相邻结点的实际距离可能差很多，但显示时只按整数坐标画。半球的例子里红色簇被压得很紧，SOM 投影上看不出来（要靠图 14.17 的线框图才能发现流形折回自己）。原文明确说「在二维显示里用的距离不参与计算」，所以不要指望从 SOM 图上读出密度。

### 14.4.5 与第 13 章 LVQ 的对照 {#s-14-4-5}

SOM 与第 13 章的原型方法（LVQ，learning vector quantization）共享同一个骨架：都维护一组原型、每个样本找最近原型、胜出原型朝样本移动。差别只有一处，但在数学上很关键：

| | 更新式 | 对应 |
|---|---|---|
| LVQ（硬判别，$j^\star$ 唯一） | $m_k\leftarrow m_k+\alpha(x_i-m_k)$，只对 $k=j^\star$ | 在线 $K$-means 一步 |
| SOM（软邻域） | $m_k\leftarrow m_k+\alpha_t h(\lVert\ell_{j^\star}-\ell_k\rVert)(x_i-m_k)$，对所有邻居 $k$ | (14.46)(14.47) |

也就是说，**SOM 是在 LVQ 的更新方向场里插入了 $h_t$ 这个拓扑权重**。$h_t\to\mathbb{1}_{\{k=j^\star\}}$ 时回到 LVQ/$K$-means，$h_t$ 恒为 1（所有原型都动）时所有原型朝同一个样本收缩、地图塌缩。这是理解 SOM 全部行为的单一视角。

最后提一下与下一节的联系：SOM 可以看成**主曲线/主曲面的离散版本**——主曲线是一条被数据点拉弯但尽量不自交的连续曲线，SOM 的格网就是它的离散采样点，两者的目标函数都在最小化「数据到曲面的距离平方」。而 SOM 的初始化，原书建议直接取数据二维主成分平面上的规则格网（见图 14.21 右图到图 14.16 左图的对应），这就是下面两节要推的东西。

## 14.5 主成分、曲线与曲面 {#s-14-5}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5">原文 §14.5</a>

主成分在 §3.4.1 已经出现过一次，在那里它是岭回归收缩机制的解释工具：把 $X$ 旋转到主成分坐标系，岭回归对大特征值方向的收缩最小、对小特征值方向的收缩最大，于是 $\hat\beta_j^{\text{ridge}}=(\lambda_j+d)^{-1}\lambda_j\hat\beta_j^{\text{OLS}}$ 看上去像是在按「特征值大小」打折。那个视角里的主成分只是**一组坐标**；§14.5.1 换成一个几何视角：主成分是数据在所有秩 $q$ 线性流形里的最优逼近，两条准则（最大投影方差、最小重构误差）给出同一个答案。

§14.5 这一整节讲的是「用低维结构逼近高维数据」，分五个方向推进，难度递进：

- **§14.5.1 主成分（PCA）**：最优逼近是**线性**的，用一个 $p\times q$ 的正交矩阵 $V_q$ 加一个位置向量 $\mu$ 表示，$q=1$ 是直线、$q=2$ 是平面。
- **§14.5.2 主曲线与主曲面**：把直线/平面换成弯曲的一维/二维流形，用样条基 $\phi(\lambda)^\top$ 参数化，损失函数里出现**曲率惩罚**，并且有一个自洽条件（沿曲线方向的投影不能再改进损失）。这一节的例子是 Procrustes 形状平均与签名识别，即旋转 Procrustes 问题 (14.56) 与它的解 (14.57)。
- **§14.5.3 谱聚类**：把 PCA 的「最小化 $X^\top X$ 的特征值和」换成「最小化拉普拉斯矩阵的非零特征值」，从连续数据跳到图数据，与第 14.3.6 的谱分割同源。
- **§14.5.4 核主成分**：把 $\mathrm{RSS}(V_q)$ 里的 $X^\top X$ 换成核矩阵 $K$，得到不要求线性流形的降维。
- **§14.5.5 稀疏主成分**：给载荷加 $\ell_1$ 惩罚，用 $z_m=\lVert v_m\rVert_1$ 与相关关系 $\mathrm{Corr}(x_j,z_m)=v_{mj}$ 交替求解，把载荷压到少数几个特征上。

这一节的数学工具只有四个：二次型在正交约束下的极值（Rayleigh 商，见预备知识 L1、L4、O1）、正交投影（预备知识 L2）、SVD 与迹恒等式（预备知识 L3）、以及 Frobenius 范数下的范数分解。下面把它们一一落到可手算的形式上。

### 14.5.1 主成分分析 {#s-14-5-1}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-1">原文 §14.5.1</a>

#### (14.49) 秩 $q$ 线性流形

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5">原文 §14.5</a>

数据 $x_1,\dots,x_N\in\mathbb{R}^p$。一个秩 $q$ 的线性流形就是「过某点、方向张成 $q$ 维子空间」的仿射超平面：

$$
f(\lambda)=\mu+V_q\lambda
$$

$$\eqno{14.49}$$

其中 $\mu\in\mathbb{R}^p$ 是位置向量，$V_q=[v_1,\dots,v_q]$ 是 $p\times q$ 矩阵、列是正交单位向量（$V_q^\top V_q=\mathbf 1_q$），$\lambda\in\mathbb{R}^q$ 是 $q$ 个参数。$q=1$ 时是一条有向直线，$q=2$ 时是一个有向平面（原书图 14.20、14.21）。

> **基础知识** · 中心化
>
> 后面所有公式都假设 $x$ 已经中心化，即 $\bar x=\frac1N\sum_{i=1}^N x_i=0$。不中心化时把所有 $x_i$ 换成 $x_i-\bar x$ 即可，而 (14.51) 就是这一句话的公式化。中心化后
>
> $$\sum_{i=1}^N(x_i-\bar x)=0,\qquad \frac1N\sum_{i=1}^N (x_i-\bar x)(x_i-\bar x)^\top=\Sigma$$
>
> 其中 $\Sigma$ 是协方差矩阵（预备知识 L5）。两种归一化约定：$\Sigma=\frac1N X^\top X$（population，除以 $N$）与 $\Sigma=\frac{1}{N-1}X^\top X$（样本，除以 $N-1$）。两者只差一个全局倍数 $\frac{N}{N-1}$，**主成分方向完全相同**，只有特征值的绝对尺度差这个因子。

#### (14.50) 重构误差与两处部分优化

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-1">原文 §14.5.1</a>

拟合 (14.49) 就是最小化**重构误差**（reconstruction error）：

$$
\mathrm{RSS}=\sum_{i=1}^{N}\min_{\mu,V_q,\lambda_i}\lVert x_i-\mu-V_q\lambda_i\rVert_2^2
$$

$$\eqno{14.50}$$

三重 `min` 看着吓人，但三个参数块可以**依次精确优化**，前两步就是普通最小二乘。

**第一步：对 $\mu$ 求导。** 把目标看作 $\sum_i\lVert (x_i-\mu)-V_q\lambda_i\rVert_2^2$，对 $\mu$ 用 $\nabla_\mu\lVert a-\mu\rVert_2^2=2(\mu-a)$：

$$
\nabla_\mu\sum_{i=1}^{N}\lVert x_i-\mu-V_q\lambda_i\rVert_2^2=2\sum_{i=1}^{N}(\mu+V_q\lambda_i-x_i)=0
$$

因为 $\lambda_i$ 的最优值不依赖 $\mu$（平移不变），这一式对 $N$ 个点求和后：

$$
N\mu+V_q\sum_{i=1}^N\lambda_i-\sum_{i=1}^N x_i=0
$$

在每点都取 $\lambda_i=V_q^\top(x_i-\bar x)$ 时 $\sum_i\lambda_i=0$（因为 $V_q^\top$ 满秩、$\sum_i(x_i-\bar x)=0$），故 $N\mu=\sum_i x_i$，即最优位置向量就是样本均值：

$$
\hat\mu=\bar x=\frac1N\sum_{i=1}^{N}x_i
$$

$$\eqno{14.51}$$

**第二步：对每个 $\lambda_i$ 求导（部分回归 / multiple regression）。** 固定 $\mu=\bar x$ 与 $V_q$，令 $a_i=x_i-\bar x$，目标对 $\lambda_i$ 可分离：

$$
\sum_{i=1}^{N}\min_{\lambda_i}\lVert a_i-V_q\lambda_i\rVert_2^2=\sum_{i=1}^{N}\big(\lVert a_i\rVert_2^2-a_i^\top V_qV_q^\top a_i\big)
$$

这一步用到两个恒等式，都可以直接展开验证：

$$
\min_{\lambda}\lVert a-V_q\lambda\rVert_2^2=\lVert (I_p-P)a\rVert_2^2,\qquad P=V_qV_q^\top,\qquad P^2=P,\ P^\top=P
$$

（$P^2=V_qV_q^\top V_qV_q^\top=V_qI_qV_q^\top=P$。）所以最小值在 $\lambda=V_q^\top a$ 处取到，残差 $a-V_qV_q^\top a$ 与每个 $v_j$ 正交：

$$
v_j^\top a_i-v_j^\top V_q\lambda_i=\langle v_j,a_i\rangle-\langle v_j,V_qV_q^\top a_i\rangle=0
$$

于是最优得分为

$$
\hat\lambda_i=V_q^\top(x_i-\bar x),\qquad j=1,\dots,q
$$

$$\eqno{14.52}$$

**第三步：对 $V_q$ 求导。** 把前两步代回，剩下的目标是

$$
\mathrm{RSS}(V_q)=\sum_{i=1}^{N}\Big\lVert (x_i-\bar x)-V_qV_q^\top(x_i-\bar x)\Big\rVert_2^2
$$

$$\eqno{14.53}$$

即：在所有 $q$ 维正交投影中，选使 $\sum_i\lVert x_i-\bar x\rVert_2^2=\sum_i\lVert V_qV_q^\top(x_i-\bar x)\rVert_2^2$ 最大的那个。原书写成 `$\sum\min_V\lVert\cdot\rVert^2$`，那个外层 `$\min_V$` 就是 (14.53) 要做的事；把外层求和号与内层最小值的分工读清楚才不会混淆。

(14.53) 已经是「选 $q$ 个正交方向」的形式，接下来的问题是怎么选。

#### 最大方差准则的完整推导 {#s-14-5-1-variance}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5">原文 §14.5</a>

**问题**：找单位向量 $v$ 使 $x_i$ 在 $v$ 上的投影 $v^\top x_i$ 的方差最大。原书的说法是「$Xv_1$ 在所有线性组合中方差最高，$Xv_2$ 在与 $v_1$ 正交的线性组合中方差最高，以此类推」。

> **基础知识** · 方差的二次型形式与 Rayleigh 商
>
> 令 $Z=X^\top v$（$Z$ 是 $v$ 对应的得分向量，$X$ 为 $N\times p$）。由 $E[Z]=Xv$ 与协方差的线性变换律 $\mathrm{Cov}(aX)=a\mathrm{Cov}(X)a^\top$（预备知识 L5）：
>
> $$\mathrm{Var}(Z_i)=\mathrm{Var}(v^\top x_i)=\sum_{j}\sum_{j'}v_jv_{j'}\mathrm{Cov}(x_j,x_{j'})=v^\top\Sigma v$$
>
> 其中 $\mathrm{Var}(Z_i)=\frac1N\sum_{i=1}^N(z_i-\bar z)^2$、$\bar z=v^\top\bar x=0$（中心化后）。

于是第一个主成分的准则是

$$
v_1=\arg\max_{\lVert v\rVert_2=1}\ v^\top\Sigma v
$$

这是个**等式约束下的二次型极值问题**。用拉格朗日乘子（预备知识 O1），取

$$
L(v,\lambda)=v^\top\Sigma v-\lambda\big(v^\top v-1\big)
$$

**一阶条件**（$\Sigma$ 对称，所以 $\nabla_v v^\top\Sigma v=2\Sigma v$，见预备知识 L4）：

$$
\nabla_v L=2\Sigma v-2\lambda v=0\quad\Longleftrightarrow\quad \Sigma v=\lambda v
$$

**这就是特征值方程**：驻点必须是 $\Sigma$ 的特征向量。**代回目标值**（这一步给出「极值 = 特征值」这个关键对应）：

$$
v^\top\Sigma v=v^\top(\lambda v)=\lambda\,v^\top v=\lambda
$$

所以目标函数在驻点处的值恰好等于拉格朗日乘子。**充分性**：把任意单位 $v$ 在正交特征基 $\{v_m\}$ 下展开 $v=\sum_m a_mv_m$、$\sum_ma_m^2=1$，则

$$
v^\top\Sigma v=\sum_m\sum_{m'}a_ma_{m'}\,v_m^\top\Sigma v_{m'}=\sum_m\lambda_m a_m^2\le\lambda_{\max}\sum_ma_m^2=\lambda_{\max}
$$

等号当且仅当非零的 $a_m$ 只出现在 $\lambda_m=\lambda_{\max}$ 处。所以 $v_1$ 必是最大特征值对应的特征向量（特征值重数大于 1 时不唯一，取该特征子空间内任一单位向量都对）。这就是预备知识 L1 的「二次型极值判别」在**单位球约束**上的版本。

**第二个及以后的成分**要在正交约束下重复：$\max v_m^\top\Sigma v_m$ s.t. $\lVert v_m\rVert=1$、$\langle v_j,v_m\rangle=0\ (j<m)$。做法是**去化（deflation）**：由上一步知 $\Sigma v_m$ 与 $v_1,\dots,v_{m-1}$ 分量正交，所以只需在补空间内解同一个问题。用拉格朗日函数 $L=v_m^\top\Sigma v_m-\lambda_m(v_m^\top v_m-1)-2\sum_{j<m}\gamma_jv_j^\top v_m$：

$$
\nabla_{v_m}L=2\Sigma v_m-2\lambda_mv_m-2\sum_{j<m}\gamma_jv_j=0\ \Longrightarrow\ \Sigma v_m=\lambda_mv_m+\sum_{j<m}\gamma_jv_j
$$

两边左乘 $v_m^\top$（$v_m\perp v_j$ 对 $j<m$）得 $v_m^\top\Sigma v_m=\lambda_m$；右乘、沿 $v_j$ 逐个取内积得 $\gamma_j=0$，于是 $\Sigma v_m=\lambda_mv_m$。结论：

$$
\Sigma=V\Lambda V^\top,\quad v_m\ \text{是第 } m\ \text{大特征值的特征向量}
$$

> **结果** · 逐次最大方差准则
>
> $$v_m=\arg\max_{v:\ \lVert v\rVert_2=1,\ \langle v_j,v\rangle=0\ (j<m)}v^\top\Sigma v,\qquad v_m^\top\Sigma v_m=\lambda_m$$
>
> 这就是 §14.5.1 结尾那句「$Xv_1$ 方差最高，$Xv_2$ 在与 $v_1$ 正交的组合中方差最高，以此类推」的完整含义：$v_m$ 是**在 $\operatorname{span}\{v_1,\dots,v_{m-1}\}$ 的正交补里**方差最大的方向。

#### 最小重构误差准则的完整推导 {#s-14-5-1-rss}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-1">原文 §14.5.1</a>

**问题**：换一个准则 —— 在所有秩 $q$ 线性流形里选重构误差最小的那个。等价地（这是最有用的写法），把 $x_i$ 在前 $q$ 个得分 $z_1,\dots,z_q$ 上回归：

$$
\hat x_{ij}=\sum_{m=1}^{q}\beta_{jm}z_{im},\qquad z_{im}=v_m^\top x_i
$$

$$
\mathrm{RSS}=\sum_{i=1}^{N}\sum_{j=1}^{p}\Big(x_{ij}-\sum_{m=1}^{q}\beta_{jm}z_{im}\Big)^2
$$

每个 $j$ 是一个独立的回归问题（多个回归变量 $z_1,\dots,z_q$）。**正规方程**：对 $\beta_{jm}$ 求偏导置零

$$
\frac{\partial \mathrm{RSS}}{\partial \beta_{jm}}=-2\sum_{i=1}^{N}z_{im}\Big(x_{ij}-\sum_{m'=1}^{q}\beta_{j'm'}z_{im'}\Big)=0\quad\Longleftrightarrow\quad \sum_{i=1}^{N}z_{im}x_{ij}=\sum_{m'=1}^{q}\beta_{j'm'}\sum_{i=1}^{N}z_{im'}z_{im'}
$$

这个「残差与所有得分正交」的正规方程有唯一解，因为得分矩阵 $Z=[z_1,\dots,z_q]=XV_q$ 列满秩：$Z^\top Z=V_q^\top X^\top X V_q=N\,\mathrm{diag}(\lambda_1,\dots,\lambda_q)$，其对角元全为正。

**把解算出来**。左边用 $z_m=Xv_m$：

$$
\sum_{i=1}^{N}z_{im}x_{ij}=(Xv_m)^\top x_i\big|_{j}=\big(X^\top Xv_m\big)_j=(N\Sigma v_m)_j=N\lambda_m\,v_{mj}
$$

（用了 $\Sigma v_m=\lambda_m v_m$ 和 $X^\top X=N\Sigma$。）右边：$\sum_{i}z_{im}z_{im'}=N\,\mathrm{Cov}(z_m,z_{m'})=N\lambda_m\mathbb{1}_{mm'}$（由最大方差准则的结论 $\mathrm{Cov}(z_m,z_{m'})=v_m^\top\Sigma v_{m'}=\lambda_{m'}v_{m'}^\top v_m$，见下面「得分互不相关」）。代入正规方程：

$$
N\lambda_m v_{mj}=N\lambda_m\sum_{m'=1}^{q}\beta_{j'm'}\mathbb{1}_{mm'}
$$

因为 $N\lambda_m>0$（$\lambda_m>0$ 当数据在该方向上有变异；否则该方向无信息，见坑），约掉：

$$
\beta_{jm}=v_{mj}
$$

$$
\hat x_{ij}=\sum_{m=1}^{q}\beta_{jm}z_{im}=\sum_{m=1}^{q}v_{mj}\,v_m^\top x_i=\big(Px_i\big)_j,\qquad P=V_qV_q^\top
$$

> **结果** · 得分-载荷的对偶关系
>
> $$z_{im}=v_m^\top x_i\ (\text{score}),\qquad \beta_{jm}=v_{mj}\ (\text{loading})$$
>
> **载荷就是特征向量本身**。重构是同一个投影矩阵 $P=V_qV_q^\top$：$z=Zx$（把 $\mathbb{R}^p$ 压到 $\mathbb{R}^q$）与 $\hat x=Px$（把 $\mathbb{R}^p$ 投回 $q$ 维子空间）用的是同一组 $v_m$。所以「降维」与「重构」不是两个不同的设计，它们是投影与投影的复合。

#### 两条准则的等价性（核心推导） {#s-14-5-1-equivalence}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5">原文 §14.5</a>

上一小节说最小 RSS 解出 $\beta_{jm}=v_{mj}$，最大方差准则说 $v_m$ 是 $\Sigma$ 的特征向量。现在把它们接起来，证明两个准则给**同一个** $V_q$。

**第一步：最优解的残差只剩被丢掉的分量。** 取上小节求出的最优 $\beta_{jm}=v_{mj}$，则

$$
\sum_{m=1}^{q}\beta_{jm}z_{im}=\sum_{m=1}^{q}v_{mj}v_m^\top x_i=\big(V_qV_q^\top x_i\big)_j=(Px_i)_j
$$

所以每个坐标的残差是 $x_{ij}-(Px_i)_j=\big[(I_p-P)x_i\big]_j$。注意残差**并不与得分逐个为零**（那需要 $q=p$），而是在 $\{x_i\}$ 张成的子空间里与 $\operatorname{col}(V_q)$ 正交：$\langle (I_p-P)x_i,v_m\rangle=0$ 对 $j\le q$。

**第二步：把平方和展开并识别为「被丢弃的方差和」。** 逐 $j$ 求和，残差向量是 $(I_p-P)x_i$，它的平方范数可以在**被保留的基**与**被丢弃的基**两组上同时展开。保留侧贡献零（第一步的正交性），故

$$
\mathrm{RSS}_q=\sum_{i=1}^{N}\lVert(I_p-P)x_i\rVert_2^2=\sum_{i=1}^{N}\sum_{m>q}\big(v_m^\top x_i\big)^2
$$

推导这一步有两种写法，都值得写出：

- 用 Pythagoras 恒等式（预备知识 L2）：$\lVert x_i\rVert_2^2=\lVert Px_i\rVert_2^2+\lVert(I_p-P)x_i\rVert_2^2$。对 $i$ 求和：$\sum_i\lVert x_i\rVert_2^2=\sum_m\sum_i z_{im}^2+\mathrm{RSS}_q$。而 $\sum_{i=1}^N z_{im}^2=N\,v_m^\top\Sigma v_m=N\lambda_m$（中心化后 $\sum_i z_{im}^2=\sum_i v_m^\top x_ix_i^\top v_m=v_m^\top(X^\top X)v_m$）。于是
- $$\mathrm{RSS}_q=\sum_{m=1}^{p}N\lambda_m-N\sum_{m=1}^{q}\lambda_m=N\sum_{m>q}\lambda_m$$

**第三步：换成最大方差准则的语言。** $N$ 是常数，$\lambda_m$ 只依赖 $m$，所以

$$
\arg\min_{V_q}\mathrm{RSS}_q=V_q\ \text{取前 } q\ \text{个最大特征值对应的特征向量}
$$

三步串起来就是完整链条：

> **结果** · 两条准则等价
>
> $$\mathrm{RSS}_q=\sum_{m>q}\sum_{i=1}^{N}(v_m^\top x_i)^2=N\sum_{m>q}\lambda_m$$
>
> $$\min_{V_q}\mathrm{RSS}_q\quad\Longleftrightarrow\quad\max_{V_q}\sum_{m=1}^{q}\lambda_m\quad\Longleftrightarrow\quad\max_{\lVert v\rVert=1}v^\top\Sigma v$$
>
> 三个「$\max$」分别是：最小化被丢弃的方差和、最大化保留的方差和、单个方向上的最大方差。

用 $X=UDV^\top$ 的语言（见下一小节）这还可以写成 $\mathrm{RSS}_q=\sum_{m>q}d_m^2$ —— 被丢弃的正好是**被截断的奇异值的平方和**。这就是「PCA = 最佳秩 $q$ 近似」的全部含义。

> **坑** · 两个前提必须成立，否则链条断掉：
>
> - **中心化**。不中心化时 $\sum_i z_{im}^2=v_m^\top X^\top Xv_m=N\,v_m^\top(\Sigma+\bar x\bar x^\top)v_m$，多出秩 1 的项 $N(\bar x^\top v_m)^2$，主成分会去找「穿过原点的最佳平面」而不是「过均值的最佳平面」。
> - **$\lambda_m>0$** 才能约掉。若某特征值为 0（数据在该方向上是常数），对应 $Z^\top Z$ 奇异、$\beta$ 不唯一，但那也说明该方向没有信息，删掉即可。原书取 $d_i\ge0$ 递减排序、$V_q$ 取前 $q$ 列，正是自动跳过零方向。

#### 得分互不相关 {#s-14-5-1-uncorrelated}

主成分最常被引用的性质：不同成分的得分**线性无关**（因为 $v_m$ 正交），而且**互不相关**。后者要算，不能靠「线性无关所以独立」——线性无关与独立是两件事，这里只是不相关。

$$
E[z_m z_k]=v_m^\top E[x_ix_i^\top]v_k=v_m^\top\Sigma v_k
$$

交换标量再交换两列：$v_m^\top\Sigma v_k=(v_m^\top\Sigma v_k)^\top=v_k^\top\Sigma v_m$（$\Sigma$ 对称）。右端用特征向量关系 $\Sigma v_k=\lambda_kv_k$：

$$
v_k^\top\Sigma v_k=v_k^\top\lambda_kv_k=\lambda_k\,v_k^\top v_k=\lambda_k
$$

所以

$$
E[z_mz_k]=v_m^\top\Sigma v_k=\lambda_k\,v_m^\top v_k=\lambda_k\,\mathbb{1}_{mk}
$$

$$
\mathrm{Corr}(z_m,z_k)=\frac{E[z_mz_k]-E[z_m]E[z_k]}{\mathrm{sd}(z_m)\mathrm{sd}(z_k)}=0\quad(m\ne k),\qquad \mathrm{Var}(z_m)=\lambda_m
$$

> **结果** · 得分的协方差矩阵是对角的
>
> $$\mathrm{Cov}(Z)=\frac1NX^\top XV_qV_q^\top=V_q\Lambda_qV_q^\top,\qquad \mathrm{Var}(z_m)=\lambda_m$$
>
> 主成分坐标系是一个**白化**（decorrelated）坐标系：旋过去之后各分量独立地携带方差 $\lambda_1\ge\lambda_2\ge\dots$。这就是为什么主成分得分可以直接当作无监督模型的输入特征（而且不需要标准化，见 §14.5.5 稀疏 PCA 里 $z_m=\lVert v_m\rVert_1$ 的写法）。

同时注意总方差守恒：$\sum_{m=1}^{p}\lambda_m=\mathrm{tr}(\Sigma)=\sum_j\mathrm{Var}(x_j)$（迹不变性，预备知识 L3），即 $\sum_m\mathrm{Var}(z_m)=\sum_j\mathrm{Var}(x_j)$。前 $q$ 个成分解释的比例就是 $\sum_{m\le q}\lambda_m/\sum_m\lambda_m$。

#### (14.54) SVD 视角与 (14.55) 两个成分的显式形式 {#s-14-5-1-svd}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-1">原文 §14.5.1</a>

把中心化后的观测按行堆成 $N\times p$ 矩阵 $X$，做**奇异值分解**：

$$
X=UDV^\top
$$

$$\eqno{14.54}$$

$U$ 是 $N\times p$ 列正交矩阵（$U^\top U=\mathbf 1_p$），列 $u_j$ 是**左奇异向量**；$V$ 是 $p\times p$ 列正交矩阵（$V^\top V=\mathbf 1_p$），列 $v_j$ 是**右奇异向量**；$D$ 是 $p\times p$ 对角阵，对角元 $d_1\ge d_2\ge\dots\ge d_p\ge0$ 是**奇异值**。这是数值分析里的标准分解，有很多算法（Golub & Van Loan）。

**主成分就是 $V$ 的列。** 推导：$X^\top X=VDU^\top UDV^\top=VD^2V^\top$，所以

$$
\Sigma=\frac1N X^\top X=V\Big(\frac{D^2}{N}\Big)V^\top
$$

$$
\lambda_m=\frac{d_m^2}{N},\qquad v_m=V_{:m}
$$

即 $\Sigma$ 的特征向量（右奇异向量）与特征值（奇异值的平方除以 $N$）直接由 SVD 给出，**不必先形成 $\Sigma$**——这是数值上推荐 SVD 的原因：$\Sigma$ 可能半正定、秩亏、且条件数平方（预备知识 N1），而 $X$ 的 SVD 没有这些问题。

**得分矩阵。** $Z_q=[z_1,\dots,z_q]=XV_q=V_q^\top X$ 转置 $=U_qD_q$（因为 $V_q^\top X=(V^\top X)_{1:q}=(UD^\top)_{1:q}=U_qD_q$）。所以原书说「$X$ 的主成分是 $UD$ 的列」，且**$N$ 个最优 $\hat\lambda_i$ 就是前 $q$ 个主成分，即 $N\times q$ 矩阵 $U_qD_q$ 的 $N$ 行**：

$$
\hat\lambda_i=u_{i1}d_1v_1+u_{i2}d_2v_2+\dots+u_{iq}d_qv_q
$$

对 $q=2$，参数化模型显式写成（原书图 14.21 左图那条平面）：

$$
\hat f(\lambda)=\bar x+\lambda_1v_1+\lambda_2v_2
$$

$$\eqno{14.55}$$

**$q=1$ 的几何读法**（图 14.20）：$x_i$ 在直线 $\{\beta v_1\}$ 上的最近点是 $u_{i1}d_1v_1$，沿直线从原点的距离 $|\hat\lambda_i|=|u_{i1}d_1|$，法向残差长度是 $d_1\sqrt{1-u_{i1}^2}$。所以「主成分得分 $\lambda_i$」不是随便选的坐标，而是**沿直线的有向正交投影长度**。

#### 方差解释与重构 {#s-14-5-1-variance-share}

把 (14.55) 往前推一步：**载荷平方和**。既然 $\beta_{jm}=v_{mj}$，那么每个原始特征 $x_j$ 被前 $q$ 个成分重构的部分是 $\sum_{m\le q}\beta_{jm}^2=\sum_{m\le q}v_{mj}^2$，它不超过 1（因为 $\sum_m v_{mj}^2=1$，$\mathbf 1$ 与 $v_m$ 的勾稽归一），且 $\sum_{j}\sum_{m\le q}v_{mj}^2=q$。所以：

- $\sum_m z_{im}^2=\lVert x_i\rVert_2^2$ 仅当 $q=p$ 时成立；一般地 $\sum_m z_{im}^2=\lVert V^\top x_i\rVert_2^2=\lVert Px_i\rVert_2^2+\lVert(I_p-P)x_i\rVert_2^2$，即保留部分平方和 $=$ 总平方和 $-$ RSS。
- 被丢弃的比例（每个点）$=\lVert(I_p-P)x_i\rVert_2^2/\lVert x_i\rVert_2^2$，对全体平均后与 $\sum_{m>q}\lambda_m/\sum_m\lambda_m$ 同阶，这正是原书图 14.24 画的奇异值谱所表达的信息。

原书手写数字例子：256 个可能的主成分中，**约 50 个解释 90% 的变异、12 个解释 63%**。这两个数就是上面的累计比例 $\sum_{m\le q}\lambda_m/\sum_m\lambda_m$ 在 $q=12$ 和 $q=50$ 处的值（用 $\lambda_m=d_m^2/N$ 算）。左图把 $v_1,v_2$ 显示成图像：$v_1$（水平方向）主要体现「三的下半截变长」，$v_2$（竖直方向）体现笔画粗细——载荷的可解释性正是靠 $\beta_{jm}=v_{mj}$ 这个对偶关系。

> **坑** · 手写数字那组数据的像素天然相关（同一张图的像素之间、同一类数字之间相关更强），所以它的奇异值谱明显高于「随机打乱每列」得到的对照谱。**高维数据的低秩性往往是数据本身的冗余，不是普遍规律**；判断降维是否值得做，要拿这种随机化对照，而不是只看累计方差曲线。

#### 标准化：用协方差还是相关矩阵 {#s-14-5-1-scaling}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-1">原文 §14.5.1</a>

PCA 的优化问题里**没有**任何关于尺度的先验，方向完全由 $\Sigma$ 决定，而 $\Sigma_{jk}$ 是带量纲的：$x$ 用米还是厘米，$\Sigma$ 就差 $10^4$，主成分方向随之改变。所以必须显式决定「是否除以标准差」：

- **不标准化**（用协方差矩阵）：特征量纲相同的物理量、有可比性的测量值。$\Sigma=\frac1NX^\top X$。
- **标准化**（用相关矩阵）：把每个特征换成 $x_j/\hat\sigma_j$ 再做 PCA，等价于在

  $$R=\mathrm{diag}\big(\hat\sigma_1^{-1},\dots,\hat\sigma_p^{-1}\big),\qquad \Sigma_{\text{corr}}=R^\top\Sigma R$$

  上做特征分解（$x$ 已中心化时 $\Sigma_{\text{corr},jk}=\hat\rho_{jk}$）。相关矩阵的对角全是 1，所以「总变异」在标准化版本里是 $p$，前 $q$ 个成分解释 $q/p$ 的比例；不标准化版本的总变异是 $\sum_j\hat\sigma_j^2$。**这两个累计比例不能互相比较**，这是选标准化时最常被忽略的坑。

设标准化矩阵为 $D=R^{-1}=\mathrm{diag}(\hat\sigma_1,\dots,\hat\sigma_p)$，数据换成 $\tilde X=XD$，则 $\tilde X^\top\tilde X=D X^\top X D$，SVD 也随之变化：$\tilde X=\tilde U\tilde D\tilde V^\top$，原坐标下的载荷是 $\hat v_m=D\tilde v_m$。所以标准化只是「先做一次可逆的线性预变换，结论再变换回来」，算法结构完全不变。

第 3.5.1 节（PCR）里同一个问题是用**交叉验证选主成分个数**解决的：标准化与否也应该用留出误差或 CV 选，而不是默认。原文没有给出确定规则，只指出两者是不同的模型。

#### 与 LDA 的关系、population 版本与数值算法 {#s-14-5-1-relations}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5">原文 §14.5</a>

**PCA 与第 4 章的 LDA。** 第 4.3 节的线性判别分析在类内协方差相同的假设 $\Sigma_W=\Sigma_B$（或各类等协方差矩阵等于总体协方差 $\Sigma$）下，判别函数

$$
\delta_k(x)=\log\pi_k-\frac12\log\lVert\Sigma_k\rVert-\frac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)+\text{const}
$$

对单个高斯类时最速下降方向是 $\Sigma^{-1}(\mu_k-\bar\mu)$；在 $\Sigma_k=\Sigma$ 的共同协方差假设下（各方向权重相同），**方差最大的方向就是分类信息量最大的方向**，于是判别方向退化为「总体协方差矩阵的**主**特征向量」，分类投影的坐标即 $z=v_m^\top x$（无偏时 $\propto\sqrt{\lambda_m}$）。这正是 PCA 的第一个成分。反过来说，PCA 就是「只有一个类、且协方差相等的 LDA」——无监督版本与监督版本的差别只在是否用类均值。

**PCA 没有「样本 vs 总体」的两种版本之分**（与第 3 章的回归不同）：一旦 $\Sigma=\frac1NX^\top X$（或 $\frac1{N-1}X^\top X$）被固定，$V_q$ 和 $\hat\mu=\bar x$ 就定死了，没有需要另行论证的一致性/有效性差别。唯一可选的是**中心化 vs 不中心化**（用 $\bar x$ 或用原点）。

**数值上怎么算**（预备知识 N1–N3）：

- 若 $p\ll N$：直接形成 $\Sigma$（$p\times p$），用 Jacobi / QR 迭代求对称特征分解，或只用**幂法**（power iteration）求最大的一个特征对：$v^{(t+1)}\leftarrow\Sigma v^{(t)}/\lVert\Sigma v^{(t)}\rVert_2$。幂法每步是 $O(p^2)$，一次得到一对 $(\lambda,v)$；要前 $q$ 个就做 $q$ 次去化幂法，或直接用分块幂法同时更新 $q$ 个方向。$N$ 大但能形成 $\Sigma$ 时，$\Sigma$ 的条件数是 $X$ 的平方，所以精确解难；实践中常用随机 SVD 或 Lanczos 取前几个奇异对。
- 若 $N\ll p$（$p>N$ 时 $\Sigma$ 必然秩亏）：**对 $X$ 做（截断）SVD**，只保留前 $q$ 个奇异对。这样 $\Sigma$ 的非零特征值个数是 $\min(N,p)$，不需要处理零方向。

> **坑** · 幂法的收敛速度依赖谱间隙 $\lvert\lambda_q/\lambda_{q+1}\rvert$：间隙小时收敛极慢（$O(\lvert\lambda_{q+1}/\lambda_q\rvert^{\,t})$）。当 $q$ 取得较大、$\lambda_q\approx\lambda_{q+1}$ 时，直接上 Jacobi、Lanczos 或 SVD，别用幂法硬磨。

#### (14.56) 一个立刻用到 SVD 的例子：Procrustes 问题 {#s-14-5-1-procrustes}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-1">原文 §14.5.1</a>

主成分用完 SVD 就结束本章的一半故事；另一半的第一个例子是**带旋转的形状配准**。设 $X_1,X_2\in\mathbb{R}^{N\times p}$ 是两组 $N$ 个对应点（第 $i$ 行是同一个「地标」在两件物品上的位置，见原书图 14.25 的两个手写 S），$R\in O(p)$ 正交矩阵（可含反射），$\mu\in\mathbb{R}^p$ 是平移，求解

$$
\min_{\mu,R}\ \lVert X_2-\big(X_1R+\mathbf 1\mu^\top\big)\rVert_F,\qquad \lVert X\rVert_F^2=\operatorname{tr}(X^\top X)
$$

$$\eqno{14.56}$$

这就是 Procrustes 问题（原书以非洲劫匪 Procrustes 的「把客人拉长压扁来套上铁床」的典故命名）。解法只有三步，且全部是上面 PCA 论证的复用：

1. **对 $\mu$ 求导**（与 (14.51) 同型，只是 $R$ 固定）：$\nabla_\mu\lVert X_2-X_1R-\mathbf 1\mu^\top\rVert_F^2=2\mathbf 1(N\mu^\top-\mathbf 1^\top(X_2-X_1R))=0$，故最优平移把两者的行重心对齐：$\hat\mu=\bar x_2-\hat R\,\bar x_1$。代回后 $\mu$ 完全消失，剩下的问题只用中心化矩阵 $\tilde X_1=X_1-\mathbf 1\bar x_1^\top$、$\tilde X_2=X_2-\mathbf 1\bar x_2^\top$。
2. **改写成迹**（用 $\lVert A-B\rVert_F^2=\operatorname{tr}(A^\top A)+\operatorname{tr}(B^\top B)-2\operatorname{tr}(A^\top B)$，把 $\operatorname{tr}(\tilde X_1^\top\tilde X_1)$ 看成常量）：

   $$\min_R\ \operatorname{tr}(\tilde X_2^\top\tilde X_2)-2\operatorname{tr}\big(R^\top\tilde X_1^\top\tilde X_2\big)\quad\Longleftrightarrow\quad\max_{R\in O(p)}\operatorname{tr}\big(R^\top\tilde X_1^\top\tilde X_2\big)$$

   这与 PCA 的 $\max \operatorname{tr}(V_q^\top\Sigma V_q)$ 是**同一个问题**（正交群上的迹极值）。
3. **解**（把 $\tilde X_1^\top\tilde X_2$ 做 SVD）落在下一分片：$\hat R=UV^\top$、$\hat\mu=\bar x_2-\hat R\bar x_1$，对应原书 (14.57)，这里只给出「为什么答案是 SVD 的 $UV^\top$」的线索 —— 迹最大时 $\tilde X_1\hat R$ 与 $\tilde X_2$ 的夹角为零，即把 $\tilde X_1$ 的奇异向量逐个转到 $\tilde X_2$ 的奇异向量方向；$d_m=0$ 的方向上取向任意，所以解不唯一。

> **结果** · PCA 与 Procrustes 的共同骨架
>
> $$\max_{U^\top U=\mathbf 1}\operatorname{tr}(U^\top A)=\sum_{m} d_m\quad\Longleftrightarrow\quad U=\text{左奇异向量}\times\text{右奇异向量}^\top$$
>
> 这条「在正交约束下最大化迹」的结论，PCA 里是 $\operatorname{tr}(V^\top\Sigma V)$ 取前 $q$ 个特征值，Procrustes 里是 $\operatorname{tr}(R^\top\tilde X_1^\top\tilde X_2)$ 取奇异值之和。差别只在 $A$ 是对称半正定的（PCA）还是一般矩阵（Procrustes）。

本分片覆盖 (14.46)–(14.56)。自组织映射给出在线更新 (14.46)、带邻域函数的更新 (14.47) 与批量加权重心 (14.48)，并把「拓扑约束」与 $K$-means 的差别定位到「更新谁、权重怎么衰减」这一步；主成分部分从线性流形模型 (14.49) 与重构误差 (14.50) 出发，先做两次部分最小化得到 (14.51)(14.52)，把 $V_q$ 的极小化写成 (14.53)，再分别用最大方差准则与得分-载荷回归推出同一个解，并给出两者等价性的完整链条 $\mathrm{RSS}_q=N\sum_{m>q}\lambda_m$；随后是 SVD 解 (14.54)、两个成分的显式形式 (14.55)、标准化问题，以及作为「正交群上迹极值」的第一个例子的 Procrustes 问题 (14.56)。