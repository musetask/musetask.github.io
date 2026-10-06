本分片接着前两个分片的编号 (14.57) 往下推。原书这一段的路线是：先把 Procrustes 形状平均与主曲线的**自洽条件**写清楚，再进入 14.5.3–14.5.5 的**谱方法**（图拉普拉斯、核主成分、稀疏主成分），然后是 14.6 的**非负矩阵分解与原型分析**，最后是 14.7 的**独立成分分析**。数学工具集中在四个点：

- **二次型极值**：Rayleigh 商定理 + 拉格朗日乘子（预备知识 L1、L4、O2）
- **SVD 与正交约束下的极值**：Procrustes 与它的迹形式（预备知识 L3、O2）
- **共轭与对偶**：稀疏主成分、NMF 的乘性更新（预备知识 O3、O4）
- **信息论**：熵、互信息、负熵，以及高斯最大熵定理（预备知识 P5）

每一节都把原书压成一行结论的式子补成可手算的步骤。凡是原书没有写出中间步骤的（例如 (14.63) 之后的「为什么取最小特征向量」、(14.74) 的乘性更新、(14.85) 的 $\log|\det A|$ 项），本分片都给出完整推导。

---

## 14.5.2b 旋转 Procrustes 解的显式形式 {#s-14-5-2b}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-2">原文 §14.5.2</a>

前一分片已经把 Procrustes 问题 (14.56) 化成了「在正交群上求迹的极值」，但只给了答案。这里把 (14.57)–(14.60) 四个式子的每一步算出来。

> **基础知识** · 正交群上的 von Neumann 不等式
>
> 设 $A,B\in\mathbb{R}^{p\times p}$，$A=U_A D_A V_A^\top$、$B=U_B D_B V_B^\top$ 是奇异值分解，奇异值都非负递减。对任意 $R\in O(p)$（即 $R^\top R=I$），
>
> $$\lVert A-BR\rVert_F^2=\lVert A\rVert_F^2+\lVert B\rVert_F^2-2\mathrm{tr}(BR^\top A)\ \ge\ \lVert A\rVert_F^2+\lVert B\rVert_F^2-2\sum_{k=1}^p \sigma_k(A)\sigma_k(B)$$
>
> 等号成立当且仅当 $R=U_AU_B^\top$。这条不等式在正交群取遍极值时反复用到，下面给出它的证明。

### 14.5.2b.1 中心化与迹形式 {#s-14-5-2b-1}

问题 (14.56) 是在 $\mu$ 与正交阵 $R$ 上极小化配准误差

$$
\min_{\mu,R}\ \lVert X_2-X_1R-\mathbf{1}\mu^\top\rVert_F^2,\qquad R^\top R=I
$$

其中 $X_1,X_2$ 是 $N\times p$，$R$ 是 $p\times p$ 正交阵，$\mu$ 是 $p$ 维位置向量（$\mathbf{1}\mu^\top$ 的每一行都是 $\mu^\top$，即把每个观测平移同一个向量）。

第一步：记列均值 $\bar x_1=N^{-1}X_1^\top\mathbf{1}$、$\bar x_2=N^{-1}X_2^\top\mathbf{1}$，中心化矩阵 $\tilde X_1=X_1-\mathbf{1}\bar x_1^\top$、$\tilde X_2=X_2-\mathbf{1}\bar x_2^\top$。因为 $R$ 与 $\mathbf{1}$ 交换，

$$
X_2-X_1R-\mathbf{1}\mu^\top=\tilde X_2-\tilde X_1R+\mathbf{1}\bigl(\bar x_2^\top-\mu^\top-\bar x_1^\top R\bigr)
$$

右边第二项的每一行都是同一个向量，由行之间的正交性（见预备知识 L2 末尾的 Pythagoras 恒等式）得

$$
\lVert X_2-X_1R-\mathbf{1}\mu^\top\rVert_F^2=\lVert \tilde X_2-\tilde X_1R\rVert_F^2+N\bigl\lVert \bar x_2-\mu^\top-\bar x_1^\top R\bigr\rVert_2^2
$$

所以对 $R$ 的极小化与 $\mu$ 无关，而 $\mu$ 的最优解（把上式第二项对 $\mu$ 求导置零）就是原书给的「配准后中心重合」条件。

第二步：展开剩下的 Frobenius 范数。用 $\lVert A\rVert_F^2=\mathrm{tr}(A^\top A)=\sum_{i,j}A_{ij}^2$（预备知识 L1）以及 $\mathrm{tr}(PQ)=\mathrm{tr}(QP)$：

$$
\lVert \tilde X_2-\tilde X_1R\rVert_F^2=\mathrm{tr}(\tilde X_2^\top\tilde X_2)+\mathrm{tr}\bigl(R^\top\tilde X_1^\top\tilde X_1R\bigr)-2\mathrm{tr}\bigl(\tilde X_2^\top\tilde X_1R\bigr)
$$

第三步：$R^\top\tilde X_1^\top\tilde X_1R$ 的迹可以搬到外面。设 $B=R^\top\tilde X_1^\top\tilde X_1R$，由 $\mathrm{tr}(B)=\mathrm{tr}(B^\top)$，

$$
\mathrm{tr}\bigl(R^\top\tilde X_1^\top\tilde X_1R\bigr)=\mathrm{tr}\bigl(R^\top\tilde X_1^\top\tilde X_1RR^\top\bigr)=\mathrm{tr}\bigl(\tilde X_1^\top\tilde X_1\bigr)=\lVert\tilde X_1\rVert_F^2
$$

（第三步等号用 $RR^\top=I$，$R$ 是方阵所以左逆也是右逆。）这一项**与 $R$ 无关**，于是 (14.56) 的 $R$-部分完全化为

$$
\min_{R^\top R=I}\ \bigl(-2\mathrm{tr}(\tilde X_2^\top\tilde X_1R)\bigr)
\quad\Longleftrightarrow\quad
\max_{R^\top R=I}\ \mathrm{tr}(RS),\qquad S=\tilde X_1^\top\tilde X_2
$$

即：**在正交群上最大化一个线性函数的迹**。

### 14.5.2b.2 解 $R=\hat R$ 与 $\hat\mu$ {#s-14-5-2b-2}

取 $S$ 的 SVD $S=UDV^\top$（预备知识 L3），$U,V$ 正交、$D=\mathrm{diag}(d_1,\dots,d_p)$，$d_1\ge d_2\ge\dots\ge d_p\ge0$。要证的事只有一件：$\mathrm{tr}(RUDV^\top)$ 的最大值是 $\sum_k d_k$。

$$
\mathrm{tr}(RUDV^\top)=\mathrm{tr}\bigl(DV^\top RU\bigr)=\sum_{k=1}^p D_{kk}\,(V^\top RU)_{kk}=\sum_{k=1}^p d_k\,Q_{kk},\qquad Q:=V^\top RU
$$

$Q$ 是正交阵（正交阵的乘积仍是正交阵）。对任意正交阵 $Q$ 与任意向量 $x$ 有 $x^\top Qx\le\lVert x\rVert^2$，因为 $x^\top Qx=x^\top\frac{Q+Q^\top}{2}x\le\lVert x\rVert$（取 $x=e_k$ 即得 $Q_{kk}\le1$）。所以 $Q_{kk}\le1$，又 $d_k\ge0$，于是

$$
\sum_{k=1}^p d_kQ_{kk}\le\sum_{k=1}^p d_k=\mathrm{tr}(D)
$$

等号条件：所有 $d_k>0$ 处都要 $Q_{kk}=1$；因 $\lVert Qe_k\rVert=1$ 而 $Q_{kk}=\langle Qe_k,e_k\rangle=1$，由 Cauchy–Schwarz 的等号条件得 $Qe_k=e_k$，即 $Q=I$。若某些 $d_k=0$（$S$ 秩亏），那些位置不要求 $Q_{kk}=1$，这就是解不唯一的唯一来源。于是

$$
Q=V^\top RU=I\ \Longrightarrow\ R=UV^\top
$$

代回第一步的配准条件，得原书 (14.57)：

$$
\hat R=UV^\top,\qquad \hat\mu=\bar x_2-\hat R\bar x_1^\top \eqno{14.57}
$$

> **推导** · 一阶条件也能验证（KKT 自洽）
>
> 带约束的 Lagrange 函数
>
> $$
> \mathcal{L}(R,\Lambda)=\mathrm{tr}(RS)-\frac{1}{2}\mathrm{tr}\bigl(\Lambda(R^\top R-I)\bigr)
> $$
>
> 其中 $\Lambda$ 对称。$\frac{\partial}{\partial R}$ 得 $S=\Lambda R$，即 $S^\top R=R^\top S$。取 $R=UV^\top$：$S^\top R=VDU^\top UV^\top=VD$，$R^\top S=VU^\top UDV^\top=VD$，两边相等，条件满足。注意这里的条件是 $S^\top R=R^\top S$ 而不是 $S=2\lambda R$，因为 $S$ 一般不对称——这正是「Procrustes 有唯一解」与「一般线性最小二乘要 $S$ 对称」的差别。
>
> **结果** · 最小 Procrustes 距离
>
> $$\text{Procrustes dist}^2(X_1,X_2)=\lVert\tilde X_1\rVert_F^2+\lVert\tilde X_2\rVert_F^2-2\sum_{k=1}^p d_k$$
>
> 其中 $d_k$ 是 $S=\tilde X_1^\top\tilde X_2$ 的奇异值。

### 14.5.2b.3 带缩放的 Procrustes (14.58) {#s-14-5-2b-3}

允许逐点缩放 $\beta>0$ 时，问题是

$$
\min_{\beta,R}\ \lVert X_2-\beta X_1R\rVert_F^2 \eqno{14.58}
$$

同样先中心化（$\beta$ 与 $R$ 都与 $\mathbf{1}$ 交换），得到 $\min_{\beta,R}\lVert\tilde X_2-\beta\tilde X_1R\rVert_F^2$。展开：

$$
\mathrm{tr}(\tilde X_2^\top\tilde X_2)+\beta^2\lVert\tilde X_1\rVert_F^2-2\beta\,\mathrm{tr}(RS)
$$

固定 $R$，这是关于 $\beta$ 的严格凸二次函数，极小点 $\beta^\ast=\mathrm{tr}(RS)/\lVert\tilde X_1\rVert_F^2$；代入得目标值 $\lVert\tilde X_2\rVert_F^2-\mathrm{tr}(RS)^2/\lVert\tilde X_1\rVert_F^2$，关于 $R$ 单调（$\mathrm{tr}(RS)\ge0$ 时）单调递减，故 $R$ 仍取 $UV^\top$。于是

$$
\hat\beta=\frac{\mathrm{tr}(\hat R\tilde X_1^\top\tilde X_2)}{\lVert\tilde X_1\rVert_F^2}=\frac{\mathrm{tr}(D)}{\lVert\tilde X_1\rVert_F^2}
$$

正是原书所说「$R$ 的解与前面一样，而 $\hat\beta=\mathrm{tr}(D)/\lVert X_1\rVert_F^2$」。**缩放因子与旋转不分离**：数据先除以 $\hat\beta$ 再做 Procrustes，等价于先做 Procrustes 再除以 $\hat\beta$。

### 14.5.2b.4 Procrustes 平均 (14.59) 与仿射不变平均 (14.60) {#s-14-5-2b-4}

$L$ 个形状 $X_\ell$（都中心化）的平均问题：

$$
\min_{\{R_\ell\},\,M}\ \sum_{\ell=1}^L\lVert X_\ell R_\ell-M\rVert_F^2 \eqno{14.59}
$$

交替算法：先固定 $M$ 求 $L$ 个 (14.56)（每个解是 (14.57)）；再固定 $R_\ell$ 求 $M$。第二步是线性的：

$$
\frac{\partial}{\partial M}\sum_\ell\lVert X_\ell R_\ell-M\rVert_F^2=2\sum_\ell(M-X_\ell R_\ell)=0
\ \Longrightarrow\ M=\frac1L\sum_{\ell=1}^L X_\ell R_\ell
$$

单调性可以逐半步证明：记 $J(\{R\},M)=\sum_\ell\lVert X_\ell R_\ell-M\rVert^2_F$。第一步把每个 $R_\ell$ 换成使 $J(\{R_\ell\},M)$ 最小的 $\hat R_\ell$，故 $J$ 不增；第二步把 $M$ 换成上面的一阶条件解（$J$ 对 $M$ 是严格凸二次，故是全局最小），故 $J$ 又不增。所以迭代单调下降并收敛到局部极小。

带缩放与仿射的版本 (14.60) 有闭式解：

$$
\sum_{\ell=1}^L\min_{A_\ell}\ \lVert X_\ell A_\ell-M\rVert_F^2,\qquad A_\ell\ \text{任意可逆} \eqno{14.60}
$$

> **推导** · 为什么 $M$ 取 $\bar H$ 的前 $p$ 个特征向量
>
> 固定 $M$，先解内层的 $\min_{A_\ell}$。因为 $\lVert X_\ell A_\ell-M\rVert_F^2$ 按列可分，等价于对 $M$ 的每一列 $m_j$ 做最小二乘投影（预备知识 L2）：
>
> $$\min_{a}\lVert X_\ell a-m_j\rVert_2^2\ \Rightarrow\ \hat a=X_\ell^\dagger m_j,\qquad \min=\lVert m_j\rVert^2-m_j^\top H_\ell m_j$$
>
> 其中 $X_\ell^\dagger=(X_\ell^\top X_\ell)^{-1}X_\ell^\top$，$H_\ell=X_\ell(X_\ell^\top X_\ell)^{-1}X_\ell^\top$ 是 $\mathrm{col}(X_\ell)$ 上的正交投影（$H_\ell^2=H_\ell$，$\mathrm{tr}(H_\ell)=p$）。残差恰为 $m_j(I-H_\ell)$。对 $j=1,\dots,p$ 求和：
>
> $$J(M)=\sum_{\ell=1}^L\mathrm{tr}\Bigl(M^\top(I-H_\ell)M\Bigr)=\mathrm{tr}(M^\top M)-\mathrm{tr}\Bigl(M^\top\Bigl(\sum_{\ell=1}^LH_\ell\Bigr)M\Bigr)$$
>
> 加上标准化 $M^\top M=I$ 后第一项变成常数 $p$，于是**极小化 $J$ 等价于在 $M^\top M=I$ 下极大化 $\mathrm{tr}(M^\top\bar H M)$**（$\bar H=\frac1L\sum_\ell H_\ell$，用 $\bar H$ 与 $\sum_\ell H_\ell$ 只差正因子，不影响特征向量）。这一步正是 (14.57) 用过的 von Neumann 论证：设 $\bar H=\sum_{k=1}^p\mu_kz_kz_k^\top$（$\mu_1\ge\dots\ge\mu_p\ge0$，$\{z_k\}$ 正交），则
>
> $$\mathrm{tr}(M^\top\bar H M)=\sum_{j,k}\mu_k\bigl(m_j^\top z_k\bigr)^2\ \le\ \sum_{k=1}^p\mu_k\sum_{j=1}^p\bigl(m_j^\top z_k\bigr)^2\ \le\ \sum_{k=1}^p\mu_k$$
>
> 第一步用 $\mu_k\le\mu_{\le p}$ 时的 $\sum_k\mu_ka_k\le\sum_k\mu_{\le p}a_k$；第二步用 $\{z_k\}$ 是正交基，$\sum_j(m_j^\top z_k)^2\le1$（$M^\top M=I$ 表明 $\sum_k m_j^\top z_k\,m_{j'}^\top z_k=\delta_{jj'}$，即 $\{m_j\}$ 是压缩 $\{z_k\}$ 的等距系）。等号要求 $\{m_j\}$ 恰为 $\{z_1,\dots,z_p\}$。
>
> **结果** · 仿射不变平均的闭式解（对应原文步骤 1–2）
>
> $$H_\ell=X_\ell(X_\ell^\top X_\ell)^{-1}X_\ell^\top,\qquad \bar H=\tfrac1L\sum_{\ell=1}^LH_\ell,\qquad M\ =\ \bar H\ \text{的前 } p\ \text{个特征向量}$$
>
> 之所以是**投影矩阵的平均**而不是 $\sum_\ell X_\ell^\top X_\ell$ 的特征向量：$H_\ell$ 把每个形状的贡献按它的子空间（而不是按它的尺度）加权，因此结果对每个 $X_\ell$ 的整体缩放不敏感，这正是「仿射不变」三个字的来源。
>
> **坑** · $A_\ell=X_\ell^\dagger M$ 可能奇异（当 $m_j$ 落在 $\mathrm{col}(X_\ell)$ 内时残差为零、最小二乘解不唯一），这不影响目标值。

> **坑** · (14.60) 必须加标准化约束（如 $M^\top M=I$）才有非平凡解，否则取 $M=X_\ell A_\ell^{-1}$ 就能让目标为 0。另外解只到正交变换（旋转/镜像）不唯一，见原文脚注 4。

---

## 14.5.2c 主曲线的自洽条件与交替算法 {#s-14-5-2c}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-2">原文 §14.5.2</a>

主曲线把 PCA 的直线推广成光滑曲线 $f(\lambda)\in\mathbb{R}^p$，参数 $\lambda$ 可以取弧长。对每个数据 $x$，令 $\lambda_f(x)$ 是 $f$ 上离 $x$ 最近的点的参数。称 $f$ 是 $X$ 的主曲线，如果

$$
f(\lambda)=E\bigl(X\mid \lambda_f(X)=\lambda\bigr) \eqno{14.61}
$$

即「落在同一段曲线上的所有点是**它自己的平均**」。这就是自洽性（self-consistency）。

> **基础知识** · 交替最小化的两个条件
>
> 一个目标 $J(\theta,\eta)$ 对两组参数分别可分时，交替更新 $\theta\leftarrow\arg\min J(\cdot,\eta)$、$\eta\leftarrow\arg\min J(\theta,\cdot)$ 在每一步都不增 $J$，故 $J$ 单调下降。但**收敛点只保证是局部极小**；只有当每步的最优解对另一组参数唯一且问题严格凸时，才保证唯一收敛点。

有限样本下的算法是 (14.62) 的两步交替：

$$
\text{(a)}\ \hat f_j(\lambda)\ \leftarrow\ E\bigl(X_j\mid \hat\lambda(X)=\lambda\bigr);\qquad
\text{(b)}\ \hat\lambda(x)\ \leftarrow\ \arg\min_{\lambda}\ \lVert x-\hat f(\lambda)\rVert_2 \eqno{14.62}
$$

(a) 用散点光滑（把 $X_j$ 对 $\hat\lambda$ 做回归）估计条件均值，(b) 是最近点投影。两步各自是凸的（在各自的可行集上），但联合起来不是——这与 SOM、K-means 的结论一样：单调下降 + 局部解。

> **推导** · 线性光滑器如何退化成幂方法
>
> 原文说「若散点光滑用线性最小二乘，则过程收敛到第一主成分，且等价于求矩阵最大特征向量的幂方法」。逐项验证。设 $X$ 已中心化，$X=UDV^\top$（SVD），取第 $K$ 个主成分得分 $s_{iK}=d_Ku_{iK}$，$\lambda=d_K^2$（因为 $X^\top X=VD^2V^\top$，见预备知识 L3）。对 $X_j$ 关于 $s_{iK}$ 做含截距的最小二乘 $\hat f_j(\lambda)=a_j+b_j\lambda$，斜率是普通最小二乘解 $\hat\beta_{jk}=(\sum_i x_{ij}s_{iK})/\sum_i s_{iK}^2$。分子 $=\sum_i (u_{jK}d_Ku_{iK})u_{iK}d_K=d_K^2u_{jK}=\lambda u_{jK}$（用到 $U^\top U=I$ 的第 $K$ 列），分母 $=\sum_i d_K^2u_{iK}^2=\lambda$（同样 $u_{iK}^2$ 求和为 1）。故
>
> $$\hat\beta_{jK}=\frac{u_{jK}}{N\lambda},\qquad \text{新方向向量}\ \propto\ \lambda\hat\beta_{jK}=\frac{u_{jK}}N$$
>
> 新方向仍是 $u_{jK}$（只是差一个正因子），说明线性情形下这一步不动点就是第一主成分方向。而把它写成迭代映射 $v\mapsto X^\top Xv$：把 $v=Ne^{-1}…$ 直接算，$\hat f_j$ 的截距 $a_j=\bar x_j=0$，故新得分 $\tilde s_{iK}=\lambda\hat\beta_{jK}x_{ij}$，方向向量 $\tilde v_j=\lambda\hat\beta_{jK}=u_{jK}/N$，于是 $X\tilde v=u_K$，$v\mapsto Xu_K\propto X(X^\top Xu_K/\lambda)=XX^\top v/\lambda$，正是幂迭代 $v\leftarrow XX^\top v$（见预备知识 N 组的幂方法说明）。收敛性来自 $\lVert X^\top Xv\rVert^2$ 沿迭代方向单调增，且当 $v$ 落在 $\lambda_1/\lambda_2$ 的本征间隙外时误差以 $(\lambda_2/\lambda_1)^t$ 衰减。
>
> 主曲面是同一套东西的二维版本：$f(\lambda_1,\lambda_2)=[f_1,\dots,f_p]$，(a) 换成二维光滑器。它与 SOM 的区别是：主曲面给每个数据点一个自己的原型 $f(\hat\lambda_1(x_i),\hat\lambda_2(x_i))$，SOM 只有少量共享原型。

---

## 14.5.3 谱聚类 {#s-14-5-3}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-3">原文 §14.5.3</a>

$K$-means 用球形/椭球形度量分组，遇到同心的非凸簇（图 14.29 左上）就失效。谱聚类的做法是：先把观测变成一张**相似图**，再把「找非凸簇」变成「切图」，最后用图的拉普拉斯矩阵的特征向量做嵌入。这一节是本分片数学最密的部分，先把需要的三条定理备齐。

> **基础知识** · 谱方法三条定理
>
> **（1）Rayleigh 商定理**。设 $A=A^\top\in\mathbb{R}^{N\times N}$，特征分解 $A=\sum_{k=1}^N\lambda_ku_ku_k^\top$（$\lambda_1\ge\dots\ge\lambda_N$，$\{u_k\}$ 正交标准）。对任意 $v\ne0$，令 $c_k=u_k^\top v$，则
>
> $$\frac{v^\top Av}{v^\top v}=\frac{\sum_k\lambda_kc_k^2}{\sum_kc_k^2}=\lambda_1-\frac{\sum_k(\lambda_1-\lambda_k)c_k^2}{\sum_kc_k^2}$$
>
> 立刻得到：$\min_{v\ne0}\dfrac{v^\top Av}{\lVert v\rVert^2}=\lambda_1$（取 $v=u_1$），$\max=\lambda_N$；且等号条件是 $v$ 落在极值特征值对应的特征空间里。
>
> **（2）Ky Fan 极值定理**（Rayleigh 商的矩阵版）。$\max_{M^\top M=I_K}\mathrm{tr}(M^\top AM)=\sum_{k=1}^K\lambda_k$，等号当且仅当 $M$ 的列张成 $\lambda_1,\dots,\lambda_K$ 的特征空间。证明同上：设 $C=Q^\top M$（$A=Q\Lambda Q^\top$），则 $\mathrm{tr}(M^\top AM)=\sum_{jk}\lambda_jC_{jk}^2\le\sum_{j\le K}\lambda_j$，最后一步把权重按 $\lambda$ 排序后用重排不等式（$\{C_{jk}^2\}$ 与 $\{1\}$ 的配对和以配大者为最大）。
>
> **（3）带等式约束二次型的 Lagrange 乘子（KKT）**。$\min_y q(y)=\tfrac1n\mathrm{Tr}(Y^\top AY)$ s.t. $G(y)=0$（$G(y)$ 为矩阵等式），Lagrange 函数 $\mathcal{L}(y,\Lambda)=q(y)-\mathrm{tr}(\Lambda^\top G(y))$，$\Lambda$ 对称。稳定点条件 $\nabla_y\mathcal{L}=\tfrac1n(2AY-2\Lambda)=0$ 即 $AY=\Lambda y$；二阶充分条件由 $\tfrac1n(A-\Lambda)\succ 0$ 在约束流形上给出。在等式约束下 Slater 条件自动满足（零矩阵总在可行集里），故 KKT 必要且（在紧约束集上）充分。
>
> **（4）图拉普拉斯的基本性质**。$W=W^\top\ge0$（对称非负权重，对角一般为零），$g_i=\sum_{i'}w_{ii'}$，$G=\mathrm{diag}(g_1,\dots,g_N)$，$L=G-W$。则
>
> - $x^\top Lx=\sum_{i,i'}w_{ii'}(x_i-x_{i'})^2/2\ge0$，故 $L\succeq 0$；
> - $L\mathbf{1}=0$，即 $\mathbf{1}$ 是特征值 $0$ 的特征向量；
> - 若图有 $m$ 个连通分量，则零特征值有 $m$ 个，几何重数等于代数重数（$L$ 对称），零特征空间由各分量的示性向量张成。证明：把顶点按分量重排，$W$（与 $L$）变成分块对角，每块是不可约的非负对称矩阵，其谱半径 $\rho=\max_i g_i$（Perron–Frobenius），是单特征值，其余特征值严格小于 $\rho$。

### 14.5.3.1 从相似图到拉普拉斯 (14.63) {#s-14-5-3-1}

起点是 $N\times N$ 的成对相似度 $s_{ii'}\ge0$。把它们看成无向相似图 $G=\langle V,E\rangle$ 的边权，得到邻接矩阵 $W$（$\mathcal{N}_K$-近邻图：$(i,i')\in\mathcal{N}_K$ 当且仅当 $i$ 是 $i'$ 的 $K$ 近邻之一或反之，只保留这些权重、其余置零；全连通图则保留全部 $w_{ii'}=s_{ii'}$，局部性由核尺度 $c$ 控制）。顶点 $i$ 的度 $g_i=\sum_{i'}w_{ii'}$，$G=\mathrm{diag}(g_i)$。图拉普拉斯

$$
L=G-W \eqno{14.63}
$$

称为**未归一化**拉普拉斯；归一化版本用度数把节点标准化，例如 $\tilde L=I-G^{-1}W$。谱聚类算法：取 $L$ 的 $m$ 个**最小**特征值对应的特征向量组成 $Z\in\mathbb{R}^{N\times m}$（跳过平凡的常数向量），再对 $Z$ 的行做 $K$-means。

### 14.5.3.2 为什么是最小特征向量 (14.64) {#s-14-5-3-2}

原文只给一行：$f^\top Lf$ 等于边权乘坐标差平方和的一半。它的含义与「取最小特征向量」的最优性需要分开证明。

第一步，算 $f^\top Lf$。$f^\top Lf=\sum_{i,i'}f_i(g_i-f_{i'})w_{ii'}$ 是 $f^\top Gf-f^\top Wf$ 的分量写法，注意对称性使得
$$
f^\top Wf=\sum_{i,i'}f_if_{i'}w_{ii'}=\tfrac12\sum_{i,i'}(f_i+f_{i'})^2w_{ii'}
$$
（用 $w_{ii'}=w_{i'i}$ 把 $\tfrac12\sum(f_i^2+f_{i'}^2)w_{ii'}$ 拆出来等于 $\sum_i f_i^2g_i$），于是

$$
f^\top Lf=\sum_{i=1}^N f_i^2g_i-\sum_{i,i'}f_if_{i'}w_{ii'}=\frac12\sum_{i=1}^N\sum_{i'=1}^Nw_{ii'}(f_i-f_{i'})^2 \eqno{14.64}
$$

推导：
$$
\frac12\sum_{i,i'}w_{ii'}(f_i-f_{i'})^2=\frac12\sum_{i,i'}w_{ii'}(f_i^2+f_{i'}^2-2f_if_{i'})
$$
其中前两项
$$
=\frac12\sum_i f_i^2\sum_{i'}w_{ii'}+\frac12\sum_{i'}f_{i'}^2\sum_iw_{ii'}=\sum_i f_i^2g_i
$$

（$w$ 对称），第三项 $=\sum_{i,i'}f_if_{i'}w_{ii'}$。得证。

第二步，语义。$f^\top Lf$ 小 ⟺ 所有**大权边**两端的坐标差小 ⟺ 相邻的观测在 $f$ 这张「地图」上坐标相近。于是**把连通分量映到同一个点**就是让 $f^\top Lf$ 变小的最强方式；同时，如果图连通，$f^\top Lf=0$ 只可能在 $f$ 为常向量时取到（由 (14.64)：和为零要求每条边 $w_{ii'}>0$ 处 $f_i=f_{i'}$，连通即全体相等），所以常向量是零特征值唯一的特征向量。若图有 $m$ 个分量，按分量示性向量重排则 $L$ 分块对角，零特征值恰有 $m$ 个。这就是为什么**取「除 $\mathbf{1}$ 外最小的 $m-1$ 个特征向量」**：它们近似各分量的示性向量。

第三步，**最优性**。把「找一个把相邻点放在一起的低维嵌入」写成 Rayleigh 商问题：

$$
\min_{f}\ \frac1N\mathrm{Tr}(F^\top LF)\quad\text{s.t.}\quad \frac1N F^\top F=I_K \qquad (F\in\mathbb{R}^{N\times K})
$$

- 上界（Rayleigh 商的迹形式）：由基础 (2)（Ky Fan），
  $$\frac1N\mathrm{Tr}(F^\top LF)=\frac1N\sum_{k=1}^K\sum_{j}\lambda_ju_j^\top f_kf_k^\top u_j=\frac1N\sum_{j\le K}\lambda_j\sum_{k\le K}(u_j^\top f_k)^2\le\frac1N\sum_{j=1}^K\lambda_j$$
  等号当且仅当 $\{f_1,\dots,f_K\}$ 张成 $\{u_1,\dots,u_K\}$（最小特征值对应的特征空间）。
- 下界（KKT 给出的必要条件）：Lagrange 函数
$$
\mathcal{L}(F,\Lambda)=\frac1N\mathrm{Tr}(F^\top LF)-\mathrm{tr}\bigl(\Lambda(F^\top F-NI)\bigr)
$$
。对 $F$ 求导（预备知识 L4：$\nabla\mathrm{Tr}(F^\top LF)=2Lf$ 对 $L$ 对称）得
  $$2\bigl(LF-\Lambda F\bigr)=0\ \Longrightarrow\ LF=\Lambda F$$
  故最优解的每一列都是 $L$ 的特征向量（$\Lambda$ 的对角元即对应的特征值）。两侧对 $\Lambda$ 求导给出可行性条件 $\frac1NF^\top F=I$。所以「最优解必是特征向量」这一步是 KKT 给的，「取最小的几个」这一步是 Ky Fan 给的。

> **结果** · 谱聚类的嵌入
>
> $$L U=U\Lambda,\qquad Z=\bigl[u_2,u_3,\dots,u_{K+1}\bigr]\ \in\mathbb{R}^{N\times K}$$
>
> 然后对 $Z$ 的 $N$ 行做 $K$-means。（丢掉 $u_1\propto\mathbf{1}$ 是因为它不携带任何信息。）

第四步，**为什么 $K$-means 在 $Z$ 的行上有效**。$z_i=u_{2i},\dots$ 这一列把「在图上不连通」的点分开：若图由两个几乎不连通的子图组成，则 (14.64) 的和被两组内部项主导，于是 $u_2$ 在两组上近似取两个不同常数（因为组内边权大、$f$ 在组内要接近，而组间要求不强），这正是图 14.29 右下角的效果。$K$-means 在 $\mathbb{R}^K$ 里对「两团常数」当然毫无困难。

> **坑** · 实践中有四个自由度要选：相似图类型（全连通 or 近邻）、近邻数 $k$ 或核尺度 $c$、取几个特征向量、簇数 $K$。图 14.29 的玩具例子里 $k\in[5,200]$ 都好用（$k=200$ 即全连通），$k<5$ 变差；而右上角谱图上前三个特征值与其余并没有明显间隙，所以「取几个」这一步没有原则性判据。这是谱聚类最实际的困难。

### 14.5.3.3 归一化拉普拉斯与 NJW 谱聚类的对偶推导 {#s-14-5-3-3}

原文说「归一化版本有很多，例如 $\tilde L=I-G^{-1}W$」，并指出令 $P=G^{-1}W$ 就得到一个图上的随机游走转移矩阵。下面把归一化版本做完整：**从迹比形式出发，经 Lagrange 对偶得到 $Y=UD_m^{1/2}$，再说明为什么 $D_m^{1/2}$ 被丢掉**。

第一步，Ng–Jordan–Weinberger (NJW) 形式：找 $Y\in\mathbb{R}^{N\times m}$ 最大化「簇内边权」与「割边权 + 节点体积」之比，

$$
\max_{Y}\ \frac{\mathrm{Tr}(Y^\top WY)}{\mathrm{Tr}(Y^\top GY)}\quad\text{s.t.}\quad \frac1nY^\top Y=I_m
$$

为什么要分母：因为簇被切开时，切掉的边权和节点度数都小；分母让「体积小的簇」不被算法忽略。对比 (14.64) 的分母恒为 $\frac12\sum_i g_i=\mathrm{tr}(W)$（常值），所以这是 (14.64) 的真正推广。

第二步，迹比 = 迹商。令 $\lambda^\star$ 为下式的最小值：

$$
\lambda^\star=\min_{Y}\ \frac{\mathrm{Tr}(Y^\top WY)}{\mathrm{Tr}(Y^\top GY)}\ \text{s.t.}\ \tfrac1nY^\top Y=I_m
$$

因为 $\lambda^\star$ 是 Rayleigh 商 $\{Y^\top WY\}/\{Y^\top GY\}$ 的最小值，有 $\mathrm{Tr}(Y^\top WY)\ge\lambda^\star\mathrm{Tr}(Y^\top GY)$，反过来也成立（取到最小值的 $Y$），故 $\max$ 与 $\min$ 等价。

第三步，Lagrange 对偶。原问题带一个**分式**约束，标准技巧是把不等式 $\mathrm{Tr}(Y^\top WY)\ge\lambda\mathrm{Tr}(Y^\top GY)$ 的拉格朗日函数取对偶。固定 $\lambda$，考虑

$$
\min_{Y}\ \mathrm{Tr}\bigl(Y^\top(L+\lambda G)Y\bigr)\quad\text{s.t.}\ \tfrac1nY^\top Y=I_m \qquad\text{（原问题）}
$$

它的对偶是：对对称乘子 $\Lambda$，

$$
\max_\Lambda\ \min_{Y}\ \mathcal{L}(Y,\Lambda),\qquad \mathcal{L}(Y,\Lambda)=\mathrm{Tr}\bigl(Y^\top(L+\lambda G)Y\bigr)-\mathrm{tr}\bigl(\Lambda(Y^\top Y-nI_m)\bigr)
$$

因为 $(L+\lambda G)\succeq 0$ 且 $L\succ 0$（图连通时），$\mathcal{L}$ 对 $Y$ 是凸的、对 $\Lambda$ 是线性的，满足 Slater 条件（取 $Y$ 为任意 $n\times m$ 列正交阵即可），强对偶成立（见预备知识 O4）。

第四步，内层极小与 KKT。固定 $\Lambda$，
$$
\partial\mathcal{L}/\partial Y=2\bigl((L+\lambda G)Y-\Lambda Y\bigr)
$$
故内层最优 $Y_\Lambda$ 满足

$$
(L+\lambda G)Y_\Lambda=\Lambda Y_\Lambda,\qquad \tfrac1nY_\Lambda^\top Y_\Lambda=I_m
$$

再对 $\Lambda$ 求导（可行方向 $-A$ 在 $\mathcal{L}$ 上的方向导数为 $-\mathrm{Tr}(A^\top A)$，须 $\ge0$，即 $A^\top A\preceq 0$，迫使 $A=0$）得同一组条件——所以互补条件自动满足，KKT 就是这两式。

第五步，解出来。记 $L+\lambda G$ 的特征值 $\mu_1\le\dots\le\mu_n$，对应正交特征向量 $u_1,\dots,u_n$。对满足 $\frac1nY^\top Y=I_m$ 的 $Y$，

$$
\mathrm{Tr}\bigl(Y^\top(L+\lambda G)Y\bigr)=\sum_{j}\mu_j\sum_{k\le m}(u_j^\top y_k)^2\ \ge\ \sum_{j=1}^m\mu_j
$$

（把权重 $\mu_j$ 配到最大的 $\sum_k(u_j^\top y_k)^2$ 上；这些平方和之和为 $m$，且 $\sum_{j\le m}\mu_j$ 最小）。等号当且仅当 $Y$ 的列张成 $\{u_1,\dots,u_m\}$，再由 $\frac1nY^\top Y=I_m$ 得唯一解

$$
Y_\star=\sqrt{n}\,U_mD_m^{1/2}
$$

其中 $U_m=[u_1,\dots,u_m]$，$D_m=\mathrm{diag}(\mu_1,\dots,\mu_m)$。归一化掉常数 $\sqrt{n}$ 并按列缩放（列的缩放不改变「哪些点被 K-means 分到一组」，只改变它们在 $\mathbb{R}^m$ 里的疏密），**丢掉 $D_m^{1/2}$** 就是 NJW 算法的第一步：

$$
Y\ \leftarrow\ U_m
$$

即 $L+\lambda G$ 的前 $m$ 个最小特征向量。

第六步，为什么 $\lambda=1$ 时它就是归一化拉普拉斯 $\tilde L=I-G^{-1}W$。记 $S=G^{-1/2}$（要求各 $g_i>0$；孤立点要先删或补边）。作代换 $Y=SZ$，则

$$
Y^\top GY=Z^\top S^\top GSZ=Z^\top Z,\qquad Y^\top WY=Z^\top G^{-1/2}WG^{-1/2}Z
$$

于是迹比变成

$$
\frac{\mathrm{Tr}(Y^\top WY)}{\mathrm{Tr}(Y^\top GY)}=\frac{1}{n}\,\mathrm{Tr}\bigl(Z^\top P_sZ\bigr),\qquad P_s:=G^{-1/2}WG^{-1/2}
$$

而约束那一项因为 $Y^\top Y=Z^\top S^\top SZ=Z^\top Z$（$S$ 对称），**原封不动地变成** $\frac1nZ^\top Z=I_m$，即 $Z$ 的列标准正交。注意 $P_s$ 与随机游走矩阵 $P=G^{-1}W$ **相似**（$P_s=S^{-1}PS$，$S$ 对称可逆），故谱相同。于是迹比的极大值 $=\frac1m\sum_{j\le m}\mu_j(P)$（Ky Fan 定理，$P_s$ 对称），最优 $Z$ 是 $P_s$ 的前 $m$ 个特征向量。

- 因为 $Py=P_s(Sy)$（用 $SP_s=WG^{-1}=P$），$P_s$ 的特征向量 $\{z_j\}$ 一一对应 $P$ 的特征向量 $\{Sz_j\}$。
- $P$ 的最大特征值 $\mu$ 与 $\tilde L=I-P$ 的特征值 $1-\mu$ 换序，所以「$P$ 的前 $m$ 大特征向量」$=$「$\tilde L=I-G^{-1}W$ 的前 $m$ 小特征向量」。
- 最后回到 $Y=G^{-1/2}Z$：**$Y$ 的第 $i$ 行被乘上了 $g_i^{-1/2}$**，这就是 NJW 算法里的「按度数加权 / 行归一化」。

$$\tilde L=I-G^{-1}W,\qquad Y=G^{-1/2}\bigl[\text{其前 } m\ \text{小特征向量}\bigr]$$

> **结果** · NJW 谱聚类的两步
>
> 1. 算 $P=G^{-1}W$（行随机）或 $L=G-W$，取前 $m$ 个特征向量 $Y=U_m$（行归一化到 $\lVert y_i\rVert=1$ 后再做 $K$-means）。
> 2. 输出 $Y$ 的行分组。
>
> 几何解释：$Y$ 的行被放在球面上（因为行归一化），球面上的「簇」就是原图上强连通的部分；随机游走很少从一组跳到另一组。

> **坑** · 行归一化这一步不是可有可无的：不同顶点的度数差别大时，不归一化会让高阶节点主导欧氏距离，行归一化相当于把每个点放到单位球面上比较方向。

---

## 14.5.4 核主成分与谱方法的关系 {#s-14-5-4}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-4">原文 §14.5.4</a>

### 14.5.4.1 双中心化的 gram 矩阵 (14.65) {#s-14-5-4-1}

线性 PCA 可以完全用内积矩阵 $K=XX^\top$（$N\times N$）算。中心化的那一步在 gram 矩阵里长什么样？设 $M=\mathbf{1}\mathbf{1}^\top/N$，中心化矩阵 $\tilde X=(I-M)X$。则

$$
\tilde X\tilde X^\top=(I-M)X X^\top(I-M)=(I-M)K(I-M)
$$

第一步用了 $\tilde X^\top=X^\top(I-M)$（$M$ 对称）与 $(I-M)^2=I-M$（$M^2=M$，因为
$$
M^2=\frac{\mathbf{1}(\mathbf{1}^\top\mathbf{1})\mathbf{1}^\top}{N^2}=\frac{N\mathbf{1}\mathbf{1}^\top}{N^2}=M
$$
所以

$$
\tilde K=(I-M)K(I-M)=UD^2U^\top,\qquad Z=UD \eqno{14.65}
$$

维度核对：$U\in\mathbb{R}^{N\times N}$ 正交，$D=\mathrm{diag}(d_1,\dots,d_N)$，$\tilde K$ 的特征值是 $d_m^2$；$\tilde K$ 的 $m$ 特征值正是中心化数据协方差 $\frac1N\tilde X^\top\tilde X=\frac1N X^\top(I-M)X$ 的 $N$ 倍（见预备知识 L3 的 $X^\top X$ 与 SVD 对应）。得分 $Z=UD$：$Z^\top Z/N=D U^\top U D/N=D^2/N$，即样本方差是 $d_m^2/N$。

还要看出 $U\perp\mathbf{1}$：$U^\top\mathbf{1}=0$ 吗？$\tilde K\mathbf{1}=(I-M)K(I-M)\mathbf{1}=0$，$\tilde K$ 对称，其零特征空间正交于其它特征空间，故第 $k\ge2$ 个特征向量满足 $u_k^\top\mathbf{1}=0$，即 $u_k\perp M$，于是 $(I-M)u_k=u_k$。**这一点后面要用两次**（一是从 $\tilde K$ 回到 $K$，二是核主成分的系数公式）。

### 14.5.4.2 主成分函数与 RKHS 范数 (14.66) {#s-14-5-4-2}

把「找最大方差的投影方向」推广成「找最大方差的函数」。设 $K$ 是核，$\mathcal{H}_K$ 是它生成的再生核希尔伯特空间，$\{\phi(x)\}$ 是内积 $\langle\phi(x),\phi(x')\rangle=K(x,x')$ 的特征空间。第一个主成分函数 $g_1\in\mathcal{H}_K$ 解

$$
\max_{g\in\mathcal{H}_K}\ \mathrm{Var}_{\mathcal{T}}\,g(X)\quad\text{s.t.}\ \lVert g\rVert_{\mathcal{H}_K}=1 \eqno{14.66}
$$

$\mathrm{Var}_{\mathcal{T}}$ 是训练集 $\mathcal{T}$ 上的样本方差，$\lVert g\rVert_{\mathcal{H}_K}=1$ 这个约束同时控制函数的**大小**和**粗糙度**（越光滑的核在 RKHS 里范数越小，所以要范数大就得允许函数抖）。

> **推导** · 解是有限维的，系数 $\alpha_{jm}=u_{jm}/d_m$
>
> 第一步，$\mathrm{Var}_{\mathcal{T}}\,g(X)=\frac1N\sum_i\bigl(g(x_i)-\overline{g}\bigr)^2$，其中 $\overline{g}=N^{-1}\sum_ig(x_i)$。把 $g$ 限制成 $\mathrm{span}\{K(\cdot,x_1),\dots,K(\cdot,x_N)\}$ 的形式 $g(x)=\sum_{j}c_jK(x,x_j)$（再生性：任何 $g\in\mathcal{H}_K$ 都可以这样写但系数不唯一，取核表示即可）。由 $\overline{g}=\frac1N\sum_ig(x_i)$ 与 $g(x_i)=\sum_jc_jK(x_i,x_j)$，令 $K\mathbf{1}=\kappa$（每个分量 $\kappa_i=\sum_jK(x_i,x_j)$），则
>
> $$\mathrm{Var}_{\mathcal{T}}\,g=\frac1N\sum_i\bigl(\textstyle\sum_jc_j(K_{ij}-\kappa_j/N)\bigr)^2=\frac1N c^\top\tilde K\,\tilde K\,c=\frac1Nc^\top\tilde K^2c$$
>
> 第二步，RKHS 范数就是核矩阵诱导的内积（标准 RKHS 事实，可直接验证：取 $g=\sum_jc_jK(\cdot,x_j)$ 与 $K(\cdot,x_i)$ 作内积，得
>
> $$\langle\sum_{j=1}^Nc_jK(\cdot,x_j),K(\cdot,x_i)\rangle_{\mathcal{H}_K}=\sum_{j=1}^Nc_jK(x_i,x_j)$$
>
> 对 $c_i$ 求导即得范数公式），于是
>
> $$\lVert g\rVert^2_{\mathcal{H}_K}=\sum_{i,j}c_ic_jK(x_i,x_j)=c^\top Kc$$
>
> 所以 (14.66) 是 $\max\frac{c^\top\tilde K^2c}{Nc^\top Kc}$。取 $c_m=u_m/d_m$ 验证它是 (14.66) 的最优解：
>
> - 约束：$(I-M)u_m=u_m$ 且 $\tilde Ku_m=d_m^2u_m$ 给出 $K(I-M)Ku_m=d_m^2u_m$，即 $K^2u_m=d_m^2u_m$，故 $c_m^\top Kc_m=u_m^\top K^2u_m/d_m^2=1$。
> - 目标值：$c_m^\top\tilde K^2c_m=u_m^\top\tilde K^2u_m/d_m^2=d_m^4/d_m^2=d_m^2$，于是 $\mathrm{Var}=d_m^2/N$，这正是第一个主成分的样本方差（与 14.5.4.1 一致）。
> - 最优性：用同样的「配对不等式」，对任意可行的 $c$ 令 $C=[c_1,\dots,c_m]$，$\tilde K^2=UD^4U^\top$，则
>   $$c^\top\tilde K^2c=\sum_k d_k^4\bigl(u_k^\top c\bigr)^2\ \le\ \sum_{k\le m}d_k^4\sum_{j\le m}\bigl(u_k^\top c_j\bigr)^2\ \le\ \sum_{k\le m}d_k^4$$
>   最后一步用 $\{u_k\}$ 正交、$\sum_k u_k^\top c_j\,u_k^\top c_{j'}=\langle c_j^\top\tilde K c_{j'}\rangle$ 的压缩性（标准压缩不等式，见预备知识 L2 投影定理：$\sum_k\langle c_j,u_k\rangle^2\le\lVert c_j\rVert^2$）。等号当 $\sum_k d_k^4(u_k^\top c_j)^2\le d_1^4\lVert c_j\rVert^2$ 取到时，$\{c_j\}$ 必须落在 $\mathrm{span}\{u_1,\dots,u_m\}$ 上，结合归一化得 $c_j=u_j/d_j$。
>
> 第三步，得分公式。$y_i=g_m(x_i)=\sum_j\alpha_{jm}K(x_i,x_j)$，$\alpha_{jm}=u_{jm}/d_m$。在特征空间里这更好懂：令 $w_m=\sum_i\alpha_{mi}\phi(x_i)$，则 $\lVert w_m\rVert^2=\alpha_m^\top K\alpha_m=1$，且中心化后的样本方差 
$$
=\frac1N\sum_i\langle w_m,\tilde\phi_i\rangle^2=\frac1N\alpha_m^\top\tilde K\tilde K\alpha_m=\frac{d_m^2}N
$$

与线性 PCA 完全平行。
>
> 第二个主成分函数只需再加 $\langle g_1,g_2\rangle_{\mathcal{H}_K}=0$，即 $\alpha_1^\top K\alpha_2=0$。

### 14.5.4.3 径向核与 $I-\tilde K$ (14.67)(14.68) {#s-14-5-4-3}

用径向核时

$$
K(x,x')=\exp\Bigl(-\frac{\lVert x-x'\rVert^2}{2c}\Bigr) \eqno{14.67}
$$

（原文写 $c$ 在指数里带 $\tfrac12$，等价于尺度参数 $\sigma^2=c/2$。）这时 $K$ 的元素就是谱聚类里相似度矩阵 $S$ 的元素；$W$ 是 $K$ 的**近邻截断版**。

> **推导** · 径向核是一个无穷维内积
>
> 记 $\sigma^2=c/2$。用 $a=x-x'$ 且 $a^\top y=\lVert x\rVert^2-2x^\top x'+\lVert x'\rVert^2$，把指数函数展开成幂级数：
>
> $$\exp\Bigl(-\frac{\lVert x-x'\rVert^2}{2\sigma^2}\Bigr)=\exp\Bigl(-\frac{\lVert x\rVert^2}{2\sigma^2}\Bigr)\exp\Bigl(-\frac{\lVert x'\rVert^2}{2\sigma^2}\Bigr)\sum_{k=0}^\infty\frac{1}{k!}\Bigl(\frac{2x^\top x'}{2\sigma^2}\Bigr)^k$$
>
> 而
>
> $$
> \sum_{k}\frac{1}{k!}\bigl(\frac{2x^\top x'}{2\sigma^2}\bigr)^k=\sum_{k}\sum_{a_1,\dots,a_k}\frac{1}{k!}\prod_{m}\bigl(\frac{x_mx'_m}{\sigma^2}\bigr)^{a_m}=\sum_{a\in\mathbb{N}^p}\frac{(2x)_a(x')_a}{\sigma^{2|a|}a!}
> $$
>
> 其中 $a!=\prod_m a_m!$，$(2x)_a=\prod_m(2x_m)^{a_m}$。于是核可以写成内积，特征向量是**无穷维**的：
>
> $$\phi_k(x)=e^{-\lVert x\rVert^2/(2\sigma^2)}\cdot\frac{(2x/\sqrt{2}\,\sigma^2)^k}{\sqrt{k!}},\qquad \langle\phi(x),\phi(x')\rangle=\sum_{k=0}^\infty\phi_k(x)\phi_k(x')=K(x,x)$$
>
> 这就是「核 PCA 相当于先做非线性特征展开再在特征空间里做 PCA」这句话的精确含义（Schölkopf 等 1999）。注意 $\phi_0$ 那一维是常数特征，其余坐标是 $x$ 的各阶单项式——所以线性 PCA 只看 $\phi_1$，核 PCA 看的是全部无穷多项式，天然能表达非线性。

核主成分找的是 $\tilde K$ 的**最大**特征值对应的特征向量。等价地，先把特征值「翻转」：因为 $I-\tilde K$ 的特征值是 $1-d_m^2$，$d_m^2$ 越大它越小，所以「$\tilde K$ 的最大特征向量」$=$「下面这个矩阵的最大特征向量」

$$
I-\tilde K \eqno{14.68}
$$

即原文的意思：核 PCA 等价于用 $I-\tilde K$ 当「拉普拉斯」。**它就是谱聚类的拉普拉斯 (14.63) 的核版本**，差别只有两处：$K$ 被中心化成 $\tilde K$；(14.63) 的 $G$ 把度数放在对角线上，而 $I-\tilde K$ 的对角线是 $1-\tilde K_{ii}$（$K_{ii}=1$，$\tilde K_{ii}=1-2/N+1/N^2$ 算出来是 $1-\frac{2}{N}+\frac1{N^2}$），且完全不携带度数信息。这是 14.5.4 与 14.5.3 的全部联系。

> **坑** · 图 14.30 的教训：核 PCA 对核的尺度与形式非常敏感。径向核 $c=2$ 不能分开两组，$c=10$ 可以；而把近邻截断的 $W$ 当核用则两者都不行。可推论：谱聚类成功靠的是**近邻截断**（把远点的权重压到 0），不是靠核本身。

### 14.5.4.4 欧氏距离、马氏距离与核 {#s-14-5-4-4}

这一小节补一条本书常用的联系（下一章 14.9 的 LTSA/ISOMAP 也靠它）。

> **基础知识** · 欧氏距离与马氏距离
>
> 均值 $\mu$、协方差 $\Sigma\succ 0$ 的 $p$ 维随机向量，$\Sigma=E[(X-\mu)(X-\mu)^\top]$。马氏距离平方
>
> $$d_M^2(x,x')=(x-x')^\top\Sigma^{-1}(x-x')=(x-x')^\top\Sigma^{-1/2}\Sigma^{-1/2}(x-x')=\bigl\lVert \Sigma^{-1/2}x-\Sigma^{-1/2}x'\bigr\rVert_2^2$$
>
> 最后一步只用内积的双线性性。所以**马氏距离就是白化（whitening）之后的欧氏距离**：令 $z=\Sigma^{-1/2}x$，$E[zz^\top]=I$，即 $z$ 的各分量不相关且方差为 1，此时欧氏距离就是标准化欧氏距离。

三种常见情形：

- $\Sigma=I$：$d_M=d_2$，没有度量学习。
- $\Sigma=\sigma^2I$：$d_M=d_2/\sigma$，只差一个正因子，**簇结构在相似图上完全不变**（$s_{ii'}=\exp(-d_{ii'}^2/c)$ 里 $c$ 可以吸收 $\sigma^2$）。
- $\Sigma$ 一般：$d_M$ 会**放大**方差小（噪声大）的方向上的差异、压缩方差大的方向。这就是「白化后 PCA 为什么更合理」：在 $z$ 坐标里 $E[zz^\top]=I$，主成分方向上的方差才有可比性。

与核的联系：把 (14.67) 里的欧氏距离换成马氏距离，得到「马氏径向核」

$$
K_\Sigma(x,x')=\exp\Bigl(-\frac12(x-x')^\top\Sigma^{-1}(x-x')\Big)=\bigl\langle\phi(\Sigma^{-1/2}x),\phi(\Sigma^{-1/2}x')\bigr\rangle
$$

用 14.5.4.3 的展开式（把 $x$ 换成 $\Sigma^{-1/2}x$）立刻得到：**在马氏径向核上做核 PCA，等价于在白化后的线性数据上做普通 PCA 再非线性展开**。一般对称正定的 $\Sigma^{-1}$ 可以分解为 $\sum_k\zeta_kv_kv_k^\top$（特征分解），则

$$
K_\Sigma(x,x')=\prod_{k=1}^p\exp\Bigl(-\frac{\zeta_k(v_k^\top x-v_k^\top x')^2}{2}\Bigr)=\prod_{k=1}^p\langle\psi_k(x),\psi_k(x')\rangle
$$

每个因子是一个一维高斯核，于是特征空间是 $p$ 个一维无穷维空间的张量积。这解释了谱聚类里「$P=G^{-1}W$ 随机游走」与「马氏度量」的关系：$G^{-1}W$ 的行归一化本质上就是按度数的平方根做一次预白化（$W$ 与 $G^{-1/2}WG^{-1/2}$ 有相同的非零特征值，后者对称）。

---

## 14.5.5 稀疏主成分 {#s-14-5-5}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-5-5">原文 §14.5.5</a>

主成分的载荷向量 $v_j$ 用来解释「哪些变量起作用」，但常常所有坐标都不为零，解释困难。三种稀疏化的做法都基于 lasso 型（$L_1$）惩罚。

### 14.5.5.1 SCoTLASS (14.69) {#s-14-5-5-1}

Joliffe 等 (2003) 从「最大方差」出发，求

$$
\max_{v}\ v^\top(XX^\top)v\quad\text{s.t.}\quad \sum_{j=1}^p\lvert v_j\rvert\le t,\qquad \lVert v\rVert_2=1 \eqno{14.69}
$$

- $\lVert v\rVert_2=1$ 定尺度（否则目标随 $\lVert v\rVert$ 无界放大）。
- $L_1$ 球 $\lVert v\rVert_1\le t$ 是 $L_2$ 球的**内接多面体**（$t=1$ 时是 $2^p$ 个顶点都在 $\lVert v\rVert_2=1$ 上的交叉多面体）。在多面体上极大化一个凸二次型 $v^\top X^\top Xv$（$X^\top X\succeq 0$），最优值在**顶点**取到，而顶点就是稀疏向量：每个坐标除一个外全为 0。这就是「绝对值约束把 $v$ 逼成稀疏」的严格理由。
- 后续主成分的做法是再加正交约束 $v_k\perp v_1,\dots,v_{k-1}$。$\lVert v\rVert_1$ 球面不是凸集，所以整个问题非凸。

> **推导** · (14.69) 的 KKT 与 lasso 型的一阶条件
>
> 把 $\lVert v\rVert_1\le t$ 用一个非负乘子 $\nu$ 写成不等式约束，$\lVert v\rVert_2=1$ 用对称乘子 $\Lambda$ 写：
>
> $$\mathcal{L}(v,\nu,\Lambda)=v^\top X^\top Xv+\nu\bigl(\lVert v\rVert_1-t\bigr)-\mathrm{tr}\bigl(\Lambda(v^\top v-1)\bigr)$$
>
> $v\ne0$ 时 $v^\top X^\top Xv$ 可微，$\lVert v\rVert_1$ 不可微，逐分量写 $\lVert v\rVert_1=\sum_j\sigma_jv_j$，$\sigma_j=\mathrm{sign}(v_j)\in\{-1,1\}$，在**不变号区域**内（$v_j\ne0$）$\partial\lVert v\rVert_1/\partial v_j=\sigma_j$。于是
>
> $$2X^\top Xv+\nu\sigma-2\Lambda v=0,\qquad \nu\ge0,\qquad \nu\bigl(t-\lVert v\rVert_1\bigr)=0$$
>
> 两种情形：
>
> - **$\nu=0$**（约束不起作用，$\lVert v\rVert_1<t$）：$(X^\top X-\Lambda)v=0$，$v$ 必须是 $X^\top X$ 的特征向量，且被挑的是**最大**特征值那个。
> - **$\nu>0$**（$\lVert v\rVert_1=t$）：$v$ 在 $L_1$ 球面的顶点上，即 $\lVert v\rVert_0\le1$ 个非零坐标。写成 $v=\sum_{j\in S}\beta_je_j$（$S=\{j:\beta_j\ne0\}$），代入上式得
>   $$2X^\top X_{:,S}\beta_S+\nu\mathrm{sign}(\beta_S)=2\Lambda_S\beta_S$$
>   其中 $\Lambda_S$ 的对角元为 $X_{S,:}^\top X_{:,S}$ 的 Rayleigh 值加上 $\lVert\beta\rVert^{-2}$ 之类的尺度项——这正是 **lasso 的一阶最优性条件**：$\lVert\beta\rVert_1\le t$ 时 $X^\top Xv\propto\mathrm{sign}(v)$ 逐坐标成立（$\nu$ 就是那个比例常数）。也就是说 (14.69) 可以看成「lasso 在 $t$ 下的解与 PCA 方向的匹配」。
>
> 这解释了原书的两条评注：$t\to\infty$ 退回普通第一主成分（$\nu=0$ 情形）；$\lVert v\rVert_1=t$ 固定、$t$ 变小时解在顶点集上跳变，所以解对 $t$ **不连续**，这是非凸的代价。

### 14.5.5.2 Zou 等 (2006) 的单成分弹性网 (14.70) {#s-14-5-5-2}

改从「最小重构误差」出发（就是 14.5.1 的 $\min\lVert X-Xv\theta^\top\rVert_F^2$ 加惩罚）：

$$
\min_{\theta,v}\ \sum_{i=1}^N\bigl\lVert x_i-\theta v^\top x_i\bigr\rVert_2^2+\lambda\lVert v\rVert_2^2+\lambda_1\lVert v\rVert_1\quad\text{s.t.}\ \lVert\theta\rVert_2=1 \eqno{14.70}
$$

$x_i$ 是 $X$ 的第 $i$ 行，$\theta\in\mathbb{R}^p$，$v\in\mathbb{R}^p$，$\lVert\theta\rVert_2=1$ 仍然是为定尺度。

> **推导** · 先消去 $\theta$：为什么 $\lambda_1=0$ 时解是第一主成分
>
> 固定 $v$，记 $a=X^\top Xv$，$S_0=\mathrm{tr}(X^\top X)$。展开重构误差：
>
> $$\sum_i\lVert x_i-\theta(v^\top x_i)\rVert^2=S_0-2\theta^\top a+\theta^\top X^\top X\theta\,(v^\top X^\top Xv)$$
>
> （最后一项来自 $\sum_i(v^\top x_i)^2=\lVert Xv\rVert_2^2=v^\top X^\top Xv$。）固定 $v$ 时它对 $\theta$ 是凸二次（$X^\top X\succeq 0$），在 $\lVert\theta\rVert_2=1$ 上的最大值在 $\theta=a/\lVert a\rVert$，最小值在 $\theta=-a/\lVert a\rVert$，代入得
>
> $$\min_{\lVert\theta\rVert=1}\sum_i\lVert x_i-\theta v^\top x_i\rVert^2=S_0-2\lVert a\rVert_2=S_0-2\sqrt{v^\top X^\top Xv}$$
>
> 于是 (14.70) 里关于 $v$ 的部分等价于
>
> $$\max_v\ 2\sqrt{v^\top X^\top Xv}-\lambda\lVert v\rVert_2^2-\lambda_1\lVert v\rVert_1$$
>
> （$v$ 的尺度被 $2\sqrt{\cdot}$ 与两个惩罚项一起定了，所以 $\lVert v\rVert_2=1$ 不必再加。）当 $\lambda=\lambda_1=0$ 时，极值点满足
>
> $$\nabla\Bigl[\sqrt{v^\top X^\top Xv}\Bigr]=\frac{X^\top Xv}{\sqrt{v^\top X^\top Xv}}=\mu v,\qquad \mu=\sqrt{v^\top X^\top Xv}=\lambda_{\max}$$
>
> 也就是 $X^\top Xv=\lambda_{\max}v$，$v$ 是**最大**特征值对应的特征向量。这给出原书第一条性质（$N>p$ 时解为第一主成分方向）。当 $\lambda>0,\lambda_1=0$ 时，条件变成
>
> $$\frac{X^\top Xv}{\sqrt{v^\top X^\top Xv}}=\Bigl(\mu+\frac{\lambda\lVert v\rVert_2}{\lVert v\rVert_2^2}\Bigr)v\ \propto\ v$$
>
> 即 $v$ 仍是 $X^\top X$ 的特征向量，只是特征值被 $\lambda$ 平移（$\lVert v\rVert_2$ 由惩罚项定出 $\lVert v\rVert_2=\sqrt{\mu/(\mu+\lambda)}$），方向不变——这正是原书第二条性质。同时这解释了为什么 $p\gg N$ 时必须 $\lambda>0$：此时 $X^\top X$ 秩为 $N<p$，$\mathrm{null}(X^\top X)$ 里的非零向量都满足 $\lambda_1=0$ 时的驻点条件（因为 $\sqrt{v^\top X^\top Xv}=0$），解不唯一；$\lambda\lVert v\rVert_2^2$ 把它们排除了。
>
> 最后，$\lambda_1\lVert v\rVert_1$ 的作用：它是**收缩**（shrinkage），使解趋于稀疏。注意与 (14.69) 不同，这里的可行集是 $\ell_2$ 球加两个惩罚，问题在 $(\theta,v)$ 上联合非凸，但凸可微（$\lVert v\rVert_1$ 除外），可以用次梯度法。

### 14.5.5.3 多个成分 (14.71) 与交替法 {#s-14-5-5-3}

$$
\min_{\Theta,V}\ \sum_{i=1}^N\sum_{k=1}^K\bigl\lVert x_i-\Theta v_k^\top x_i\bigr\rVert_2^2+\lambda\sum_{k=1}^K\lVert v_k\rVert_2^2+\lambda_1\sum_{k=1}^K\lVert v_k\rVert_1 \quad\text{s.t.}\ \Theta^\top\Theta=I_K \eqno{14.71}
$$

$V\in\mathbb{R}^{p\times K}$（第 $k$ 列是 $v_k$），$\Theta\in\mathbb{R}^{p\times K}$。原书说「(14.71) 对固定另一个参数时各自凸，交替求解」；下面把两个子问题都算完。

**固定 $V$ 解 $\Theta$（Procrustes 型）**：记 $M=X^\top X$，$B=MV$，$S_0=\mathrm{tr}(M)$。
$$
\sum_i\sum_k\lVert x_i-\Theta v_k^\top x_i\rVert^2=\sum_k\bigl(\lVert \Theta v_k^\top X^\top\rVert^2-2\langle\Theta v_k^\top X^\top, Xv_k\rangle+\lVert Xv_k\rVert^2\bigr)
$$
再用 $\mathrm{tr}(\Theta^\top M\Theta)=K S_0-\sum_k v_k^\top(M-S_0I)Mv_k$ 与 $\sum_k\mathrm{tr}(\Theta^\top B)$ 整理得

$$
\sum_{i,k}\bigl(\lVert x_i-\Theta v_k^\top x_i\bigr\rVert^2\bigr)=S_0+\mathrm{tr}\bigl(\Theta^\top M\Theta\bigr)-2\mathrm{tr}(\Theta^\top B)
$$

第一项与 $\Theta$ 无关；约束 $\Theta^\top\Theta=I_K$ 下极小化后两项，等价于**极大化** $2\mathrm{tr}(\Theta^\top B)-S_0$。由 (14.57) 的 von Neumann 论证（取 $B=UDV^\top$、$\Theta=UV^\top$）：最优 $\Theta$ 是 $M=X^\top X$ 的**前 $K$ 个特征向量**组成的矩阵，与 $V$ 无关。

**固定 $\Theta$ 解 $V$（$K$ 个弹性网）**：把目标按 $v_k$ 分开，惩罚项也是可分的，于是解成 $K$ 个独立的 (14.70)（把 $\Theta v_k$ 看作原来的 $\theta$，其范数为 1）。这正是原书说的「等价于 $K$ 个弹性网问题」。

**收敛性**：每半步都在固定另一组参数时取到全局最优（或凸可微问题的驻点），故目标单调不增，收敛到局部极小。

---

## 14.6 非负矩阵分解与原型分析 {#s-14-6}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-6">原文 §14.6</a>

### 14.6.1 NMF 模型与泊松似然 (14.72)(14.73) {#s-14-6-1}

数据与成分都假设非负，这是 PCA 的一个替代品（适合图像这类非负数据）：

$$
X\approx WH \eqno{14.72}
$$

$W\in\mathbb{R}^{N\times r}$、$H\in\mathbb{R}^{r\times p}$、$r\le\max(N,p)$，且 $x_{ij},w_{ik},h_{kj}\ge0$。$W,H$ 由最大化下式得到：

$$
\sum_{i=1}^N\sum_{j=1}^pL(W,H)=\sum_{i=1}^N\sum_{j=1}^p\bigl[x_{ij}\log(WH)_{ij}-(WH)_{ij}\bigr] \eqno{14.73}
$$

> **推导** · (14.73) 为什么是对数似然
>
> 设 $x_{ij}$ 相互独立，$x_{ij}\sim\mathrm{Poisson}(\lambda_{ij})$，$\lambda_{ij}=(WH)_{ij}=\sum_kw_{ik}h_{kj}$。泊松对数似然（去掉与参数无关的常数 $-\sum\log x_{ij}!$）是
>
> $$\ell=\sum_{i,j}\bigl(x_{ij}\log\lambda_{ij}-\lambda_{ij}\bigr)$$
>
> 换元 $\lambda_{ij}=(WH)_{ij}$ 就得到 (14.73)。似然在 $\lambda_{ij}>0$ 时非负 $\iff$ $\sum\lambda_{ij}\le\sum x_{ij}$（用 $\log x\le x-1$：$\sum x_j\log\lambda_j-\sum\lambda_j\ge\sum x_j\log x_j-\sum x_j$），所以 (14.73) 的极大值 $\le\sum_{ij}x_{ij}\log\frac{x_{ij}}{\bar x}-\sum x_{ij}+\sum x_{ij}$ 一类的界在 $\lambda_{ij}=x_{ij}$（即完美拟合）处取到——说明「$WH$ 应逼近 $X$」是似然自己要求的。

### 14.6.2 乘性更新的推导 (14.74) {#s-14-6-2}

Lee 和 Seung (2001) 的交替算法：

$$
w_{ik}\ \leftarrow\ w_{ik}\frac{\sum_{j=1}^p h_{kj}\,x_{ij}/(WH)_{ij}}{\sum_{j=1}^p h_{kj}}\ ;\qquad
h_{kj}\ \leftarrow\ h_{kj}\frac{\sum_{i=1}^N w_{ik}\,x_{ij}/(WH)_{ij}}{\sum_{i=1}^N w_{ik}} \eqno{14.74}
$$

> **推导** · 辅助函数（majorization）如何造出乘性更新
>
> 乘性更新的思路：找辅助函数 $\Psi(W,H)$ 满足 (a) 在当前点与 $L$ 取值相同；(b) 固定 $H$ 时 $\Psi$ 关于各个 $w_{ik}$ **可分离**且逐坐标有闭式解，于是能像坐标下降一样一步解出整个 $W$。做到 (b) 的关键是让 $\log$ 的自变量对 $w_{ik}$ 可分离。
>
> 第一步，Jensen 不等式。凹函数 $\log$ 与权重 $\alpha_k\ge0,\ \sum_k\alpha_k=1$、$t_k>0$：$\sum_k\alpha_k\log t_k\le\log\bigl(\sum_k\alpha_kt_k\bigr)$。对固定的 $(i,j)$ 取
>
> $$\alpha_k=\frac{w_{ik}}{c_{ik}},\qquad c_{ik}=\sum_{k'}w_{ik'},\qquad t_k=\lambda^*_{ij}$$
>
> 得
>
> $$\sum_{k=1}^r w_{ik}\log\lambda^*_{ij}=c_{ik}\sum_k\frac{w_{ik}}{c_{ik}}\log\lambda^*_{ij}\ \le\ c_{ik}\log\Bigl(\sum_k\frac{w_{ik}}{c_{ik}}\lambda^*_{ij}\Bigr)$$
>
> 右端是关于 $w_{i\cdot}$ 的**可分离**函数（每个 $i$ 只含自己那一行），这就是全部技巧所在。
>
> 第二步，把对数似然写进这个界。把 $\lambda_{ij}=\sum_kw_{ik}h_{kj}$ 代入 $\sum_{i,j}x_{ij}\log\lambda_{ij}$，对每个 $(i,j)$ 用第一步的界；$\sum_{i,j}x_{ij}\lambda_{ij}$ 已经关于 $w_{ik}$ 可分（因为 $\lambda_{ij}$ 对 $w$ 是线性的），保持不动。得辅助函数
>
> $$\Psi(W,H)=\sum_{i=1}^N\sum_{j=1}^p\Bigl[x_{ij}\log\Bigl(\sum_{k=1}^r\tfrac{w_{ik}}{c_{ik}}\lambda^*_{ij}\Bigr)-\lambda_{ij}\Bigr]+\text{常数}$$
>
> 第三步，选 $\lambda^*_{ij}$ 使 $\Psi$ 在当前点等于 $L$。约定 $c_{ik}$ 取当前行和，取
>
> $$\lambda^*_{ij}=\frac{w_{ik}\lambda_{ij}}{c_{ik}}\ \Longrightarrow\ \sum_k\frac{w_{ik}}{c_{ik}}\lambda^*_{ij}=\lambda_{ij}\frac{\sum_kw_{ik}^2}{c_{ik}^2}$$
>
> 在当前点 $\sum_kw_{ik}^2=c_{ik}^2$（每个 $c_{ik}$ 只取一个非零贡献），故这个和恰为 $\lambda_{ij}$，于是 $\Psi(W,H)=L(W,H)$，性质 (a) 成立。
>
> 第四步，逐坐标极小化 $\Psi$。第三步代入后，$\Psi$ 中含 $w_{ik}$ 的项形如 $a\log w_{ik}-bw_{ik}+\mathrm{const}$（$\log\lambda^*_{ij}$ 里含一个 $\log w_{ik}$），唯一极小点在 $w^\star=a/b$：一阶条件 $a/w-b=0$，二阶导 $-a/w^2<0$，是全局最小。取
>
> $$a=\sum_{j=1}^p\frac{x_{ij}}{\lambda_{ij}},\qquad b=\sum_{j=1}^ph_{kj}$$
>
> 得
>
> $$w_{ik}\ \leftarrow\ w_{ik}\,\frac{\sum_{j=1}^p h_{kj}\,x_{ij}/\lambda_{ij}}{\sum_{j=1}^p h_{kj}}$$
>
> 这正是 (14.74) 的第一个式子。对 $H$ 完全对称（此时权重取 $\sum_iw_{ik}$，$\lambda^*_{ij}=h_{kj}\lambda_{ij}/\sum_iw_{ik}$，把 $w$ 与 $h$、$i$ 与 $j$ 对调）得
>
> $$h_{kj}\ \leftarrow\ h_{kj}\,\frac{\sum_{i=1}^N w_{ik}\,x_{ij}/\lambda_{ij}}{\sum_{i=1}^N w_{ik}}$$
>
> 即 (14.74) 的第二个式子。两个式子都是「当前值 $\times$ 增益系数」的形式：$W$ 的增益是 $W,H$ 的**列和归一化**下 $x_{ij}/\lambda_{ij}$ 的加权平均，$H$ 的增益是 $W$ 的**行和归一化**下同样的加权平均。
>
> 第五步，收敛性。每半步 $\Psi$ 下降，而 Jensen 给出 $L\le\Psi$，且在每步的当前点 $\Psi=L$，所以每半步
>
> $$L(W_{\text{新}})\le\Psi(W_{\text{新}})\le\Psi(W_{\text{旧}})=L(W_{\text{旧}})$$
>
> 即 $L$ 单调不增，收敛到 $(W,H)$ 空间的局部极大。这是**单调迭代（majorization-minimization）**的标准用法；原书说它同时与 log-linear 模型的迭代比例缩放（IPF）有关，结构完全一样：那里也是对偶函数加乘性更新。
>
> **坑** · 迭代保持 $W,H$ 的**列和/行和为 1** 的不变性（因为更新只乘正数），所以要靠初始化进入正确的「单纯形尺度」；同时 $X=WH$ 精确成立时分解**不唯一**：图 14.34 里 $h_1,h_2$ 可以取在数据与坐标轴之间的「开空间」里的任意非负向量，都给出精确重构。所以结果依赖初值。

### 14.6.3 原型分析 (14.75)(14.76)(14.77) {#s-14-6-3}

Cutler 和 Breiman (1994)：用**本身是数据点的凸组合**的原型来近似数据点。模型形式与 NMF 相同：

$$
X\approx WH \eqno{14.75}
$$

$W\in\mathbb{R}^{N\times r}$、$H\in\mathbb{R}^{r\times p}$，但现在要求 $w_{ik}\ge0$ 且 $\sum_{k=1}^rw_{ik}=1$（对每个 $i$）：即 $X$ 的每个数据点是 $r$ 个原型（$H$ 的行）的**凸组合**。同时要求原型本身也是凸组合：

$$
H=BX \eqno{14.76}
$$

$B\in\mathbb{R}^{r\times N}$，$b_{ki}\ge0$、$\sum_{i=1}^Nb_{ki}=1$。于是原型落在数据的凸包上——这就是「原型/archetype」两个词的含义。目标函数

$$
J(W,B)=\lVert X-WHX\rVert^2=\lVert X-WBX\rVert^2 \eqno{14.77}
$$

在 $W,B$ 上交替极小化。

> **推导** · 固定 $B$ 时 $W$ 的 KKT：每行至多 $r$ 个非零
>
> 记 $A$ 为 $\mathbb{R}^{N\times p}$ 矩阵，其第 $k$ 行为 $\tilde b_k^\top$（即原型的转置），$A=BX$ 转置。则 $(WBX)_{i:}=w_i^\top A$，目标按行可分：
>
> $$\min_{w_i\ \in\ \Delta_r^{1}}\ J_i(w_i)=\lVert x_i-w_i^\top A\rVert^2=\lVert x_i\rVert^2-2w_i^\top c_i+w_i^\top Qw_i$$
>
> 其中 $c_i=A x_i\in\mathbb{R}^r$，$Q=A^\top A$（$r\times r$，$\mathrm{rank}(Q)\le\mathrm{rank}(A)\le\min(N,p)$）。Lagrange 函数
>
> $$
> \mathcal{L}(w,\mu,\nu)=\lVert x_i-w^\top A\rVert^2-\mu(w^\top\mathbf{1}-1)-\nu^\top w
> $$
>
> 其中 $\mu\in\mathbb{R}$ 自由、$\nu\ge0$。KKT：
>
> $$2Qw-c_i=\mu\mathbf{1}+\nu,\qquad \nu\ge0,\qquad w\ge0,\qquad \nu_kw_{ik}=0,\qquad \sum_kw_{ik}=1$$
>
> 第三式（**互补松弛**）是关键：$\nu_k>0\Rightarrow w_{ik}=0$。设活跃集 $S=\{k:w_{ik}>0\}$，$\lvert S\rvert=r'$，则在这些坐标上 $\nu_{ik}=0$，于是 $\nu$ 消失、$2Q_{SS}w_S=c_{i,S}+\mu\mathbf{1}_S$：
>
> $$w_S=\frac12Q_{SS}^{-1}c_{i,S}+\frac{\mu}{2}Q_{SS}^{-1}\mathbf{1}_S,\qquad \sum_{k\in S}w_k=1\ \Longrightarrow\ \mu=\frac{2\bigl(1-\tfrac12\mathbf{1}_S^\top Q_{SS}^{-1}c_{i,S}\bigr)}{\mathbf{1}_S^\top Q_{SS}^{-1}\mathbf{1}_S}$$
>
> 即「给定 $S$，权重是 $c_{i,S}$ 的一个仿射函数，只剩标量 $\mu$ 由归一化定出」。可行性与互补条件变成三个不等式：$\mu/2+Q_{SS}^{-1}c_{i,S}\ge0$、$S^c$ 上 $\nu_S^c=2Q_{S^cc}c_{i,S^c}-\mu\mathbf{1}-\mathbf{1}(2Q_{S^Sc}c_{i,S})\ge0$、以及 $\sum_{k\in S^c}w_k=0\iff\mu$ 与 $S^c$ 的互补条件同时成立。这些条件正是 QP 的 **active-set（主变量）方法**逐步加/删变量时可验证的一组不等式（Goldfarb–Idnani 算法），所以原型分析的一个 $W$-步可以在 $\mathcal{O}(r^3)$ 里解出。
>
> 所以解是「把 $x_i$ 投影到 $S$ 个原型的凸包上」，而 $S$ 由 KKT 的互补条件挑选；由于 $r'$ 个权重之和为 1 且落在 $\mathrm{row}(A)$ 的仿射包上，而 $\mathrm{rank}(Q)\le\min(N,p)$，有效原型数不超过 $\min(N,p)$——这解释了「即使 $r>p$ 近似也不完美」。这同时解释了原书「每个子问题都是凸优化」，以及「$K$-means 是它的特例：$w_i$ 取单位行向量」。
>
> 固定 $W$ 时对 $B$ 的子问题**完全对称**（原型是数据点的凸组合），KKT 形式一样。
>
> **坑** · (14.77) 的 $W,B$ 联合非凸（两个单纯形的乘积），所以只收敛到局部极小。而且 (14.76) 意味着即使 $r>p$ 近似也不完美——原型受限于数据凸包。

> **结果** · NMF 与原型分析的对照
>
> | | NMF (14.72) | 原型分析 (14.75) |
> |---|---|---|
> | 约束 | 非负 | 非负 + 行（列）和为 1 |
> | 原型位置 | 任意非负向量 | 必须在数据凸包上 |
> | 关心的输出 | $W$ 的列（非负基） | $H$ 的行（原型数据点） |
> | 秩 | 要求 $r\le p$ | $r\le N$，可 $r>p$ |
> | 与 $K$-means | 接近 VQ | $w_i$ 取单位向量即 $K$-means |

---

## 14.7.1 潜变量模型与因子分析 {#s-14-7-1}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-7-1">原文 §14.7.1</a>

多变量数据常被看成若干**不可直接测量的源**的间接测量：问卷答题反映智力与心理能力、EEG 传感器测头皮不同位置记录脑活动、股价变化反映市场信心等外部驱动力。因子分析找这些潜源，ICA 是它更强的竞争者。

SVD (14.54) 本身就有一个潜变量表示。写 $S=\sqrt{N}U$，$A^\top=DV^\top/\sqrt{N}$，则 $X=SA^\top$，即每个观测都是若干个 $S$ 列的线性组合。因为 $U$ 列正交且 $X$ 各列中心化（故 $X^\top\mathbf{1}=0$，$U^\top\mathbf{1}=0$），所以 $S$ 的列中心、互不相关、方差为 1。写成潜变量模型

$$
\begin{cases}
X_1=a_{11}S_1+a_{12}S_2+\cdots+a_{1p}S_p\\
X_2=a_{21}S_1+a_{22}S_2+\cdots+a_{2p}S_p\\
\ \vdots\\
X_p=a_{p1}S_1+a_{p2}S_2+\cdots+a_{pp}S_p
\end{cases}
\qquad\text{即}\qquad X=AS \eqno{14.78}
$$

但这不令人满意：给定**任何**正交 $p\times p$ 阵 $R$，都可以写

$$
X=AS=(AR^\top)(RS)=A^\ast S^\ast \eqno{14.79}
$$

$$
\mathrm{Cov}(S^\ast)=R\,\mathrm{Cov}(S)\,R^\top=RI R^\top=I
$$

（第二步用 $R^{-1}=R^\top$，以及 $\mathrm{Cov}(RS)=R\,\mathrm{Cov}(S)R^\top$ 的线性变换公式。）所以分解**不唯一**，无法把某个特定的潜变量认定为真正的源。SVD 的优点只是：任一截断到秩 $q<p$ 的分解都以最优方式逼近 $X$。

经典因子分析模型缓解了这个问题。取 $q<p$：

$$
\begin{cases}
X_1=a_{11}S_1+\cdots+a_{1q}S_q+\varepsilon_1\\
X_2=a_{21}S_1+\cdots+a_{2q}S_q+\varepsilon_2\\
\ \vdots\\
X_p=a_{p1}S_1+\cdots+a_{pq}S_q+\varepsilon_p
\end{cases}
\qquad\text{即}\qquad X=AS+\varepsilon \eqno{14.80}
$$

$S$ 是 $q$ 个潜变量（因子），$A$ 是 $p\times q$ 载荷矩阵，$\varepsilon_j$ 是互不相关的零均值扰动。参数都落在协方差阵里：

$$
\Sigma=AA^\top+D_\varepsilon \eqno{14.81}
$$

推导：$X=AS+\varepsilon$，$E[\varepsilon\mid S]=0$ 且 $\mathrm{Cov}(\varepsilon)=D_\varepsilon$ 对角，于是（线性性 + 交叉项消失）

$$
\mathrm{Cov}(X)=A\,\mathrm{Cov}(S)A^\top+A\,\mathrm{Cov}(S,\varepsilon)+A^\top\,\mathrm{Cov}(\varepsilon,S)+D_\varepsilon=AA^\top+D_\varepsilon
$$

其中
$$
\mathrm{Cov}(S,\varepsilon)=E[(S-ES)\mathrm{E}\varepsilon^\top]=E[(S-ES)]E[\varepsilon^\top]=0
$$
（因为扰动零均值、条件期望为零）。

> **推导** · 三个可核查的事实
>
> **（1）$S_\ell$ 高斯且互不相关 $\Rightarrow$ 独立。** 对联合高斯，$\mathrm{Cov}=0$ 蕴含独立（因为 $\mathrm{Cov}=0$ 使相关系数矩阵为单位阵，联合密度 $\propto\exp(-\frac12x^\top\Sigma^{-1}x)$ 分解成边缘密度的乘积）。所以一套教育测验分数可以被认为由「智力」「驱动力」等独立因子驱动。$A$ 的列是**因子载荷**，用来命名与解释因子。
>
> **（2）可辨识性问题依然存在。** 因为对任何 $q\times q$ 正交 $R$，$(A,AR^\top)$ 在 (14.81) 中与 $(AR^\top,R\cdot)$ 等价：
>
> $$\bigl(AR^\top\bigr)\bigl(AR^\top\bigr)^\top=ARR^\top A^\top R^\top R=AA^\top$$
>
> （用了 $R^\top R=I$。）这让分析者可以在旋转后的因子中挑「更好解释」的版本，是因子分析主观性的根源，也是它在当代统计学里不受欢迎的原因之一。
>
> **（3）SVD 决定因子子空间。** 若取 $\mathrm{Var}(\varepsilon_j)=\sigma^2$ 全相同，则 $\Sigma-\sigma^2I=AA^\top$，$\mathrm{rank}=q$。设 $\Sigma$ 的特征值 $\lambda_1\ge\dots\ge\lambda_p$，$A$ 的列空间被 $\Sigma$ 的特征向量张成（因为 $\mathrm{range}(AA^\top)$ 是 $\Sigma$ 的不变子空间，且 $\Sigma$ 对称、特征值非退化——重根时取其张成的整个不变子空间）。所以 $\Sigma-\sigma^2I$ 的列空间就是 $\mathrm{col}(A)$，用 SVD 取前 $q$ 个特征向量即可估计 $A$ 的子空间（再在子空间里解剩余的 $\sigma^2$ 与载荷比）。
>
> **（4）因子分析建模的是相关结构，不是协方差结构。** 因为每个 $X_j$ 有自己的 $\varepsilon_j$，只要有 $\mathrm{Var}(\varepsilon_j)>0$，$\Sigma$ 的对角就「污染」了。用 $D_\varepsilon^{-1/2}$ 前后乘 (14.81)（习题 14.14）得
>
> $$D_\varepsilon^{-1/2}\Sigma D_\varepsilon^{-1/2}=\bigl(D_\varepsilon^{-1/2}A\bigr)\bigl(D_\varepsilon^{-1/2}A\bigr)^\top+I$$
>
> 右边第二项是单位阵、与 $A$ 无关，所以**相关阵的分解与协方差阵的分解只差一次逐变量缩放**。这是因子分析与 PCA 的本质区别（见 14.7.2 末尾的对照）。

---

## 14.7.2 独立成分分析 {#s-14-7-2}

<a class="src" href="../esl/ch14-unsupervised-learning.html#s-14-7-2">原文 §14.7.2</a>

ICA 模型与 (14.78) 形式完全一样，只是 $S_\ell$ 被假设**统计独立**而非互不相关。直觉上：互不相关只固定了二阶交叉矩（协方差），统计独立固定**全部**交叉矩。多出来的矩条件让 $A$ 可以被唯一确定。由于多元高斯的分布由二阶矩决定，高斯是唯一的例外——高斯独立成分也只能确定到一个旋转，与前面一样。所以只要假设 $S_\ell$ **独立且非高斯**，(14.78) 与 (14.80) 的可辨识性问题就消除了。

无损一般性可以假设 $X$ 已被**白化**成 $\mathrm{Cov}(X)=I$（用上面的 SVD 即可做到）。由于 $\mathrm{Cov}(S)=I$，有 $I=AA^\top$，即 $A$ 正交。于是解 ICA 变成：找正交 $A$ 使 $S=A^\top X$ 的各分量独立（非高斯）。

### 14.7.2.1 熵、互信息与 (14.82)–(14.85) {#s-14-7-2-1}

微分熵：若随机变量 $Y$ 的密度是 $g(y)$，

$$
H(Y)=-\int g(y)\log g(y)\,dy \eqno{14.82}
$$

信息论的一条著名结论：**在方差相同的随机变量里，高斯的熵最大**。互信息度量 $Y$ 各分量之间的依赖：

$$
I(Y)=\sum_{j=1}^pI(Y_j)=H(Y_1)+\cdots+H(Y_p)-H(Y)=\sum_{j=1}^pH(Y_j)-H(Y) \eqno{14.83}
$$

> **推导** · (14.83) 是 KL 散度的定义式
>
> 设 $g$ 是 $Y$ 的联合密度，$g_j$ 是第 $j$ 个边缘密度，则 $g_j(y_j)=\int g(y)\,dy_{-j}$。展开定义
>
> $$\mathrm{KL}(g\,\|\,\textstyle\prod_jg_j)=\int g(y)\log\frac{g(y)}{\prod_{j=1}^pg_j(y_j)}dy=-\int g\log g\,dy+\sum_{j=1}^p\int g(y)\log g_j(y_j)dy$$
>
> 而由 $y_j$ 与 $y_{-j}$ 独立，
>
> $$
> \int g(y)\log g_j(y_j)dy=E\bigl[\log g_j(Y_j)\bigr]=-\int g_j(y_j)\log g_j(y_j)dy_j=-H(Y_j)
> $$
>
> 代回即得 (14.83)。$I(Y)\ge0$ 是 KL 散度非负性（$\mathrm{KL}(f\|g)=-E_f\log\frac gf$ 关于 $f$ 凸，在 $f=g$ 处取最小 0），$I(Y)=0\iff g=\prod_jg_j\iff$ 独立。所以 (14.83) 是「独立性」的连续版本。

若 $\mathrm{Cov}(X)=I$、$Y=A^\top X$、$A$ 正交，容易算出

$$
\sum_{j=1}^pI(Y_j)=\sum_{j=1}^pH(Y_j)-H(X)-\log\lvert\det A\rvert\sum_{j=1}^p1=\sum_{j=1}^pH(Y_j)-H(X) \eqno{14.84}
$$

$$
\sum_{j=1}^pI(Y_j)=H(Y_j)-H(X) \eqno{14.85}
$$

> **推导** · $\log\lvert\det A\rvert$ 从哪来：熵的换元公式
>
> **熵的换元公式**：若 $Y=A^\top X$（$A$ 可逆），$X$ 的密度 $f$，则
>
> $$H(Y)=H(X)+\log\lvert\det A\rvert$$
>
> 证明：密度变换公式给出 $g(y)=f(Ay)/\lvert\det(A^\top)\rvert=f(Ay)/\lvert\det A\rvert$（$x=Ay$，$\frac{\partial x}{\partial y}=A$）。代入 (14.82)：
>
> $$H(Y)=-\int g(y)\log\frac{f(Ay)}{\lvert\det A\rvert}dy=-\int f(x)\log\frac{f(x)}{\lvert\det A\rvert}dx=H(X)+\log\lvert\det A\rvert\int f(x)dx=H(X)+\log\lvert\det A\rvert$$
>
> 其中用了换元 $x=Ay$ 把积分变成对 $f$ 的积分（$\int g(y)dy=\int f(x)dx=1$）。对 $p$ 维向量同理：$\int f(x)\log\lvert\det A\rvert\,dx=\log\lvert\det A\rvert$。
>
> 现在代入：$A$ 正交 $\Rightarrow AA^\top=I$，$|\det A|^2=\det A\det A^\top=\det I=1$，故 $|\det A|=1$，$\log|\det A|=0$，(14.84) 立刻简化成 (14.85)。
>
> **结果**：在正交约束下极小化 $I(Y)=I(A^\top X)$ 等价于极小化 $\sum_jH(Y_j)$，也等价于**最大化各分量的负熵**（高斯分量使熵最大）。
>
> **坑** · 这个换元公式也解释了为什么熵本身不是不变量：只有正交变换（行列式 $\pm1$）才保持熵。所以「$A$ 必须正交」这个白化后的假设是整段推导的关键前提；若 $A$ 奇异（成分数少于 $\min(N,p)$，即 (14.80) 型模型），公式要用伪行列式 $\sum\log|\lambda_j|$ 而不是 $\log|\det A|$，这正是 factor analysis 版 ICA 的做法。

### 14.7.2.2 负熵与对比函数 (14.86)(14.87) {#s-14-7-2-2}

为了方便，Hyvärinen 和 Oja (2000) 不用 $H(Y_j)$ 而用**负熵**

$$
J(Y_j)=H(Z_j)-H(Y_j) \eqno{14.86}
$$

其中 $Z_j$ 是与 $Y_j$ **方差相同**的高斯随机变量。$J\ge0$，度量 $Y_j$ 偏离高斯的程度。书中 ICA 解用的是近似

$$
J(Y_j)\approx\bigl[E\,G(Y_j)-E\,G(Z_j)\bigr]^2,\qquad G(u)=\frac1a\log\cosh(au),\quad 1\le a\le2 \eqno{14.87}
$$

把期望换成样本平均即可计算。$Z_j\sim N(0,\sigma^2)$ 时 $E\,G(Z_j)=E\,G(Y_j)$ 当 $Y_j$ 也是高斯，所以 $\bigl[E G(Y_j)-E G(Z_j)\bigr]^2=0$，符合定义。

> **推导** · (14.86) 等于 KL 散度，(14.87) 为什么合法
>
> **第一步，$J$ 就是 KL 散度。** 取 $\varphi_\sigma(y)=(2\pi\sigma^2)^{-1/2}e^{-y^2/(2\sigma^2)}$，则
>
> $$J(Y_j)=H(Z_j)-H(Y_j)=-E\log\varphi_\sigma(Y_j)+E\log g(Y_j)=\int g(y)\log\frac{g(y)}{\varphi_\sigma(y)}dy=\mathrm{KL}(g\|\varphi_\sigma)$$
>
> 故 $J\ge0$，等号当且仅当 $g=\varphi_\sigma$。
>
> **第二步，高斯最大熵的证明（用 $\log$ 的切线不等式，可逐步复核）。** 设 $E[Y]=0,\ \mathrm{Var}(Y)=\sigma^2$。取任意一点 $y_0$ 作为切点（为简化取 $y_0=0$，$\varphi'_\sigma(0)=0$）：
>
> $$-\log g(y)\ \le\ -\log\varphi_\sigma(y_0)-\frac{\varphi'_\sigma(y_0)}{\varphi_\sigma(y_0)}(y-y_0)+\frac12\frac{\varphi''_\sigma(y_0)}{\varphi_\sigma(y_0)}(y-y_0)^2\ \text{（由 }-\log\ \text{凸）}$$
>
> 在 $y_0=0$ 处有 $\varphi'_\sigma(0)=0$，$\frac{\varphi''_\sigma(0)}{\varphi_\sigma(0)}=-1/\sigma^2$，$-\log\varphi_\sigma(0)=\frac12\log(2\pi\sigma^2)$，故对每个 $y$
>
> $$-\log g(y)\le\frac12\log(2\pi\sigma^2)+\frac{y^2}{2\sigma^2}$$
>
> 两边乘 $g(y)$ 积分：
>
> $$
> H(Y)\le\frac12\log(2\pi\sigma^2)+\frac{1}{2\sigma^2}E[Y^2]=\frac12\log(2\pi\sigma^2)+\frac12=\frac12\log(2\pi e\sigma^2)
> $$
>
> 等号要求 $-\log g$ 在 0 处取到切线，即 $g=\varphi_\sigma$。**所以 $J(Y_j)\ge0$ 且只有高斯使它为 0**；用单位方差时 $H(Z_j)=\frac12\log(2\pi e)$。
>
> **第三步，(14.87) 的合理性。** $G'(u)=\tanh(au)$，$G''(u)=a^2\mathrm{sech}^2(au)>0$，所以 $G$ 严格凸、**在 $u=0$ 取唯一最小**。对任何单位方差随机变量 $Y$，$E[G(Y)]\ge G(E[Y)]=G(0)$（Jensen），等号当且仅当 $Y$ 退化到常数；而在方差相同的约束下，高斯使 $E[G(Y)]$ **最小**（这是负熵近似理论里的结论，可由 $G$ 的偶函数性与高斯最大熵定理一起得到：若把 $G$ 的最小值挪到 $E[G(Z)]$，则 $\bigl(E[G(Y)]-E[G(Z)]\bigr)$ 就是「比高斯更尖/更平」的程度的代理）。因此 $\bigl(E[G(Y)]-E[G(Z)]\bigr)^2$ 与 $J(Y)$ 同号同量级，可以代替 $J$ 做优化，且**可微、只依赖一阶矩的样本平均**，故能用来跑不动点迭代（下一小节）。经典但更不稳健的替代是基于四阶矩（峰度）的指标：$\bigl(E[Y^4]-3\bigr)^2$，因为高斯的四阶矩是 3、其它分布偏离之。

### 14.7.2.3 快速不动点算法（由 (14.87) 推出） {#s-14-7-2-3}

这一小节把「方向 $w$ 的快速不动点更新」完整推出来，它是 ICA 与**探索性投影追踪**（exploratory projection pursuit，见 14.7.3）共同的引擎。原书把它印在 §14.7.4 的 (14.96)，不在本分片的编号范围内，这里作为 (14.87) 的直接推论给出。

要解决的问题：白化数据（$E[XX^\top]=I$）下，沿单位方向 $w$ 投影 $y=w^\top x$，用 (14.87) 的近似极大化 $J(y)$：

$$
\max_{\lVert w\rVert_2=1}\ \bigl(E\,G(w^\top X)-c_G\bigr)^2,\qquad c_G:=E\,G(Z),\ Z\sim N(0,1)
$$

因为 $E[G(w^\top X)]\ge$ 高斯值，等价于极大化 $E[G(w^\top X)]$。注意当 $\lVert w\rVert=1$ 且 $X$ 已白化时 $\mathrm{Var}(w^\top X)=w^\top E[XX^\top]w=1$ 自动成立，不需要再约束方差。

> **推导** · 五步得到 $w^{+}\propto E[Xg'(w^\top X)]-E[XX^\top g''(w^\top X)]w$
>
> **第一步，Lagrange 函数。** 等价地写目标 $F(w)=E[G(w^\top X)]$，约束 $w^\top w=1$，$\Lambda$ 为对称乘子：
>
> $$\mathcal{L}(w,\Lambda)=E[G(w^\top X)]-\frac12\Lambda(w^\top w-1)$$
>
> **第二步，一阶条件。** 对 $w$ 求梯度（预备知识 L4 的链式法则：$\nabla_w(w^\top X)=X$，且 $\frac{\partial}{\partial w}G(w^\top x)=g(w^\top x)x$，$g=G'$）：
>
> $$E\bigl[g(w^\top X)X\bigr]=\Lambda w$$
>
> 两边左乘 $w^\top$ 并用 $E[w^\top X\,g(w^\top X)]$ 记作 $\kappa$：$E[w^\top Xg(w^\top X)]=\mathrm{tr}(\Lambda)$。所以 $\Lambda=\kappa I$，**一阶条件等价于**
>
> $$E\bigl[Xg(w^\top X)\bigr]=\kappa\,w$$
>
> （这个结论很重要：它说明不必显式处理正交约束。）
>
> **第三步，在 $w$ 附近线性化。** 令 $\Delta w$ 小，把 $E[Xg((w+\Delta w)^\top X)]$ 在 $w$ 处作一阶展开（用 $g'=g'$ 记 $G''$）：
>
> $$E[Xg((w+\Delta w)^\top X)]\approx E[Xg(w^\top X)]+E\bigl[Xg'(w^\top X)(\Delta w)^\top X\bigr]$$
>
> 把一阶条件右边的 $\kappa(w+\Delta w)$ 也在 $w$ 处展开（$\kappa=\kappa(w)$ 依赖 $w$，写 $\Delta\kappa=\partial\kappa$）：
>
> $$E[Xg(w^\top X)]+E[Xg'(w^\top X)(\Delta w)^\top X]=\kappa w+\kappa\Delta w+\Delta\kappa\,w$$
>
> 用第二步消掉左边第一项，得
>
> $$E\bigl[Xg'(w^\top X)(\Delta w)^\top X\bigr]=\kappa\Delta w+\Delta\kappa\,w$$
>
> **第四步，左乘 $w^\top$ 消掉未知量。** $\Delta\kappa$ 是待定量，其它都已知：
>
> $$\Delta w^\top E\bigl[XX^\top g'(w^\top X)\bigr]=\kappa w^\top\Delta w+\Delta\kappa\ \Longrightarrow\ \Delta\kappa=\Delta w^\top E\bigl[XX^\top g'(w^\top X)\bigr]-\kappa\lVert\Delta w\rVert^2$$
>
> 代回第三步：
>
> $$E\bigl[Xg'(w^\top X)(\Delta w)^\top X\bigr]=\Delta w^\top E\bigl[XX^\top g'(w^\top X)\bigr]w$$
>
> 这一步把「增量方向」显式解出来：设
>
> $$
> M:=E\bigl[XX^\top g'(w^\top X)\bigr]-(\kappa-\lVert\Delta w\rVert^2/\lVert\Delta w\rVert)E[XX^\top]
> $$
>
> 近似为 $E[XX^\top g'(w^\top X)]-\kappa E[XX^\top]$，则
>
> $$\Delta w\ \parallel\ E\bigl[XX^\top g'(w^\top X)\bigr]-\kappa\,E[XX^\top]$$
>
> **第五步，把方向写成不动点。** 上一步的增量与「一阶条件残差」的方向一致（这是 Newton 步的性质：增量 $\parallel\nabla$ 的线性化残差），故不动点迭代取
>
> $$w^{+}\ \propto\ E\bigl[Xg(w^\top X)\bigr]-E\bigl[XX^\top g'(w^\top X)\bigr]w \ \triangleq\ F(w)$$
>
> 每步用 $\lVert w^{+}\rVert=1$ 归一化。白化数据时 $E[XX^\top]=I$，这就是教科书里的形式；一般数据时用 $E[XX^\top]$ 代替 $I$（等价于先白化）。经验上把两点的 $X$ 换成样本均值 $\frac1N\sum_ix_i$。
>
> **对 $G(u)=\log\cosh(\alpha u)$ 的化简（FastICA 的实用形式）。** $g(u)=G'(u)=\alpha\tanh(\alpha u)$，$g'(u)=G''(u)=\alpha^2\bigl(1-\tanh^2(\alpha u)\bigr)$。关键一步用 Stein 引理：对白化数据，
>
> $$E\bigl[Xg'(w^\top X)\bigr]_{ij}=E\bigl[X_iX_jg'(w^\top X)\bigr]=w_j\,E\bigl[X_i g'(w^\top X)\bigr]=w_iw_j\,E\bigl[g'(w^\top X)\bigr]$$
>
> 第一个等号是 $X_i$ 与 $X_j$ 不相关（白化）所以 $X_iX_j$ 可拆；第二个等号是 Stein 恒等式 $E[X_jf(w^\top X)]=E[\partial_j f(w^\top X)]=w_jE[f(w^\top X)]$（对光滑 $f$，由分部积分 $\int x_jf\,dx=0$ 边界条件下成立，预备知识 P 组的 Stein 引理）；第三个等号再用一次 Stein。于是
>
> $$E\bigl[XX^\top g'(w^\top X)\bigr]=E[g'(w^\top X)]\,\mathrm{diag}(w_k^2)\ \text{（逐元素）}\quad\Rightarrow\quad w^{+}\ \propto\ E[X\tanh(\alpha w^\top X)]-\alpha\Bigl(1-E[\tanh^2(\alpha w^\top X)]\Bigr)w$$
>
> 原书 (14.96) 写的是 $a_j\leftarrow E[\hat g'_j(a_j^\top X)X]-E[\hat g''_j(a_j^\top X)]a_j$，正是上式在「$E[XX^\top]=I$、$w$ 单位长」下的写法（第二项里的 $E[\hat g_j'']$ 就是标量版的对角近似）。
>
> **多成分与去相关。** 对每个 $j$ 独立跑上面的迭代得到 $a_j$，再用 Gram–Schmidt 对 $\{a_j\}$ 正交化（等价于每步后做 $\tfrac12(a_j-\sum_{j'<j}(a_j^\top a_{j'})a_{j'})$ 归一化），否则 $A$ 不会正交、$\log|\det A|\ne0$、(14.85) 失效。也可以把更新写成矩阵形式 $A\leftarrow\frac{1}{N}\sum_ig_j'(A^\top X)X^\top-A\,\mathrm{diag}(\dots)$，然后做对称去白化。

> **结果** · ICA 算法（FastICA）
>
> 1. 白化：$X\to XW$，$W=(X^\top X)^{-1/2}$。
> 2. 随机初始化正交 $A$。
> 3. 重复：对每个 $j$，$a_j\leftarrow E[\tanh(a_j^\top X)X]-E[1-\tanh^2(a_j^\top X)]a_j$，再正交化。
> 4. 输出 $S=A^\top X$。
>
> **坑** · 收敛到的是**不动点**（临界点），不保证是全局最优；而且如果两个独立成分的分布恰好「同一个方向上非高斯性一样强」，算法可能收敛到它们的混合。实践中要对初值跑多次、用对比函数值 $C_j(a_j)=\frac1N\sum_i\hat g_j(a_j^\top x_i)$ 选最好（这正是原书 (14.95) 的 $C(A)=\sum_jC_j(a_j)$）。

### 14.7.2.4 ICA 与 PCA、LDA、因子分析的区别 {#s-14-7-2-4}

| | PCA | 因子分析 | ICA | 线性判别（LDA） |
|---|---|---|---|---|
| 模型 | $X=AS$，$\mathrm{Cov}(S)=I$ | $X=AS+\varepsilon$，$\mathrm{Cov}(\varepsilon)=D_\varepsilon$ | $X=AS$，$S$ 独立非高斯 | 有 $Y$，$X\mid Y$ 的类内协方差 $\Sigma_W$ |
| 优化什么 | $\max\mathrm{Var}(a^\top X)$ | $\max\log L(\Sigma=AA^\top+D_\varepsilon)$ | $\max\sum_jJ(a_j^\top X)$ | $\max\mathrm{Tr}(\Sigma_W^{-1}\Sigma_B)$ |
| 用了哪些矩 | 二阶 | 二阶 | 全部（通过熵/负熵） | 二阶（且用到 $Y$） |
| 可辨识性 | 只到旋转（但取前 $K$ 个就有意义） | 只到旋转 | 只要非高斯就唯一 | 需要 $\Sigma_W$ 非奇异 |
| 分量正交 | 是（白化） | 否 | 是（白化后） | 否 |

三点具体说明：

- **PCA 与 ICA 的差别只有「矩的阶数」。** PCA 找的是**方差最大**的方向，所以输出看起来像高斯（手写数字例子里 5 个 PCA 分量的联合分布都像高斯）；ICA 找的是**最不接近高斯**的方向，所以分量长尾（图 14.39、14.40：ICA 第 5 分量专门挑出「长拖尾的三」）。两者都先做白化（图 14.37 的上/中面板），区别只在白化之后如何旋转。
- **ICA 与 LDA 的差别是「有没有 $Y$」。** LDA 的目标是 Fisher 判别比 $\mathrm{Tr}(\Sigma_W^{-1}\Sigma_B)$，其中 $\Sigma_B$ 来自类均值之间的散布，需要标签；ICA 完全无监督，其「非高斯性」相当于把「每个方向上的边缘分布尽量稀疏」当作目标。
- **ICA 就是一种因子旋转。** 从这个角度看，ICA 与心理学里的 varimax、quartimax 是同一类东西：它们都从因子分析的解出发，找旋转。只是 varimax 最大化载荷的「简单性」（四阶矩的某种组合），ICA 最大化负熵（严格更大范围的准则）。因此 ICA 对初值敏感、可能有多个不动点，与一般旋转方法同病。

### 14.7.2.5 直接估计联合密度 (14.88)(14.89) {#s-14-7-2-5}

独立成分按定义有乘积型联合密度

$$
f_S(s)=\prod_{j=1}^pf_j(s_j) \eqno{14.88}
$$

（这就是「独立」的定义：联合密度等于边缘密度之积。）下面给出直接估计这个密度的方法（Hastie 和 Tibshirani 2003，R 包 `ProDenICA`）。既然 (14.90)–(14.94) 属于本分片编号范围之外，这里只推 (14.89) 的构造与它的约束。

$$
f_j(s_j)=\varphi(s_j)e^{g_j(s_j)} \eqno{14.89}
$$

称为**倾斜高斯**（tilted Gaussian）：$\varphi$ 是标准高斯密度，$g_j$ 满足密度归一化条件。具体地，把 (14.89) 代入归一化条件 $\int f_j=1$：

$$
\int\varphi(t)e^{g_j(t)}dt=1\ \Longleftrightarrow\ E_{\varphi}[e^{g_j(Z)}]=1
$$

代入后联合密度是 $\prod_j\varphi(s_j)e^{g_j(s_j)}$。用泰勒展开 $e^{g}\approx1+g+\frac12g^2$ 可以看出「倾斜」的含义：取 $g$ 的一次项 $-a s$（$a$ 非零）就得到 $\varphi(t-a)$，即**平移**的高斯——但这会把均值弄偏；所以真正的解在两个约束下才有意义：

1. **归一化**：$E_\varphi[e^{g_j}]=1$（保证 $f_j$ 是密度）；
2. 原书说可以进一步证明解满足 $E[s_j]=0$、$\mathrm{Var}(s_j)=1$（习题 14.18），即两个矩约束被 $\varphi$ 的选择「吸收」掉了。

> **推导** · 归一化约束与矩条件
>
> 密度为 $f_j(s)=\varphi(s)e^{g(s)}$，则（用 $\varphi'/\varphi=-s$）
>
> $$E_f[s]=\int s\varphi(s)e^{g(s)}ds=-\int\varphi'(s)e^{g(s)}ds=-\bigl[\varphi e^g\bigr]_{-\infty}^{+\infty}+\int\varphi(s)g'(s)e^{g(s)}ds=\int\varphi(s)g'(s)e^{g(s)}ds$$
>
> 边界项为零（密度可积）。同理 $\mathrm{Var}_f(s)=E_f[s^2]-E_f[s]^2$。所以「$f_j$ 的均值零、方差一」变成关于 $g'$ 的两个积分条件。取 $g'(s)=cs$（即 $g$ 是二次的）时 $E_f[s]=c\,E_f[s^2]$，令 $c=0$ 就得到零均值族中最简单的一支 $g(s)=\frac{c}{2}s^2$，此时
>
> $$f(s)\propto\varphi(s)e^{cs^2/2}$$
>
> 这正是**高斯族内部的「精度重整」**；归一化常数把 $c$ 吸收，使 $f$ 仍是标准正态。也就是说：「(14.89) 里 $g$ 的一次项会被归一化条件吃掉，真正携带信息的是 $g$ 的非线性部分」。原书正是沿着这条线索：增大惩罚 $\lambda_j$ 会把 $\hat g_j$ 拉向常数，从而 $f_j\to\varphi$。
>
> **坑** · (14.89) 的模型**过参数化**：不对 $g_j$ 加限制时，$g_j$ 可以任意复杂，极小化/极大化没有唯一解。原书的修补办法是在 (14.91) 里减两个罚项：一个是上面这个归一化约束，另一个是粗糙度罚项 $\int\lambda_j\{g_j\}(t)\,\mathrm{d}t$，它保证解 $\hat g_j$ 是以观测值 $s_{ij}=a_j^\top x_i$ 为节点的四次样条。

---
