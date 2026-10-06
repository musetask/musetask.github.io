## 3.7 降维的线性方法 {#s-3-7}

前两节我们一直在做同一件事：把 $\arg\min_\beta\lVert y-X\beta\rVert^2$ 写清楚，然后往目标函数里加罚款。这一节换个角度——**先把 $p$ 维的预测子压到 $m$ 维，再对 $m$ 个新变量做最小二乘**。逻辑是：$p$ 大时 $\mathrm{Var}(\hat\beta_j)$ 随 $1/(N-p)$ 爆炸（见 §3.2 与 (3.8)），把自由度限制到 $m\ll N$ 就把方差压回去了。

三种线性降维方法都写成「响应/预测子被投影到一个 $m$ 维子空间」的形式，但**投影方向怎么选**是它们的全部区别：

- **PCR**：方向与 $y$ 无关，只由 $X$ 的协方差结构（主成分）决定；
- **CCA**：方向同时看 $X$ 和 $Y$，使两组变量的相关系数最大；
- **PLS**：方向由「交替最小化残差」这一优化过程隐式决定，不是一个显式的特征值问题。

本节的核心结论是：在标准模型下，**PLS 的得分方向与 PCR / CCA 的方向重合**，差别只在用几个分量。推导见 §3.7.4。对应原文 <a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-7">原文 §3.7</a>。

### 3.7.1 主成分回归的三个式子 {#s-3-7-1}

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-7">原文 §3.7.1 主成分回归</a>

**问题**：把「只用前 $m$ 个主成分做线性回归」写成 $\hat y$ 和 $\hat\beta$ 的闭式，并说明得分 $z_m$ 本身是 $X$ 的什么组合。ESL 对应 (3.61)(3.62)。

> **基础知识** · 记号与两个正交分解
>
> 全部数据中心化：$x_j$ 是 $X\in\mathbb{R}^{N\times p}$ 的第 $j$ 列（已减去均值 $\bar x_j$），$\tilde y=y-\bar y\mathbf 1\in\mathbb{R}^N$（下文为省字写成 $y$）。记内积 $\langle a,b\rangle=a^\top b$。
>
> $X$ 的谱分解（见预备知识 L3）：
>
> $$X^\top X=\sum_{k=1}^{p}d_k^2v_kv_k^\top,\qquad V^\top V=I$$
>
> 主成分得分 $z_k:=Xv_k=d_ku_k$，其中 $u_k=Xv_k/d_k$ 是 $X^\top X$ 的特征向量 $v_k$ 在 $x$ 空间的像。**关键正交性**（可直接展开验证）：
>
> $$\langle z_k,z_{k'}\rangle=v_k^\top X^\top Xv_{k'}=v_k^\top\Big(\sum_{l}d_l^2v_lv_l^\top\Big)v_{k'}=d_k^2\,v_k^\top v_{k'}=d_k^2\delta_{kk'}$$
>
> 最后一步用到 $V^\top V=I$ 与 $V\Lambda V^\top V=\Lambda$。整个第 14 章的 PCA 与这里的 PCR 用的是同一套式子。

**(a) 拟合值**：对固定的 $z_m$，只有一个自由参数 $\hat\theta_m$。

$$
\hat y_{(m)}=\bar y\,\mathbf 1+\hat\theta_m z_m,\qquad \hat\theta_m=\frac{\langle z_m,y\rangle}{\langle z_m,z_m\rangle}=\frac{v_m^\top X^\top y}{d_m^2}\ \eqno{3.61}
$$

> **推导** · 为什么 (3.61) 是唯一解
>
> 目标函数只剩一维：$R(\theta)=\sum_{i=1}^N(\tilde y_i-\theta z_{mi})^2$。展开成两个标量积
>
> $$R(\theta)=\lVert y\rVert^2-2\theta\langle z_m,y\rangle+\theta^2\lVert z_m\rVert^2$$
>
> （第二项用了 $\sum_i\theta^2z_{mi}^2=\theta^2\langle z_m,z_m\rangle$。）求导
>
> $$\frac{dR}{d\theta}=-2\langle z_m,y\rangle+2\theta\lVert z_m\rVert^2,\qquad \frac{d^2R}{d\theta^2}=2\lVert z_m\rVert^2>0$$
>
> 二次项系数严格正 ⟹ $R$ 严格凸，唯一极小点令一阶导为零，得 $\hat\theta_m=\langle z_m,y\rangle/\langle z_m,z_m\rangle$。分子分母再代入 $z_m=Xv_m$：$\langle z_m,y\rangle=v_m^\top X^\top y$，$\langle z_m,z_m\rangle=v_m^\top X^\top Xv_m=d_m^2$。
>
> 由此得到一个后面到处要用的**对称投影算子**：
>
> $$\hat y_{(m)}=X\frac{v_mv_m^\top}{d_m^2}X^\top y \qquad\Longrightarrow\qquad H_m=\frac{v_mv_m^\top X^\top Xv_m}{d_m^4}=\frac{v_mv_m^\top}{d_m^2}$$
>
> 这里 $H_mv_m = v_m(v_m^\top v_m)/d_m^2=v_m/d_m^2=v_m/d_m^2$，且 $H_mx=H_mXv_m = v_m(v_m^\top Xv_m)/d_m^2=v_m$，故 $H_m$ 确实是从 $x$ 空间映到 $\mathrm{span}(z_m)$ 的正交投影（$H_m^\top=H_m$、$H_m^2=H_m$，见预备知识 L2）。

**(b) 系数**：把 (3.61) 的 $\hat y_{(m)}$ 写成 $X\hat\beta$。

$$
\hat\beta_{\rm pcr}^{(m)}=\frac{X^\top z_m}{\langle z_m,z_m\rangle}=\frac{X^\top Xv_m}{d_m^2}=\hat\theta_m v_m\ \eqno{3.62}
$$

> **推导** · 从拟合值到系数
>
> $\hat\beta$ 必须使 $X\hat\beta=\hat\theta_mXv_m$。取 $\hat\beta=(X^\top X)^{-1}X^\top\hat y_{(m)}$（见预备知识 L2 的投影定理），代入 $\hat y_{(m)}=\hat\theta_mXv_m$：
>
> $$\hat\beta=(X^\top X)^{-1}X^\top\hat\theta_mXv_m=\hat\theta_m v_m$$
>
> 中间一步只用了 $X^\top X\,v_m=d_m^2v_m$。所以 (3.62) 的信息量是：**PCR 的系数向量就是第 $m$ 个主成分方向乘上一个标量**，标量由 $y$ 决定、方向完全由 $X$ 决定。

**(c) 得分是原预测子的组合**：

$$
z_m=\sum_{j=1}^{p}\phi_{mj}x_j,\qquad \phi_{mj}=\frac{\langle x_j,z_m\rangle}{d_m^2}=v_{mj}\ \text{（ESL 的等价记法：}\ \phi_{mj}=\frac{\langle x_j,y\rangle}{\sum_{j'\le m}\langle x_{j'},y\rangle}\text{）}
$$

> **推导** · $\phi_{mj}$ 的来历
>
> 第一步：$z_m=Xv_m=\sum_j v_{mj}x_j$ 是定义，所以 $\phi_{mj}=v_{mj}$ 是同义反复。
>
> 第二步（**可用的形式**）：由 (3.62)，$\hat\beta_{\rm pcr}^{(m)}$ 的第 $j$ 个分量是 $\hat\theta_m v_{mj}$。另一方面，对 $\hat\beta_{\rm pcr}^{(m)}$ 逐分量看回归系数与变量-响应内积的关系：
>
> $$\hat\theta_m v_{mj}=\frac{v_m^\top X^\top y}{d_m^2}v_{mj}=\frac{\langle z_m,y\rangle\,v_{mj}}{d_m^2}=\frac{\langle z_m,y\rangle}{d_m^2}\cdot\frac{\langle x_j,z_m\rangle}{d_m^2}\cdot\frac{d_m^2}{\langle x_j,z_m\rangle}\ldots$$
>
> 直接算更干净：$\langle x_j,z_m\rangle=\langle x_j,Xv_m\rangle=v_m^\top X^\top x_j$。而由 $X^\top x_j=\sum_k d_k(u_k^\top x_j)v_k$（对 $X=\sum_k d_ku_kv_k^\top$ 取 $X^\top$ 再作用 $x_j$，用 $V^\top V=I$ 收集），得
>
> $$\langle x_j,z_m\rangle=v_m^\top X^\top x_j=d_m\,(u_m^\top x_j)\quad\Longrightarrow\quad v_{mj}=\frac{\langle x_j,z_m\rangle}{d_m^2}$$
>
> 第三步（ESL 用的那一步）：把 $\langle x_j,z_m\rangle$ 换成 $\langle x_j,y\rangle$。设残差 $r_m=y-\sum_{k\le m}\hat\theta_kz_k$，则 $r_m\perp\mathrm{span}(z_1,\ldots,z_m)$，于是
>
> $$\langle x_j,y\rangle=\langle x_j,\hat y_{(m)}\rangle+\langle x_j,r_m\rangle=\sum_{k\le m}\hat\theta_k\langle x_j,z_k\rangle+\langle x_j,r_m\rangle$$
>
> 这里**不能**把最后一项丢掉：$r_m\perp \mathrm{span}(z_1,\ldots,z_m)\subseteq\mathrm{col}(X)$，但 $x_j$ 本身在 $\mathrm{col}(X)$ 里。只有当 $r_m\perp x_j$ 时才有 $\langle x_j,y\rangle=\hat\theta_m\langle x_j,z_m\rangle$，代入 (3.61) 的 $\hat\theta_m=\langle z_m,y\rangle/d_m^2$ 得
>
> $$\phi_{mj}=\frac{\langle x_j,z_m\rangle}{d_m^2}=\frac{\langle x_j,y\rangle}{\hat\theta_m d_m^2}=\frac{\langle x_j,y\rangle}{\langle z_m,y\rangle/\langle z_m,z_m\rangle\cdot\langle z_m,z_m\rangle}\ \text{（ESL 记法）}$$
>
> 这正是 上面的 $\phi_{mj}$ 式 右边那个「$\langle x_j,y\rangle$ 除以一个共同的归一化量」的形式：分子是变量与响应的内积，分母是该得分与响应的内积。

> **结果** · PCR 的三条性质
>
> - **方向与 $y$ 无关**：$\hat\beta_{\rm pcr}^{(m)}$ 的方向永远是 $v_m$，$y$ 只影响长度 $\hat\theta_m$。所以加不加一个新的响应变量不会改变主成分方向（除非它改变了 $X^\top X$，即 $X$ 本身变了）。
> - **方差随 $m$ 下降**：$\hat\theta_m = u_m^\top y/d_m$，在独立同方差噪声下 $\mathrm{Var}(\hat\theta_m)=\sigma^2/d_m^2$，而 $d_m^2$ 随 $m$ 递减，故 RSS 随 $m$ 递减。同时 $H_m$ 的迹 $=d_m^2/d_m^2=1$，即每个主成分恰好用掉 1 个自由度（对比 §3.6.4 的 (3.60)）。
> - **单变量系数不稀疏**：$\hat\beta_{\rm pcr}^{(m)}=\hat\theta_m v_m$，$v$ 有 $p$ 个非零分量（一般情形），所以 $m=1$ 也能「用」所有 $p$ 个预测子。这与后面 §3.8 的 lasso 形成尖锐对比。

> **坑**
>
> 1. $d_m^2=0$（第 $m$ 个特征值为 0）时 (3.61)(3.62) 都无定义。此时应把 $m$ 限在 $\mathrm{rank}(X)$ 以内，或者用伪逆 $X^\dagger$（见预备知识 L3）。
> 2. 中心化是必需的：$X^\top X$ 度量的是**中心化后**的协方差（差一个 $N$ 的因子）。不中心化时第一个主成分通常就是常数向量 1，PCR 会退化成「只预测均值」。
> 3. 上面的 $\phi_{mj}$ 式 中带 $\langle x_j,y\rangle$ 的形式只在 $r_m\perp x_j$ 时严格成立（最保险的情形是 $m=p$，此时 $r_m=0$）。原文为行文简洁省略了 $\langle x_j,r_m\rangle$ 一项，实践中的 $\phi_{mj}$ 应按 $\langle x_j,z_m\rangle/d_m^2$ 计算。

### 3.7.2 主成分作为预测子：两种说法 {#s-3-7-2}

**问题**：主成分的经典定义是「方差最大」，但 上面的 $\phi_{mj}$ 式 用到的却是「预测 $y$」。这两种说法是一回事吗？(3.63)(3.64) 给出答案：不完全是。

> **基础知识** · 二次型极值（见预备知识 C2、L3）
>
> $\max_{\lVert\alpha\rVert_2=1}\alpha^\top A\alpha$ 的解是 $A$ 最大特征值对应的单位特征向量，值是该特征值。推导：谱分解 $A=V\Lambda V^\top$，令 $\alpha=Vz$，约束变成 $\lVert z\rVert=1$，则 $\alpha^\top A\alpha=\sum_i\lambda_i z_i^2\le\lambda_1\sum_iz_i^2=\lambda_1$，等号在 $z=e_1$。

**(a) 经典说法：方差最大**

$$
\max_{\alpha}\ \mathrm{Var}(X\alpha)\ \text{ s.t. } \lVert\alpha\rVert_2=1\ \eqno{3.63}
$$

> **推导** · (3.63) 的解
>
> 中心化后 $\mathrm{Var}(X\alpha)=\alpha^\top\mathrm{Cov}(X)\alpha\cdot N$，而 $N\mathrm{Cov}(X)=X^\top X=\sum_kd_k^2v_kv_k^\top$，比例因子与 $\alpha$ 无关。由上面的二次型极值定理，(3.63) 的极值点是 $\alpha=v_1$，极值是 $d_1^2$。**这个优化完全没出现 $y$**，这就是「主成分是纯 $X$ 的对象」的来源。

**(b) 预测导向的说法**

$$
\max_{\alpha}\ \mathrm{Corr}^2(y,X\alpha)\,\mathrm{Var}(X\alpha)\ \eqno{3.64}
$$

> **推导** · (3.64) 的解 **不是** $v_1$
>
> 记 $s:=X^\top y\in\mathbb{R}^p$，$S:=X^\top X$。三个量分别是
>
> $$\mathrm{Cov}(y,X\alpha)=\alpha^\top s,\qquad \mathrm{Var}(y)=\frac{\lVert y\rVert^2}{N},\qquad \mathrm{Var}(X\alpha)=\alpha^\top S\alpha$$
>
> 所以 (3.64) 的目标函数等于
>
> $$F(\alpha)=\mathrm{Corr}^2\cdot\mathrm{Var}=\frac{(\alpha^\top s)^2}{\mathrm{Var}(y)}\cdot\frac{\alpha^\top S\alpha}{\alpha^\top S\alpha}=\frac{1}{\mathrm{Var}(y)}\cdot\frac{(\alpha^\top s)^2}{\alpha^\top S\alpha}$$
>
> **$\mathrm{Var}(X\alpha)$ 被约掉了**——这是本节最值得记住的一步：$\mathrm{Corr}^2$ 里的分母 $\mathrm{Var}(X\alpha)$ 与乘在外面的 $\mathrm{Var}(X\alpha)$ 正好抵消。于是 (3.64) 与 (3.63) 优化的是**完全不同的两个泛函**。
>
> 用 Cauchy–Schwarz（见预备知识 L1）估计分子：$(\alpha^\top s)^2=\big(\alpha^\top S^{1/2}\cdot S^{-1/2}s\big)^2\le(\alpha^\top S\alpha)\big(s^\top S^{-1}s\big)$，等号成立当且仅当 $S^{1/2}\alpha\propto S^{-1/2}s$，即 $\alpha\propto S^{-1}s$。故
>
> $$\hat\alpha_{\rm pred}=\frac{S^{-1}s}{\lVert S^{-1}s\rVert},\qquad F_{\max}=\frac{s^\top S^{-1}s}{\mathrm{Var}(y)}=\frac{N\,s^\top S^{-1}s}{\lVert y\rVert^2}$$
>
> 而 $S^{-1}s=(X^\top X)^{-1}X^\top y=\hat\beta_{\rm ls}$（(3.6)）。**所以 (3.64) 的答案是最小二乘方向，而不是第一主成分方向。**

> **结果** · 两种说法何时一致
>
> $v_1=S^{-1}s/\lVert\cdot\rVert$ 当且仅当 $s$ 是 $S$ 的特征向量，即 $X^\top y\parallel v_1$。一般情形下二者不同，因为：
>
> | 说法 | 优化对象 | 解 | 用在哪 |
> |---|---|---|---|
> | (3.63) 方差最大 | $X$ 单独 | $v_1$，与 $y$ 无关 | 无监督降维、可视化、去噪 |
> | (3.64) 预测相关最大 | $X$ 与 $y$ 联合 | $\hat\beta_{\rm ls}$ | 监督降维（这才是「降维回归」想要的） |
>
> PCR 站在 (3.63) 这一边（方向由 $X$ 定），CCA/PLS 站在 (3.64) 这一边。**这就是 PCA 回归与 PLS 回归的分野**。

> **坑** · (3.64) 写成无约束形式时目标函数在 $\alpha=0$ 附近是 $0/0$（$\mathrm{Corr}$ 未定义）。数学上 $F(\alpha)$ 的表达式在 $\alpha\ne0$ 时与 $\alpha$ 的方向无关（只有齐次性），所以补上 $\lVert\alpha\rVert_2=1$ 不改变解；但要保证 $s\notin\mathrm{null}(S)$，否则最大值退化。

### 3.7.3 典型相关分析：两视图模型与特征值问题 {#s-3-7-3}

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-7">原文 §3.7.3 典型相关分析</a>

**问题**：当响应是 $K$ 维向量、预测子分成两组时，如何找「同时能解释两组」的投影方向？CCA 给出答案，而它的最优性条件是一个**广义特征值问题**——本节要把它推出来。

> **基础知识** · 拉格朗日乘子与广义特征值问题（见预备知识 O1）
>
> $\max_u\, u^\top M u$ s.t. $u^\top A u=1$（$M$、$A$ 对称，$A$ 正定）的拉格朗日函数 $L=u^\top Mu-\gamma(u^\top Au-1)$，一阶条件 $2Mu=2\gamma Au$，即 $M u=\gamma A u$。代入约束：$\gamma=u^\top Mu/u^\top Au$，故 $\gamma\le\lambda_{\max}(A^{-1/2}MA^{-1/2})$（由 Cauchy–Schwarz），最大值在对应特征向量处取到。

**(a) 两视图模型**

$$
Y_k=f_k(X)+\varepsilon_k,\qquad k=1,\dots,K\ \eqno{3.65}
$$

$$
X_j=g_j(Y)+\delta_j,\qquad j=1,\dots,p\ \eqno{3.66}
$$

(3.65) 是本章的线性模型（$X\in\mathbb{R}^{N\times p}$，$Y\in\mathbb{R}^{N\times K}$）；(3.66) 是把角色互换的同一模型。**两侧同时被建模**是 CCA 与单向 PLS 的区别：CCA 只用样本 $\{X_i,Y_i\}_{i=1}^N$，不假设哪个是「结果」。

**(b) CCA 的定义**

$$
\max_{u\in\mathbb{R}^K,\ v\in\mathbb{R}^p}\ \mathrm{Corr}^2(Yu,\,Xv)\ \eqno{3.67}
$$

> **推导** · CCA 最优权重 = 最小残差 = 广义特征值问题

记三个 $N\times N$ 的二阶矩矩阵（都对称）：$S_{XX}=X^\top X$，$S_{XY}=X^\top Y$，$S_{YY}=Y^\top Y$。展开相关系数：

$$\mathrm{Corr}^2(Yu,Xv)=\frac{\big(u^\top S_{YX}v\big)^2}{\big(u^\top S_{YY}u\big)\big(v^\top S_{XX}v\big)}$$

**第一步：归一化消掉分母。** (3.67) 在 $(u,v)\mapsto(-u,v)$、$(u,-v)$ 下不变，故可以加两个等式约束 $u^\top S_{YY}u=1$、$v^\top S_{XX}v=1$（这两个约束可行：取 $u=S_{YY}^{-1}e_1/\lVert\cdot\rVert$ 即可）。此时目标变成 $\max\, (u^\top S_{YX}v)^2$。

**第二步：写出两个一阶条件。** 拉格朗日函数

$$L(u,v,\gamma_1,\gamma_2)=2u^\top S_{YX}v-\gamma_1\big(u^\top S_{YY}u-1\big)-\gamma_2\big(v^\top S_{XX}v-1\big)$$

（系数 $2$ 是为了消掉公因子。）两个偏导为零（用 $\nabla_u(u^\top Mu)=2Mu$，见预备知识 L4）：

$$\frac{\partial L}{\partial u}=2S_{YX}v-2\gamma_1S_{YY}u=0,\qquad \frac{\partial L}{\partial v}=2S_{YX}^\top u-2\gamma_2S_{XX}v=0$$

从第一个解出 $u$：$S_{YY}$ 可逆（$K\le N$ 且满秩），故 $u=\frac1{\gamma_1}S_{YY}^{-1}S_{YX}v\propto S_{YY}^{-1}S_{YX}v$。从第二个解出 $v\propto S_{XX}^{-1}S_{YX}^\top u$。**把后者代回前者**：

$$S_{YY}^{-1}S_{YX}\cdot\frac1{\gamma_2}S_{XX}^{-1}S_{YX}^\top\cdot\frac1{\gamma_1}S_{YY}^{-1}S_{YX}v\ \propto\ v$$

整理成关于 $u$ 的特征值问题（把 $S_{YX}^\top u=\gamma_2S_{XX}v$ 用回第一式）：

$$S_{YX}\big(\tfrac1{\gamma_2}S_{XX}^{-1}S_{YX}^\top\big)\big(\tfrac1{\gamma_1}S_{YY}^{-1}S_{YX}v\big)\ \ldots$$

更直接：两式合成为 $u=\gamma_1^{-1}\gamma_2^{-1}S_{YY}^{-1}S_{YX}S_{XX}^{-1}S_{YX}^\top u$，而回代时 $\gamma_2$ 已被吸收，故标准写法是

$$v_m\ \propto\ S_{XX}^{-1}S_{XY}\,u_m\ \ \text{（$X$ 空间权重）}\qquad\text{（$X$ 空间权重）}$$

$$u_m\ \propto\ S_{YY}^{-1}S_{YX}\,v_m\ \ \text{（$Y$ 空间权重）}\qquad\text{（$Y$ 空间权重）}$$

$$S_{YY}^{-1}S_{YX}S_{XX}^{-1}S_{XY}\,u_m=\rho_m^2\,u_m$$

(3.67c) 就是**CCA 的特征值问题**：$S_{YY}^{-1}S_{YX}S_{XX}^{-1}S_{XY}$ 一般不对称，所以它是广义特征值问题而非通常的特征值问题；把它对称化：令 $a=S_{YY}^{-1/2}u$、$b=S_{XX}^{-1/2}v$，则 (3.67) 的分子变为

$$u^\top S_{YX}v=a^\top\underbrace{\Big(S_{YY}^{-1/2}S_{YX}S_{XX}^{-1/2}\Big)}_{=:T}b,\qquad T\ \text{对称正半定（是 }X^\top\Sigma_X^{-1}Y\ \text{的 Gram 阵）}$$

约束变成 $\lVert a\rVert=\lVert b\rVert=1$，于是 $\mathrm{Corr}^2=\big(a^\top Tb\big)^2\le\lambda_1^2(T)$（Cauchy–Schwarz），等号在 $a=b=$ 顶级奇异向量处。所以 $\rho_m^2$ 就是 $T$ 的第 $m$ 个奇异值之平方。

**第三步：同一个特征值问题也来自「最小残差」。** 这一步说明 CCA 也可以被看成降维回归。取 $v=S_{XX}^{-1}S_{XY}u$，则

$$\mathrm{Corr}^2(Yu,Xv)=\frac{u^\top S_{YX}S_{XX}^{-1}S_{XY}u}{u^\top S_{YY}u}=1-\frac{u^\top\underbrace{\big(S_{YY}-S_{YX}S_{XX}^{-1}S_{XY}\big)}_{=:M}u}{u^\top S_{YY}u}$$

（用了 $S_{YX}=S_{XY}^\top$，所以 $S_{YX}S_{XX}^{-1}S_{XY}$ 对称。）$M\succeq0$ 是残差协方差：$M=(Y-\hat Y_{\rm ls})^\top(Y-\hat Y_{\rm ls})$，其中 $\hat Y_{\rm ls}=XS_{XX}^{-1}S_{XY}$。于是「最大化 $\mathrm{Corr}^2$」$\iff$「在 $u^\top S_{YY}u=1$ 下最小化 $u^\top Mu$」。对这个最小化问题用拉格朗日乘子：$2Mu-2\rho^2S_{YY}u=0$，即

$$Mu=\rho^2S_{YY}u\iff S_{YY}^{-1}Mu=\rho^2u\iff S_{YY}^{-1}S_{YX}S_{XX}^{-1}S_{XY}u=\rho^2u$$

与 (3.67c) **完全相同**。

> **结果** · 一个特征值问题，三种读法
>
> $$S_{YY}^{-1}S_{YX}S_{XX}^{-1}S_{XY}u_m=\rho_m^2u_m$$
>
> 1. $\max\mathrm{Corr}^2(Yu,Xv)$ 的驻点（第一、二步）；
> 2. $\min\,u^\top(S_{YY}-S_{YX}S_{XX}^{-1}S_{XY})u$ s.t. $u^\top S_{YY}u=1$ 的驻点（第三步）；
> 3. $u^\top M u$ 的广义 Rayleigh 商，其中 $M$ 是残差 $Y-\hat Y_{\rm ls}$ 的 Gram 阵。
>
> 而 $v_m\propto S_{XX}^{-1}S_{XY}u_m$ (3.67a) 恰是「用 $u_m$ 去线性预测 $Y$，再把 $X$ 投影过去」的方向。**这正是 PLS 的权重**，见下一小节。

> **坑**
>
> 1. CCA 的 $\rho_m^2$ 是相关系数的平方，故 $0\le\rho_m^2\le1$；若 $S_{XX}$ 或 $S_{YY}$ 秩亏，$S^{-1}$ 不存在，必须用伪逆并把零空间的贡献单独处理（等价于先做一次删列）。
> 2. $K>1$ 时「典型相关系数」有 $K$ 个，**没有唯一的自然顺序之外的选择**；若只取前 $m$ 个，剩下的方向由约束归一化方式决定，需要一个明确约定。
> 3. 上面所有式子都要求**列中心化**，否则 $S_{XX}$ 含均值方向，得分会被常数向量主导。

### 3.7.4 偏最小二乘：交替最小化的驻点 {#s-3-7-4}

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-7">原文 §3.7.4 偏最小二乘</a>

**问题**：PLS 不直接解 (3.67) 的特征值问题，而是交替最小化一个双线性目标。这个交替过程在收敛点上满足什么条件？结论是：**得分方向退化成 PCA / CCA 的方向**。

> **基础知识** · 矩阵微分（见预备知识 L4）与迹循环律
>
> $\nabla_B\mathrm{tr}(B^\top M)=M$（$M$ 常矩阵）；$A$ 对称时 $\nabla_B\mathrm{tr}(B^\top AB)=(A+A^\top)B=2AB$。两者都在 L4 里有逐分量证明。
>
> **维度约定**（本小节全部式子按此读）：$X\in\mathbb{R}^{N\times p}$，$Y\in\mathbb{R}^{N\times K}$，$B\in\mathbb{R}^{p\times K}$；$U_m\in\mathbb{R}^{K\times m}$ 是响应权重，$YU_m\in\mathbb{R}^{N\times m}$ 是响应得分，$U_{-m}\in\mathbb{R}^{m\times K}$ 是响应载荷。

**(a) PLS 目标函数**

$$
\hat B_{\rm rr}^{(m)}=\arg\min_B\ \sum_{i=1}^{N}\big(y_i-B^\top x_i\big)^\top\Sigma^{-1}\big(y_i-B^\top x_i\big)\ \eqno{3.68}
$$

> **推导** · 把 (3.68) 写成迹并求一阶条件
>
> 设 $R=Y-XB\in\mathbb{R}^{N\times K}$。单项是 $\sum_i r_i^\top\Sigma^{-1}r_i=\sum_i\mathrm{tr}(r_i^\top\Sigma^{-1}r_i)=\mathrm{tr}(R^\top\Sigma^{-1}R)$。所以 (3.68) 的目标函数是 $F(B)=\mathrm{tr}(R^\top\Sigma^{-1}R)$。展开成三项：
>
> $$F(B)=\mathrm{tr}(Y^\top\Sigma^{-1}Y)-2\mathrm{tr}\big(B^\top X^\top\Sigma^{-1}Y\big)+\mathrm{tr}\big(B^\top (X^\top\Sigma^{-1}X)B\big)$$
>
> 逐项求导：
>
> $$\nabla_BF=-2X^\top\Sigma^{-1}Y+2(X^\top\Sigma^{-1}X)B=2X^\top\Sigma^{-1}(XB-Y)$$
>
> 第二个等号用了 $\Sigma^{-1}$ 对称 ⟹ $X^\top\Sigma^{-1}X$ 对称，于是 $\nabla_B\mathrm{tr}(B^\top AB)=2AB$。令 $\nabla_BF=0$：
>
> $$X^\top\Sigma^{-1}(Y-XB)=0\ \Longrightarrow\ \Sigma^{-1}(Y-XB)\ \perp\ \mathrm{col}(X)$$

这就是**一阶条件**：残差矩阵在 $\Sigma^{-1}$ 度量下与 $X$ 的列空间正交。$\Sigma=I$ 时它退化成最小二乘正规方程 $X^\top(Y-XB)=0$。

**(b) 权重矩阵**

$$
\hat B_{\rm rr}^{(m)}=\hat U_m\,\hat U_{-m}\ \eqno{3.69}
$$

其中 $\hat U_m=(X^\top X)^{-1}X^\top(YU_m)\in\mathbb{R}^{p\times m}$ 是「$X$ 空间里被 (3.68) 的一阶条件 投影过的得分方向」，$\hat U_{-m}=U_{-m}\in\mathbb{R}^{m\times K}$ 是响应载荷（把得分映回 $Y$）。于是 $\hat B_{\rm rr}^{(m)}$ 与 $YU_m$ 的关系是

$$\hat B_{\rm rr}^{(m)}=\hat U_mU_{-m}\ \Longleftrightarrow\ \hat Y_{\rm rr}^{(m)}:=X\hat B_{\rm rr}^{(m)}=H\,(YU_m)U_{-m}$$

即「先把 $Y$ 投到 $m$ 个得分上、再用 $H$ 映回 $x$ 空间」。这个乘积的形状是本小节所有结论的载体。

**(c) 最终系数**

$$
\hat B_{\rm rr}^{(M)}=(X^\top X)^{-1}X^\top\,(Y\,U_m)\,U_{-m}\ \eqno{3.70}
$$

> **推导** · (3.70) 是 (3.69) 的 $B$-更新步
>
> PLS 的一步迭代是：给定 $U_{-m}$，求 $B$ 使 $(Y\hat U_{-m}^\top)$ 被 $\mathrm{col}(X)$ 最好地解释。把这步写成
>
> $$\min_B\ \lVert YU_{-m}^\top-XB U_{-m}^\top\rVert^2\ \text{ s.t. } BU_{-m}^\top\ \text{给定}$$
>
> 固定 $U_{-m}$（列满秩）时它等价于 $\min_{T}\lVert YU_{-m}^\top-XT\rVert^2$，其中 $T=BU_{-m}^\top$，其解是标准最小二乘 $T=(X^\top X)^{-1}X^\top YU_{-m}^\top$（见 (3.6)）。回代并乘回 $U_{-m}$：
>
> $$B=(X^\top X)^{-1}X^\top YU_{-m}^\top(U_{-m}^\top U_{-m})^{-1}U_{-m}^\top$$
>
> 把 $(U_{-m}^\top U_{-m})^{-1}$ 吸收进响应权重的定义（把 $U_{-m}$ 正交归一化，$U_{-m}^\top U_{-m}=I$），$B=(X^\top X)^{-1}X^\top YU_{-m}^\top=(X^\top X)^{-1}X^\top(YU_{-m})U_{-m}^\top$。令 $U_{-m}^\top:=U_{-m}$，$YU_{m}:=YU_{-m}^\top$，得 (3.70)。

**(d) 拟合值**

$$
\hat Y_{\rm rr}^{(m)}=H\,Y_{\rm pcr}^{(m)}\ \eqno{3.71}
$$

其中 $H=X(X^\top X)^{-1}X^\top$ 是帽子矩阵（对称幂等，$\mathrm{tr}(H)=p$，见预备知识 L2），$Y_{\rm pcr}^{(m)}$ 是用前 $m$ 个主成分得到的秩 $m$ 拟合。

> **核心推导** · 为什么 $\hat U_m$ 同时是 PCR 和 PLS 的方向
>
> 这是 §3.7 最需要补的一步，ESL 用一句话带过。
>
> **第一步（PLS 驻点）：** 在交替最小化的收敛点上，两个更新步都是最优的。
>
> - $B$-步给出 (3.68) 的一阶条件：$\Sigma^{-1}(Y-XB)\perp\mathrm{col}(X)$；
> - $U$-步给出 $\hat U_{-m}$ 是 $YU_{-m}^\top$ 的**最佳秩 $m$ 近似**的左奇异方向。等价地（Eckart–Young 一阶条件）：令 $E=Y-XB$，则
>
> $$\hat U_{-m}^\top\big(YU_{-m}^\top-XB U_{-m}^\top\big)=\hat U_{-m}^\top E U_{-m}^\top=0\ \Longrightarrow\ \hat U_{-m}^\top E=0$$
>
> **第二步（残差方向可以扔掉）：** 设 $H=X(X^\top X)^{-1}X^\top$，$E_0=(I-H)Y$。由 $H$ 对称幂等，$\mathrm{col}(E_0)\perp\mathrm{col}(X)$。现在看 $Y^\top Y=H Y^\top H Y+E_0^\top E_0$（展开：$Y=HY+E_0$，$Y^\top Y=H^\top Y^\top H Y + Y^\top (I-H^\top)E_0+\ldots$，交叉项用 $\mathrm{col}(E_0)\perp\mathrm{col}(X)=\mathrm{col}(HY)$ 消掉）。若 $E_0^\top E_0$ 的谱范数不超过 $HY^\top H Y$ 的第 $m$ 个特征值，则 $Y$ 与 $HY$ 的前 $m$ 个左奇异子空间**重合**，即
>
> $$\mathrm{col}(\hat U_{-m})=\mathrm{span}\big\{\text{前 } m \text{ 个左奇异向量 of } Y\big\}=\mathrm{span}\big\{\text{前 } m \text{ 个左奇异向量 of } HY\big\}=\mathrm{span}\{v_1,\ldots,v_m\}\cdot(\text{在 }x\text{ 空间})$$
>
> 而 $\mathrm{col}(HY)\subseteq\mathrm{col}(X)$，故还有 $\hat U_{-m}^\top(I-H)=0$。
>
> **第三步（与 CCA 对上）：** 用 上面的条件式 与 $\hat U_{-m}^\top(I-H)=0$：
>
> $$\hat U_{-m}^\top E U_{-m}^\top=0,\quad E=Y-HY\ \Longrightarrow\ \hat U_{-m}^\top H YU_{-m}^\top=\hat U_{-m}^\top YU_{-m}^\top$$
>
> 两边左乘 $\Sigma^{-1}$，展开 $HY=XS_{XX}^{-1}S_{XY}$，得
>
> $$\hat U_{-m}^\top S_{XX}^{-1}S_{XY}\,U_{-m}^\top=\hat U_{-m}^\top\Sigma^{-1}Y\,U_{-m}^\top$$
>
> 这与 上面两条权重式与特征值式 的联合形式 $\Sigma^{-1}\cdot$ 特征值方程逐字对应：**$\hat U_{-m}$ 就是 CCA 的 $u_m$，也是 $\mathrm{span}(z_1,\ldots,z_m)$。**
>
> **第四步（结论）：** 把第三步代回 $B$ 的表达式，$B\hat U_{-m}^\top$（即拟合的响应）落在 $\mathrm{span}(HY)\cap$（得分张成的 $m$ 维子空间），而这正是 $Y_{\rm pcr}^{(m)}$ 所张成的空间。故
>
> $$\hat Y_{\rm rr}^{(m)}=X\hat B_{\rm rr}^{(m)}Y=H\,Y_{\rm pcr}^{(m)}$$
>
> 即 (3.71)。取 $m=1$ 时两者**完全相等**；$m$ 增大后 PLS 开始偏离 PCR，但都落在同一个 $\mathrm{span}(z_1,\ldots,z_m)$ 之内（若 $m\ge K$ 则 PLS 也等于 PCR）。

> **结果** · 三种方法的关系一览
>
> | 方法 | 方向来源 | 是否需要 $Y$ | $m=1$ 时 |
> |---|---|---|---|
> | PCR | $X^\top X$ 的前 $m$ 个特征向量 | 否 | 自己的方向 |
> | CCA | $S_{YY}^{-1}S_{YX}S_{XX}^{-1}S_{XY}$ 的前 $m$ 个特征向量 | 是（同时用 $X,Y$） | 与 PCR 方向不同 |
> | PLS | 交替最小化的驻点 | 是 | **等于 PCR** |
>
> $m=1$ 时 PLS 系数还有闭式：$B_{\rm pls}^{(1)}=S_{XX}^{-1}X^\top y\cdot\frac{\langle v_1,y\rangle}{\langle z_1,y\rangle}\cdot\frac{1}{1}$，即沿 $v_1$ 的单个主成分回归——这就是「PLS 的一步就是 PCR」。

> **坑**
>
> 1. PLS 的交替最小化**不保证收敛到全局最优**（双线性目标），但保证单调下降，因此每个聚点都是驻点，上面的推导只用到「驻点」性质，不受影响。
> 2. 第二步的条件「$E_0^\top E_0$ 的谱范数不超过 $HY^\top HY$ 的第 $m$ 个特征值」在 $m$ 小、$N/p$ 大、信号足够强时成立；$m$ 接近 $K$ 时结论总是成立。这是「PLS 与 PCR 只在小 $m$ 时相近」的定量依据。
> 3. $\Sigma$ 若与 $X$ 相关，$\hat B$ 不再是 (3.6) 的最小二乘解，(3.71) 的等式也不成立——GLS 加权只在 $\Sigma$ 已知且与 $X$ 独立时是安全的。

### 3.7.5 CCA 与 ridge 之间的连续过渡 {#s-3-7-5}

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-7">原文 §3.7.5 CCA 与 ridge</a>

**问题**：CCA 用的方向是「最大化 $\mathrm{Corr}^2$」，ridge 用的是「最小化残差」。它们看起来是两件事，但存在一个参数 $c\ge0$ 把它们连成一条线：$c=0$ 是 CCA，$c\to\infty$ 是 ridge。

> **基础知识** · 关于对称幂等算子的分解（见预备知识 L1、L3）
>
> 任何 $H=H^\top=H^2$ 可以写成 $H=W\Lambda W^\top$，$\Lambda$ 是 0/1 对角矩阵（$\Lambda_{mm}=1$ 当 $w_m$ 在 $\mathrm{col}(H)$ 内）。核对：$H^2=H\Rightarrow H=H^\top H$，故 $\mathrm{col}(H)=\mathrm{range}(H^\top)$，正交分解后特征值只能取 0 或 1。**这是所有「$c=0$ 用前 $m$ 个分量、$c\to\infty$ 用全部」的写法的基础。**

$$
\hat B_{c+w}=\hat B\,\hat\Lambda\,\hat U^{-1},\qquad \hat\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_p),\ \ \lambda_m^2=\frac{c}{p}\ \eqno{3.72}
$$

$$
\hat\Lambda=\hat\Lambda(c),\qquad \hat\Lambda(0)=0,\qquad \lim_{c\to\infty}\frac{\hat\Lambda(c)}{c}=\hat I\ \eqno{3.73}
$$

$$
\hat Y_{c+w}=H\,Y_{S_{c+w}}\ \eqno{3.74}
$$

$$
\hat Y_{{\rm ridge},c+w}=A_\lambda\,Y_{S_{c+w}},\qquad A_\lambda=\big(I-\Pi_\lambda\big)^\top\ \text{的一般形式 } (X^\top X+\lambda I)^{-1}X^\top X\ \eqno{3.75}
$$

> **推导** · 三个因子的分工
>
> 把 (3.72) 的因子逐个解释（这也是原文那句话的意思）：
>
> - $\hat U$：$Y$ 的（正交归一化的）典型方向 $u_1,\ldots,u_p$ 组成的 $N\times p$ 矩阵，$\hat U^\top\hat U=I$，于是 $\hat U^{-1}=\hat U^\top$，**$\hat U^{-1}$ 把 $y$ 转到典型坐标系**；
> - $\hat\Lambda$：在典型坐标系里对每个坐标乘一个权重 $\lambda_m$，即**收缩只作用在典型方向上**；
> - $\hat B$：把典型坐标映回 $X$ 空间的系数矩阵（在 (3.67a) 的 $v_m\propto S_{XX}^{-1}S_{XY}u_m$ 上乘 $\lambda_m$）。
>
> 于是 $\hat B\Lambda\hat U^{-1}y=\hat B\Lambda(\text{典型坐标})$，与 (3.72) 一致。
>
> **端点 $c=0$。** 由 (3.73)，$\hat\Lambda(0)=0$，此时权重重数为 0，留在解里的是**不在**前 $m$ 个典型方向上的部分——把 (3.72) 写成
>
> $$\hat B_{c+w}=\hat B(\hat I-\hat\Lambda_c)\hat U^\top\ \text{的形式（等价，因为 }\hat U^\top Y\hat U\ \text{只在前 }m\text{ 个坐标上有信息）}$$
>
> 就看到 $c=0$ 时只有前 $m$ 个典型坐标被保留 ⟹ 拟合是 CCA 的 $m$ 分量解。
>
> **端点 $c\to\infty$。** 由 (3.73) 的 $\hat\Lambda(c)/c\to\hat I$，$\hat B_{c+w}/c\to\hat B\hat U^{-1}$。而 §3.4.2 中 ridge 的系数是
>
> $$\hat\beta_{\rm ridge}=(X^\top X+\lambda I)^{-1}X^\top y=\Big(I-(X^\top X+\lambda I)^{-1}\lambda\Big)\cdot\hat\beta_{\rm ls}\ \Longrightarrow\ A_\lambda=(X^\top X+\lambda I)^{-1}X^\top X$$
>
> 逐项核对：$(X^\top X+\lambda I)^{-1}X^\top X=I-\lambda(X^\top X+\lambda I)^{-1}$，故 $A_\lambda$ 是 $I$ 减去一个秩亏修正，正是 (3.75) 的形式，且 $\lim_{\lambda\to0}A_\lambda=I$（退化成 OLS）、$\lim_{\lambda\to\infty}A_\lambda=0$（完全收缩）。
>
> **拟合值的一致性。** (3.74) 用 $H$，(3.75) 用 $A_\lambda$，两者的区别正是「投影」与「ridge 平滑」的区别：$H$ 对称幂等（$H^\top=H$，$H^2=H$），$A_\lambda$ 一般既不对称也不幂等（$A_\lambda^\top=A_\lambda$ 但 $A_\lambda^2=A_\lambda-\lambda(X^\top X+\lambda I)^{-1}A_\lambda\ne A_\lambda$）。这与 §3.6.4 中 (3.60) 的两个定义何时重合是同一个问题。

> **结果** · 这条线的三种特例
>
> $$\begin{cases} c=0: & \text{CCA 的 } m \text{ 分量解，方向由 }\mathrm{Corr}^2\text{ 决定}\\[2pt] c>0: & \text{中间状态：把 CCA 方向按 }\lambda_m^2=c/p\text{ 加权后重建}\\[2pt] c\to\infty: & \text{ridge（所有方向按 }A_\lambda\text{ 收缩）}\end{cases}$$
>
> 实践意义：$c$ 提供了一个**一维**的复杂度旋钮，可以像 ridge 的 $\lambda$ 那样用交叉验证选；而 $\lambda_m^2=c/p$ 说明重数按 $m$ 的分位数取，前几个典型方向收缩最强。

> **坑**
>
> 1. $\hat U^{-1}$ 要求 $\hat U$ 列满秩，即 $K\le\mathrm{rank}(Y)$；$Y$ 有重复列时必须先合并或删列。
> 2. (3.72)(3.73) 的记号在不同版本里 $\hat\Lambda$ 与 $\hat B$ 的取法略有差别（等价地写成 $\hat\Lambda$ 作用于**保留**方向还是**丢弃**方向）。稳定的内容是三个端点与三个因子的分工，以及 (3.74)(3.75) 的区别。
> 3. $c\to\infty$ 给出的是 **ridge 而非 lasso**；这条线里没有任何一步会产生 $\ell_1$ 式的稀疏性。稀疏性必须靠 §3.8 的 $\ell_1$ 或非凸惩罚。

## 3.8 收缩方法的一般理论 {#s-3-8}

前面所有方法——岭回归、lasso、PCR、PLS、CCA——都能塞进同一个模板：往「数据拟合优度」后面加一个「复杂度罚款」。本节把这个模板写出来，说明四件事：罚款 $J(\beta)$ 该长什么样；$J$ **凸**时优化有多容易（KKT + 坐标下降 + 分离变换）；$J$ **非凸**时会坏掉什么（局部极小、必须二分 $\lambda$）；一般损失函数（不只是平方损失）下还剩下什么。统一的模板是

$$
\hat\beta(\lambda)=\arg\min_{\beta}\ \big\{R(\beta)+\lambda J(\beta)\big\}\ \eqno{3.76}
$$

**问题**：把 (3.76) 的两个部件分别定义清楚，并对常用罚款做统一比较。这是本节一切讨论的出发点，对应 (3.76)(3.77)。本节最终要回答的元问题是：**什么时候收缩是好事，什么时候它是数据挖掘。**

### 3.8.1 统一形式与惩罚对照表 {#s-3-8-1}

> **基础知识** · 凸性与 $\ell_q$ 球（见预备知识 L1、O1）
>
> $J$ 凸 ⟹ $\{J\le c\}$ 是凸集 ⟹ (3.76) 是凸优化 ⟹ KKT 必要充分、强对偶成立（见预备知识 O1、O2），$\nabla_\beta(R+\lambda J)=0$ 是全局最优的**充分**条件。
> $\ell_1$ 球 $\{\lVert\beta\rVert_1\le1\}$ 的极端点恰是 $p$ 个坐标轴方向 $\{e_j\}$，$\ell_2$ 球没有极端点——这一条事实是 §3.8.4「$\ell_0$ 罚等价于子集选择」的根源，也是 §3.8.6 centered lasso 的动机。

**损失部件**（(3.77)）：

$$
R(\beta)=\sum_{i=1}^{N}L\big(y_i,\ \beta_0+\textstyle\sum_{j=1}^{p}x_{ij}\beta_j\big)\ \eqno{3.77}
$$

$L$ 是逐点损失。平方损失 $L(y,\eta)=\tfrac12(y-\eta)^2$ 给出最小二乘 / ridge / lasso；对数损失给出 logistic 回归的 lasso 版本（见 (3.89)）。**惩罚部件与损失部件的分工**：$R$ 管「对数据解释得多好」，$J$ 管「愿意付出多少偏差」，$\lambda$ 是两者之间的兑换率——这就是 $\lambda$ 有时被称为「复杂度罚款的 Lagrange 乘子」的原因（它确实是 (3.79) 的对偶变量）。

**惩罚部件 $J(\beta)$ 的对照表**：

| $J(\beta)$ | 名称 | 凸 | 解的行为 |
|---|---|---|---|
| $\sum_j\mathbf 1\{\beta_j\ne0\}$ | $\ell_0$ / 最佳子集 | 否 | 精确稀疏，但需枚举 $2^p$ 个子集（(3.41)） |
| $\sum_j\beta_j^2$ | ridge / $\ell_2$ | 是 | 全部非零，均匀收缩，$\hat\beta_j\to0$ |
| $\sum_j\lvert\beta_j\rvert$ | lasso / $\ell_1$ | 是 | 精确稀疏，收缩量恒为 $\lambda$：$\hat\beta_j=S_{\lambda}(\hat\beta_j^{\rm ls})$ |
| $\lambda_1\lVert\beta\rVert_1+\lambda_2\lVert\beta\rVert_2^2$ | elastic net | 是 | 稀疏 + 保留成组效应，见 (3.91) |
| $\sum_j w_j\lvert\beta_j\rvert$ | adaptive / 加权 lasso | 是 | 用 $w_j$ 放大不重要变量的阈值，见 (3.81) |
| $\sum_\ell\lVert\beta^{(\ell)}\rVert_2$ | group lasso | 是 | 整组进出，见 (3.80) |
| 有界斜率的分段线性罚 | MCP / SCAD | 否 | 稀疏但不过度收缩，见 (3.82) |

> **结果** · 凸性带来的三件事
>
> - **最优性**：$\nabla_\beta(R+\lambda J)=0$ 是**充分**条件（凸情形下也是必要条件）；
> - **唯一性**：$R$ 严格凸（平方损失 + $X$ 满秩）⟹ 解唯一，$\ell_1$ 罚不会破坏唯一性，因为 $\{J\le c\}$ 是凸集，两点中点仍在集合内；
> - **可分性**：$J=\sum_jJ(\beta_j)$ ⟹ 可以逐坐标更新，这是 LARS 与坐标下降的根据（(3.90)、§3.8.6）。

> **坑** · $J$ 不能依赖 $y$，否则 (3.76) 退化。例如 $J(\beta)=\lVert\beta-\hat\beta_{\rm ls}\rVert_2^2$ 给出 $\hat\beta=\hat\beta_{\rm ls}$（完全抵消）。这正是 (3.81) 里权重 $w_j$ 只能是「第一阶段估计的**固定**函数」、必须在两阶段之间冻结的原因。

### 3.8.2 lasso 的两种等价形式 {#s-3-8-2}

**问题**：lasso 写成惩罚形式 (3.76) 很常见，但它的**原始形式**（约束形式）给出更精确的稀疏化描述，而两种形式的参数之间有精确对应。这是 (3.78)(3.79)，也是 §3.5 约束形式 (3.42) 的推广。

> **基础知识** · KKT、互补松弛与强对偶（见预备知识 O2、O1）
>
> $\min_\beta f(\beta)$ s.t. $g(\beta)\le s$ 的 KKT 四条：原始可行 $g\le s$；对偶可行 $\lambda\ge0$；**互补松弛** $\lambda(g(\beta^\star)-s)=0$；平稳性 $\nabla f+\lambda\nabla g=0$。强对偶（Slater + 凸）保证原始最优值 = 对偶最优值。
> 一个关键事实：**$\lambda$ 是约束值 $s$ 的严格减函数**。它由互补松弛给出——$s$ 放松一个量，最优的 $\lambda$ 必须相应下降，否则平稳性方程的解会超出可行集。

$$
\hat\beta_t=\arg\min_{\beta}\ \lVert\beta\rVert_1\ \ \text{s.t.}\ \ \big\lVert X^\top(y-X\beta)\big\rVert_\infty\le s\ \eqno{3.78}
$$

$$
\hat\beta_s=\arg\min_{\beta}\ \big\lVert X^\top(y-X\beta)\big\rVert_\infty\ \ \text{s.t.}\ \ \lVert\beta\rVert_1\le t\ \eqno{3.79}
$$

> **推导** · 两式通过拉格朗日函数互换
>
> 记 $r(\beta)=y-X\beta$，$g(\beta):=\lVert X^\top r(\beta)\rVert_\infty$，$h(\beta)=\lVert\beta\rVert_1$。
>
> 对 (3.78) 作拉格朗日松弛（见预备知识 O1）：
>
> $$L(\beta,\lambda)=h(\beta)+\lambda\big(g(\beta)-s\big),\qquad \lambda\ge0$$
>
> **一阶条件。** 记 $\kappa_j=\lVert x_j\rVert^2$，$c_j(\beta)=\langle x_j,r(\beta)\rangle$。注意 $\frac{\partial c_j}{\partial\beta_j}=-\langle x_j,x_j\rangle=-\kappa_j$，而 $\frac{\partial h}{\partial\beta_j}=\mathrm{sign}(\beta_j)$。所以 KKT 的平稳性给出（$\beta_j\ne0$ 时）
>
> $$\mathrm{sign}(\beta_j)-\lambda\kappa_jc_j(\beta)=0\quad\Longrightarrow\quad c_j(\beta)=\frac{\mathrm{sign}(\beta_j)}{\lambda\kappa_j}$$
>
> 而 $\beta_j=0$ 时用次梯度条件（$\partial\lVert\beta\rVert_1/\partial\beta_j$ 在 0 处是 $[-1,1]$ 中任一值）：
>
> $$\lvert\lambda\kappa_jc_j(\beta)\rvert\le1\quad\Longleftrightarrow\quad\lvert c_j(\beta)\rvert\le\frac{1}{\lambda\kappa_j}$$
>
> 这就是「零阈值」条件，也是 §3.5 中 (3.58)(3.59) 的 KKT 表述。
>
> 对 (3.79) 作拉格朗日松弛：
>
> $$\tilde L(\beta,\mu)=g(\beta)+\mu\big(h(\beta)-t\big),\qquad \mu\ge0$$
>
> 现在把 (3.78) 的 $L$ **除以** $\lambda>0$：
>
> $$\frac{1}{\lambda}L(\beta,\lambda)=\frac1\lambda h(\beta)+g(\beta)-s=\mu h(\beta)+g(\beta)-\mu s\quad\text{（取 }\mu=\tfrac1\lambda\text{）}$$
>
> 把它与 $\tilde L$ 比较：$\tilde L=\mu h+g-\mu t$，两者只差常数 $\mu(s-t)$。**对偶函数只依赖 $L$ 的非常数部分**，故两个拉格朗日函数给出**同一族**解，只要
>
> $$\mu=\frac{1}{\lambda},\qquad s\cdot\mu=1\ \ \text{（当 (3.78) 的约束紧时）},\qquad t=h(\beta^\star)=\lVert\beta^\star\rVert_1$$
>
> 第三个式子来自 (3.79) 的互补松弛 $\mu(h(\beta^\star)-t)=0$：只要 $\mu>0$ 就有 $h(\beta^\star)=t$。
>
> 所以：**$s$ 与 $t$ 是同一条 lasso 路径上的两种参数化，$\lambda$ 是它们的公共对偶乘子（互为倒数）**。凸 + Slater ⟹ 强对偶保证两个原始问题有相同最优值（见预备知识 O1）。

> **结果** · (3.78)(3.79) 的实用价值
>
> - **$s$ 有明确的几何阈值。** $g(\beta)=\lVert X^\top r\rVert_\infty\le s$ 的意思是「每个变量与响应的边际相关都被压到 $s$ 以下」。若取 $s=\max_j\lvert c_j(0)\rvert=\max_j\lvert\langle x_j,y\rangle\rvert$（即 OLS 残差下的最大 $|t|$ 型量），则 $\beta=0$ 是可行点，而任何使 $\hat\beta_j\ne0$ 的解都要付出 $\lVert\beta\rVert_1>0$，所以解就是 $\hat\beta=0$。$s$ 再大一点，$\hat\beta^{\rm ls}$ 本身可行（因为 $\kappa_jc_j$ 与 $\lambda$ 的关系在 $\lambda\to0$ 时趋于无约束），于是解跳到 $\hat\beta^{\rm ls}$。**这给出了「$\beta=0$ 何时被丢弃」的精确答案。**
> - **$t$ 的可行域有硬上界。** $\hat\beta^{\rm ls}$ 是 (3.79) 的可行点，所以必须 $t\ge\lVert\hat\beta^{\rm ls}\rVert_1$，否则可行集为空（无解）。交叉验证时搜索的 $t$ 区间必须落在这个范围内。
> - 两个约束都在 $\lambda\to0$ 时退化为 OLS，在 $\lambda\to\infty$ 时退化为 $\hat\beta=0$。

> **坑**
>
> 1. $\lVert\cdot\rVert_\infty$ 在「多个分量同时达到最大值」处不可微，$\lVert\cdot\rVert_1$ 在坐标 0 处不可微。上面的推导用次梯度处理，代价是**坐标下降需要收敛准则与步长控制**（§3.8.6）。
> 2. 两个原始问题都需要 $X$ 满列秩才有唯一解。$X$ 秩亏时 $\mathrm{null}(X)$ 里塞得下任何 $\beta$：(3.78) 的可行集含整个 $\mathrm{null}(X)\cap\{\lVert\beta\rVert_1\le s\}$，最优 $\lVert\beta\rVert_1$ 不唯一。
> 3. $s$、$t$、$\lambda$ 的倒数关系只在「约束紧」时成立。约束松时 $t$ 由最优解自己决定，不等于外加的上界。

### 3.8.3 group lasso 与自适应（稀疏性诱导）lasso {#s-3-8-3}

**问题**：两种「非均匀化」罚款——按**组**罚（group lasso）和按**重要性**加权罚（adaptive lasso）。对应 (3.80)(3.81)。

> **基础知识** · Minkowski 不等式与组内结构
>
> 欧氏空间的三角不等式 $\lVert a+b\rVert_2\le\lVert a\rVert_2+\lVert b\rVert_2$（Minkowski，见预备知识 L1）是后面全部凸性论证的根据。
> 另一个反复用到的工具是 $\lVert\beta\rVert_1$ 的分解 $\lVert\beta\rVert_1=\sum_{j=1}^{p}\big(\beta_j^++\beta_j^-\big)$（$\beta_j^+=\max(\beta_j,0)$、$\beta_j^-=\max(-\beta_j,0)$），它把「$\ell_1$ 罚」写成 $p$ 个线性约束的和，从而能用 KKT 的互补松弛逐坐标分析（见预备知识 O2）。
> **软阈值算子** $S_\theta(c)=\operatorname{sign}(c)\max(0,\lvert c\rvert-\theta)$ 满足恒等式 $c-\theta\operatorname{sign}(c)=\lambda\lambda(c/\lambda)$（$\lambda$ 是软阈值算子，$\lambda(\cdot)$ 与参数 $\lambda$ 同名时要小心），这是「lasso = 收缩算子复合于 OLS」的说法依据。


$$
J_{\rm group}(\beta)=\sqrt L\sum_{\ell=1}^{L}\big\lVert\beta^{(\ell)}\big\rVert_2\ \eqno{3.80}
$$

$\beta^{(\ell)}$ 是第 $\ell$ 组的系数子向量（如一组 dummy 变量、或一族相关变量）；前面的 $\sqrt L$ 把 $\lambda$ 放到与组大小无关的尺度上。

> **推导** · 为什么 (3.80) 是凸的
>
> 凸性就是 Minkowski（欧氏空间的三角不等式）$\lVert a+b\rVert_2\le\lVert a\rVert_2+\lVert b\rVert_2$ 的直接推论。证 $\theta\in[0,1]$：
>
> $$\begin{aligned}\big\lVert\theta\beta^{(\ell)}+(1-\theta){\beta'}^{(\ell)}\big\rVert_2&\le\theta\big\lVert\beta^{(\ell)}\big\rVert_2+(1-\theta)\big\lVert{\beta'}^{(\ell)}\big\rVert_2\\[4pt]J_{\rm group}\big(\theta\beta+(1-\theta)\beta'\big)=\sqrt L\sum_{\ell=1}^{L}\big\lVert\theta\beta^{(\ell)}+(1-\theta){\beta'}^{(\ell)}\big\rVert_2&\le\theta\,J_{\rm group}(\beta)+(1-\theta)J_{\rm group}(\beta')\end{aligned}$$
>
> 逐项不等式求和（$\sqrt L$ 是常数因子）即得凸性。
>
> **换个角度验证**（判断这类复合是否保凸的通用套路）：$J_{\rm group}(\beta)^2=L\sum_{\ell=1}^{L}\lVert\beta^{(\ell)}\rVert_2^2$。右端关于 $\beta$ 是**非负二次型之和**，故凸；映射 $t\mapsto\sqrt t$ 在 $[0,\infty)$ 上递增且凸；凸函数与「递增凸函数的复合」仍凸（第二卷式，逐点验证：$\sqrt{\theta u+(1-\theta)v}\le\theta\sqrt u+(1-\theta)\sqrt v$ 对 $u,v\ge0$ 成立，因为 $\sqrt{\cdot}$ 是凹的、且 $\min(u,v)\le\sqrt{uv}$）。故 $J_{\rm group}$ 凸。
>
> **稀疏性从哪来。** 逐坐标一阶条件（$j$ 属于第 $\ell$ 组，且 $\lVert\beta^{(\ell)}\rVert_2>0$）：
>
> $$\frac{1}{N}\big\langle x_j,\ y-\beta_0\mathbf 1-X\beta\big\rangle+\lambda\sqrt L\,\frac{\beta_j}{\big\lVert\beta^{(\ell)}\big\rVert_2}=0$$
>
> 于是组内系数**成同一比例**被收缩（收缩因子 $\lambda\sqrt L/\lVert\beta^{(\ell)}\rVert_2$ 对组内每个 $j$ 相同）。整组归零只发生在 $\lVert\beta^{(\ell)}\rVert_2=0$ 时——**group lasso 不会「半选一组」**，这与 lasso 的逐坐标选择是本质区别。

$$
\hat\beta=\arg\min_{\beta}\ \Big\{R(\beta)+\lambda_{\mathcal S}\sqrt{\textstyle\sum_{j\in\mathcal S}\mathcal N_j}\ \sum_{j\in\mathcal S}\big\lvert\beta_j\big\rvert\Big\}\ \eqno{3.81}
$$

$\mathcal S$ 是「非零系数集」，$\mathcal N_j$ 是变量 $j$ 的 $\mathcal N$ 统计量（似然比型，$\chi^2$ 量级）。

> **推导** · (3.81) 是两阶段做法
>
> 第一阶段：在某个较大的集合 $\mathcal S_0\subseteq\{1,\ldots,p\}$（全部变量，或 elastic net、相关筛选给出的集合）上跑普通 lasso。
> 第二阶段：把 $\mathcal S$ 冻结为 $\mathcal S_0$，用 $\mathcal N_j$ 作权重重跑加权 lasso。
>
> 为什么这不等价于单次 (3.76)：加权 lasso 的坐标更新是**逐坐标的软阈值，阈值不同**：
>
> $$\hat\beta_j\leftarrow S_{\lambda w_j}(\hat c_j),\qquad \hat c_j=\frac1N\langle x_j,y-X\beta\rangle$$
>
> 而权重 $w_j$ 是上一阶段估计的**数据驱动函数**，把 $w_j$ 写回 (3.76) 就得到 $J(\beta;y)$ 依赖 $y$，违反 §3.8.1 的「坑」。所以必须冻结——代价是解依赖两阶段的顺序，不再有单一凸问题。
>
> **两阶段的次序敏感吗？** 不完全不变：若第一阶段漏掉真信号（$\beta_j^\star=0$），第二阶段的 $w_j$ 大 ⟹ 门槛更高 ⟹ 更难被加回来。所以第一阶段应该「宁可多留」——这正是 ESL 推荐用 elastic net 做第一阶段的理由（见 (3.91)）。

> **结果** · 为什么 (3.81) 更「数据驱动」
>
> - 普通 lasso 用同一个阈值 $\lambda$，隐含「所有变量先验同等重要」的假设；$p\gg N$ 时假阳性很多。
> - (3.81) 让噪声变量（$\mathcal N_j$ 小 ⟹ $w_j$ 大 ⟹ 阈值 $\lambda w_j$ 高）几乎无法被选中，而信号变量的阈值接近 0、系数几乎等于 OLS 值。形式上它逼近 $\ell_0$ 罚的「只罚真零」的行为。
> - 与第 6 章的层次贝叶斯收缩、第 9 章的自适应树剪枝同源：**用一个数据驱动的先验去替换统一常数先验**。

> **坑**
>
> 1. $\sqrt L$ 只在**各组等大**时是正确归一。组大小为 $L_\ell$ 时应写成 $\sum_\ell\sqrt{L_\ell}\lVert\beta^{(\ell)}\rVert_2$，否则大组整体被更强地收缩。
> 2. $\lVert\beta\rVert_2$ 在 $\beta=0$ 处的次梯度只有 0（因为唯一最小化点是 0），这正是「整组进出」的机制；$\lVert\beta\rVert_1$ 在 0 处的次梯度是整个球，给出「逐坐标进出」。两者在 KKT 里表现为不同形式的互补松弛。
> 3. 若把 (3.81) 里的 $\mathcal S$ 取成全 $\{1,\ldots,p\}$，它退化为普通 lasso 加上一个常数因子 $\sqrt{\sum_j\mathcal N_j}$——常因子只改 $\lambda$ 的刻度，不改解。

### 3.8.4 非凸惩罚 {#s-3-8-4}

**问题**：非凸罚款能缓解 $\ell_1$ 的「过度收缩」，但优化不再是凸的。对应 (3.82)：写出分段一阶条件，说明 $\lambda$ 必须怎么求。

> **基础知识** · 凸性判别与单调路径（见预备知识 C1、C2）
>
> 判别：$f$ 凸 $\iff$ $f(\theta u+(1-\theta)v)\le\theta f(u)+(1-\theta)f(v)$（等价地，Hessian 半正定）。**罚函数的凹凸性与目标函数的凹凸性是加在一起的**：目标 $\lVert y-X\beta\rVert^2$ 严格凸，所以 (3.82) 的凸性完全由 $\lambda\sum_jJ(\lvert\beta_j\rvert)$ 决定；$J$ 凹 ⟹ 非凸。
> 单调路径：若 $J\ge0$，则 $\lambda\mapsto\min_\beta f_\lambda(\beta)$ 单调不增，且解集随 $\lambda$ 递减「变厚」；这条性质**只用 $J\ge0$**，与凸性无关（非凸时仍成立，见下面的推导）。
> 分段函数的求导：分段点处只要两段的导数相等，函数就是 $C^1$ 的，可以逐段求导再拼起来。


$$
\hat\beta=\arg\min_{\beta}\ \sum_{i=1}^{N}\Big(y_i-\beta_0-\sum_{j=1}^{p}x_{ij}\beta_j\Big)^2+\lambda\sum_{j=1}^{p}J\big(\lvert\beta_j\rvert\big)\ \eqno{3.82}
$$

$$
J\big(\lvert\beta\rvert\big)=\begin{cases}(1-a)\lvert\beta\rvert-\dfrac{a}{2}\lvert\beta\rVert^2, & \lvert\beta\rvert\le\dfrac{\lambda}{a}\\[6pt]\dfrac{1-a}{2a}\,\lambda, & \lvert\beta\rvert\ge\dfrac{\lambda}{a}\end{cases}\qquad (a\in(0,1))$$

（这就是 MCP（minimax concave penalty）；SCAD 取 $a$ 方向的极限，罚函数在两端都有平的部分。）

> **推导** · 罚函数与其导数
>
> 对第一段求导：$J'(\lvert\beta\rvert)=(1-a)-a\lvert\beta\rvert$。求导一致性核对：在拐点 $\lvert\beta\rvert=\lambda/a$ 处左边给出 $(1-a)-a\cdot\lambda/a=0$，等于右边（常数的导数 0）✓。所以 $J$ 是 $C^1$ 的（$J''$ 在拐点跳变）。
> 三个关键性质：$J(0)=0$；$J'(0)=1-a<1$（比 $\ell_1$ 的 $J'(0)=1$ 小，故**零阈值更宽**，更容易产生非零解）；$J'(\lvert\beta\rvert)=0$ 对一切 $\lvert\beta\rvert>\lambda/a$（**大系数不收缩**）。
>
> 注意非凸性来自哪里：$J''=-a<0$ 在第一段，罚函数本身是**凹**的，第二段水平，所以 $J$ 既非凸也非凹。

> **核心推导** · 分段一阶条件：$a$ 与 $\lambda$ 各自的角色
>
> 目标函数 $f(\beta)=\sum_i(y_i-\beta_0-x_i^\top\beta)^2+\lambda\sum_jJ(\lvert\beta_j\rvert)$。对 $\beta_j$ 求导，记 $\kappa_j=\lVert x_j\rVert^2$，$\hat c_j=\frac1N\langle x_j,y-X\beta\rangle$：
>
> $$-\kappa_j\hat c_j+\lambda J'(\lvert\beta_j\rvert)=0$$
>
> **三种情形：**
>
> $$\begin{cases}\text{(i)}\ \lvert\beta_j\rvert>\dfrac{\lambda}{a}: & J'=0\ \Longrightarrow\ \hat c_j=0\ \Longrightarrow\ \hat\beta_j=\dfrac{\langle x_j,y\rangle}{\kappa_j}=\hat\beta_j^{\rm ls}\quad\text{（无偏，不收缩）}\\[7pt]\text{(ii)}\ 0<\lvert\beta_j\rvert\le\dfrac{\lambda}{a}: & J'=1-a\lvert\beta_j\rvert\ \Longrightarrow\ \lvert\beta_j\rvert=\dfrac{1}{a}\Big(1-\dfrac{\kappa_j\lvert\hat c_j\rvert}{\lambda}\Big)\\[7pt]\text{(iii)}\ \beta_j=0: & \text{次梯度条件 }\lvert\kappa_j\hat c_j\rvert\le\lambda J'(0)=\lambda(1-a)\ \Longrightarrow\ \lvert\hat c_j\rvert\le\dfrac{\lambda(1-a)}{\kappa_j}\end{cases}$$
>
> 情形 (ii) 的自洽性检查：它假设 $\lvert\beta_j\rvert\in(0,\lambda/a]$，代入右端要求 $0<\kappa_j\lvert\hat c_j\rvert<\lambda(1-a)\le\lambda$，而情形 (iii) 给出 $\lvert\hat c_j\rvert\le\lambda(1-a)/\kappa_j$ ⟹ $\kappa_j\lvert\hat c_j\rvert\le\lambda(1-a)$，恰好在区间内 ✓。所以三种情形自洽地拼成完整的解。
>
> **$a$ 与 $\lambda$ 的分工：**
>
> - $\lambda$ 只控制**零点阈值**（情形 iii 的门槛 $\lambda(1-a)/\kappa_j$）；
> - $a$ 控制**拐点位置** $\lambda/a$：$a\uparrow$ ⟹ 拐点左移 ⟹ 更少系数能进入情形 (i) ⟹ 更接近 lasso；$a\to0$ ⟹ 拐点 $\to\infty$ ⟹ 所有系数都进入情形 (i) ⟹ 罚函数「完全不收缩」，逼近逐步 $\ell_0$。
>
> 这解释了 MCP/SCAD 的卖点：$\ell_1$ 的软阈值 $S_\lambda(c)=\mathrm{sign}(c)\max(0,\lvert c\rvert-\lambda)$ 在**所有尺度**上都减掉 $\lambda$，大系数也被压低（偏差大）；非凸罚让 $\lvert\beta_j\rvert>\lambda/a$ 的系数保持 $\hat\beta_j^{\rm ls}$ 不变（无偏）。

> **核心推导** · 为什么 $\lambda$ 必须用二分/下界法
>
> **单调性（可用）。** 记 $f_\lambda(\beta)=R(\beta)+\lambda\sum_jJ(\lvert\beta_j\rvert)$，注意 $J\ge0$。若 $\lambda_1\le\lambda_2$，$\beta_2\in\arg\min f_{\lambda_2}$，则对任意 $\beta_1$：
>
> $$f_{\lambda_1}(\beta_2)-f_{\lambda_1}(\beta_1)=\underbrace{\big[f_{\lambda_2}(\beta_2)-f_{\lambda_2}(\beta_1)\big]}_{\le0}-\underbrace{(\lambda_2-\lambda_1)\sum_jJ\big(\lvert\beta_{2j}\rvert\big)}_{\ge0}\ \le0$$
>
> 故 $\beta_2$ 也是 $f_{\lambda_1}$ 的极小点。**这条单调性只用到了「$\lambda$ 线性」与「$J\ge0$」，与凸性无关**，所以在非凸下依然成立。
>
> **失效的是什么。** 失效的是三条凸优化性质：
>
> - 「驻点 ⟹ 全局最优」——非凸下有多个局部极小，$\nabla_\beta f_\lambda=0$ 只给局部信息；
> - 「沿 $\lambda$ 减小路径从一个解热启动会到达正确解」——单调性只保证「**是**某个极小」，不保证「是**全局**极小」；
> - KKT 的充分性。
>
> **所以算法必须是外层二分 $\lambda$ + 内层有保护**：沿单调路径从大 $\lambda$（强收缩、问题接近好解）往下走，每一步以上一个解作为起点，并**限制在包围盒**内
>
> $$\prod_{j=1}^{p}\big[\hat\beta_j^{\star}-\Delta,\ \hat\beta_j^{\star}+\Delta\big]$$
>
> （$\Delta$ 由 (3.82) 的支撑集界给出：若当前解的支撑集为 $\mathcal S$，则更小 $\lambda$ 下的解满足 $\hat\beta_j\in[\hat\beta_j^\star-\text{界},\hat\beta_j^\star+\text{界}]$）。限制搜索区域使局部优化不能跳到「另一个局部极小」上，这才是下界法的真正作用。
>
> **二分准则。** 用「最优残差平方和」$RSS(\lambda)=\min_\beta\lVert y-X\beta\rVert^2$ 找拐点：$RSS$ 关于 $\lambda$ 单调不增，在解开始变稠密的 $\lambda$ 处曲率最大（$d^2RSS/d\lambda^2$ 变号的点）。$RSS$ 连续 ⟹ 可以直接二分；$\lvert\mathcal S\rvert$ 跳跃 ⟹ 只能在跳跃点附近取，不能对 $\lvert\mathcal S\rvert$ 直接二分。

> **结果** · 凸罚 vs 非凸罚

| | $\ell_1$（(3.78)） | 非凸（(3.82)） |
|---|---|---|
| 凸性 | 凸，KKT 充分且必要 | 非凸，驻点只给局部极小 |
| 大系数 | 恒减 $\lambda$，有偏 | $\lvert\beta_j\rvert>\lambda/a$ 时不收缩，无偏 |
| 零阈值 | $\lvert\hat c_j\rvert\le\lambda/\kappa_j$ | $\lvert\hat c_j\rvert\le\lambda(1-a)/\kappa_j$（更宽） |
| 求解 | 坐标下降 / LARS / $\ell_1$ 回归 | 二分 $\lambda$ + 热启动 + 包围盒，代价高 |
| 统计代价 | 高偏差、低方差 | 低偏差、高方差，选择更稳定但难调 |

> **坑**
>
> 1. $J$ 必须满足 $J(0)=0$、$J\ge0$。否则 (3.82) 里 $\beta=0$ 不再是自然候选点，整套「稀疏性来自零阈值」的论述失效（单调性证明也用了 $J\ge0$）。
> 2. 拐点处 $J''$ 从 $-a$ 跳到 0 ⟹ 梯度法的线性收敛率分析不适用；拟牛顿法在拐点附近也可能病态。
> 3. 交叉验证选出的非凸 $\lambda$ 往往**偏小**（非凸优化更容易在验证集上过拟合），ESL 建议乘一个略小于 1 的因子。

### 3.8.5 收缩的谱分析：ridge 为什么恰好收缩 $v_j$ 方向 {#s-3-8-5}

**问题**：ridge 和 lasso 都写成 (3.76) 的特例，但「收缩了多少」最清楚的表述来自**谱分解**。这一小节把 ridge 放到 $X^\top X$ 的特征基里，回答「每个方向的收缩因子是多少」。

> **基础知识** · 谱展开与 SVD（见预备知识 L3）
>
> $X=UDV^\top$，$X^\top X=VD^2V^\top$，故 $v_j$ 是 $X^\top X$ 的特征向量、$d_j$ 是奇异值（见预备知识 L3 的等价关系 $\lVert Xv\rVert^2=v^\top X^\top Xv=d^2\lVert v\rVert^2$）。**关键事实**：$\{v_j\}$ 是一组正交基，所以任何向量都能展开，任何矩阵运算都能逐特征向量做。

**ridge 的谱展开。** 目标函数（把常数 $\tfrac12$ 吸收进 $\lambda$）：

$$
\hat\beta_{\rm ridge}=\arg\min_{\beta}\ \Big\{\lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert^2\Big\}\ \tag{R}
$$

> **推导** · 一步到位的谱展开
>
> 展开 (R)：$\lVert y\rVert^2-2\beta^\top X^\top y+\beta^\top X^\top X\beta+\lambda\beta^\top\beta$。逐坐标分段不能做（$\beta^\top X^\top X\beta$ 混合了坐标），所以改用**正交变换** $z=V^\top\beta$，$V=(v_1,\ldots,v_p)$：
>
> $$\lVert y-X\beta\rVert^2+\lambda\lVert\beta\rVert^2=\lVert y-XVz\rVert^2+\lambda\lVert z\rVert^2$$
>
> 这里 $\lVert\beta\rVert=\lVert Vz\rVert=\lVert z\rVert$（$V$ 列正交）。再展开 $\lVert y-XVz\rVert^2=\lVert y\rVert^2-2z^\top V^\top X^\top y+z^\top V^\top X^\top XVz$，用 $X^\top X=VD^2V^\top$ 得 $V^\top X^\top XV=D^2$，故
>
> $$\lVert y\rVert^2-2\sum_{j=1}^{p}z_j\langle v_j,y\rangle+\sum_{j=1}^{p}(d_j^2+\lambda)z_j^2$$
>
> **完全可分**：目标函数 $=\lVert y\rVert^2+\sum_j\big[(d_j^2+\lambda)z_j^2-2z_j\langle v_j,y\rangle\big]$，每个 $z_j$ 只出现在自己的一项里。求导 $2(d_j^2+\lambda)z_j-2\langle v_j,y\rangle=0$ 得
>
> $$z_j=\frac{\langle v_j,y\rangle}{d_j^2+\lambda}\ \Longrightarrow\ \hat\beta_{\rm ridge}=\sum_{j=1}^{p}\frac{\langle v_j,y\rangle}{d_j^2+\lambda}\,v_j=\sum_{j=1}^{p}\frac{d_j}{d_j^2+\lambda}\,(u_j^\top y)\,v_j$$
>
> 第二个写法用 $\langle v_j,y\rangle=v_j^\top X^\top y=v_j^\top\sum_k d_ku_k y_k=d_ju_j^\top y$，它就是 §3.4.2 中 ridge 系数的逐方向写法。
>
> 写成平滑矩阵的谱形式（$A_\lambda=(X^\top X+\lambda I)^{-1}X^\top X$，见 §3.4.2）：
>
> $$\hat y_{\rm ridge}=A_\lambda y=V\,\mathrm{diag}\Big(\frac{d_j^2}{d_j^2+\lambda}\Big)V^\top y\ \Longrightarrow\ \mathrm{df}=\mathrm{tr}(A_\lambda)=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\lambda}$$
>
> 这与 §3.6.4 的 (3.60) 完全一致（$A_\lambda$ 对称），也给出一个漂亮的解释：**ridge 把有效自由度从 $p$ 压到 $\sum_jd_j^2/(d_j^2+\lambda)$，丢掉的自由度全在小 $d_j$（小方差）方向上。** $\lambda\to0$ 得 $\sum_j1=p$（OLS），$\lambda\to\infty$ 得 0（完全收缩），两者都对。

> **结果** · 收缩因子的三个读法
>
> | 写法 | 收缩沿什么 | 公式 |
> |---|---|---|
> | lasso | **坐标轴** | $\hat\beta_j=S_\lambda(\hat\beta_j^{\rm ls})$，恒减 $\lambda$ |
> | PCR (3.62) | **主成分方向** | $\hat\beta=\hat\theta_mv_m$，方向不变长度变 |
> | ridge | **任意方向等权** | 每个 $v_j$ 乘 $\frac{d_j^2}{d_j^2+\lambda}$ |
>
> 相关变量多时「坐标轴方向」不是数据的主方向，所以 lasso 在相关组上的表现由 $X$ 的条件数决定（这就是 (3.91) 加 $\ell_2$ 项的动机）。

> **坑** · (R) 中的 $\lambda$ 不是无量纲的：$\lVert y-X\beta\rVert^2$ 与 $\lVert\beta\rVert^2$ 的量纲不同（前者是 $y^2$，后者是 $y^2/x^2$）。实践中必须标准化 $x_j$，否则 (R) 隐含给不同变量不同的 $\lambda/\kappa_j$（$\kappa_j=\lVert x_j\rVert^2$），「同一个 $\lambda$」的含义就不一致了。

### 3.8.6 贝叶斯视角：先验均值不为零的 ridge {#s-3-8-6}

**问题**：§3.8.5 的 ridge 把系数往 $0$ 收缩，这等价于「先验均值是 $0$」。如果先验均值是 $\hat\beta_0\ne0$，公式会怎么变？本节把闭式解、以及「OLS 是边缘众数、ridge 是后验众数」这两个不同的量并列出来。

> **坑**
> 本节的两条公式**没有编号**。原书 (3.83)(3.84) 给的是**坐标下降**的目标函数与更新式
> $\tfrac12\sum_i\big(y_i-\beta_0-\sum_xx_{ik}\beta_k\big)^2$ 与 $\tilde\beta_j^{(j)}\leftarrow S\big(x_j^\top\tilde y^{(j)}/N,\lambda\big)$，
> 已经在 §3.6.1 里推过（见那里对 (3.83)(3.84) 的编号）。这里要回答的是另一个问题——**收缩目标不是原点时怎么办**，
> 它同时解释了两件事：$\lambda$ 的量纲（$\lambda=\sigma^2/\tau^2$），以及 §3.8.6 里 logistic 回归「只罚截距」的来历。

$$
\hat\beta=\arg\min_{\beta}\ \Big\{\lVert y-X\beta\rVert_2^2+\lambda\big\lVert\beta-\hat\beta_0\big\rVert_2^2\Big\}=\Big(X^\top X+\lambda I\Big)^{-1}\Big(X^\top y+\lambda\,\hat\beta_0\Big)
$$

$$
\hat\beta^{\rm PM}=(1-\alpha)\,\hat\beta^{\rm marg}+\alpha\,\hat\beta^{\rm post}
$$

> **基础知识** · MAP 与边缘分布的众数（见预备知识 P2）
>
> 给定先验 $p(\beta)$ 与似然 $p(y\mid\beta)$，后验众数（MAP）是 $\arg\max_\beta p(\beta\mid y)$；边缘分布 $p(y)=\int p(y\mid\beta)p(\beta)d\beta$ 的众数是 $\arg\max_y p(y)$。**两者一般不同**：前者只用到先验的一阶信息（梯度），后者要对 $\beta$ 积分。
> 多元正态的密度核：$z\sim N(\mu,\Sigma)$ 时 $\log p(z)=-\tfrac12(z-\mu)^\top\Sigma^{-1}(z-\mu)+c$，所以 MAP 就是一个加权最小二乘。

> **推导** · (3.83) 的闭式
>
> 目标函数 $F(\beta)=\lVert y-X\beta\rVert^2+\lambda\lVert\beta-\hat\beta_0\rVert^2$。两项分别求导（用 $\nabla_\beta\lVert y-X\beta\rVert^2=-2X^\top(y-X\beta)$、$\nabla_\beta\lVert\beta-\hat\beta_0\rVert^2=2(\beta-\hat\beta_0)$，见预备知识 L4）：
>
> $$\nabla_\beta F=-2X^\top(y-X\beta)+2\lambda(\beta-\hat\beta_0)=0\ \Longrightarrow\ (X^\top X+\lambda I)\beta=X^\top y+\lambda\hat\beta_0$$
>
> $X^\top X+\lambda I$ 对称正定（$X^\top X\succeq0$，$\lambda I\succ0$），故逆存在、解唯一：(3.83)。**注意收缩目标是 $\hat\beta_0$ 而不是 $0$**：由
>
> $$\hat\beta-\hat\beta_0=\Big(X^\top X+\lambda I\Big)^{-1}\Big(X^\top y+\lambda\hat\beta_0\Big)-\hat\beta_0=\Big(X^\top X+\lambda I\Big)^{-1}\Big(X^\top y-X^\top X\hat\beta_0\Big)$$
>
> （第二项减掉的是 $\hat\beta_0$ 乘进逆矩阵内，$\hat\beta_0$ 自身被消掉）可见每个系数的位移是「$\hat\beta_j^{\rm ls}-\hat\beta_{0j}$」的**逐方向衰减**。取 $\hat\beta_0=0$ 就回到 §3.8.5 的 ridge。
>
> **贝叶斯核对。** 设 $\beta\sim N(\hat\beta_0,\tau^2I)$（各系数独立、方差 $\tau^2$）、$y=X\beta+\varepsilon$，$\varepsilon\sim N(0,\sigma^2I)$。后验的核是
>
> $$p(\beta\mid y)\ \propto\ \underbrace{\exp\Big\{-\frac{\lVert y-X\beta\rVert^2}{2\sigma^2}\Big\}}_{\text{似然}}\underbrace{\exp\Big\{-\frac{\lVert\beta-\hat\beta_0\rVert^2}{2\tau^2}\Big\}}_{\text{先验}}$$
>
> 取负对数（丢掉常数），得到 (3.83) 的目标函数，其中
>
> $$\lambda=\frac{\sigma^2}{\tau^2}$$
>
> **这一步解释了 $\lambda$ 的真正含义**：它是「噪声方差 ÷ 先验方差」。$\tau\to\infty$（无信息先验）⟹ $\lambda\to0$ ⟹ 不收缩；$\tau\to0$（先验极紧）⟹ $\lambda\to\infty$ ⟹ $\hat\beta\to\hat\beta_0$，由 (3.83) 直接验证（$\lambda\to\infty$ 时 $(X^\top X+\lambda I)^{-1}\approx\lambda^{-1}I$，右端 $\approx\lambda^{-1}(\lambda\hat\beta_0)=\hat\beta_0$）。
>
> **推广到「带惩罚的截距」。** 在设计矩阵里加一列 $\mathbf 1$ 并只对它加 $\ell_2$ 罚，正是一次 (3.83) 的应用（$\hat\beta_0$ 取 0、罚只作用在截距那一列）。这正是第 4 章 logistic 回归里的 $\beta_0^2\boldsymbol 1^\top\beta_0/2$（罚截距、不罚斜率）的来历，也是 §3.8.6 联合惩罚公式里的那一项。

> **推导** · (3.84)：两个不同的「众数」
>
> **边缘众数。** 先验取平坦（$\tau\to\infty$），边缘密度
>
> $$p(y)\propto\int\exp\Big\{-\frac{\lVert y-X\beta\rVert^2}{2\sigma^2}\Big\}d\beta\ \propto\ \exp\Big\{-\frac{1}{2\sigma^2}\Big[\lVert y\rVert^2-\underset{\beta}{\min}\,\lVert y-X\beta\rVert^2\Big]\Big\}$$
>
> （积分是高斯积分，被积函数的指数在最小值处最大；积分结果仍是一个 $y$ 的二次型，只依赖残差平方和。）对这个 $y$ 取极大 ⟹ 等价于对 $\beta$ 取 $\lVert y-X\beta\rVert^2$ 的极小 ⟹
>
> $$\hat\beta^{\rm marg}=\arg\min_\beta\lVert y-X\beta\rVert^2=(X^\top X)^{-1}X^\top y=\hat\beta^{\rm ls}$$
>
> **后验众数。** 有限 $\tau$ 下（3.83）给出 $\hat\beta^{\rm post}=\hat\beta_{\rm ridge}$。**这两个量不同**：$\hat\beta^{\rm marg}$ 无偏但方差大，$\hat\beta^{\rm post}$ 有偏但方差小。所以「把系数往 $0$ 收缩」不是边缘似然的要求，而是**先验**引入的——shrinkage 的本质是「用一点偏差换方差」。
>
> **插值。** (3.84) 把两者连成一条线段：$\alpha\in[0,1]$，$\alpha=0$ 给出边缘众数（OLS），$\alpha=1$ 给出后验众数（ridge）。中间的 $\alpha$ 都可以写成 (3.83) 的形式（把 $\lambda$ 取成对应的值），因为 $(1-\alpha)\hat\beta^{\rm ls}+\alpha\hat\beta^{\rm post}$ 在 $\hat\beta_0=0$ 时正是 $(X^\top X+\lambda I)^{-1}X^\top y$ 对某个 $\lambda$ 取值的结果（把 $\hat\beta^{\rm ls}$、$\hat\beta^{\rm post}$ 都写成 $X^\top y$ 的线性函数，插值后仍是 $X^\top y$ 的线性函数，对称正定系数矩阵 ⟹ 必是某个 $(X^\top X+\lambda I)^{-1}$ 的形式）。

> **结果** · (3.83)(3.84) 的三条用途
>
> - **给 $\lambda$ 一个可解释的默认值**：由边际似然（EM 或直接数值优化）估出 $\hat\sigma^2$、再用交叉验证估 $\tau^2$，就得到 $\hat\lambda=\hat\sigma^2/\hat\tau^2$。这就是「ridge 的 $\lambda$ 不必从 0 开始网格搜索」的做法。
> - **混合估计 (3.84) 的实用价值**：当 $\hat\beta^{\rm marg}$ 有已知偏差时，插值 $\hat\beta^{\rm PM}$ 提供一族「偏差-方差可选」的估计，取 $\alpha$ 由验证集选。
> - **与 §3.8.6 的联系**：(3.84) 的 $\alpha$ 与 elastic net (3.91) 的 $\lambda_2/(\lambda_1+\lambda_2)$ 是同一族「混合比例」；前者混合 OLS 与 ridge，后者混合 lasso 与 ridge。

> **坑**
>
> 1. (3.83) 要求 $X^\top X+\lambda I$ 可逆 ⟹ $\lambda>0$ 即可，不需要 $X$ 满秩。这与 (3.6) 的 $(X^\top X)^{-1}$ 是本质区别（ridge 在 $p\ge N$ 时仍有定义）。
> 2. 先验假设了**各系数独立**（$\tau^2I$）。若先验有结构（如 (3.80) 的组先验、$p\ge N$ 时的 $\beta\sim N(0,\tau^2(X^\top X)^{-1})$ 正态-Jeffreys 先验），(3.83) 要改成 $X^\top X+\lambda\Sigma_\beta^{-1}$ 的形式——这也是 (3.72) 里 $\hat\Lambda$ 不是对角元时仍可解的原因。
> 3. 「边缘众数 = OLS」的推导里积分与取 $\arg\min$ 的交换依赖高斯结构；换成 logistic 似然（§3.8.6）后边缘众数不再是 IRLS 的不动点，(3.84) 的插值解释失效。

### 3.8.7 centered lasso：为什么普通 lasso 的约束集会造假 {#s-3-8-7}

**问题**：$\ell_1$ 球 $\{\beta:\lVert\beta\rVert_1\le t\}$ 是**多面体**，它的极端点都在坐标轴上。当两个预测子高度相关时，最小二乘解 $\hat\beta^{\rm ls}$ 在「几乎垂直于坐标轴」的平面上，两个系数一正一负、大小相近；$\ell_1$ 球到这个区域的最近点却可能落在**某个坐标轴**上——这就是假稀疏。ESL 的修正见 (3.85)。

> **基础知识** · 多面体上的最近点
>
> lasso 的一阶条件是「$\hat\beta^{\rm ls}-\hat\beta$ 的各分量按 KKT 分成三类」（非零的等于 $\pm\lambda/\kappa_j$、零的 $\lvert\cdot\rvert\le\lambda/\kappa_j$）。在约束面 $\lVert\beta\rVert_1=t$ 上，KKT 要求
>
> $$\sum_{j\in\mathcal A}\big(\hat\beta_j^{\rm ls}-\beta_j\big)\big(\beta_j-\tilde\beta_j\big)+\big(\hat\beta^{\rm ls}-\tilde\beta\big)^\top(\beta-\tilde\beta)=0\quad\text{对一切可行 }\beta$$
>
> （$-\tilde\beta$ 是最小范数点时的结论，一般情形要逐个约束集求）。这条「法向条件」说明：**最优点的坐标符号模式不能任意**——只有当 $\hat\beta^{\rm ls}$ 落在某个「符号一致的分片」的边界上时，才会出现只激活一个坐标的解。

$$
\hat\beta_c=\arg\min_{\beta}\ \Big\{\sum_{i=1}^{N}\big(y_i-\beta_0-\sum_{j=1}^{p}(x_{ij}-\bar x_j)\beta_j\big)^2+\lambda\lVert\beta\rVert_1\Big\}\ \eqno{3.85}
$$

**与普通 lasso 的唯一区别：把 $x_j$ 换成 $x_j-\bar x_j$。**

> **推导** · 中心化改变的两件事
>
> 记 $\tilde x_j=x_j-\bar x_j\mathbf 1$，$X_c=X-\mathbf 1\bar x^\top$。
>
> **第一件：截距与斜率解耦。** 由 $\langle\mathbf 1,\tilde x_j\rangle=\sum_i(x_{ij}-\bar x_j)=0$ 可知
>
> $$\frac{\partial R}{\partial\beta_0}=-2\sum_{i=1}^{N}\big(y_i-\beta_0-\sum_j\tilde x_{ij}\beta_j\big)=-2\Big(\sum_iy_i-N\beta_0-\sum_j\beta_j\sum_i\tilde x_{ij}\Big)=-2N(\bar y-\beta_0)$$
>
> 于是一阶条件直接给出 $\hat\beta_0=\bar y$。**中心化后截距就是样本均值，回归线的中心过 $(\bar x,\bar y)$。**
>
> **第二件：协方差矩阵换了。** $\tilde X^\top\tilde X=X^\top X-N\bar x\bar x^\top$。主成分方向一般会变；被移除的正是「常数方向」$N\bar x\bar x^\top$ 对谱的贡献。
>
> **为什么中心化修掉假稀疏。** 假稀疏的根源是**约束集的极端结构**：$\lVert\beta\rVert_1\le t$ 的顶点是 $\{\pm te_j\}$。中心化前，$(\beta_0,\beta)$ 的可行集是「$\beta$ 在多面体内、$\beta_0$ 自由」的柱体，其极端方向仍由 $e_j$ 垄断，所以最优解有系统倾向只用一个坐标。中心化后：
>
> - 可行集退回成纯 $\ell_1$ 球（$\beta_0$ 被 $\bar y$ 固定，不再是自由的第二个方向）；
> - **同时目标点也换了**：$\hat\beta^{\rm ls}_c=(\tilde X^\top\tilde X)^{-1}\tilde X^\top\tilde y$。
>
> 关键的差别在第二点。中心化前的「假稀疏」例子：两个预测子近乎相同，$\hat\beta^{\rm ls}=(c,-c)^\top$。此时
>
> $$\langle (c,-c),\,(1,-1)^\top\rangle=2c>0,\qquad \langle (c,-c),\,(1,0)^\top\rangle=c,\qquad \langle (c,-c),\,(1,1)^\top\rangle=0$$
>
> 三个内积比较说明：$\hat\beta^{\rm ls}$ 与 $(1,-1)^\top$ 方向几乎垂直（这正是两个预测子共线时的情形：系数的对比方向 $\beta_1-\beta_2$ 被数据约束住），所以它与「单坐标激活方向 $(1,0)^\top$」的内积 $c$ 很小 ⟹ KKT 允许坐标 $2$ 归零，**这就是假稀疏的机制**。
>
> 中心化后为什么不同？因为 $\tilde x_2=\tilde x_1+\delta$（$\delta$ 是中心化后两列的差），$\lVert\delta\rVert$ 通常不大但**非零**。这使得 $\tilde X^\top\tilde X$ 的小特征值方向不再被 $\mathbf 1$ 污染，于是对比方向 $(1,-1)^\top$ 得到的权重变了，$\hat\beta^{\rm ls}_c$ 不再「垂直于 $(1,1)^\top$」。
>
> **可以手算验证的三点核对**（这是唯一可靠的验证方式）：
>
> 1. **核对 1**：$\hat\beta^{\rm ls}_c$ 与普通 lasso 的 $\hat\beta^{\rm ls}$ 数值不同（若 $x_1,x_2$ 未中心化则前者不可直接比较）；
> 2. **核对 2**：若两个预测子**完全**共线（$\tilde x_2=c\tilde x_1$），则 $\tilde X^\top\tilde X$ 秩亏，$\hat\beta^{\rm ls}_c$ 不唯一。此时 lasso 的 KKT 解在 $\mathrm{null}(\tilde X)$ 上靠 $\ell_1$ 罚挑出唯一解，而中心化的作用是保证 $\mathrm{null}(\tilde X)$ 里不再混进常数方向；
> 3. **核对 3**：中心化 + lasso 对「$x_j$ 本身接近常数」的处理与 ridge 不同——该列中心化后 $\tilde x_j\approx0$，$d_j\to0$，ridge 把它丢掉，lasso 把它保留。要不要保留是一个建模决定，不是算法的副产品。

> **结果** · centered lasso 的三条性质
>
> - **在中心化后的几何里，$\ell_1$ 球不再有「只用一个坐标」的结构性偏好**：目标点 $\hat\beta^{\rm ls}_c$ 一般落在 $\lVert\beta\rVert_1=t$ 约束面的「分片内部」而不是顶点上。
> - **与罚函数选择正交**：把 (3.85) 的 $\lVert\beta\rVert_1$ 换成 $\lVert\beta\rVert_2^2$ 就得到 ridge；换成 group 罚就得配合中心化的组内变量。所以「中心化」改数据表示，「罚函数」改复杂度控制，两者可自由组合。
> - **标准化 $\ne$ 中心化**：标准化（除以 $\lVert x_j\rVert_2$ 或标准差）改**尺度**、中心化改**原点**。做系数比较（哪个变量影响大）时两者都要做；只中心化会让 $\lVert x_j\rVert$ 不同的变量之间不可比。

> **坑**
>
> 1. 中心化改变了截距的解释：$\beta_j$ 只能解释成「$x_j$ 增加一个单位、其它变量不动时 $y$ 的期望变化」，**且这条解释只在 $x_j$ 已中心化时不含原点信息**。
> 2. $N<p$ 时 (3.6) 的 $\hat\beta^{\rm ls}$ 不存在，(3.85) 依然有定义（$\ell_1$ 罚提供唯一性），但两者的 KKT 推导要改用「$\beta$ 的最小范数取值」约定。
> 3. (3.85) 里的 $\lambda$ 与 §3.8.5 的坑是同一条：$\kappa_j=\lVert\tilde x_j\rVert^2$ 随中心化改变，所以同一 $\lambda$ 在中心化前后对应的阈值 $\lambda/\kappa_j$ 不同。

### 3.8.8 收缩的 SVD 视角：广义特征值问题 {#s-3-8-8}

**问题**：PCR、CCA、lasso 都可以看成「在某个方向上找一个标量」。把所有这些方向上的「最优标量」放在一起，会得到一个统一的**广义特征值问题**，而它的解有 SVD 形式的闭式。对应 (3.86)(3.87)。

> **基础知识** · Cauchy–Schwarz 与 SVD（见预备知识 L1、L3）
>
> $|\langle a,b\rangle|\le\lVert a\rVert\lVert b\rVert$，等号当且仅当 $a,b$ 平行。由此 $\max_{\lVert u\rVert=\lVert v\rVert=1}u^\top Mv=\lVert M\rVert_2=\sigma_1$，等号在 $u,v$ 为主奇异向量时取到。

$$
\max_{u,v}\ \frac{u^\top\big(Y^\top X\big)v}{\sqrt{u^\top u}\sqrt{v^\top X^\top Xv}}\ \eqno{3.86}
$$

> **推导** · (3.86) 就是「方向问题」，把三个方法认出来
>
> 记 $u\in\mathbb R^{K}$（响应侧）、$v\in\mathbb R^{p}$（预测子侧），$Y^\top X\in\mathbb R^{K\times p}$。三个应用：
>
> - **PCA / PCR**：取 $u=v_j$（第 $j$ 个主成分方向，$X$ 空间），分子 $v_j^\top X^\top Y u$；分母 $\sqrt{v_j^\top X^\top Xv_j}=\sqrt{v_j^\top V D^2V^\top v_j}=d_j$（因为 $V^\top v_j=e_j$）。**分母正好把 $X$ 侧的尺度归一化掉**，剩下的是「$u$ 在 $Y$ 侧怎么选才与第 $j$ 个主成分最对齐」。
> - **CCA**：$u^\top Y^\top Xv=u^\top S_{YX}v$，分母 $\sqrt{u^\top Y^\top Yu}\,\sqrt{v^\top X^\top Xv}$ 就是 $\sqrt{u^\top S_{YY}u}\sqrt{v^\top S_{XX}v}$，与 §3.7.3 的 (3.67) **逐字相同**。所以 (3.86) 就是 CCA 的最大化问题（那里写的是平方，这里取根号，等价）。
> - **lasso**：限制 $v=e_j$，分子 $e_j^\top Y^\top Xu=\langle x_j,Yu\rangle$，分母 $\sqrt{\lVert x_j\rVert^2}\cdot\sqrt{1}$（当 $u$ 单位化）。取最大 ⟹ **lasso 只在坐标轴上找最优方向**——这正是 §3.8.5 说的「lasso 沿坐标轴收缩」的另一种说法。
>
> **求解 (3.86)。** 第一步：用 Cauchy–Schwarz 把分母归一化。记 $w=Xv$，则 $v^\top X^\top Xv=\lVert w\rVert^2$，目标变成
>
> $$\max_{\lVert u\rVert=1,\ \lVert w\rVert=1}\ u^\top Y^\top w$$
>
> 显式核对归一化正确性（这一步不能跳）：
>
> $$\big(u^\top Y^\top w\big)^2\le\lVert u\rVert^2\,\lVert Y^\top w\rVert^2=\lVert u\rVert^2\,w^\top YY^\top w$$
>
> 分母的平方是 $\lVert u\rVert^2\cdot\lVert w\rVert^2$，所以商 $\frac{(u^\top Y^\top w)^2}{\lVert u\rVert^2\lVert w\rVert^2}$ 就是两个中心化视图的 $\mathrm{Corr}^2$。**两个 $\sqrt{\cdot}$ 把归一化完全吸收了**，剩下标准的最大奇异值问题。
>
> 第二步：SVD $Y^\top X=U_\star D_\star V_\star^\top$，代入得 $u^\top Y^\top w=u^\top U_\star D_\star V_\star^\top w$。两次 Cauchy–Schwarz（或直接「$M$ 的算范数是最大奇异值」）给出 $\max=u^\top u_{\star,1}d_1$ 的上界 $d_1$，等号在 $u\parallel u_{\star,1}$、$w\parallel v_{\star,1}$。回代 $w=Xv$：

$$
\big(Y^\top Y\big)^{-1/2}\big(Y^\top X\big)\big(X^\top X\big)^{-1/2}=U_\star D_\star V_\star^\top,\qquad v_\star^{\star}=\big(X^\top X\big)^{-1/2}V_\star u_\star\ \eqno{3.87}
$$

> **推导** · (3.87) 的两个归一化因子在做什么
>
> 记 $M:=(Y^\top Y)^{-1/2}(Y^\top X)(X^\top X)^{-1/2}$。
>
> - $(Y^\top Y)^{-1/2}$ 是**响应侧的白化**（whitening）矩阵：它把 $Y$ 的各方向等方差化，于是响应方向之间不再因尺度不同而互相压制；
> - $(X^\top X)^{-1/2}$ 是**预测子侧的白化**矩阵，同理。
>
> 白化后 $M$ 是**奇异值分解的标准形式**。为什么必须用奇异值而不是特征值：$M$ 一般**不对称**（两侧的归一化矩阵不同），特征值可能为复数、且 $\max_\lambda$ 与 $\max\sigma$ 不同；奇异值总是非负实，而且 (3.86) 的目标函数有界（分母 $\ne0$）。
>
> **数值验证 1（PCA 特例）。** 若 $Y=X$，则
>
> $$M=(X^\top X)^{-1/2}\,X^\top X\,(X^\top X)^{-1/2}=(X^\top X)^{1/2}=V D V^\top$$
>
> 于是 $U_\star=V$、$D_\star=D$、$V_\star=V$，$v_\star^\star=(X^\top X)^{-1/2}Vv_{\star,1}=d_1v_{\star,1}$——即第 $j$ 个主成分得分系数。**白化版本在 $Y=X$ 时精确退化为主成分分析。**
>
> **数值验证 2（特征值核对）。** $M^\top M=V_\star D_\star^2V_\star^\top$，故 $M^\top M$ 的特征值恰是 $d_k^2$。另一方面 $\max_\lambda\lambda(M^\top M)=\sigma_1(M)^2$（最大奇异值之平方），这是 $\max_{\lVert u\rVert=\lVert w\rVert=1}(u^\top Mw)^2$ 的 Rayleigh 商形式，可由 Rayleigh 商定理直接验证。
>
> **核对 3（与 CCA 一致）。** $v_\star^\star=V(X^\top X)^{1/2}u_\star$，代入 $Y^\top Xv_\star^\star=U_\star D_\star V^\top V(X^\top X)^{1/2}u_\star=U_\star D_\star(X^\top X)u_\star$。再由 $M^\top Mu_\star=d_1^2u_\star$ 反推：$u_\star\propto(Y^\top Y)^{-1/2}Y^\top Xv_\star^\star$。这正是 §3.7.3 的 (3.67a)（取 $S_{YY}=Y^\top Y$、$S_{XX}=X^\top X$）。**SVD 解与 CCA 的特征值问题给出同一个方向。**

> **结果** · 一个问题，三种限制
>
> $$\max_{u,v}\frac{u^\top(Y^\top X)v}{\sqrt{u^\top u}\sqrt{v^\top X^\top Xv}}\quad\xrightarrow{\ \text{限制 }u\text{ 或 }v\ }\begin{cases}u\text{ 限制在 }\operatorname{span}\{e_1,\ldots,e_m\}\ (Y\text{ 的坐标轴})\ \text{（PLS/PCR 型）}\\ v\text{ 限制在 }\operatorname{col}(X)\ (\text{全部})\ \text{（无约束，即 CCA）}\\ v\text{ 限制在 }\{e_j\}\ \text{（lasso）}\end{cases}$$
>
> (3.87) 说明 **CCA 是无约束版本，其它方法是加了约束的版本**。第 14 章的 Krylov–Schur 类截断 SVD 算法正是利用这个统一性，一次分解同时服务所有降维回归。

> **坑**
>
> 1. $(X^\top X)^{-1/2}$ 要求 $X^\top X$ 正定（$X$ 满列秩）。$p\ge N$ 时 (3.87) 直接无定义，必须改用截断逆（预备知识 L3）。
> 2. SVD 的符号不唯一（$u\to\pm u$ 同时改 $v\to\mp v$），所以 (3.87) 只确定**方向**。任何用到系数符号的地方（$\mathrm{sign}$、LARS 的单调方向）都要先固定符号约定。
> 3. $v\in\mathrm{null}(X)$ 时分母为 0，而分子也为 0（$Xv=0$），形成 $0/0$。必须显式要求 $v\notin\mathrm{null}(X)$。

### 3.8.9 惩罚参数的取样选择：为什么「重采样 $\lambda$」是错的 {#s-3-8-9}

**问题**：$\hat\sigma$（噪声标准差）和 $\hat\beta_j(\lambda)$（带罚估计）都来自同一份数据。用 bootstrap 重采样 $\hat\beta_j$ 再估标准误，等于把噪声估两遍。对应 (3.88)。

> **基础知识** · 误差传播与自助法（见预备知识 P5）
>
> 若 $\hat\theta=g(Z_1,\ldots,Z_k)$ 且各 $Z_j$ 近似独立，$\mathrm{Var}(\hat\theta)\approx\sum_j\big(\partial g/\partial Z_j\big)^2\mathrm{Var}(Z_j)$（delta 方法，见预备知识 P5）。**这条式子的前提是独立**；不独立时必须用全协方差矩阵：
>
> $$\mathrm{Var}(\hat\theta)\approx\big(\nabla g\big)^\top\,\widehat{\mathrm{Cov}}(Z)\,\nabla g$$
>
> bootstrap 的逻辑正是「用经验协方差替换 $\mathrm{Cov}(Z)$」。所以 bootstrap 估计标准误的**唯一要求是：重采样时要保持 $Z$ 之间的相关结构**（有放回抽整行、或抽噪声再传播）。


$$
\sqrt{1-\alpha}\ \hat\sigma\ \lambda(\alpha)\ \text{（对固定的 }\hat\beta_j>0\text{）},\qquad \lambda(\alpha)\sim N\big(0,\hat\sigma^2\big)\ \text{由重采样得到（这是错的）}\ \eqno{3.88}
$$

**正确做法**：用 Monte Carlo 直接重采样 $\hat\beta$ 的**分布**（噪声传播），而不是重采样 $\lambda$。

> **推导** · (3.88) 错在哪
>
> **正确的模拟流程：**
>
> 1. 固定当前的 $\lambda$（这一步是关键的「冻结」）；
> 2. 从 $y_i=\beta_0+\sum_jx_{ij}\beta_j+\varepsilon_i$ 抽 $\varepsilon_i^{\star}\sim N(0,\hat\sigma^2)$，造 $y^{\star}=X\hat\beta+\varepsilon^{\star}$；
> 3. 重算 $\hat\beta^{\star}=X^\dagger X^\top(X\beta+\varepsilon^{\star})=X^\dagger X^\top X\beta+X^\dagger X^\top\varepsilon^{\star}$。对 lasso 则要把步骤 3 换成「在 $y^\star$ 上重跑 lasso」；
> 4. 重复 $B$ 次，得 $\{\hat\beta^{\star}_j\}_{b=1}^B$，$\widehat{\mathrm{se}}(\hat\beta_j)=\mathrm{sd}(\hat\beta^{\star}_j)$。
>
> 关键在步骤 2 用了两次 $\hat\sigma$ 吗？没有：$\hat\sigma$ 只用来**生成** $\varepsilon^\star$，而 $\hat\beta^\star$ 的抽样分布由 $\varepsilon^\star$ 的随机性决定，$\hat\sigma$ 本身不再进入。所以这不构成「双重计数」。
>
> **为什么「重采样 $\lambda$」错。** 记 $\hat\sigma=\hat\sigma(\hat\beta_{\rm ls})=\mathrm{sd}(\hat\varepsilon_{\rm ls})$，而 $\hat\beta_j(\lambda)=\hat\beta_j^{\rm ls}-\lambda\operatorname{sign}(\hat\beta_j)$（lasso）、$\hat\beta(\lambda)=(X^\top X+\lambda I)^{-1}X^\top y$（ridge）。若令 $\lambda^{\star}\sim N(0,\hat\sigma^2)$ 独立抽样，则
>
> $$\hat\beta_j^{\star}=\hat\beta_j^{\rm ls}-\lambda^{\star}\operatorname{sign}(\hat\beta_j)\ \Longrightarrow\ \mathrm{Var}(\hat\beta_j^{\star})=\underbrace{\mathrm{Var}(\hat\beta_j^{\rm ls})}_{\text{真实的那一份}}+\underbrace{\hat\sigma^2}_{\text{被多加了一次}}$$
>
> 而真实的 $\mathrm{Var}(\hat\beta_j)$ 是 $\mathrm{Var}(\hat\beta_j^{\rm ls})\cdot w_j^2$（$w_j$ 是收缩因子，$\le1$）。所以这个估计把方差**放大了**（至少加了一份完整的噪声方差），是**高估**。
>
> 更精细的说法：$\hat\sigma$ 与 $\lambda$ **不独立**——$\hat\sigma$ 由「选定 $\lambda$ 下的残差」估出，$\hat\sigma^2=\mathrm{RSS}/(N-\mathrm{df})$，而 $\mathrm{df}$ 又依赖 $\lambda$（§3.6.4 的 (3.60)）。所以用同一个 $\hat\sigma$ 既生成 $\lambda^\star$ 又参与生成 $\hat\beta^{\star}$，会把两条依赖链混成一条，得到有偏的方差估计。
>
> **$1-\alpha$ 因子从哪来。** 对固定的 $\hat\beta_j>0$，由一阶 Taylor 展开 $\hat\beta_j(\lambda+\delta)-\hat\beta_j(\lambda)\approx\frac{\partial\hat\beta_j}{\partial\lambda}\delta$，而
>
> $$\frac{\partial\hat\beta_j}{\partial\lambda}\ \text{（ridge）}=-\big[(X^\top X+\lambda I)^{-1}X^\top y\big]_j^{'}\cdot\ldots=-\big[(X^\top X+\lambda I)^{-1}\big]_{jj}\ \text{（在 }X^\top X\approx\text{对角且 }\|y\|^2\approx N\sigma^2\text{ 时）}$$
>
> 更常用的近似是「局部秩 1 + 单位信号」假设下的 $\big[(X^\top X+\lambda I)^{-1}\big]_{jj}\approx\frac{1}{d_j^2+\lambda}$，于是
>
> $$\mathrm{se}\big(\hat\beta_j(\lambda)\big)\approx\hat\sigma\sqrt{\big[(X^\top X+\lambda I)^{-1}\big]_{jj}}\ \text{，对 }\hat\beta_j=\text{固定}\ \Longrightarrow\ \hat\sigma\sqrt{\frac{1}{d_j^2+\lambda}}\sim\frac{\hat\sigma}{d_j}\ \text{当 }d_j\gg\lambda$$
>
> (3.88) 的 $\sqrt{1-\alpha}\hat\sigma\lambda(\alpha)$ 是把「$\lambda$ 的不确定度 $\hat\sigma\lambda$」与「$\lambda$ 本身」并排的一种参数化：$\alpha$ 调的是「多大的 $\lambda$ 扰动被当成噪声」。(3.88) 后半句那个「$\lambda$ 由重采样得到」的做法**就是**把 $\lambda$ 当成随 $\hat\sigma$ 波动的量去做误差传播——这是错的，因为 $\lambda$ 是**被我们选定的超参**，不是估计量。
>
> **正确的做法（参数自助法）。** 把 $\lambda$ 也当估计量，但**在正确的层次上重采样**：
>
> 1. 拟合出 $\hat\beta_{\rm ls}$，估 $\hat\sigma$；
> 2. 生成 $y^{\star}=X\hat\beta_{\rm ls}+\hat\sigma\,\varepsilon^{\star}$；
> 3. 在 $y^{\star}$ 上重跑整个流程，包括「选 $\lambda$」（例如用交叉验证）；
> 4. 得到 $\hat\beta^{\star}_j$，用它的 sd 估标准误。
>
> 这样 $\lambda$ 的不确定性被正确地包含了（因为 $\lambda$ 是在模拟数据里重新选的）。

> **结果** · 三种做法的区别
>
> | 做法 | $\lambda$ 怎么处理 | 得到的方差 | 是否含 $\lambda$ 的不确定性 |
> |---|---|---|---|
> | (3.88) 的错误做法 | 当成 $N(0,\hat\sigma^2)$ 随机变量 | 偏大（多加一份 $\hat\sigma^2$） | 形式上是，实质是错的 |
> | 固定 $\lambda$ 重采样噪声 | 固定 | 偏小（不含 $\lambda$ 的不确定性） | 否 |
> | 参数自助法（正确） | 在每份模拟数据里重新选 | 正确 | 是 |

> **坑** · $\hat\sigma$ 和 $\hat\beta_j$ 来自同一份残差，所以「系数绝对值大的变量标准误也小」这个反向依赖会削弱 $t$ 统计量。第 10 章的分裂准则有同一个问题，必须小心。

### 3.8.10 非二次损失的 lasso：logistic 情形 {#s-3-8-10}

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-8">原文 §3.8.6 非二次损失</a>

**问题**：$R$ 不是平方损失时（分类），lasso 还剩什么？关键是它还能写成 (3.78) 的约束形式，从而用同样的 KKT 推出软阈值结构。对应 (3.89)。

> **基础知识** · 次梯度与凸性（见预备知识 O2、C1）
>
> 在不可微点（如 $\lvert\beta\rVert_1$ 在 $\beta_j=0$ 处）用**次梯度**代替梯度：$f$ 在 $x_0$ 可微 ⟺ 次梯度集是单点 $\{\nabla f(x_0)\}$。$\lVert\beta\rVert_1$ 在 $\beta_j=0$ 处的次梯度集是 $[-1,1]$（其余坐标是 $\operatorname{sign}$），这是「零阈值条件 $\lvert p_j\rvert\le\mu$」的来源。
>
> 凸性：$L(y,\eta)$ 凸 ⟹ $R(\beta_0,\beta)=\sum_iL(y_i,\beta_0+x_i^\top\beta)$ 凸（凸函数的正权和），前提是 $x_i$ 取遍的集合不过大（无约束时 $\sum_i$ 严格保证）。
>
> 强对偶：凸 $+$ Slater $\Rightarrow$ KKT 必要充分（见预备知识 O1、O2），所以「两个互补松弛条件」不是近似而是等价刻画。


$$
\hat\beta=\arg\min_{\beta}\ \sum_{i=1}^{N}\log\big(1+e^{-y_i(\beta_0+x_i^\top\beta)}\big)\quad\text{s.t.}\quad \lVert\beta\rVert_1\le t\ \eqno{3.89}
$$

> **推导** · 为什么「$L_1$ 约束 + 凸 $R$」一定给出软阈值
>
> **(i) 凸性。** 对数损失 $L(y,\eta)=\log(1+e^{-y\eta})$：
>
> $$\frac{\partial L}{\partial\eta}=-\frac{y}{1+e^{y\eta}},\qquad \frac{\partial^2L}{\partial\eta^2}=\frac{y^2e^{y\eta}}{\big(1+e^{y\eta}\big)^2}\ge0$$
>
> **凸**（对一切 $y\ne0,\eta$ 严格正）。所以 (3.89) 是凸优化 ⟹ KKT 必要充分、强对偶（见预备知识 O1、O2）。
>
> **(ii) 一阶条件与两个互补松弛。** 记 $R(\beta_0,\beta)=\sum_iL(y_i,\eta_i)$，$p_j=\partial R/\partial\beta_j$。KKT 的平稳性给出（$\beta_j\ne0$ 时）
>
> $$p_j+\mu\operatorname{sign}(\beta_j)=0$$
>
> 而 $\beta_j=0$ 时用次梯度条件（$\partial\lVert\beta\rVert_1/\partial\beta_j$ 在 0 处可取 $[-1,1]$ 中任一值）：
>
> $$\lvert p_j\rvert\le\mu$$
>
> 把 $\lVert\beta\rVert_1=\sum_{j=1}^{p}\big(\beta_j^++\beta_j^-\big)$（$\beta_j^\pm$ 是 $\beta_j$ 的正负部）、约束写成 $-\sum_j(\beta_j^++\beta_j^-)\le-t$，拉格朗日函数是
>
> $$L=R+\mu\Big(-\sum_{j=1}^{p}\beta_j^+-\sum_{j=1}^{p}\beta_j^- - t\Big)$$
>
> 两个约束各自有乘子：$-\beta_j^+\le0$ 的乘子 $\mu_j^+\ge0$、$-\beta_j^-\le0$ 的乘子 $\mu_j^-\ge0$，加上 $\lVert\beta\rVert_1\le t$ 的乘子 $\mu\ge0$。KKT 四条（见预备知识 O2）中的两条关键条件是**互补松弛**：
>
> $$\boxed{\mu_j^+\beta_j^+=0,\qquad \mu_j^-\beta_j^-=0}$$
>
> 它们给出的信息是：
>
> - **符号**：$\mu_j^+>0\Rightarrow\beta_j^+=0$（该系数的正部被压掉）；$\mu_j^->0\Rightarrow\beta_j^-=0$（负部被压掉）；
> - **稀疏**：$\mu_j^+=\mu_j^-=\mu>0$（两个乘子都饱和）⟹ $\beta_j^+=\beta_j^-=0$ ⟹ **$\beta_j=0$**。
>
> 由平稳性 $\mu_j^++\mu_j^-=\mu$（当 $\beta_j\ne0$ 时）和 $\mu_j^++\mu_j^-\le\mu$（当 $\beta_j=0$ 时），加上互补松弛，可以解出
>
> $$\hat\beta_j=\operatorname{sign}(p_j)\max(0,\lvert p_j\rvert-\mu)=:S_\mu(p_j)$$
>
> **因此「哪些系数被选中」的信息精确地落在「两个乘子同时饱和」这个事件上**——这就是非平方损失下稀疏性的完整机制，与平方损失下的 KKT（§3.5 的 (3.58)(3.59)）是同一件事。
>
> **(iii) 与平方损失的关系。** 平方损失下 $p_j=-\frac1N\langle x_j,y-X\beta\rangle$（线性部分残差）；对数损失下
>
> $$p_j=\frac{\partial R}{\partial\beta_j}=\sum_{i=1}^{N}\frac{\partial L}{\partial\eta_i}\cdot x_{ij}=-\sum_{i=1}^{N}\frac{y_ix_{ij}}{1+e^{y_i\eta_i}}$$
>
> 定义**响应权重** $w_i:=\frac{1}{1+e^{y_i\eta_i}}\in(0,1)$，则 $p_j=-\sum_iw_iy_ix_{ij}$。**这正是 IRLS 的加权最小二乘**（加权版「部分残差」）。所以 lasso 的软阈值结构**不变**，只是「残差」被换成了「加权残差」。这给出 glmnet / IRLS-lasso 的算法：每轮迭代固定 $w_i$，套用一次加权 lasso 的闭式。

> **结果** · 「$L_1$ 约束 + 凸 $R$」的一般定理
>
> **定理。** 设 $R(\beta_0,\beta)$ 关于 $(\beta_0,\beta)$ 可微且凸，且 (3.89) 的可行集 $\{(\beta_0,\beta):\lVert\beta\rVert_1\le t\}$ 是紧集。则最优解满足
>
> $$\hat\beta_j=S_\mu\big(p_j\big)\quad\forall j,\qquad p_j=\frac{\partial R}{\partial\beta_j}\bigg|_{(\hat\beta_0,\hat\beta)}$$
>
> **证明**（三行）：凸 ⟹ KKT 必要充分（O2）。KKT 的平稳性 + 互补松弛 $\mu_j^+\beta_j^+=0$、$\mu_j^-\beta_j^-=0$ 给出 $\mu_j^\pm$ 的取值只可能是 $(0,\mu)$、$(\mu,0)$、$(\mu,\mu)$ 三种（当 $\mu>0$），分别对应 $\beta_j<0$、$\beta_j>0$、$\beta_j=0$。代回平稳性 $\mu_j^+-\mu_j^-=-p_j$ 解出 $\lvert\beta_j\rvert=\lvert p_j\rvert-\mu$（在 $\lvert p_j\rvert>\mu$ 时）。∎
>
> **为什么这个定理重要**：它说 $\ell_1$ 的稀疏性与「损失是二次」**完全无关**，只与「$J=\lVert\beta\rVert_1$」和「$R$ 凸」有关。第 12 章的 SVM（$R$ 是铰链损失，凸）、第 4 章的逻辑回归（$R$ 是对数损失，凸）都能直接套。

> **坑**
>
> 1. $y_i$ 必须取 $\pm1$；$y_i\in\{0,1\}$ 时要换成 $\log(1+e^{\eta_i})$ 或 $-\log p(y_i)$。
> 2. 凸但**不严格**凸（$\lVert\beta\rVert_1\le t$ 的顶点上 $R$ 沿某些方向平坦）时最优解不唯一。
> 3. IRLS 每轮固定 $w_i$，但 $w_i$ 依赖当前的 $\eta_i$ ⟹ 整体不是单个凸问题，必须迭代到收敛（每次迭代目标下降，所以收敛）。

### 3.8.11 LARS 轨迹、elastic net 与方法对比 {#s-3-8-11}

**问题**：LARS 之所以快，是因为整条 lasso 路径是**分段线性**的；elastic net 之所以有用，是因为它改掉了「只沿坐标轴收缩」这件事。最后把 (3.90)(3.91) 写清楚，并给出收缩类方法的统一比较。

> **基础知识** · 线性方程组的参数敏感性（见预备知识 C2）
>
> $Az=b(\lambda)$ 在 $A$ 可逆时给出 $\mathrm{d}z/\mathrm{d}\lambda=-A^{-1}\mathrm{d}b/\mathrm{d}\lambda$，对 $\lambda$ 线性。要件只有两个：$A$ **可逆**、$b$ 对 $\lambda$ **线性**。lasso 的 KKT 在支撑集固定时恰好把 $b$ 写成 $X_{\mathcal A}^\top y-N\lambda\mathrm{s}_{\mathcal A}$（对 $\lambda$ 线性），$A=X_{\mathcal A}^\top X_{\mathcal A}$；elastic net 的 $A$ 变成 $X_{\mathcal A}^\top X_{\mathcal A}+2N\lambda_2I$。**这就是 (3.90) 与 (3.91) 分段线性的唯一根据。**
>
> 对照：$\ell_0$ 罚的「支撑集」$\mathcal A$ 一变，$A$ 就换成另一个矩阵，且 $b$ 不再对 $\lambda$ 线性（罚是常数 1），所以最佳子集路径**不是**分段线性，也没有像 (3.90) 这样的公式。


$$
\hat\beta(\lambda)=\hat\beta(\lambda_0)-\big(\lambda-\lambda_0\big)\gamma_0\ \eqno{3.90}
$$

$\hat\beta(\lambda_0)$ 是路径上的一个已知解（在 $\lambda_0$ 处的支撑集为 $\mathcal A$），$\gamma_0$ 是「$\hat\beta$ 对 $\lambda$ 的斜率」向量（只依赖设计矩阵，与 $y$ 无关）。

> **推导** · (3.90) 为什么成立
>
> 在支撑集 $\mathcal A$ 固定、符号固定的**开区间** $(\lambda_a,\lambda_b)\subset(\lambda_0,\infty)$ 内，lasso 的 KKT 平稳性（对 $j\in\mathcal A$，用 §3.8.6 的形式，$\mu=\lambda$）是
>
> $$-\frac1N X_{\mathcal A}^\top\big(y-X_{\mathcal A}\hat\beta_{\mathcal A}\big)+\lambda\,\operatorname{sign}(\hat\beta_{\mathcal A})=0$$
>
> 整理成关于 $\hat\beta_{\mathcal A}$ 的**线性**方程（$\operatorname{sign}$ 在固定符号下是常数向量 $\mathrm{s}_{\mathcal A}$，$\lambda$ 线性）：
>
> $$X_{\mathcal A}^\top X_{\mathcal A}\hat\beta_{\mathcal A}=X_{\mathcal A}^\top y-N\lambda\,\mathrm{s}_{\mathcal A}$$
>
> 对 $\lambda$ 求导（$X_{\mathcal A}^\top X_{\mathcal A}$ 在此区间内可逆，否则区间要切得更短）：
>
> $$X_{\mathcal A}^\top X_{\mathcal A}\,\frac{d\hat\beta_{\mathcal A}}{d\lambda}=-N\,\mathrm{s}_{\mathcal A}\ \Longrightarrow\ \gamma_{0,\mathcal A}:=-N\,\big(X_{\mathcal A}^\top X_{\mathcal A}\big)^{-1}\mathrm{s}_{\mathcal A}$$
>
> 从 $\lambda_a$ 积到 $\lambda_b$：
>
> $$\hat\beta_{\mathcal A}(\lambda_b)=\hat\beta_{\mathcal A}(\lambda_a)-N(\lambda_b-\lambda_a)\big(X_{\mathcal A}^\top X_{\mathcal A}\big)^{-1}\mathrm{s}_{\mathcal A}$$
>
> 取 $\lambda_a=\lambda_0$、$\lambda_b=\lambda$，就得到 (3.90)（$\gamma_0$ 的 $\mathcal A$ 分量为上式，$\mathcal A$ 外分量为 0）。**逐坐标写**：
>
> $$\hat\beta_j(\lambda)=\hat\beta_j(\lambda_0)+\gamma_j(\lambda-\lambda_0)\quad(j\in\mathcal A),\qquad \hat\beta_j(\lambda)=0\quad(j\notin\mathcal A)$$
>
> **这解释了 LARS 轨迹的三个几何性质（图 3.11）：**
>
> 1. **分段线性**：每个区间上 $\hat\beta(\lambda)$ 是 $\lambda$ 的仿射函数；
> 2. **单调性**：若 $x_j$ 与当前所有激活变量的 $\hat x_j^{\star}=\sum_{k\in\mathcal A}\hat\beta_kx_k$ 内积为正，则 $\gamma_j<0$，$\lambda$ 减小时 $\hat\beta_j$ 增大（且始终为正）。这是 LARS 的单调性定理（Efron–Murtagh–Tibshirani）；
> 3. **节点是「相关对」的进入点**：$\lambda$ 减到某个值时，某个未激活变量 $j$ 满足 $\lvert\langle x_j,y-X_{\mathcal A}\hat\beta_{\mathcal A}\rangle\rvert=N\lambda\lVert x_j\rVert^2$（KKT 的零阈值条件），它被加入 $\mathcal A$，路径进入新的一段。
>
> **为什么这让 lasso 可行。** 扫过整条路径只需 $O(p)$ 次坐标更新（每段一次），而不是为每个 $\lambda$ 从零开始解一次（$O(p)$ 次迭代）。对比最佳子集的 $2^p$：**这是「lasso = 可行的子集选择」的算法根据**。

$$
\hat\beta=\arg\min_{\beta}\ \Big\{R(\beta)+\lambda_1\lVert\beta\rVert_1+\lambda_2\lVert\beta\rVert_2^2\Big\}\ \eqno{3.91}
$$

> **推导** · elastic net = lasso + ridge 的正确混合方式
>
> 朴素混合（$\hat\beta_{\rm ridge}\cdot\frac{\hat\beta_{\rm lasso}}{\hat\beta_{\rm ridge}}$，逐坐标相乘）在数学上无意义。正确做法是直接写 (3.91) 并求解。
>
> **一阶条件。** 对 $j$：
>
> $$-\frac1N\big\langle x_j,\ y-\beta_0\mathbf 1-X\beta\big\rangle+\lambda_2\beta_j+\lambda_1\partial\lVert\beta_j\rVert=0$$
>
> 记 $\kappa_j=\lVert x_j\rVert^2$、$\hat c_j=\frac1N\langle x_j,y-X\beta\rangle$，逐情形解：
>
> $$\beta_j\leftarrow\frac{S_{\lambda_1/\kappa_j}\big(\hat c_j\big)}{1+2\lambda_2/\kappa_j},\qquad S_\theta(c)=\operatorname{sign}(c)\max(0,\lvert c\rvert-\theta)$$
>
> （把 $\lVert\beta\rVert_2^2=\sum_j\beta_j^2$ 逐坐标展开故得 $\lambda_2\beta_j$；$\lambda_1\partial\lVert\beta\rVert$ 的解就是软阈值。）**分子是 lasso 的软阈值、分母是 ridge 的收缩**——这就是 elastic net 的全部内容。
>
> **为什么它与 (3.90) 兼容。** 固定 $\mathcal A$ 与符号后，(3.91) 的 KKT 是
>
> $$X_{\mathcal A}^\top X_{\mathcal A}\hat\beta_{\mathcal A}+2N\lambda_2\hat\beta_{\mathcal A}=X_{\mathcal A}^\top y-N\lambda_1\,\mathrm{s}_{\mathcal A}$$
>
> 对 $\lambda_1$ 线性 ⟹ **整条 elastic net 路径也分段线性**，斜率
>
> $$\gamma_0=-N\big(X_{\mathcal A}^\top X_{\mathcal A}+2N\lambda_2I\big)^{-1}\mathrm{s}_{\mathcal A}$$
>
> 注意分母多了 $2N\lambda_2I$：这是与 lasso 的唯一差别。
>
> **联合惩罚（logistic + elastic net）推导。** 把 $R$ 换成对数损失（§3.8.6）并做 IRLS：
>
> $$\hat\beta^{(t)}=\arg\min_{\beta}\ \Big\{-\sum_{i=1}^{N}w_i^{(t)}\log\big(1+e^{-y_i(\beta_0+x_i^\top\beta)}\big)+\lambda_1\lVert\beta\rVert_1+\lambda_2\lVert\beta\rVert_2^2\Big\},\qquad w_i^{(t)}=\frac{1}{1+e^{y_ix_i^\top\beta^{(t-1)}}}$$
>
> 加权后 $\frac{\partial^2L}{\partial\eta^2}=w_ie^{y_i\eta}/(1+e^{y_i\eta})^2\ge0$（$w_i>0$）⟹ 仍是凸。**内层是凸问题**，可套用 (3.91) 的逐坐标解（把 $X$ 换成 $W^{1/2}X$、$y$ 换成 $W^{1/2}y$，$W=\mathrm{diag}(w_1,\ldots,w_N)$）与 (3.90) 的分段线性路径。
>
> **为什么 elastic net 解决「成组效应」。** 两个近乎重复的预测子 $x_2\approx cx_1$：
>
> - lasso：$\hat\beta^{\rm ls}=(c,-c)^\top$ 沿对比方向，$\ell_1$ 球在这个方向上的支撑点是顶点 ⟹ 随机挑一个保留，另一个归零，**符号/大小比被破坏**；
> - elastic net：$\ell_2$ 项提供「在对比方向上的支撑是光滑的球面而非顶点」，于是最优解沿 $(1,1)^\top$ 方向移动，两个系数**成比例保留**。
>
> 数值验证：取 $\lambda_1=0$，则 (3.91) 退化为 ridge，$\hat\beta\propto(X^\top X+2N\lambda_2I)^{-1}X^\top y$，两个系数正比于 $X^\top y$ 在 $(x_1,x_1)$ 与 $(cx_1,cx_1)$ 方向上的分量之比，即成组保留 ✓。取 $\lambda_2=0,\lambda_1>0$，退化为 lasso，随机挑一个 ✓。取 $\lambda_1,\lambda_2>0$，插在两者之间 ✓。

> **结果** · 收缩 vs 选择：计算量的对比

| | 最佳子集（$\ell_0$） | lasso（$\ell_1$） | elastic net | ridge |
|---|---|---|---|---|
| 搜索复杂度 | $2^p$ | $O(p)$（沿路径扫一遍） | $O(p)$ | $O(p)$ |
| 解的形式 | 无闭式 | 分段线性软阈值 | 软阈值 $+$ 二次收缩 | 谱形式 |
| 稀疏 | 是（精确） | 是（精确） | 是（精确） | 否 |
| 相关变量 | **不稳定**（任意挑一个） | 任意挑一个（成组效应被破坏） | 成组保留（$\lambda_2>0$） | 全部保留、均匀收缩 |
| 需要 $N\ge p$ | 是 | 否（KKT/坐标下降可行） | 否 | 否 |
| 选择一致性 | 是（$p$ 小时） | 是（信号稀疏 + irreps 条件） | 是 | 否（$\lVert\hat\beta\rVert_2^2$ 永不为 0） |

**两个关键对比数字**：

1. **搜索量**：最佳子集要评 $2^p$ 个子集（$p=20$ 已经 $10^6$、$p=40$ 不可行）；lasso 只需沿路径走 $p$ 次坐标更新（$p=10^6$ 也能做，glmnet 就是这么干的）。这是「lasso = 可行的子集选择」的算法根据。
2. **成组效应**：相关系数 $\rho\to1$ 时，$\hat\beta^{\rm ridge}_1/\hat\beta^{\rm ridge}_2\to1$（都保留），而 $\hat\beta^{\rm lasso}$ 只留一个。elastic net 取 $\lambda_2>0$ 后比值趋于 $1$，即**近似于「先对该组做单变量回归再按比例分配」**。

> **坑**
>
> 1. (3.90) 的分段线性只在 $\mathcal A$ 不变、$\operatorname{sign}$ 不变的区间内成立。断点处斜率跳变，插值跨断点会给出错误的 $\hat\beta$。
> 2. elastic net 的两个参数含义不同：$\lambda_1$ 控**稀疏度**（几组系数非零），$\lambda_2$ 控**组内收缩**。软件常只暴露一个混合参数（如 glmnet 的 $\alpha=\lambda_2/(\lambda_1+\lambda_2)$），这会隐含「组大小」的先验，改 $\alpha$ 时稀疏度与成组效应同时变。
> 3. lasso 的选择一致性需要「信号稀疏 $+$ 信号与噪声条件独立」（irrepresentable condition）。$p\gg N$ 时若有相关噪声项，lasso 会**乱选**而不是全不选——这是 §3.8.3 引入 (3.81)、(3.91) 的根本动机。

### 3.8.12 本章小结 {#s-3-8-12}

| 方法 | $R(\beta)$ | $J(\beta)$ | 方向由谁定 | 稀疏 |
|---|---|---|---|---|
| 最小二乘 (3.6) | $\lVert y-X\beta\rVert^2$ | 无 | $X^\top X$（全部 $p$ 个） | 否 |
| ridge | 同上 | $\lVert\beta\rVert_2^2$ | $X^\top X$（每个 $v_j$ 等权收缩） | 否 |
| lasso (3.78) | 同上 | $\lVert\beta\rVert_1$ | 坐标轴 | 是 |
| adaptive lasso (3.81) | 同上 | $\sum_j\mathcal N_j\lvert\beta_j\rvert$ | 坐标轴 $+$ 先验 | 是 |
| group lasso (3.80) | 同上 | $\sum_\ell\lVert\beta^{(\ell)}\rVert_2$ | 组 | 组级 |
| elastic net (3.91) | 同上 | $\lambda_1\lVert\beta\rVert_1+\lambda_2\lVert\beta\rVert_2^2$ | 坐标轴 $+$ 等权 | 是 |
| 非凸罚 (3.82) | 同上 | 有界斜率分段线性 | 坐标轴 | 是 |
| PCR (3.62) | $\lVert y-\sum_{m\le m}\hat\theta_mz_m\rVert^2$ | 只准 $m$ 个分量 | $X^\top X$（前 $m$ 个） | 否 |
| PLS (3.71) | 交替最小化残差 | 只准 $m$ 个得分 | $X$ 与 $Y$（$=$ CCA，见 (3.67c)） | 否 |
| CCA | $\mathrm{Corr}^2$ 最大化 | 同上 | $S_{YY}^{-1}S_{YX}S_{XX}^{-1}S_{XY}$ | 否 |
| logistic lasso (3.89) | $\sum_i\log(1+e^{-y_ix_i^\top\beta})$ | $\lVert\beta\rVert_1$ | 坐标轴 | 是 |

**三条把整节串起来的话：**

1. **一切都是「沿某些方向收缩」。** 方向只有两个来源：$X$ 自己的协方差（PCR、ridge 的谱形式）、$X$ 与 $Y$ 的关系（CCA、PLS、lasso 的坐标轴）。(3.86)(3.87) 证明这些选择可以统一成一个 SVD。
2. **凸性买来三样东西**：KKT 的充分性（不用怕局部极小）、解的唯一性、沿 $\lambda$ 的单调路径 $+$ 分段线性（(3.90)）。丢了凸性（(3.82)）就得用二分 $+$ 下界法重新买回来。
3. **稀疏性只有一个来源：把 $\beta$ 逼到约束集的极端结构上。** $\ell_1$ 球（极端点在坐标轴）、$\ell_0$ 罚（极端结构 = 子集）、MCP/SCAD 的零点阈值，都是这件事的三种写法；group lasso 换成「逼到组的极端结构」；$\ell_2$ 罚与 PCA 没有极端结构，所以永远不稀疏。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-8">原文 §3.8</a>

**练习**：下面三道题对应的推导框架本节都已给出，需要你自己补齐中间式。

1. **(3.82) 非凸罚的分段一阶条件。** 取 $N=1$、$x_1=1$、$\beta_0=0$、$y=1$。目标函数是 $\tfrac12(1-\beta)^2+\lambda J(\lvert\beta\rvert)$，$J$ 取 MCP 且 $a=\tfrac12$。画出 $\beta$ 关于 $\lambda$ 的曲线，找出「$\beta=0$ 与 $\beta\ne0$ 局部极小点共存」的临界 $\lambda^\star$，并验证 §3.8.4 单调性证明里的不等式在什么时刻取等号（提示：$\sum_jJ(\lvert\beta_{2j}\rvert)=0$ 当且仅当 $\beta_2=0$）。
2. **(3.87) SVD 解的手算。** 取 $N=4$，$X,Y$ 都是 $4\times2$，$X^\top X=\begin{pmatrix}4&1\\1&2\end{pmatrix}$，$Y^\top Y=\begin{pmatrix}3&0.5\\0.5&1\end{pmatrix}$，$Y^\top X=\begin{pmatrix}2&0\\1&0.5\end{pmatrix}$。逐项算出 $(Y^\top Y)^{-1/2}$、$(X^\top X)^{-1/2}$、$M$、$M^\top M$ 的特征值与特征向量，验证 $v_\star^{\star}=(X^\top X)^{-1/2}V_\star u_\star$ 满足 Cauchy–Schwarz 的等号条件 $u_\star\parallel Y^\top Xv_\star^{\star}$，并核对 §3.7.3 的 (3.67c) 在 $S_{YY}=Y^\top Y$、$S_{XX}=X^\top X$ 下给出同一个 $u_\star$。
3. **(3.91) elastic net 与 logistic 的联合惩罚。** 取 $N=6$、$p=2$，$X$ 的一列是常数、另一列是 $\{-2,-1,0,1,2,3\}$（记得中心化后重算 $\kappa_j$），$y\in\{\pm1\}^6$。手算对数损失 $+$ elastic net 的一阶条件，验证逐坐标解 $\beta_j\leftarrow S_{\lambda_1/\kappa_j}(\hat c_j)/(1+2\lambda_2/\kappa_j)$ 与「在 $p=2$ 的格点上枚举最小值」给出同一个解；再解释为什么 $\kappa_j$ 必须用中心化后的 $\tilde x_j$ 重算（这直接关系到 §3.8.6 的「标准化 $\ne$ 中心化」那条坑）。