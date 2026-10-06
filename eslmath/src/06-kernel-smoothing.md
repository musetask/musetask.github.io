---
id: m06
n: "6"
title: 核平滑方法
title_en: Kernel Smoothing Methods
desc: 核密度估计的偏差、Nadaraya–Watson 的偏差–方差权衡、局部线性与局部多项式、局部似然与时序平滑
prev: m05
next: m07
prev_title: 第 5 章 基展开与正则化
next_title: 第 7 章 模型评估与选择
---

# 6 核平滑方法 {#s-6}

核平滑的核心想法只有一句：**在每个查询点 $x_0$ 处，只用靠近它的那些观测，在一个小邻域里重新拟合一个简单模型**。这个简单模型可以是常数（局部常数、最近邻平均）、直线（局部线性）、多项式（局部多项式），也可以是任何一个参数模型（局部似然）。邻域的宽度由核 $K_\lambda(x_0,x)$ 控制，$\lambda$ 是本章唯一需要从数据里调出的参数。

本章的推导围绕一条主线展开：把「加权平均」写成对 $f$ 的核加权积分，然后把核在 $x_0$ 处作 Taylor 展开，看**偏差是 $h^2$ 阶、方差是 $h^{-1}$ 阶**（一维）这件事怎么来的；由此得到最优带宽 $h_{\mathrm{opt}}\sim n^{-1/5}$，再对核函数本身用 Cauchy–Schwarz 优化，就得到 Epanechnikov 核。第二条主线是局部回归：把加权最小二乘写成一阶条件，推出等效核（equivalent kernel）$l_i(x_0)$，并证明局部线性在二阶意义下无偏——这就是「自动核改造」（automatic kernel carpentry）的严格含义。

## 6.1 一维核平滑器 {#s-6-1}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-1">原文 §6.1</a>

第 2 章里我们用 $k$ 近邻平均估计回归函数 $E(Y\mid X=x)$：

$$
\hat f(x)=\mathrm{Ave}\big(y_i\mid x_i\in N_k(x)\big) \eqno{6.1}
$$

$N_k(x)$ 是距 $x$ 平方距离最近的 $k$ 个点构成的集合，$\mathrm{Ave}$ 表示算术平均。它的逻辑是把条件期望的定义「放松」为邻域上的平均。

问题在图 6.1 左图：$\hat f(x)$ 关于 $x$ 是**跳跃**的。把 $x_0$ 从左往右移动，最近邻集合保持不变，直到右边某个 $x_i$ 越过左边某个 $x_i'$，$\hat f$ 就发生一次跳变。这个跳跃没有任何统计含义，纯粹是「平均」这个离散操作带来的。

自然的修补是：**不给邻域内的点同等权重，而让权重随距离平滑地衰减到 0**。这样拟合出的 $\hat f(x)$ 就连续了，并且在紧支撑核的情形下还多阶可微。得到的估计量叫 Nadaraya–Watson 核加权平均 (6.2)：

$$
\hat f(x)=\frac{\displaystyle\sum_{i=1}^N K_\lambda(x_0,x_i)\,y_i}{\displaystyle\sum_{i=1}^N K_\lambda(x_0,x_i)} \eqno{6.2}
$$

注意 (6.2) 只是 (6.1) 的「软化」：最近邻对应 $K\equiv\mathbf 1$（盒核），此时分母恰为 $k$。所以本章可以看成第 2 章那一行公式的连续化版本，而所有的偏差–方差分析都是在问同一个问题：**软化到什么程度最合适**。

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-2">原文 §6.2</a>

## 6.2 核宽度的选择与偏差–方差权衡 {#s-6-2}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-2">原文 §6.2</a>

### 6.2.1 核的定义与三个实用细节

核写成「一个固定形状 $D$ + 一个宽度 $\lambda$」的形式 (6.3)：

$$
K_\lambda(x_0,x)=D\!\left(\frac{\lvert x-x_0\rvert}{\lambda}\right) \eqno{6.3}
$$

原书用的 Epanechnikov 二次核 (6.4) 是

$$
D(t)=\begin{cases}\tfrac34\,(1-t^2), & \lvert t\rvert\le 1,\\[2pt] 0, & \text{otherwise}\end{cases} \eqno{6.4}
$$

验证它是一个核（密度）：$\int_{-1}^1 \tfrac34(1-t^2)dt=\tfrac34(2-\tfrac23)=\tfrac34\cdot\tfrac43=1$，且 $\int t D(t)dt=0$（对称）。

另一个常用的紧支撑核是 tri-cube (6.6)：

$$
D(t)=\begin{cases}(1-\lvert t\rvert^3)^3, & \lvert t\rvert\le 1,\\[2pt] 0, & \text{otherwise}\end{cases} \eqno{6.6}
$$

它在支撑边界处**两阶连续可微**（Epanechnikov 在边界处一跳，不连续），尾部更平坦（像盒核），所以 LOESS 默认用它。第三个是 Gauss 核 $D=\varphi$，无界支撑，$\lambda$ 是标准差。三者的对比见原文图 6.2。

> **坑** · 三个实用细节
> - $\lambda$ 要选：窄窗 → $\hat f(x_0)$ 只平均很少的 $y_i$，方差接近单个 $y_i$ 的方差，但偏差小；宽窗 → 方差小，偏差大（用到的 $f(x_i)$ 未必接近 $f(x_0)$）。
> - 度量窗宽（$h_\lambda(x)$ 常数）保持偏差为常数、方差反比于局部密度；最近邻窗宽反过来，方差为常数、偏差反比于局部密度。
> - $x_i$ 有并列值时：把并列的 $y_i$ 先平均成一个观测，再给它一个总权重 $w_i$，最后在加权平均前把 $w_i$ 乘进核权重。

需要让窗宽随 $x_0$ 变化（自适应，如最近邻）时，把 $\lambda$ 换成一个宽度函数 $h_\lambda(x_0)$ (6.5)：

$$
K_\lambda(x_0,x)=D\!\left(\frac{\lvert x-x_0\rvert}{h_\lambda(x_0)}\right) \eqno{6.5}
$$

(6.3) 里 $h_\lambda(x_0)=\lambda$；对 $k$ 近邻，$h_k(x_0)=\lvert x_0-x_{[k]}\rvert$，$x_{[k]}$ 是第 $k$ 近的 $x_i$。带观测权重的版本要求邻域总权重精确等于 $k$：若第 $k$ 个点的权重 $w_j$ 会让总权重超过 $k$，允许用小数份（fractional part）计入。

### 6.2.2 核密度估计的偏差：完整推导

现在做本章最核心的一次计算。设 $x_1,\dots,x_N\overset{\mathrm{iid}}{\sim}f$，在 $x_0$ 处用 Parzen 估计

$$
\hat f_X(x_0)=\frac1N\sum_{i=1}^N K_\lambda(x_0,x_i)
$$

> **基础知识**
> - Taylor 展开（含余项）：$f(x_0+u)=f(x_0)+f'(x_0)u+\tfrac12f''(x_0)u^2+o(u^2)$，只要 $f$ 二阶连续可微（见预备知识 C1）。
> - 换元积分与卷积恒等式（见预备知识 C3）：
> $$\int h(u)K(x_0-x-u)du=\int h(x_0-x-v)K(v)dv$$
> - 核的矩：$\mu_0(K)=\int K=1$，$\mu_1(K)=\int uK(u)du$，$\mu_2(K)=\int u^2K(u)du$；对奇核 $\mu_1=0$。

**第一步：算 $E[\hat f_X(x_0)]$。** 逐项取期望，再把 Taylor 展开代进去：

$$
E[\hat f_X(x_0)]=\frac1N\sum_{i=1}^N E\!\left[K_\lambda(x_0,x_i)\right]=\frac1N\sum_{i=1}^N\int K_\lambda(x_0,x)\,f(x)\,dx
$$

（因为 $E[K_\lambda(x_0,x_i)]=\int K_\lambda(x_0,x)f(x)dx$，用到 $x_i$ 的密度就是 $f$。）

把 $f$ 在 $x_0$ 处展开，其中 $u=x-x_0$：

$$
\int K_\lambda(x_0,x)f(x)dx=\int K_\lambda(x_0,x)f(x_0)dx+\int K_\lambda(x_0,x)(x-x_0)f'(x_0)dx+\frac12\int K_\lambda(x_0,x)(x-x_0)^2f''(x_0)dx+o(h^2)
$$

**第二步：逐项化简这三个积分**——这里正是换元积分起作用的地方。令 $u=(x-x_0)/\lambda$，$x-x_0=\lambda u$，$dx=\lambda du$：

1. 一阶项（$\lambda^2$）：
$$\int K_\lambda(x_0,x)(x-x_0)dx=\lambda^2\int u\,K(u)du=\lambda^2\mu_1(K)=0\ \ (\text{$K$ 为偶函数})$$

2. 二阶项（$\lambda^3$）：
$$\int K_\lambda(x_0,x)(x-x_0)^2dx=\lambda^3\int u^2K(u)du=\lambda^3\mu_2(K)$$

3. 常数项：$\int K_\lambda(x_0,x)dx=\int K(u)du=\mu_0(K)=1$。

（对自适应带宽 (6.5)，第三式变成 $h_\lambda(x_0)\int K= h_\lambda(x_0)$，一阶项变成 $h_\lambda(x_0)^2\mu_1(K)$——这就是「边界处核不对称导致偏差」的来源，见 §6.3.5。）

**第三步：汇总。** 把上面三式代回第二步的展开式。余项 $o(u^2)$ 乘上 $\int K_\lambda dx=\lambda$ 后是 $o(\lambda^3)$，比 $\lambda^2$ 项高阶，可以丢掉：

$$
E[\hat f_X(x_0)]-f(x_0)=\frac{\lambda^2}{2}\mu_2(K)f''(x_0)+o(\lambda^2)
$$

> **结果** · 核密度估计的偏差是 $O(\lambda^2)$，系数为 $\tfrac12\mu_2(K)f''(x_0)$。只要 $f''$ 有界，$\lambda\to0$ 时偏差自动消失，这叫**一致收敛**。

### 6.2.3 矩匹配带宽与 $h_{\mathrm{opt}}$

再算方差。各项 $K_\lambda(x_0,x_i)/N$ 独立，故（见预备知识 P1 的方差可加性）

$$
\mathrm{Var}[\hat f_X(x_0)]=\frac1{N^2}\sum_{i=1}^N\mathrm{Var}\big[K_\lambda(x_0,x_i)\big]=\frac1N\mathrm{Var}\big[K_\lambda(x_0,x_1)\big]
$$

$$=\frac1N\left(\int K_\lambda^2f-\left(\int K_\lambda f\right)^2\right)=\frac{\lambda^{-1}}{N}\int K(u)^2f(x_0+\lambda u)du-\frac{\lambda^{-1}}{N}\big(\ldots\big)^2=O(\lambda^{-1})$$

即标准结论 $\mathrm{Var}\approx\frac{\lambda^{-1}}{N}\int K^2f(x_0)$，量级 $\lambda^{-1}$。于是

$$
\mathrm{MSE}(\lambda)\approx\frac{\lambda^4}{4}\mu_2(K)^2f''(x_0)^2+\frac{\lambda^{-1}}{N}\int u^2K(u)^2f(x_0)\,du
$$

把 $f(x_0+\lambda u)\to f(x_0)$ 当作常数，第二项写成 $\lambda^{-1}\hat R(K)f(x_0)/N$，$\hat R(K)=\int u^2K^2$。对 $\lambda$ 求导并令其为零（见预备知识 C1）：

$$
\frac{\partial\mathrm{MSE}}{\partial\lambda}=\lambda^3\mu_2(K)^2f''(x_0)^2-\frac{\sigma^2\hat R(K)}{N\lambda^2}=0\ \Longrightarrow\ \lambda^5=\frac{\sigma^2\hat R(K)}{N\mu_2(K)^2f''(x_0)^2}
$$

这里省略了 $f''(x_0)$（要用差商或局部二次回归估计），实践中它被吸收进带宽常数。$\hat R(K)=\int u^2K(u)^2du$ 可直接算出：Epanechnikov 为 $\frac9{16}\int_{-1}^1(u^2-2u^4+u^6)du=\frac9{16}\cdot\frac{16}{105}=\frac3{35}\approx0.0857$；Gauss 为 $\frac1{2\pi}\int u^2e^{-u^2}du=\frac1{4\sqrt{\pi}}\approx0.1411$。核形状因此通过组合量 $\hat R(K)/\mu_2(K)^2$ 进入：Epanechnikov 为 $\frac{3/35}{1/25}=\frac{15}{7}\approx2.14$，Gauss 为 $\frac{1}{4\sqrt\pi}\approx0.141$。指数 $1/5$ 是关键：它来自「$4$ 阶偏差项 vs $(-1)$ 阶方差项」的平衡。

> **结果** · 一维最优带宽的速率是 $N^{-1/5}$，$\sigma^2$ 只影响常数不影响速率。这就是为什么核平滑的带宽对样本量不敏感：$N$ 涨 100 倍，$h$ 只缩 2.5 倍。

### 6.2.4 高维时带宽的缩放

$d$ 维情形 (6.24) 用乘积 Gauss 核：

$$
\hat f_X(x_0)=\frac1N\sum_{i=1}^N\prod_{k=1}^d\varphi_\lambda\!\left(\frac{x_{0k}-x_{ik}}{\lambda}\right)=\frac{1}{N\lambda^d}\sum_{i=1}^N K_\lambda^d(x_0,x_i)
$$

同样的 Taylor 展开给出偏差仍是 $O(\lambda^2)$（只有二阶导参与），但方差变成 $O(\lambda^{-d})$，因为一个半径 $\lambda$ 的球里有效样本量是 $\propto N\lambda^d$：

$$
\mathrm{MSE}(\lambda)\approx\frac{\lambda^4}{4}\mu_2(K)^2f''(x_0)^2+\frac{\sigma^2}{N\lambda^d\int K^2}
$$

严格求导：$\lambda^3\mu_2^2f''^2=\dfrac{d\,\sigma^2}{N\int K^2}\lambda^{-d-1}$，即

$$
\lambda^{d+5}=\frac{d\,\sigma^2}{N\mu_2(K)^2f''(x_0)^2\int K^2}\ \Longrightarrow\ \lambda\asymp N^{-1/(d+5)}
$$

文献（包括原书关于维数的讨论）常写成 $N^{-1/(d+4)}$。差别只来自一个常数因子：$4\cdot\frac{\lambda^4}{4}$ 的 $4$ 若在「两项比例平衡」时被丢掉，就得到 $\lambda^4\propto\lambda^{-d}$，即 $N^{-1/(d+4)}$。两者同阶，$d=1$ 时分别为 $N^{-1/6}$ 与 $N^{-1/5}$——一维的 $N^{-1/5}$ 是严格求导的结果，也正是 Silverman 经验规则里的指数。

> **坑** · 维数从 1 到 10，指数从 $1/5$ 掉到 $1/14$，要维持同样精度的「相对」带宽，所需样本量按 $N^{d/4}$ 指数爆炸。更要命的是，为了让 $N\lambda^d\propto$ 常数，$\lambda^d$ 必须按 $N^{-1}$ 缩，即每个坐标方向的带宽只有最近邻平均的一半。这就是第 2 章「维数灾难」在平滑语境下的精确含义，也是本章所有方法在高维失效的根源。

## 6.3 局部回归 {#s-6-3}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-1-1">原文 §6.1.1</a>

### 6.3.1 局部线性：矩阵解与等效核

局部加权平均的问题在边界处最明显：那里邻域不对称，核被截掉一半，偏差不再随 $h\to0$ 消失（详见 §6.3.5）。修补办法是**在邻域里拟合直线而不是常数**。每个 $x_0$ 解一个独立的加权最小二乘问题 (6.7)：

$$
\min_{\alpha(x_0),\beta(x_0)}\ \sum_{i=1}^N K_\lambda(x_0,x_i)\big[y_i-\alpha(x_0)-\beta(x_0)x_i\big]^2 \eqno{6.7}
$$

估计值只取拟合直线在 $x_0$ 处的值：$\hat f(x_0)=\hat\alpha(x_0)+\hat\beta(x_0)x_0$。记 $b(x)^\top=(1,x)$，$B$ 是 $N\times2$ 回归矩阵（第 $i$ 行 $b(x_i)^\top$），$W(x_0)=\mathrm{diag}(K_\lambda(x_0,x_1),\dots,K_\lambda(x_0,x_N))$。因为加权平方和 $\sum_i w_i(y_i-b^\top\beta)^2=y^\top W y-2\beta^\top B^\top W y+\beta^\top B^\top W B\beta$ 对 $\beta$ 求梯度并令零：

$$
\frac{\partial}{\partial\beta}\big(\cdots\big)=-2B^\top W y+2B^\top WB\beta=0\ \Longrightarrow\ B^\top WB\,\hat\beta=B^\top W y
$$

$B^\top WB$ 是 $2\times2$ 对称正定阵（$B$ 满秩且至少一个权重 $>0$），故

$$
\hat f(x_0)=\hat\alpha(x_0)+\hat\beta(x_0)x_0=b(x_0)^\top\hat\beta(x_0)=b(x_0)^\top(B^\top W(x_0)B)^{-1}B^\top W(x_0)y \eqno{6.8}
$$

写开就是三个标量公式：$S_0=\sum_i w_i$，$S_1=\sum_i w_ix_i$，$S_2=\sum_i w_ix_i^2$，$T_0=\sum_i w_iy_i$，$T_1=\sum_i w_ix_iy_i$，则 $\hat\beta=(T_1-T_0S_1/S_0)/(S_2-S_1^2/S_0)$，$\hat\alpha=(T_0-\hat\beta S_1)/S_0$。

关键是 (6.8) 可以写成 $y$ 的线性组合：

$$
\hat f(x_0)=\sum_{i=1}^N l_i(x_0)\,y_i,\qquad l(x_0)=B(x_0)(B^\top W(x_0)B)^{-1}B^\top W(x_0) \eqno{6.9}
$$

其中 $B(x_0)=\big(b(x_0)^\top,\dots,b(x_N)^\top\big)^\top$。$l_i(x_0)$ 不含 $y$，称为**等效核**（equivalent kernel）。图 6.4 把它画出来：$l_i(x_0)$ 在 $x_0$ 附近的形状与加权平均用的 $K_\lambda$ 很像，但边界处**自动被削平了一侧**，这正是它能修正偏差的原因。

### 6.3.2 局部线性二阶无偏：完整证明

> **基础知识**
> - 一阶 Taylor（余项 $o(h)$，见预备知识 C1）：$f(x_i)=f(x_0)+f'(x_0)(x_i-x_0)+\frac12f''(x_0)(x_i-x_0)^2+o(\lvert x_i-x_0\rvert^2)$。
> - 加权正规方程的一阶条件：对 (6.7) 的目标函数求 $\partial/\partial\beta_1$，得 $\sum_i w_i e_i(x_i-x_0)=0$，其中 $e_i=y_i-b(x_i)^\top\hat\beta(x_0)$。

**第一步：把 (6.9) 的期望写出来。** 因为 $\hat\beta$ 不依赖 $y$，$E[\hat f(x_0)]=\sum_i l_i(x_0)E[y_i]=\sum_i l_i(x_0)f(x_i)$。逐项 Taylor 展开并求和 (6.10)：

$$
E[\hat f(x_0)]=\sum_{i=1}^N l_i(x_0)f(x_i)=f(x_0)\underbrace{\sum_{i=1}^N l_i(x_0)}_{(a)}+f'(x_0)\underbrace{\sum_{i=1}^N (x_i-x_0)l_i(x_0)}_{(b)}+\frac12f''(x_0)\sum_{i=1}^N (x_i-x_0)^2l_i(x_0)+R \eqno{6.10}
$$

**第二步：证两个恒等式。**

**恒等式 1：$\sum_{i=1}^N l_i(x_0)=1$。** 由 (6.9)，$l(x_0)=b(x_0)^\top(B^\top W(x_0)B)^{-1}B^\top W(x_0)$，所以

$$\sum_{i=1}^N l_i(x_0)=\mathbf 1^\top l(x_0)=b(x_0)^\top(B^\top WB)^{-1}B^\top W\mathbf 1$$

$b(x_0)=(1,x_0)^\top$ 就是 $\mathbf 1\in\mathbb{R}^N$，故 $B^\top W\mathbf 1=B^\top Wb(x_0)$，代进去：

$$\sum_{i=1}^N l_i(x_0)=b(x_0)^\top(B^\top WB)^{-1}B^\top WB\,b(x_0)=b(x_0)^\top I\,b(x_0)=1\cdot1=1$$

（$(B^\top WB)$ 可逆：它等于 $\tilde B^\top\tilde B$，对任何 $c\neq0$ 有 $c^\top(B^\top WB)c=\lVert\tilde Bc\rVert^2>0$。）

**恒等式 2：$\sum_{i=1}^N (x_i-x_0)l_i(x_0)=0$。** 换成**中心化**的基：$\tilde b(x)=(1,x-x_0)^\top$，$\tilde B$ 的第 $i$ 行为 $\tilde b(x_i)^\top$。关键观察：(6.7) 的目标函数在 $x_i\mapsto x_i-x_0$ 下不变（直线 $\alpha+\beta x=\alpha'+\beta'(x-x_0)$，取 $\beta'=\beta$、$\alpha'=\alpha+\beta x_0$ 即可一一对应），而中心化模型在 $x_0$ 处的取值 $\tilde b(x_0)^\top\tilde\beta=(1,0)\tilde\beta=\tilde\beta_0$ 恰等于 $\hat\alpha+\hat\beta x_0$。所以 $\hat f(x_0)$ 这个线性泛函不变，(6.9) 的系数也不变，即

$$l(x_0)=\tilde b(x_0)^\top(\tilde B^\top W\tilde B)^{-1}\tilde B^\top W=(\tilde B^\top W\tilde B)^{-1}\tilde B^\top W\tilde B\,\tilde\beta$$

最后一式说明 $l(x_0)$ 落在 $\tilde B$ 的列空间里，故 $\tilde B^\top Wl(x_0)=0$。取其第 2 个分量（$\tilde B$ 的第 2 列是 $x_i-x_0$，$W$ 是对角阵）：

$$\sum_{i=1}^N (x_i-x_0)l_i(x_0)=\Big(\tilde B^\top Wl(x_0)\Big)_2=0$$

证毕。这个论证**完全不用**「窗口完整」这一假设，所以在边界处同样成立。

**第三步：收尾。** 把两个恒等式代回 (6.10)：$f(x_0)$ 项系数为 1、$f'(x_0)$ 项系数为 0，于是

$$
E[\hat f(x_0)]=f(x_0)+\frac12f''(x_0)\sum_{i=1}^N(x_i-x_0)^2l_i(x_0)+R
$$

又因 $l_i=O(\frac1{Nh})$、只有 $O(Nh)$ 个非零项，$\sum_i(x_i-x_0)^2l_i=O(h^2)$，所以 $\mathrm{Bias}=O(h^2)$：局部线性在二阶 Taylor 意义下**无偏**。这就是「自动核改造」的严格内容。

> **结果** · 局部常数与局部线性的偏差阶不同：
> $$\mathrm{Bias}\big[\hat f_{m=0}\big]=-\frac{m_K}{h}f'(x_0)+\frac{h^2}{2}\mu_2(K)f''(x_0)+\cdots,\qquad \mathrm{Bias}\big[\hat f_{m=1}\big]=\frac{h^2}{2}\sum_i(x_i-x_0)^2l_i(x_0)f''(x_0)+o(h^2)$$
> 其中 $m_K=\int uK(u)du$（见预备知识 C3）：$m_K=0$（对称核）时局部常数的偏差已无 $O(h)$ 项，但在边界处 $m_K\neq0$，故仍有 $O(h)$ 偏差。

### 6.3.3 局部多项式

不止线性。设 $b(x)$ 是 $X$ 中次数不超过 $d$ 的多项式项向量（例如 $d=1,p=2$ 时 $b(x)=(1,x_1,x_2)$；$d=2$ 时 $b(x)=(1,x_1,x_2,x_1^2,x_2^2,x_1x_2)$；$d=0$ 时 $b(x)=1$）。在每个 $x_0$ 解

$$
\min_{\beta(x_0)}\ \sum_{i=1}^N K_\lambda(x_0,x_i)\big[y_i-b(x_i)^\top\beta(x_0)\big]^2 \eqno{6.11}
$$

解为 $\hat\beta(x_0)=(B^\top W(x_0)B)^{-1}B^\top W(x_0)y$，与 (6.8) 同型；取值 $\hat f(x_0)=b(x_0)^\top\hat\beta(x_0)=\hat\alpha_0(x_0)$（中心化后就是截距项）。

（这里 $\min K_\lambda$ 只是原文对 $K_\lambda$ 的修饰，意思是「（取自）核 $K_\lambda$ 的权重」。）解为 $\hat\beta(x_0)=(B^\top W(x_0)B)^{-1}B^\top W(x_0)y$，与 (6.8) 同型。取值 $\hat f(x_0)=b(x_0)^\top\hat\beta(x_0)$。

(6.10) 的展开立刻给出：$\sum_i l_i(x_0)=1$，$\sum_i (x_i-x_0)^jl_i(x_0)=0\ (j=1,\dots,d)$（练习 6.2 的推广），于是

$$
\mathrm{Bias}\big[\hat f_{m=d}\big]=f(x_0)\Big(\sum_i l_i-1\Big)+\sum_{j=1}^d f^{(j)}(x_0)\Big(\sum_i(x_i-x_0)^j l_i\Big)+\sum_{j=d+1}^\infty \frac{f^{(j)}(x_0)}{j!}\sum_i (x_i-x_0)^j l_i
$$

只有 $j\ge d+1$ 的项残留：$\mathrm{Bias}=O(h^{d+1})$。

> **坑** · 降偏差要付方差：$y_i=f(x_i)+\varepsilon_i$、$\varepsilon_i$ 独立零均值方差 $\sigma^2$ 时
> $$\mathrm{Var}[\hat f(x_0)]=\mathrm{Var}\Big(\sum_i l_i\varepsilon_i\Big)=\sum_i l_i^2\sigma^2=\sigma^2\lVert l(x_0)\rVert_2^2$$
> 而 $\lVert l(x_0)\rVert_2$ 随 $d$ 单调增（练习 6.3），因为 $d$ 越大解越靠近数据（插值），$l$ 越尖。所以 $d$ 的选择是偏差–方差权衡。

经验法则（原文收集的经验）：局部线性在边界处以很小的方差代价大幅降偏差；局部二次在边界几乎无助、方差却大增；局部二次在内部曲率大的地方最有用；渐近上奇数次的局部多项式优于偶数次（因为渐近 MSE 由边界效应主导）。

### 6.3.4 Epanechnikov 核的最优性：完整推导

现在把优化做到底：先对 $h$ 求最优带宽，再对核形状本身求最优。

**第一步：局部常数估计的偏差。** 设 $y_i=f(x_i)+\varepsilon_i$，$\varepsilon_i$ 独立、$E=0$、$V=\sigma^2$，$x_i$ 固定。在 $x_0$ 处作局部常数估计 $\hat\alpha(x_0)=\frac{\sum_i w_iy_i}{\sum_i w_i}$，$w_i=K_\lambda(x_0,x_i)$，取 $\sum_i w_i\approx N\lambda\int K=N\lambda$。把 $E[\hat\alpha]=\frac{\sum_i w_if(x_i)}{\sum_i w_i}$ 与 $f(x_0)$ 相减，并用 $f(x_i)-f(x_0)\approx f'(x_0)(x_i-x_0)+\frac12f''(x_0)(x_i-x_0)^2$：

- 一阶项：$\frac{f'(x_0)\sum_iw_i(x_i-x_0)}{\sum_iw_i}\approx\frac{f'(x_0)\,N\lambda^2\mu_1(K)}{N\lambda}=\lambda\mu_1(K)f'(x_0)$，对称核下为 $0$；
- 二阶项：$\frac{f''(x_0)\sum_iw_i(x_i-x_0)^2}{2\sum_iw_i}\approx\frac{f''(x_0)\,N\lambda^3\mu_2(K)}{2N\lambda}=\frac{h^2}{2}\mu_2(K)f''(x_0)$。

所以

$$
B=\frac{h^2}{2}\mu_2(K)f''(x_0)+o(h^2)
$$

**第二步：方差。** $\mathrm{Var}[\hat\alpha]=\sigma^2\frac{\sum_i w_i^2}{(\sum_i w_i)^2}$，而 $\sum_i w_i^2\approx N\lambda\int K^2$，$(\sum_i w_i)^2\approx(N\lambda)^2$，故

$$
V=\frac{\sigma^2}{nh\int K^2}
$$

（这里用了核估计里的标度关系，见预备知识 P1 的黎曼和 $\frac1N\sum_i g(x_i)\approx\int g$，对 $g(x)=\lambda^{-1}K^2(\frac{x-x_0}{h})$，即得 $\frac1N\sum g(x_i)=\frac1{Nh}\int K^2$。）

**第三步：MSE 与最优带宽。**

$$
\mathrm{MSE}=B^2+V=\frac{h^4}{4}\mu_2(K)^2f''(x_0)^2+\frac{\sigma^2}{nh\int K^2}
$$

对 $h$ 求导（见预备知识 C1）：$\frac{\partial \mathrm{MSE}}{\partial h}=h^3\mu_2(K)^2f''^2-\frac{\sigma^2}{nh^2\int K^2}=0$，即

$$
h_{\mathrm{opt}}=\left(\frac{\sigma^2}{n\,\mu_2(K)^2f''(x_0)^2\int K^2}\right)^{1/5}
$$

代入回 MSE 得 $\mathrm{MSE}_{\min}=\frac54\big(\tfrac14\mu_2^2f''^2\big)^{1/5}\big(\tfrac{\sigma^2}{n\int K^2}\big)^{4/5}$，即经典结论：MSE 随 $n^{-4/5}$ 收敛。

**第四步：核形状本身的最优。** 上式里核只通过组合量 $\mu_2(K)^2\int K^2$ 出现。由于缩放 $K\mapsto K_c$ 使 $\mu_2\to c^2\mu_2$、$\int K^2\to c^{-1}\int K^2$，等价于在约束

$$
\int K=1,\qquad \int u^2K=\mu_2\ \text{（固定二阶矩）}
$$

下最小化 $\int K^2$。对任意 $K$ 与任意实数 $a$，加 $a$ 到约束里再放大：

$$
\int u^2K=\mu_2\ \Longrightarrow\ \int (u^2-a)K=\mu_2-a
$$

用 Cauchy–Schwarz（$\int gh\le\sqrt{\int g^2\int h^2}$，见预备知识 L1；两个积分都在 $K$ 的支撑 $[-a,a]$ 上取，故收敛）：

$$
(\mu_2-a)^2=\Big[\int_{-a}^{a}(u^2-a)K(u)du\Big]^2\le\Big(\int_{-a}^{a}(u^2-a)^2du\Big)\Big(\int_{-a}^{a}K(u)^2du\Big)
$$

故

$$
\int K^2\ \ge\ \frac{(\mu_2-a)^2}{\int(u^2-a)^2du}\qquad(\forall a,\ \text{使 }K\ge0)
$$

等号成立当且仅当 $K\propto(a-u^2)$（在 $[-a,a]$ 上取正号部分）。取 $a=1$（把 $K$ 的支撑标准化到 $[-1,1]$）即得 $K\propto(1-u^2)_+$——**Epanechnikov 核** (6.4)，归一化常数由 $\int_{-1}^1\frac34(1-t^2)dt=\frac34\cdot\frac43=1$ 定出。验算二阶矩：$\int u^2D=\frac34\int_{-1}^1(u^2-u^4)du=\frac34\cdot\frac45=\frac15$，$\int D^2=\frac9{16}\int_{-1}^1(1-2u^2+u^4)du=\frac9{16}\cdot\frac{16}{15}=\frac35$。

**结果** · Epanechnikov 核 (6.4) 在「给定二阶矩的所有核中使 $\int K^2$ 最小」，因此最小化 $\mu_2^2\int K^2$，从而最小化 MSE (6.4)。它同时有紧支撑（用最近邻窗宽时必需）。

> **数值** · 三核对比（都标定为 $\int K=1$）：$\mu_2^2\int K^2$ 越小越好。
>
> | 核 | $\mu_2$ | $\int K^2$ | $\mu_2^2\int K^2$ |
> |---|---|---|---|
> | Epanechnikov $\frac34(1-t^2)_+$ | $1/5$ | $3/5$ | $3/125=0.024$ |
> | tri-cube $(1-\lvert t\rvert^3)^3$ | $1/6$ | $0.949$ | $0.0264$ |
> | Gauss $\varphi(t)$ | $1$ | $1/(2\sqrt{2\pi})\approx0.1995$ | $\approx0.1995$ |
>
> tri-cube 的积分要展开：$\int_0^1 t^2(1-t^3)^3dt=\frac13-\frac32+\frac13-\frac1{12}=\frac1{12}$，两侧乘 $2$ 得 $\mu_2=\frac16$；$\int D^2=2\int_0^1(1-t^3)^6dt=2\big(1-\frac64+\frac{15}7-\frac{20}{10}+\frac{15}{13}-\frac6{16}+\frac1{19}\big)\approx0.949$。Gauss 核用 $\int\varphi^2=\frac1{2\pi}\int e^{-2u^2}du=\frac1{2\pi}\sqrt{\frac\pi2}=\frac1{2\sqrt{2\pi}}$。
>
> Epanechnikov 最优（数值上与 Cauchy–Schwarz 下界 $0.024$ 相等，说明下界可达），Gauss 核差约 8 倍——这是「紧支撑核更省数据」的第一个定量证据。

### 6.3.5 伪边界与核的改造

上面的偏差公式每一项都悄悄用了一个前提：$\sum_i w_i\approx N\lambda\int K=N\lambda$，也就是**窗口完整地落在 $x_0$ 附近**。$x_0$ 靠近定义域边界时这个前提不成立。

具体地，若 $x_0$ 右侧 $x_i$ 不存在（$x_0$ 是最大值），则 $x_0+x\lambda$ 之后的样本为空，$\int K$ 不足 $1$，且非对称截断使得

$$
\int_{-\infty}^{x_0+\lambda}K_\lambda(x_0,x)dx=\int_{-1}^{0}D(t)dt=1-\frac12=1-\frac12\ne1
$$

（Epanechnikov 左半侧积分 $=\frac34\int_{-1}^0(1-t^2)dt=\frac34(1-\frac13)=\frac12$。）修正这一项后，偏差里重新出现了 $O(\lambda)$ 项：

$$
\mathrm{Bias}\approx\frac{\int K(x_0,x)\cdot 1_{\{x\le b\}}(x)\cdot\big(f(x)-f(x_0)\big)dx}{\int K(x_0,x)1_{\{x\le b\}}dx}
$$

分母小于 $Nh$，分母与分子里的截断不对称共同造成偏差；极端地若 $x_0$ 就是样本最大值，整个窗口只剩左半，$m_K\neq0$。

> **坑** · 三种修法
> - **历史上**：直接改核。局部常数下要用核 $K_\mathrm{adj}(x_0,x)=K(x_0,x)\mathbf 1\{x\le b\}-\big(m_K/m_2\big)(x-b)\mu_2(K)K(x_0,x)$ 之类的修正项，靠渐近 MSE 理论推出来，实现繁琐且只是有限样本下的近似。
> - **局部线性**：把 $x_i$ 中心化到 $x_0$（即在 $x-x_0$ 上拟合），一阶条件 $\sum_iw_ie_i(x_i-x_0)=0$ **自动消掉**那一项 $O(\lambda)$ 偏差。§6.3.2 的证明其实不需要窗口完整——它只用了 $\sum_i(x_i-x_0)l_i=0$ 和 $\sum_il_i=1$，这两个恒等式与数据是否被截断无关。这就是「自动核改造」（automatic kernel carpentry）。
> - **二次核**：$\hat f(x_0)=\sum_iw_i\big[f(x_0)+\tfrac12f''(x_0)(x_i-x_0)^2\big]+\sum_iw_ie_i(x_i-x_0)^2$，偏差降到 $O(\lambda^3)$，但要估计二阶导，方差大增，且在**真正**的边界上帮助很小。

### 6.3.6 局部多项式的几何解释

把局部多项式看成「在 $x_0$ 附近的 $Y$ 对 $X$ 的条件期望的 Taylor 展开」，就明白为什么截距项就是估计值。

模型 $Y=f(X)+\varepsilon$ 在 $x_0$ 处的 Taylor 展开（前 $d$ 阶）为

$$
E[Y\mid X=x_0+(x-x_0)]=f(x_0)+f'(x_0)(x-x_0)+\cdots+\frac{f^{(d)}(x_0)}{d!}(x-x_0)^d+R
$$

在 $x=x_0$ 处右边只剩 $f(x_0)+R$。所以局部多项式回归做的事就是：在 $x_0$ 的邻域内用数据估计这 $d$ 个 Taylor 系数，而 $(\hat\beta_0,\dots,\hat\beta_d)\to(f,f',\dots,f^{(d)}/d!)$，**$\hat\beta_0$ 就是 $\hat f(x_0)$**。这就解释了两件事：一是为什么中心化（用 $x_i-x_0$ 作自变量）不改变 $\hat f(x_0)$，只改变系数解释；二是为什么 $d$ 越大偏差越小（截断的 Taylor 阶越高）而方差越大（要估的系数越多，$l$ 向量越尖）。

> **坑** · Epanechnikov 核在支撑边界处不连续（$D(1)=0$ 但导数跳变），且 $\int K^2$ 之外还需 $\int|K|<\infty$、$\int uK<\infty$ 等正则条件。实践中常用 bicube $D(t)=(1-\lvert t\rvert^3)^3$（三阶连续可微）替代，或对 Epanechnikov 做平滑/截断处理，否则 $\hat f$ 只连续不可微。

## 6.4 $\mathbb{R}^p$ 中的局部回归与结构化局部模型 {#s-6-4}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6">原文 §6.3</a>

### 6.4.1 多维局部回归

Nadaraya–Watson 在 $d$ 维就是把常数局部替换成 $d$ 维核加权的常数；局部线性则在 $X$ 空间里局部拟合一个**超平面**。实现简单，且在边界上明显优于局部常数，所以是默认选择。公式就是 (6.12) 在 $d$ 维的形式：$b(x)$ 是次数 $\le d$ 的 $x$ 的多项式项向量（$d=1,p=2$：$b=(1,x_1,x_2)$；$d=2$：$b=(1,x_1,x_2,x_1^2,x_2^2,x_1x_2)$；$d=0$：$b=1$），每个 $x_0\in\mathbb{R}^d$ 解

$$
\min_{\beta(x_0)}\ \sum_{i=1}^N K_\lambda^d(x_0,x_i)\big[y_i-b(x_i)^\top\beta(x_0)\big]^2 \eqno{6.12}
$$

核通常取径向（radial）形式 (6.13)：

$$
K_\lambda^d(x_0,x)=D\!\left(\frac{\lVert x-x_0\rVert}{\lambda}\right) \eqno{6.13}
$$

其中 $\lVert\cdot\rVert$ 是欧氏范数。**因为欧氏范数依赖各坐标的单位**，平滑前应把每个预测变量标准化（例如到单位标准差）。

> **坑** · 一维时边界效应是个小麻烦，$d\ge2$ 时是大麻烦：处在（球）边界上的样本比例随 $d$ 增大趋于 1（维数灾难的一种表现）。直接改核去处理 $d$ 维边界会非常混乱，尤其边界不规则时。**局部多项式回归在任何维数下都能无缝地做到所需阶数的边界修正**——这也是图 6.8 里星形（极不规则）边界上的天文数据仍能被合理拟合的原因。

但局部回归在 $d\gg2,3$ 时就没什么用了：不可能同时保持「局部性」（低偏差）和「邻域里样本够多」（低方差），除非总样本量按 $2^d$ 指数增长（与 §2.4 的结论一致）。而且高维下 $\hat f(X)$ 的可视化本身就很困难，而这往往是平滑的主要目的。

### 6.4.2 结构化核

默认的球核 (6.13) 对各坐标一视同仁，所以「标准化到单位标准差」是自然的默认。更一般地，用半正定矩阵 $A$ 给不同坐标加权：

$$
K_{\lambda,A}(x_0,x)=D\!\left(\frac{(x-x_0)^\top A\,(x-x_0)}{\lambda}\right) \eqno{6.14}
$$

要强调的是参数必须是 $A^\top A$ 形式，即 $\bigl((x-x_0)^\top A(x-x_0)\bigr)$ 是二次型；等价地写成 $K_{\lambda,A}=D\bigl(\lVert x-x_0\rVert_A/\lambda\bigr)$，$\lVert\cdot\rVert_A=\sqrt{(x-x_0)^\top A(x-x_0)}$。整个坐标或方向可以被降权甚至剔除：若 $A$ 对角，调 $A_{jj}$ 就是调 $X_j$ 的影响；令 $A_{jj}=0$ 就是完全忽略该坐标。$A\succeq0$ 保证距离非负且满足三角不等式的推广（$\lVert x\rVert_A$ 是内积诱导的范数，见预备知识 L1）。

预测变量很多且高度相关时（模拟信号或图像数字化而来就是这种），可以用预测变量的协方差函数定制 $A$，让它少关注高频对比成分（练习 6.4）。也可以**学** $A$ 的参数：第 11 章的投影寻踪回归正是这个路子，$A$ 的低秩版本给出 $\hat f(X)$ 的 ridge form。但更一般的 $A$ 模型过于繁琐，所以下面转向结构化的回归函数。

### 6.4.3 结构化回归函数

要拟合 $E(Y\mid X)=f(X_1,\dots,X_p)$，其中每一级交互都可能存在。自然的想法是先做 ANOVA 分解 (6.15)，然后**去掉某些高阶项**引入结构：

$$
f(X_1,X_2,\dots,X_p)=\alpha+\sum_{j=1}^p g_j(X_j)+\sum_{k<\ell}g_{k\ell}(X_k,X_\ell)+\cdots \eqno{6.15}
$$

- **可加模型**只保留主效应：$f(X)=\alpha+\sum_{j=1}^p g_j(X_j)$。
- 二阶模型保留至多二阶交互，以此类推。
- 第 9 章的 backfitting 算法正是拟合这类低阶交互模型的迭代方法。

> **结果** · backfitting 的关键细节：在可加模型里，若除 $g_k$ 外所有项都已知，则 $g_k$ 的估计**就是**把 $Y-\sum_{j\neq k}g_j(X_j)$ 对 $X_k$ 做一维局部回归。轮流对每个函数做一遍，重复到收敛。整个过程只需要**一维**局部回归，因而完全避开了维数灾难。

一个重要的特例是**变系数模型 (varying coefficient model)**：把 $p$ 个预测变量分成 $(X_1,\dots,X_q)$（$q<p$）和其余变量组成的向量 $Z$，假设条件线性

$$
f(X)=\alpha(Z)+\beta_1(Z)X_1+\cdots+\beta_q(Z)X_q \eqno{6.16}
$$

给定 $Z$ 时这是一个线性模型，但每个系数可以随 $Z$ 变化。自然的拟合方式是局部加权最小二乘：在每个 $z_0$ 解

$$
\min_{\alpha(z_0),\beta(z_0)}\ \sum_{i=1}^N K_\lambda(z_0,z_i)\Big[y_i-\alpha(z_0)-\sum_{j=1}^q x_{ij}\beta_j(z_0)\Big]^2 \eqno{6.17}
$$

这就是 (6.7) 的向量化版本：权重核作用在 $z$ 上，回归矩阵用 $x$。图 6.10 用它拟合人体主动脉直径关于年龄的关系（系数随性别与主动脉位置变化）。

> **坑** · (6.17) 与 (6.7) 的关键区别：**局部化发生在不同的变量上**。做外推时 $z$ 与 $x$ 的角色不能互换；若 $z$ 是「时间」，就得到「按时间做窗口」的局部线性模型，这与第 9 章的时变参数模型思路一致。

## 6.5 局部似然与其他模型 {#s-6-5}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-5">原文 §6.5</a>

### 6.5.1 局部似然

局部回归与变系数模型的思想可以推广到**任何**参数模型，只要拟合方法接受观测权重。设每个观测有参数 $\theta_i=\theta(x_i)=x_i^\top\beta$（对协变量线性），对 $\beta$ 的推断基于对数似然 $\ell(\beta)=\sum_{i=1}^N\ell(y_i,x_i^\top\beta)$。局部化的做法是用 $x_0$ 附近的似然推断 $\theta(x_0)=x_0^\top\beta(x_0)$：

$$
\ell\big(\beta(x_0)\big)=\sum_{i=1}^N K_\lambda(x_0,x_i)\,\ell\big(y_i,x_i^\top\beta(x_0)\big)
$$

许多似然模型（尤其是含 logistic 与 log-linear 的 GLM 族）都以线性方式引入协变量，局部似然就是把全局线性模型放松为**局部**线性。另一种变体是用不同的变量来定义局部性和关联 $\theta$：

$$
\ell\big(\theta(z_0)\big)=\sum_{i=1}^N K_\lambda(z_0,z_i)\,\ell\big(y_i,\eta(x_i,\theta(z_0))\big)
$$

例如 $\eta(x,\theta)=x^\top\theta$，最大化局部似然就拟合出一个变系数模型 $\theta(z)$（即 (6.16) 的变系数形式）。**注意 $\ell$ 是逐观测的**对数似然贡献（不是总似然），这一点让局部化保持了「样本量」的语义。

第三种用法是**自回归时序模型**。$k$ 阶 AR：$y_t=\beta_0+\beta_1y_{t-1}+\cdots+\beta_ky_{t-k}+\varepsilon_t$。记滞后向量 $z_t=(y_{t-1},\dots,y_{t-k})$，模型看起来就是标准线性模型 $y_t=z_t^\top\beta+\varepsilon_t$，通常用最小二乘拟合。用核 $K(z_0,z_t)$ 做**局部**最小二乘，就让模型随序列的短期历史而变化。

> **坑** · 这与传统的「按时间窗口」变化的动态线性模型是两回事：这里局部性作用在**滞后向量空间**（即历史模式的相似性）上，而不是纯粹的时间先后。

作为局部似然的例子，考虑第 4 章 (4.36) 的多类线性 logistic 回归的局部版本。数据是特征 $x_i$ 与类别 $g_i\in\{1,2,\dots,J\}$，线性模型为

$$
\Pr(G=j\mid X=x)=\frac{e^{\beta_{j0}+\beta_j^\top x}}{1+\sum_{k=1}^{J-1}e^{\beta_{k0}+\beta_k^\top x}},\qquad \beta_{J0}=0,\ \beta_J=0 \eqno{6.18}
$$

$J$ 类模型的局部对数似然可以写成

$$
\ell\big(\beta(x_0)\big)=\sum_{i=1}^N K_\lambda(x_0,x_i)\sum_{j=1}^J\mathbf 1\{g_i=j\}\log\frac{e^{\beta_{j0}(x_0)+\beta_j(x_0)^\top(x_i-x_0)}}{1+\sum_{k=1}^{J-1}e^{\beta_{k0}(x_0)+\beta_k(x_0)^\top(x_i-x_0)}} \eqno{6.19}
$$

把它展开成「分子项减分母项」：记 $\eta_{ji}(x_0)=\beta_{j0}(x_0)+\beta_j(x_0)^\top(x_i-x_0)$，则

$$
\sum_{j=1}^J\mathbf 1\{g_i=j\}\log\frac{e^{\eta_{ji}(x_0)}}{1+\sum_{k=1}^{J-1}e^{\eta_{ki}(x_0)}}=\eta_{g_i i}(x_0)-\log\Big(1+\sum_{k=1}^{J-1}e^{\eta_{ki}(x_0)}\Big)
$$

（因为只有 $j=g_i$ 的项系数为 1。分母对每个 $i$ 都一样。$\beta_{J0}=\beta_J=0$ 使 $J$ 类的指数项为 $e^0=1$。）这就是原书 (6.19) 的实际计算形式。

三点说明：用 $g_i$ 作下标挑出合适的分子；$\beta_{J0}=\beta_J=0$ 是模型定义；**局部回归在 $x_0$ 处中心化**，所以拟合出的后验概率在 $x_0$ 处就是

$$
\hat{\Pr}(G=j\mid X=x)=\frac{e^{\hat\beta_{j0}(x_0)}}{1+\sum_{k=1}^{J-1}e^{\hat\beta_{k0}(x_0)}} \eqno{6.20}
$$

> **结果** · 中心化让 (6.20) 变得极其干净：$\hat\beta_j(x_0)^\top(x_i-x_0)$ 在 $i\to x_0$ 时消失，只有截距进入分类器。所以「局部」只体现在系数是 $x$ 的函数上，推断本身仍在无约束 logit 尺度上进行，从而保留了局部线性的偏差修正。

由于 $G$ 是二值指标，也可以直接平滑这个 0/1 响应而不必用似然——那就等价于**局部常数** logistic 回归（练习 6.5），会丢掉局部线性的偏差修正；在无约束 logit 尺度上操作更自然。用 logistic 回归通常还要求参数估计的标准误，这同样可以局部做（用局部 Fisher 信息矩阵的逆的平方根），从而给出拟合流行率的逐点标准误带（原文图 6.12）。

> **数值** · 图 6.12 的心脏疾病数据上，局部线性 logistic 拟合 SBP 与 CHD，暴露了一个用传统方法可能注意不到的异常（高 SBP 段）。图 6.14 用**密度**方法（下一节）给同一问题答案，两者差别见 §6.6.2。

与局部 logistic 关系密切且能规避维数问题的是第 9 章的**广义可加模型**（GAM），它假设回归函数是可加结构。朴素贝叶斯与 GAM 的关系，类比于 LDA 与 logistic 回归的关系（§4.4.5）：logit 变换后可加，但拟合方式不同（练习 6.9）。

## 6.6 核密度估计与分类 {#s-6-6}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-6">原文 §6.6</a>

### 6.6.1 核密度估计

设随机样本 $x_1,\dots,x_N$ 抽自密度 $f_X(x)$，要估计 $f_X(x_0)$。同样的局部思路给出

$$
\hat f_X(x_0)=\frac{\#\{x_i\in\mathcal{N}(x_0)\}}{N\lambda} \eqno{6.21}
$$

其中 $\mathcal{N}(x_0)$ 是 $x_0$ 附近宽度为 $\lambda$ 的度规邻域。这个估计是「bumpy」的，所以更偏好平滑的 Parzen 估计：

$$
\hat f_X(x_0)=\frac1N\sum_{i=1}^N K_\lambda(x_0,x_i) \eqno{6.22}
$$

它以随距 $x_0$ 递减的权重计入附近的观测。此时 $K_\lambda(x_0,x)=\varphi(\lvert x-x_0\rvert/\lambda)$ 是流行选择。记 $\varphi_\lambda$ 为零均值、标准差 $\lambda$ 的 Gauss 密度，则

$$
\hat f_X(x)=\frac1N\sum_{i=1}^N\varphi_\lambda(x-x_i)=\int\varphi_\lambda(x-v)\,d\hat F_N(v)=(\hat F_N\star\varphi_\lambda)(x) \eqno{6.23}
$$

即**经验分布 $\hat F_N$ 与 $\varphi_\lambda$ 的卷积**：$\hat F_N$ 在每个 $x_i$ 处放质量 $1/N$（是跳跃的），$\hat f_X$ 相当于给每个观测加独立的 Gauss 噪声后平滑。

> **数值** · 若 $K_\lambda(x_0,x)=\varphi_\lambda(x-x_0)$，$\varphi_\lambda(t)=\frac1\lambda\varphi(t/\lambda)$，则
> $$\int \varphi_\lambda(x-x_0)\,dx=1\quad(\text{换元 } v=(x-x_0)/\lambda),\qquad \int t\,\varphi_\lambda(t)dt=0\ (\varphi\ \text{偶})$$
> 所以偏差 $=\frac{\lambda^2}{2}\mu_2(\varphi)f''=\frac{\lambda^2}{2}f''(x_0)$，方差 $=\frac{\lambda^{-1}}{N}\int\varphi^2f(x_0)=\frac{f(x_0)}{2\sqrt{2\pi}N\lambda}$。Gauss 核的 MSE 最优带宽 $(\approx0.9\,N^{-1/5}\min(\hat\sigma,(\text{IQR}/1.34)))$ 正是 Silverman 规则的来源。

Parzen 密度估计是局部平均的对应物，后续改进沿局部回归的思路（在**对数**尺度上做局部回归，见 Loader (1999)），本章不展开。在 $\mathbb{R}^p$ 中 Gauss 密度估计的自然推广就是用乘积核：

$$
\hat f_X(x_0)=\frac1N\sum_{i=1}^N\prod_{k=1}^d\varphi_\lambda\!\left(\frac{x_{0k}-x_{ik}}{\lambda}\right)=\frac1{N\lambda^d}\sum_{i=1}^N K^d(x_0,x_i) \eqno{6.24}
$$

其中 $K^d=\prod_{k=1}^d\varphi$（各坐标独立核之积），有效样本量 $\propto N\lambda^d$。

### 6.6.2 核密度分类与朴素贝叶斯

用非参数密度估计做分类很直接：$J$ 类问题里分别在各类内拟合 $\hat f_j(X)$，再配合类先验估计 $\hat\pi_j$（通常是样本比例），用 Bayes 定理得

$$
\hat\pi_j\hat f_j(x_0)=\hat{\Pr}(G=j\mid X=x_0)=\frac{\hat\pi_j\hat f_j(x_0)}{\sum_{k=1}^J\hat\pi_k\hat f_k(x_0)} \eqno{6.25}
$$

图 6.14 用它估计 CHD 的流行率。图 6.12（局部 logistic）与图 6.14（密度法）的差别主要在高 SBP 区：那里两类数据都稀疏，而 Gauss 核用**度规**带宽，密度估计在该区域又低又差（高方差）；局部 logistic (6.20) 用 tri-cube 核配 $k$-NN 带宽，**在那里把核自动加宽**，再用局部线性假设在 logit 尺度上把估计抹平。

> **坑** · 若分类是最终目标，把各类密度分别学得很好既不必要也可能有误导。原文图 6.15 里类密度是多峰的，但**后验比**很光滑；为捕捉这些与后验无关的特征去追求粗糙高方差的密度拟合，反而伤害分类。对二类问题，真正需要估计准的只是决策边界 $\{x\mid\Pr(G=1\mid X=x)=\tfrac12\}$ 附近的区域。

**朴素贝叶斯分类器**在特征空间维数高、密度估计不吸引人时特别合适（也叫「Idiot's Bayes」）。它假设给定 $G=j$ 时各特征 $X_k$ 条件独立：

$$
\Pr(G=\ell\mid X)=\frac{\pi_\ell f_\ell(X)}{\sum_{j=1}^J\pi_jf_j(X)}=\frac{\pi_\ell\prod_{k=1}^p f_{\ell k}(X_k)}{\sum_{j=1}^J\pi_j\prod_{k=1}^p f_{jk}(X_k)} \eqno{6.26}
$$

独立性假设一般不成立，但极大简化了估计：各 $f_{jk}$ 可分别用**一维**核密度估计——这是原始朴素贝叶斯（用单变量 Gauss 表示边缘密度）的推广；若某个 $X_j$ 是离散的，用合适的直方图估计即可，从而无缝混合特征类型。尽管假设乐观，朴素贝叶斯常胜过复杂得多的替代方法，原因与图 6.15 相同：个别类密度估计的偏差未必伤到后验概率，靠近决策区时尤其如此。

以第 $J$ 类为基准类，对 (6.26) 作 logit 变换（见预备知识 O1 的对数与 KL 讨论 / P1 的矩条件）：

$$
\log\frac{\Pr(G=\ell\mid X)}{\Pr(G=J\mid X)}=\log\frac{\pi_\ell}{\pi_J}+\sum_{k=1}^p\log\frac{f_{\ell k}(X_k)}{f_{Jk}(X_k)}=\alpha_\ell+g_\ell(X_k) \eqno{6.27}
$$

（展开：$\log\frac{\pi_\ell\prod_kf_{\ell k}(X_k)}{\pi_J\prod_kf_{Jk}(X_k)}=\log\frac{\pi_\ell}{\pi_J}+\sum_k\log\frac{f_{\ell k}(X_k)}{f_{Jk}(X_k)}$，记 $\alpha_\ell=\log(\pi_\ell/\pi_J)$、$g_{\ell k}(X_k)=\log\frac{f_{\ell k}(X_k)}{f_{Jk}(X_k)}$。）

> **结果** · (6.27) 具有**广义可加模型**的形式，但拟合方式与第 9 章的 GAM 完全不同：朴素贝叶斯逐类独立估计边缘密度再组合，GAM 直接最小化一个惩罚化的加权偏差（如 $\sum_j g_j(X_j)$）目标。两者关系类比 LDA 与 logistic 回归。

## 6.7 径向基函数、核与维数 {#s-6-7}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-7">原文 §6.7</a>

第 5 章里 $f(x)=\sum_{j=1}^M\beta_jh_j(x)$，其中一些基函数本身就是**局部**定义的（B 样条在 $\mathbb{R}$ 局部，张量积在 $\mathbb{R}^p$ 局部），有些则不局部（截断幂基、sigmoid 基）。**径向基函数（RBF）把这两件事合并**：直接把核函数当作基函数——每个观测点处放一个核，观测权重就是它的系数。

### 6.7.1 RBF 展开与重整化

$$
f(x)=\sum_{j=1}^M\beta_j\,K_{\lambda_j}(\xi_j,x),\qquad K_{\lambda_j}(\xi_j,x)=D\!\left(\frac{\lVert x-\xi_j\rVert}{\lambda_j}\right) \eqno{6.28}
$$

$D$ 常取标准 Gauss 密度。$D$ 取 Gauss 时这个模型就是 RBF 网络（$\xi_j,\lambda_j$ 起「权重」作用）。学习 $\{\lambda_j,\xi_j,\beta_j\}$ 的两条路线：

1. **同时最小化平方和**：
$$
\min_{\lambda_j,\xi_j,\beta_j}\ \sum_{i=1}^N\left[y_i-\sum_{j=1}^M\beta_j\exp\left(-\frac{\lVert x_i-\xi_j\rVert^2}{2\lambda_j^2}\right)\right]^2 \eqno{6.29}
$$
这个准则非凸、有多个局部极小，优化算法与神经网络类似。

2. **先估 $\{\lambda_j,\xi_j\}$ 再估 $\beta_j$**：给定前者，后者就是普通最小二乘（正规方程，见预备知识 L2）。$\{\lambda_j,\xi_j\}$ 常**无监督**地用 $X$ 的分布来定——例如对训练 $x_i$ 拟合 Gauss 混合密度，同时给出中心 $\xi_j$ 与尺度 $\lambda_j$；或用聚类定原型、固定 $\lambda_j=\lambda$ 当超参数。缺点是 $\Pr(Y\mid X)$（尤其 $E[Y\mid X]$）对「作用集中在哪里」没有任何发言权；好处是简单。

若强行令所有 $\lambda_j=\lambda$，会产生**空洞**——$\mathbb{R}^p$ 中没有任何核有支撑的区域（图 6.16 上panel）。**重整化径向基函数**（renormalized RBF）避免此问题：

$$
h_j(x)=\frac{D(\lVert x-\xi_j\rVert/\lambda)}{\sum_{k=1}^M D(\lVert x-\xi_k\rVert/\lambda)} \eqno{6.30}
$$

于是 $\sum_jh_j(x)=1$ 恒成立，且 $h_j\to1$（$\xi_j$ 附近）。**Nadaraya–Watson 估计 (6.2) 正是重整化 RBF 展开的一个特例**：

$$
\hat f(x_0)=\sum_{i=1}^N y_i\frac{K_\lambda(x_0,x_i)}{\sum_{j=1}^N K_\lambda(x_0,x_j)}=\sum_{i=1}^N y_ih_i(x_0),\qquad h_i(x_0)=\frac{K_\lambda(x_0,x_i)}{\sum_{j=1}^N K_\lambda(x_0,x_j)} \eqno{6.31}
$$

即在每个观测处放一个基函数 $\xi_i=x_i$，系数 $\hat\beta_i=y_i$，$i=1,\dots,N$。（这就是原文紧接其后引用第 5 章 (5.50) 的原因——那个解正是核诱导的正则化问题。）

> **坑** · **维数是本章所有方法的天花板**。$\mathbb{R}^p$ 中半径 $\lambda$ 的球内样本量 $\propto N\lambda^d$，要把偏差压到 $O(h^2)$ 需要 $\lambda$ 随 $d$ 缩小（§6.2.4 给了指数 $1/(d+4)$ 或 $1/(d+5)$），于是球内样本按 $N^{-d/4}$ 爆炸式减少，$\lambda^{-d}$ 的方差又同时上升。局部回归在 $d\gg2,3$ 时失效，第 2 章已从更一般的角度论证了同样的结论。

## 6.8 混合模型、计算量与练习 {#s-6-8}

<a class="src" href="../esl/ch06-kernel-smoothing-methods.html#s-6-8">原文 §6.8</a>

### 6.8.1 混合模型：核方法的一种极限

混合模型是密度估计的有用工具，也可以看成一种核方法。Gauss 混合模型为

$$
f(x)=\sum_{m=1}^M\alpha_m\varphi(x;\mu_m,\Sigma_m),\qquad \sum_{m=1}^M\alpha_m=1 \eqno{6.32}
$$

参数通常用极大似然配合第 8 章的 EM 算法拟合。两个特例直接连回本章：

- 若协方差约束为标量 $\Sigma_m=\sigma_mI$，(6.32) 就是**径向基展开**（对照 (6.28)）：$\varphi(x;\mu,\sigma^2I)\propto \frac1\sigma D\bigl(\lVert x-\mu\rVert/\sigma\bigr)$。
- 若进一步固定 $\sigma_m=\sigma>0$ 且 $M\uparrow N$，则 (6.32) 的极大似然估计**逼近核密度估计** (6.22)，取 $\hat\alpha_m=1/N$、$\hat\mu_m=x_m$。

第二个特例值得算一下：代入 $\sigma$ 固定的 Gauss 核，

$$
f(x)=\frac1N\sum_{m=1}^N\frac1{\sigma\sqrt{2\pi}}\exp\left(-\frac{\lVert x-x_m\rVert^2}{2\sigma^2}\right)
$$

这正是 (6.24) 的一维 Gauss 版（$\varphi_\sigma(x-x_m)$）。所以「混合模型 $\to$ KDE」的过渡是连续的：混合模型是**自适应**的密度估计（每个成分有自己的中心与尺度），KDE 是把所有成分放在数据点、固定同一尺度的特例。

> **坑** · (6.32) 的似然在 $M$ 增、$\sigma_m\to0$ 时无界（练习 6.11）：取 $\Sigma_1=\varepsilon^2I$、$\alpha_1=\varepsilon$，在 $x=\mu_1$ 处密度值 $\propto\alpha_1/\varepsilon=\varepsilon/\varepsilon=1$，配合其余成分仍在概率单纯形内，$M=N$ 时可以精确记住每个点。所以实践中要么固定 $\sigma_m$，要么用惩罚似然/BIC（第 7 章）选 $M$。

用 Bayes 定理，各类分别拟合混合密度即得到灵活的 $\Pr(G\mid X)$ 模型（第 12 章详述）。原文图 6.17 演示：对心脏疾病数据的 Age 拟合两成分 Gauss 混合（$\Sigma_1,\Sigma_2$ 不约束相等），EM **不使用** CHD 标签，得 $\hat\mu_1=36.4,\hat\Sigma_1=157.7,\hat\alpha_1=0.7$ 与 $\hat\mu_2=58.0,\hat\Sigma_2=15.6,\hat\alpha_2=0.3$（第一个成分标准差极大，近似均匀密度）。混合模型还给出「第 $i$ 个观测属于成分 $m$」的概率：

$$
\hat r_{im}=\frac{\hat\alpha_m\varphi(x_i;\hat\mu_m,\hat\Sigma_m)}{\sum_{k=1}^M\hat\alpha_k\varphi(x_i;\hat\mu_k,\hat\Sigma_k)} \eqno{6.33}
$$

> **数值** · 阈值化 $\hat r_{i2}>0.5$ 得到 $\hat\delta_i$，与 CHD 标签对照：No 组 $232/70$，Yes 组 $76/84$，错误率 $\frac{70+76}{462}\approx32\%$——与用 CHD 作响应的线性 logistic 回归（最大似然）错误率相同，尽管混合模型**根本没看过** CHD 标签。

### 6.8.2 计算量

核方法与局部回归、密度估计都是**基于记忆**的方法：模型就是整个训练集，拟合在评估/预测时完成，对许多实时应用不可行。

- 单点 $x_0$ 的拟合约 $O(N)$ 浮点运算（除平方核等过度简化情形）。
- $M$ 个基函数的展开单次评估 $O(M)$，通常 $M\sim O(\log N)$；基函数方法的初始代价至少 $O(NM^2+M^3)$。
- 核方法的平滑参数 $\lambda$ 通常**离线**确定（如用交叉验证），代价 $O(N^2)$ 浮点运算。

S-PLUS/R 的 `loess` 与 `locfit`（Loader, 1999）用三角剖分方案：先在 $M$ 个精心选择的位置精确计算拟合（$O(NM)$），再用混合（blending）技术插值到别处（每次评估 $O(M)$）。

### 6.8.3 练习与补充 {#s-6-8-3}

> **推导** · 局部线性无偏的核心恒等式（练习 6.2）
> **结果** · 局部线性下 $\sum_{i=1}^N(x_i-x_0)l_i(x_0)=0$：由 §6.3.2 恒等式 2 得证（中心化基 $\tilde B$ 的第 2 列与 $\tilde B^\top Wl(x_0)=0$ 的第 2 个分量）。推广到 $k$ 次局部多项式：中心化基的各列为 $1,(x_i-x_0),\dots,(x_i-x_0)^k$，正规方程 $\tilde B^\top Wl(x_0)=0$ 逐分量给出
> $$\sum_{i=1}^N(x_i-x_0)^jl_i(x_0)=\Big(\tilde B^\top Wl(x_0)\Big)_{j+1}=0,\qquad j=1,\dots,k$$
> 再加上恒等式 1（$\sum_il_i(x_0)=1$，$\tilde B$ 的第 1 列是 $\mathbf 1$），偏差点 $j=0$ 的项系数为 $1$、$j=1,\dots,k$ 的项**全部为零**，只剩 $j\ge k+1$：$\mathrm{Bias}=O(h^{k+1})$。这就是「局部多项式自动把 $k$ 阶 Taylor 项正交化掉」的含义。

> **推导** · 留一交叉验证残差（练习 6.7）
> 线性平滑器 $\hat f=S_\lambda y$，$S_\lambda$ 对称且 $\sum_j\{S_\lambda\}_{ij}=1$。留一残差 $\tilde e_i=y_i-\hat f^{(-i)}(x_i)$，其中 $\hat f^{(-i)}=\frac{(S_\lambda y)_i-\{S_\lambda\}_{ii}y_i}{1-\{S_\lambda\}_{ii}}$。直接算：
> $$\tilde e_i=\frac{(y_i-(S_\lambda y)_i)}{1-\{S_\lambda\}_{ii}}$$
> 即 $\mathrm{PRESS}(\lambda)=\sum_{i=1}^N\Big(\frac{y_i-\hat f(x_i)}{1-\{S_\lambda\}_{ii}}\Big)^2$，只需一次拟合即可算完（原书指出这对局部回归特别简单）。

> **推导** · GCV 是 $PE(\lambda)$ 的无偏估计（练习 6.10，(6.34)(6.35)）
> 设 $y_i=f(x_i)+\varepsilon_i$，$x_i$ 固定，$\mathrm{Var}(\varepsilon_i)=\sigma^2$。在样本内的预测误差
> $$\mathrm{PE}(\lambda)=\frac1N\sum_{i=1}^N PE_i(\lambda),\qquad PE_i(\lambda)=E\Big[\big(y_i^\ast-f_\lambda(x_i)\big)^2\Big] \eqno{6.34}$$
> 逐点平均后：$\frac1N\sum_iPE_i(\lambda)=\frac1N\lVert f_\lambda-f\rVert^2+\sigma^2=\frac1N\lVert(S_\lambda-I)f\rVert^2+\sigma^2$。
> 而样本内平均平方残差
> $$\mathrm{ASR}(\lambda)=\frac1N\sum_{i=1}^N\big(y_i-\hat f_\lambda(x_i)\big)^2=\frac1N\lVert(I-S_\lambda)(f+\varepsilon)\rVert^2=\frac1N\lVert(I-S_\lambda)f\rVert^2+\frac{\sigma^2}{N}\lVert(I-S_\lambda)\rVert_F^2$$
> （用了交叉项 $2f^\top(I-S_\lambda)^\top(I-S_\lambda)\varepsilon$ 的期望为 $0$。）故 ASR 是 $PE$ 的**乐观（低估）**估计，低估量是 $\frac{\sigma^2}{N}\cdot\mathrm{trace}(S_\lambda^\top S_\lambda)$——收缩掉的那部分对应「被平滑吸收的自由度」。GCV 做无偏修正：
> $$\mathrm{GCV}_\lambda(\lambda)=\mathrm{ASR}(\lambda)\Big/\big(1-\mathrm{trace}(S_\lambda)/N\big)^2\ \propto\ \mathrm{ASR}(\lambda)+\frac{2\sigma^2}{N}\mathrm{trace}(S_\lambda) \eqno{6.35}$$
> 有效自由度定义同第 5 章：$\mathrm{df}=\mathrm{trace}(S_\lambda)$，$\{S_\lambda\}_{ij}=l_i(x_j)$ 由等效核 (6.8) 逐点给出。原文图 6.7 把局部线性（span $40\%$，$\mathrm{df}=5.86$）与校准到同样 df 的平滑样条的等效核对比，二者定性相似。

> **数值** · 练习 6.11：Gauss 混合的似然无界
> 取 $M=N$、$\alpha_m=1/N$、$\Sigma_m=\varepsilon^2I$、$\mu_m=x_m$。在 $x=x_1$ 处，
> $$\sum_m\alpha_m\varphi(x_1;\mu_m,\Sigma_m)=\frac{1}{N}\cdot\frac{1}{\varepsilon\sqrt{2\pi}}e^{0}+\sum_{m=2}^N\frac1N\cdot\frac1{\varepsilon\sqrt{2\pi}}e^{-\lVert x_1-x_m\rVert^2/2\varepsilon^2}\ \xrightarrow{\varepsilon\to0}\ \frac{1}{N\varepsilon\sqrt{2\pi}}\to\infty$$
> 所以要固定 $\sigma_m$ 或用信息准则控制模型大小。

> **坑** · 练习 6.8（联合密度的条件均值 = Nadaraya–Watson）
> 用乘积核 $\phi_\lambda(X)\phi_\lambda(Y)$ 估计 $(X,Y)$ 的联合密度，得到 $E(Y\mid X=x)$ 是两个核光滑的比值；取 $Y$ 为离散型则用分类的核估计，等价于 (6.25) 的密度分类器。

> **延伸** · 练习 6.4（$A=\Sigma^{-1}$ 的含义）：$K_{\lambda,A}$ 用 $\Sigma^{-1}$ 诱导的马氏距离，等价于先把 $X$ 白化再做球核平滑——这与第 4 章 LDA 的白化一致；$A=I$ 相当于只标准化到单位方差，忽略相关性。要抑制高频成分，取 $A=\Sigma^{-1}$ 的低秩截断或对 $\Sigma$ 做平滑（$A=(Sigma+cI)^{-1}$，$c$ 越大越接近 $I$）；要完全忽略高频，令 $A$ 在对应特征方向上取 0。