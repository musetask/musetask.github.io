## 3.2 线性回归模型与最小二乘 {#s-3-2}

这一节是整章的地基。书里用三行公式写完的内容（最小二乘的 $\arg\min$、正规方程、高斯假设下的精确分布），我们把它拆成六个小节：先求「$\hat\beta$ 是什么」，再说「$\hat\beta$ 是什么的概率分布」，然后是系数估计的偏差与方差，最后是数值算法。

模型本身只有一行：把响应变量 $Y$ 写成常数项加 $p$ 个预测变量的线性组合。

$$
f(X)=\beta_0+\sum_{j=1}^p X_j\beta_j \eqno{3.1}
$$

$X_j$ 是一个长度为 $N$ 的向量（第 $i$ 个观测上第 $j$ 个预测变量的取值），$\beta_0$ 是截距，$\beta_j$ 是第 $j$ 个系数的**真值**。要强调的关键点：$f$ 关于**系数是线性的**，关于 $X_j$ 未必线性。预测变量 $X_j$ 可以有五种来源：

- 数值型变量的原始取值 $x_{ij}$；
- 变换后的数值，例如 $X_j = \log x_j$、$X_j = x_j^2$、$X_j = e^{-x_j}$；
- 基展开（basis expansion）：把 $x_j$ 在一组基函数 $\phi_1,\dots,\phi_{p_j}$ 上展开，$X_j=\sum_k\phi_k(x_j)$，例如多项式基；
- 哑变量编码（dummy coding）：定性变量按水平数拆成多个 0/1 列；
- 交互项：$X_j = x_{j_1}x_{j_2}$，使模型对 $X$ 本身非线性。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-2">原文 §3.2</a>

「最小」的判据是平方误差和：

$$
\mathrm{RSS}(\beta)=\sum_{i=1}^N\big(y_i-f(x_i)\big)^2=\sum_{i=1}^N\Big(y_i-\beta_0-\sum_{j=1}^p x_{ij}\beta_j\Big)^2 \eqno{3.2}
$$

选平方而不是绝对值，是因为平方可微、无条件凸、有闭式解。统计解释：当 $y_i$ 条件独立且 $E[y_i\mid x_i]=f(x_i)$ 时，$\mathrm{RSS}$ 是 $N$ 个独立同分布随机变量的和的平方，其中每个的期望是 $\mathrm{Var}(y_i\mid x_i)$。于是 $\hat\beta$ 在两个意义下「好」：它是条件均值的最小二乘近似（无偏性条件 $\beta_0+\sum_j\beta_jx_{ij}=E[y_i\mid x_i]$），也是条件方差加权和的最小值。$p=0$ 时 (3.2) 退化为 $\sum_i(y_i-\bar y)^2$，即中心化，这提醒我们截距项的作用就是吸收常数。

### 3.2.1 从残差平方和到正规方程 {#s-3-2-1}

> **基础知识** · 用到的工具
>
> - **内积与范数**（见预备知识 L1）：$\langle a,b\rangle=a^\top b$，$\mathrm{RSS}=\langle y-X\beta,\,y-X\beta\rangle$，所以 $\mathrm{RSS}$ 是 $\beta$ 的二次函数（每项 $\beta^\top X^\top X\beta$）。
> - **二次型求导**（见预备知识 L4）：$A$ 对称时 $\nabla_\beta(\beta^\top A\beta)=2A\beta$，$A$ 不对称时为 $(A+A^\top)\beta$。
> - **正定与唯一极小**（见预备知识 L1、C2）：$A$ 对称正定 ⟹ $\beta^\top A\beta$ 严格凸，$\nabla=0$ 处即唯一全局极小，且可用配方写成 $\beta^\top A\beta=(\beta-\beta^\star)^\top A(\beta-\beta^\star)+b^\top A^{-1}b$。
> - **投影定理**（见预备知识 L2）：$\mathrm{col}(X)$ 中的唯一点 $\hat y$ 使 $y-\hat y\perp\mathrm{col}(X)$。
> - **SVD 与秩**（见预备知识 L3）：$X=UDV^\top$ 给出 $\ker(X)=\ker(V^\top)$。

**第 1 步：写出矩阵形式。** 把 (3.2) 逐项展开成双重求和再收集二次项：

$$
\sum_{i=1}^N\Big(y_i-\beta_0-\sum_{j=1}^p x_{ij}\beta_j\Big)^2=\sum_{i=1}^N y_i^2-2\sum_{i=1}^N y_i\Big(\beta_0+\sum_{j=1}^p x_{ij}\beta_j\Big)+\sum_{i=1}^N\Big(\beta_0+\sum_{j=1}^p x_{ij}\beta_j\Big)^2 \eqno{3.3}
$$

最后一项的常数部分是 $N\beta_0^2$，线性部分是 $2\beta_0\sum_ix_{ij}\beta_j$，二次部分是 $\sum_{i,j}x_{ij}x_{ik}\beta_j\beta_k$。把这些收集起来就是

$$
\mathrm{RSS}(\beta)=(y-X\beta)^\top(y-X\beta)=y^\top y-2\beta^\top X^\top y+\beta^\top X^\top X\beta
$$

其中 $X$ 是 $N\times(p+1)$ 矩阵，第 $i$ 行为 $(1,x_{i1},\dots,x_{ip})^\top$，$y=(y_1,\dots,y_N)^\top$，$\beta=(\beta_0,\beta_1,\dots,\beta_p)^\top$。核对一遍：$(y-X\beta)^\top(y-X\beta)=\sum_{i,k}(y_i-\sum_j x_{ij}\beta_j\cdot I_{jk})(y_k-\sum_\ell x_{k\ell}\beta_\ell)$，对 $i\ne k$ 的交叉项成对抵消，只剩 $i=k$，即 (3.3)。

**第 2 步：求一阶导。** 直接对分量求导，再翻译成矩阵。对 $\beta_j$ 求导（$j\ge1$）：

$$
\frac{\partial\mathrm{RSS}}{\partial\beta_j}=2\sum_{i=1}^N\Big(y_i-\beta_0-\sum_{k=1}^p x_{ik}\beta_k\Big)(-x_{ij})=-2\sum_{i=1}^N x_{ij}\big(y_i-\beta_0-\sum_{k=1}^p x_{ik}\beta_k\big)
$$

对 $\beta_0$ 求导：$\frac{\partial\mathrm{RSS}}{\partial\beta_0}=-2\sum_i\big(y_i-\beta_0-\sum_kx_{ik}\beta_k\big)$，正是上式在 $j=0$（令 $x_{i0}=1$）时的特例。矩阵语言里，$X^\top(y-X\beta)$ 的第 $j$ 分量 $=\sum_i x_{ij}(y_i-\sum_kx_{ik}\beta_k)$，故

$$
\frac{\partial \mathrm{RSS}}{\partial\beta}=-2X^\top(y-X\beta),\qquad \frac{\partial^2 \mathrm{RSS}}{\partial\beta\,\partial\beta^\top}=2X^\top X \eqno{3.4}
$$

海森矩阵的推导只需再求一次导：$\frac{\partial X^\top(y-X\beta)}{\partial\beta}=-X^\top X$，所以 $\frac{\partial^2\mathrm{RSS}}{\partial\beta\partial\beta^\top}=-2\cdot(-X^\top X)=2X^\top X$。

> **结果** · $X^\top X$ 对称，所以二阶导存在且在 $\beta$ 无关；$\mathrm{RSS}$ 是严格凸的（二阶充分条件，见预备知识 C2），因此**局部极小即全局极小，且只要存在就是唯一的**。这解释了为什么最小二乘不需要任何迭代算法。

**第 3 步：正规方程。** 令 (3.4) 的第一个式子为零：

$$
X^\top(y-X\beta)=0\ \Longleftrightarrow\ X^\top y=(X^\top X)\beta \eqno{3.5}
$$

这一步的统计含义非常重要：$X^\top(y-X\beta)=0$ 说残差 $r=y-X\beta$ 与 $X$ 的**每一列**都正交，特别地 $r\perp$ 截距列（即 $\sum_ir_i=0$）。

**第 4 步：解出来。** 若 $X$ 列满秩（$\mathrm{rank}(X)=p+1$），则 $X^\top X$ 正定可逆：

$$
\hat\beta=(X^\top X)^{-1}X^\top y \eqno{3.6}
$$

$X^\top X$ 正定的证明：设 $\beta\ne0$，则 $X\beta\ne0$（否则 $\beta$ 在零空间里，与列满秩矛盾），故 $\beta^\top X^\top X\beta=\lVert X\beta\rVert^2>0$（见预备知识 L1）。所以 (3.5) 有唯一解。

**第 5 步：拟合值与帽子矩阵。**

$$
\hat y=X\hat\beta=X(X^\top X)^{-1}X^\top y=H y \qquad H:=X(X^\top X)^{-1}X^\top \eqno{3.7}
$$

验证 $H$ 是正交投影。转置：$H^\top=(X(X^\top X)^{-1}X^\top)^\top=X\big((X^\top X)^{-1}\big)^\top X^\top$，而 $X^\top X$ 对称 ⟹ $(X^\top X)^{-1}$ 对称（预备知识 L1：$A^{-1}$ 对称 ⟺ $A$ 对称），故 $H^\top=H$。幂等：

$$
H^2=X(X^\top X)^{-1}\underbrace{X^\top X}_{(X^\top X)^{-1}(X^\top X)^{-1}}X^\top=X(X^\top X)^{-1}X^\top=H
$$

用到的唯一一步是 $(X^\top X)^{-1}(X^\top X)=I$。对称且幂等 $\iff$ 正交投影（预备知识 L2），所以

$$
\hat y=\mathop{\mathrm{Proj}}_{\mathrm{col}(X)}y,\qquad r=(I-H)y\perp \mathrm{col}(X)
$$

**第 6 步：几何解释。** 在 $\mathbb{R}^N$ 里，$\mathrm{col}(X)$ 是一个 $p+1$ 维子空间，$y$ 落在它外面。$\hat\beta$ 使 $\hat y$ 是 $\mathrm{col}(X)$ 中离 $y$ 最近的点，残差 $r\perp\hat y$。于是勾股定理给出 Pythagoras 分解

$$
\lVert y\rVert^2=\lVert \hat y\rVert^2+\lVert r\rVert^2=\hat y^\top H y+r^\top r
$$

（用到 $H^2=H$、$H^\top=H$：$\hat y^\top Hy=(Hy)^\top Hy=\lVert Hy\rVert^2$，$r^\top r=y^\top(I-H)^2y=y^\top(I-H)y$）。这个恒等式是后面所有方差分解与 $F$ 检验的起点。

> **坑** · (3.6) 需要 $\mathrm{rank}(X)=p+1$，即 $N\ge p+1$ 且设计矩阵列满秩（无共线）。$X^\top X$ 奇异时 (3.6) **不存在**，但 (3.7) 的 $\hat y$ 仍然唯一——因为投影唯一。用 SVD $X=U_rD_rV_r^\top$（$r=\mathrm{rank}(X)$，见预备知识 L3）可把解写成
>
> $$\hat\beta=V_rD_r^{-1}U_r^\top y=X^+y,\qquad \hat y=U_rU_r^\top y$$
>
> 其中 $X^+=V_rD_r^{-1}U_r^\top$ 是 Moore–Penrose 伪逆。它给出所有解中欧氏范数最小的那一个，代价是 $\hat\beta$ 的坐标依赖基的选择（换基会改变 $\hat\beta$，但 $\hat y$ 不变）。

### 3.2.2 高斯误差模型与精确抽样分布 {#s-3-2-2}

有了 $\hat\beta$ 的表达式，接下来问：它的**不确定性**有多大。为此必须加一个概率模型——高斯线性模型：

$$
Y=\beta_0+\sum_{j=1}^p X_j\beta_j+\varepsilon=X\beta+\varepsilon,\qquad \varepsilon\sim N(0,\sigma^2 I_N) \eqno{3.9}
$$

这里 $\varepsilon\sim N(0,\sigma^2I_N)$ 的含义是：$N$ 个误差独立、均值 0、方差都是 $\sigma^2$。于是 $y=X\beta+\varepsilon$，$E[y]=X\beta$，$\mathrm{Cov}(y)=\sigma^2 I_N$。

> **基础知识** · 本小节要用的分布性质
>
> - **线性变换律**（预备知识 L5）：$Z=aX\Rightarrow E[Z]=aE[X]$，$\mathrm{Cov}(aX)=a\,\mathrm{Cov}(X)\,a^\top$；等价地 $\mathrm{Var}(\sum_k a_kZ_k)=\sum_{ij}a_ia_j\mathrm{Cov}(Z_i,Z_j)$。
> - **高斯向量的二次型**（预备知识 P4）：$Z\sim N(\mu,I_p)$，$A$ 对称、特征值 $\lambda_1,\dots,\lambda_p$，则 $Z^\top AZ=\sum_i\lambda_i Z_i^2+\mu^\top A\mu$，即加权 $\chi^2$ 求和；若所有 $\lambda_i=1$ 就是 $\chi^2_p$。
> - **$t$ 分布定义**（预备知识 P4）：$Z\sim N(0,1)$、$V\sim\chi^2_\nu$ 独立，$\frac{Z}{\sqrt{V/\nu}}\sim t_\nu$。
> - **加权 $\chi^2$ 求和定理**（预备知识 P4）：$\sum_i\lambda_i\chi^2_{1,i}\sim\sigma^2\chi^2_{\sum\lambda_i}$ 当且仅当权全为 $\sigma^2$（更一般地只有权相等时才简化为单个 $\chi^2$）。
> - **迹与期望**（预备知识 L4）：$E[u^\top A v]=u^\top EA\,v$，$\sum_i a_{ii}=\mathrm{tr}(A)$。

**第 1 步：$\hat\beta$ 的方差。** 由 (3.6) 与 $y=X\beta+\varepsilon$：

$$
\hat\beta=(X^\top X)^{-1}X^\top y=(X^\top X)^{-1}X^\top(X\beta+\varepsilon)=(X^\top X)^{-1}X^\top X\beta+(X^\top X)^{-1}X^\top\varepsilon=\beta+A\varepsilon,\qquad A=(X^\top X)^{-1}X^\top
$$

> 这里 $X$ 是 $N\times(p+1)$，所以 $A=(X^\top X)^{-1}X^\top$ 是 $(p+1)\times N$，与噪声 $\varepsilon\in\mathbb{R}^N$ 同形相乘。
> **记号陷阱**：$A$ 是「把 $N$ 维噪声映到 $p+1$ 维系数」的那个矩阵，所以出现的一定是 $A\varepsilon$，不是 $A\varepsilon$；
> 而 $AA^\top$ 是 $(p+1)\times(p+1)$（$=(X^\top X)^{-1}$）、$A^\top A$ 是 $N\times N$（$=$ 帽子矩阵 $H$）。两个顺序千万别混，下面那张形状对照表值得记住。

用协方差的线性变换律（预备知识 L5）与 $\mathrm{Cov}(\varepsilon)=\sigma^2 I_N$：

$$
\mathrm{Var}(\hat\beta)=A(\sigma^2I_N)A^\top=\sigma^2 A A^\top=\sigma^2 (X^\top X)^{-1}X^\top X(X^\top X)^{-1}=\sigma^2(X^\top X)^{-1}
$$

（第一步用协方差的线性变换律 $\mathrm{Cov}(Az)=A\,\mathrm{Cov}(z)\,A^\top$，见预备知识 L5；中间一步把 $A$ 与 $A^\top$ 的具体形状代进去，
中间三项 $(X^\top X)^{-1}\cdot X^\top X\cdot (X^\top X)^{-1}$ 正好约成 $(X^\top X)^{-1}$。）因此

$$\mathrm{Var}(\hat\beta)=\sigma^2(X^\top X)^{-1} \eqno{3.8}$$

顺带记一个后面要反复用的**形状对照**，避免把 $A^\top A$ 与 $AA^\top$ 搞反：

| 矩阵 | 形状 | 等于 |
|---|---|---|
| $A=(X^\top X)^{-1}X^\top$ | $(p+1)\times N$ | — |
| $AA^\top$ | $(p+1)\times(p+1)$ | $(X^\top X)^{-1}$ |
| $A^\top A$ | $N\times N$ | $H=X(X^\top X)^{-1}X^\top$ |

分量形式：$\mathrm{Var}(\hat\beta_j)=\sigma^2\big[(X^\top X)^{-1}\big]_{jj}=:\sigma^2v_j$。注意 $v_j$ 就是帽子矩阵的第 $j$ 个对角元（预备知识 L2 中 $x^\top(X^\top X)^{-1}x$ 对应 $X$ 的某列时的结论，用 Sherman–Morrison 秩一更新可验证 $h_{jj}=x_j^\top(X^\top X)^{-1}x_j$），它是「第 $j$ 个系数被数据挤压的程度」。

**第 2 步：$\sigma^2$ 的估计与无偏性。** 定义

$$
\hat\sigma^2=\frac{1}{N-p-1}\sum_{i=1}^N(y_i-\hat y_i)^2=\frac{r^\top r}{N-p-1}
$$

为什么除以 $N-p-1$ 而不是 $N$？因为残差不是 $N$ 个自由量。先证 $E[\mathrm{RSS}]=(N-p-1)\sigma^2$。由 $r=y-\hat y=(I-H)y$ 与 $y=X\beta+\varepsilon$：

$$
r=(I-H)(X\beta+\varepsilon)=\underbrace{(I-H)X}_{=\,X-HX=\,X-X(X^\top X)^{-1}X^\top X=\,X-HX=\,0}\beta+(I-H)\varepsilon=(I-H)\varepsilon
$$

（这里 $HX=X$，因为 $X(X^\top X)^{-1}X^\top X=X$。）于是 $\mathrm{RSS}=r^\top r=\varepsilon^\top(I-H)^\top(I-H)\varepsilon=\varepsilon^\top(I-H)\varepsilon$，用到 $(I-H)^\top(I-H)=(I-H)^2=I-H$（对称幂等）。又因 $(I-H)\beta=0$，可以把 $\varepsilon$ 换成 $y$：

$$
\mathrm{RSS}=y^\top(I-H)y
$$

取期望：$E[\mathrm{RSS}]=E[y]^\top(I-H)E[y]+\mathrm{tr}\big((I-H)\mathrm{Cov}(y)\big)=0+\sigma^2\mathrm{tr}(I-H)=\sigma^2(N-\mathrm{tr}(H))$。而 $\mathrm{tr}(H)=\mathrm{tr}(X(X^\top X)^{-1}X^\top)=\mathrm{tr}\big((X^\top X)^{-1}(X^\top X)\big)=\mathrm{tr}(I_{p+1})=p+1$（迹的循环律，预备知识 L4）。故

$$
E[\mathrm{RSS}]=(N-p-1)\sigma^2\ \Longrightarrow\ E[\hat\sigma^2]=\sigma^2
$$

$\hat\sigma^2$ 无偏。自由度 $N-p-1$ 的来源：$\mathrm{col}(X)$ 是 $p+1$ 维，$y$ 所在空间是 $N$ 维，$y$ 到子空间的**正交分量**只有 $N-p-1$ 个。

**第 3 步：$\hat\beta$ 的精确分布。** $A\varepsilon$ 是高斯向量的线性变换，故仍高斯；均值 $A^\top X\beta=\beta$（上面已算），协方差 $\sigma^2(X^\top X)^{-1}$：

$$
\hat\beta\sim N\big(\beta,(X^\top X)^{-1}\sigma^2\big) \eqno{3.10}
$$

更深的表述（后面 (3.15) 的置信椭球要用）。注意 $A^\top A=X(X^\top X)^{-1}X^\top=H$ 正是帽子矩阵，于是

$$\frac{(\hat\beta-\beta)^\top(X^\top X)(\hat\beta-\beta)}{\sigma^2}=\frac{(A\varepsilon)^\top(A\varepsilon)}{\sigma^2}=\frac{\varepsilon^\top A A^\top\varepsilon}{\sigma^2}=\frac{\varepsilon^\top H\varepsilon}{\sigma^2}\ \sim\ \chi^2_{p+1}$$

最后一步的权重为什么全是 $1$？因为 $H$ 是**对称幂等**矩阵（$H^\top=H,\ H^2=H$），所以它的特征值只能是 0 或 1；$\mathrm{tr}(H)=p+1$ 给出恰好 $p+1$ 个特征值 1（$\mathrm{tr}$ 等于特征值之和，见预备知识 L3）。把 $\varepsilon$ 按 $H$ 的特征向量分解成 $\sum_i z_i v_i$（$z_i\overset{\text{iid}}{\sim}N(0,\sigma^2)$），则
$$\varepsilon^\top H\varepsilon=\sum_i\lambda_i z_i^2=\sum_{i=1}^{p+1}z_i^2\sim\sigma^2\chi^2_{p+1}$$

> **坑** · 两个容易搞错的地方：
>
> 1. **不要写成 $\sum_i h_{ii}\chi^2_{1,i}$。** $h_{ii}$ 是 $H$ 的对角元，权重应当是它的**特征值**。由于 $H$ 幂等，非零特征值全为 $1$，所以权重恰好全为 $1$，这也是为什么结论是干净的 $\chi^2_{p+1}$ 而不是加权 $\chi^2$。
>
>    （加权 $\chi^2$ 求和的规则见预备知识 P4。）
>
> 2. **不要说 $A\varepsilon$「落在一个低维子空间里」。** 它的协方差是 $\sigma^2AA^\top=\sigma^2(X^\top X)^{-1}$，这是满秩的 $(p+1)\times(p+1)$ 矩阵，所以 $A\varepsilon$ 张成**整个** $\mathbb{R}^{p+1}$。真正被限制住方向的是**乘上 $X^\top X$ 之后**的马氏平方长度（上面那个二次型），因为 $H$ 把 $\varepsilon$ 投影到了 $\mathrm{col}(X)$ 里。

$$
\frac{(\hat\beta-\beta)^\top(X^\top X)(\hat\beta-\beta)}{\sigma^2}=\frac{(A\varepsilon)^\top(A\varepsilon)}{\sigma^2}=\frac{\varepsilon^\top A A\varepsilon}{\sigma^2}=\frac{\varepsilon^\top H\varepsilon}{\sigma^2}=\sum_{i=1}^{p+1}h_{ii}\,\chi^2_{1,i}\ \sim\ \chi^2_{p+1}
$$

最后一步用预备知识 P4 的加权 $\chi^2$ 求和，并注意 $\sum_ih_{ii}=\mathrm{tr}(H)=p+1$、且 $\varepsilon$ 各分量独立正态（$h_{ii}\chi^2_1$ 相互独立）。等价地，$\varepsilon=H\varepsilon+(I-H)\varepsilon$，$H\varepsilon$ 是 $\mathrm{col}(X)$ 上的高斯向量（协方差 $\sigma^2H$），而 $H^\top H\varepsilon$ 的平方和 $=\sum_ih_{ii}\varepsilon_i^2$ 只在权重为常数时才等于 $\chi^2_{p+1}$——这正是 $h_{ii}=1$ 的极限情形，一般情形靠迹恒等式。

**第 4 步：$\hat\sigma^2$ 的精确分布（核心）。** 沿用第 2 步的 $\mathrm{RSS}=\varepsilon^\top(I-H)\varepsilon$。设 $A=I-H$，它是对称幂等矩阵（$A^\top=A$，$A^2=A$），特征值只能是 0 或 1。由 $\mathrm{tr}(H)=p+1$、$\mathrm{tr}(A)=N-p-1$ 以及「对称幂等矩阵的迹 = 特征值之和 = 秩」，得

$$
A\ \text{的特征值：}\underbrace{1,\dots,1}_{N-p-1\ \text{个}},\ \underbrace{0,\dots,0}_{p+1\ \text{个}}
$$

用预备知识 P4（$\varepsilon\sim N(0,I_N)$，$A$ 对称特征值 $\lambda_i$）：$\varepsilon^\top A\varepsilon\sim\sum_{i=1}^N\lambda_i\chi^2_{1,i}$。这里 $\lambda_i$ 是 $N-p-1$ 个 1 与 $p+1$ 个 0，故 $\sum\lambda_i\chi^2_{1,i}$ 就是 $N-p-1$ 个独立 $\chi^2_1$ 之和 $\sim\chi^2_{N-p-1}$：

$$
(N-p-1)\hat\sigma^2\ \sim\ \sigma^2\chi^2_{N-p-1} \eqno{3.11}
$$

直观解释：残差向量 $r=(I-H)\varepsilon$ 躺在 $(I-H)$ 的值空间里，这个空间是 $\mathrm{col}(X)^\perp$，维数 $N-p-1$。在这个空间内 $I-H$ 就是恒等算子，所以 RSS 是恰好 $N-p-1$ 个独立 $\chi^2_1\sigma^2$ 的和；投影到 $\mathrm{col}(X)$ 的那 $p+1$ 个方向被完全「吸收」进 $\hat\beta$，不贡献误差。

> **结果** · (3.11) 同时给出 $E[\hat\sigma^2]=\sigma^2$（$\chi^2_\nu$ 的期望是 $\nu$）与 $\mathrm{Var}(\hat\sigma^2)=2\sigma^4/\nu$。它保证 $\hat\sigma$ 不会系统性地偏大或偏小，这是把 $z$ 统计量换成 $t$ 统计量的唯一理由。

**第 5 步：单个系数的 $t$ 统计量。** 由 (3.10)，$(\hat\beta_j-\beta_j)/(\sigma\sqrt{v_j})\sim N(0,1)$，其中 $v_j=\big[(X^\top X)^{-1}\big]_{jj}$。把 $\sigma$ 换成 $\hat\sigma$：

$$
z_j=\frac{\hat\beta_j}{\hat\sigma\sqrt{v_j}}
=\underbrace{\frac{\hat\beta_j-\beta_j}{\sigma\sqrt{v_j}}}_{\sim N(0,1)}\cdot\underbrace{\frac{\sigma}{\hat\sigma}}_{\text{倒换}}\cdot\frac1{\sqrt{v_j}}
$$

其中 $\hat\beta_j-\beta_j$ 只依赖 $A\varepsilon$，而 $\hat\sigma^2=\varepsilon^\top A\varepsilon/(N-p-1)$（$A=I-H$），两者分别正交投影在 $\mathrm{col}(X)$ 与其正交补上，故**独立**（高斯向量不相关即独立）。又 $\hat\sigma/\sigma=\sqrt{V/\nu}$，$V\sim\chi^2_\nu$，$\nu=N-p-1$。代入 $t$ 分布的定义（预备知识 P4）：

$$
\frac{\hat\beta_j-\beta_j}{\hat\sigma\sqrt{v_j}}\ \sim\ t_{N-p-1}\ \text{（自由度 $\nu=N-p-1$）}
$$

(3.10) 给出 $\hat\beta_j$ 的中心，用 $\hat\beta_j-\beta_j$ 代替 $\hat\beta_j$ 不改变 (3.10) 那种「中心在 $\beta_j$」的性质，故书里直接写成

$$
z_j=\frac{\hat\beta_j}{\hat\sigma\sqrt{v_j}} \eqno{3.12}
$$

**第 6 步：$F$ 统计量（本章最核心的推导之一）。** 要检验「一组系数同时为零」。设 $X_0$ 是 $X_1$ 的列子集：$X_0=X_1S$，$S$ 是 $(p_0+1)\times(p_1+1)$ 选择矩阵。模型 1 含 $p_1+1$ 个参数，模型 0 含 $p_0+1$ 个（嵌套，$p_1>p_0$）。

第 1 小步：把两个残差平方和写成二次型。记 $\hat y_0=X_0(X_0^\top X_0)^{-1}X_0^\top y$，$\hat y_1=Hy$。则 $\mathrm{RSS}_0=\lVert y-\hat y_0\rVert^2=y^\top(I-P_0)y$，$\mathrm{RSS}_1=y^\top(I-P_1)y$。

第 2 小步：证明两残差正交。$r_1=y-\hat y_1\perp\mathrm{col}(X_1)$（正规方程，3.2.1 第 5 步），而 $\mathrm{col}(X_0)\subseteq\mathrm{col}(X_1)$，故 $r_1^\top P_0y=0$；同理 $r_1^\top\hat y_1=0$。于是

$$
r_0^\top r_1=(y-\hat y_0)^\top r_1=y^\top r_1-\hat y_0^\top r_1=0-0=0
$$

第 3 小步：分子化为单个二次型。$\mathrm{RSS}_0-\mathrm{RSS}_1=\lVert r_0\rVert^2-\lVert r_1\rVert^2=\lVert r_0-r_1\rVert^2+2r_0^\top r_1=\lVert r_0-r_1\rVert^2$，而 $r_0-r_1=(I-P_0)y-(I-P_1)y=(P_1-P_0)y$，故

$$
\mathrm{RSS}_0-\mathrm{RSS}_1=y^\top(P_1-P_0)y
$$

第 4 小步：证明 $A=P_1-P_0$ 是秩 $p_1-p_0$ 的正交投影。**对称**显然。**幂等**：由 $X_0=X_1S$ 可算出 $P_1P_0=X_1(X_1^\top X_1)^{-1}X_1^\top X_1S(S^\top X_1^\top X_1S)^{-1}S^\top X_1^\top=X_1S(S^\top X_1^\top X_1S)^{-1}S^\top X_1^\top=P_0$（用了 $(X_1^\top X_1)^{-1}(X_1^\top X_1)=I$），同理 $P_0P_1=P_0$。于是

$$
A^2=(P_1-P_0)^2=P_1-P_0P_1-P_0P_1+P_0=P_1-P_0=A
$$

（用了 $P_1^2=P_1$、$P_0^2=P_0$。）**秩**：$A$ 的值空间 $\subseteq\mathrm{col}(X_1)$（因 $P_1v\in\mathrm{col}(X_1)$），且 $\subseteq\mathrm{col}(X_0)^\perp$（因对任意 $w\in\mathrm{col}(X_0)$，$w^\top Av=(P_1w)^\top v=0$，因为 $P_1w\in\mathrm{col}(X_1)\perp\mathrm{col}(X_0)$）。这个交的维数是 $(p_1+1)-(p_0+1)=p_1-p_0$。

第 5 小步：求两个二次型的分布。用 $\mathrm{RSS}_k=y^\top(I-P_k)y$，因 $(I-P_k)X\beta=0$（与第 2 步同理），故 $\mathrm{RSS}_k=\varepsilon^\top(I-P_k)\varepsilon$。$I-P_1$ 与 $P_1-P_0$ 都是对称幂等投影，秩分别为 $N-p_1-1$ 与 $p_1-p_0$（对称幂等矩阵的迹=秩=特征值之和，特征值只有 0 和 1）。用预备知识 P4：

$$
\frac{\mathrm{RSS}_0-\mathrm{RSS}_1}{\sigma^2}\sim\chi^2_{p_1-p_0},\qquad \frac{\mathrm{RSS}_1}{\sigma^2}\sim\chi^2_{N-p_1-1}
$$

第 6 小步：证明两者独立。高斯向量的两个二次型 $u^\top Bu$ 与 $u^\top Cu$（$B,C$ 对称）在 $BC=0$ 时独立。这里

$$
(P_1-P_0)(I-P_1)=P_1-P_0-P_1^2+P_0P_1=P_1-P_0-P_1+P_0=0
$$

第 7 小步：相除得到 $F$ 分布。两个 $\sigma^2$ 约掉，按 $F$ 分布的定义（分子分母各自被自由度整除，独立 $\chi^2$）：

$$
F=\frac{(\mathrm{RSS}_0-\mathrm{RSS}_1)\big/(p_1-p_0)}{\mathrm{RSS}_1\big/(N-p_1-1)}\sim F_{p_1-p_0,\;N-p_1-1} \eqno{3.13}
$$

> **结果** · 自由度来源完全不同：分子自由度 $p_1-p_0$ = 被检验为零的参数个数，分母自由度 $N-p_1-1$ = 残差自由度。
>
> > **坑**
> > - (3.13) 的整个推导依赖**正态假设** (3.9) 与**嵌套性**（$X_0$ 的列是 $X_1$ 的列的子集）。若两个模型不嵌套，$P_1-P_0$ 不再是投影，分子就不是 $\chi^2$。
> > - $\mathrm{RSS}_0>\mathrm{RSS}_1$ 恒成立（增加变量不会增大 RSS），这是 $F$ 统计量的取值恒 $\ge0$ 的原因，也保证 $p$ 值有意义。
> > - 分母用 $\mathrm{RSS}_1$ 而不是 $\mathrm{RSS}_0$：因为只有 $\mathrm{RSS}_1$ 的自由度等于 $N-p_1-1$，且它与分子独立。

**第 7 步：置信区间。** 双侧水平 $\alpha$ 的置信区间由 $t$ 分布的对偶得到（预备知识 P5）：把「接受 $H_0:\beta_j=\beta_j^{(0)}$」与「$\beta_j^{(0)}$ 落在区间内」对应起来，

$$
\big(\hat\beta_j-z^{(1-\alpha/2)}\sqrt{v_j}\,\hat\sigma,\ \hat\beta_j+z^{(1-\alpha/2)}\sqrt{v_j}\,\hat\sigma\big) \eqno{3.14}
$$

其中 $z^{(1-\alpha/2)}$ 是标准正态分位数：$\alpha=0.05$ 时 $z^{(0.975)}=1.96$，单侧 $\alpha=0.05$ 时 $z^{(0.95)}=1.645$。

**第 8 步：联合置信集（$p+1$ 个系数一起）。** 把第 3 步的二次型结果用 $\hat\sigma$ 缩放：

$$
C_\beta=\Big\{\beta:\ (\hat\beta-\beta)^\top X^\top X(\hat\beta-\beta)\le\hat\sigma^2\,\chi^2_{p+1}(1-\alpha)\Big\} \eqno{3.15}
$$

覆盖概率可直接验证：$\Pr\big(\beta\in C_\beta\big)=\Pr\big((\hat\beta-\beta)^\top X^\top X(\hat\beta-\beta)/\sigma^2\le\chi^2_{p+1}(1-\alpha)\big)=\Pr(\chi^2_{p+1}\le\chi^2_{p+1}(1-\alpha))=1-\alpha$。

**几何形状。** 集合 $C_\beta$ 是 $\beta$ 空间中的**椭球**：约束可以写成

$$
(\hat\beta-\beta)^\top\Big(\frac{X^\top X}{\hat\sigma^2}\Big)(\hat\beta-\beta)\le\chi^2_{p+1}(1-\alpha)
$$

即中心 $\hat\beta$、形状矩阵为 $X^\top X$ 的马氏距离不超过半径。取 $X^\top X=V\Lambda V^\top$（$\lambda_j>0$，$V$ 正交，见预备知识 L3），令 $w=V^\top(\hat\beta-\beta)$，则 $w^\top\Lambda w=\sum_{j=1}^{p+1}\lambda_jw_j^2$。所以椭球的**主轴方向就是 $X^\top X$ 的特征向量方向**，第 $j$ 根半轴的长度为 $\sqrt{\chi^2_{p+1}(1-\alpha)\hat\sigma^2/\lambda_j}$：$\lambda_j$ 大（该方向数据信息多）的方向上椭球扁，$\lambda_j$ 小（接近共线）的方向上椭球长——这正是共线性在置信域上的表现。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-2">原文 §3.2</a>

> **坑** · (3.14) 与 (3.15) 的覆盖概率**同时**对所有 $\beta_j$ 成立（联合置信集比逐个区间同时覆盖的区间更宽一点以外的严格版本）。不要把它们当成 $p+1$ 个独立区间相乘；正确的整体覆盖率就是上面的 $1-\alpha$。

### 3.2.3 数值例子：前列腺癌数据的 $F$ 统计量 {#s-3-2-3}

前列腺癌数据（$N=67$ 个病人，$p=8$ 个预测变量：log PSA、p Gleason 分级、两项最具预测力的临床指标等的 8 个量）上，比较两个嵌套线性模型：

| 模型 | 参数个数 | RSS | 残差自由度 |
|---|---|---|---|
| 简约模型 $0$（4 个系数被约束为 0） | $p_0+1=5$ | $\mathrm{RSS}_0=32.81$ | $67-5=62$ |
| 全模型 $1$ | $p_1+1=9$ | $\mathrm{RSS}_1=29.43$ | $67-9=58$ |

代入 (3.13)：

$$
F=\frac{(32.81-29.43)\big/(9-5)}{29.43\big/(67-9)}=\frac{3.38/4}{29.43/58}=\frac{0.845}{0.50741}=1.665\approx1.67 \eqno{3.16}
$$

$F(4,58)$ 的 0.95 分位数约为 2.49，$p$ 值约 0.17。所以四个被剔除的系数**联合**不显著。逐个看（(3.14) 口径）会得到更长的区间：$2t_{58}(0.975)=2.002$，标准误约 0.75 时区间半宽约 1.5，而半衰期 PSA 的效应只有 0.64 左右——这正是书里强调「联合检验比逐个检验更宽容」的数值来源。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-2">原文 §3.2</a>

### 3.2.4 系数的偏差与最小方差 {#s-3-2-4}

不假设正态也能讨论估计质量。把 (3.9) 里的 $\varepsilon\sim N$ 换成只要「零均值、同方差、不相关」，令

$$
\hat\theta=a^\top\hat\beta=a^\top(X^\top X)^{-1}X^\top y \eqno{3.17}
$$

即系数的任意线性组合。它的期望（预备知识 L5，$E[X^\top y]=X^\top X\beta$）：

$$
E[\hat\theta]=a^\top(X^\top X)^{-1}X^\top X\beta=a^\top\beta=\theta \eqno{3.18}
$$

所以 $\hat\beta$ 是**无偏**的，方差由 (3.8) 给出：$\mathrm{Var}(a^\top\hat\beta)=\sigma^2a^\top(X^\top X)^{-1}a$。

最小方差性质：在所有形如 $\tilde\theta=c^\top y$ 的线性估计中，OLS 达到最小方差。

$$
\mathrm{Var}(a^\top\hat\beta)\ \le\ \mathrm{Var}(c^\top y)\quad\text{对一切 }c\text{ 成立} \eqno{3.19}
$$

推导：约束是「对任意 $\beta$，$c^\top y$ 在 $y=X\beta+\varepsilon$ 下应逼近 $\theta=a^\top\beta$」，这要求 $c^\top X=a^\top$，即 $X^\top c=a$。此时 $\mathrm{Var}(c^\top y)=\sigma^2c^\top c$。把 $c$ 沿 $\mathrm{col}(X)$ 正交分解为 $c=c_\parallel+c_\perp$，其中 $c_\parallel\in\mathrm{col}(X)$、$c_\perp\perp\mathrm{col}(X)$：$X^\top c=a$ 只约束 $c_\parallel$，而 $\lVert c\rVert^2=\lVert c_\parallel\rVert^2+\lVert c_\perp\rVert^2\ge\lVert c_\parallel\rVert^2$。约束 $X^\top c_\parallel=a$ 的范数最小解是 $c_\parallel=X(X^\top X)^{-1}a$（再叠加任何 $X^\top w=0$ 的分量只会增大范数）。于是

$$
\mathrm{Var}(c^\top y)=\sigma^2c^\top c\ge\sigma^2a^\top(X^\top X)^{-1}X^\top X(X^\top X)^{-1}a=\sigma^2a^\top(X^\top X)^{-1}a=\mathrm{Var}(a^\top\hat\beta)
$$

且等号恰在 $c=X(X^\top X)^{-1}a$ 时成立——这正是 OLS 的系数（此时 $c^\top y=a^\top(X^\top X)^{-1}X^\top y=a^\top\hat\beta$，估计量本身相同）。

一般估计量 $\tilde\theta$ 的均方误差可以分解成「方差」与「偏差平方」两部分：

$$
\mathrm{MSE}(\tilde\theta)=E\big[(\tilde\theta-\theta)^2\big]=\mathrm{Var}(\tilde\theta)+\big[E(\tilde\theta)-\theta\big]^2 \eqno{3.20}
$$

推导：$E[(\tilde\theta-\theta)^2]=E[(\tilde\theta-E\tilde\theta+E\tilde\theta-\theta)^2]$，交叉项 $2E[(\tilde\theta-E\tilde\theta)(E\tilde\theta-\theta)]$ 为零，因为第一项均值为 0；第二项是常数。这个公式（预备知识 P1、P2）的意义：任何通过引入偏差（正则化、收缩、删变量）来降低方差的手段，都是在 (3.20) 的两项之间搬砖。

> **结果** · OLS 同时做到**无偏**（(3.18)）与**线性类中最小方差**（(3.19)）。第 6 章的岭回归正好利用 (3.20)：它牺牲一点偏差换取方差的大幅下降，从而降低 MSE。

### 3.2.5 估计误差与新点预测误差 {#s-3-2-5}

有了 $\hat\beta-\beta=A\varepsilon$，就可以回答两个问题：系数本身误差多大？在新观测点上预测的误差多大？

**系数的均方误差。** 无偏 ⟹ 偏差项为零，于是 (3.20) 给出

$$
\mathrm{MSE}(\hat\beta)=E\big[(\hat\beta-\beta)(\hat\beta-\beta)^\top\big]=\mathrm{Var}(\hat\beta)=\sigma^2(X^\top X)^{-1}
$$

整体标准误 $\sqrt{\mathrm{tr}(\hat\beta^\top\hat\beta)}=\lVert\hat\beta\rVert_2$，单个系数的标准误 $\hat\sigma\sqrt{v_j}$。这条等式说明：**小 $\hat\sigma$ 不等于系数精确**，还必须看 $X^\top X$ 的条件数。书里第 3.4.2 讨论配方差差与共线性时说「variance inflation factor」正是 $\big[(X^\top X)^{-1}\big]_{jj}$ 被非截距列的相关系数放大后的结果。

**新观测点。** 新点上 $Y_0=f(x_0)+\varepsilon_0$，$\varepsilon_0\sim N(0,\sigma^2)$，与训练误差独立：

$$
Y_0=f(x_0)+\varepsilon_0,\qquad \hat f(x_0)=x_0^\top\hat\beta=x_0^\top\beta+x_0^\top A\varepsilon \eqno{3.21}
$$

拟合值的方差：$\mathrm{Var}(\hat f(x_0))=x_0^\top\mathrm{Var}(\hat\beta)x_0$，代入 (3.8) 得 $\sigma^2x_0^\top(X^\top X)^{-1}x_0$。均方误差用 (3.20)：

$$
\mathrm{Var}\big[\hat f(x_0)\big]=\sigma^2x_0^\top(X^\top X)^{-1}x_0,\qquad \mathrm{MSE}\big[\hat f(x_0)\big]=\mathrm{Var}\big[\hat f(x_0)\big]+\big[f(x_0)-E\hat f(x_0)\big]^2 \eqno{3.22}
$$

> **结果** · (3.22) 的第二项是**偏差**：如果线性形式 (3.1) 本身规格正确，$f(x_0)=\beta_0+\sum_jx_{0j}\beta_j$ 就是真实均值，偏差为零；若用了多项式展开或哑变量，$f$ 是真实条件均值的近似，就产生平方偏差。这两项的此消彼长是第 7 章「模型复杂度」曲线和第 3.7 维数缩减的动机。

**为什么 (3.22) 的方差 $\le\sigma^2$？** 需要 $x_0^\top(X^\top X)^{-1}x_0\le1$。当 $x_0$ 恰是第 $j$ 个训练设计行时成立：用 Sherman–Morrison（秩一更新，见预备知识 L3）把 $X$ 拆成 $X_{-j}$ 与 $x_j$，记 $a_j=x_j^\top(X_{-j}^\top X_{-j})^{-1}x_j$，则

$$
(X^\top X)^{-1}=(X_{-j}^\top X_{-j})^{-1}-\frac{(X_{-j}^\top X_{-j})^{-1}x_jx_j^\top(X_{-j}^\top X_{-j})^{-1}}{1+a_j}
\ \Longrightarrow\ x_j^\top(X^\top X)^{-1}x_j=\frac{a_j}{1+a_j}
$$

另一方面，由 (3.7) 与帽矩阵的投影性质，$h_{jj}=e_j^\top He_j=e_j^\top X(X^\top X)^{-1}X^\top e_j=\frac{a_j}{1+a_j}$（同一个量）。$H$ 是正交投影，特征值只有 0 与 1，所以 $h_{jj}=e_j^\top He_j\le1$。**引理**：$e_j^\top He_j\le1$，因为 $e_j=H e_j+(I-H)e_j$ 给出 $1=\lVert e_j\rVert^2=\lVert He_j\rVert^2+\lVert(I-H)e_j\rVert^2\ge\lVert He_j\rVert^2=1-\lVert(I-H)e_j\rVert^2$。

对训练点之外的 $x_0$，$x_0^\top(X^\top X)^{-1}x_0$ 可以大于 1（外推时方差爆炸）。$x_0^\top(X^\top X)^{-1}x_0$ 这个二次型就是**杠杆值**（leverage）的连续版本，第 3.5.2 的 ridge 与第 5 章的 Cook 距离都建立在它上面。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-2">原文 §3.2</a>

### 3.2.6 单个预测变量的闭式解 {#s-3-2-6}

把 (3.9) 的高斯模型换成「零均值、同方差、不相关」的更弱假设，记为

$$
Y=X\beta+\varepsilon,\qquad E[\varepsilon]=0,\ \mathrm{Cov}(\varepsilon)=\sigma^2I_N \eqno{3.23}
$$

(3.23) 是后面所有「估计误差」论证的最小假设集：正态只在 (3.10)–(3.16) 的**精确分布**处才需要，无偏性只需 (3.23)。

只有一个预测变量且不截距时，正规方程的每个分量都有显式解。令 $x=(x_1,\dots,x_N)^\top$ 为该列：

$$
\hat\beta_j=\frac{\sum_{i=1}^N x_{ij}y_i}{\sum_{i=1}^N x_{ij}^2},\qquad r=y-X\hat\beta \eqno{3.24}
$$

为什么？$X^\top(y-X\beta)=0$ 在第 $j$ 列上就是 $\sum_ix_{ij}(y_i-x_{ij}\beta_j)=0$，其中 $\sum_i x_{ij}^2$ 与 $\beta_j$ 无关，可直接解出（要求 $\sum_i x_{ij}^2>0$，即该列不全为零）。

用内积记号（预备知识 L1）重写。$\langle x,y\rangle=\sum_{i=1}^N x_iy_i=x^\top y$，是「两列的逐点乘积之和」：

$$
\langle x,y\rangle=\sum_{i=1}^N x_iy_i=x^\top y \eqno{3.25}
$$

于是

$$
\hat\beta=\frac{\langle x,y\rangle}{\langle x,x\rangle},\qquad \hat y=\frac{\langle x,y\rangle}{\langle x,x\rangle}x \eqno{3.26}
$$

**有截距的一元回归**：此时模型是 $y_i=\beta_0+\beta_1x_i+\varepsilon_i$。截距列是 $\mathbf1=(1,\dots,1)^\top$。两条正规方程：$\sum_i(y_i-\beta_0-\beta_1x_i)=0$（故 $\hat\beta_0=\bar y-\hat\beta_1\bar x$）与 $\sum_ix_i(y_i-\beta_0-\beta_1x_i)=0$。代入前者：

$$
\sum_ix_i(y_i-\bar y+\hat\beta_1\bar x-\hat\beta_1x_i)=0\ \Longrightarrow\ \sum_ix_i(y_i-\bar y)=\hat\beta_1\sum_ix_i(x_i-\bar x)
$$

两边同除以 $\sum_ix_i$，再用恒等式 $\sum_ix_i(y_i-\bar y)=\sum_i(x_i-\bar x)(y_i-\bar y)$（因为 $\sum_i(x_i-\bar x)=0$，加上 $\bar y\sum_ix_i$ 不改变和），得

$$
\hat\beta_1=\frac{\langle x-\bar x\mathbf1,\,y\rangle}{\langle x-\bar x\mathbf1,\,x-\bar x\mathbf1\rangle}=\frac{\sum_{i=1}^N(x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^N(x_i-\bar x)^2} \eqno{3.27}
$$

分母就是 $x$ 的样本方差 $S^2$ 乘 $N$。中心化这个步骤在数值上很关键：直接解 $\begin{pmatrix}N&\sum x_i\\ \sum x_i & \sum x_i^2\end{pmatrix}\binom{\beta_0}{\beta_1}=\binom{\sum y_i}{\sum x_iy_i}$ 会因为量纲差异损失精度。

**正交基下的一般情形（多预测变量，不截距）**：设设计矩阵第 $j$ 列是向量 $z_j$（未标准化）。同理

$$
\hat\beta_p=\frac{\langle z_p,y\rangle}{\langle z_p,z_p\rangle} \eqno{3.28}
$$

方差由 (3.8) 给出，分量形式 $\mathrm{Var}(\hat\beta_p)=\sigma^2\big[(X^\top X)^{-1}\big]_{pp}$。若 $X$ 的列已**正交**（$\langle z_p,z_q\rangle=0$，$p\ne q$），则 $X^\top X=\mathrm{diag}(\lVert z_1\rVert_2^2,\dots)$，逆是对角的，于是

$$
\mathrm{Var}(\hat\beta_p)=\frac{\sigma^2}{\langle z_p,z_p\rangle}=\frac{\sigma^2}{\lVert z_p\rVert_2^2} \eqno{3.29}
$$

**推论**：$X$ 列正交时各个 $\hat\beta_p$ 互不相关（协方差矩阵对角），因此逐个检验可以独立解释。方差与该列的**长度平方**成反比——数据把这条「方向」上的信息收集得越多，系数的方差越小。这解释了第 3.7 节 PCR 的一个直觉：先按 $\lVert z_p\rVert$ 排序，把信息量最大的方向留下。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-2">原文 §3.2</a>

> **坑** · (3.24)、(3.26)–(3.29) 都是**不截距**或**单变量**的特例，形式上比一般情形简单，不能直接套到有截距的多变量情形上。书里 (3.27) 的 $\bar x$ 中心化是「去截距」在数值上的等价操作。

### 3.2.7 QR 分解与 SVD：数值上稳定的最小二乘 {#s-3-2-7}

> **基础知识** · SVD 与 QR（预备知识 L3）
>
> - 任何 $X\in\mathbb{R}^{N\times(p+1)}$ 有 SVD $X=UDV^\top$，$D=\mathrm{diag}(d_1,\dots,d_r)$，$d_i\ge0$ 递减，$r=\mathrm{rank}(X)$；$U,V$ 列正交。
> - 若 $d_i>0$（列满秩），则 $X^\top X=VD^2V^\top$，$(X^\top X)^{-1}=VD^{-2}V^\top$，特征向量是 $V$ 的列，特征值 $d_j^2$。
> - 满秩时总能选到 $Q\in\mathbb{R}^{N\times(p+1)}$ 列正交、$R\in\mathbb{R}^{(p+1)\times(p+1)}$ 可逆，使 $X=QR$（**薄 QR**；要求 $R$ 上三角时需 $N\ge p+1$）。

数值上不要用 $(X^\top X)^{-1}$：平方条件数 $\mathrm{cond}(X^\top X)=\mathrm{cond}(X)^2$，会平方地损失精度。正确做法是直接分解 $X$。

**第 1 步：换基。** 设 $X=Z\Gamma$，其中 $Z\in\mathbb{R}^{N\times(p+1)}$ 列满秩，$\Gamma\in\mathbb{R}^{(p+1)\times(p+1)}$ 可逆。$Z$ 可以理解为「一组基向量作为列」，$\Gamma$ 是基与原坐标之间的变换矩阵：

$$
X=Z\Gamma \eqno{3.30}
$$

由于 $Z\Gamma$ 与 $X$ 的列空间相同，投影 $P_X$ 与 $\hat y$ 与基的选择无关。

**第 2 步：把 $Z$ 的列正交化。** 令 $D=\mathrm{diag}(\lVert z_1\rVert_2,\dots,\lVert z_{p+1}\rVert_2)$（各列非零），则 $ZD^{-1}$ 列正交。这给出

$$
X=\underbrace{ZD^{-1}}_{Q}\cdot\underbrace{D\Gamma}_{R}=QR \eqno{3.31}
$$

插入 $D^{-1}D=\mathbf1$ 只是把「缩放因子在哪一边」显式写出来：$X=ZD^{-1}D\Gamma$。两种写法给出**完全相同的 $\hat y$**，因为 $\hat y$ 只依赖列空间；但 $\hat\beta$ 会变，因为它是坐标相关的量（第 3.5.2 会再次强调这一点）。

**第 3 步：用 $Q,R$ 求 $\hat\beta$。** $\mathrm{RSS}=(y-X\beta)^\top(y-X\beta)$，由 $\nabla_\beta\mathrm{RSS}=0$（3.2.1 第 2 步）得 $X^\top X\beta=X^\top y$，即 $QR\beta$ 的最小二乘。因为 $Q^\top Q=I$，可以左乘 $Q^\top$：

$$
Q^\top Q\,R\beta=Q^\top y\ \Longrightarrow\ R\beta=Q^\top y\ \Longrightarrow\ \hat\beta=R^{-1}Q^\top y \eqno{3.32}
$$

这一步比 (3.6) 好：它只解一个 $(p+1)\times(p+1)$ 的三角方程 $R\beta=Q^\top y$（回代，$O((p+1)^2)$），不需要形成 $X^\top X$ 也不需要求逆。

**第 4 步：拟合值。**

$$
\hat y=X\hat\beta=QR R^{-1}Q^\top y=QQ^\top y \eqno{3.33}
$$

$QQ^\top$ 是 $Q$ 的列空间（即 $\mathrm{col}(X)$）上的正交投影：$QQ^\top$ 对称，$(QQ^\top)^2=QQ^\top QQ^\top=QQ^\top$（用 $Q^\top Q=I$）。这与 (3.7) 的 $X(X^\top X)^{-1}X^\top$ 是同一个投影矩阵（两者都投影到 $\mathrm{col}(X)$，投影唯一），但 (3.33) 不含任何逆矩阵的显式形式。

**与 SVD 的对照。** 代入 SVD：$X=UDV^\top$。取 $Q=UD^{-1}$、$R=DV^\top$，则 $QR=UD^{-1}DV^\top=X$，且 $Q^\top Q=D^{-1}U^\top UD^{-1}=D^{-1}D^{-1}$… 注意这不列正交；正确做法是取 $Q=U$、$R=DV^\top$，此时 $Q^\top Q=I$，于是

$$
\hat\beta=(DV^\top)^{-1}U^\top y=V D^{-1}U^\top y=X^+y,\qquad \hat y=UU^\top y
$$

**这就是伪逆的来历**：当 $\mathrm{rank}(X)<p+1$ 时，$U$ 只保留前 $r$ 列，$\hat y=U_rU_r^\top y$ 仍然良定义（3.2.1 的坑），而 $\hat\beta$ 由 $V_rD_r^{-1}U_r^\top y$ 选出最小范数的那个。

**为什么 QR 比 SVD 便宜、但 SVD 更稳。** $X=QR$ 只用 Householder 反射（正交变换），$O(N(p+1)^2)$ 且精度几乎不损失；$X=UDV^\top$ 额外付出 $O(N(p+1)^2)$ 做双对角化，得到**全部**信息（奇异值 = $\mathrm{col}(X)$ 的条件数）。当列接近共线时 $d_{\min}\to0$，$D^{-1}$ 放大数值误差，所以病态问题必须用 SVD 截断（只保留 $d_i$ 大的方向），这就是第 3.4 节数值稳定 QR 的动机。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-2">原文 §3.2</a>

## 3.3 推断与系数显著性 {#s-3-3}

3.2.2 已经把 (3.12)、(3.13)、(3.14)、(3.15) 四个统计量的**分布**从高斯假设和加权 $\chi^2$ 性质（预备知识 P4）里推了出来。§3.3 要做的是把它们翻译成决策语言，并指出这个翻译为什么容易出错。

**双侧检验与区间的对偶。** 由 $t$ 分布的对偶性（预备知识 P5），对 $\beta_j=\beta_j^{(0)}$ 的水平 $\alpha$ 检验与「$\beta_j^{(0)}$ 落在 (3.14) 区间内」等价。于是「不拒绝」等价于「区间包含 0」，这就是为什么回归表里「$p$ 值 < 0.05」与「95% 置信区间不含 0」是同一句话。

**全局检验。** $F$ 统计量 (3.13) 检验 $p_1-p_0$ 个系数同时为零。特例：$p_0=0$（模型只有截距 $\bar y$）且 $p_1=p$，则

$$
F=\frac{\mathrm{TSS}-\mathrm{RSS}}{p}\Big/\frac{\mathrm{RSS}}{N-p-1},\qquad \mathrm{TSS}=\sum_{i=1}^N(y_i-\bar y)^2=\lVert(I-H)y\rVert^2+\lVert Hy\rVert^2
$$

即「模型是否比只报均值有任何改进」。书里第 3.5.4 讨论「$F$ 统计量与信息准则」时用的就是这个版本。

**三个必须记住的陷阱。**

- **不能把 $p$ 值当概率。** $p$ 值 $=\Pr_{H_0}(|T|\ge|T_{\text{obs}}|)$，是在「$H_0$ 为真」的**假想世界**里重复抽样得到 $|T|$ 至少这么大频繁的概率；它绝不是 $\Pr(H_0\mid\text{data})$。数据多时任何微小效应都会显著。
- **$F$ 与 $t^2$ 的关系只对单参数成立。** 只检验一个系数时，$F_{1,\nu}=t_\nu^2$（因为 $\chi^2_1=W^2$）；但 $F$ 统计量与「逐个做 $t$ 检验」不是同一件事，它们的自由度与标度都不同。
- **多重比较。** 检验 $k$ 个系数各取水平 $\alpha$，整体出错概率 $1-(1-\alpha)^k$。Bonferroni 校正取 $\alpha/k$；Benjamini–Hochberg 控制 FDR（错误发现比例），在高维稀疏情形（$p\gg N$）更合适。相关检验的 Bonferroni 过于保守，ESL 后面会回到这一点。

**共线性如何进入 (3.12)。** $v_j=\big[(X^\top X)^{-1}\big]_{jj}$。若第 $j$ 列与其它列高度相关，$X^\top X$ 近奇异，$v_j$ 被放大，$\hat\sigma\sqrt{v_j}$ 变大，$z_j$ 变小——「系数不显著」往往不是「效应为零」而是「设计矩阵设计得差」。第 6 章用岭回归压平条件数以缓解这个问题，第 3.5 用 LARS 路径直接看哪个变量在「挤掉」哪个变量。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-3">原文 §3.3</a>

## 3.4 多输出回归 {#s-3-4}

前面所有内容都是**单输出**：$y\in\mathbb{R}$。多输出回归（multi-output / multi-response regression）同时回归 $K$ 个响应，第 $k$ 个有自己的系数：

$$
Y_k=\beta_{0k}+\sum_{j=1}^p X_j\beta_{jk}+\varepsilon_k \eqno{3.34}
$$

把它合起来记成 $Y_k=f_k(X)+\varepsilon_k$，其中 $f_k$ 是与 $f$ 同类型的线性形式，只是系数换成 $\beta_{jk}$：

$$
Y_k=f_k(X)+\varepsilon_k \eqno{3.35}
$$

其中 $E[\varepsilon_k]=0$、$\mathrm{Var}(\varepsilon_k)=\sigma_k^2$、不同 $k$ 的误差**协方差**为 $\sigma_{kk'}$（一般不为 0，这就是为什么 §3.4 末尾要讨论 GLS）。矩阵形式：$Y\in\mathbb{R}^{N\times K}$ 第 $i$ 行是 $y_i^\top=(y_{i1},\dots,y_{iK})^\top$，$B\in\mathbb{R}^{(p+1)\times K}$ 第 $k$ 列是 $\beta_k=(\beta_{0k},\dots,\beta_{pk})^\top$，$E\in\mathbb{R}^{N\times K}$ 第 $i$ 行为 $\varepsilon_i^\top$：

$$
Y=XB+E \eqno{3.36}
$$

总残差平方和把 $K$ 个输出的残差平方直接加起来：

$$
\mathrm{RSS}(B)=\sum_{k=1}^K\sum_{i=1}^N\big(y_{ik}-f_k(x_i)\big)^2 \eqno{3.37}
$$

**第 1 步：写成迹的形式。** 记 $i$ 行的残差向量 $r_i=y_i-x_i^\top B\in\mathbb{R}^{K}$（注意 $x_i$ 是第 $i$ 行的设计行，$x_i^\top B$ 是第 $i$ 行的预测向量）。则

$$
\sum_{k=1}^K(y_{ik}-f_k(x_i))^2=\lVert y_i-x_i^\top B\rVert_2^2=\big(y_i-x_i^\top B\big)^\top\big(y_i-x_i^\top B\big)
$$

对 $i$ 求和得到标量 $\sum_i r_i^\top r_i$。这正好是 $(Y-XB)^\top(Y-XB)$ 的迹，因为对 $N\times K$ 矩阵 $R$，$\mathrm{tr}(R^\top R)=\sum_{i,k}R_{ik}^2$（$R^\top R$ 是 $K\times K$，其迹 $=\sum_k\sum_iR_{ik}^2$），且 $R=Y-XB$。所以

$$
\mathrm{RSS}(B)=\mathrm{tr}\big[(Y-XB)^\top(Y-XB)\big] \eqno{3.38}
$$

也可以反过来用迹的循环律 $\mathrm{tr}(AB)=\mathrm{tr}(BA)$（预备知识 L4）写 $\mathrm{tr}[(Y-XB)(Y-XB)^\top]$，再按行展开成 $\sum_i\lVert r_i\rVert_2^2$。

**第 2 步：求极小。** 用矩阵微分（预备知识 L4 的 $\frac{\partial\mathrm{tr}(G^\top FA)}{\partial F}=GA$，取 $G=-(Y-XB)=XB-Y$、$A=I$）。更直接地沿用单输出的二次函数求导（预备知识 L4）：$B\mapsto\mathrm{tr}[(Y-XB)^\top(Y-XB)]$ 是二次函数，其梯度为

$$
\nabla_B\mathrm{RSS}(B)=\frac{\partial}{\partial B}\mathrm{tr}\big[(Y-XB)^\top(Y-XB)\big]=-2X^\top(Y-XB)
$$

（可分三步：$\mathrm{tr}[(Y-XB)^\top(Y-XB)]$ 对 $Y-XB$ 线性，二次型求导给出 $2$ 倍残差，再乘上 $\frac{\partial(Y-XB)}{\partial B}=-X$ 得到 $-2X^\top(Y-XB)$。）令梯度为零：$X^\top(Y-XB)=0$，故

$$
\hat B=(X^\top X)^{-1}X^\top Y \eqno{3.39}
$$

**第 3 步：看清它的含义。** $\hat B=(X^\top X)^{-1}X^\top Y$ 把单输出公式逐列应用——第 $k$ 列 $\hat\beta_k=(X^\top X)^{-1}X^\top y_k$。等价地，$\hat B$ 等于对每个输出单独做 OLS。$\hat Y=X\hat B=X(X^\top X)^{-1}X^\top Y=HY$：投影矩阵 $H$ 对所有输出**共用**。方差：$\mathrm{Cov}(\hat b_k)=A^\top\mathrm{Cov}(y_k)A$，其中 $A=(X^\top X)^{-1}X^\top$，即 $\mathrm{Cov}(\hat b_k)=\sigma_k^2(X^\top X)^{-1}$（用 3.2.2 第 1 步的 $A^\top A=H=(X^\top X)^{-1}$）。输出之间的协方差 $\sigma_{kk'}$ 只会让 $\hat b_k$ 与 $\hat b_{k'}$ 相关，不改变各自的边际方差。

> **结果** · (3.39) 与 (3.6) 逐列对应：一个 $N\times(p+1)$ 的设计矩阵服务 $K$ 个输出，计算一次 QR/SVD 分解（3.2.7）就能同时解出全部 $K$ 个系数向量。这与「$K$ 次独立 OLS」的计算量相同，但只需分解一次。

> **坑** · (3.37) 的 $\mathrm{RSS}(B)$ 把 $K$ 个输出的**不可比**尺度直接相加：若输出 1 是「身高（米）」、输出 2 是「收入（万元）」，贡献会被尺度主导。应当先标准化 $Y$，或改成加权形式。

**异方差协方差矩阵：GLS。** 若同一观测上 $K$ 个误差的协方差矩阵 $\Sigma=\mathrm{Cov}(\varepsilon_i)\in\mathbb{R}^{K\times K}$ 不是对角的，最小二乘不再是有效估计（虽然仍无偏），应改用广义最小二乘，把每个观测的残差向量用 $\Sigma^{-1}$ 度量：

$$
\mathrm{RSS}(B;\Sigma)=\sum_{i=1}^N\big(y_i-f(x_i)\big)^\top\Sigma^{-1}\big(y_i-f(x_i)\big) \eqno{3.40}
$$

其中 $f(x_i)=x_i^\top B$。求导给出正规方程 $X^\top\Sigma^{-1}X\,B=X^\top\Sigma^{-1}Y$，解为 $\hat B=(X^\top\Sigma^{-1}X)^{-1}X^\top\Sigma^{-1}Y$——**注意它不等于 (3.39)**，除非 $\Sigma=\sigma^2 I_K$。这个「加权」就是度量学习的雏形：不同输出方向按其在协方差下的精度定权重。若 $\Sigma$ 不可逆（例如 $K>p+1$，$X$ 不满秩导致 $\Sigma$ 与设计耦合退化），用 $\Sigma^{+}$ 伪逆替代 $\Sigma^{-1}$，公式形式不变（这时也正是 $p\gg N$ 时正则化变得必要的场景，见第 3.5 节）。

<a class="src" href="../esl/ch03-linear-methods-for-regression.html#s-3-4">原文 §3.4</a>

> **坑** · (3.40) 的 GLS 需要 $\Sigma$ **已知**，实际中常靠残差迭代估计（可行 GLS）。另外当 $K$ 个输出完全独立（$\Sigma$ 对角）时，GLS 退化为逐输出各自加权，与 (3.39) 加权的逐列 OLS 等价。

