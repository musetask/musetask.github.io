---
id: prereq
n: "0"
title: 数学预备知识
title_en: Mathematical Preliminaries
desc: 推导全书所需的数学工具：线性代数、微积分、概率统计、最优化、数值分析。每条都给出可复核的公式。
prev_title: 封面
next: m02
next_title: 第 2 章 监督学习概述
---

# 0 数学预备知识 {#s-0}

读 ESL 的时候最常卡住的地方不是「看不懂结论」，而是「中间那一步凭什么」。这本书大量使用五类工具：矩阵代数、微积分（极值与求导）、概率统计、优化理论、数值计算。本章把它们一次性重建，后面每一章都只引用这里的编号（`L1`、`C2`、`P5`、`O3` 这样的标签）。

写法上遵循一个原则：**每条工具都给出可以手算验证的形式**，并指出初学者最容易记错的地方。

---

## 0.1 L · 线性代数 {#s-0-1}

### 0.1.1 内积、范数与正定性（L1） {#s-0-1-1}

> **基础知识** · 内积空间 $\mathbb{R}^p$
>
> 定义：对 $a,b\in\mathbb{R}^p$，
>
> $$\langle a,b\rangle = a^\top b = \sum_{k=1}^p a_k b_k,\qquad \|a\|^2 = \langle a,a\rangle$$
>
> 于是 $\|a\|=\sqrt{\sum_k a_k^2}$。Cauchy–Schwarz 不等式 $|\langle a,b\rangle|\le\|a\|\|b\|$ 来自等号条件：$\|a-\lambda b\|^2\ge 0$ 对一切 $\lambda$ 取最小。
>
> **正定矩阵**：对称矩阵 $A=A^\top$ 满足 $x^\top A x>0$ 对一切 $x\ne0$ 成立，称为正定（positive definite）。等价说法很多，逐条有用：
>
> - $A$ 正定 $\iff$ 所有特征值 $\lambda_i>0$
> - $A$ 正定 $\iff$ 可以写成 $A=C^\top C$，$C$ 是列满秩的「方根」矩阵
> - $A$ 正定 $\iff$ $A=L^\top L$，$L$ 上三角（Cholesky 分解）
> - $A$ 半正定：把 $>0$ 换成 $\ge0$，等价于 $\lambda_i\ge0$
>
> **二次型极小判别**：若 $A$ 对称，
>
> $$\min_x x^\top A x \ \text{有非零解当且仅当}\ \lambda_{\min}(A)\le 0$$
>
> 这个判别法是全书写得最多的一行推导，值得记住。

$$
x^\top A x=\sum_{i=1}^p \lambda_i (v_i^\top x)^2
$$

推导：用 $A=Q\Lambda Q^\top$（谱分解），令 $z=Q^\top x$，则 $x^\top A x = \Lambda z^\top z=\sum_i\lambda_i z_i^2$。每个 $\lambda_i$ 的系数是 $z_i^2\ge0$，所以只要有一个 $\lambda_i\le0$，取 $z=e_i$ 就能让它 $\le0$。

> **坑**：正定性要求**对称**。一般矩阵 $A$ 只要对称部分 $\tfrac12(A+A^\top)$ 正定，就有 $x^\top A x>0$（对称部分可以半正定而不失结论）。ESL 里 $X^\top X$ 天然对称，正定性只用在这个自然情形上。

### 0.1.2 投影与最小二乘（L2） {#s-0-1-2}

> **基础知识** · 投影定理
>
> 设 $C$ 是 $\mathbb{R}^n$ 的子空间，$v\in\mathbb{R}^n$。存在唯一的 $p\in C$ 与 $r\perp C$ 使得 $v=p+r$，其中 $p$ 是 $C$ 中离 $v$ 最近的点。这个 $p$ 称为 $v$ 在 $C$ 上的**正交投影**。
>
> 证明只用到两条：$p$ 是最近点 $\Rightarrow$ $\langle v-p,\;c-p\rangle\ge0$ 对一切 $c\in C$；取 $c=p+w$（$w\in C$）得 $\langle v-p,w\rangle\ge0$ 对一切 $w\in C$，即残差与整个子空间正交。$C$ 闭且有限维时该极小点存在。

矩阵形式：$C=\mathrm{col}(X)$，$X$ 为 $n\times p$ 满列秩，则

$$
P_X = X(X^\top X)^{-1}X^\top,\qquad \hat y = P_X y,\qquad \hat\beta=(X^\top X)^{-1}X^\top y
$$

推导：$\hat\beta$ 使残差 $y-X\hat\beta$ 与每个列向量正交，即 $X^\top(y-X\hat\beta)=0$，解出即得。$P_X$ 的两条性质可以直接验证：

$$
P_X^\top = P_X,\qquad P_X^2 = P_X
$$

（$P_X^2 = X(X^\top X)^{-1}X^\top X(X^\top X)^{-1}X^\top = X(X^\top X)^{-1}X^\top$。）对称且幂等 $\iff$ 正交投影。

> **数值** · 残差平方和可分解为
>
> $$\|y\|^2 = \|P_Xy\|^2 + \|(I-P_X)y\|^2$$
>
> 这是 Pythagoras 恒等式在 $\mathbb{R}^N$ 中的应用，也是第 3 章所有 RSS 分析的出发点。

### 0.1.3 特征分解、SVD 与秩（L3） {#s-0-1-3}

> **基础知识** · 谱分解
>
> 实对称矩阵 $A$ 有正交特征系 $\{\lambda_i,v_i\}$：$Av_i=\lambda_i v_i$，$V^\top V=I$。于是 $A=Q\Lambda Q^\top$。
>
> **奇异值分解（SVD）**：任何 $X\in\mathbb{R}^{N\times p}$ 可以写成
>
> $$X = U D V^\top$$
>
> 其中 $D=\mathrm{diag}(d_1,\dots,d_r)$，$d_i\ge0$ 递减，$r=\mathrm{rank}(X)$；$U,V$ 列正交。$d_i^2$ 就是 $X^\top X$ 的特征值，所以**SVD 是对称特征分解在非对称矩阵上的对应物**。

ESL 反复用到的一个等价关系，值得单独推导一遍：

$$
X^\top X = V D^2 V^\top \quad\Longleftrightarrow\quad X^\top X \cdot v_j = d_j^2 v_j
$$

推导：$X=UDV^\top \Rightarrow X^\top X = VD U^\top U D V^\top = VD^2V^\top$。后向由 $X^\top Xv = d^2v \Rightarrow \|(Xv)\|^2 = v^\top X^\top Xv = d^2\|v\|^2$，再由正交性定出 $Xv=d_ju_j$。

> **坑** · 秩亏时 (3.6) 的 $(X^\top X)^{-1}$ 不存在。但**拟合值仍然唯一**：$P_X$ 有无穷多个矩阵表达式，投影本身唯一。ESL 第 3 章专门强调了这一点。
>
> 解决办法是改用 $V_rD_r^{-1}U_r^\top$（截断逆）或伪逆 $X^+=V_rD_r^{-1}U_r^\top$。

### 0.1.4 矩阵微积分（L4） {#s-0-1-4}

> **基础知识** · 梯度与雅可比
>
> 标量函数 $g:\mathbb{R}^p\to\mathbb{R}$ 的梯度 $\nabla g(x)\in\mathbb{R}^p$，分量 $\partial g/\partial x_k$。向量函数的雅可比 $J\in\mathbb{R}^{m\times p}$，$J_{ij}=\partial f_i/\partial x_j$。
>
> **微分恒等式**：把 $dx$ 当作形式小量，$dg=\sum_k \frac{\partial g}{\partial x_k}dx_k$。
>
> 四条常用求导法则（都可由定义验证）：

$$
\nabla_x (a^\top x) = a,\quad
\nabla_x (x^\top A x) = (A+A^\top)x \ (\text{一般}),\quad
\nabla_x (x^\top A x) = 2Ax \ (\text{$A$ 对称})
$$

第三条的推导（这是全书出现频率最高的推导之一）：设 $A$ 对称，

$$
x^\top Ax=\sum_{i,j}A_{ij}x_ix_j,\quad \frac{\partial}{\partial x_k}=\sum_j A_{kj}x_j+\sum_i A_{ik}x_i
$$

第一个和是 $(Ax)_k$，第二个用 $A^\top=A$ 得 $(A^\top x)_k$，两者相等故为 $2(Ax)_k$。

矩阵对矩阵：设 $F$ 为 $N\times M$，$G$ 为 $N\times M$，$A$ 为 $M\times N$（注意维度，则 $FA$ 是 $N\times N$，迹才有意义），则

$$
\frac{\partial \mathrm{tr}(G^\top F A)}{\partial F} = GA^\top
$$

推导：令 $H$ 与 $F$ 同形（$N\times M$），考察一阶增量

$$
\mathrm{tr}\big(G^\top(F+H)A\big)-\mathrm{tr}(G^\top F A)=\mathrm{tr}(G^\top H A)
$$

用迹循环律把 $G^\top$（$M\times N$）挪到最右端：

$$
\mathrm{tr}(G^\top H A)=\mathrm{tr}(A G^\top H)=\mathrm{tr}\big((GA^\top)^\top H\big)=\langle GA^\top,\ H\rangle
$$

（逐个核对维度：$A G^\top$ 是 $M\times M$，$(GA^\top)^\top=AG^\top$，它与同形的 $H$ 做内积得 $N\times M$ 的矩阵。）由微分定义 $dg=\langle \nabla_F g,\ dF\rangle$，得

$$
\nabla_{\!F}\,\mathrm{tr}(G^\top F A)=GA^\top
$$

> **坑** · 三个最容易踩的地方：
>
> 1. **别把 $GA$ 和 $GA^\top$ 搞反。** $\langle GA^\top,H\rangle$ 里的 $GA^\top$ 是 $M\times N$ 才与 $H$ 同形；写成 $GA$（$N\times M$）虽然也"能凑"，但那是另一个式子 $\mathrm{tr}(G^\top H A^\top)$ 的结果，不是本题的。
> 2. **迹循环律的适用条件**：$\mathrm{tr}(AB)=\mathrm{tr}(BA)$ 要求 $A$ 是 $m\times n$、$B$ 是 $n\times m$（两个乘积都是 $m\times m$）。它**不是**"只有方阵才能用"，也不是无条件成立——若 $AB$ 与 $BA$ 尺寸不同，迹根本没有定义。
> 3. **先检查维度**再套公式。上面 $G^\top F A$ 要求 $F$ 是 $N\times M$、$A$ 是 $M\times N$，否则整个表达式无意义。

### 0.1.5 协方差与相关性（L5） {#s-0-1-5}

> **基础知识** · 协方差矩阵
>
> 随机向量 $Z\in\mathbb{R}^p$，均值 $\mu=E[Z]$，协方差 $\mathrm{Cov}(Z)=E[(Z-\mu)(Z-\mu)^\top]$。分量 $\mathrm{Cov}(Z_i,Z_j)=E[(Z_i-\mu_i)(Z_j-\mu_j)]$，标准差 $\sigma_i=\sqrt{\mathrm{Var}(Z_i)}$。
>
> 相关系数 $\mathrm{Corr}(Z_i,Z_j)=\frac{\mathrm{Cov}(Z_i,Z_j)}{\sigma_i\sigma_j}$。
>
> **线性变换律**（ESL 第 14 章反复用）：
>
> $$Z=aX \ \Rightarrow\ E[Z]=aE[X],\qquad \mathrm{Cov}(aX)=a\,\mathrm{Cov}(X)\,a^\top$$

推导：$E[aX]=aE[X]$ 由期望的线性性；$\mathrm{Cov}(aX)=E[(aX-aE[X])(aX-aE[X])^\top]=a\,E[(X-E[X])(X-E[X])^\top]\,a^\top$。第二个等号是矩阵乘法的分配律。

方差分解：$\mathrm{Var}(\sum_k a_k Z_k)=\sum_{ij}a_ia_j\,\mathrm{Cov}(Z_i,Z_j)$ —— 这是**二次型形式的方差公式**，第 6、7、14 章的偏差计算都建立在它之上。

---

## 0.2 C · 微积分与极值 {#s-0-2}

### 0.2.1 无约束极值与海森矩阵（C1） {#s-0-2-1}

> **基础知识**
>
> $g:\mathbb{R}^p\to\mathbb{R}$ 可微，内点 $x_0$ 为局部极小 ⟹ $\nabla g(x_0)=0$（必要条件）。
>
> 二阶判别：设 $\nabla^2 g(x_0)$（海森矩阵，对称，$\partial^2g/\partial x_i\partial x_j$）存在，
>
> $$\nabla^2g(x_0)\ \begin{cases} \text{正定} &\Rightarrow \text{严格局部极小}\\ \text{负定} &\Rightarrow \text{严格局部极大}\\ \text{不定} &\Rightarrow \text{非极值（鞍点）}\\ \text{半正定} &\Rightarrow \text{不能判定} \end{cases}$$

二阶判别的证明只用泰勒展开：$g(x_0+h)=g(x_0)+\nabla g(x_0)^\top h+\frac12h^\top Hh+o(\|h\|^2)$。在内点极值处 $\nabla g(x_0)=0$，剩下

$$g(x_0+h)-g(x_0)=\frac12h^\top Hh+o(\|h\|^2)$$

若 $\lambda_{\min}(H)>0$，则 $h^\top Hh\ge\lambda_{\min}\|h\|^2$，二阶项至少 $\frac{\lambda_{\min}}{2}\|h\|^2>0$；而 $o(\|h\|^2)$ 是三阶以上小量，被它压过，故充分小的 $h\ne0$ 都有 $g(x_0+h)>g(x_0)$，即严格局部极小。$\lambda_{\max}(H)<0$ 同理给出严格极大。

> **坑** · **「半正定 $\Rightarrow$ 局部极小」是错的**，这是最常被误抄的一条：半正定只是**必要**条件，不是充分条件。反例 $g(x)=x^3$ 在 $x_0=0$ 处 $\nabla g(0)=0$、$g''(0)=0\succeq0$，但 $0$ 是**拐点**不是极小。要断定极小，要么证明 $\nabla^2g$ 在某邻域内**正定**，要么直接用结构（例如整体凸 $\Rightarrow$ 任意驻点是全局极小）。另外「不定 $\Rightarrow$ 鞍点」只在 $p\ge2$ 时严格：$p=1$ 时海森是 $1\times1$，非零就已经能定极性。

**梯度下降**：沿最陡下降方向 $-\nabla g$ 移动，

$$
x^{(m+1)} = x^{(m)} - \gamma_m \nabla g(x^{(m)})
$$

局部收敛要求步长不过大；严格说，若 $\nabla g$ Lipschitz 连续且 $\gamma<2/L$，迭代线性收敛。

### 0.2.2 二次函数（贯穿全书的核心） {#s-0-2-2}

> **基础知识** · 二次函数极小
>
> $g(\beta)=\frac12\beta^\top A\beta - b^\top\beta$，$A$ 对称正定。则
>
> $$\nabla g = A\beta - b = 0 \Rightarrow \beta^\star=A^{-1}b,\qquad g(\beta^\star)=-\tfrac12 b^\top A^{-1}b$$

推导第二步：$A\beta^\star=b \Rightarrow b^\top A^{-1}b = (A\beta^\star)^\top A^{-1}A\beta^\star = (\beta^\star)^\top\beta^\star = \beta^{\star\top}A^{-1}b$。故 $g(\beta^\star)=\frac12\beta^{\star\top}b-b^\top\beta^\star=-\frac12 b^\top A^{-1}b$。

**配方（completion of squares）**，几乎所有「为什么 ridge 会收缩」的论证都靠它：

$$
\beta^\top A\beta - 2b^\top\beta = (\beta-A^{-1}b)^\top A(\beta-A^{-1}b) - b^\top A^{-1}b
$$

推导：右边展开 $(\beta-A^{-1}b)^\top A(\beta-A^{-1}b) = \beta^\top A\beta - 2\beta^\top b + b^\top A^{-1}b$。移项即得。

> **结果** · 对任意正定 $A$，函数 $\beta\mapsto\beta^\top A\beta$ 是**严格凸**的。因此 $A$-带约束的极小问题有唯一解，不存在局部非全局极小。这是第 5、12、18 章所有凸性论证的根据。

### 0.2.3 积分技巧：分部积分与换元（C3） {#s-0-2-3}

> **基础知识**
>
> 分部积分：$\frac{d}{dx}[f(x)g(x)]=f'g+fg'$，故 $\int_a^b f'g = [fg]_a^b - \int_a^b fg'$。
>
> 换元：$\int g(u)u'(x)dx = \int g(u)du$。
>
> **卷积恒等式**（局部估计的通用工具，第 6 章反复用）：
>
> $$\int h(x-u)\,du = \int h(v)\,dv \ (\text{换元 } v=x-u),\qquad \int u\,h(x-u)du = x\int h - \int v h(v)dv$$

第二条的推导：令 $v=x-u$，则 $u=x-v$，$du=-dv$，

$$\int u h(x-u)du = \int (x-v)h(v)dv = x\int h(v)dv - \int vh(v)dv$$

第三种更常用的形式：核 $K$ 是概率密度（$\int K=1$）时

$$\int u\,K(x-u)du = x\!\!\int K(v)dv-\!\!\int vK(v)dv = x - m_K,\quad m_K:=\int vK(v)dv
$$

这就是 Nadaraya–Watson 估计量偏差 $-m_K/h$ 的全部来源。

---

## 0.3 P · 概率统计 {#s-0-3}

### 0.3.1 矩、大数定律与中心极限（P1） {#s-0-3-1}

> **基础知识**
>
> 均值 $E[Z]=\int z\,dP$，二阶矩 $E[Z^2]$，$\mathrm{Var}(Z)=E[Z^2]-E[Z]^2$。
>
> **方差可加性**：若 $Z_1,\dots,Z_N$ 独立同分布，则 $\mathrm{Var}(\bar Z)=\sigma^2/N$。推导：
>
> $$\mathrm{Var}(\bar Z)=\mathrm{Var}\Big(\tfrac1N\sum_i Z_i\Big)=\tfrac1{N^2}\sum_{ij}\mathrm{Cov}(Z_i,Z_j)=\tfrac{N}{N^2}\sigma^2=\frac{\sigma^2}{N}$$
>
> 用到 $\mathrm{Cov}(Z_i,Z_j)=0\ (i\ne j)$。
>
> **弱大数定律（WLLN）**：独立同分布、$E|Z|<\infty$ 时 $\bar Z_N\to\mu$ 依概率。
>
> **中心极限定理（CLT）**：独立同分布、均值 $\mu$、方差 $\sigma^2<\infty$ 时
>
> $$\frac{\sqrt N(\bar Z_N-\mu)}{\sigma}\ \xrightarrow{d}\ N(0,1)$$

CLT 的推导思路：中心化后 $Z_i-\mu$ 独立同分布零均值，$\sum_{i=1}^N(\frac{Z_i-\mu}{\sigma/\sqrt N})$ 是标准化独立随机变量之和，其特征函数为

$$
\phi_{S_N}(t)=\prod_{i=1}^N\Big(1-\frac{t^2}{2N}+o(1/N)\Big)\to e^{-t^2/2}
$$

Levy 连续性定理把特征函数收敛翻译成依分布收敛。

**为什么 CLT 是全书统计推断的支柱**：$\beta^\top\hat\beta$ 这类二次型的渐近正态性、$\chi^2$ 检验、$F$ 检验的极限分布，全部由 CLT 加「函数映射定理」得到。ESL 第 3 章的 $F$ 分布、第 7 章的 bootstrap 渐近正态性都只是 CLT 的一次应用。

### 0.3.2 条件期望与全期望公式（P2） {#s-0-3-2}

> **基础知识** · 塔性质（tower property）
>
> 对随机变量 $X,Y$ 与函数 $g$：
>
> $$E[g(Y)] = E\big[E[g(Y)\mid X]\big]$$
>
> 推导：右边 $=E\big[\int g(y)dP_{Y\mid X=y}(y)\big]$，把条件分布看作关于 $(x,y)$ 的边缘分布的条件核，再用边缘化。
>
> **全方差公式**：
>
> $$\mathrm{Var}(Y) = E\big[\mathrm{Var}(Y\mid X)\big] + \mathrm{Var}\big(E[Y\mid X]\big)$$
>
> 推导：记 $m(X)=E[Y\mid X]$，则 $Y-m(X)+m(X)-E[Y]$ 两项正交（条件期望的正交投影性质），
>
> $$E[(Y-E[Y])^2]=E[(Y-m(X))^2]+E[(m(X)-E[Y])^2]+2E[(Y-m(X))(m(X)-E[Y])]$$
>
> 最后一项为零：由条件期望的定义，$E[Z\cdot 1_{\{X\in A\}}]=E[E[Z\mid X]1_{\{X\in A\}}]$，取 $Z=Y-m(X)$、$A=\{m(X)>E[Y]\}$ 之类即可得零。

**这个公式是偏差–方差分解的引擎**。ESL 第 2、7、15 章反复说的「把预测误差写成条件均值减去真值」就来自这里。

### 0.3.3 指数族与极大似然（P3） {#s-0-3-3}

> **基础知识** · 指数族
>
> 密度形如
>
> $$p(y\mid\eta)=h(y)\,\exp\big(\eta^\top T(y)-A(\eta)\big)$$
>
> 其中 $A(\eta)=\log\int h(y)e^{\eta^\top T(y)}dy$ 是配分函数的对数。
>
> **对数配分函数的导数给出矩**：设积分收敛且可交换求导与积分，则
>
> $$A'(\eta)=E_\eta[T(Y)],\qquad A''(\eta)=\mathrm{Var}_\eta\,T(Y)$$
>
> 推导：对 $A(\eta)=\log\int h(y)e^{\eta^\top T}dy$ 求导，分子分母同时含 $e^{\eta^\top T}$，商法则给出 $A'=(\int T h e^{\eta^\top T})/(\int h e^{\eta^\top T})$，即期望。二次求导：再对 $\eta_j$ 求导，把 $\frac{\partial}{\partial\eta_j}e^{\eta^\top T}=T_je^{\eta^\top T}$ 代进去，得到期望的加权形式，整理成 $E[T_jT_k]-E[T_j]E[T_k]$。

> **坑** · $A''(\eta)\ge0$ 说明 $A$ 凸，这保证了负对数似然 $\ell(\eta)=\sum_i[A(\eta_i)-\eta_i^\top T(y_i)]$ 是凸函数，逻辑回归、泊松回归因此有唯一极小。ESL 第 4 章用这个论证说明逻辑回归解的唯一性。

### 0.3.4 抽样分布：$t$、$\chi^2$、$F$（P4） {#s-0-3-4}

> **基础知识** · 高斯向量的二次型
>
> 设 $Z\sim N(\mu,I_p)$，$A$ 对称，特征值 $\lambda_1,\dots,\lambda_p$。则
>
> $$Z^\top AZ \ \sim\ \sum_{i=1}^p \lambda_i \chi^2_1,\qquad Z^\top A Z-\mu^\top A\mu=\sum_i\lambda_i\chi^2_1$$

推导（正交变换法）：设 $q_i$ 为 $A$ 的单位特征向量，$W=Q^\top(Z-\mu)$。$W\sim N(0,I_p)$（正交变换不改变标准高斯的分布，因为每个分量仍是标准正态且不相关），故

$$Z^\top A Z = (Z-\mu)^\top A(Z-\mu)+\mu^\top A\mu = W^\top A W + \mu^\top A\mu = \sum_i \lambda_i W_i^2 + \text{常数}$$

独立性和 $\chi^2_1$ 分布：$W_i\overset{\text{iid}}{\sim}N(0,1)$，$W_i^2\sim\chi^2_1$。

**加权 $\chi^2$ 求和的三大结论**（ESL 第 3 章 (3.11)、(3.15) 的全部内容）：

| 结论 | 条件 | 结果 |
|---|---|---|
| 正定性保持 | 所有 $\lambda_i>0$ | $\sum\lambda_i\chi^2_1 \sim \sigma^2\chi^2_{\sum \lambda_i}$（加权尺度） |
| $t$ 分布 | $\sigma^2$ 未知 | $(\hat\theta-\theta)/(\hat\sigma \sqrt{v})\sim t_\nu$ |
| $F$ 分布 | 两个嵌套二次型 | $\frac{Q_1/(p_1-p_0)}{Q_2/(N-p_1-1)}\sim F_{p_1-p_0,\,N-p_1-1}$ |

> **结果** · **$t$ 分布的定义**：若 $Z\sim N(0,1)$，$V\sim\chi^2_\nu$ 独立，则 $T=Z/\sqrt{V/\nu}\sim t_\nu$。
>
> 这就是书里说「把 $\sigma$ 换成 $\hat\sigma$ 后 $z_j$ 服从 $t_{N-p-1}$」的原因：$\hat\beta_j-\beta_j$ 的分子是标准正态的倍数，分母换成 $\hat\sigma$ 后分母是 $\chi^2_\nu/\nu$ 的平方根。

> **坑** · $t$ 与 $F$ 的自由度来源不同：$t$ 的 $\nu=N-p-1$ 来自**残差自由度**（估了 $p+1$ 个参数），$F$ 的分母自由度也来自残差；分子自由度来自**被检验的参数个数**。搞混这两者是常见错误。

### 0.3.5 假设检验与置信区间（P5） {#s-0-3-5}

> **基础知识** · 检验的逻辑
>
> - 原假设 $H_0:\theta=\theta_0$，统计量 $T(\text{data})$ 在 $H_0$ 下的分布已知。
> - $p$ 值 $=\Pr\bigl(\lvert T\rvert\ge\lvert T_{\text{obs}}\rvert\,\bigr)$ —— **不是** $P(H_0\mid\text{data})$。
> - 置信区间 $\{[\,l(\text{data}),\,u(\text{data})\,]\}$ 覆盖 $\theta$ 的概率是 $1-\alpha$，**参数是固定的，随机的是区间**。
>
> **双侧置信区间来自检验的对偶**：水平 $\alpha$ 的检验 $\{ \text{接受 }H_0\}$ 与水平 $1-\alpha$ 的置信集 $\{ \theta: \text{不接受 } H_0:\theta\}$ 互补。

正态情形：已知 $\sigma$ 时 $\hat\theta\sim N(\theta,\sigma^2/n)$，

$$\frac{\sqrt n(\hat\theta-\theta)}{\sigma}\sim N(0,1)\ \Rightarrow\ \theta\in \Big[\hat\theta - z^{(1-\alpha/2)}\frac{\sigma}{\sqrt n},\ \hat\theta+z^{(1-\alpha/2)}\frac{\sigma}{\sqrt n}\Big]$$

---

## 0.4 O · 最优化理论 {#s-0-4}

### 0.4.1 拉格朗日乘子与对偶（O1） {#s-0-4-1}

> **基础知识**
>
> 等式约束 $\min_\beta f(\beta)\ \text{s.t.}\ g(\beta)=0$。用 $L(\beta,\lambda)=f(\beta)-\lambda^\top g(\beta)$，则最优解满足 $\nabla_\beta L=0,\ \lambda^\top g=0$。
>
> **凸问题的对偶**：原问题
>
> $$\min_\beta f_0(\beta)\quad \text{s.t. } f_i(\beta)\le0,\ i=1,\dots,m$$
>
> 拉格朗日函数 $L=f_0+\sum_i\lambda_i f_i$，$\lambda\ge0$。对偶函数
>
> $$g(\lambda)=\inf_\beta L(\beta,\lambda)$$
>
> 弱对偶性（不需要任何凸性）：对任意可行 $(\beta,\lambda)$，$g(\lambda)\le L(\beta,\lambda)=f_0(\beta)$，故 $g(\lambda)\le f_0(\beta)$。
>
> Slater 条件（存在严格可行点 $f_i<0$）下，$f_0$ 凸时强对偶成立：$\sup_\lambda g(\lambda)=\min_\beta f_0(\beta)$。

> **坑** · **对偶总是凸问题**。这是 SVM、lasso 的对偶形式能带来算法优势的根本原因：即使原问题非凸，对偶仍凸，可以放心用梯度法。ESL 第 12 章反复利用这一点。

### 0.4.2 KKT 条件（O2） {#s-0-4-2}

> **基础知识** · 四个条件
>
> 凸问题 + Slater ⟹ 最优解 $(\beta^\star,\lambda^\star)$ 恰好是下列条件的解：
>
> - **原始可行**：$f_i(\beta^\star)\le0$
> - **对偶可行**：$\lambda^\star\ge0$
> - **互补松弛**：$\lambda_i^\star f_i(\beta^\star)=0$
> - **平稳性**：$\nabla f_0(\beta^\star)+\sum_i\lambda_i^\star\nabla f_i(\beta^\star)=0$
>
> 推导：把 KKT 的四个条件组合成
>
> $$L(\beta,\lambda)-L(\beta^\star,\lambda^\star)=\sum_i\lambda_i f_i(\beta)-\sum_i\lambda_i^\star f_i(\beta^\star)$$
>
> 逐项非负（$\lambda_i\ge0,\ f_i\le0$；互补松弛使 $-\lambda_i^\star f_i(\beta^\star)\ge0$）。若再满足平稳性，右端等于
>
> $$(\beta-\beta^\star)^\top\big[\nabla f_0(\beta^\star)+\sum_i\lambda_i^\star\nabla f_i(\beta^\star)\big]=0$$

**互补松弛是「什么时候约束起作用」的回答**：$\lambda_i^\star>0$ 才有可能 $f_i(\beta^\star)=0$（约束紧）。ESL 第 3 章 lasso 的 KKT 条件 (3.58)(3.59) 正是这一条的直接应用——不活跃变量的零阈值条件就来自这里。

### 0.4.3 EM 算法（O3） {#s-0-4-3}

> **基础知识**
>
> 设观测数据 $y$，缺失（或未观测）数据 $z$，完整数据的联合密度
>
> $$p_\theta(y,z)=p_\theta(y\mid z)\,p_\theta(z)$$
>
> 定义对数似然的期望
>
> $$Q(\theta,\theta')=E_{\theta'}\big[\log p_\theta(Y,Z)\mid Y=y\big]$$
>
> 迭代：$\theta^{(m+1)}=\arg\max_\theta Q(\theta,\theta^{(m)})$。

> **推导** · EM 单调提升对数似然。令 $\ell(\theta)=\log p_\theta(y)$，$r(\theta,z)=p_\theta(z\mid y)/p_\theta(y)$ 为后验。则 Jensen 不等式给出
>
> $$\ell(\theta)-Q(\theta,\theta')\ \ge\ E_{\theta'}\Big[\log\frac{r(\theta',Z)}{r(\theta,Z)}\Big] = \mathrm{KL}\big(r(\theta')\ \|\ r(\theta)\big)\ \ge0$$
>
> 第一个不等号来自 Jensen 对凹函数 $\log$（$E\log a\le\log Ea$），第二个来自 KL 非负。**等号成立当且仅当** $r(\theta)=r(\theta')$（后验不变），此时 $\theta=\theta'$ 也是极小。因此每次迭代不降低似然。

> **坑** · EM 只是**收敛到驻点**，不保证全局最大（除非似然凹，如高斯混合的对数似然在无约束情形下是凹的但约束使其非凹）。初始化敏感，会陷局部极值。这是第 14 章反复强调的。

### 0.4.4 坐标下降与随机梯度（O4） {#s-0-4-4}

> **基础知识** · 坐标下降
>
> 对可分目标 $F(\beta)=\sum_{j=1}^p F_j(\beta_j)$，固定其余坐标只优化第 $j$ 个：
>
> $$\beta_j\leftarrow\arg\min_t F_j(t)$$
>
> 每次都单调下降（因为 $F(\beta)\le F(\beta^{\pm})$）。

**lasso 的坐标下降解（ESL (3.84)）**：$R(\beta)=\frac{1}{2N}\|y-X\beta\|_2^2+\lambda\|\beta\|_1$。只更新 $\beta_j$，令残差 $r=y-X\beta+X_j\beta_j$，则

$$
\beta_j \leftarrow S\!\left(\frac1N x_j^\top r,\ \lambda\right),\qquad
S(u,\lambda)=\begin{cases}u,&|u|\le\lambda\\ u-\lambda\,\mathrm{sign}(u),&|u|>\lambda\end{cases}
$$

> **推导** · 固定其它坐标，目标变成
>
> $$\frac1{2N}\sum_i\big(r_i-x_{ij}\beta_j\big)^2+\lambda|\beta_j|$$
>
> 对 $\beta_j$ 求导：$-\frac1N x_j^\top(r-x_j\beta_j)+\lambda\,\mathrm{sign}(\beta_j)=0$。分三种情形：
>
> - $\beta_j>0$：$\frac1N x_j^\top r=\frac1N\|x_j\|^2\beta_j+\lambda \Rightarrow \beta_j=\frac{1}{N\|x_j\|^2}(x_j^\top r-N\lambda)$
> - $\beta_j<0$：同理 $\beta_j=\frac{1}{N\|x_j\|^2}(x_j^\top r+N\lambda)$
> - $\beta_j=0$：需要 $|x_j^\top r/N|\le\lambda$（KKT 互补松弛）
>
> 三种情形合并即软阈值算子 $S$。ESL (3.84) 写的就是这个形式：$\tilde\beta_j^{(j)}\leftarrow S\!\big(\frac1N\sum_i\tilde y_i^{(j)}x_{ij},\ \lambda\big)$。

> **坑** · 常见的一个错误推论：「软阈值算子满足 $S(u-\lambda,\lambda)=S(u,\lambda)-\lambda$，所以可以把 $\lambda$ 提出来」。**这个恒等式不成立。** 取 $u=1.5\lambda$：左边 $S(0.5\lambda,\lambda)=0.5\lambda$，右边 $S(1.5\lambda,\lambda)-\lambda=(1.5\lambda-\lambda)-\lambda=-0.5\lambda$，两者不等。真正成立的是**正齐次性** $S(cu,c\lambda)=c\,S(u,\lambda)\ (c>0)$，以及只在 $u\ge0$（或只在 $u\le0$）一侧成立的$S(u+\lambda,\lambda)=S(u,\lambda)+\lambda$。这类「看起来显然的平移」正是坐标下降实现里最常见的 bug 来源——务必直接用上面那个 $S$ 去验，而不是套恒等式。\n> **随机梯度**：$F(\beta)=E[L(\beta,Z)]$ 时，$\nabla F=E[\nabla L(\beta,Z)]$，用单样本梯度 $\nabla L(\beta,Z^{(m)})$ 代替即 SGD。步长 $\gamma_m=1/(\lambda_0+\lambda_1m)$ 保证 $\sum\gamma_m=\infty$ 且 $\sum\gamma_m^2<\infty$，从而几乎必然收敛。

### 0.4.5 判决理论：风险与贝叶斯分类（O5） {#s-0-4-5}

> **基础知识** · 条件风险
>
> 动作空间 $\{1,\dots,K\}$，损失 $L(g,Y)$，后验 $p_k(x)=P(Y=k\mid X=x)$。条件风险
>
> $$R(g\mid x)=\sum_k L(g,k)\,p_k(x)$$
>
> **贝叶斯分类规则**：$g^*(x)=\arg\min_k R(g=k\mid x)$。0-1 损失下 $R(g=k\mid x)=p_k(x)$，故规则是「选后验最大的类」。
>
> 推导：全风险 $R(g)=E[R(g(X)\mid X)]$（塔性质）。若 $g^*$ 逐点最小化条件风险，则
>
> $$R(g^*)=E[R(g^*\mid X)]\le E[R(g\mid X)]=R(g)$$
>
> 这是「最优」的严格含义，也是第 4 章所有「贝叶斯最优」陈述的依据。

> **坑** · $p_k(x)$ 是**后验**，不是**先验**。0-1 损失下先验与似然加权后相互抵消，但判别分析里类先验 $\pi_k$ 必须保留（第 4 章 LDA 公式中 $\log\frac{\pi_k}{\pi_l}$ 一项）。

---

## 0.5 N · 数值与计算量 {#s-0-5}

### 0.5.1 病态、条件数与秩（N1） {#s-0-5-1}

> **基础知识**
>
> - **秩**：$\mathrm{rank}(X)=\dim\,\mathrm{col}(X)$。$\mathrm{rank}(X)=\mathrm{rank}(X^\top X)$，$\mathrm{rank}(X)\le\min(N,p)$。
> - **条件数**：$\kappa(X)=\sigma_{\max}/\sigma_{\min}$（$X$ 满秩时）。相关矩阵 $\mathrm{cor}(x_j,x_l)$ 接近 $\pm1$ 时 $\kappa\to\infty$，最小二乘解剧烈震荡。
> - **机器精度**：双精度 $\epsilon\approx2.2\times10^{-16}$。**有效秩** $d_{\rm eff}=\sum_j \sigma_j^2/\sum_{j\le d}\sigma_j^2$ 是判断数值可信度的实用指标。

> **数值** · 正确的做法是 SVD 或 QR，而不是显式求 $(X^\top X)^{-1}$。理由由误差分析给出：
>
> $$\frac{\|\Delta\beta\|}{\|\beta\|}\ \lesssim\ \kappa_2(X)\,\epsilon$$
>
> 推导思路：$\Delta\beta = A^{-1}\Delta b$，$\frac{\|\Delta\beta\|}{\|\beta\|}\le\frac{\|A^{-1}\|}{\|A\|}\frac{\|A\|\|\Delta b\|}{\|\Delta b\|}$，第二项为条件数。$\kappa_2(X)=\sqrt{\lambda_{\max}(X^\top X)/\lambda_{\min}(X^\top X)}$，而 $\lambda_{\min}(X^\top X)=\sigma_{\min}^2$，故 $\kappa_2(X^\top X)=\kappa_2(X)^2$ —— **正规方程把条件数平方了**，这是不能直接消元的数值理由。

### 0.5.2 复杂度计数（N2） {#s-0-5-2}

> **基础知识** · 矩阵乘法的代价
>
> 直接矩阵乘法 $N\times p$ 乘 $p\times N$ 需 $O(N^2p)$。但用 SVD 可以看清「有效秩」才是真正的计算量：

| 运算 | 直接算法 | 用 SVD / 低秩 |
|---|---|---|
| $(X^\top X)^{-1}X^\top y$ | $O(N^2p)$ | $O(Npr)$，$r=\mathrm{rank}(X)$ |
| 全部 $N$ 个 $\hat y_i$ | 同上 | $O(Npr)$ |
| 第 $i$ 个留一法预测 | $O(pr^2)$ | $O(r)$ |

> **结果** · 第 5 章的 LOOCV 精确公式之所以重要，正是因为它把「$N$ 次重新拟合」的 $O(N^2p^2)$ 降到 $O(Npr)$。

### 0.5.3 梯度 / 随机梯度复杂度（N3） {#s-0-5-3}

> **基础知识** · 一次迭代代价
>
> 逻辑回归（4 章）与 lasso（5、18 章）的坐标下降每次更新需要 $x_j^\top r$，代价 $O(N)$；一轮扫过 $p$ 个坐标是 $O(Np)$。SGD 每次是 $O(p)$。所以 SGD 的优势在 $p$ 极大时的**内存**而非时间：只需驻留梯度向量。

---

## 0.6 这套工具怎么用在后面的章节里 {#s-0-6}

| 章节 | 主要依赖 | 典型例子 |
|---|---|---|
| 2 监督学习概述 | L2, P1, P2, O5 | 偏差–方差分解、交叉验证 |
| 3 线性回归 | L1–L4, P4 | 正规方程 (3.6)、$F$ 统计量 (3.13) |
| 4 线性分类 | P3, O5, C2 | 逻辑回归的极大似然 |
| 5 基展开与正则化 | L1, C2, O2, O4 | lasso 的 KKT (3.58) |
| 6 核平滑 | C3, P1 | 核估计的偏差 $-m_K/h$ |
| 7 模型评估 | L2, P1, P2 | LOOCV 公式 |
| 8 模型推断与平均 | P1, P2 | bagging 方差公式 |
| 9–10 树与提升 | O1, O5, P2 | AdaBoost 的 margin 推导 |
| 11 神经网络 | L4, C1 | 后向传播、权重衰减 |
| 12 支持向量机 | O1, O2, L1 | 最大间隔几何与对偶 |
| 13 最近邻 | O5, P2 | 一致性收敛 |
| 14 无监督学习 | L3, O3 | PCA 推导、EM |
| 15–16 森林与集成 | P2 | 平均的偏差–方差公式 |
| 17 图模型 | P3 | Gibbs 分布、配分函数 |
| 18 高维问题 | L3, O2 | lasso 的支撑集、LOCO |

<a class="src" href="index.html">返回封面</a>