## 线性代数

@src 全书第 1 章 · [1.5 Notation and Simple Matrix Algebra](../isl/02-introduction.html#s-notation-and-simple-matrix-algebra)

第 1 章只用了几行矩阵记号，但后面每一章都建立在同一套代数上。这里把它讲全。

### 向量与矩阵的基本运算

$$
x_i = \begin{pmatrix} x_{i1} \\ x_{i2} \\ \vdots \\ x_{ip} \end{pmatrix} \in \R^{p}, \quad
X = \begin{pmatrix} x_{11} & \cdots & x_{1p} \\ \vdots & & \vdots \\ x_{n1} & \cdots & x_{np} \end{pmatrix} \in \R^{n \times p}
$$

**基础**：$X$ 有 $n$ 行 $p$ 列；第 $i$ 行 $x_i^\top$ 是第 $i$ 个观测的 $p$ 个变量值。
向量默认是列向量，所以 $x_i$ 写成列、$x_i^\top$ 写成行。

**结论**：$X^\top X$ 是 $p \times p$ 矩阵，$(X^\top X)_{jk} = \sum_{i=1}^n x_{ij}x_{ik}$，
即第 $j$、$k$ 个变量的**列交叉乘积和**。$XX^\top$ 是 $n \times n$，
$(XX^\top)_{ik} = \sum_{j=1}^n x_{ij}x_{kj}$，即第 $i$、$k$ 个**观测**之间的相似度。
这个区别在后面很重要：$X^\top X$ 管变量，$XX^\top$ 管样本。

**易错点**：$x_i$ 是 $p \times 1$，$x_i^\top$ 是 $1 \times p$。$x_{ij}$ 的第一个下标永远是行
（观测），第二个永远是列（变量）。原书式 (1.1) 特意把 $x_{ij}$ 画成列向量列出，
就是为了强调这一点。

### 均值、平方和与协方差矩阵

$$
\bar{x}_j = \frac{1}{n}\sum_{i=1}^{n} x_{ij}, \quad
S_{jj} = \frac{1}{n-1}\sum_{i=1}^{n}\left(x_{ij} - \bar{x}_j\right)^2, \quad
S_{jk} = \frac{1}{n-1}\sum_{i=1}^{n}\left(x_{ij}-\bar{x}_j\right)\left(x_{ik}-\bar{x}_k\right)
$$

**基础**：样本均值、样本方差（除以 $n-1$ 使其无偏）、样本协方差。
注意 $S$ 的对角线是方差、非对角线是协方差，合起来是一个 $p \times p$ 对称矩阵。

**推导**（为什么是 $n-1$）：设 $x_{ij} = \mu_j + \eta_{ij}$，$E(\eta_{ij}) = 0$。则

$$
E\left[\sum_{i=1}^n (x_{ij}-\bar{x}_j)^2\right] = E\left[\sum_i \eta_{ij}^2\right] - nE(\bar{x}_j - \mu_j)^2 = n\sigma_j^2 - n\cdot\frac{\sigma_j^2}{n} = (n-1)\sigma_j^2
$$

最后一步用了 $\mathrm{Var}(\bar{x}_j) = \sigma_j^2/n$（$n$ 个独立同分布项平均后方差缩小 $n$ 倍）。

**结论**：除以 $n-1$ 恰好抵消抽样波动，$S_{jj}$ 是 $\sigma_j^2$ 的无偏估计。

**易错点**：书里 $S_{jj}$ 前面写的是 $1/(n-1)$，但在最小二乘的正规方程里除以
$n$ 或 $n-1$ 会差一个整体因子，$\hat{\boldsymbol\beta}$ 不受影响。真正会受影响的是
$\hat{\sigma}^2$（MSE）与标准误，所以别混用。

### 矩阵求逆存在的条件

$$
(X^\top X)^{-1} \text{ 存在} \iff X \text{ 列满秩} \iff \text{不存在 } \boldsymbol\beta \ne 0 \text{ 使 } X\boldsymbol\beta = 0
$$

**基础**：$X \in \R^{n \times p}$，$X^\top X$ 是 $p \times p$。$X$ 列满秩意味着 $p \le n$
且 $p$ 个变量线性无关。

**推导**：对任意 $\boldsymbol\beta \ne 0$，

$$
\boldsymbol\beta^\top X^\top X \boldsymbol\beta = (X\boldsymbol\beta)^\top(X\boldsymbol\beta) = \|X\boldsymbol\beta\|_2^2 > 0
$$

（最后一步用 $X\boldsymbol\beta \ne 0$）。所以 $X^\top X$ 对任意非零向量二次型为正，
即正定，从而可逆。

**结论**：这就是第 3 章反复提到的警告——变量共线性会让 $X^\top X$ 接近奇异，
求逆数值上不稳定，$\hat{\beta}$ 的方差爆炸。

**易错点**：$X^\top X$ 可逆 $\iff$ $X$ 列满秩，**不需要** $n \ge p$ 之外的条件，
但也不意味着 $X$ 本身可逆（$X$ 是 $n\times p$，$n>p$ 时必然不可逆）。
用 $(X^\top X)^{-1}X^\top$ 代替 $X^+$（伪逆）是对的。

### 正交性、最小二乘与投影

$$
\hat{\boldsymbol\beta} = \underset{\boldsymbol\beta}{\arg\min} \|X\boldsymbol\beta - \mathbf{y}\|_2^2
$$

**基础**：$\|\cdot\|_2$ 是欧氏范数，$\|\mathbf{v}\|_2^2 = \mathbf{v}^\top\mathbf{v}$。

**推导**（正规方程的来历）。设 $L(\boldsymbol\beta) = (X\boldsymbol\beta - \mathbf{y})^\top(X\boldsymbol\beta - \mathbf{y})$。
逐项展开：

$$
L = \boldsymbol\beta^\top X^\top X\boldsymbol\beta - 2\boldsymbol\beta^\top X^\top \mathbf{y} + \mathbf{y}^\top\mathbf{y}
$$

对 $\boldsymbol\beta$ 求梯度：$X^\top X$ 是对称的，所以

$$
\nabla_{\boldsymbol\beta} L = 2X^\top X\boldsymbol\beta - 2X^\top \mathbf{y} = 0
$$

（最后一项 $\mathbf{y}^\top\mathbf{y}$ 与 $\boldsymbol\beta$ 无关，梯度为 0。）
移项得 $X^\top X\hat{\boldsymbol\beta} = X^\top\mathbf{y}$，即
$\hat{\boldsymbol\beta} = (X^\top X)^{-1}X^\top\mathbf{y}$。

**结论**：$L$ 是凸函数（$X^\top X \succeq 0$），驻点即全局最小。几何上，$X\hat{\boldsymbol\beta}$
是 $\mathbf{y}$ 在 $\mathrm{col}(X)$ 上的**正交投影**：残差 $\mathbf{y} - X\hat{\boldsymbol\beta}$
与每个 $x_i$ 都正交。

**易错点**：不能写 $(X^\top X)^{-1} = X^{-1}(X^\top)^{-1}$，除非 $X$ 本身方阵可逆，
而这正是共线性时要避免的情形。数值实现里更常用 QR 分解或 SVD 而不是显式求逆。

### 特征分解、协方差矩阵与 PCA

$$
\Sigma = \frac{1}{n-1}(X-\mathbf{1}\bar{x}^\top)^\top(X-\mathbf{1}\bar{x}^\top)
= \frac{1}{n-1}X_c^\top X_c \succ 0
$$

$$
\Sigma = V\Lambda V^\top, \quad \Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_p), \quad \lambda_1 \ge \lambda_2 \ge \cdots
$$

**基础**：对称矩阵一定可正交对角化（谱定理）；特征向量张成正交子空间；
$\lambda_1$ 是最大的特征值，$\lambda_1 = \max_{\|v\|=1} v^\top \Sigma v$（瑞利商）。

**推导**（为什么第一主成分是最大特征值对应的方向）。设 $v_1$ 为单位向量，
考虑 $v_1$ 方向上的方差 $\mathrm{Var}(v_1^\top x) = v_1^\top \Sigma v_1$。
由特征方程 $\Sigma v_1 = \lambda_1 v_1$，左乘 $v_1^\top$（用 $v_1^\top v_1 = 1$）得
$v_1^\top\Sigma v_1 = \lambda_1$。而对任意单位 $v$，写成 $v = \sum_k a_k v_k$，
则 $v^\top\Sigma v = \sum_k a_k^2\lambda_k \le \lambda_1\sum_k a_k^2 = \lambda_1$。

**结论**：最大方差方向就是 $\lambda_1$ 对应的特征向量；解释的总方差比例是
$\lambda_k / \sum_{j}\lambda_j$。第 12 章的 PCA 就是这两行。

**易错点**：$\Sigma$ 里的 $X_c$ 必须中心化，否则第一主成分只是样本均值方向。
另一个坑：当 $n \ll p$ 时 $X^\top X$ 是 $p \times p$ 但秩最多 $n$，有 $p-n$ 个零特征值，
必须用 SVD 而不是 $\Sigma$ 的特征分解（或者说 $\Sigma = \frac{1}{n-1}V_r\Sigma_r V_r^\top$，
只用非零部分）。

### 奇异值分解

$$
X = U D V^\top, \quad D = \mathrm{diag}(\sigma_1,\ldots,\sigma_r,\underbrace{0,\ldots,0}_{p-r}), \quad \sigma_k = \sqrt{\lambda_k(X^\top X)}
$$

**基础**：$V$ 的列是 $X^\top X$ 的特征向量，$\sigma_k$ 是对应特征值的平方根，
$U$ 的列是 $XV$ 归一化后的结果。

**推导**（$U$ 的列从哪来）：设 $V = (v_1,\ldots,v_p)$，$X^\top Xv_k = \sigma_k^2 v_k$。
左乘 $X$：$X^\top Xv_k = \sigma_k^2 v_k$ 两边整理得 $X(X^\top X)v_k = \sigma_k^2 Xv_k$，即
$\left(X^\top X\right)^2 Xv_k = \sigma_k^4 Xv_k$。所以 $u_k = Xv_k/\sigma_k$ 是
$(X^\top X)^2$ 的特征向量，$\|Xv_k\| = \sqrt{v_k^\top X^\top Xv_k} = \sigma_k$，归一化即得。

**结论**：$X^\top X = V D^2 V^\top$，$XX^\top = U D^2 U^\top$。用 SVD 可以完全避开求逆：
$X^+ = V D^{-1}U^\top$（只对 $\sigma_k > 0$ 取逆）。岭回归的解
$\hat{\boldsymbol\beta}_\lambda = V\,\mathrm{diag}(\sigma_k/(\sigma_k^2+\lambda))\,U^\top\mathbf{y}$
就是这样写出来的（见第 6 章）。

**易错点**：$\sigma_k = 0$ 的方向上 $D^{-1}$ 不存在。SVD 把「$X^\top X$ 不可逆」
这件事显式地暴露在 $\sigma_k$ 上，比盲目求逆稳健得多。

### 施密特正交化

给定线性无关的 $a_1,\ldots,a_m$：

$$
u_1 = a_1, \quad u_k = a_k - \sum_{j<k}\frac{u_j^\top a_k}{\|u_j\|_2^2}u_j, \quad
e_k = \frac{u_k}{\|u_k\|_2}
$$

**基础**：$u_j$ 两两正交且 $\mathrm{span}\{u_1,\ldots,u_k\} = \mathrm{span}\{a_1,\ldots,a_k\}$。

**推导**（正交性）：对 $j<k$，$u_j^\top u_k = u_j^\top a_k - \sum_{l<k}\frac{u_l^\top a_k}{\|u_l\|^2}u_l^\top u_j
= u_j^\top a_k - \frac{u_j^\top a_k}{\|u_j\|^2}\|u_j\|^2 = 0$。用归纳假设「$u_1,\ldots,u_{k-1}$ 已正交」。

**结论**：这一步在第 9 章构造 SVM 的基函数、第 10 章的神经网络初始化里都会用到；
第 7 章的正交基展开也建立在此。
