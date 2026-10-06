## 微积分与矩阵求导

@src 全书通用 · 第 3、4、7、9、10 章的推导都要用到

统计学习的核心动作是「求一个函数在某个集合上的最小值」。这需要三样东西：
怎么算梯度、怎么判断最小值唯一、怎么求带约束的解。

### 偏导数与梯度

设 $g:\R^p \to \R$。偏导数与梯度：

$$
\frac{\partial g}{\partial x_j} = \lim_{h\to 0}\frac{g(x + he_j) - g(x)}{h}, \qquad
\nabla g(x) = \begin{pmatrix} \partial g/\partial x_1 \\ \vdots \\ \partial g/\partial x_p \end{pmatrix}
$$

**基础**：$\nabla g$ 的第 $j$ 个分量是沿第 $j$ 个坐标轴的方向导数。梯度指向 $g$ 增长最快的方向。

**结论**：无约束极小点的必要条件是 $\nabla g(x^\ast) = 0$（驻点条件）。
若 $g$ 是二次函数，这一条就够了；一般函数还要检查是不是极小点。

**易错点**：$\nabla g$ 是**列向量**且形状为 $p \times 1$。梯度下降的更新
$x \leftarrow x - \eta \nabla g(x)$ 里 $\eta > 0$ 叫学习率；符号写成加号就是梯度上升。
不要把 $\partial g/\partial x$ 和 $\partial x/\partial g$ 搞混。

### 链式法则（向量形式）

$$
\frac{\partial}{\partial x}\big(f(Ax)\big) = A^\top \nabla f(Ax), \qquad
\frac{\partial}{\partial x}\big(x^\top A x\big) = (A + A^\top)x
$$

**推导**（第一个式子）。设 $z = Ax$，则 $x = A^{-1}z$（$A$ 可逆时），于是
$f(Ax) = f(z) = g(z)$，$\nabla_z f(z) = \nabla_z g(z)$。由多元链式法则，
$\nabla_x (g\circ A) = A^\top \nabla_z g$。$A$ 不可逆时逐分量验证同样成立。

**推导**（第二个式子，更常用）。逐分量展开，用乘积法则：

$$
\frac{\partial}{\partial x_k}\left(\sum_{i}\sum_j x_i A_{ij} x_j\right)
= \sum_j A_{kj}x_j + \sum_i x_i A_{ik}
= (A^\top x)_k + (Ax)_k
$$

第三步把 $i,j$ 的哑指标重新命名为统一的 $k$，并注意 $A$ 与 $A^\top$ 的行列互换关系。

**结论**：$x^\top A x$ 在 $A$ **对称**时梯度简化为 $2Ax$。这是最小二乘正规方程
$\nabla = 2X^\top X\boldsymbol\beta - 2X^\top\mathbf{y} = 0$ 的来源。

**易错点**：$x^\top A x$ 只有当 $A$ 对称时才等于 $x^\top \frac{A + A^\top}{2}x$；
非对称时只有对称部分有贡献（斜对称部分被消掉，因为 $x^\top K x = 0$ 对任意斜对称 $K$ 成立）。
写梯度前先检查对称性，能省掉一半推导。

### 雅可比矩阵与海森矩阵

$$
J_{ij} = \frac{\partial g_i}{\partial x_j} \ (m \times p), \qquad
H_{jk} = \frac{\partial^2 g}{\partial x_j \partial x_k} \ (p \times p)
$$

**推导**（$H$ 是对称的）：混合偏导可交换（$g$ 二阶连续可微时），$\partial^2 g/\partial x_j\partial x_k
= \partial^2 g/\partial x_k\partial x_j$，所以 $H = H^\top$。

**结论**：$H \succ 0$（所有特征值为正）$\iff$ $g$ 在该点严格凸，驻点是唯一全局最小。
$H \succeq 0$（半正定）$\iff$ 凸，驻点集是凸集。第 6 章岭回归的唯一性证明、
第 4 章逻辑回归的凸性，都用这个判据。

**易错点**：「$H$ 正定」比「$H$ 对称」强。只对称不定（如 $\mathrm{diag}(1,-1)$）的点是鞍点，
驻点但不是极小点——这正是过拟合高维模型时训练误差还能继续下降的原因。

### 泰勒展开

$$
g(x + \Delta) = g(x) + \nabla g(x)^\top \Delta + \frac{1}{2}\Delta^\top H(x)\Delta + o(\|\Delta\|^2)
$$

**基础**：多元泰勒公式，$g$ 二阶可微。

**结论**：牛顿法就是让这个展开沿 $\Delta$ 的一阶项为零来选步长：
$0 = \nabla g^\top\Delta + \frac12\Delta^\top H\Delta$。对 $\Delta$ 求导并令其为零
（把 $\Delta$ 当变量），得 $\nabla g + H\Delta = 0$，即 $\Delta = -H^{-1}\nabla g$。

**易错点**：牛顿法只有在 $H$ 正定时才保证单调下降；$H$ 奇异时要用伪逆或阻尼
（Levenberg–Marquardt），这正是第 10 章训练神经网络的实践做法。

### 链式法则：反向传播的形式

对复合 $L(\boldsymbol\theta) = g(\boldsymbol\theta^{(L)}, \ldots, \boldsymbol\theta^{(1)})$，
中间层 $\boldsymbol\theta^{(\ell)} = h_\ell(\boldsymbol\theta^{(\ell-1)})$：

$$
\frac{\partial L}{\partial \boldsymbol\theta^{(\ell-1)}}
= \left(\frac{\partial \boldsymbol\theta^{(\ell)}}{\partial \boldsymbol\theta^{(\ell-1)}}\right)^\top
  \frac{\partial L}{\partial \boldsymbol\theta^{(\ell)}}
$$

**基础**：这是链式法则的矩阵形式——误差从输出层往回传，每层把梯度转置后相乘。

**结论**：$L$ 关于第 $1$ 层参数的梯度是各层雅可比转置的连乘。第 10 章的式 (10.23)
到 (10.25) 就是这个式子在两层网络上的具体展开。

**易错点**：雅可比的维度方向容易搞反：$J = \partial \boldsymbol\theta^{(\ell)}/
\partial \boldsymbol\theta^{(\ell-1)}$ 是 $p_\ell \times p_{\ell-1}$，所以回传时要转置。
另外梯度是**对参数**的偏导，不是对输入的偏导，两者不能混。

### 约束极值：拉格朗日乘子法

求 $\min_x g(x)$ s.t. $h_j(x) = 0$（$j=1,\ldots,q$）。构造

$$
\mathcal{L}(x,\lambda) = g(x) - \sum_{j=1}^{q}\lambda_j h_j(x)
$$

**推导**。在最优点处，等高面 $g = c$ 与约束曲面相切。设曲线的切向量为 $v$
（$Dh(x)v = 0$）。相切意味着 $\nabla g \cdot v = 0$ 对所有这样的 $v$ 成立，
即 $\nabla g$ 落在 $\{v : Dh\,v=0\}$ 的正交补里，而这个正交补由 $Dh$ 的行向量张成，
所以 $\nabla g = \sum_j \lambda_j \nabla h_j$。这等价于 $\nabla_x\mathcal{L} = 0$。

**结论**：KKT 条件——(i) 驻点 $\nabla g = \sum_j\lambda_j\nabla h_j$；
(ii) 可行性 $h_j = 0$；(iii) 互补松弛（对不等式约束 $\lambda_j \ge 0$，$\lambda_j h_j = 0$）。

**易错点**：$\lambda$ 的符号约定在不同书里不一致（写 $+ \sum \lambda_j h_j$ 或
$-\sum\lambda_j h_j$ 都有人用）。关键是 $\nabla g$ 必须是各约束梯度的**线性组合**。
第 9 章 SVM 用的是「最小化 $g$，对偶最大化」，所以最后一步是取对偶的上确界——
这一步很多人会忘。

### 软阈值算子

$$
S_\lambda(t) = \begin{cases} t - \lambda & t > \lambda \\ 0 & |t| \le \lambda \\ t + \lambda & t < -\lambda \end{cases}
\quad\text{即}\quad S_\lambda(t) = \mathrm{sign}(t)\max(|t|-\lambda,\, 0)
$$

**推导**（它是最小化问题的解）。考虑一维问题
$\min_\beta \frac{1}{2}(\beta - z)^2 + \lambda|\beta|$。

- 若 $z > \lambda$：$|\beta|$ 在 $\beta>0$ 区域解析，目标函数在 $\beta>0$ 上的导数
  为 $\beta - z + \lambda$，令其为 0 得 $\beta = z - \lambda > 0$，自洽，解就是 $z-\lambda$。
- 若 $|z| \le \lambda$：在 $\beta > 0$ 区域导数 $\beta - z + \lambda > 0$（恒正，函数递增），
  在 $\beta < 0$ 区域导数 $\beta - z - \lambda < 0$（恒负，函数递减），所以极小在 $\beta = 0$，此时次梯度 $\lambda\,\partial|\beta|\big|_{0} \in [-\lambda,\lambda]$ 能吸收掉残差。
- 若 $z < -\lambda$：对称得 $\beta = z + \lambda$。

**结论**：$S_\lambda$ 是「LASSO 收缩」的一维版本；它的分段线性形状正是 LASSO 解稀疏的原因——
落在 $[-\lambda,\lambda]$ 内的坐标被**精确置零**。

**易错点**：这里的 $\lambda$ 与岭回归的 $\lambda$ 数值不可比（一个是 $\ell_1$ 罚项，
一个是 $\ell_2$ 罚项，量纲差一个因子）。不要拿两章的 $\lambda$ 直接对比。
另外 $S_\lambda$ 在 0 处不可导，必须用次梯度，写成导数会出错。
