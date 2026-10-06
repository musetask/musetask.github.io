## 正则化、约束与坐标下降

@src 全书通用 · 第 6、7、9 章的核心计算工具

### 惩罚化与约束的等价

$$
\min_{\boldsymbol\beta}\; \underbrace{\mathrm{RSS}(\boldsymbol\beta)}_{拟合} + \underbrace{\lambda\,\mathrm{pen}(\boldsymbol\beta)}_{惩罚}
\qquad\Longleftrightarrow\qquad
\min_{\boldsymbol\beta}\; \mathrm{RSS}(\boldsymbol\beta)\quad \text{s.t.}\quad \mathrm{pen}(\boldsymbol\beta) \le c
$$

**推导**（两个问题解集相同）。设 $L(\boldsymbol\beta) = \mathrm{RSS}$，$P = \mathrm{pen}$，
且 $P$ 是凸的。

- 方向一：给定无约束问题的最优拉格朗日乘子 $\lambda^\ast \ge 0$，则惩罚问题的解
  $\hat\beta$ 满足 $\nabla L(\hat\beta) + \lambda^\ast\nabla P(\hat\beta) = 0$。
  又 $\nabla P(\hat\beta)^\top\hat\beta \le 0$（因为 $P(0)$ 是自然的最小点，见下），
  所以
  $L(\hat\beta) = -(\lambda^\ast)^{-1}\nabla L(\hat\beta)^\top \hat\beta
  \ge -(\lambda^\ast)^{-1}\nabla P(\hat\beta)^\top\hat\beta \ge 0$，
  即 $L(\hat\beta) \le L(\boldsymbol\beta) + \lambda^\ast P(\hat\beta)$ 对所有 $\boldsymbol\beta$ 成立。
  整理即 $\hat\beta$ 是带约束问题的最优解。
- 方向二：取 $c = P(\hat\beta_{\text{pen}})$，由对偶强对偶（Slater 条件：$P(0) < c$，
  在岭回归里只要 $c>0$ 就成立）得两个问题最优值相等。

**结论**：$\lambda$ 与 $c$ 一一对应（$\lambda$ 越大，$c$ 越小）。所以「调 $\lambda$」和
「调约束大小」是同一件事的两种说法。LASSO 里这个结论只在 $P$ 是 $\ell_1$ 范数时
需要非凸性论证，细节见第 6 章。

**易错点**：这个等价性要求 $P$ **凸**且 $P(0) = 0$ 是最小点。$\ell_0$ 范数（非零元素个数）
非凸，等价性不成立——这正是 $\ell_0$ 惩罚难优化、要用贪心近似的原因。

### 岭回归解的显式形式

$$
\hat{\boldsymbol\beta}_\lambda = (X^\top X + \lambda I)^{-1} X^\top \mathbf{y}
$$

**推导**。目标函数 $F(\boldsymbol\beta) = \mathrm{RSS} + \lambda\|\boldsymbol\beta\|_2^2$。
把 $\|\boldsymbol\beta\|^2$ 展开进 RSS 的表达式并求梯度：

$$
\nabla F = -2X^\top(\mathbf{y} - X\boldsymbol\beta) + 2\lambda\boldsymbol\beta
= 2\big[(X^\top X + \lambda I)\boldsymbol\beta - X^\top\mathbf{y}\big] = 0
$$

所以 $(X^\top X+\lambda I)\hat{\boldsymbol\beta}_\lambda = X^\top\mathbf{y}$，两边左乘逆矩阵即得。
而 $X^\top X$ 半正定、$\lambda I$ 正定，故 $X^\top X + \lambda I$ 正定可逆，**任何** $\lambda>0$
都保证唯一解——这是岭回归相对 OLS 的关键好处。

**易错点**：$I$ 是 $p\times p$ 单位阵，对应**截距列也参与惩罚**。原书明确说明
标准做法是把第 1 列设为全 1（截距）**并且不惩罚它**（用 $D$ 矩阵或中心化变量实现）。
如果直接把截距一起惩罚掉，$\hat\beta_0$ 会被拉向 0，模型解释就错了。

### 特征值视角与收缩率

设 $X^\top X = V\Sigma^2V^\top$（SVD/特征分解），$\mathbf{y}$ 在 $V$ 上的坐标为 $\tilde{\mathbf{y}}$。
则

$$
\hat{\boldsymbol\beta}_\lambda = V\,\mathrm{diag}\!\left(\frac{\sigma_k^2}{\sigma_k^2+\lambda}\right)\tilde{\mathbf{y}}
$$

**推导**。代入 $\hat{\boldsymbol\beta}_\lambda = (V\Sigma^2V^\top + \lambda I)^{-1}V\Sigma^2 U^\top\mathbf{y}$。
用 $V^\top V = I$ 得 $(V\Sigma^2V^\top+\lambda I)^{-1} = V(\Sigma^2+\lambda I)^{-1}V^\top$。
所以 $\hat{\boldsymbol\beta}_\lambda = V(\Sigma^2+\lambda I)^{-1}\Sigma^2V^\top\mathbf{y}
= V\,\mathrm{diag}\big(\frac{\sigma_k^2}{\sigma_k^2+\lambda}\big)\,\tilde{\mathbf{y}}$。

**结论**：第 $k$ 个方向的系数被**乘上** $\frac{\sigma_k^2}{\sigma_k^2+\lambda} \in (0,1)$。
$\lambda$ 越大，$\sigma_k$ 小的方向被压得越厉害；$\lambda\to\infty$ 时全部系数归零。
这就是「ridge 把系数拉向 0，但不会精确置零」这句话的全部数学内容。

**易错点**：收缩发生在特征值**小**的方向，不是系数小的方向。系数小但方差方向大的
变量会保留系数；方差小但系数大的变量被压掉——这正是 ridge 稳定共线性变量的机制。

### LASSO 的软阈值与解的稀疏性

LASSO 目标 $\min \frac{1}{2n}\mathrm{RSS} + \lambda\|\boldsymbol\beta\|_1$，
在正交基（$X^\top X = I$，可由标准化 + 旋转达到）下坐标可分离：

$$
\hat\beta_k = S_{\lambda n}\big(\tilde y_k\big) = \mathrm{sign}(\tilde y_k)\max(|\tilde y_k| - \lambda n, 0)
$$

**结论**：$\ell_1$ 的分段线性罚项有平台区，落在平台内的坐标被精确置零
（软阈值推导见微积分那一节）。所以 LASSO 解天然稀疏，且「变量选择」与「系数估计」
一步完成，不需要先筛选变量。

**易错点**：稀疏性**依赖**系数坐标轴的选择。原书指出，如果先做正交变换
（把相关变量合成主成分）再罚 $\ell_1$，就变成了 PCA 式降维，$\ell_1$ 在正交基上
不再产生稀疏。$\ell_1$ 的价值恰恰在于它在原始变量轴上制造平台。

### 坐标下降

对 $\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta} F(\boldsymbol\beta)$，循环：

1. 固定 $\beta_{k}$ 以外所有分量，$F$ 关于 $\beta_k$ 是二次的（因为 $F$ 是可分的二次函数）。
2. 令该方向导数为 0，闭式解出 $\beta_k$；
3. $k \leftarrow k + 1$（循环）直到收敛。

以 LASSO 为例，记 $r = \mathbf{y} - \sum_{j \ne k}\beta_j x_j$（当前残差），
则关于 $\beta_k$ 的目标为 $\frac{1}{2n}\|r + \beta_k x_k\|^2 + \lambda|\beta_k|$，
导数为 $\frac{1}{n}x_k^\top(r + \beta_k x_k) + \lambda\,\partial|\beta_k| = 0$，
在 $\beta_k \ne 0$ 时 $\beta_k = \frac{x_k^\top r}{n\|x_k\|^2} - \frac{n\lambda}{\|x_k\|^2}$，
再经软阈值即得一步更新式。

**结论**：LASSO 没有闭式解（$\ell_1$ 不可微），但坐标下降收敛快且实现简单，
所以原书与 `glmnet` 都用它。注意每次更新都可能是「置零」。

**易错点**：坐标下降对 $\ell_1$ 有效是因为**可分**；对一般耦合罚项（如 elastic net）
需要处理耦合项。另一个坑是变量尺度不均时 $\lambda$ 的效果随尺度变化，
所以必须先标准化特征。

### 凸性判别与强对偶

**定义**。$f$ 凸 $\iff f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y)$，$\theta\in[0,1]$。
可微时等价于 $\nabla^2 f \succeq 0$（海森半正定）。

**Slater 条件**：存在严格可行的点（约束取等号时自动满足）。Slater + 凸性 $\Rightarrow$
强对偶：$\min g$ s.t. $h=0$ 的原问题最优值等于 $\max_\lambda \min_\theta\mathcal{L}(\theta,\lambda)$ 的对偶最优值。

**结论**：SVM 训练是凸问题（$f$ 二次、约束线性），满足 Slater，所以对偶问题的解能
还原原问题的解；这就是第 9 章能从对偶直接读出决策函数的原因。

**易错点**：强对偶是「原问题最优值 = 对偶最优值」，**不是**「原问题解 = 对偶问题解」。
还需要互补松弛（$\mu_j h_j = 0$）才能把解对应起来——这就是为什么 SVM 里只有支持向量
（$\alpha_i>0$）才对决策函数有贡献。

### 随机梯度与 SGD

目标 $F(\boldsymbol\theta) = \frac1n\sum_i f_i(\boldsymbol\theta)$。每步用单样本的梯度：

$$
\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta_t \nabla f_{i_t}(\boldsymbol\theta)
$$

**推导（为什么无偏）**：$\E[\nabla f_{i_t}(\boldsymbol\theta)] = \frac1n\sum_i\nabla f_i(\boldsymbol\theta) = \nabla F(\boldsymbol\theta)$，
因为 $i_t$ 均匀抽样。所以这是对真实梯度的无偏估计，$\eta_t$ 取常数时过程仍围绕最优点抖动
（收敛到半径 $O(\eta)$ 的邻域）。取递减步长 $\eta_t \propto t^{-2/3}$（强凸时 $\propto t^{-1}$）
可以收敛到最优点。

**结论**：第 10 章的随机梯度下降就是这个式子；小批量梯度下降是同一想法的近似
（用 $m$ 个样本的均值代替单个，偏差 $O(1/m)$，方差更小）。

**易错点**：学习率过大发散、过小收敛慢；SGD 的解不像精确梯度下降那样落在精确驻点上。
调学习率比调网络结构更重要（实践上如此）。
