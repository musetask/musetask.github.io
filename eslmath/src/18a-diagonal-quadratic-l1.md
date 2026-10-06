## 18.1 引言 {#s-18-1}

本章处理一类在基因组学、蛋白质组学里变得常见的问题：特征数 $p$ 远大于样本数 $N$，写作 $p\gg N$。第 2 章已经算过，高维单位球里的点几乎全挤在半径 $\sqrt p$ 附近，最近邻距离的相对散布趋于零，「最近邻」因此失去判别力；第 14 章的中位半径判别器就是为此设计的补救。本章从「为什么 $N>p$ 时代的教科书方法会崩」开始，给出两类对策：一类用**结构假设**（特征近似独立 → 对角 LDA、§18.2）压掉参数个数，另一类用**惩罚**（$L_2$ 二次正则化 §18.3、$L_1$ 正则化 §18.4）压掉估计方差。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-1">原文 §18.1</a>

原文用一个小模拟来说明「少拟合反而更好」。生成 $N=100$ 个样本，$p$ 个两两相关系数 0.2 的标准高斯特征，响应按线性模型生成

$$
Y_i=\beta_0+\sum_{j=1}^p X_{ij}\beta_j+\varepsilon_i,\qquad \varepsilon_i\sim N(0,\sigma^2)\ \eqno{18.1}
$$

系数 $\beta_j$ 也取标准正态，于是「显著的」单变量系数个数随 $p$ 增长（$p=20,100,1000$ 时分别约为 9、33、331）。在这个数据上拟合岭回归（见第 3 章 §3.4.1），$p=20$ 时最优 $\lambda=0.001$（平均自由度 20），$p=100$ 时 $\lambda=100$（35），$p=1000$ 时 $\lambda=1000$（43）。用 $t_j=\hat\beta_j/\widehat{\mathrm{se}}[\hat\beta_j]$ 衡量信号强度，三个情形下 $|t_j|$ 的中位数分别是 2.0、0.6、0.2，超过 2 的个数是 9.8、1.2、0.0。

**结果** 这个「自由度」对比是全章的钥匙：$p$ 越大，能被数据分辨出来的系数越少，于是必须收缩得越狠。岭回归在 $p<N$ 时能利用特征间的相关结构（$\lambda$ 小即好），$p\gg N$ 时样本里根本没有足够信息去估那么大的协方差矩阵，必须重度正则化。本章后面所有的 $\lambda$、$\Delta$、$\gamma$，本质上都在回答同一个问题：**这个方向上到底有多少自由度可用**。

## 18.2 对角线性判别分析与最近收缩质心 {#s-18-2}

例子是 SRBCT 基因表达数据：$p=2308$ 个基因（列）、$N=63$ 个样本（行），每个表达值是 log 比值 $\log(R/G)$；样本分四类（BL、EWS、NB、RMS），另有 20 个测试样本。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-2">原文 §18.2</a>

$p\gg N$ 时不能做完整 LDA（类内协方差矩阵秩最多 $N-1$，必奇异）。最简单的正则化是假设类内协方差矩阵**是对角的**，即类内特征近似独立。这个假设在现实中很少严格成立，但 $p\gg N$ 时根本没有数据去估那些依赖关系，独立性把参数个数从 $O(Kp^2)$ 降到 $O(Kp)$，往往换来有效且可解释的分类器。

### 18.2.1 对角 LDA 的判别分数与分类规则 {#s-18-2-1}

> **基础知识** · 二次型与展开（见预备知识 L1、C2）
>
> $x^\top x=\sum_jx_j^2$；$(x-a)^\top D(x-a)=\sum_jd_j(x_j-a_j)^2$。展开技巧：把含 $k$ 的项与不含 $k$ 的项分开，含 $k$ 的一次项要靠 $2x^\top xa$ 收集，二次项 $a^\top a$ 对固定的 $x$ 是**逐类常数**。

在类内协方差为 $\mathrm{diag}(s_1^2,\dots,s_p^2)$、各类共享同一协方差的前提下，类 $k$ 的判别分数为

$$
\delta_k(x)=-\sum_{j=1}^p\frac{\bigl(x_j^{\star}-\bar x_{kj}\bigr)^2}{s_j^2}+\log\pi_k \eqno{18.2}
$$

其中 $s_j$ 是第 $j$ 个基因的合并类内标准差，$\bar x_{kj}=\sum_{i\in C_k}x_{ij}/N_k$ 是类 $C_k$ 中该基因的均值，$\bar x_k=(\bar x_{k1},\dots,\bar x_{kp})^\top$ 叫类 $C_k$ 的**质心**，$\pi_k$ 是类先验（$\sum_{k=1}^K\pi_k=1$）。第一项是 $x^{\star}$ 到第 $k$ 个质心的（负）标准化平方距离，第二项是先验修正。分类规则就是取分数最大的类

$$
\hat g(x^{\star})=\ell\quad \text{使得}\quad \delta_\ell(x^{\star})=\max_{k=1}^K\delta_k(x^{\star}) \eqno{18.3}
$$

**推导：为什么判别分数是线性的。** 把平方逐项展开成三部分：

$$
\sum_{j=1}^p\frac{\bigl(x_j^{\star}-\bar x_{kj}\bigr)^2}{s_j^2}=\sum_{j=1}^p\frac{(x_j^{\star})^2}{s_j^2}-2\sum_{j=1}^p\frac{x_j^{\star}\bar x_{kj}}{s_j^2}+\sum_{j=1}^p\frac{\bar x_{kj}^2}{s_j^2}
$$

第一项 $\sum_j(x_j^{\star})^2/s_j^2$ 与 $k$ 无关（同一个 $x^{\star}$），可以丢掉。第三项 $\sum_j\bar x_{kj}^2/s_j^2=\lVert\tilde x_k\rVert^2$ 是**逐类常数**（$\tilde x_k=(\bar x_{k1}/s_1,\dots,\bar x_{kp}/s_p)^\top$），不能丢，但它是常数。第二项是关于 $x^{\star}$ 的线性函数，于是

$$
\delta_k(x^{\star})=2\sum_{j=1}^p\frac{x_j^{\star}\bar x_{kj}}{s_j^2}-\sum_{j=1}^p\frac{\bar x_{kj}^2}{s_j^2}+\log\pi_k=2x^{\star\top}\tilde x_k-\lVert\tilde x_k\rVert^2+\log\pi_k
$$

**结果** (18.2) 展开后就是一个**线性判别式**，系数向量是标准化质心 $\tilde x_k$，阈值是 $\lVert\tilde x_k\rVert^2/2$ 加上先验项。类 $k$ 与类 $\ell$ 的判别边界由 $\delta_k(x^{\star})=\delta_\ell(x^{\star})$ 给出：

$$
x^{\star\top}(\tilde x_k-\tilde x_\ell)=\frac{\lVert\tilde x_k\rVert^2-\lVert\tilde x_\ell\rVert^2}{2}+\frac12\log\frac{\pi_\ell}{\pi_k}
$$

即一个垂直于标准化质心之差的超平面；先验不等时它不再平分两个质心，而是朝先验小的一侧平移。

### 18.2.2 与最近类中心（欧氏距离）的等价性 {#s-18-2-2}

**推导。** 令 $z_j^{\star}=x_j^{\star}/s_j$、$\tilde x_{kj}=\bar x_{kj}/s_j$，即在标准化坐标中记 $z^{\star}=(z_1^{\star},\dots,z_p^{\star})^\top$。则

$$
\sum_{j=1}^p\frac{\bigl(x_j^{\star}-\bar x_{kj}\bigr)^2}{s_j^2}=\sum_{j=1}^p\bigl(z_j^{\star}-\tilde x_{kj}\bigr)^2=\lVert z^{\star}-\tilde x_k\rVert^2
$$

所以「标准化平方距离最小 $\Leftrightarrow$ 标准化坐标中欧氏距离最近 $\Leftrightarrow$ 欧氏最近类中心」。若再假设 $s_1=\dots=s_p=s$，则 $\tilde x_k=\bar x_k/s$，距离 $\lVert z^{\star}-\tilde x_k\rVert^2=\frac1{s^2}\lVert x^{\star}-\bar x_k\rVert^2$，同一个正数缩放不改变 argmin，于是**完全退化为原始坐标中的欧氏最近类中心（NCC）**。

> **坑** 「等价于最近类中心」只在先验相等（或先验项被吸收）时严格成立。若 $\pi_k$ 不等，(18.2) 与 NCC 的差别是判别平面被 $\frac12\log(\pi_\ell/\pi_k)$ 平移；这时它就是第 6 章的**朴素贝叶斯**（各类独立同方差的多元正态，naive Bayes），而不是纯几何最近邻。

这个分类器在高维下常常很有效（Bickel and Levina, 2004 称之为「独立性规则」，并证明它常在高维问题中胜过标准 LDA）。SRBCT 数据上它对 20 个测试样本错了 5 个。缺点是**用了全部 $p$ 个特征**，无法解释。下面的收缩正是为了自动丢掉特征。

### 18.2.3 标准化对比量：把质心偏差换算成 z 分数 {#s-18-2-3}

要做收缩，先要把质心偏差放到「噪声尺度」上度量。第 $j$ 个基因、类 $k$ 与全体均值的对比量标准化为

$$
d_{kj}=\frac{\bar x_{kj}-\bar x_j}{m_k\,(s_j+s_0)},\qquad m_k^2=\frac{1}{N_k}-\frac1N \eqno{18.4}
$$

$\bar x_j=\sum_{i=1}^Nx_{ij}/N$ 是总体均值，$s_0>0$ 是一个小常数（通常取 $s_j$ 的中位数）。

**推导：分子里的方差为什么是 $m_k^2\sigma^2$。** 在类内方差为常数 $\sigma^2=s_j^2$ 的正态假设下，$\bar x_{kj}$ 与 $\bar x_j$ 的协方差不能忽略：

$$
\mathrm{Var}(\bar x_{kj}-\bar x_j)=\mathrm{Var}(\bar x_{kj})+\mathrm{Var}(\bar x_j)-2\mathrm{Cov}(\bar x_{kj},\bar x_j)=\frac{\sigma^2}{N_k}+\frac{\sigma^2}{N}-2\frac{\sigma^2}{N}=\sigma^2\Big(\frac1{N_k}-\frac1N\Big)=m_k^2\sigma^2
$$

其中 $\mathrm{Cov}(\bar x_{kj},\bar x_j)=\mathrm{Cov}\big(\frac1{N_k}\sum_{i\in C_k}x_{ij},\frac1N\sum_{i=1}^Nx_{ij}\big)=\frac{|C_k|}{N_kN}\sigma^2=\frac{\sigma^2}{N}$。所以对比量的标准差是 $m_k\sigma_j$，除以它就得到近似 $z$ 分数——这正是 $m_k$ 这个因子的来源。$s_0$ 的作用是防止 $\bar x_{kj}-\bar x_j$ 因表达式数值接近零（log 比值可能接近 0）而产生虚假的大 $d_{kj}$。

### 18.2.4 软阈值收缩：NSC 的核心 {#s-18-2-4}

把 $d_{kj}$ 朝零做**软阈值**（图 18.2 的橙色曲线）：

$$
d_{kj}^{\prime}=\operatorname{sign}(d_{kj})\bigl(\lvert d_{kj}\rvert-\Delta\bigr)_+ \eqno{18.5}
$$

$(u)_+=\max(u,0)$。也可换成**硬阈值**（自动变量选择的更朴素版本）：

$$
d_{kj}^{\prime}=d_{kj}\cdot\mathbb{I}\bigl(\lvert d_{kj}\rvert\ge\Delta\bigr) \eqno{18.6}
$$

$\Delta$ 是待定参数（例子里用 10 折交叉验证选，取到 $\Delta=4.34$，选出 43 个基因）。每个 $d_{kj}$ 的绝对值减少 $\Delta$（过零则归零）。

**核心推导：软阈值是「平方误差 + $L_1$ 惩罚」的极小点。** 这一步是 NSC 的全部理论内容（原文指向练习 18.2：把 (18.7) 看成类均值的 lasso 型估计）。

*第 1 步：写出估计问题。* 对固定的 $(k,j)$，要估的量是类均值与总体均值之差 $\theta_{kj}=\mu_{kj}-\mu_j$。取目标函数

$$
\Psi_{kj}(\theta)=\sum_{i\in C_k}\bigl(x_{ij}-\mu_j-\theta\bigr)^2+2\lambda_{kj}\lvert\theta\rvert
$$

*第 2 步：解一维问题。* $\Psi$ 是凸的（凸二次 + 凸惩罚），逐段求导。设 $u=\bar x_{kj}-\bar x_j$，注意 $\sum_{i\in C_k}(x_{ij}-\mu_j-\theta)^2$ 关于 $\theta$ 的导数是 $-2\sum_{i\in C_k}(x_{ij}-\mu_j-\theta)=-2(N_ku-N_k\theta)$。于是

$$
\frac{\partial\Psi}{\partial\theta}=\begin{cases} -2N_k(u-\theta)+2\lambda_{kj}=0, & \theta>0 \\[2pt] -2N_k(u-\theta)-2\lambda_{kj}=0, & \theta<0 \\[2pt] \text{满足 } \lvert -2N_ku\rvert\le 2\lambda_{kj}, & \theta=0\end{cases}
$$

第一行给出 $\theta=u-\lambda_{kj}/N_k$，第二行给出 $\theta=u+\lambda_{kj}/N_k$，第三行要求 $\lvert u\rvert\le\lambda_{kj}/N_k$。三行合起来就是

$$
\hat\theta_{kj}=\operatorname{sign}(u)\bigl(\lvert u\rvert-\lambda_{kj}/N_k\bigr)_+
$$

即**把观测值 $u$ 朝零减一个固定的量**——正是软阈值。

*第 3 步：换算回 $d$ 的单位。* 由第 2 步的结果 $\hat\theta_{kj}=u-\lambda_{kj}\operatorname{sign}(u)\mathbb{I}(\lvert u\rvert>\lambda_{kj})$，两边同除以 $m_k(s_j+s_0)\approx m_ks_j$，得

$$
\hat d_{kj}=d_{kj}-\frac{\lambda_{kj}}{N_km_ks_j}\operatorname{sign}(d_{kj})\mathbb{I}\bigl(\lvert d_{kj}\rvert>\tfrac{\lambda_{kj}}{N_km_ks_j}\bigr)=\operatorname{sign}(d_{kj})\bigl(\lvert d_{kj}\rvert-\Delta_{kj}\bigr)_+,
\quad \Delta_{kj}:=\frac{\lambda_{kj}}{N_km_ks_j}
$$

**结果** 软阈值 (18.5) 就是「平方风险 + $L_1$ 惩罚」的最优解，阈值 $\Delta$ 与惩罚参数 $\lambda$ 只差一个正的比例因子 $1/(N_km_ks_j)$。所以 NSC 是一个**逐坐标独立的 lasso**：$L_1$ 惩罚在这里不是用来「选变量」的装饰，而是从最小化风险直接推出来的收缩形式。

**推导：为什么用软阈值而不是硬阈值。** 同样的思路换成 0-1 惩罚：$\Psi^{\prime}(\theta)=\sum_{i\in C_k}(x_{ij}-\mu_j-\theta)^2+\lambda\mathbb{I}(\theta\ne0)$。在 $\theta\ne0$ 区间内极小点是 $\theta=u$，代价为 $\lambda$；在 $\theta=0$ 处代价为 $u^2$。所以解为

$$
\hat\theta_{kj}=\begin{cases} u, & \lvert u\rvert^2>\lambda \\ 0, & \lvert u\rvert^2\le\lambda\end{cases}\ \Longleftrightarrow\ d_{kj}^{\prime}=d_{kj}\mathbb{I}\bigl(\lvert d_{kj}\rvert\ge\Delta\bigr),\quad \Delta=\sqrt{\lambda}/(m_ks_j)
$$

比较两式：硬阈值是「不满足就整段扔掉」，软阈值是「所有坐标都减一点」。软阈值连续、在 0 处斜率跳变但函数连续（图 18.2），所以原文偏好软阈值：它同时降方差（每个坐标都朝 0 拉一点）和做变量选择（$|d|$ 小的归零），而且不产生阈值附近的硬边界。

> **坑** 软阈值的偏差是**有偏的**：$\hat d_{kj}$ 在 $d_{kj}$ 很小时不是「无偏地保留」而是「过度收缩到接近 0」。这与 lasso 一样，用偏差换方差。若某个弱但确定的信号恰好被切到 0，NSC 没有任何补救机制——这是它与 §18.4 里 elastic net 的重要差别（后者用 $L_2$ 项把相关特征拉住，避免整组归零）。

### 18.2.5 反变换回质心与类概率 {#s-18-2-5}

把标准化变换反过来，得到收缩后的质心：

$$
\bar x_{kj}^{\prime}=\bar x_j+m_k(s_j+s_0)\,d_{kj}^{\prime} \eqno{18.7}
$$

代入 $d_{kj}^{\prime}$（用软阈值）就是 $\bar x_{kj}^{\prime}=\bar x_j+m_ks_j\,\operatorname{sign}(d_{kj})(\lvert d_{kj}\rvert-\Delta)_+$（忽略 $s_0$ 的一次修正），即**质心朝总体均值收缩**；用硬阈值时是「要么原样、要么完全等于总体均值 $\bar x_j$」。在 (18.2) 里用 $\bar x_{kj}^{\prime}$ 替换 $\bar x_{kj}$ 即得 NSC 分类器。

判别分数还能给出类概率估计：

$$
\hat p_k(x^{\star})=\frac{\exp\{\delta_k(x^{\star})\}}{\sum_{\ell=1}^K\exp\{\delta_\ell(x^{\star})\}} \eqno{18.8}
$$

**推导：为什么 (18.8) 就是后验概率。** 高斯类密度 $f_k(x)=|2\pi\Sigma_k|^{-1/2}\exp\{-\tfrac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k)\}$，贝叶斯公式给 $\Pr(G=k\mid x)\propto\pi_kf_k(x)$。对角 LDA 的 $\delta_k$ 就是 $\log\pi_kf_k(x)$（差一个与 $k$ 无关的常数），所以

$$
\Pr(G=k\mid x)=\frac{\pi_kf_k(x)}{\sum_\ell\pi_\ell f_\ell(x)}=\frac{e^{\delta_k(x)}}{\sum_\ell e^{\delta_\ell(x)}}
$$

即 (18.8)。**结果** 它是 softmax 形式，可以用来给分类结果排序，或在最大概率低于阈值时**拒判**（不给出分类），在高维小样本里这是很实用的操作。

**结果** 只有对**至少一个类**满足 $d_{kj}^{\prime}\ne0$ 的基因参与分类，绝大多数基因被丢掉。SRBCT 数据里 2308 个基因只剩 43 个（图 18.3 的热图），$\Delta$ 在一段很宽的范围内都能给出 0 个测试错误（图 18.4 上图；不收缩时错 5/20）。注意 Fan and Fan (2008) 从理论上证明了：在 $p\gg N$ 的对角 LDA 里**必须**做某种形式的变量选择，否则误差会随 $p$ 发散。

## 18.3 带二次正则化的线性分类器 {#s-18-3}

这一节换成另一组数据：144 个病人、14 种癌症类型、16063 个基因，另有 54 个测试病人。每位病人的数据先标准化到均值 0、方差 1。下表是八种方法的比较（$\lambda$ 由交叉验证选）：

| 方法 | CV 错分（SE） | 测试错分 | 用的基因数 |
|---|---|---|---|
| 最近收缩质心 | 35 (5.0) | 17 | 6 520 |
| $L_2$ 惩罚判别分析 | 25 (4.1) | 12 | 16 063 |
| 支持向量分类器 | 26 (4.2) | 14 | 16 063 |
| lasso 回归（one vs all） | 30.7 (1.8) | 12.5 | 1 429 |
| $k$ 近邻 | 41 (4.6) | 26 | 16 063 |
| $L_2$ 惩罚多项逻辑回归 | 26 (4.2) | 15 | 16 063 |
| $L_1$ 惩罚多项逻辑回归 | 17 (2.8) | 13 | 269 |
| elastic net 惩罚多项回归 | 22 (3.7) | 11.8 | 384 |

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-3">原文 §18.3</a>

表里最刺眼的一点：方法 2、3、6（三种二次正则化方法）**用了全部 16063 个基因**，而 $N=144$。这是 $L_2$ 惩罚的本质——它只缩小系数，不把任何系数压到 0。

### 18.3.1 正则化判别分析 (RDA) {#s-18-3-1}

线性判别分析需要求逆一个 $p\times p$ 的类内协方差矩阵；$p\gg N$ 时它秩最多 $N<p$，必然奇异。RDA 的做法是把这个估计**朝它自己的对角**收缩：

$$
\hat\Sigma(\gamma)=\gamma\hat\Sigma+(1-\gamma)\mathrm{diag}(\hat\Sigma),\qquad \gamma\in[0,1] \eqno{18.9}
$$

**核心推导：$\gamma$ 的插值性质与偏差–方差的连续调节。**

*第 1 步：两个端点。* $\gamma=0$ 时 $\hat\Sigma(0)=\mathrm{diag}(\hat\Sigma)$，这正是 §18.2 的对角 LDA，也就是「不收缩版本的最近收缩质心」；$\gamma=1$ 时 $\hat\Sigma(1)=\hat\Sigma$，$p\le N-1$ 时就是标准 LDA。

*第 2 步：正定性。* $\hat\Sigma$ 对称半正定（是 $X_c^\top X_c$ 的形式，$X_c$ 为中心化数据矩阵），$\mathrm{diag}(\hat\Sigma)$ 在所有 $s_j>0$ 时正定。所以

$$
\hat\Sigma(\gamma)=\mathrm{diag}(\hat\Sigma)+\gamma\bigl(\hat\Sigma-\mathrm{diag}(\hat\Sigma)\bigr)\succ0\quad(\gamma>0)
$$

凸组合的半正定性（预备知识 L1）保证任意 $\gamma>0$ 都可逆：$\hat\Sigma$ 的零特征值被抬到 $(1-\gamma)$ 倍的对角元上，最小特征值至少 $\gamma\min_j s_j^2>0$。这就是 RDA 解决奇异性的全部机制。

*第 3 步：把「$L_2$ 惩罚」与 $\gamma$ 挂钩。* 若把收缩目标换成标量矩阵 $\bar\sigma^2I$（$\bar\sigma^2$ 是合并方差的标量版本），则

$$
\bigl[\gamma\hat\Sigma+(1-\gamma)\bar\sigma^2I\bigr]^{-1}=\frac1{\bar\sigma^2}\Bigl[\tfrac{1}{\bar\sigma^2}\hat\Sigma+\tfrac{1-\gamma}{\gamma}I\Bigr]^{-1}=\frac1{\bar\sigma^2}\Bigl[\tfrac{1}{\bar\sigma^2}\hat\Sigma+\lambda I\Bigr]^{-1}
$$

即 $\lambda=\bar\sigma^2(1-\gamma)/\gamma$，也就是 $\gamma=\bar\sigma^2/(\bar\sigma^2+\lambda)$。**结果** $\gamma$ 与岭惩罚参数 $\lambda$ 是一一对应的：$\gamma\to0\Leftrightarrow\lambda\to\infty$（退化成欧氏最近类中心），$\gamma\to1\Leftrightarrow\lambda\to0$（回到 LDA）。(18.9) 用对角作目标是比标量更精细的选择（每个特征按自己的尺度收缩），原文说它「很像」岭回归把总协方差往对角（标量）矩阵收缩，这句话在标量目标下是**严格等价**的。

*第 4 步：偏差–方差公式。* 把 LDA 看成对类别指示变量做最小二乘回归（见第 12 章 (12.57) 的最优评分表示），$\hat\beta(\lambda)=(X^\top X+\lambda I_p)^{-1}X^\top y$。设 $Z=X^\top X$、噪声 $e\sim N(0,\sigma^2I_N)$，则 $\hat\beta=(Z+\lambda I)^{-1}(Z\beta+X^\top e)$：

$$
E[\hat\beta]-\beta=\bigl[(Z+\lambda I)^{-1}Z-I\bigr]\beta=-\lambda(Z+\lambda I)^{-1}\beta
$$

$$
\text{偏差}^2=\lambda^2\beta^\top(Z+\lambda I)^{-2}\beta,\qquad \mathrm{tr}\bigl(\mathrm{Var}(\hat\beta)\bigr)=\sigma^2\sum_{j=1}^p\frac{z_j}{(z_j+\lambda)^2}
$$

方差那一行的推导用了两个恒等式：$(Z+\lambda I)^{-1}Z=I-\lambda(Z+\lambda I)^{-1}$，以及 $\mathrm{tr}(Z+\lambda I)^{-1}=\sum_j(z_j+\lambda)^{-1}$（$z_j$ 是 $Z$ 的特征值）。把 $Z$ 谱分解后，风险为

$$
R(\lambda)=\sum_{j=1}^p\frac{\lambda^2\beta_j^2}{(z_j+\lambda)^2}+\sigma^2\sum_{j=1}^p\frac{z_j}{(z_j+\lambda)^2}
$$

逐项求导（设 $\beta_j$ 是 $\beta$ 在 $Z$ 的特征基下的坐标）：

$$
\frac{\mathrm dR}{\mathrm d\lambda}=2\sum_{j=1}^p\frac{z_j\bigl(\lambda\beta_j^2-\sigma^2\bigr)}{(z_j+\lambda)^3}
$$

**结果**（a）$z_j=0$ 的方向（$p-N$ 个不可辨识方向）对方差的贡献是 $0$，但偏差贡献是 $\lambda^2\beta_j^2/\lambda^2=\beta_j^2$，即被**完全收缩到 0**——这是免费的。（b）$z_j$ 大的方向（噪声大的方向）方差贡献 $\sigma^2z_j/(z_j+\lambda)^2$ 迅速衰减。（c）若所有 $z_j=z$，求导式退化为 $\frac{2z}{(z+\lambda)^3}\bigl(\lambda\lVert\beta\rVert^2-p\sigma^2\bigr)=0$，最优解显式为 $\hat\lambda=p\sigma^2/\lVert\beta\rVert^2$；$p=1$ 时就是熟知的 $\hat\lambda=\sigma^2/\hat\beta^2$。这三条合起来就是「$p\gg N$ 时必须重度收缩」的定量理由，也是 (18.9) 里 $\gamma$ 由交叉验证选的依据（表 18.1 第 2 行，所有 $\gamma\in(0.002,0.550)$ 都给出相同的 CV 与测试误差）。

> **坑** (18.9) 收缩的是**协方差矩阵**，类中心本身没有收缩；Guo et al. (2006) 讨论了把两者一起收缩的进一步版本。另外注意 RDA 与 NSC 的对应只在 $\gamma=0$ 处成立：NSC 是**逐坐标**做 $L_1$ 型收缩，RDA 是对**矩阵整体**做 $L_2$ 型收缩，NSC 会产生精确零而 RDA 不会。

### 18.3.2 带二次正则化的逻辑回归 {#s-18-3-2}

用对称形式的多类逻辑模型（第 4 章 (4.17)）：

$$
\Pr(G=k\mid X=x)=\frac{\exp\bigl(\beta_{k0}+x^\top\beta_k\bigr)}{\sum_{\ell=1}^K\exp\bigl(\beta_{\ell0}+x^\top\beta_\ell\bigr)},\qquad k=1,\dots,K \eqno{18.10}
$$

有 $K$ 个 log-odds 系数向量 $\beta_1,\dots,\beta_K$。用二次惩罚最大化惩罚化的对数似然：

$$
\bigl(\{\hat\beta_{k0},\hat\beta_k\}_{k=1}^K\bigr)=\arg\max_{\{\beta_{k0},\beta_k\}}\left\{\sum_{i=1}^N\sum_{k=1}^K\log\Pr(g_i\mid x_i)-\frac{\lambda}{2}\sum_{k=1}^K\lVert\beta_k\rVert_2^2\right\} \eqno{18.11}
$$

**推导：二次惩罚自动解决了参数化的冗余。** 对 $\beta_k$ 求梯度（记 $P_{ik}=\Pr(g_i=k\mid x_i)$、$y_{ik}=\mathbb{I}(g_i=k)$）：

$$
\frac{\partial}{\partial\beta_k}\sum_i\sum_k\log P_{ik}=X^\top\bigl(P_k-y_k\bigr),\qquad \frac{\partial}{\partial\beta_k}\Bigl(-\tfrac{\lambda}{2}\sum_k\lVert\beta_k\rVert^2\Bigr)=-\lambda\beta_k
$$

KKT（预备知识 O2）给出 $X^\top(P_k-y_k)=\lambda\hat\beta_k$。把 $k=1,\dots,K$ 的 $K$ 个等式相加：

$$
\lambda\sum_{k=1}^K\hat\beta_k=X^\top\Bigl(\sum_kP_k-\sum_ky_k\Bigr)=X^\top(\mathbf 1_N-\mathbf 1_N)=\mathbf 0
$$

**结果** $\sum_{k=1}^K\hat\beta_k=\mathbf 0$，即每个坐标上 $K$ 个系数之和被强制为零（原练习 18.3）。这自动消掉了「同时给所有 $\beta_k$ 加同一个向量」造成的不可辨识。注意常数项 $\beta_{k0}$ **不**惩罚（否则也会被压成 0，模型失去截距）。

**推导：$\lambda$ 小时它逼近 LDA 方向。** 在等先验、无截距的简化下，$P_{ik}$ 在 $\beta\approx0$ 附近一阶展开为 $P_{ik}\approx\frac1K+\frac1Kx_i^\top\beta_k$。代入 KKT：

$$
\lambda\hat\beta_k=\sum_i\Big(\frac1K+\frac1Kx_i^\top\hat\beta_k-y_{ik}\Bigr)x_i=\frac1KX^\top X\hat\beta_k-X^\top y_k
$$

（用了 $\sum_ix_i=\mathbf0$ 与中心化），即 $\bigl(\frac1KX^\top X-\lambda I\bigr)\hat\beta_k=X^\top y_k$，而 $X^\top y_k=\sum_{i\in C_k}x_i-N\bar x=N(\bar x_k-\bar x)$。**结果**

$$
\hat\beta_k=\Bigl(\tfrac1K X^\top X-\lambda I\Bigr)^{-1}N(\bar x_k-\bar x)
$$

小 $\lambda$ 时 $\hat\beta_k\propto\bar x_k-\bar x$，正是 LDA 的判别方向；惩罚沿着 $X^\top X$ 的奇异方向收缩。**坑** 这个「$L_2$ 惩罚逻辑回归 $=$ 沿奇异方向收缩的 LDA」只在 $\beta$ 很小（一阶近似有效）时成立；$\lambda\to0$ 且数据可分时系数发散，需要重标化（Rosset et al., 2004a：$\lambda\to0$ 时重标化的解收敛到最大间隔分类器，即 §12.2 的 SVM）。表 18.1 第 6 行就是这个模型（15 个测试错误，用满 16063 个基因）。问题凸，可以用 Newton 法或 Friedman et al. (2010) 的路径算法。

### 18.3.3 支持向量分类器 {#s-18-3-3}

二类 SVM 见第 12 章。$p>N$ 时它格外有吸引力：除非不同类有完全相同的特征向量，$N$ 个点在 $p$ 维空间里几乎总是**严格可分**的，所以不需任何正则化也能找到间隔最大的分离超平面。有点反直觉的是，当 $p\gg N$ 时**未正则化**的 SVM 常常和最好的正则化版本一样好——过拟合在这里似乎不是问题，部分原因是错分损失对样本的移动不敏感。

**推导：对偶为什么与 $p$ 无关。** 软间隔原始问题（$C$ 是「正则化」参数）与拉格朗日函数

$$
\min_{\beta,\xi}\ \frac12\lVert\beta\rVert^2+C\sum_{i=1}^N\xi_i\ \ \text{s.t.}\ \ g_i(\beta^\top x_i+\beta_0)\ge1-\xi_i,\ \xi_i\ge0
$$

$$
L=\tfrac12\lVert\beta\rVert^2+C\sum_i\xi_i-\sum_i\alpha_i\bigl[g_i(\beta^\top x_i+\beta_0)-1+\xi_i\bigr]-\sum_i\mu_i\xi_i
$$

对 $\beta,\beta_0,\xi$ 求导并令零：$\beta=\sum_i\alpha_ig_ix_i$，$\sum_i\alpha_ig_i=0$，$\alpha_i\bigl[g_i(\beta^\top x_i+\beta_0)-1+\xi_i\bigr]=0$，$0\le\alpha_i\le C$。代回得对偶

$$
\max_{\alpha}\ \sum_{i=1}^N\alpha_i-\frac12\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jg_ig_j\,\underbrace{x_i^\top x_j}_{(XX^\top)_{ij}},\quad 0\le\alpha_i\le C,\ \sum_i\alpha_ig_i=0
$$

**结果** 对偶里只有 $N$ 个变量 $\alpha_i$ 和 $N\times N$ 的 Gram 矩阵 $XX^\top$，**自由度的个数与 $p$ 无关**；解出来以后

$$
\hat g(x)=\hat\beta_0+\sum_{i\in\text{SV}}\hat\alpha_ig_i\,x_i^\top x
$$

的预测代价是 $O(\lvert\text{SV}\rvert\cdot p)$，训练代价是 $O(\lvert\text{SV}\rvert^2)$（Gram 矩阵本身 $O(pN^2)$ 算出一次）。这就是它比 $L_2$ 惩罚判别分析更实用的技术原因：后者必须显式处理 $p\times p$ 的 $\hat\Sigma$（除非用 §18.3.5 的捷径）。表 18.1 第 3 行测试错 14，且误差对 $C$ 的选择不敏感（$C>0.001$ 都可以）；由于 $p>N$ 可分，取 $C=\infty$ 即可让超平面完美分开训练数据。

$K>2$ 的推广：一对多（ova）与一对一（ovo）。ovo 要算全部 $\binom K2$ 个二类分类器，测试点归为「赢得最多对局」的类；ova 算 $K$ 个二类分类器，取置信度（到超平面的有符号距离）最高的类。Vapnik (1998)、Weston and Watkins (1999) 提出了更复杂的多类判别准则。Tibshirani and Hastie (2007) 的**间隔树**把 SVM 排成二叉树，像 CART 一样做层次化分类。

### 18.3.4 特征选择 {#s-18-3-4}

$p$ 大时特征选择不只是科学要求，还是必要的。RDA、逻辑回归、SVM **都不自动做**特征选择，因为它们用的都是二次惩罚：所有特征都有非零权重。原文提到的临时办法是**递归特征消减（RFE）**（Guyon et al., 2002）：从权重最小的特征开始逐个删掉并重新拟合。

**结果** 在这个例子里 RFE 失败了——Ramaswamy et al. (2001) 报告 SVM 的准确率随基因数从 16063 开始减少就下降，而训练样本只有 144 个。原文对这个反常行为没有解释。这是「高维下特征未必多余」的一个警示：$p$ 大并不意味着每个坐标都是噪声，$L_2$ 惩罚把所有坐标按比例缩小反而是安全的。

**结果** 三种方法都可以用核改成非线性判别边界（§18.3 末段）。动机本是提高复杂度，但 $p\gg N$ 时模型已经足够复杂，过拟合始终是危险；即便如此，径向核在这些高维问题上**有时**反而更好——因为径向核会压低相距很远的点之间的内积，从而对离群点更稳健，而高维里「离群点很多」是常态。不过表 18.1 里给 SVM 加径向核反而更差。

### 18.3.5 $p\gg N$ 时的计算捷径 {#s-18-3-5}

这一小节的技术适用于**任何**「参数线性 + 系数二次惩罚」的方法，包括本节三种以及带二次正则化的神经网络（§11.5.2）。几何直觉：三维空间里两点总在一条直线上，$p$ 维空间里 $N$ 个点落在一个 $(N-1)$ 维仿射子空间里——所以 $p$ 维的问题其实只有 $N$ 个自由度。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-3-5">原文 §18.3.5</a>

**第 1 步：SVD 分解。** 数据矩阵 $X$ 是 $N\times p$，作奇异值分解

$$
X=UDV^\top,\qquad U\ \text{是}\ N\times N\ \text{正交},\ V\ \text{是}\ p\times N\ \text{列正交},\ D=\mathrm{diag}(d_1,\dots,d_N),\ d_1\ge\dots\ge d_N\ge0 \eqno{18.12}
$$

> **基础知识** · SVD 与秩（见预备知识 L3）
>
> $X^\top X=VD^2V^\top$，$XX^\top=UD^2U^\top$：**两者非零特征值完全相同**（都是 $d_j^2$），特征向量分别是 $V$ 与 $U$ 的列。这条「同一个谱的两种看法」是本小节全部结论的来源。Eckart–Young：$\lVert X-X_r\rVert_F^2=\sum_{j>r}d_j^2$ 是所有秩 $\le r$ 逼近中的最小值。

**第 2 步：换一个小矩阵。** 记 $R=UD$（$N\times N$，行记为 $r_i^\top$），则

$$
X=RDV^\top=RV^\top,\qquad R=XV \eqno{18.13}
$$

（第二式验证：$XV=UDV^\top V=UD=R$，因为 $V^\top V=I_N$。）关键的两个恒等式随之而来：

$$
R^\top R=D^2U^\top UD=D^2,\qquad R^\top=UD^\top
$$

即 $R$ 的列**两两正交**，第 $j$ 列的范数是 $d_j$。

**第 3 步：岭回归解的形式变换。** 岭回归的显式解为

$$
\hat\beta=\bigl(X^\top X+\lambda I_p\bigr)^{-1}X^\top y \eqno{18.14}
$$

用第 1 步的 $X^\top X=VD^2V^\top$，注意到

$$
\bigl(VD^2V^\top+\lambda I_p\bigr)V=V\bigl(D^2V^\top V+\lambda I_N\bigr)=V\bigl(D^2+\lambda I_N\bigr)
$$

（第一步用 $V^\top V=I_N$），所以 $\bigl(X^\top X+\lambda I_p\bigr)^{-1}V=V(D^2+\lambda I_N)^{-1}$。再代 $X^\top y=VDU^\top y$：

$$
\hat\beta=V\bigl(D^2+\lambda I_N\bigr)^{-1}DU^\top y=V\bigl(R^\top R+\lambda I_N\bigr)^{-1}R^\top y \eqno{18.15}
$$

**结果** (18.14) 与 (18.15) 是**同一个向量**：$\hat\beta=V\hat\theta$，其中 $\hat\theta$ 是只用 $N$ 个「观测」$(r_i,y_i)$ 做的岭回归估计——每个「预测变量」$r_i$ 其实是一个 $N$ 维向量，共 $N$ 个预测变量。拟合值也完全一致：由 (18.13) 的第 $i$ 行读出 $x_i^\top=r_i^\top V^\top$，所以

$$
x_i^\top\beta=r_i^\top V^\top V\theta=r_i^\top\theta,\qquad \lVert\beta\rVert^2=\theta^\top V^\top V\theta=\lVert\theta\rVert^2
$$

后一个等式是关键：**二次惩罚在旋转 $V$ 下不变，线性模型的拟合值在旋转下等变**，所以损失函数值与惩罚值都逐项相等，不只是极小点相同。代价从 $O(p^3)$（对 $p\times p$ 的 $X^\top X$ 求逆）降到 $O(pN^2)$。

**第 4 步：一般定理。** 对任何「线性模型 + 二次惩罚」的问题，设 $f^\star(r_i)=\theta_0+r_i^\top\theta$，$r_i$ 如 (18.13)，考虑下面**一对**优化问题：

$$
(\hat\beta_0,\hat\beta)=\arg\min_{\beta_0,\beta\in\mathbb{R}^p}\ \sum_{i=1}^N L\bigl(y_i,\beta_0+x_i^\top\beta\bigr)+\frac{\lambda}{2}\lVert\beta\rVert_2^2 \eqno{18.16}
$$

$$
(\hat\theta_0,\hat\theta)=\arg\min_{\theta_0,\theta\in\mathbb{R}^N}\ \sum_{i=1}^N L\bigl(y_i,\theta_0+r_i^\top\theta\bigr)+\frac{\lambda}{2}\lVert\theta\rVert_2^2 \eqno{18.17}
$$

**则** $\hat\beta_0=\hat\theta_0$ 且 $\hat\beta=V\hat\theta$。

**推导。** 把 (18.16) 的目标限制在 $\{\beta=Vu:\ u\in\mathbb{R}^N\}$ 这个子空间上（当 $p>N$ 时 $\mathrm{col}(X)\subseteq\mathrm{row}(V)$，最优解必在这个子空间里），令 $\theta=u$。逐项用第 3 步的两个恒等式：

$$
\sum_iL(y_i,\beta_0+x_i^\top\beta)=\sum_iL(y_i,\beta_0+r_i^\top u)\ \ (\text{完全相等}),\qquad \lVert Vu\rVert_2^2=\lVert u\rVert_2^2
$$

所以子空间上的限制问题**逐点等于** (18.17)，即 (18.17) 在 (18.16) 上取到相同的极小值；又因为 (18.16) 凸，极小点唯一，故 $\hat\beta=V\hat\theta$、截距相同。$\blacksquare$

**结果** 这个定理说：把 $p$ 个向量 $x_i$ 换成 $N$ 个向量 $r_i$ 做同样的惩罚拟合，再乘一次矩阵变回去。几何上我们把特征旋转到一个坐标系，使前 $N$ 个坐标之后全为零；旋转是允许的，因为二次惩罚在旋转下不变、线性模型是等变的。它适用于本节所有方法、正则化逻辑回归、LDA（练习 18.6）、SVM、带二次正则化的神经网络。$\lambda$ 通常由交叉验证选，但只需在原数据上构造一次 $R$，之后每一折 CV 都直接用它做数据（练习 18.12）。SVM 的**核技巧**（§12.3.7）用的是同一个降维思想：$N\times N$ 的 Gram 矩阵 $K=XX^\top=UD^2U^\top$，与 $R$ 携带完全相同的信息（练习 18.13 用 $K$ 的 SVD 拟合带岭惩罚的逻辑回归）。

> **坑** 这条捷径**不适用于 lasso**：$\lVert Vu\rVert_1=\lVert u\rVert_1$ 不成立。$V$ 的旋转把 $L_1$ 球变成一个平行多面体，稀疏解不在 $\{Vu\}$ 里，所以 §18.4 的方法必须直接用坐标下降。

## 18.4 带 L1 正则化的线性分类器 {#s-18-4}

§18.3 的方法都用 $L_2$ 惩罚，所有估计系数都非零，因此**不做**特征选择（表 18.1 第 2、3、6 行用满 16063 个基因）。本节换成 $L_1$ 惩罚，得到自动的特征选择。

**lasso 与 $p>N$ 下的稀疏性**

回顾第 3 章 §3.4.2 的 lasso，写成 Lagrange 形式（3.52）：

$$
\min_{\beta_0,\beta_1,\dots,\beta_p}\ \sum_{i=1}^N\Bigl(y_i-\beta_0-\sum_{j=1}^p x_{ij}\beta_j\Bigr)^2+\lambda\sum_{j=1}^p\lvert\beta_j\rvert \eqno{18.18}
$$

$L_1$ 惩罚的后果是：当 $\lambda$ 足够大时，一部分 $\hat\beta_j$ **精确等于零**。§3.8.1 的 LARS 算法能高效算出所有 $\lambda$ 的 lasso 解。关键的一条（Rosset and Zhu, 2007 等，由凸对偶得到）：

> **结果** 当 $p>N$ 时，对**任意** $\lambda$，lasso 解的非零系数个数至多为 $N$；$\lambda\to0$ 时 lasso 精确拟合训练数据。
>
> 推导思路：KKT 给出 $\mathbf 0\in\partial\ell(\hat\beta)+\lambda\partial\lVert\beta\rVert_1$，其中 $\ell$ 是光滑的，所以存在向量 $z$ 使 $\nabla\ell(\hat\beta)+\lambda s=z$，$s\in\partial\lVert\hat\beta\rVert_1$。这说明 $\hat\beta\in\mathrm{row}(X)$，而 $\dim\mathrm{row}(X)=N$；再由 $L_1$ 球面（一个多面体）与 $N$ 维子空间的交至多 $N$ 个顶点，得非零坐标数 $\le N$。这就是 lasso 提供「（严重）形式的特征选择」的机制。

把响应编码成 $\pm1$ 并对预测取阈值（通常 0），lasso 回归就变成二类分类器；多类可用 §18.3.3 的 ova / ovo（表 18.1 第 4 行，ova，测试错 12.5，选 1429 个基因）。

**分类 lasso：用 $L_1$ 惩罚逻辑回归**

更自然的做法是用 lasso 惩罚去正则化逻辑回归。文献里有多个实现，包括类似 LARS 的路径算法（Park and Hastie, 2007）；因为路径分段光滑但**非线性**，精确方法比 LARS 慢、$p$ 大时不太可行。Friedman et al. (2010) 给出很快的算法：用 §18.3.2 的对称多项逻辑模型 (18.10)，写出惩罚化的对数似然准则

$$
\min_{\{\beta_{jk}\}}\ \sum_{i=1}^N\sum_{k=1}^K\sum_{j=1}^p\max\Bigl\{\log\Pr(g_i=k\mid x_i)-\lambda\lvert\beta_{kj}\rvert\Bigr\} \eqno{18.19}
$$

（对比 (18.11) 的 $L_2$ 版本：那里是 $-\frac\lambda2\sum_k\lVert\beta_k\rVert_2^2$，逐坐标取平方；这里逐坐标取绝对值。）原文把惩罚写在 $\max$ 括号里，这是文献常用的 min–max 写法：恒等式 $\max\{a,b\}=\tfrac12\bigl(a+b+\lvert a-b\rvert\bigr)$ 表明它等于「似然项与惩罚项取较大者」在每个坐标上的和。下面按与它对齐的标准形式「$\sum_i\sum_k\log\Pr(g_i=k\mid x_i)-\lambda\sum_k\sum_j\lvert\beta_{kj}\rvert$」做推导，两者的阈值规则完全一样（见第 3 步）。

Friedman et al. 的算法在一串预先选定的 $\lambda$ 上用**循环坐标下降**（§3.8.6）算精确解，利用两个事实：$p\gg N$ 时解稀疏；相邻 $\lambda$ 的解很相似。表 18.1 第 7 行是它（自动特征选择选了 269 个基因），测试错 13。Genkin et al. (2007) 虽然用贝叶斯观点表述，实际算的也是这个惩罚化极大似然问题的后验众数。

**核心推导：$L_1$ 的阈值规则怎么来的。** 坐标下降每次只更新一个坐标。固定其余坐标，记 $z_k=\sum_{j'\ne j}\beta_{kj'}x_{ij'}$，要解的一维问题是

$$
\min_{b}\ \sum_{i=1}^N\sum_{k=1}^K\Bigl[\log\Pr(g_i=k\mid x_i)\ \text{（其中 }\beta_{kj}\ \text{换为 } b)\Bigr]+\lambda\lvert b\rvert
$$

*第 1 步：求导并分离符号。* 对 $b>0$，一阶条件是 $\partial\ell/\partial b=-\lambda$；对 $b<0$ 是 $\partial\ell/\partial b=+\lambda$；在 $b=0$ 处要求 $\lvert\partial\ell/\partial b\rvert\le\lambda$（次梯度条件，预备知识 O2）。

*第 2 步：定义偏相关得分。* 记 $r_j=\bigl(-\partial\ell/\partial\beta_j\bigr)\big|_{\beta_j=0}$，即「置零其余坐标时，把 $\beta_j$ 推向正方向会损失多少」。逻辑模型里可以显式算出

$$
r_j=\sum_{i=1}^N\sum_{k=1}^K\Bigl[P_{ik}-y_{ik}\Bigr]\,x_{ij}\Big|_{b=0}
$$

*第 3 步：$L_2$ vs $L_1$ 的对比。* 若惩罚是 $\frac\lambda2b^2$，一阶条件 $-\ell'(b)=\lambda b$ 给出 $\hat b=r_j/\lambda$——**永远非零**（除非 $r_j=0$）。若惩罚是 $\lambda\lvert b\rvert$，一阶条件 $-\ell'(b)=\lambda\operatorname{sign}(b)$ 给出 $\hat b=\operatorname{sign}(r_j)(\lvert r_j\rvert-\lambda)_+$——**$\lvert r_j\rvert\le\lambda$ 时精确为 0**。

**结果** $L_1$ 惩罚的阈值规则来自「一阶条件 + 次梯度条件」这一步：非零条件是 $\lvert r_j\rvert>\lambda$，零值条件是 $\lvert r_j\rvert\le\lambda$，两者的分界就是软阈值。这与 §18.2 里 NSC 的软阈值是同一个数学结构（那里是从「平方误差 $+\lambda\lvert\beta\rvert$」的一维极小点推出来的），区别只在 $r_j$ 是「系数梯度」而不是「标准化后的样本均值」。

*第 4 步：路径。* (18.19) 里的 $\lambda$ 扫描给出一条系数路径。图 18.5 的左图是白血病数据（Golub et al., 1999）上二类 lasso 逻辑回归的路径：7129 个基因、38 个样本（27 个 ALL、11 个 AML），另有 34 个测试样本（20、14）。数据线性可分，所以 $\lambda=0$ 处解不唯一（练习 18.11），且 $\lambda$ 很小时解退化，所以路径在拟合概率趋近 0 和 1 处被截断；左图有 19 个非零系数。$\lambda\to0$ 是合适的极限，对应测试集 3/34 个错分（图 18.6；右图用二项偏差度量，更平滑）。小样本使这些曲线有相当大的抽样波动。$p\gg N$ 时所有正则化逻辑回归的极限系数都发散，所以实际软件会显式或隐式设 $\lambda>0$ 的下界；不过**重标化**后的系数收敛，可以看作线性最优分离超平面的有趣替代品。

**Elastic net：$L_1$ 与 $L_2$ 的折中**

基因组应用里变量之间常强相关（基因倾向于在分子通路上协同作用）。lasso 惩罚对一组强相关变量中「选哪个」相当无所谓（练习 3.28），而岭惩罚倾向于把相关变量的系数**拉向彼此**（练习 3.29）。**elastic net** 惩罚（Zou and Hastie, 2005）是折中：

$$
\lambda\sum_{j=1}^p\Bigl\{\alpha\lvert\beta_j\rvert+\tfrac{(1-\alpha)}{2}\beta_j^2\Bigr\} \eqno{18.20}
$$

第二项鼓励高度相关的特征被**平均**，第一项在这些被平均的特征上鼓励稀疏解。于是多项逻辑回归问题变成

$$
\min_{\{\beta_{jk}\}}\ \sum_{i=1}^N\sum_{k=1}^K\sum_{j=1}^p\max\Bigl\{\log\Pr(g_i=k\mid x_i)-\lambda\bigl[\alpha\lvert\beta_{kj}\rvert+\tfrac{(1-\alpha)}{2}\beta_{kj}^2\bigr]\Bigr\} \eqno{18.21}
$$

**推导：elastic net 的 KKT 阈值规则。** 对坐标 $\beta_{kj}$ 求导（非零处），把二次项的 $-2\times\frac12$ 与一阶条件合并：

$$
\frac{\partial\ell}{\partial\beta_{kj}}+\lambda(1-\alpha)\beta_{kj}+\lambda\alpha\,\operatorname{sign}(\beta_{kj})=0
$$

设 $r_{kj}=-\partial\ell/\partial\beta_{kj}$，整理成不动点形式 $\beta_{kj}=\operatorname{sign}(r_{kj})\cdot g(\lvert r_{kj}\rvert)$，其中

$$
\hat\beta_{kj}=\operatorname{sign}(r_{kj})\Bigl(\frac{\lvert r_{kj}\rvert}{\lambda(1-\alpha)}-\frac{\alpha}{1-\alpha}\Bigr)_+
$$

**结果** elastic net 就是「**带偏移的软阈值**」：阈值是 $\lambda\alpha/(1-\alpha)$ 而不是 $\lambda$，收缩强度是 $1/(1-\alpha)$ 而不是 1。$\alpha=1$ 退化为 lasso（图 18.5 左图，19 个非零系数），$\alpha=0$ 退化为岭（图 18.5 右图 $\alpha=0.8$，39 个非零系数、量级更小——这正是「平均效应」的体现）。$\lambda\to0$ 的重标化极限：$\alpha=0$ 时与 SVM 的最大间隔解重合但**选中全部 7129 个基因**，$\alpha=1$ 时与一个 $L_1$ 分离超平面（Rosset et al., 2004a）重合且**至多 38 个基因**；$\alpha$ 从 1 减小，分离超平面里的基因单调增多。表 18.1 第 8 行用 $\alpha\in[0.05,1]$（20 个值）与 $\log$ 尺度上 100 个 $\lambda$ 做二维交叉验证，最小 CV 误差在 $\alpha\in[0.75,0.80]$。

### 18.4.1 应用：蛋白质质谱 {#s-18-4-1}

蛋白质质谱用来分析血液里的蛋白质，可用于诊断疾病。样本 $i$ 观测到许多飞行时间 $t_j$ 上的强度 $x_{ij}$；飞行时间与蛋白质的质荷比 $m/z$ 有已知关系，所以谱图上某个 $t_j$ 处出现一个峰，就说明存在有对应质量与电荷的蛋白质。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-4-1">原文 §18.4.1</a>

数据取自 Adam et al. (2003)：健康人与前列腺癌患者的平均谱（图 18.7）。总共 16898 个 $m/z$ 取值点，范围 2000 到 40000；完整数据集有 157 个健康人与 167 个癌症病人，目标是找出区分两组的 $m/z$ 点。这是一个**函数型数据**的例子：预测变量可以看作 $m/z$ 的函数。数据先标准化（基线扣除与归一化），并限制在 $m/z$ 2000–40000 之间，然后同时用 NSC 与 lasso 回归：

| 方法 | 测试错分 / 108 | 用的点数 |
|---|---|---|
| 最近收缩质心 | 34 | 459 |
| lasso | 22 | 113 |
| lasso on peaks | 28 | 35 |

**结果** 拟合得更紧的 lasso 测试错分明显更低（22 对 34），但**可能不给科学上有用的解**：理想的质谱应把样本分解成其组成蛋白质，这些蛋白质应表现为谱图上的峰。lasso 不对峰做任何特殊处理，所以不难理解只有一部分非零权重落在峰附近。另外，同一蛋白质在不同谱图上的峰可能落在略有不同的 $m/z$ 值上，要识别公共峰就需要样本间的 $m/z$ **扭曲对齐（warping）**。

原文的补救办法：先用标准峰提取算法对每条谱提峰，217 条训练谱共得 5178 个峰；把峰的**位置**沿 $\log(m/z)$ 轴做层次聚类，在高度 $\log(0.005)$ 处水平切树，对每个簇取峰位置平均，得到 728 个公共峰及其峰心。再判断每个公共峰在每条谱里是否存在、峰高多少（不存在则记 0），得到一个 $217\times728$ 的峰高矩阵，用它做 lasso 回归，测试谱同样按这 728 个峰打分。

**结果** 最后一行（lasso on peaks：28 个错分、35 个峰）不如直接对原始谱做 lasso，但对生物学家更有用：给出 35 个待研究的峰位置。反过来它也提示「峰与峰之间的位置差」可能含有判别信息，而第 2 行的 lasso 位置本身也值得进一步检查。

### 18.4.2 函数型数据的 fused lasso {#s-18-4-2}

上例的特征有天然顺序（由 $m/z$ 决定）。更一般地，函数型特征 $x_i(t)$ 按某个指标变量 $t$ 排序。本书已经讨论过几种利用这种结构的方法：把 $x_i(t)$ 用样条、小波或 Fourier 基展开，用基系数作预测变量再做回归（§5.3）；或者反过来用这些基表示**系数**（§12.6 的惩罚化判别分析，显式控制系数的平滑度）。这些方法的共同点是**均匀地**平滑系数。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-4-2">原文 §18.4.2</a>

fused lasso（Tibshirani et al., 2005）是更自适应的做法：修改 lasso 惩罚，使它显式利用特征的有序性。

$$
\min_{\beta_0,\beta_1,\dots,\beta_p}\ \sum_{i=1}^N\Bigl(y_i-\beta_0-\sum_{j=1}^p x_{ij}\beta_j\Bigr)^2+\lambda_1\sum_{j=1}^p\lvert\beta_j\rvert+\lambda_2\sum_{j=1}^{p-1}\lvert\beta_{j+1}-\beta_j\rvert \eqno{18.22}
$$

**结果** 这个准则的每一项都是凸函数，故极小点存在且唯一（只要 $X$ 列满秩，平方损失项已严格凸，$L_1$ 与差分惩罚都不破坏这一性质）。第一项惩罚鼓励稀疏，第二项鼓励在指标 $j$ 上平滑——它是 $\beta$ 的**总变差** $\mathrm{TV}(\beta)$。

若底层指标 $t$ 不等距（值 $t_j$），(18.22) 的差分惩罚应换成除以差值的**差商**形式：

$$
\lambda_2\sum_{j=1}^{p-1}\frac{\lvert\beta_{j+1}-\beta_j\rvert}{\lvert t_{j+1}-t_j\rvert} \eqno{18.23}
$$

即级数的每一项都有一个「惩罚修正因子」，使惩罚对 $t$ 的间距不敏感。

**核心推导：差分惩罚为什么强制系数分段常数。** 看坐标 $\beta_j$（$1<j<p$）的一阶条件。记 $\mathcal L(\beta)=\sum_i(y_i-\beta_0-x_i^\top\beta)^2$，记 $s_j\in\{-1,0,1\}$ 为 $\partial\lvert\beta_j\rvert/\partial\beta_j$ 的取值（$\beta_j=0$ 时取 0），$u_j,v_j\in\{-1,0,1\}$ 分别来自相邻两个差分项，则

$$
0=\frac{\partial\mathcal L}{\partial\beta_j}+\lambda_1 s_j+\lambda_2\bigl(u_j-v_j\bigr)
$$

*第 1 步：平台内部的相消。* 若 $\beta_{j-1}=\beta_j=\beta_{j+1}$，两个差分项对 $\beta_j$ 的贡献是 $+1$ 与 $-1$（同取 $\operatorname{sign}(\beta_j-\beta_{j-1})$），互相抵消，于是 KKT 退化为「$\ell_1$ 项的一阶条件」

$$
\frac{\partial\mathcal L}{\partial\beta_j}=\frac{\partial}{\partial\beta_j}\Bigl(\sum_i(y_i-\beta_0-x_i^\top\beta)^2\Bigr)=-2\sum_i\hat r_ix_{ij}=-\lambda_1s_j
$$

其中 $\hat r_i=y_i-\hat\beta_0-x_i^\top\hat\beta$ 是残差。

*第 2 步：一般 $X$ 下的结论。* 在整个平台上 $s_j$ 是常数（$\beta_j$ 同号），所以 $\sum_i\hat r_ix_{ij}$ 在平台上取同一个值。这是关于**残差**的条件，不直接说 $\hat\beta$ 逐坐标相等——对一般设计矩阵只能说「同一平台上各坐标的残差加权预测和相同」。

*第 3 步：$X=I$ 时结论变强。* 若 $x_i=e_i$，则 $\sum_i\hat r_ix_{ij}=\hat r_j$，上式直接给出 $\hat r_j=-\lambda_1s_j/2$，即

$$
\hat\beta_j=\hat\beta_0+y_j-\frac{\lambda_1}{2}
$$

**结果** 在 $X=I_N$ 的情形下，**同一平台上所有坐标都等于同一个值 $y_j-\lambda_1/2$ 的公共常数**：$\hat\beta_j=\hat\beta_0+\bar y-\lambda_1/2$，其中 $\bar y=\sum_iy_i/N$（因为 $\sum_j\hat\beta_j$ 的约束与 $\lambda_1$ 项共同作用）。平台高度与平台长度无关，只有平台的位置（在哪里断开）由 $\lambda_2$ 决定。这就是「分段常数」的严格来源：$\lambda_1$ 项钉死平台高度，$\lambda_2$ 项决定平台边界。

更直观的三个特例：$\lambda_1=0$ 时只剩 $\lambda_2$，解是分段的**线性**（TV 去噪）；$\lambda_2=0$ 时退化为普通 lasso（无平台约束）；$\lambda_1,\lambda_2\to\infty$ 时整条谱压成同一个常数。

**特例：fused lasso 信号逼近器。** 当预测变量矩阵是单位矩阵 $X=I_N$（$N\times N$）时，fused lasso 退化成用来逼近序列 $\{y_i\}_{i=1}^N$ 的**信号逼近器**：

$$
\min_{\beta_0,\beta_1,\dots,\beta_N}\ \sum_{i=1}^N\bigl(y_i-\beta_0-\beta_i\bigr)^2+\lambda_1\sum_{i=1}^N\lvert\beta_i\rvert+\lambda_2\sum_{i=1}^{N-1}\lvert\beta_{i+1}-\beta_i\rvert \eqno{18.24}
$$

**推导：$\lambda$ 路径的结构。** 把 (18.24) 的 KKT 写成差分 $\gamma_j=\hat\beta_{j+1}-\hat\beta_j$ 的形式：若 $\lvert\gamma_j\rvert<\lambda_2$，两个差分项相消，$\hat\beta_j$ 被钉在平台上；若 $\lvert\gamma_j\rvert>\lambda_2$，该位置的次梯度可以取 $\pm1$，允许跳变，代价是 $\lambda_2\lvert\gamma_j\rvert$。于是「是否跳变」由 $\lambda_2\lvert\gamma_j\rvert$ 与平方损失在这一步的收益 $\bigl(y_{j+1}-y_j\bigr)^2/2$ 相比决定。固定 $\lambda_2$ 让 $\lambda_1$ 变化时，平台的高度随 $\lambda_1/2$ 线性移动，只有当某个 $\lvert\gamma_j\rvert$ 越过 $\lambda_2$ 时平台的**边界**才改变。**结果** 解关于 $\lambda_1$ 是分段线性的，只在有限的若干「节点」处改变分段结构——这与回归 lasso 解关于 $\lambda$ 分段线性的结论平行（§3.4.2）。

图 18.8 是 Tibshirani and Wang (2007) 的例子：CGH 阵列数据，测量肿瘤样本中每个基因拷贝数与正常样本之比的近似 $\log_2$ 值，横轴是基因的染色体位置。癌细胞里基因常被扩增或缺失，而**这些事件倾向于出现在连续区域里**，所以平滑后的信号估计（深红色）里显著非零的区域可以用来定位基因的扩增与缺失。**结果** 这正是 fused lasso 的用武之地：$\lambda_2$ 抑制噪声级抖动，$\lambda_1$ 把无关区域压到 0。还有二维版本：参数排成像素网格，对左、右、上、下四个方向的差分都加惩罚，可用于去噪或图像分类，Friedman et al. (2007) 为一维和二维 fused lasso 给出了快速的广义坐标下降算法。
