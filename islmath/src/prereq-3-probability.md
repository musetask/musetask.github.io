## 概率论与统计推断

@src 全书通用 · 第 2、4、11、12、13 章的推导都要用到

### 随机变量、期望与方差

**基础**：随机变量 $X$ 是样本空间上的函数。期望是对取值按概率加权：

$$
\E[X] = \sum_x x\,P(X = x), \qquad \E[X] = \int x\,f_X(x)\,dx
$$

方差定义为离均值的平方期望：

$$
\mathrm{Var}(X) = \E\left[(X - \E[X])^2\right] = \E[X^2] - (\E[X])^2
$$

**推导**（第二个等式）。把被开方项展开：

$$
\E[(X-\mu)^2] = \E[X^2 - 2\mu X + \mu^2] = \E[X^2] - 2\mu\E[X] + \mu^2
$$

代入 $\mu = \E[X]$：$= \E[X^2] - 2(\E[X])^2 + (\E[X])^2 = \E[X^2] - (\E[X])^2$。
第一步用了期望的**线性性**——即使 $X$ 与 $\mu$ 不独立也成立，因为 $\mu$ 是常数。

**结论**：方差的非负性直接来自定义。书里式 (2.7) 的
$\E[(y_0-\hat f(x_0))^2] \ge \mathrm{Var}(\epsilon)$ 就是靠这个非负性 + Cauchy–Schwarz
（见下面偏差–方差分解）。

**易错点**：$\mathrm{Var}(X) = \E[X^2] - (\E[X])^2$ 里两项相减会严重抵消，
数值上要小心；协方差同理。$\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$（不是 $a$）。

### 期望的线性性与方差的可加性

$$
\E[aX + bY] = a\E[X] + b\E[Y], \qquad
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X,Y)
$$

**推导**。线性性：$\E[aX+bY] = \int (ax+by)\,dP = a\int x\,dP + b\int y\,dP$。
方差：展开 $\E[(X+Y)^2] - (\E[X+Y])^2$，
$\E[(X+Y)^2] = \E[X^2] + 2\E[XY] + \E[Y^2]$，
$(\E[X]+\E[Y])^2 = (\E[X])^2 + 2\E[X]\E[Y] + (\E[Y])^2$，
相减得 $\mathrm{Var}(X) + \mathrm{Var}(Y) + 2(\E[XY] - \E[X]\E[Y])$，
而括号里就是 $\mathrm{Cov}(X,Y)$。

**结论**：期望无条件线性；方差只在**独立**（更弱地，互不相关）时可加。
第 13 章 Bonferroni 校正只用期望的线性性；FDR 推导里才需要协方差项。

**易错点**：「独立 $\Rightarrow$ 不相关」成立，反之不成立。多个 p 值往往正相关，
这正是 Benjamini–Hochberg 比 Bonferroni 更有功效的原因。

### 大数定律与中心极限定理

**大数定律**：$x_1,\ldots,x_n$ 独立同分布、$\E[x_i]=\mu$、$\mathrm{Var}(x_i)=\sigma^2$，则

$$
\frac{1}{n}\sum_{i=1}^n x_i \xrightarrow[n\to\infty]{} \mu
\quad\text{几乎必然},\qquad
\E\left[\left(\frac{1}{n}\sum_i x_i - \mu\right)^2\right] = \frac{\sigma^2}{n}
$$

**推导**（第二个式子，用方差可加性）。记 $\bar x = \frac1n\sum_i x_i$：

$$
\mathrm{Var}(\bar x) = \frac{1}{n^2}\mathrm{Var}\left(\sum_i x_i\right)
= \frac{1}{n^2}\sum_i \mathrm{Var}(x_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
$$

而 $\E[\bar x] = \frac1n\sum_i\mu = \mu$，所以 $\mathrm{Var}(\bar x) = \E[(\bar x-\mu)^2]$。

**结论**：样本均值的标准差是 $\sigma/\sqrt{n}$。这是「为什么大样本有用」的全部理由，
也是第 3 章所有大样本近似（$t$ 分布、自由度 $n-p$）的来源。

**中心极限定理**：$(n/\sigma)(\bar x - \mu) \Rightarrow \mathcal{N}(0,1)$。

**易错点**：$1/\sqrt{n}$ 这个因子只对**独立**样本成立。相关样本（时间序列、聚类数据）
的有效样本量小于 $n$，标准误被低估。第 5 章讲交叉验证的方差时也强调这一点。

### 正态分布与标准化

$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad
z = \frac{x-\mu}{\sigma}, \quad \Phi(z) = \int_{-\infty}^z \frac{1}{\sqrt{2\pi}}e^{-t^2/2}\,dt
$$

**基础**：$X \sim \mathcal{N}(\mu,\sigma^2)$ 记作 $X \overset{d}{=} \mu + \sigma Z$，$Z\sim\mathcal{N}(0,1)$。

**推导**（线性变换的性质）。若 $X\sim\mathcal{N}(\mu,\sigma^2)$、$a\ne0$，
则 $aX+b \sim \mathcal{N}(a\mu+b, a^2\sigma^2)$。反推：设 $Y=aX+b$，
$y = ax+b \Rightarrow x = (y-b)/a$，$dx = dy/a$，

$$
f_Y(y) = f_X\left(\frac{y-b}{a}\right)\frac{1}{|a|}
= \frac{1}{\sigma\sqrt{2\pi}}\exp\left(-\frac{((y-b)/a-\mu)^2}{2\sigma^2}\right)\frac{1}{|a|}
= \frac{1}{\sqrt{2\pi}|a|\sigma}\exp\left(-\frac{(y - (a\mu+b))^2}{2a^2\sigma^2}\right)
$$

正是均值 $a\mu+b$、标准差 $|a|\sigma$ 的正态密度。

**结论**：正态分布在**线性变换下封闭**。第 3 章的 $\hat\beta_1 \pm 2\,\mathrm{SE}$
置信区间、第 13 章的 z 检验，全靠这一条。$t$ 分布则是「正态除以 $\sqrt{\chi^2/k}$」的结果。

**易错点**：$X\sim\mathcal{N}(\mu,\sigma^2)$ 的第二个参数是**方差**，
而 $\mathrm{sd}(X) = \sigma$。书里 $\sigma^2 = \mathrm{Var}(\epsilon)$ 与
$\hat\sigma^2 = \mathrm{MSE}$ 说的是同一件事的两个层次。

### 偏差–方差分解

设 $Y = f(X) + \epsilon$，$E[\epsilon|X] = 0$，$\mathrm{Var}(\epsilon) = \sigma^2$。
$\hat f$ 只依赖训练数据（视为固定），$x_0$ 固定：

$$
\E_{y_0,\mathcal{D}}\left[(y_0 - \hat f(x_0))^2\right]
= \left[\E_{\mathcal{D}}\hat f(x_0) - f(x_0)\right]^2
+ \mathrm{Var}_{\mathcal{D}}\!\left(\hat f(x_0)\right) + \sigma^2
$$

**推导**（逐步）。记 $m := \E_{\mathcal{D}}\hat f(x_0)$（对训练集求平均），$b := m - f(x_0)$。
把被开项写成 $(f(x_0) + \epsilon - \hat f) = \epsilon - (\hat f - m) - b$。设
$u = \hat f - m$，则 $E[u] = 0$，且 $y_0$ 与训练集独立，故 $u$ 与 $\epsilon$ 独立。

$$
\E[(y_0-\hat f)^2] = \E[(\epsilon - u - b)^2] = \E[\epsilon^2] + \E[u^2] + b^2
- 2\E[\epsilon u] - 2b\E[\epsilon] + 2b\E[u]
$$

交叉项逐个消失：$\E[\epsilon u] = \E[\epsilon]\E[u] = 0$（独立 + 零均值）；
$\E[\epsilon] = 0$；$\E[u] = 0$。剩下
$\E[\epsilon^2] = \mathrm{Var}(\epsilon) + (\E\epsilon)^2 = \sigma^2$，
$\E[u^2] = \mathrm{Var}(u) = \mathrm{Var}(\hat f)$，$b^2$ 就是平方偏差。

**结论**：这就是原书式 (2.7)。三项分别对应**平方偏差**、**方差**、**不可约误差**。
它解释了式 (2.3)：即使 $\hat f = f$ 也剩下 $\mathrm{Var}(\epsilon)$。

**易错点**：$b$ 是常数、$u$ 是随机的，平方偏差里不能有方差。混了这两种「方差」
是这一节最容易犯的错。另外不可约性是**模型层面**的：$\E[\epsilon|X]\ne 0$（比如漏了变量）
时 $f$ 本身就没定义对，方差分解的形式会变。

### 极大似然估计

给定样本 $x_1,\ldots,x_n$ 独立同分布，密度 $f(x;\theta)$。似然与对数似然：

$$
L(\theta) = \prod_{i=1}^n f(x_i;\theta), \qquad
\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i;\theta)
$$

$\hat\theta = \arg\max_\theta \ell(\theta)$（或 $\arg\max L$）。

**推导要点**（第 4 章 4.3.2 的做法）。以伯努利为例，$f(x;p) = p^x(1-p)^{1-x}$：

$$
\ell(p) = \sum_{i=1}^n \big[x_i\log p + (1-x_i)\log(1-p)\big], \quad
\frac{\partial \ell}{\partial p} = \sum_i \frac{x_i}{p} - \sum_i\frac{1-x_i}{1-p}
= \frac{n\bar x}{p} - \frac{n(1-\bar x)}{1-p}
$$

令 0（用分式合并：$\frac{n\bar x(1-p) - n(1-\bar x)p}{p(1-p)} = \frac{n(\bar x - p)}{p(1-p)}=0$）得
$p = \bar x$。二阶导 $-\frac{n\bar x}{p^2} - \frac{n(1-\bar x)}{(1-p)^2} < 0$，确为极大。

**结论**：$L(\hat\theta) = \prod_i f(x_i;\hat\theta)$ 是参数的似然；极大似然估计
就是让观测到的数据「最不像是偶然发生的」的那组参数。第 4 章用同一手法推出
多项分布的 LDA、高斯分布的 QDA、朴素贝叶斯的 $\hat\pi_k$。

**易错点**：$\hat\theta$ **依赖样本**，所以 $\hat f$ 有随机性；偏差–方差分解里的方差
正是这个随机性带来的。混淆「参数是常数」与「估计是随机变量」会让整个分解讲不通。

### 似然比与 p 值

$$
p = P_{H_0}\left(\text{观测到的手上统计量至少和实际观察到的一样极端}\right)
$$

对双侧检验，若检验统计量 $T \sim \mathcal{N}(0,1)$ 近似，取
$p = 2\{1 - \Phi(|T_{\text{obs}}|)\}$。等价地
$p = 2\,\Phi(-|T_{\text{obs}}|)$。

**基础**：p 值是**在原假设为真**的前提下，数据落到如此极端区域的概率。
它不是「原假设为真的概率」。

**结论**：$\Phi$ 的对称性 $\Phi(-z) = 1 - \Phi(z)$ 给出上面的两形式。
第 13 章 FDR–thresholding 的 q 值定义为
$q = \min\{q' : P_1(\#\{p_i \le q'\}\ge k) \le kq'\}$，与 p 值同族。

**易错点**：p 值小不等于效应大。第 13 章明确说「大的 p 值不一定意味着无效应」，
因为效应大小与样本量共同决定 p 值。

### 置信区间

$$
\hat\theta \pm z_{1-\alpha/2}\,\hat{\mathrm{se}}(\hat\theta)
$$

**推导**（为什么是这个形式）。由 CLT，$\frac{\hat\theta-\theta}{\hat{\mathrm{se}}}\approx \mathcal{N}(0,1)$。
要求覆盖概率为 $1-\alpha$：$P\left(\left|\frac{\hat\theta-\theta}{\hat{\mathrm{se}}}\right|
\le z_{1-\alpha/2}\right) = 1-\alpha$。把 $\hat{\mathrm{se}}$ 乘进去再解出 $\theta$ 的范围即得。

**结论**：$95\%$ 区间对应 $z_{0.975} = 1.96$，故常写成 $\hat\theta \pm 2\hat{\mathrm{se}}$。
回归系数标准误 $\hat{\mathrm{se}}(\hat\beta_j) = \sqrt{\hat\sigma^2\,[(X^\top X)^{-1}]_{jj}}$，
用 $t_{n-p}$ 代替正态分位数更精确。

**易错点**：频率派的 95% 区间**不是**「$\theta$ 有 95% 的概率落在这里」；
贝叶斯解释才是（此时它是可信区间）。第 11 章生存分析里两套都出现。
