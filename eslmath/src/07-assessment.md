---
id: m07
n: "7"
title: 模型评估与选择
title_en: Model Assessment and Selection
desc: 偏差–方差分解、训练误差的乐观性与协方差公式、样本内误差估计（Cp/AIC/BIC/有效自由度/MDL）、VC 维与结构风险最小化、交叉验证与广义交叉验证、bootstrap 的 .632 与 .632+ 估计量
prev: m06
next: m08
prev_title: 第 6 章 核平滑方法
next_title: 第 8 章 模型推断与平均
---

# 7 模型评估与选择 {#s-7}

前面六章都在讲「怎么拟合」。本章讲的是更难受的问题：**拟合出来的东西，到底准不准**。这是全书唯一一章专门处理「不确定性」的章节，它的结论会回头影响第 8 章的 bagging、第 12 章 SVM 的模型选择和第 18 章的样本量计算。

本章的逻辑主线有两条。**第一条是「把误差拆开」**：任何一个预测误差都能拆成「不可约误差 + 平方偏差 + 方差」，拆开以后才能看清模型复杂度在做什么（第 3 节）。**第二条是「用同一批数据估计那个被拆出来的量」**：训练误差乐观多少（用协方差估计，第 4 节）、样本内误差要加多少修正（Cp/AIC/BIC，第 5、7 节）、模型复杂度该用什么量来数（有效自由度、VC 维，第 6、9 节）、没有留出数据时怎么把数据复用起来（交叉验证、bootstrap，第 10、11 节）。

写作约定：本章一律用平方误差损失与连续响应 $Y=f(X)+\varepsilon$，其中 $E[\varepsilon]=0$、$\mathrm{Var}(\varepsilon)=\sigma_\varepsilon^2$（见预备知识 P1）。分类情形在 (7.5)(7.6) 给出对应的损失函数，其余部分只需把「平方」换成「0–1」即可。

---

## 7.1 引言 {#s-7-1}

评估的对象有两个，必须分清：

- **模型选择（model selection）**：估计不同模型的性能，挑最好的那个。
- **模型评估（model assessment）**：模型定下来之后，估它在**新数据**上的预测误差。

两者的区别在实践后果上很具体。做一次模型选择时，我们会检查 $M$ 个模型的误差，然后挑最小的那个；被挑中的那个模型的误差估计天然偏低——因为「最小」这件事本身用了数据。评估则是拿一个**已经定死**的模型去估误差，没有这一步挑选，因此没有这个偏差。第 8 章 (8.5) 的 BMA 权重 $\mathrm{Pr}(\mathcal M_m\mid Z)$ 就是把这个选择过程也算进去的正确做法。

本章的记号约定：$\mathcal T$ 表示训练集，$x_i\in\mathbb{R}^p$，$y_i$ 为响应，$X$ 为 $N\times p$ 设计矩阵（第 $i$ 行为 $x_i^\top$），$S$ 为 $N\times N$ 平滑（帽子）矩阵。$\lVert\cdot\rVert$ 与 $\operatorname{tr}$ 分别表示欧氏范数与迹。

如果数据很多，最干净的做法是随机切成三份：训练集用来拟合，验证集用来选择，测试集用来评估。测试集应该锁在「金库」里，只在最后打开一次。如果反复用测试集来挑模型，最终选中模型的测试误差会明显低估真误差——这是选择效应（selection effect），第 8 章会给出精确形式。

三份各占多少没有一个通用规则，取决于数据的信噪比与训练样本量。一个典型切法是训练 50%、验证 25%、测试 25%。图 7.1 用的就是这种情形：$N=50$ 的训练集，100 次重复，画出 $\mathrm{err}$（浅蓝）与 $\mathrm{Err}_{\mathcal T}$（浅红）随模型复杂度的变化，$\mathrm{Err}$ 与 $E[\mathrm{err}]$ 是两条实线。浅红曲线整体高于实线的间隙，就是 (7.9) 里偏差与方差之和。

数据不够切三份时，本章的方法分两类：**解析近似**（AIC、BIC、MDL、SRM、GCV）与**数据复用**（交叉验证、bootstrap）。前者的共同骨架是「训练误差 + 一个只依赖复杂度 $d$ 的修正项」，所以关键是「怎么数复杂度」；后者的共同骨架是「把数据复用成若干个训练/测试对」，所以关键是「这些训练集有多像原来的训练集」。本章要回答的核心问题是：这些方法各自估计的是哪个量、偏多少、方差多大。

三条主线在本章的位置：(7.9) 的偏差–方差分解告诉我们理想复杂度在哪里；7.4 的乐观度公式告诉我们训练误差要加多少才不是自夸；7.6 的 $\operatorname{trace}(S)$ 告诉我们「$d$」在不规则模型里该换成什么。这三件事分别对应「目标」「偏差修正」「复杂度度量」，缺一不可。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-1">原文 §7.1</a>

---

## 7.2 误差的来源与损失函数 {#s-7-2}

### 7.2.1 四种损失函数 {#s-7-2-1}

要衡量 $Y$ 与 $\hat f(X)$ 之间的差距，先选损失函数 $L(Y,\hat f(X))$。连续响应最常用的两个是

$$
L(Y,\hat f(X))=\begin{cases}\bigl(Y-\hat f(X)\bigr)^2 & \text{平方误差}\\ \bigl|Y-\hat f(X)\bigr| & \text{绝对误差}\end{cases}\eqno{7.1}
$$

**测试误差**（又称泛化误差）是独立测试样本上的误差：

$$
\mathrm{Err}_{\mathcal T}=E\bigl[L(Y,\hat f(X))\mid\mathcal T\bigr]\eqno{7.2}
$$

这里训练集 $\mathcal T$ 固定，只对总体里的 $(X,Y)$ 取期望。相关的还有**期望预测误差**

$$
\mathrm{Err}=E\bigl[L(Y,\hat f(X))\bigr]=E\bigl[\mathrm{Err}_{\mathcal T}\bigr]\eqno{7.3}
$$

(7.3) 的第二个等号就是塔性质（预备知识 P2）：对训练集再取一次期望，得到完全平均掉随机性的量。**分清 $\mathrm{Err}_{\mathcal T}$ 与 $\mathrm{Err}$ 是本章最重要的一件事**——前者条件于训练集，后者只依赖总体分布。前者依赖我们手上这份数据，后者是可分析的；后面会看到，绝大部分方法估的是后者。

**训练误差**是训练样本上的平均损失：

$$
\mathrm{err}=\frac{1}{N}\sum_{i=1}^{N}L\bigl(y_i,\hat f(x_i)\bigr)\eqno{7.4}
$$

离散响应 $G\in\{1,\dots,K\}$ 时，损失函数换成

$$
L\bigl(G,\hat G(X)\bigr)=I\bigl(G\ne\hat G(X)\bigr)\ (\text{0–1 损失}),\qquad \sum_{k=1}^{K}I(G=k)\bigl(-2\log\hat p_k(X)\bigr)\eqno{7.5}
$$

$$
=-2\log\hat p_G(X)\ \ (\text{两倍负对数似然})\eqno{7.6}
$$

-2 乘对数似然叫 deviance。取 -2 的好处是：高斯情形下它与平方误差只差常数（第 3 章 $\sum_i(y_i-\hat f(x_i))^2/\sigma_\varepsilon^2 = -2\log\text{lik}$），所以同一套推导可以两边复用。对应的训练误差是样本对数似然

$$
\mathrm{err}=\frac{1}{N}\sum_{i=1}^{N}\bigl(-\log\hat p_{g_i}(x_i)\bigr)\eqno{7.7}
$$

更一般地，若 $Y$ 的密度是 $\mathrm{Pr}_{\theta(X)}(Y)$，就把损失定义成

$$
L(Y,\theta(X))=-2\cdot\log\mathrm{Pr}_{\theta(X)}(Y)\eqno{7.8}
$$

### 7.2.2 模型选择与模型评估的区别 {#s-7-2-2}

给定调参 $\alpha$（控制复杂度），预测写作 $\hat f_\alpha(x)$。我们想找的是让平均测试误差最小的 $\alpha$，也就是 (7.3) 曲线的极小点。但在做这件事之前要先分清两个目标：模型选择关心「哪一族模型最好」，模型评估关心「最终选中的这一个误差是多少」。这两件事对方法的要求不同——选择只在乎**相对**大小，评估在乎**绝对**精度。这一差别在 7.11.1 节会再次出现：给所有候选加同一个常数不改变选择结果，但会破坏评估。

> **坑** · 平方误差与 0–1 损失下最优的 $\alpha$ 可能完全不同。偏差–方差**加法**分解只在平方误差下成立；0–1 损失下预测误差不是平方偏差与方差之和（习题 7.2 会算出精确形式 (7.62)）。所以「用回归的折中点去做分类」是错的。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-2">原文 §7.2</a>

---

## 7.3 偏差–方差分解 {#s-7-3}

### 7.3.1 三项分解 {#s-7-3-1}

**问题**：书里 (7.9) 把平方误差损失下的期望预测误差拆成三项。每一项从哪来？

**第一步：写出误差。** 固定输入点 $x_0$，模型 $Y=f(X)+\varepsilon$，则

$$
\mathrm{Err}(x_0)=E\bigl[(Y-\hat f(x_0))^2\mid X=x_0\bigr]
$$

**第二步：代入误差模型并展开。** $Y-\hat f(x_0)=f(x_0)-\hat f(x_0)+\varepsilon$。注意 $\hat f(x_0)$ 只依赖训练集，与 $\varepsilon$ 独立，且 $E[\varepsilon]=0$：

$$
E\bigl[(Y-\hat f(x_0))^2\mid x_0\bigr]=E\bigl[(f-\hat f+\varepsilon)^2\mid x_0\bigr]=E\bigl[(f-\hat f)^2\mid x_0\bigr]+2E[f-\hat f\mid x_0]E[\varepsilon]+\mathrm{Var}(\varepsilon)
$$

**交叉项为零**（用 $E[\varepsilon]=0$），剩下

$$
E\bigl[(f-\hat f)^2\mid x_0\bigr]=E\bigl[\bigl(f-E\hat f+E\hat f-\hat f\bigr)^2\mid x_0\bigr]=\bigl[f(x_0)-E\hat f(x_0)\bigr]^2+E\bigl[\bigl(\hat f(x_0)-E\hat f(x_0)\bigr)^2\mid x_0\bigr]
$$

**第三步：读出三项。**

$$
=\sigma_\varepsilon^2+\mathrm{Bias}^2\bigl(\hat f(x_0)\bigr)+\mathrm{Var}\bigl(\hat f(x_0)\bigr)\eqno{7.9}
$$

> **结果** · 平方误差损失下的期望误差 = **不可约误差** $\sigma_\varepsilon^2$ + **平方偏差** + **方差**。
>
> - $\sigma_\varepsilon^2$：$Y$ 围绕真均值的波动，任何估计都躲不开，除非 $\sigma_\varepsilon^2=0$。
> - $\mathrm{Bias}(\hat f(x_0))=E[\hat f(x_0)]-f(x_0)$：平均估计与真均值之差。
> - $\mathrm{Var}(\hat f(x_0))$：$\hat f$ 围绕自己均值的散布。
>
> 模型越复杂，偏差越小、方差越大；最优复杂度在中间。这是全章图 7.1 的全部内容。

**这个分解依赖的假设**：$E[\varepsilon]=0$（否则出现 $2(f-E\hat f)E[\varepsilon]$ 交叉项）；$\hat f(x_0)$ 与 $\varepsilon$ 独立（数据含噪且噪声与设计无关）；平方误差损失（换成 0–1 损失第二步就断了）。

**为什么这三项的取舍必然是「此消彼长」**。把 $\hat f$ 看作一族随复杂度 $\alpha$ 变化的估计器：$\alpha$ 大 ⇒ 能拟合更多结构 ⇒ $E[\hat f(x_0)]$ 更接近 $f(x_0)$ ⇒ $\mathrm{Bias}^2\downarrow$；但 $\hat f$ 用训练数据的地方更多 ⇒ 它对 $\mathcal T$ 的敏感度更高 ⇒ $\mathrm{Var}\uparrow$。两者相加的极小点就是图 7.1 实红曲线的谷底。

**图 7.3 的分类情形给出一个反直觉的结果**：右下角图里，线性分类器的预测误差最小点与回归情形一样在 $p\ge10$，但从 $p=1$ 改进到 $p=10$ 的幅度大得多；而 k 近邻分类器的误分类率随 $k$ 增大到 20 一直不升，尽管平方偏差在上升。原因是**决策边界附近的容错空间**：若真概率为 0.9 而估计的均值为 0.6，平方偏差 $(0.6-0.9)^2$ 可观，但只要都在 $\frac12$ 的同一侧就一步不错。所以 0–1 损失下偏差与方差的相互作用（习题 7.2 的 (7.62)），最优点可能与平方误差下的最优点不同。

### 7.3.2 两个特例：k 近邻与线性模型 {#s-7-3-2}

**k 近邻回归**。设训练输入 $x_i$ 固定，随机性只来自 $y_i$。记 $x_{(\ell)}$ 为 $x_0$ 的第 $\ell$ 近邻，则 $\hat f_k(x_0)=\frac1k\sum_{\ell=1}^k y_{(\ell)}$，于是

$$
\hat f_k(x_0)=\frac1k\sum_{\ell=1}^{k}\bigl[f(x_{(\ell)})+\varepsilon_{(\ell)}\bigr]\ \Rightarrow\ E[\hat f_k(x_0)]=\frac1k\sum_{\ell=1}^k f(x_{(\ell)}),\qquad \mathrm{Var}(\hat f_k(x_0))=\frac{k\sigma_\varepsilon^2}{k^2}=\frac{\sigma_\varepsilon^2}{k}
$$

代入 (7.9)：

$$
\mathrm{Err}(x_0)=\sigma_\varepsilon^2+\left[f(x_0)-\frac1k\sum_{\ell=1}^k f\bigl(x_{(\ell)}\bigr)\right]^2+\frac{\sigma_\varepsilon^2}{k}\eqno{7.10}
$$

$k$ 与复杂度反向对应：$k$ 小偏差小方差大，$k$ 大反之。

**最小二乘线性模型**。$X$ 为 $N\times p$ 设计矩阵，$\hat f_p(x_0)=x_0^\top\hat\beta = x_0^\top(X^\top X)^{-1}X^\top y$，记

$$
h(x_0)=X(X^\top X)^{-1}x_0\ \ (\mathbb{R}^N\text{ 向量}),\qquad \hat f_p(x_0)=h(x_0)^\top y
$$

方差是 $h(x_0)$ 与 $y$ 的二次型（预备知识 L5 的二次型方差公式）：

$$
\mathrm{Var}\bigl[\hat f_p(x_0)\bigr]=h(x_0)^\top(\sigma_\varepsilon^2I_N)h(x_0)=\lVert h(x_0)\rVert^2\,\sigma_\varepsilon^2
$$

$$
\mathrm{Err}(x_0)=\sigma_\varepsilon^2+\bigl[f(x_0)-E\hat f_p(x_0)\bigr]^2+\lVert h(x_0)\rVert^2\sigma_\varepsilon^2\eqno{7.11}
$$

**为什么平均以后方差项变成 $p/N$**：这是全章最有用的一步计算。把 $x_i$ 代入 $h(x_i)=X(X^\top X)^{-1}x_i$，则

$$
\frac1N\sum_{i=1}^{N}\lVert h(x_i)\rVert^2=\frac1N\sum_{i=1}^{N}x_i^\top(X^\top X)^{-1}X^\top X(X^\top X)^{-1}x_i=\frac1N\operatorname{tr}\!\Bigl((X^\top X)^{-1}\sum_{i=1}^{N}x_ix_i^\top\Bigr)=\frac1N\operatorname{tr}(I_p)=\frac pN
$$

用了三条：$(X^\top X)^{-1}$ 对称所以两个因子可以夹在一起；$\sum_i x_ix_i^\top = X^\top X$；$\operatorname{tr}(I_p)=p$（预备知识 L1）。于是**样本内**误差

$$
\frac1N\sum_{i=1}^{N}\frac1N\sum_{i'=1}^{N}\mathrm{Err}(x_i)=\sigma_\varepsilon^2+\frac1N\sum_{i=1}^{N}\bigl[f(x_i)-E\hat f(x_i)\bigr]^2+\frac pN\sigma_\varepsilon^2\eqno{7.12}
$$

> **坑** · (7.12) 里的 $p/N$ 只在**固定设计**（$x_i$ 固定、只有 $y_i$ 随机）时精确成立；随机设计下要对 $X$ 也取期望，结论形式不变但要用期望设计矩阵。ridge 情形只要把 $h(x_0)=X(X^\top X+\alpha I)^{-1}x_0$ 代入 (7.11)，偏差项也一起变。

**k 近邻的复杂度参数为什么是 $N/k$**。$k$ 近邻估计量是核平滑的特例（Nadaraya–Watson，预备知识 C3 的卷积恒等式给出其偏差 $\approx-m_K/h$，在等距 $k$ 邻居下 $h\propto1/k$）。它的「参数个数」应理解为 $N/k$，这正是习题 7.6 的结论，也是图 7.7 里 KNN 的 VC 维取 $N/k$ 的理由。

### 7.3.3 模型偏差与估计偏差 {#s-7-3-3}

**问题**：(7.14) 把平方偏差再拆成两块，为什么？

先定义「最优线性逼近」。设模型空间是 $\{x^\top\beta\}$，记

$$
\beta^\star=\arg\min_\beta E\bigl[(X^\top\beta-f(X))^2\bigr]\eqno{7.13}
$$

期望对 $X$ 的分布取。把平均平方偏差写成三段之和再展开：

$$
E_{x_0}\bigl[f(x_0)-E\hat f_\alpha(x_0)\bigr]^2=E_{x_0}\bigl[f(x_0)-x_0^\top\beta^\star+x_0^\top\beta^\star-E(x_0^\top\hat\beta_\alpha)\bigr]^2
$$

记 $b(x_0)=f(x_0)-x_0^\top\beta^\star$，$a(x_0)=x_0^\top\bigl(\beta^\star-E\hat\beta_\alpha\bigr)$。展开 $E(b+a)^2=E[b^2]+E[a^2]+2E[ba]$。**交叉项为零**，因为

$$
E_{x_0}[x_0b_{\rm lin}(x_0)]=E[X]\bigl(\beta^\star-E\hat\beta_\alpha\bigr)=0
$$

这里 $b_{\rm lin}$ 是 $x_0^\top(\beta^\star-E\hat\beta_\alpha)$：由 (7.13) 的定义，$E[XX^\top](\beta^\star-\tilde\beta)=0$ 对一切 $\tilde\beta$ 成立（$\beta^\star$ 是 $E[XX^\top]$ 加权的最小二乘解）。于是

$$
E_{x_0}\bigl[f(x_0)-E\hat f_\alpha(x_0)\bigr]^2=E_{x_0}\bigl[f(x_0)-x_0^\top\beta^\star\bigr]^2+E_{x_0}\bigl[x_0^\top\beta^\star-E(x_0^\top\hat\beta_\alpha)\bigr]^2\eqno{7.14}
$$

$$
=\operatorname{Ave}\bigl[\text{模型偏差}\bigr]^2+\operatorname{Ave}\bigl[\text{估计偏差}\bigr]^2
$$

> **结果** · 平方偏差 = **模型偏差**（真函数与模型空间里最近点的距离，只能靠换更丰富的模型类来减小）+ **估计偏差**（样本估计的均值与那个最近点的距离，只能靠减小方差来减小）。
>
> 最小二乘的估计偏差**恰好为零**（$\hat\beta$ 无偏）；ridge、lasso、best-subset 的估计偏差为正，正是它们用一点偏差换取方差下降的原因。图 7.2 里「shrunken fit」偏离「closest fit in population」的那一段就是估计偏差。

**图 7.2 的读法**：黑色圆点标「closest fit in population」即 $x^\top\beta^\star$，它是线性空间里离真函数最近的点；蓝色阴影区域是训练样本中观测到的真值周围的 $\sigma_\varepsilon$ 噪声带；黄色大圆是最小二乘拟合的方差 $\lVert h(x_0)\rVert^2\sigma_\varepsilon^2$ 的示意。若改用更少预测变量或把系数向零收缩，就得到图中的「shrunken fit」：它离 $\beta^\star$ 更远（多了一块**估计偏差**），但圆（方差）更小。**当方差下降超过平方偏差上升时，压缩就是划算的**——这就是所有正则化方法的统一判据。

**判据写成不等式**。设收缩使平方偏差增加 $\Delta b$，方差减少 $\Delta v$，则 (7.14)+(7.11) 给出总误差变化 $\Delta b-\Delta v$。对岭回归可以精确算出：令 $S_\alpha=X(X^\top X+\alpha I)^{-1}X^\top$，则 $\hat f_\alpha(x)=x_0^\top(X^\top X+\alpha I)^{-1}X^\top y=h_\alpha(x_0)^\top y$，

$$
\Delta v=\Bigl(\lVert h_\alpha(x_0)\rVert^2-\lVert h(x_0)\rVert^2\Bigr)\sigma_\varepsilon^2\le0,\qquad \Delta b=\Bigl(\beta^\star-E\hat\beta_\alpha\Bigr)^\top E[XX^\top]\Bigl(\beta^\star-E\hat\beta_\alpha\Bigr)\ge0
$$

（$\hat\beta_\alpha=(X^\top X+\alpha I)^{-1}X^\top y$，$E\hat\beta_\alpha=(X^\top X+\alpha I)^{-1}X^\top E[y]$，故 $E\hat\beta_\alpha-\beta^\star$ 由收缩量决定，可显式算出。）两者的 $\alpha$ 依赖性不同，正是最优 $\alpha$ 存在的理由。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-3">原文 §7.3</a>

---

## 7.4 训练误差的乐观性 {#s-7-4}

这一节是本章的**技术核心**，后面所有解析方法（Cp、AIC、BIC、GCV）都是它的推论。

### 7.4.1 分清三个量 {#s-7-4-1}

$$
\mathrm{Err}_{\mathcal T}=E_{X_0,Y_0}\bigl[L(Y_0,\hat f(X_0))\mid\mathcal T\bigr]\eqno{7.15}
$$

$$
\mathrm{Err}=E_{\mathcal T}E_{X_0,Y_0}\bigl[L(Y_0,\hat f(X_0))\bigr]\eqno{7.16}
$$

$$
\mathrm{err}=\frac1N\sum_{i=1}^{N}L\bigl(y_i,\hat f(x_i)\bigr)\eqno{7.17}
$$

(7.15) 中 $(X_0,Y_0)$ 是**全新**的一点，与训练集独立。(7.16) 里 $\mathrm{Err}$ 是纯总体量。

(7.17) 通常明显小于 $\mathrm{Err}_{\mathcal T}$，因为同一批数据既用来拟合又用来评估。为了把这种「自夸」量化出来，先考虑**样本内**误差：只在训练点上重新采一次响应。

$$
\mathrm{Err}_{\mathrm{in}}=\frac1N\sum_{i=1}^{N}E_{Y_0}\bigl[L(Y_i^0,\hat f(x_i))\mid\mathcal T\bigr]\eqno{7.18}
$$

$Y_i^0$ 表示在**同一个** $x_i$ 上新观测到的响应。定义**乐观度**与**平均乐观度**：

$$
\mathrm{op}\equiv\mathrm{Err}_{\mathrm{in}}-\mathrm{err}\eqno{7.19}
$$

$$
\omega\equiv E_y(\mathrm{op})\eqno{7.20}
$$

预测点 $x_i$ 固定、只对训练响应 $y$ 取期望，所以记 $E_y$ 而不是 $E_{\mathcal T}$。$\mathrm{op}$ 通常为正：训练误差低估了预测误差。

> **坑** · $\mathrm{Err}_{\mathrm{in}}$ 本身不是我们真正关心的量（未来的 $X$ 不会恰好等于训练点），但它**便于**推导且与 $\mathrm{err}$ 只差一个可算的量。所有基于「训练误差 + 乐观度」的方法都是这个思路。

**为什么必须先算 $\mathrm{Err}_{\mathrm{in}}$**：因为 $\mathrm{Err}_{\mathcal T}$ 的被积函数里 $\hat f$ 依赖于 $\mathcal T$、测试点依赖于新的 $X_0$，两处随机性纠缠在一起，没法分离。而 $\mathrm{Err}_{\mathrm{in}}$ 固定了评估点 $x_i$，只剩响应 $Y_i^0$ 一处随机性，于是 $E_{Y_0}$ 可以一步步算。算出乐观度后，再补上「评估点不重合」的那部分，就得到真正想估的 $\mathrm{Err}$——这正是 7.5 节 $C_p$/AIC 的由来。

### 7.4.2 核心推导：乐观度 = 协方差之和 {#s-7-4-2}

**问题**：把 (7.18) 与 (7.17) 都展开，证明 (7.21)。

**准备工作：两个约定。** 设计固定，$y_i=f_i+\varepsilon_i$，其中 $f_i=f(x_i)$ 视为常数、$\varepsilon_i$ 独立同分布（均值 0、方差 $\sigma_\varepsilon^2$）。记 $\hat y_i=\hat f(x_i)$，$\bar{\hat f}_i=E_y[\hat f(x_i)]$（只对训练响应取期望）。

**第一步：算 $\mathrm{Err}_{\mathrm{in}}$。** 在 $x_i$ 处新观测的响应 $Y_i^0$ 与训练集独立，故

$$
E_{Y_0}\bigl[(Y_i^0-\hat y_i)^2\mid\mathcal T\bigr]=E_{Y_0}\bigl[(Y_i^0)^2\mid\mathcal T\bigr]-2\hat y_iE_{Y_0}[Y_i^0]+\hat y_i^2=(f_i^2+\sigma_\varepsilon^2)-2f_i\hat y_i+\hat y_i^2=(f_i-\hat y_i)^2+\sigma_\varepsilon^2
$$

$$
\mathrm{Err}_{\mathrm{in}}=\frac1N\sum_{i=1}^{N}(f_i-\hat y_i)^2+\sigma_\varepsilon^2\tag{A}
$$

注意 (A) 里 $\hat y_i$ 仍是随机量，$f_i$ 是常数。

**第二步：算 $\mathrm{err}$。** 把两个因子各自拆成「均值 + 中心化」：

$$
f_i-\hat y_i=\underbrace{(f_i-\bar{\hat f}_i)}_{A_i}-\underbrace{(\hat y_i-\bar{\hat f}_i)}_{B_i},\qquad y_i-\hat y_i=A_i-B_i+\varepsilon_i
$$

其中 $E_y[B_i]=0$、$E_y[\varepsilon_i]=0$。平方：

$$
E_y\bigl[(y_i-\hat y_i)^2\bigr]=E_y[A_i^2]+E_y[B_i^2]+E_y[\varepsilon_i^2]+\underbrace{2E_y[-A_iB_i]}_{0}+\underbrace{2E_y[A_i\varepsilon_i]}_{0}+\underbrace{2E_y[-B_i\varepsilon_i]}_{-2\,\mathrm{Cov}(\hat y_i,\varepsilon_i)}
$$

三个交叉项为零的理由必须逐个说清，不能略：

- $E_y[A_iB_i]=0$：$A_i$ 只依赖固定的 $x_i$，故 $E_y[A_i]=0$，与 $B_i$ 无关。
- $E_y[A_i\varepsilon_i]=0$：同样 $A_i$ 均值为零且与 $\varepsilon_i$ 独立。
- $E_y[B_i\varepsilon_i]=\mathrm{Cov}(\hat y_i,\varepsilon_i)$：由 $B_i=\hat y_i-\bar{\hat f}_i$ 且 $E_y[\varepsilon_i]=0$。

**第三步：把协方差换成 $\mathrm{Cov}(\hat y_i,y_i)$。** 因为 $\hat y_i$ 是 $y$ 的线性（广义）函数、不含 $\varepsilon_i$ 的常数部分，

$$
\mathrm{Cov}(\hat y_i,\varepsilon_i)=\mathrm{Cov}(\hat y_i,y_i-f_i)=\mathrm{Cov}(\hat y_i,y_i)-f_i\,\mathrm{Cov}(\hat y_i,1)=\mathrm{Cov}(\hat y_i,y_i)
$$

（用了 $f_i$ 是常数。）$E_y[\varepsilon_i^2]=\sigma_\varepsilon^2$ 也代进去：

$$
\mathrm{err}=\frac1N\sum_{i=1}^{N}A_i^2+\frac1N\sum_{i=1}^{N}B_i^2+\sigma_\varepsilon^2+\frac2N\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)\tag{B}
$$

**第四步：作差。** 由 (A)，$\mathrm{Err}_{\mathrm{in}}=\frac1N\sum_i(f_i-\hat y_i)^2+\sigma_\varepsilon^2=\frac1N\sum_i(A_i-B_i)^2+\sigma_\varepsilon^2=\frac1N\sum_iA_i^2+\frac1N\sum_iB_i^2+\sigma_\varepsilon^2$（$E_y[A_iB_i]=0$）。与 (B) 相减，**前两项完全抵消**，只剩协差项：

$$
\mathrm{op}=\frac2N\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)
$$

$$
\frac2N\sum_{i=1}^{N}\omega_i=\frac2N\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)\eqno{7.21}
$$

（$\omega_i:=E_y[\,(y_i^0-\hat y_i)^2-(y_i-\hat y_i)^2\,]$ 是第 $i$ 个观测的乐观度，$\omega=\frac1N\sum_i\omega_i$。）合并 (7.19)(7.20) 得到期望版本的 (7.22)：

$$
\frac1N\sum_{i=1}^{N}E_y\bigl(\mathrm{Err}_{\mathrm{in}}\bigr)=E_y\bigl(\mathrm{err}\bigr)+\frac2N\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)\eqno{7.22}
$$

> **结果** · **训练误差低估了多少，完全由「$y_i$ 对自己的预测 $\hat y_i$ 的影响有多强」决定。**
>
> - 拟合得越狠，$\mathrm{Cov}(\hat y_i,y_i)$ 越大，乐观度越大。
> - 0–1 损失下 $\hat y_i\in\{0,1\}$ 是分类结果；熵损失下 $\hat y_i\in[0,1]$ 是拟合概率。结论形式不变。
> - **插值情形**（$p=N$，$\mathrm{tr}(S)=N$）：由 (7.23)(7.32)，$\sum_i\mathrm{Cov}(\hat y_i,y_i)=N\sigma_\varepsilon^2$，故 $\mathrm{op}=2\sigma_\varepsilon^2$，与「训练误差为 0、测试误差约 $2\sigma_\varepsilon^2$」这一著名事实完全一致。

**三种「误差」的对照表**。把本节四个量放在一起，读者最容易混淆的就是它们：

| 量 | 表达式 | 评估点 | 响应的随机性 | 与真实测试误差的关系 |
|---|---|---|---|---|
| 训练误差 | $\mathrm{err}$ (7.17) | 训练点 $x_i$ | 与 $\hat f$ 共享同一批 $y$ | 乐观，最小 |
| 样本内误差 | $\mathrm{Err}_{\mathrm{in}}$ (7.18) | 训练点 $x_i$ | 独立新抽 $Y_i^0$ | 等于 $\mathrm{err}+\mathrm{op}$ |
| 条件测试误差 | $\mathrm{Err}_{\mathcal T}$ (7.15) | 新点 $X_0$ | 新抽 | 不小于 $\mathrm{Err}_{\mathrm{in}}$ |
| 期望测试误差 | $\mathrm{Err}$ (7.16) | 新点，再对 $\mathcal T$ 平均 | 新抽 | 本章方法实际估的量 |

四行单调不减，而 $\mathrm{Err}_{\mathrm{in}}\le\mathrm{Err}_{\mathcal T}$ 这一步书里没有展开，理由是 $x_0$ 恰落在某个训练点（或其邻域）时拟合更准、误差更小，取期望后必然如此。

**线性情形化简。** 若 $\hat y=Sy$，$S$ 为 $N\times N$ 线性平滑矩阵，则

$$
\mathrm{Cov}(\hat y_i,y_i)=\mathrm{Cov}\Bigl(\sum_{j=1}^N S_{ij}y_j,\ y_i\Bigr)=\sum_{j=1}^{N}S_{ij}\mathrm{Cov}(y_j,y_i)=S_{ii}\sigma_\varepsilon^2
$$

（用了 $\mathrm{Cov}(y_j,y_i)=\sigma_\varepsilon^2\delta_{ij}$，因为误差独立。）这一行是整节的枢纽：它把「预测对 $y_i$ 的影响」压缩成一个数 $S_{ii}$（第 $i$ 个观测的**杠杆值**），而且这个数只依赖设计、不含 $y$，因此**可估**。

若拟合用了 $d$ 个基函数，最小二乘下 $S=X(X^\top X)^{-1}X^\top$ 是到 $d$ 维空间的投影，$\operatorname{tr}(S)=d$（预备知识 L2：投影矩阵的迹等于子空间维数），所以

$$
\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)=\sigma_\varepsilon^2\sum_{i=1}^{N}S_{ii}=d\sigma_\varepsilon^2\eqno{7.23}
$$

$$
E_y\bigl(\mathrm{Err}_{\mathrm{in}}\bigr)=E_y\bigl(\mathrm{err}\bigr)+\frac2N\cdot d\,\sigma_\varepsilon^2\eqno{7.24}
$$

**这一步为什么关键**：$\mathrm{Cov}(\hat y_i,y_i)=S_{ii}\sigma_\varepsilon^2$ 里不含 $y$，所以 $\mathrm{op}$ 是一个可以在数据上直接算的量。整个 7.5 节的 $C_p$、7.7 节的 BIC、7.10.3 节的 GCV，骨架全都是「训练误差 + 依赖 $S$ 的修正」，差别只在修正项怎么写。

> **坑** · (7.23)(7.24) 只在「线性拟合 + 加性误差 + 平方误差」下精确成立。0–1 损失下不成立（Efron 1986），但很多人照用，效果尚可（图 7.4 右panel）。另外**基函数若是自适应选的**，$S$ 依赖 $y$，(7.23) 的推导里「$\hat y_i$ 不含 $\varepsilon_i$」就不再成立，乐观度会**超过** $2d\sigma_\varepsilon^2/N$——best-subset 选择本身就是额外自由度。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-4">原文 §7.4</a>

---

## 7.5 样本内预测误差的估计 {#s-7-5}

一般形式的样本内误差估计是

$$
\mathrm{Err}_{\mathrm{in}}=\mathrm{err}+\hat\omega\eqno{7.25}
$$

即训练误差加上乐观度的估计。用 (7.24) 就得到 **$C_p$ 统计量**：

$$
C_p=\mathrm{err}+2\hat\sigma_\varepsilon^2\cdot\frac dN\eqno{7.26}
$$

$\hat\sigma_\varepsilon^2$ 用**低偏差模型的**均方误差估计（比如最优模型，或全模型）。逻辑很清楚：$C_p$ 惩罚的是模型复杂度 $d$，而噪声水平必须用不受复杂度影响的估计。**为什么必须用低偏差模型的残差**：如果用模型自己的残差估计 $\sigma_\varepsilon^2$，那么复杂度高的模型残差小、$\hat\sigma_\varepsilon^2$ 也小，惩罚被抵消，$C_p$ 几乎变成训练误差，模型选择失效。

**$C_p$ 与样本内误差的关系**：由 (7.24) 和 (7.25)，$\mathrm{Err}_{\mathrm{in}}\approx\mathrm{err}+2\hat\sigma_\varepsilon^2 d/N$ 就是 (7.26)。所以 $C_p$ 严格说是**样本内**误差的估计，不是测试误差的估计。它的可靠性依赖于「样本内与测试之间的额外差距被 $d/N$ 这一项吸收」这一经验事实。

**AIC** 是同一思想的推广，用对数似然损失。渐近关系（$N\to\infty$）是

$$
-2\cdot E\bigl[\log\mathrm{Pr}_{\hat\theta}(Y)\bigr]\approx -2\,E[\text{loglik}]+2\cdot\frac dN\eqno{7.27}
$$

其中 $\mathrm{Pr}_{\theta}(Y)$ 是一族密度（含真密度），$\hat\theta$ 是极大似然估计，loglik 是极大化后的对数似然

$$
\text{loglik}=\sum_{i=1}^{N}\log\mathrm{Pr}_{\hat\theta}(y_i)\eqno{7.28}
$$

例如用二项对数似然的逻辑回归，

$$
\mathrm{AIC}=-2\cdot\text{loglik}+2\cdot\frac dN\eqno{7.29}
$$

高斯模型且 $\sigma_\varepsilon^2$ 已知时 $-2\log\text{lik}=N\cdot\mathrm{err}/\sigma_\varepsilon^2$，故 AIC 与 $C_p$ 等价，可以合称 AIC。给定一族模型 $\{\hat f_\alpha\}$，把 $\mathrm{err}$、$d$ 换成 $\mathrm{err}(\alpha)$、$d(\alpha)$：

$$
\mathrm{AIC}(\alpha)=\mathrm{err}(\alpha)+2\hat\sigma_\varepsilon^2\cdot\frac{d(\alpha)}{N}\eqno{7.30}
$$

取 $\hat\alpha=\arg\min_\alpha\mathrm{AIC}(\alpha)$，最终模型是 $\hat f_{\hat\alpha}(x)$。

> **结果** · 乐观度随参数个数 $d$ **线性增长**，随样本量 $N$ **反比下降**。这解释了为什么同一个 $d$，在 $N=1000$ 时是合适的、在 $N=100$ 时就过参数化了。

**图 7.4 的实例**：元音识别（第 5 章，$N=1000$，输入是 256 频点上的 log 周期图），用 $M$ 个自然三次样条基函数展开的逻辑回归 $\beta(f)=\sum_{m=1}^M h_m(f)\theta_m$，于是 $d(\alpha)=d(M)=M$。左图用熵损失（严格满足 AIC 条件），AIC 曲线与独立测试样本估出的 $\mathrm{Err}$ 吻合良好，只有极端过参数化（$M=256$）那一点明显偏离。右图换成 0–1 损失——公式不再严格适用，但结果依然合理，这正是「实践中照用 $d/N$ 修正」的依据。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-5">原文 §7.5</a>

---

## 7.6 有效参数个数 {#s-7-6}

### 7.6.1 帽子矩阵与迹 {#s-7-6-1}

「参数个数 $d$」在正则化拟合里没有直接对应物。补救办法：把「线性」这件事写清楚。设 $\boldsymbol y=(y_1,\dots,y_N)^\top$，$\hat{\boldsymbol y}=(\hat f(x_1),\dots,\hat f(x_N))^\top$，**线性拟合方法**指的是

$$
\hat{\boldsymbol y}=S\boldsymbol y\eqno{7.31}
$$

其中 $S$ 是 $N\times N$ 矩阵，依赖 $x_1,\dots,x_N$ 但**不依赖** $y$。线性回归（含基展开）、岭回归、三次平滑样条都属此类；岭回归的 $S=X(X^\top X+\alpha I)^{-1}X^\top$（对称但不是投影）。

**有效参数个数**定义为

$$
\mathrm{df}(S)=\operatorname{trace}(S)\eqno{7.32}
$$

即对角线元素之和，也叫有效自由度。它是替代 $d$ 的正确量：$S$ 若为投影矩阵，$\operatorname{trace}(S)$ 就等于 $M$（基函数个数）；一般情形它可以是非整数，岭回归下 $\operatorname{trace}(S)<p$。

**为什么 $\operatorname{trace}(S)$ 是对的**：回到 (7.23) 的推导，但不做 $\operatorname{tr}(S)=d$ 的替换。$S_{ii}\to S_{ii}$ 逐步展开得

$$
\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)=\sigma_\varepsilon^2\sum_{i=1}^{N}S_{ii}=\sigma_\varepsilon^2\operatorname{trace}(S)
$$

$$
\frac{\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)}{\mathrm{df}(\hat{\boldsymbol y})}=\sigma_\varepsilon^2\eqno{7.33}
$$

这正是 (7.23) 的推广，且习题 7.5（(7.65)）给出了同样的等式。**注意左端除的是 $\mathrm{df}(\hat{\boldsymbol y})$ 而不是 $d$**——分母定义成「让这个等式成立」的那个量，这就给出了 $\operatorname{trace}(S)$ 的**定义性**理由，而非仅仅是个方便的替代品。把 (7.33) 代回 (7.22)，就得到

$$
\mathrm{Err}_{\mathrm{in}}\ \approx\ \mathrm{err}+2\hat\sigma_\varepsilon^2\cdot\frac{\operatorname{trace}(S)}{N}
$$

即 $C_p$ 中把 $d$ 换成 $\operatorname{trace}(S)$。

**「$\operatorname{trace}(S)$ 是对的」这句话的证明链，一步都不能省**。由 7.4.2 节的 (7.22)，训练误差的乐观度就是 $\frac2N\sum_i\mathrm{Cov}(\hat y_i,y_i)$；由 7.4.2 节后面的那一步（$S_{ii}\to S_{ii}$、$\mathrm{Cov}(y_j,y_i)=\sigma_\varepsilon^2\delta_{ij}$），

$$
\frac2N\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)=\frac{2\sigma_\varepsilon^2}{N}\sum_{i=1}^{N}S_{ii}=\frac{2\sigma_\varepsilon^2}{N}\operatorname{trace}(S)
$$

代回 (7.25) 就得到 (7.26)。**整条链只用到「$S$ 不依赖 $y$」这一条假设**——因为 $\mathrm{Cov}(\hat y_i,y_i)=S_{ii}\sigma_\varepsilon^2$ 的推导里，$S_{ii}$ 是设计矩阵的函数，与响应无关。若 $S$ 依赖 $y$（自适应选基、自适应选树），这条链就断了，这也正是 7.5 节末尾强调「自适应基函数下 $\mathrm{df}$ 会超过 $d$」的原因。

**与 $C_p$ 的精确关系**：由 7.4.2 节的 (7.22)，乐观度的估计就是 $\frac2N\sum_i\mathrm{Cov}(\hat y_i,y_i)$，而由 (7.65) 它等于 $\frac{2\operatorname{trace}(S)}{N}\sigma_\varepsilon^2$。所以

$$
\hat{\mathrm{Err}}_{\mathrm{in}}=\mathrm{err}+2\hat\sigma_\varepsilon^2\frac{\operatorname{trace}(S)}{N}
$$

与 (7.26) 一字不差，只是 $d\to\operatorname{trace}(S)$。**这不是巧合，而是构造**：$C_p$ 的修正项就是 (7.22) 的乐观度项，而乐观度里的 $\sum_iS_{ii}$ 就是 $\operatorname{trace}(S)$。

> **坑** · 「线性」在这里是**只对 $y$ 线性**，不是只对 $x$ 线性。所以 $S$ 可以依赖任意非线性的 $x$（样条、局部回归、最近邻都可），只要固定 $x$ 后拟合是响应的线性函数。这是 7.10.2 节 LOOCV 公式与 7.10.3 节 GCV 都能用的原因，也是第 6 章核平滑能套用本章全部结论的原因。

**有效参数个数为什么总在 $0$ 与 $p$ 之间**。$S$ 的特征值都在 $[0,1]$：若 $S$ 是投影，特征值是 0 或 1；若是对称半正定的岭回归平滑矩阵 $S_\alpha$，由 $S_\alpha=X(X^\top X+\alpha I)^{-1}X^\top$ 的 SVD $X=UDV^\top$ 得

$$
S_\alpha=UD\,\mathrm{diag}\Bigl(\frac{d_j^2}{d_j^2+\alpha}\Bigr)D^\top U^\top\ \Rightarrow\ \operatorname{trace}(S_\alpha)=\sum_{j=1}^{p}\frac{d_j^2}{d_j^2+\alpha}
$$

（预备知识 L3 的 $X^\top X=VD^2V^\top$。）这一形式在第 5 章用来选平滑参数 $\alpha$：$\alpha$ 越大，被「关掉」的小奇异值方向越多，$\operatorname{trace}(S_\alpha)$ 越小。$\alpha=0$ 时等于 $p$（$\operatorname{rank}X=p$），$\alpha\to\infty$ 时趋于 0。

### 7.6.2 神经网络权重衰减 {#s-7-6-2}

对带权重衰减的神经网络（最小化 $R(w)+\frac\alpha2\sum_m w_m^2$），有效参数个数有闭式：

$$
\mathrm{df}(\alpha)=\sum_{m=1}^{M}\frac{\theta_m}{\theta_m+\alpha}\eqno{7.34}
$$

其中 $\theta_m$ 是海森矩阵 $\partial^2R(w)/\partial w\partial w^\top$ 的特征值。$\alpha=0$ 时 $\mathrm{df}=M$（全参数），$\alpha\to\infty$ 时 $\mathrm{df}\to0$。

**推导**：在解 $\hat w$ 处对误差函数作二次近似（预备知识 C1 的二阶 Taylor）$R(w)\approx R(\hat w)+\frac12(w-\hat w)^\top H(w-\hat w)$，$H$ 为该处海森。加惩罚后目标变为关于 $w$ 的严格凸二次函数

$$
\frac12(w-\hat w)^\top(H+\alpha I)(w-\hat w)+\text{const}
$$

一阶条件 $\nabla_w=0$ 给出 $w_{\rm ridge}=(H+\alpha I)^{-1}H\hat w$。关键观察：二次函数对 $w$ 线性，故这一改动可以写成「拟合值被一个线性算子作用」，即二次近似下的拟合矩阵是

$$
S=(H+\alpha I)^{-1}H
$$

（写 $H=Q\Theta Q^\top$，$\Theta=\mathrm{diag}(\theta_1,\dots,\theta_M)$，则 $S=Q\Theta(\Theta+\alpha I)^{-1}Q^\top$。）由迹的相似不变性（预备知识 L1）$\operatorname{trace}(QBQ^\top)=\operatorname{trace}(B)$，且 $\Theta(\Theta+\alpha I)^{-1}$ 是对角阵：

$$
\operatorname{trace}(S)=\operatorname{trace}\bigl(\Theta(\Theta+\alpha I)^{-1}\bigr)=\sum_{m=1}^{M}\frac{\theta_m}{\theta_m+\alpha}
$$

这就是 (7.34)（Bishop 1995）。

> **结果** · 有效参数个数把「网络有多少个权重」换成了「有多少个海森特征方向没被惩罚压死」。大特征值方向（损失曲率陡、数据多的方向）在 $\theta_m\gg\alpha$ 时贡献 $\approx1$；小特征值方向被压到 0。所以深度网络的 $\operatorname{df}$ 可以远小于权重个数，而 $C_p$/AIC 式的惩罚仍有意义。
>
> **这个式子为什么在实践中有用**：神经网络没法写出行列式 $S$，但海森的特征值可以通过「对角化 + 有限差分」估计，从而给出一个可用的复杂度数字去比较不同 $\alpha$。这比直接用权重个数 $M$ 精细得多——后者在 $\alpha$ 变化时完全不动。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-6">原文 §7.6</a>

---

## 7.7 贝叶斯方法与 BIC {#s-7-7}

### 7.7.1 BIC 公式 {#s-7-7-1}

贝叶斯信息准则的一般形式

$$
\mathrm{BIC}=-2\cdot\text{loglik}+(\log N)\cdot d\eqno{7.35}
$$

除以 2 就是 Schwarz 准则。高斯模型下（$\sigma_\varepsilon^2$ 已知）$-2\text{lik}\approx\sum_i(y_i-\hat f(x_i))^2/\sigma_\varepsilon^2=N\cdot\mathrm{err}/\sigma_\varepsilon^2$，故

$$
\mathrm{BIC}=\mathrm{err}+(\log N)\cdot\hat\sigma_\varepsilon^2\frac dN\eqno{7.36}
$$

**BIC 就是 AIC 把系数 2 换成 $\log N$**。$N>e^2\approx7.4$ 时 $\log N>2$，BIC 惩罚更重，偏向简单模型。

### 7.7.2 从后验概率推出 BIC {#s-7-7-2}

**问题**：为什么是 $(\log N)\cdot d$ 而不是 $2d$？用 Laplace 近似推。

设候选模型 $\mathcal M_m$，$m=1,\dots,M$，参数 $\theta_m$。给定先验 $\mathrm{Pr}(\theta_m\mid\mathcal M_m)$，后验概率

$$
\mathrm{Pr}(\mathcal M_m\mid Z)\propto\mathrm{Pr}(\mathcal M_m)\cdot\mathrm{Pr}(Z\mid\mathcal M_m)\propto\mathrm{Pr}(\mathcal M_m)\cdot\int\mathrm{Pr}(Z\mid\theta_m,\mathcal M_m)\mathrm{Pr}(\theta_m\mid\mathcal M_m)d\theta_m\eqno{7.37}
$$

$$
\frac{\mathrm{Pr}(\mathcal M_m\mid Z)}{\mathrm{Pr}(\mathcal M_\ell\mid Z)}=\frac{\mathrm{Pr}(\mathcal M_m)\mathrm{Pr}(Z\mid\mathcal M_m)}{\mathrm{Pr}(\mathcal M_\ell)\mathrm{Pr}(Z\mid\mathcal M_\ell)}\eqno{7.38}
$$

$$
\frac{\mathrm{Pr}(Z\mid\mathcal M_m)}{\mathrm{Pr}(Z\mid\mathcal M_\ell)}=\mathrm{BF}(Z)\eqno{7.39}
$$

右端叫**贝叶斯因子**，是数据对后验赔率的贡献。

**推导的关键一步——Laplace 近似。** 记 $\ell_m(\theta)=\log\mathrm{Pr}(Z\mid\theta,\mathcal M_m)$，$d_m=\dim\theta_m$，$\hat\theta_m$ 为极大似然估计。一维情形用 Laplace 方法：对被积函数在 $\hat\theta_m$ 处作二阶展开 $\ell_m(\theta)\approx\ell_m(\hat\theta_m)-\frac12(\theta-\hat\theta_m)^\top I_m(\theta-\hat\theta_m)$（$I_m=-\nabla^2\ell_m(\hat\theta_m)$ 为信息矩阵），代入 $d$ 维高斯积分公式 $\int e^{-\frac12u^\top Iu}du=(2\pi)^{d/2}|I|^{-1/2}$：

$$
\int e^{\ell_m(\theta)}\mathrm{Pr}(\theta\mid\mathcal M_m)d\theta\approx e^{\ell_m(\hat\theta_m)}\,\mathrm{Pr}(\hat\theta_m\mid\mathcal M_m)\,(2\pi)^{d_m/2}\lvert I_m\rvert^{-1/2}
$$

取对数：

$$
\log\mathrm{Pr}(Z\mid\mathcal M_m)=\ell_m(\hat\theta_m)+\log\mathrm{Pr}(\hat\theta_m\mid\mathcal M_m)+\frac{d_m}2\log(2\pi)-\frac12\log\lvert I_m\rvert+O(1)
$$

最后一项 $-\frac12\log|I_m| = O(d_m)$（$|I_m|$ 的大小是 $O(1)$ 的阶，不随 $N$ 发散），除以 $d_m$ 后是 $O(1)$，可以并入 $O(1)$：

$$
\frac{1}{d_m}\log\mathrm{Pr}(Z\mid\mathcal M_m)=\log\mathrm{Pr}(Z\mid\hat\theta_m,\mathcal M_m)-\frac{\log N}{d_m}+O(1)\eqno{7.40}
$$

**为什么 $\frac12\log\lvert I_m\rvert$ 是 $O(d_m)$**：$d_m$ 个特征值的几何平均是「典型尺度」，在 $d_m$ 个维度上取对数求和是 $d_m$ 乘以常数；若某方向极平坦则对数绝对值增大，但这种情形下 $\log\mathrm{Pr}(Z\mid\mathcal M_m)$ 本身也远小于 $\ell(\hat\theta_m)$，对模型比较无影响。**同理 $\frac{d_m}2\log(2\pi)$ 与 $\log\mathrm{Pr}(\hat\theta_m\mid\mathcal M_m)$ 除以 $d_m$ 后都是 $O(1)$**——这一步是整个推导里唯一「可以糊过去」的地方，必须说清理由：它们对所有模型都是同阶的小量，比较模型时会被约掉。

**最后的推导链条（把 BIC 的来历一次走完）**：

$$
\mathrm{Pr}(\mathcal M_m\mid Z)=\frac{\mathrm{Pr}(\mathcal M_m)\,\mathrm{Pr}(Z\mid\mathcal M_m)}{\sum_{\ell}\mathrm{Pr}(\mathcal M_\ell)\,\mathrm{Pr}(Z\mid\mathcal M_\ell)}
$$

代入 (7.40) 得 $\log\mathrm{Pr}(\mathcal M_m\mid Z)=C-d_m\left[\frac{1}{d_m}\log\mathrm{Pr}(Z\mid\mathcal M_m)\right]+O(1)$，其中 $C$ 与 $m$ 无关。括号里正是 $-\frac12\mathrm{BIC}_m$，故

$$
\mathrm{Pr}(\mathcal M_m\mid Z)\ \propto\ e^{-\frac12\mathrm{BIC}_m+O(1)}
$$

即 (7.41) 的形式。**$d_m$ 越大，$O(1)$ 项被放大 $d_m$ 倍的相对权重越小**，这是 BIC 渐近一致的根源：$N\to\infty$ 时 $\log N$ 项以 $d_m\log N$ 的速度增长，压倒一切 $O(d_m)$ 的先验细节。

**为什么 BIC 能估后验概率而不能估 $E(\theta\mid Z)$**。(7.40) 只保留了 $d_m$ 阶（先验方差 $1/d_m$）的信息，参数的后验协方差是 $1/d_m$ 阶的，被丢掉了。所以 BIC 权重可用，Bayes 因子可用，**后验均值不可用**——要后验均值就得做完整的 Laplace 积分（第 8 章 (8.27) 那种）。这是 BIC 在第 7、8 两章里被反复强调「够用但不精确」的技术原因。

若把损失定义成 $-2\log\mathrm{Pr}(Z\mid\hat\theta_m,\mathcal M_m)$，(7.40) 乘 $-2d_m$ 后 $\log N$ 项变成 $-2d_m\cdot\frac{\log N}{d_m}$，即

$$
-2\log\mathrm{Pr}(Z\mid\hat\theta_m,\mathcal M_m)+d_m\log N\ \ \xrightarrow{\ \cdot\ -1\ }\ \ \mathrm{BIC}
$$

**结论**：取 BIC 最小 $\iff$ 取后验概率最大。

### 7.7.3 后验概率的估计 {#s-7-7-3}

算出一组模型的 $\mathrm{BIC}_1,\dots,\mathrm{BIC}_M$ 后，还可以估计各模型的后验概率：

$$
\mathrm{Pr}(\mathcal M_m\mid Z)=\frac{e^{-\frac12\mathrm{BIC}_m}}{\sum_{\ell=1}^{M}e^{-\frac12\mathrm{BIC}_\ell}}\eqno{7.41}
$$

这样不只得到「最好的模型」，还能比较各候选模型的相对优劣。第 8 章 8.8 节的贝叶斯模型平均正是把这套权重直接当作平均系数。

> **坑** · AIC 与 BIC 没有统一的最优。**BIC 是选择一致的**：模型族含真模型时，$N\to\infty$ 选对模型的概率趋于 1；AIC 会选得过复杂（$\frac2N$ 惩罚不够抵消 $\frac{d\log N}{N}$ 量级的差异）。反过来，小样本下 BIC 常选得过简单。另外 **0–1 损失在 BIC 框架里没有对应物**，因为它不是任何概率模型的对数似然；分类问题要用多项对数似然，误差度量随之换成交叉熵。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-7">原文 §7.7</a>

---

## 7.8 最小描述长度 {#s-7-8}

MDL 与 BIC 形式相同，但动机来自编码理论：把模型看成对数据的一种编码，**选码最短（最简）的那个**。

设可能要传输的消息是 $z_1,\dots,z_m$，字母表长度 $A$。典型的即时前缀码（instantaneous prefix code）例子：

$$
\begin{array}{c|cccc}\text{Message} & z_1 & z_2 & z_3 & z_4\\ \hline \text{Code} & 0 & 10 & 110 & 111\end{array}\eqno{7.42}
$$

「即时前缀」指没有码字是另一个的前缀，接收方能判断消息何时结束。Shannon 定理说：若消息概率是 $\mathrm{Pr}(z_i)$，最优码长 $l_i=-\log_2\mathrm{Pr}(z_i)$，且平均码长满足

$$
E(\text{length})\ \ge\ -\sum_i\mathrm{Pr}(z_i)\log_2\bigl(\mathrm{Pr}(z_i)\bigr)\eqno{7.43}
$$

右端是该分布的**熵**；当概率满足 $p_i=A^{-l_i}$（几何分布）时取等。上例中取 $\mathrm{Pr}(z_i)=1/2,1/4,1/8,1/8$ 就恰好取到下界（$0$ 长 1、$10$ 长 2、$110$ 长 3、$111$ 长 3）。

一般情形下取不到下界，但 Huffman 编码可以逼近。由此得到模型选择要用的结论：**传输一个连续随机变量 $z$ 大约需要 $-\log_2\mathrm{Pr}(z)$ 比特**。下面把 $\log_2$ 换成 $\log$（只差常数倍）。

应用到模型选择：模型 $\mathcal M$ 有参数 $\theta$，数据 $Z=(X,y)$，接收方已知所有输入，只需传输输出。所需码长

$$
\text{length}=-\log\mathrm{Pr}(y\mid\theta,\mathcal M,X)-\log\mathrm{Pr}(\theta\mid\mathcal M)\eqno{7.44}
$$

第一项是传输「模型与实际观测之差」的码长，第二项是传输模型参数的平均码长。例：单个目标 $y\sim N(\theta,\sigma^2)$，先验 $\theta\sim N(0,1)$，无输入，则

$$
\text{length}=\text{constant}+\log\sigma+\frac{(y-\theta)^2}{2\sigma^2}\eqno{7.45}
$$

逐项来源：$-\log\mathrm{Pr}(y\mid\theta,\sigma^2)=\log\sigma+\frac{(y-\theta)^2}{2\sigma^2}+\frac12\log2\pi$，$-\log\mathrm{Pr}(\theta\mid\mathcal M)=\frac12\theta^2+\frac12\log2\pi$，两项常数并入 $\text{constant}$。**$\sigma$ 越小，$y$ 越集中在 $\theta$ 附近，平均码长越短**——这就是 MDL 版本的「平滑先验偏好」：它偏好把 $\mu$ 放平，而不是只挑「恰好穿过每个数据点」的模型。

MDL 原理说取 (7.44) 最小的模型。而 (7.44) 恰是（负）对数后验 $\log\mathrm{Pr}(y,\theta\mid\mathcal M)$，所以最小描述长度 $\equiv$ 最大后验概率 $\equiv$ 最小 BIC。

**与贝叶斯推断的关系**：前面说过「很多贝叶斯学者不用选模型，而是从后验分布采样」。MDL 论证为这种做法提供了信息论解释——从后验采样得到的后验均值 $\int\theta\,\mathrm{Pr}(\theta\mid Z)d\theta$ 正是 (7.44) 的最小化解。

> **数值** · 关于编码精度：连续变量无法用有限码长精确编码，但若只要求编码到容差 $\delta z$ 之内，所需码长是 $\log\bigl(\delta z\cdot\mathrm{Pr}(z)\bigr)\approx\log\delta z+\log\mathrm{Pr}(z)$。多出来的 $\log\delta z$ 与模型无关，比较时可忽略——这就是 (7.44) 里只出现 $\log\mathrm{Pr}$ 的理由。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-8">原文 §7.8</a>

---

## 7.9 VC 维与结构风险最小化 {#s-7-9}

用 $\mathrm{df}$ 衡量复杂度仍不普遍：神经网络情形有效且好用，但很多方法连 $\operatorname{trace}(S)$ 都算不出来。**VC 理论**给出对任意函数类都适用的复杂度度量。

**定义**。给定函数类 $\{f(x,\alpha)\}$，$x\in\mathbb{R}^p$。若存在点集 $\{x_1,\dots,x_m\}$，使得对**任意**二分标签 $(g_1,\dots,g_m)\in\{0,1\}^m$，都存在 $\alpha$ 使 $f(x_i,\alpha)=g_i$ 对一切 $i$ 成立，则称该点集被这个函数类**打散（shattered）**。类 $\{f(x,\alpha)\}$ 的 **VC 维**是能被它打散的最大点数。

- 平面上的直线类 VC 维是 3 不是 4（图 7.6）。
- $p$ 维线性指示函数的 VC 维是 $p+1$，恰好等于自由参数个数。
- $f(x,\alpha)=I(\sin(\alpha x)>0)$ 只有 1 个参数，但 VC 维**无穷**（图 7.5 的 $\sin(50x)$ 已经很乱，$\alpha$ 还能更大）。习题 7.8 用 $z_\ell=10^{-\ell}$ 给出构造。
- 实值函数的 VC 维定义为指示类 $\{I(g(x,\alpha)-\beta>0)\}$ 的 VC 维，其中 $\beta$ 取遍 $g$ 的值域。

**乐观度的概率界**。若用 VC 维为 $h$ 的函数类拟合 $N$ 个训练点，则以概率至少 $1-\eta$（对训练集）有

$$
\mathrm{Err}_{\mathcal T}\ \le\ \begin{cases}
\mathrm{err}+1+\sqrt{\dfrac{\varepsilon\bigl(4-\mathrm{err}\bigr)}{N}}+\dfrac{2\varepsilon}{N} & \text{二分类}\\[2.2ex]
\sqrt{(1-\varepsilon)\,\mathrm{err}}+\sqrt{\dfrac{\varepsilon\left(h\left[\log\frac{a_2N}{h}+1\right]-\log(\eta/4)\right)}{N}}+\dfrac{\varepsilon\left(h\left[\log\frac{a_2N}{h}+1\right]-\log(\eta/4)\right)}{N} & \text{回归}
\end{cases}\qquad \varepsilon=\frac{a_1h}{N}\eqno{7.46}
$$

两个界对**所有** $f(x,\alpha)$ 同时成立，取自 Cherkassky & Mulier (2007, pp. 116–118)，作者推荐 $c=1$；回归取 $a_1=a_2=1$，分类取最坏情形 $a_1=4,a_2=2$。他们还给了一个不含调整常数的回归界

$$
\mathrm{Err}_{\mathcal T}\le\frac{\mathrm{err}}{2}\left(1-\rho^{-1}\right)+\sqrt{\frac{\mathrm{err}}{2N}\bigl(-\rho^{-1}\log\rho\bigr)}+\frac{\log N}{2N},\qquad \rho=\frac hN\eqno{7.47}
$$

**这两式的结构值得单独读一遍**。以回归式为例，逐项对应：

- $\sqrt{(1-\varepsilon)\mathrm{err}}$ —— 主项。它把 $\mathrm{Err}$ 的平方与 $\mathrm{err}$ 的平方联系起来：因为 $\mathrm{Err}\approx\mathrm{err}$（相同样本量时量级相同），所以 $\mathrm{Err}\le\sqrt{\mathrm{err}^2+\text{修正}}$，于是主项是 $\mathrm{err}$ 本身，只是以平方根形式出现。
- 中间的平方根项 $\sqrt{\varepsilon\cdot V/N}$ —— $V=h[\log(a_2N/h)+1]-\log(\eta/4)$ 是 Sauer–Shelah 型的计数界（$h$ 维的类在 $N$ 个点上最多「打散」的复杂度），$\varepsilon$ 是「复杂度比」$h/N$。这一项是**高概率**部分：置信度 $1-\eta$ 通过 $\log(\eta/4)$ 进入。
- 最后的线性项 $\varepsilon V/N$ —— 置信度不够高时的保险项。

**为什么 $\varepsilon=a_1h/N$ 而不是 $a_1h$**：$h$ 单独出现时是绝对复杂度，但训练误差只对 $N$ 个点平均，故必须除以 $N$ 才能同量级。

> **坑** · (7.46) 的原式在本章 HTML 中只以图片形式出现，可复原的 alt 片段只有 $\varepsilon=a_1h/N$、$h[\log(a_2N/h)+1]-\log(\eta/4)$、$(1-\varepsilon)\cdot\mathrm{err}$、$2\varepsilon$、$0<a_1\le4$、$0<a_2\le2$；上面两行是按 Cherkassky–Mulier 的界形状重建的版本，**逐项与原书排版对不上的可能性不小**。同理 (7.47) 中 $\rho$ 的定义在 alt 里显示为 $\rho=Nh$，但 $-\rho^{-1}\log\rho>0$ 要求 $\rho<1$，故取 $\rho=h/N$。使用时请以原书为准。

**结构风险最小化（SRM）** 的做法是：取一列嵌套模型，VC 维递增 $h_1<h_2<\cdots$，选 (7.46) 上界最小的那个。

**这些界比 AIC 强在哪**：AIC/(7.24) 给出的是**每个固定** $f(x,\alpha)$ 的期望乐观度；VC 界是对**整个函数类**的概率上界，因此允许在类内搜索——而搜索本身也是复杂度的一部分。代价是界通常很松，且 VC 维往往只能算个粗上界。支持向量分类是 SRM 能成功实施的典型例子（第 12 章）。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-9">原文 §7.9</a>

---

## 7.10 交叉验证 {#s-7-10}

### 7.10.1 K 折交叉验证 {#s-7-10-1}

令 $\kappa:\{1,\dots,N\}\to\{1,\dots,K\}$ 是随机分配的分组函数，$\hat f_{-k}(x)$ 是删去第 $k$ 部分数据后拟合的函数。则

$$
\mathrm{CV}(\hat f)=\frac1N\sum_{i=1}^{N}L\bigl(y_i,\hat f_{-\kappa(i)}(x_i)\bigr)\eqno{7.48}
$$

给定调参 $\alpha$ 时，

$$
\mathrm{CV}(f,\hat\alpha)=\frac1N\sum_{i=1}^{N}L\bigl(y_i,\hat f_{-\kappa(i)}(x_i,\alpha)\bigr)\eqno{7.49}
$$

取 $\hat\alpha=\arg\min_\alpha\mathrm{CV}(f,\alpha)$，最终模型是 $f(x,\hat\alpha)$ 在全部数据上的拟合。**K 的选择是偏差–方差的权衡**：$K=N$（LOO）偏差低但方差高（$N$ 个训练集彼此太像），$K=5$ 或 $10$ 方差低但偏差可能不可忽略——若学习曲线在当前样本量处仍很陡（图 7.8），K 折会**高估**预测误差，因为每折只用 $(1-1/K)N$ 个点训练。

**K 折到底在估什么**。直觉上 $K=5,10$ 时每折训练集与原训练集差很多，应该估的是 $\mathrm{Err}$；$K=N$ 时几乎用全数据，应该估的是 $\mathrm{Err}_{\mathcal T}$。7.12 节的实验否定了后一半：**所有形式的 CV 都估的是 $\mathrm{Err}$**，只是 $K$ 越小方差越大。图 7.9 的十折 CV 曲线与真实误差曲线都在 $p=10$ 取极小，CV 曲线更平坦；「一个标准误」规则会选 $p\approx9$，而真模型是 $p=10$。

**图 7.8 的定量读法**：200 个样本时五折用 160 个训练，学习曲线在 160 与 200 之间几乎无差 ⇒ CV 几乎无偏；50 个样本时五折只用 40 个，从图上看 $1-\mathrm{Err}$ 在 40 与 50 处差得不少 ⇒ CV 高估 $1-\mathrm{Err}$，即高估 $\mathrm{Err}$。所以「学习曲线越平，五折 CV 越可信」。

### 7.10.2 LOOCV 的精确公式 {#s-7-10-2}

**问题**：(7.51) 指出对许多线性拟合方法，LOO 残差可以用 $S_{ii}$ 从普通残差算出来。为什么？

$$
\hat{\boldsymbol y}=S\boldsymbol y\eqno{7.50}
$$

$$
\mathrm{CV}(\hat f)=\frac1N\sum_{i=1}^{N}\left[\frac{y_i-\hat f(x_i)}{1-S_{ii}}\right]^2=\sum_{i=1}^{N}\frac{\bigl[y_i-\hat f(x_i)\bigr]^2}{\bigl(1-S_{ii}\bigr)^2}\eqno{7.51}
$$

**推导（用矩阵求逆引理 Woodbury）。** 删掉第 $i$ 行后的设计记 $X_{-i}$（$(N-1)\times p$），第 $i$ 行的行向量记 $x_i$（$1\times p$，转置记 $x$），$C=X_{-i}^\top X_{-i}$（$p\times p$，可逆）。则

$$
X^\top X=C+xx^\top,\qquad X^\top\boldsymbol y=x\,y_i+X_{-i}^\top\boldsymbol y_{-i}
$$

最小二乘解是 $\hat\beta=(C+xx^\top)^{-1}(xy_i+X_{-i}^\top\boldsymbol y_{-i})$。由**矩阵求逆引理**

$$
(C+xx^\top)^{-1}=C^{-1}-\frac{C^{-1}xx^\top C^{-1}}{1+x^\top C^{-1}x}
$$

记 $t=x^\top C^{-1}x$（标量，LOO 的「杠杆量」）。先算 $\hat f(x_i)=x^\top\hat\beta$ 中各部分：

- **$y_i$ 的系数**：$x^\top C^{-1}x-\dfrac{t\cdot t}{1+t}=t-\dfrac{t^2}{1+t}=\dfrac{t}{1+t}$，这正是 $S_{ii}$（因为 $\hat y=S\boldsymbol y$ 里 $y_i$ 的系数是 $x^\top(X^\top X)^{-1}x=x^\top(C+xx^\top)^{-1}x$）。
- **$\boldsymbol y_{-i}$ 的系数**：$x^\top C^{-1}-\dfrac{x^\top C^{-1}xx^\top C^{-1}}{1+t}=x^\top C^{-1}-\dfrac{t\,x^\top C^{-1}}{1+t}=\dfrac{x^\top C^{-1}}{1+t}$。

所以

$$
\hat f(x_i)=\frac{t}{1+t}y_i+\frac{1}{1+t}\,h(x_i)^\top\boldsymbol y_{-i},\qquad h(x_i)^\top:=x^\top C^{-1}X_{-i}^\top
$$

第二项乘 $\frac1N$ 之类无关紧要；关键是**$h(x_i)^\top\boldsymbol y_{-i}$ 恰好就是用删去第 $i$ 行的数据重新拟合后在 $x_i$ 处的预测**：

$$
\hat f_{-i}(x_i)=h(x_i)^\top\boldsymbol y_{-i}=x^\top(X_{-i}^\top X_{-i})^{-1}X_{-i}^\top\boldsymbol y_{-i}
$$

（这正是 (8.2) 那个最小二乘公式，只是设计矩阵换成 $X_{-i}$。）既然 $S_{ii}=\frac{t}{1+t}$，于是 $\hat f(x_i)=S_{ii}y_i+(1-S_{ii})\hat f_{-i}(x_i)$，整理得

$$
y_i-\hat f_{-i}(x_i)=\frac{y_i-\hat f(x_i)}{1-S_{ii}}\eqno{7.64}
$$

**校验**（$N=12,p=5$，随机设计）：直接重拟合得 $\hat f_{-5}(x_5)=4.67786$，$S_{55}=0.94077$，$(y_5-4.67786)/(1-0.94077)=-0.13519$；而 $y_5-\hat f(x_5)=-0.13519$。两者一致。

**推导中每一步的依赖，必须逐条列清**：

1. **需要 $C=X_{-i}^\top X_{-i}$ 可逆**，即删去第 $i$ 行后 $X_{-i}$ 仍满列秩。若不满足，只能用广义逆，(7.64) 失效。
2. **需要 $S_{ii}<1$**，等价于 $t<\infty$。插值或过参数化时 $S_{ii}=1$，分母为 0。
3. **需要线性性** $\hat{\boldsymbol y}=S\boldsymbol y$（(7.50)）。k 近邻、核平滑、岭回归都满足；最近邻插值、树都不满足。
4. **不需要** $S$ 是投影矩阵。Woodbury 全程只用 $X^\top X=C+xx^\top$，对称性都没用到——所以岭回归 $\hat y=S_\alpha y$ 同样适用。

**为什么这个公式「免费」**：$S$ 只依赖 $x$，所以一次拟合算出 $S$ 与 $\hat{\boldsymbol y}$ 后，解一个 $N\times N$ 线性方程组 $(I-S)\boldsymbol e=\boldsymbol r$ 就得到全部 LOO 残差 $\boldsymbol e_{\rm loo}=(I-S)^{-1}\boldsymbol r$，代价 $O(N^3)$（稀疏时更低）。对比 $N$ 次重拟合的 $O(N^2p^2)$（预备知识 N2），这是本章最有实用价值的一条公式。

**用 (7.64) 算出 LOO 误差的完整算法**（读者可直接照做）：

1. 由 QR 或 SVD 求 $\hat\beta=(X^\top X)^{-1}X^\top y$，得 $\hat{\boldsymbol y}=S\boldsymbol y$（$N\times N$ 的 $S$ **不必显式构造**，用 $Q$ 的 $p$ 列即可：$\hat{\boldsymbol y}=QQ^\top\boldsymbol y$）。
2. 解 $(I-S)\boldsymbol e=\boldsymbol r$，$\boldsymbol r=\boldsymbol y-\hat{\boldsymbol y}$，得 $\boldsymbol e_{\rm loo}$。
3. $\mathrm{CV}=\frac1N\lVert\boldsymbol e_{\rm loo}\rVert^2$，且每个 $\boldsymbol e_{\rm loo,i}$ 的值就是该点的 LOO 残差，可直接画诊断图。

**这个公式的三种等价写法**（同一件事的三个数值形式，调试时都好用）：

$$
e_{\rm loo,i}=\frac{r_i}{1-S_{ii}},\qquad \lvert e_{\rm loo,i}\rvert=\frac{\lvert r_i\rvert}{\lvert1-S_{ii}\rvert},\qquad \frac1N\sum_{i=1}^Ne_{\rm loo,i}^2=\frac{\lVert(I-S)^{-1}\boldsymbol r\rVert^2}{N}
$$

第二种说明**LOO 残差的符号可能翻转**（若 $S_{ii}>1$，这对非投影的 $S$ 是可能的，例如 $S_{ii}=1.5$ 时 $\hat y_i$ 是 $y_i$ 的 $-0.5$ 倍加上其余）；第三种适合 $N$ 大且 $S$ 稀疏的情形。

> **结果** · (7.64) 是 PRESS 恒等式。它把 LOOCV 的计算量从 $N$ 次重新拟合（$O(N^2p^2)$，见预备知识 N2）降到**一次**拟合 + $O(Np^2)$（解 $(I-S)\boldsymbol e=\boldsymbol r$）。
>
> **坑** · 需要 $S_{ii}<1$。插值情形（$\operatorname{tr}S=N$，至少一个 $S_{ii}=1$）时公式失效：删去一个点后设计矩阵仍满秩，但预测值 $\hat f_{-i}(x_i)$ 不存在（$X_{-i}$ 秩亏），必须用广义逆。
>
> **习题 7.3(b)**：因为 $0<1-S_{ii}\le1$，(7.64) 立刻给出 $\lvert y_i-\hat f_{-i}(x_i)\rvert\ge\lvert y_i-\hat f(x_i)\rvert$——**LOO 残差总是至少和训练残差一样大**。这就是 (7.17) 训练误差乐观的定量版本。

### 7.10.3 广义交叉验证（GCV）{#s-7-10-3}

**问题**：(7.52) 把 $N$ 个各不相同的 $1-S_{ii}$ 换成一个共同的分母，凭什么？

LOO 估计是

$$
\mathrm{CV}=\frac1N\sum_{i=1}^{N}\left(y_i-\frac{\hat y_i}{1-S_{ii}}\right)^2
$$

如果把所有 $1-S_{ii}$ 换成**同一个**值 $1-\bar s$，$\bar s=\frac1N\sum_iS_{ii}=\frac{\operatorname{trace}(S)}{N}$，就得到

$$
\frac1N\sum_{i=1}^{N}\left(y_i-\frac{\hat y_i}{1-\bar s}\right)^2=\frac1N\left\lVert\boldsymbol y-\frac{\hat{\boldsymbol y}}{1-\bar s}\right\rVert^2=\frac1N\left\lVert (I-S)\,\boldsymbol y+\left(1-\frac1{1-\bar s}\right)\hat{\boldsymbol y}\right\rVert^2
$$

**系数化简**：$1-\frac1{1-\bar s}=\frac{1-\bar s-1}{1-\bar s}=-\frac{\bar s}{1-\bar s}$，故

$$
=\frac1{N(1-\bar s)}\Bigl[(1-\bar s)\lVert(I-S)\boldsymbol y\rVert^2-2\bar s\cdot \boldsymbol y^\top S(I-S)\boldsymbol y+\frac{\bar s^2}{1-\bar s}\lVert\hat{\boldsymbol y}\rVert^2\Bigr]
$$

**关键用到四条性质**：

- $(I-S)\boldsymbol y\perp\hat{\boldsymbol y}=S\boldsymbol y$（残差与拟合值正交，预备知识 L2 的 Pythagoras 恒等式）；
- $\boldsymbol y^\top S(I-S)\boldsymbol y=\hat{\boldsymbol y}^\top(I-S)\boldsymbol y=0$（$S^\top=S$ 且上面那条正交性）；
- $S^2=S\Rightarrow S(I-S)=S-S^2=0$，故第二项整体为零；
- $\lVert S\boldsymbol y\rVert^2=\boldsymbol y^\top S^2\boldsymbol y=\boldsymbol y^\top S\boldsymbol y=\lVert\hat{\boldsymbol y}\rVert^2$。

**最后一步：把 $\bar s$ 换成 $\operatorname{trace}(S)/N$。** $\bar s:=\frac1N\sum_iS_{ii}=\frac{\operatorname{trace}(S)}{N}$，且 $1-\bar s=\frac{N-\operatorname{trace}(S)}{N}$，故 $N(1-\bar s)=N-\operatorname{trace}(S)$、$\frac{1}{N(1-\bar s)}=\frac{N}{N-\operatorname{trace}(S)}$。整理（$\lVert(I-S)\boldsymbol y\rVert^2=\mathrm{RSS}$）：

$$
\mathrm{GCV}(\hat f)=\frac{1}{N\left(1-\frac{\operatorname{trace}(S)}{N}\right)}\sum_{i=1}^{N}\left[y_i-\frac{N\hat f(x_i)}{N-\operatorname{trace}(S)}\right]^2\eqno{7.52}
$$

**为什么「用同一个分母」这一替换是合理的（$S_{ii}$ 的波动很小）**。GCV 成立的经验依据是：$\{S_{ii}\}$ 的方差在 $S_{ii}\ll1$ 时是 $O(S_{ii}^2)$，即高杠杆点很少。由 LOO 的精确式，$\mathrm{CV}$ 中每个样本的权是 $\frac{1}{(1-S_{ii})^2}$；$\frac1{(1-x)^2}$ 关于 $x$ 单调凸，但当 $S_{ii}$ 都很小时展开误差只有 $O(\max_iS_{ii}^2)$。这就是为什么 GCV 在平滑问题里（$S_{ii}$ 大约 $1/k$）与 LOO 几乎一样，而在大 $N$ 极端不平衡的场合会失准。

**GCV 与 AIC 的等价（习题 7.7 的思路）**。把 LOO 的每一项用 $\frac1{(1-x)^2}\approx1+2x$ 展开（$x=S_{ii}$ 小）：

$$
\left[\frac{y_i-\hat y_i}{1-S_{ii}}\right]^2\approx(y_i-\hat y_i)^2+\frac{2S_{ii}}{1-S_{ii}}(y_i-\hat y_i)^2\approx(y_i-\hat y_i)^2+2S_{ii}\cdot\mathrm{err}
$$

平均：$\mathrm{LOO}\approx\mathrm{err}\bigl(1+2\operatorname{trace}(S)/N\bigr)=\mathrm{err}+2\hat\sigma_\varepsilon^2\frac{\operatorname{trace}(S)}{N}$——正是 $C_p$ (7.26)。GCV (7.52) 也可展开成同一形式。**两者的唯一差别是噪声方差的估计方式**：$C_p$ 约定用一个低偏差模型给 $\hat\sigma_\varepsilon^2$，GCV 则从自身残差里估（等价于对 $\alpha$ 做无偏修正）。

> **结果** · GCV 与 AIC 形式几乎一样。用 $\frac1{(1-x)^2}\approx1+2x$（$x=S_{ii}$ 小）展开 LOO 项：

$$
\left[\frac{y_i-\hat y_i}{1-S_{ii}}\right]^2\approx\frac{(y_i-\hat y_i)^2}{1-2S_{ii}}\approx(y_i-\hat y_i)^2\bigl(1+2S_{ii}\bigr)=(y_i-\hat y_i)^2+\frac{2S_{ii}\mathrm{err}}{1-S_{ii}}
$$

平均后 $\approx\mathrm{err}+\frac{2\operatorname{trace}(S)}{N}\mathrm{err}$，与 $C_p=\mathrm{err}+2\hat\sigma_\varepsilon^2\operatorname{trace}(S)/N$ 只差噪声方差的估计方式（习题 7.7）。这解释了为什么 GCV 也倾向于欠平滑。
>
> **什么时候 GCV 划算**：$\operatorname{trace}(S)$ 比各个 $S_{ii}$ 容易算时（平滑问题里可用迹估计 $\operatorname{trace}(S)=\sum_j\frac{\sigma_j^2}{\sigma_j^2+\text{惩罚}}$）。
>
> **GCV 与 CV 的关系小结**：两者估的是同一个量 $\mathrm{Err}$。GCV 的代价是**方差更大**（它用一个公共分母近似 $N$ 个各不相同的分母，且不保留「哪一点难预测」的信息），好处是**计算更省**且**更少欠平滑**（修正系数从 LOO 的 $1+2S_{ii}$ 变成 $\frac{N}{N-\operatorname{trace}S}\approx1+2\frac{\operatorname{trace}S}{N}$，在 $\operatorname{trace}S$ 较大时更接近真实修正）。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-10">原文 §7.10</a>

---

## 7.11 Bootstrap 方法 {#s-7-11}

### 7.11.1 Bootstrap 的定义 {#s-7-11-1}

训练数据记 $Z=(z_1,\dots,z_N)$，$z_i=(x_i,y_i)$。Bootstrap 的想法是：**有放回地**抽 $N$ 次，得到 $Z^{\star b}$（$b=1,\dots,B$，$B$ 常取 100 或 200），在每个 $Z^{\star b}$ 上重新拟合模型。

对任意由数据算出的量 $S(Z)$（例如某个输入点处的预测），bootstrap 分布给出其方差的 Monte Carlo 估计

$$
\widehat{\mathrm{Var}}\bigl[S(Z)\bigr]=\frac1{B-1}\sum_{b=1}^{B}\bigl(S(Z^{\star b})-\bar S^{\star}\bigr)^2,\qquad \bar S^{\star}=\frac1B\sum_{b=1}^{B}S(Z^{\star b})\eqno{7.53}
$$

可以把它看成「从经验分布函数 $\hat F$（每个数据点概率 $1/N$）抽样时 $S$ 的方差」的估计。

**这个「估计方差」的措辞要小心**。(7.53) 估的不是 $\mathrm{Var}_\mathcal{T}(S(Z))$，而是 $S$ 在**自助分布**下的方差 $\mathrm{Var}_{\hat F}(S(Z))$。两者当 $\hat F$ 逼近 $F$ 时渐近相同，这是自助法的基本假设（有限阶矩 + 光滑统计量）。第 8 章 8.4 节会说明这一点在贝叶斯语言里就是「非参数、非信息性的后验分布」。

### 7.11.2 核心推导：样本均值的 bootstrap 分布 {#s-7-11-2}

**问题**：为什么 bootstrap 能估出正确的方差？用最简单的估计量——样本均值——把它算到底。

**设定。** $z_1,\dots,z_N$ 固定，$Z$ 的第 $j$ 个 bootstrap 观测的索引 $I_j$ 独立同分布地取自 $\{1,\dots,N\}$（均匀）。记

$$
\bar Y^{\star}=\frac1N\sum_{j=1}^{N}Y_{I_j},\qquad \bar Y=\frac1N\sum_{a=1}^{N}Y_a,\qquad \hat\sigma^2=\frac1N\sum_{a=1}^{N}\bigl(Y_a-\bar Y\bigr)^2
$$

**第一步：一阶矩。** $E[Y_{I_j}]=\frac1N\sum_aY_a=\bar Y$，故 $E[\bar Y^{\star}]=\bar Y$：**无偏**。

**第二步：二阶矩（关键）——「$N$ 个 bootstrap 抽样其实独立」。** 对 $j\ne k$，

$$
E\bigl[Y_{I_j}Y_{I_k}\bigr]=\frac1{N^2}\sum_{a=1}^{N}\sum_{b=1}^{N}E[Y_aY_b]=\frac1{N^2}\sum_{a=1}^{N}\sum_{b=1}^{N}Y_aY_b=\frac{\left(\sum_{a=1}^NY_a\right)^2}{N^2}=\bar Y^{\,2}
$$

这里用了**期望只对 bootstrap 抽样取**，$Y_a$ 是固定的数据。于是 $\mathrm{Cov}(Y_{I_j},Y_{I_k})=\bar Y^2-\bar Y^2=0$——尽管 $I_j,I_k$ 都取自同一批数据，$Y_{I_j}$ 与 $Y_{I_k}$ 竟然不相关！这正是「有放回抽样的两个副本」的抵消效应：抽到同一个点的概率 $1/N$ 贡献 $+\mathrm{Var}(Y)$，抽到不同点的概率 $1-1/N$ 贡献 0，平均下来正好 0。

**第三步：方差。** $Y_{I_1},\dots,Y_{I_N}$ 两两不相关（$j=k$ 时是同一个变量，按方差处理），故由预备知识 P1 的方差可加性

$$
\mathrm{Var}\bigl(\bar Y^{\star}\bigr)=\frac1{N^2}\sum_{j=1}^{N}\mathrm{Var}\bigl(Y_{I_j}\bigr)=\frac1{N^2}\cdot N\cdot\hat\sigma^2=\frac{\hat\sigma^2}{N}
$$

**与真实抽样方差 $\hat\sigma^2/N$ 完全一致**。

**第四步：用多项式权重做「二阶展开」。** 换个视角更看清结构。记 $n_a=\#\{j:I_j=a\}$，则

$$
(n_1,\dots,n_N)\sim\mathrm{Mult}\Bigl(N;\tfrac1N,\dots,\tfrac1N\Bigr),\qquad \bar Y^{\star}=\sum_{a=1}^{N}w_aY_a,\quad w_a=\frac{n_a}{N}
$$

权重满足 $\sum_aw_a=1$、$E[w_a]=1/N$。**二阶量**（用 $\mathrm{Var}(n_a)=Np(1-p)=\frac{N-1}{N}$、$\mathrm{Cov}(n_a,n_b)=-Np^2=-\frac1N$）：

$$
\mathrm{Var}(w_a)=\frac{N-1}{N^{3}},\qquad \mathrm{Cov}(w_a,w_b)=-\frac1{N^{3}}\ (a\ne b)
$$

两者**大小相等、符号相反**，这正是 $\sum_aw_a=1$ 的后果。于是

$$
\mathrm{Var}\Bigl(\sum_{a=1}^{N}w_aY_a\Bigr)=\frac{N-1}{N^3}\sum_{a=1}^{N}Y_a^2-\frac1{N^3}\sum_{a\ne b}Y_aY_b=\frac1{N^3}\Bigl[N\sum_aY_a^2-\Bigl(\sum_aY_a\Bigr)^2\Bigr]=\frac{N^2\hat\sigma^2}{N^3}=\frac{\hat\sigma^2}{N}
$$

**注意这里的条件期望技巧**：给定计数 $n$，$\bar Y^{\star}=\sum_aw_aY_a$ 是**确定的**（$Y_a$ 已观测），所以

$$
\mathrm{Var}(\bar Y^{\star})=\mathrm{Var}\Bigl(E[\bar Y^{\star}\mid n]\Bigr)
$$

只由权重的二阶矩决定。也可用全方差公式分解：$E[\mathrm{Var}(\bar Y^{\star}\mid n)]+0$（前一项不为零才是问题，见下），说明**方差完全来自权重的涨落，而非响应的涨落**——这是理解 bootstrap 的关键直觉。

**第五步：非线性估计量的偏差（一阶 Taylor 展开）。** 对光滑 $g$，

$$
g(\bar Y^{\star})=g(\bar Y)+g'(\bar Y)(\bar Y^{\star}-\bar Y)+\frac12g''(\bar Y)(\bar Y^{\star}-\bar Y)^2+o\bigl(\lVert\bar Y^{\star}-\bar Y\rVert^3\bigr)
$$

取期望（$E[\bar Y^{\star}-\bar Y]=0$）得 $E[g(\bar Y^{\star})]\approx g(\bar Y)+\frac12g''(\bar Y)\cdot\frac{\hat\sigma^2}{N}$：**偏差是 $O(1/N)$**，这就是 bootstrap 对光滑统计量的一阶相合性。方差同样可控：$\mathrm{Var}(g(\bar Y^{\star}))\approx g'(\bar Y)^2\hat\sigma^2/N$，与真实抽样方差的比值 $g'(\bar Y)^2\to1$。

**第六步：为什么「自助抽样独立」这一条也适用于向量与回归**。把第二步推广：设 $\boldsymbol Z_j$ 是 $j$ 维数据向量、各分量独立同分布，则对 $j\ne k$ 有

$$
E[\boldsymbol Z_{I_j}\boldsymbol Z_{I_k}^\top]=\frac1{N^2}\sum_{a,b}E[\boldsymbol Z_a\boldsymbol Z_b^\top]=\frac{\left(\sum_{a=1}^N\boldsymbol Z_a\right)\left(\sum_{b=1}^N\boldsymbol Z_b\right)^\top}{N^2}=\bar{\boldsymbol Z}\,\bar{\boldsymbol Z}^\top
$$

**与 $j=k$ 时的 $E[\boldsymbol Z_{I_j}\boldsymbol Z_{I_j}^\top]=\frac1N\sum_aE[\boldsymbol Z_a\boldsymbol Z_a^\top]=\mathrm{Cov}(\boldsymbol Z)+\bar{\boldsymbol Z}\bar{\boldsymbol Z}^\top$ 相比，只多了 $\mathrm{Cov}(\boldsymbol Z)$ 那一项。** 这正是 8.2 节 (8.7) 的矩阵版自助分布 $\hat\beta^\star\sim N\bigl(\hat\beta,(H^\top H)^{-1}\hat\sigma^2\bigr)$ 的来源。线性泛函的方差被**精确**复制，这是自助法最有价值也最容易被忽视的性质。

**这条推导的三个可迁移要点**：

1. **「$N$ 个自助抽样其实独立」**（第二步）是自助法有效性的核心。任何**线性**泛函 $S(Z)=\sum_aaZ_a$ 都有 $\mathrm{Var}(S(Z^{\star}))=\frac1{N^2}\sum_a\mathrm{Var}(Z_a)=\frac1N\mathrm{Var}(\text{原统计量})$，即精确复制方差。第 8 章 (8.3) 的 $\mathrm{Var}(\hat\beta)=(H^\top H)^{-1}\sigma^2$ 就是这条规则的矩阵版本。
2. **权重涨落而非响应涨落**（第四步）。给定计数 $n$，$\bar Y^{\star}$ 确定，所以方差全部来自 $w$ 的随机性。对理解「为什么自助分布会低估某些相关性结构」很关键。
3. **偏差 $O(1/N)$**（第五步）。对 $C^2$ 光滑的 $g$，自助样本均值的二阶偏差恰好是 $\frac{\hat\sigma^2}{2N}g''(\bar Y)$——可用来构造 **BCa** 区间（7.11.3 节末尾提到的加速项就是 $\frac{g''(\bar Y)}{2}$ 的变体）。

### 7.11.3 用 bootstrap 估计预测误差 {#s-7-11-3}

**朴素做法及其失败**。在每个 $Z^{\star b}$ 上拟合，得到 $\hat f^{\star b}(x_i)$，然后

$$
\widehat{\mathrm{Err}}_{\mathrm{boot}}=\frac1{BN}\sum_{b=1}^{B}\sum_{i=1}^{N}L\bigl(y_i,\hat f^{\star b}(x_i)\bigr)\eqno{7.54}
$$

**这个估计一般很差。** 原因：bootstrap 样本当训练集，原训练集当测试集，两者有大量重复观测，过拟合的预测会显得好得不真实。定量地，

$$
\Pr\bigl\{y_i\text{ 出现在第 }b\text{ 个 bootstrap 样本中}\bigr\}=1-\Bigl(1-\frac1N\Bigr)^{N}\approx1-e^{-1}=0.632\eqno{7.55}
$$

**例子**（1 近邻、预测与标签独立、两类等大、真实错误率 0.5）：若 $y_i$ 出现在 bootstrap 样本里，1 近邻预测它就是自己，贡献 0；若没出现，贡献 0.5。于是

$$
E\bigl[\widehat{\mathrm{Err}}_{\mathrm{boot}}\bigr]\approx0.5\times0.368=0.184\ll 0.5
$$

差了将近三倍。（对照：真实的 $N$ 折 CV 里 $y_i$ 从不出现在训练集，错误率恒为 0.5。）**这就是为什么交叉验证必须用不重叠的训练/测试数据**。

**留一 bootstrap** 修正这个重叠问题。对每个观测 $i$，只统计那些**不含** $y_i$ 的 bootstrap 样本的预测：

$$
\widehat{\mathrm{Err}}^{(1)}=\frac1N\sum_{i=1}^{N}\frac{1}{\lvert C_{-i}\rvert}\sum_{b\in C_{-i}}L\bigl(y_i,\hat f^{\star b}(x_i)\bigr)\eqno{7.56}
$$

其中 $C_{-i}$ 是所有不含第 $i$ 个观测的 bootstrap 样本下标集合。要么取足够大的 $B$ 保证每个 $\lvert C_{-i}\rvert>0$（$E[\lvert C_{-i}\rvert]=0.368B$，$B=100$ 时约有 37 个），要么对零分母的项直接略去。

**但它仍有训练集大小偏差**：每个 bootstrap 样本平均只含 $0.632N$ 个**不同**观测，行为近似二折交叉验证。若学习曲线在 $N/2$ 处仍陡，(7.56) 会向上偏。

**$.632$ 估计量** 为此而生，把 (7.56) 往训练误差方向拉一点（常数 0.632 来自 (7.55)）：

$$
\widehat{\mathrm{Err}}^{(0.632)}=0.368\cdot\mathrm{err}+0.632\cdot\widehat{\mathrm{Err}}^{(1)}\eqno{7.57}
$$

**在过拟合情形下会崩掉**。例：1 近邻 + 标签独立 ⇒ $\mathrm{err}=0$、$\widehat{\mathrm{Err}}^{(1)}=0.5$，于是 $\widehat{\mathrm{Err}}^{(0.632)}=0.316$，而真值是 0.5。

**$.632+$ 估计量** 先度量过拟合程度。定义**无信息错误率** $\gamma$：若输入与标签独立，规则的错误率。在所有「目标 $y_i$ 与预测 $x_{i'}$」的组合上评估即可

$$
\hat\gamma=\frac1{N^2}\sum_{i=1}^{N}\sum_{i'=1}^{N}L\bigl(y_i,\hat f(x_{i'})\bigr)\eqno{7.58}
$$

二分类时（$p_1$ 是 $y_i=1$ 的比例，$q_1$ 是 $\hat f(x_{i'})=1$ 的比例）简化为

$$
\hat\gamma=p_1(1-q_1)+(1-p_1)q_1\eqno{7.59}
$$

（两项分别对应「真 1 判成 0」与「真 0 判成 1」。1 近邻满足 $q_1=p_1$，故 $\hat\gamma=2p_1(1-p_1)$；多分类推广为 $\hat\gamma=\sum_\ell p_\ell(1-q_\ell)$。）相对过拟合率为

$$
\hat R=\frac{\widehat{\mathrm{Err}}^{(1)}-\mathrm{err}}{\hat\gamma-\mathrm{err}}\eqno{7.60}
$$

从 0（无过拟合）到 1（过拟合达到无信息水平）。最终

$$
\widehat{\mathrm{Err}}^{(0.632+)}=\bigl(1-\hat w\bigr)\cdot\mathrm{err}+\hat w\cdot\widehat{\mathrm{Err}}^{(1)},\qquad \hat w=\frac{1-0.368\hat R}{1-0.368}\eqno{7.61}
$$

$\hat R=0$ 时 $\hat w=0.632$，$\hat R=1$ 时 $\hat w=1$，故结果落在 $\widehat{\mathrm{Err}}^{(1)}$ 与 $\widehat{\mathrm{Err}}^{(0.632)}$ 之间。对上面 1 近邻的例子，$\hat R=1$，得 $\hat w=1$，$\widehat{\mathrm{Err}}^{(0.632+)}=\widehat{\mathrm{Err}}^{(1)}=0.5$——**正确**。

**最后一件事：BCa 区间**。(7.61) 这类估计量只管**误差的点估计**。若要**置信区间**，标准做法是百分位法（(7.53) 的 $B$ 个自助值取分位数），但它在偏差与偏斜时覆盖不对。**偏差修正（bias-corrected）** 分位数法用自助分布的分数

$$
z_0=\Phi^{-1}\Bigl(\frac{\#\{b:S(Z^{\star b})<S(Z)\}}{B}\Bigr)
$$

把分位数水平从 $\alpha/2,\,1-\alpha/2$ 平移为 $\Phi\bigl(\Phi^{-1}(\alpha/2)+z_0\bigr)$ 与 $\Phi\bigl(\Phi^{-1}(1-\alpha/2)+z_0\bigr)$。**BCa** 再加一个加速项 $a$（由自助 jackknife 的 $g''$ 估出），两步修正后的区间才在常见情形下有正确覆盖率。这里不给 (7.61) 之外的编号公式，因为原书本章只在正文提了一句 BCa 名称，公式未编号。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-11">原文 §7.11</a>

---

## 7.12 条件误差还是期望误差 {#s-7-12}

图 7.14、7.15 做了一个专门的实验：交叉验证估的是 $\mathrm{Err}_{\mathcal T}$（条件于给定训练集）还是 $\mathrm{Err}$（期望）？

- 10 折 CV 的训练集与原训练集差得比较多，估的更接近 $\mathrm{Err}$；
- N 折（LOO）几乎用全数据，**本该**更接近 $\mathrm{Err}_{\mathcal T}$。

但实验结果相反：**10 折比 N 折更接近 $\mathrm{Err}_{\mathcal T}$**，而且两者对 $\mathrm{Err}$ 都近似无偏、10 折方差更小。

**为什么 LOO 反而不估 $\mathrm{Err}_{\mathcal T}$**：CV 曲线与条件误差曲线的相关性是**负的**。直觉解释是：训练集「好」的那次，拟合更好，$\hat f$ 在训练点附近更贴合，留出的那个点恰好错得更多。也就是说，**CV 误差与条件误差之间存在反向关联**，两种 CV 都在估计对的条件量上系统性地偏。

结论：**只凭同一份训练集的数据，很难估出「这一份训练集的」测试误差**。交叉验证与相关方法只能给出 $\mathrm{Err}$ 的合理估计。

模型选择与模型评估在这里分道扬镳：选择只在乎相对大小（给所有指标加常数不改变选择），评估在乎绝对精度——图 7.13 显示 AIC 平均高估所选模型的预测误差 30%–51%，而 CV 与 bootstrap 只高估 0%–4%。**但反过来，树这类不稳定方法上 CV 与 bootstrap 会低估约 10%**，因为「在验证集上搜索最佳树」本身受验证集影响，这时只有独立的测试集能给无偏估计。

**图 7.13 的四个场景**（与图 7.3、7.7 同一组：KNN/回归×两套数据）说明：在这四个问题上十折 CV 与 $.632+$ bootstrap 的选择质量与 AIC 相当或略差，但**评估精度高一个数量级**。

**选择效应的量级**。设模型 $\mathcal M_1,\dots,\mathcal M_M$ 的真实测试误差分别为 $e_1,\dots,e_M$（当作固定未知常数），各自带独立的估计噪声，方差 $\sigma_m^2$（LOO 的 $\sigma_m^2\approx2\sigma_\varepsilon^2 h_m/N$，图 7.14 右下panel 对这一正态近似给出了支持）。选中的模型下标 $\hat m=\arg\min_m\hat e_m$，则

$$
E[\hat e_{\hat m}-\hat e_{m^\star}]=\sum_m\Pr(\hat m=m)\bigl(E[\hat e_m]-\hat e_{m^\star}\bigr)\approx-\frac{1}{\sqrt{2\pi}}\sum_m\sigma_m\phi\bigl(\frac{\hat e_{m^\star}-\hat e_m}{\sigma_m}\bigr)
$$

（对独立高斯取最小值的经典「次优选择」公式，$\phi$ 为标准正态密度。）**关键**：即使每个 $\hat e_m$ 都无偏，取最小值这一步本身造成**向下偏差**，偏差量级是「有效噪声水平」乘 $\frac1{\sqrt{2\pi}}$，即与 $\sigma_m$ 同阶。$M$ 越大、有效噪声越小，选择效应越小；$N$ 越大、$\sigma_m$ 越小，同理。这也解释了图 7.7 的箱线图为什么总是正的。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-12">原文 §7.12</a>

---

## 7.13 习题中的关键推导 {#s-7-13}

### 7.13.1 0–1 损失下的偏差–方差（习题 7.2）{#s-7-13-1}

平方误差下「偏差 + 方差 = 误差」不适用于分类。设 $Y\in\{0,1\}$，$\mathrm{Pr}(Y=1\mid x_0)=f(x_0)$，$\hat G(x)=I\bigl(\hat f(x)>\frac12\bigr)$ 是 $\hat f$ 诱导的分类规则，$G(x)=I\bigl(f(x)>\frac12\bigr)$ 是 Bayes 规则，$\mathrm{Err}_B(x_0)=\mathrm{Pr}(Y\ne G(x_0)\mid X=x_0)$ 是不可约 Bayes 误差。

$$
\mathrm{Err}(x_0)=\mathrm{Err}_B(x_0)+\bigl|2f(x_0)-1\bigr|\mathrm{Pr}\bigl(\hat G(x_0)\ne G(x_0)\mid X=x_0\bigr)\eqno{7.62}
$$

**推导要点**（练习题原文只给了结论）：$\mathrm{Err}(x_0)=\mathrm{Pr}(Y\ne\hat G\mid x_0)$，按「$f$ 在边界的哪一侧」与「$\hat f$ 的均值在边界的哪一侧」分成四格求和；只有当 $f(x_0)$ 与 $E[\hat f(x_0)]$ 在 $\frac12$ 的**同侧**时才有额外损失，额外损失恰好等于偏差的绝对值乘判错概率。$\lvert2f(x_0)-1\rvert$ 就是「标准化偏差」，在 $f=\frac12$（决策边界上）时为 0——偏差在边界附近无害。

再用 $\hat f(x_0)\sim N\bigl(E\hat f(x_0),\mathrm{Var}(\hat f(x_0))\bigr)$ 近似，判错概率是「均值到边界的有符号距离」除以标准差再用高斯 CDF：

$$
\mathrm{Pr}\bigl(\hat G(x_0)\ne G(x_0)\mid X=x_0\bigr)\approx\Phi\!\left(\frac{\operatorname{sign}\bigl(\tfrac12-f(x_0)\bigr)\bigl(E[\hat f(x_0)]-\tfrac12\bigr)}{\sqrt{\mathrm{Var}(\hat f(x_0))}}\right)\eqno{7.63}
$$

其中 $\Phi(t)=\int_{-\infty}^t\frac{1}{\sqrt{2\pi}}\exp(-u^2/2)\,du$ 是标准正态累积分布函数。**偏差与方差在这里是乘性结合的**：若 $E[\hat f(x_0)]$ 与 $f(x_0)$ 在边界同侧（偏差符号使判错率随偏差增大），减小方差有利；若两者在**异侧**，偏差为正，此时**增大**方差反而有帮助——因为它提高了 $\hat f(x_0)$ 落到正确一侧的机会（Friedman 1997）。

**为什么 (7.62) 成立**：把事件 $\{\hat G(x_0)\ne G(x_0)\}$ 按「$f(x_0)$ 在边界的哪一侧」$c\in\{-1,+1\}$ 分类（$f>\frac12$ 记 $c=+1$）分块求和：

$$
\mathrm{Err}(x_0)-\mathrm{Err}_B(x_0)=E\bigl[c\,\Phi\bigl(\tfrac{c(\hat f-\frac12)}{\sigma}\bigr)\bigr]
$$

分界处 $\hat f=\frac12$ 的概率质量可忽略（连续近似），两块分别给出 $c\,\Phi(\cdot)$ 与 $-c\,\Phi(-c\cdot(\cdot))$，用 $\Phi(-z)=1-\Phi(z)$ 合并即得 (7.62)。其中 $c$ 只通过 $\lvert2f-1\rvert$ 影响概率权重（因为 $f$ 离边界越远，$\Pr[\hat f$ 越过边界 $]$ 越小），这正是边界-偏差项的含义。

### 7.13.2 线性平滑的三个结论 {#s-7-13-2}

**习题 7.3**：LOO 残差恒等式即 (7.64)，已由 7.10.2 节完整推出；条件是 $S_{ii}<1$。
**习题 7.5** 的结论是

$$
\sum_{i=1}^{N}\mathrm{Cov}(\hat y_i,y_i)=\operatorname{trace}(S)\,\sigma_\varepsilon^2\eqno{7.65}
$$

推导只需一步：$\mathrm{Cov}(\hat y_i,y_i)=\mathrm{Cov}(\sum_jS_{ij}y_j,y_i)=\sum_jS_{ij}\mathrm{Cov}(y_j,y_i)=S_{ii}\sigma_\varepsilon^2$，对 $i$ 求和得 $\sigma_\varepsilon^2\sum_iS_{ii}$。（习题 7.6 进一步说明 $k$ 近邻回归的有效自由度是 $N/k$：它等价于 Nadaraya–Watson 核估计，每个 $\hat f(x_i)$ 只是 $k$ 个最近 $y_j$ 的平均，故 $S_{ii}=1/k$、$\operatorname{trace}(S)=N/k$。）
**习题 7.8** 要证明的点列

$$
z_1=10^{-1},\ \dots,\ z_\ell=10^{-\ell}\eqno{7.66}
$$

任意 $\ell$ 个递减点都能被 $\{I(\sin(\alpha x)>0)\}$ 打散：给定任意二分标签，取 $\alpha$ 足够大使相邻零点正好落在选中的点之间，即可让每个点的正负号独立指定。故该类 VC 维无穷大——**参数量少不等于复杂度低**。

<a class="src" href="../esl/ch07-model-assessment-and-selection.html#s-7-12">原文习题</a>