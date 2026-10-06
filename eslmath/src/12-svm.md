---
id: m12
n: "12"
title: 支持向量机与柔性判别
title_en: Support Vector Machines and Flexible Discriminants
desc: 最大间隔的几何推导、拉格朗日对偶与 KKT、软间隔与 hinge 损失、核技巧与再生核希尔伯特空间、回归 SVM、核岭回归、以及柔性/惩罚/混合判别分析。
prev: m11
next: m13
prev_title: 第 11 章 神经网络
next_title: 第 13 章 原型方法与最近邻
---

# 12 支持向量机与柔性判别 {#s-12}

第 4 章在**两类严格可分**的前提下介绍了最优 separating hyperplane。本章做两件事：先把这一节放宽到**类重叠**（不可分）的情形，得到支持向量分类器（SVC），再通过**扩大特征空间**（核技巧）得到非线性边界，即支持向量机（SVM）。

本章的第二个主题在 §12.4 之后：直接推广线性判别分析（LDA）的三个方向——**柔性判别分析 FDA**（把 LDA 化为回归再做非线性）、**惩罚判别分析 PDA**（在图像/信号这类特征极多的情形下正则化 LDA）、**混合判别分析 MDA**（每类用多个原型）。

阅读路线建议：§12.2–12.3 只依赖拉格朗日对偶（预备知识 O1、O2）与正定性（预备知识 L1）；§12.5 引入再生核希尔伯特空间（RKHS），是本章唯一需要额外背景的部分；§12.4 之后只依赖第 4 章的 LDA 与第 5 章的惩罚回归。

<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-2">原文 §12.2</a> 讲支持向量分类器，<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-3">原文 §12.3</a> 讲核与 SVM，<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-5">原文 §12.5</a> 讲 FDA。

---

## 12.1 引言 {#s-12-1}

本章描述线性判别边界用于分类的各种推广。最优 separating hyperplane 在第 4 章已介绍（可分情形）；这里覆盖到类重叠的不可分情形。所得技术随后被推广为**支持向量机**——它通过在一个被变换过的、大得多的特征空间里构造线性边界，从而产生非线性边界。

第二组方法推广 Fisher 的线性判别分析（LDA）：包括柔性判别分析（用与 SVM 十分相似的方式构造非线性边界）、惩罚判别分析（适用于信号/图像分类这类特征极多且高度相关的问题）、以及混合判别分析（用于形状不规则的类）。

> **记号** · 本章统一约定
>
> - 训练数据 $(x_i,y_i)$，$i=1,\dots,N$，$x_i\in\mathbb{R}^p$，$y_i\in\{-1,+1\}$。
> - 判别函数 $f(x)=x^\top\beta+\beta_0$，判决规则 $G(x)=\mathrm{sign}[f(x)]$。
> - 「间隔」margin 指两条平行线 $f(x)=\pm M$ 之间的距离 $2M$。
> - 正定性、黑箱记法见预备知识 L1；对偶与 KKT 见 O1、O2。
> - 第 4 章已经证明：**$f(x)$ 给出点 $x$ 到超平面 $f(x)=0$ 的带符号距离**（当 $\|\beta\|=1$）。这一条是 §12.2.1 全部几何推导的出发点。

---

## 12.2 支持向量分类器 {#s-12-2}

### 12.2.1 最大间隔的判别集与归一化 {#s-12-2-1}

训练数据是 $N$ 对 $(x_1,y_1),\dots,(x_N,y_N)$，$x_i\in\mathbb{R}^p$，$y_i\in\{-1,+1\}$。用超平面

$$
\big\{x:\ f(x)=x^\top\beta+\beta_0=0\big\} \eqno{12.1}
$$

定义，其中 $\beta$ 是单位向量：$\lVert\beta\rVert=1$。由 $f(x)$ 诱导的分类规则是

$$
G(x)=\mathrm{sign}\big[x^\top\beta+\beta_0\big] \eqno{12.2}
$$

超平面的几何在 §4.5 复习过：$f(x)$ 给出点到超平面的带符号距离。因为两类可分，能找到 $f(x)=x^\top\beta+\beta_0$ 使 $y_if(x_i)>0$ 对所有 $i$ 成立。因此我们可以找到在类 $+1$ 与类 $-1$ 的训练点之间造出**最大间隔**的超平面（图 12.1）。优化问题

$$
\max_{M,\ \beta,\beta_0,\ \lVert\beta\rVert=1}\ M\quad \text{s.t.}\quad y_i\big(x_i^\top\beta+\beta_0\big)\ge M,\quad i=1,\dots,N \eqno{12.3}
$$

刻画了这个概念。图中的带宽为 $M$（在超平面两侧各 $M$），故总宽 $2M$，称为**间隔**（margin）。

> **推导** · 从 (12.3) 到 (12.4)：为什么间隔是 $1/\lVert\beta\rVert$
>
> **(a) 点到超平面的带符号距离。** 设超平面 $H=\{x:x^\top\beta+\beta_0=0\}$，$\beta\ne0$。任取 $x$，$x$ 到 $H$ 的垂足是 $x-\lambda\beta$，其中 $\lambda$ 由
>
> $$0=(x-\lambda\beta)^\top\beta+\beta_0=x^\top\beta+\beta_0-\lambda\lVert\beta\rVert^2\ \Longrightarrow\ \lambda=\frac{x^\top\beta+\beta_0}{\lVert\beta\rVert^2}$$
>
> 代入即可。垂足到 $x$ 的距离是 $\lVert x-(x-\lambda\beta)\rVert=\lVert\lambda\beta\rVert=|\lambda|\lVert\beta\rVert=\dfrac{|x^\top\beta+\beta_0|}{\lVert\beta\rVert}$。当 $\lVert\beta\rVert=1$ 时距离就是 $|f(x)|$。$\blacksquare$
>
> **(b) 缩放不变性与 $\beta_0$ 的归一化。** 设 $(\beta,\beta_0)\ne(0,\beta_0)$ 是任一可行对。对任意常数 $c>0$，
>
> $$g_c(x)=c\,f(x)=(c\beta)^\top x+c\beta_0$$
>
> 超平面 $g_c(x)=0$ **完全相同**（因为 $c\ne0$），符号也不变（$\mathrm{sign}[cf(x)]=\mathrm{sign}[f(x)]$）。但 (12.3) 的约束值全部放大 $c$ 倍：
>
> $$y_i\big[(c\beta)^\top x_i+c\beta_0\big]=c\,y_i\big[\beta^\top x_i+\beta_0\big]\ge cM$$
>
> 于是 $M\mapsto cM$，最优值也放大 $c$ 倍。**这说明 (12.3) 没有唯一的 $(\beta,\beta_0,M)$** —— 沿这条射线任意缩放都给出可行解且线性地改善目标。解决办法就是**归一化**：规定 $\lVert\beta\rVert=1$，(12.3) 里的这个约束正是为了消除这个尺度自由度。
>
> **(c) 用缩放换掉 $\lVert\beta\rVert=1$。** 反过来，若把 (12.3) 的归一化改成**固定 $\beta_0$ 的归一化**：取 $c=1/|\beta_0|$（可分时 $\beta_0\ne0$，否则类 $+1$ 与类 $-1$ 都在同一侧，矛盾）。记 $\tilde\beta=\beta/|\beta_0|$，$\tilde\beta_0=\mathrm{sign}(\beta_0)$，$\tilde M=M/|\beta_0|$。代入约束：
>
> $$y_i\big[\tilde\beta^\top x_i+\tilde\beta_0\big]=\frac{y_i[\beta^\top x_i+\beta_0]}{|\beta_0|}\ge\frac{M}{|\beta_0|}=\tilde M$$
>
> 而 $\tilde M=1/\lVert\tilde\beta\rVert$（因为超平面与 $f(x)=0$ 相同，距离 $M$ 用 (a) 的公式等于 $1/\lVert\beta\rVert$）。把 $\tilde M=1/\lVert\tilde\beta\rVert$ 代回约束并两边乘 $\lVert\tilde\beta\rVert$，得
>
> $$y_i\big[\tilde\beta^\top x_i+\tilde\beta_0\big]\ \ge\ 1$$
>
> **归一化 $\beta_0\in\{+1,-1\}$ 之后，$\lVert\beta\rVert$ 越小，间隔 $2M=2/\lVert\beta\rVert$ 越大**。所以 $\lVert\beta\rVert$ 最小化 $\Leftrightarrow$ 间隔最大化。去掉 $\lVert\beta\rVert=1$（$\beta$ 可以任意缩放），问题写成
>
> $$\min_{\beta,\beta_0}\ \lVert\beta\rVert\quad \text{s.t.}\quad y_i\big(x_i^\top\beta+\beta_0\big)\ge1,\quad i=1,\dots,N \eqno{12.4}$$
>
> **结论：margin $=\dfrac{1}{\lVert\beta\rVert}$，margin 带宽 $=2M=\dfrac{2}{\lVert\beta\rVert}$。** $\blacksquare$
>
> (12.4) 是**凸**优化问题：目标是严格凸二次型 $\lVert\beta\rVert$（预备知识 C2：$\frac12\beta^\top\beta$ 的系数矩阵 $I$ 正定，故目标严格凸），约束是线性不等式（预备知识 O1 的标准形）。**这保证解唯一、不存在局部非全局极小**——第 5、12、18 章所有凸性论证的根据。解的特征在 §4.5.2 给出，正是本章 §12.2.3 要推的。

### 12.2.2 软间隔：允许点落在间隔之内 {#s-12-2-2}

现在假设类在特征空间中重叠。仍想最大化 $M$，但允许某些点落在间隔的错误一侧。定义松弛变量 $\xi=(\xi_1,\dots,\xi_N)$。修改 (12.3) 的约束有两种自然方式：

$$
y_i\big(x_i^\top\beta+\beta_0\big)\ge M-\xi_i \eqno{12.5}
$$

或

$$
y_i\big(x_i^\top\beta+\beta_0\big)\ge M(1-\xi_i)\quad \forall i,\qquad \xi_i\ge0,\qquad \sum_{i=1}^N\xi_i\le \text{const} \eqno{12.6}
$$

两种选择给出不同的解。第一种看起来更自然，因为它用**到间隔的实际距离**度量重叠；第二种用**相对距离**度量，而相对距离随间隔宽度 $M$ 变化。但第一种导致**非凸**优化问题，第二种是凸的，所以 (12.6) 导出了「标准」的 SVC，以下都用它。

(12.6) 的思想是：约束 $y_i(x_i^\top\beta+\beta_0)\ge M(1-\xi_i)$ 里的 $\xi_i$ 是预测 $f(x_i)$ 落在自己间隔错误一侧的**比例**。限制 $\sum_i\xi_i$ 就限制了预测落在错误一侧的**总比例**。当 $\xi_i>1$ 时发生误分类，所以把 $\sum\xi_i$ 限制在某个 $K$，就把训练误分类总数限制在 $K$ 以内。

> **坑**：为什么 (12.5) 非凸而 (12.6) 凸？这不是美学问题。写成一个统一形式：约束是 $y_i(x_i^\top\beta+\beta_0)\ge M-\xi_i$，在 $M=1/\lVert\beta\rVert$ 的替换下变成 $y_i(\cdot)\lVert\beta\rVert-\xi_i\lVert\beta\rVert\ge1$，其中 $\lVert\beta\rVert$ 是**变量**的凹函数。在约束中引入变量的非凹函数会破坏凸性。(12.6) 把 $\xi_i$ 乘在**常数** $M$ 上，则 $\partial^2/\partial M\partial\xi_i=0$，二次型仍正定，故凸。**结论：$\xi_i$ 必须乘 $M$ 而不是减 $M$。** 这是从 (12.5) 到 (12.6) 唯一的技术差别，却是全部数学的支点。

像 §4.5.2 的 (4.48) 一样，可以去掉 $\beta$ 的范数约束、定义 $M=1/\lVert\beta\rVert$，把 (12.4) 写成等价形式

$$
\min_{\beta,\beta_0}\ \lVert\beta\rVert\quad \text{s.t.}\quad \big\{\,y_i\big(x_i^\top\beta+\beta_0\big)\ge1-\xi_i\ \ \forall i,\qquad \xi_i\ge0,\ \ \xi_i\le\text{const}\,\big\} \eqno{12.7}
$$

这是不可分情形下 SVC 的通常写法。但书里认为约束 $y_i(x_i^\top\beta+\beta_0)\ge1-\xi_i$ 里那个固定标度「1」的出现令人困惑，所以更喜欢从 (12.6) 出发。

由 (12.7) 的准则可见，**离自己类边界很远的点对塑造边界不起大作用**。这是个有吸引力的性质，也是 SVC 与 LDA 的区别所在：在 LDA 中，判别边界由类分布的协方差与类中心的**位置**共同决定。§12.3.3 会看到逻辑回归在这方面与 SVC 更相似。

计算上，把 (12.7) 重新表述成等价形式更方便：

$$
\min_{\beta,\beta_0}\ \lVert\beta\rVert+C\sum_{i=1}^N\xi_i\quad \text{s.t.}\quad \xi_i\ge0,\quad y_i\big(x_i^\top\beta+\beta_0\big)\ge1-\xi_i\ \ \forall i \eqno{12.8}
$$

其中「代价」参数 $C$ 替换了 (12.7) 里的常数；可分情形对应 $C=\infty$。

> **结果** · $C$ 与 $\sum\xi_i$ 上界的换算
>
> 比较 (12.7) 与 (12.8)：约束集相同（若把「$\xi_i\le$ const」理解为对 $\sum\xi_i$ 的约束）。目标函数 $\lVert\beta\rVert$ 对 $\xi$ 严格**递减**（$\partial\lVert\beta\rVert/\partial\xi_i=0$，即在给定 $\beta$ 下 $\xi$ 取越小越好）。于是给定 $\xi$，$\lVert\beta\rVert$ 取最小；等价地，把目标写成 $\lVert\beta\rVert+\underbrace{C\sum\xi_i}_{\text{罚项}}$ 就是标准的「拉格朗日乘子」替换：约束 $\sum\xi_i\le K$ 的乘子就是 $C$，两者的对应关系见 §12.3 的对偶推导。$C\to\infty$ 时 (12.8) 强制所有 $\xi_i=0$，退化为 (12.4)（可分情形）。
>
> **坑**：$\lVert\beta\rVert$ 的**尺度不变性**在这里仍在，只是被约束「$y_i f(x_i)\ge1-\xi_i$」固定了。也就是说 (12.8) 与 (12.7) 等价的原因正是 $\xi$ 吸收了尺度：若 $(c\beta,c\beta_0)$ 使 $\xi$ 变紧 $c$ 倍，两个目标同比例缩放，比值不变。

### 12.2.3 用拉格朗日乘子求解 {#s-12-2-3}

问题 (12.7) 是带线性不等式约束的二次规划，因此是凸优化问题。我们用拉格朗日乘子描述二次规划解。拉格朗日（primal）函数是

$$
L_P=\frac12\lVert\beta\rVert^2+C\sum_{i=1}^N\xi_i-\sum_{i=1}^N\alpha_i\big[y_i\big(x_i^\top\beta+\beta_0\big)-(1-\xi_i)\big]-\sum_{i=1}^N\mu_i\xi_i \eqno{12.9}
$$

对 $\beta,\beta_0,\xi_i$ 最小化。令各导数为零得

$$
\beta=\sum_{i=1}^N\alpha_i y_i x_i \eqno{12.10}
$$

$$\sum_{i=1}^N\alpha_i y_i=0 \eqno{12.11}
$$

$$\alpha_i=C-\mu_i,\quad \forall i \eqno{12.12}$$

外加正性约束 $\alpha_i,\mu_i,\xi_i\ge0\ \forall i$。（(12.10) 给出 $\beta=\sum\alpha_i y_ix_i$，等式约束 $\sum\alpha_i y_i=0$ 单独编号为 (12.11)，是推导中 $\partial L_P/\partial\beta_0=0$ 的结果；把 (12.10)–(12.12) 代回 (12.9)，得到对偶目标函数

$$
L_D=\sum_{i=1}^N\alpha_i-\frac12\sum_{i=1}^N\sum_{i'=1}^N\alpha_i\alpha_{i'}y_iy_{i'}\,x_i^\top x_{i'} \eqno{12.13}
$$

它给出 (12.8) 目标函数在任意可行点处的下界。在 $0\le\alpha_i\le C$ 与 $\sum_{i=1}^N\alpha_i y_i=0$ 的约束下最大化 $L_D$。除 (12.10)–(12.12) 外，KKT 条件还包括约束

$$
\alpha_i\Big[y_i\big(x_i^\top\beta+\beta_0\big)-(1-\xi_i)\Big]=0 \eqno{12.14}
$$

$$\mu_i\xi_i=0 \eqno{12.15}
$$

$$y_i\big(x_i^\top\beta+\beta_0\big)-(1-\xi_i)\ge0 \eqno{12.16}
$$

对 $i=1,\dots,N$。这些方程 (12.10)–(12.16) 一起唯一刻画原问题与对偶问题的解。

> **推导** · (12.9) → (12.10) → (12.13) 的完整步骤
>
> **第 1 步：写出原问题。** (12.8) 记作
>
> $$\min_{\beta,\beta_0,\xi}\ \tfrac12\lVert\beta\rVert^2+C\sum_i\xi_i\quad\text{s.t.}\quad \xi_i\ge0,\quad y_i(x_i^\top\beta+\beta_0)\ge1-\xi_i$$
>
> 整理成 $\min f_0$ s.t. $f_i\le0$ 的标准形（预备知识 O1）：
>
> $$f_0=\tfrac12\lVert\beta\rVert^2+C\sum_i\xi_i,\qquad f_i=y_i\big[(x_i^\top\beta+\beta_0)+\xi_i\big]-1\le0,\qquad g_i=-\xi_i\le0$$
>
> **第 2 步：拉格朗日函数。** 取乘子 $\alpha_i\ge0$（对应 $f_i$）与 $\mu_i\ge0$（对应 $g_i$）：
>
> 把 (12.9) 里的 $-\sum_i\alpha_i\big[y_if_i-(1-\xi_i)\big]$ 按 $f_i=x_i^\top\beta+\beta_0$ 展开，注意 slack 以**系数 1** 进入、不带 $y_i$：
>
> $$-\sum_i\alpha_i\big[y_if_i-1+\xi_i\big]=\sum_i\alpha_i(1-y_if_i)-\sum_i\alpha_i\xi_i$$
>
> 于是等价地
>
> $$L_P=\tfrac12\lVert\beta\rVert^2+C\sum_i\xi_i+\sum_i\alpha_i\big(1-y_i(x_i^\top\beta+\beta_0)\big)-\sum_i\alpha_i\xi_i-\sum_i\mu_i\xi_i$$
>
> 整理成 (12.9) 的形式（把 $-\alpha_i[y_i(x_i^\top\beta+\beta_0)-(1-\xi_i)]=-\alpha_i y_i(x_i^\top\beta+\beta_0)+\alpha_i-\alpha_i\xi_i$，再与 $-\mu_i\xi_i$ 合并）。$\blacksquare$
>
> **第 3 步：逐坐标求 $\inf$。**
>
> 对 $\beta$ 求梯度（用预备知识 L4：$\nabla_\beta\tfrac12\beta^\top\beta=\beta$，$\nabla_\beta x_i^\top\beta=x_i$）：
>
> $$\frac{\partial L_P}{\partial\beta}=\beta-\sum_i\alpha_i y_i x_i=0\ \Longrightarrow\ \boxed{\ \beta=\sum_{i=1}^N\alpha_i y_i x_i\ }$$
>
> 这正是 (12.10) 的第一式。
>
> 对 $\beta_0$ 求梯度：
>
> $$\frac{\partial L_P}{\partial\beta_0}=-\sum_i\alpha_i y_i=0\ \Longrightarrow\ \boxed{\ \sum_{i=1}^N\alpha_i y_i=0\ }$$
>
> 这正是 (12.11)。**这一条是后面「$y^\top\alpha=0$」的对偶等式约束的来源**（原书写成 (12.10) 里的一行，也可视为 (12.11)）。
>
> 对 $\xi_i$ 求偏导（注意 $\xi_i$ 只出现在三个和里）：
>
> $$\frac{\partial L_P}{\partial\xi_i}=C-\alpha_i-\mu_i=0\ \Longrightarrow\ \boxed{\ \alpha_i=C-\mu_i\ }$$
>
> 这正是 (12.12)。**加上 $\mu_i\ge0$ 就得到对偶可行性 $0\le\alpha_i\le C$**，这解释了为什么对偶问题的约束是盒约束。
>
> **第 4 步：代回求对偶函数。** 记 $S_0=\sum_i\alpha_i y_i$，$A=\sum_i\alpha_i$。代入 $\beta=\sum_i\alpha_i y_ix_i$：
>
> $$\tfrac12\lVert\beta\rVert^2=\tfrac12\Big\lVert\sum_i\alpha_i y_ix_i\Big\rVert^2=\tfrac12\sum_i\sum_{i'}\alpha_i\alpha_{i'}y_iy_{i'}\underbrace{x_i^\top x_{i'}}_{\langle x_i,x_{i'}\rangle}$$
>
> 二次项（由 $-\sum_i\alpha_i y_i\beta^\top x_i$ 与 $-\sum_i\alpha_i y_i\beta_0$ 两部分）：
>
> $$-\sum_i\alpha_i y_i\,x_i^\top\beta=-\sum_i\alpha_i y_i\,x_i^\top\Big(\sum_{i'}\alpha_{i'}y_{i'}x_{i'}\Big)=-\sum_i\sum_{i'}\alpha_i\alpha_{i'}y_iy_{i'}\underbrace{x_i^\top x_{i'}}_{\text{与上相同}}$$
>
> 两项一正一负，**恰好相消**（这是所有线性不可分判别对偶的关键代数）。剩下常数项：
>
> $$\sum_i\alpha_i-\underbrace{\sum_i\alpha_i y_i\beta_0}_{=0\ \text{由 }S_0=0}+\sum_i\underbrace{\alpha_i\xi_i}_{=\mu_i\xi_i-\mu_i\xi_i=0\ \text{由 }\mu_i\xi_i=0}$$
>
> 中间两段都用到了第 3 步的稳态条件。$\blacksquare$
>
> **第 5 步：强对偶。** (12.8) 是凸问题且 Slater 条件成立（取 $\xi_i>0$ 且 $\beta=0,\beta_0$ 充分大即可使所有 $f_i<0$ 严格成立），故对偶最优值等于原问题最优值，**原对偶解相同**。
>
> **第 6 步：对偶问题的最终形式。**
>
> $$\boxed{\ \max_{\alpha}\ \sum_{i=1}^N\alpha_i-\tfrac12\sum_i\sum_{i'}\alpha_i\alpha_{i'}y_iy_{i'}\langle x_i,x_{i'}\rangle\quad\text{s.t.}\quad 0\le\alpha_i\le C,\ \sum_i\alpha_i y_i=0\ }$$
>
> 这就是 (12.13) 加约束。$\blacksquare$
>
> **关键观察**：对偶问题只通过内积 $x_i^\top x_{i'}$ 依赖数据。这是 §12.3 核技巧的入口。
>
> **坑**：原问题中的 $C\sum\xi_i$ 有 $N$ 个变量 $\xi_i$，对偶只有 $N$ 个变量 $\alpha_i$——问题规模相同。但如果用变量消去（把 $\beta,\beta_0,\xi$ 都消掉），原问题有 $2p+1+N$ 个变量；**而对偶的最优值可以在 $N$ 很小的时候也算出来**。此外，对偶总是凸的（预备知识 O1），即使原问题不是。

> **结果** · (12.13) 给出原问题目标的下界
>
> 对偶函数 $g(\alpha)=\inf_{\beta,\beta_0,\xi}L_P$ 对任意 $\alpha\ge0$ 都满足 $g(\alpha)\le f_0(\text{任一可行点})$（弱对偶，见预备知识 O1）。所以 (12.13) 在 $\alpha$ 可行域上最大化得到的是**下界**，只有 Slater 成立时才是精确的。

### 12.2.4 KKT 与支持向量 {#s-12.2-4}

由 (12.10) 可见 $\beta$ 的解有形式

$$
\hat\beta=\sum_{i=1}^N\hat\alpha_i y_i x_i \eqno{12.17}
$$

其中 $\hat\alpha_i$ 只对那些约束 (12.16) 恰好取等的观测 $i$ 非零（由 (12.14)）。这些观测称为**支持向量**，因为 $\hat\beta$ 只由它们表示。在这些支持点中：

- 一些位于间隔边缘（$\hat\xi_i=0$），由 (12.15) 与 (12.12) 知它们满足 $0<\hat\alpha_i<C$；
- 其余（$\hat\xi_i>0$）满足 $\hat\alpha_i=C$（即**落在间隔错误一侧的点**）。

由 (12.14) 可见，任何一个这样的间隔点（$0<\hat\alpha_i,\ \hat\xi_i=0$）都可用来解出 $\beta_0$；数值上通常取所有解的**平均**以提高稳定性。

> **结果** · 三组支持点的精确刻画
>
> 结合 (12.14)(12.15)(12.12) 与 $y_i f(x_i)=1-\xi_i+\varepsilon_i$（$\varepsilon_i\ge0$ 为松弛量），三类支持点：
>
> $$\begin{cases}\hat\alpha_i=C,\ \hat\xi_i>0&\text{间隔错误侧的点（含误分类点）}\\ 0<\hat\alpha_i<C,\ \hat\xi_i=0&\text{恰在间隔边界 } y_if(x_i)=1\text{ 上的点}\\ \hat\alpha_i=0&\text{间隔正确侧且有余量的点，对 }\hat\beta\text{ 无贡献}\end{cases}$$
>
> **关键推论**：$\hat\alpha_i=0$ 的点对 $\hat\beta$（从而对判别边界）**毫无影响**——删掉它们，SVM 的解完全不变。这意味着 **LOOCV 误差 $\le$ 支持向量比例**。第 12.2.5 节的 Figure 12.2 中这个比例是 62% 和 85%，作为上界太松，但仍是有效的 sanity check。

最大化对偶 (12.13) 是比原问题 (12.9) 更简单的凸二次规划，可用标准技术求解（Murray et al., 1981）。给定解 $\hat\beta_0$ 与 $\hat\beta$，判决函数写作

$$
\hat G(x)=\mathrm{sign}\big[\hat f(x)\big]=\mathrm{sign}\big[x^\top\hat\beta+\hat\beta_0\big] \eqno{12.18}
$$

该过程的调节参数是代价参数 $C$。

> **坑**：$\hat\beta_0$ 的确定有三种情况：
>
> 1. 若**所有**支持点都满足 $\hat\xi_i=0$（数据可分），$\hat\beta_0$ 由任一边界点解出，且对所有这些点解一致（它们给出同一值）。
> 2. 若**部分**点满足 $\hat\xi_i>0$（$\hat\alpha_i=C$），它们给出不同的 $\hat\beta_0$（因为 $y_if(x_i)=1-\hat\xi_i<1$ 不等式）。此时只能用 $0<\hat\alpha_i<C$ 的点求解。
> 3. 若所有点都有 $\hat\xi_i>0$（无点在间隔边界上，比如 $C$ 极小时），$\hat\beta_0$ 由**任意一点**在边界 $y_if(x_i)=1-\hat\xi_i$ 上求解。
>
> 数值上取所有 $\hat\xi_i=0$ 且 $\hat\alpha_i<C$ 的点解的平均，实践中最稳。

### 12.2.5 C 的选择与混合数据例 {#s-12-2-5}

$C$ 的最优值可用第 7 章的交叉验证估计。有趣的是，**留一法交叉验证误差可以从上界被支持点比例控制**：删掉一个非支持向量不会改变解，因此这些观测被原边界正确分类，在交叉验证中仍被正确分类。

图 12.2 给出图 2.5 混合数据在两个 $C$ 值下的 SVC 边界。落在边界错误侧的点是支持向量；落在正确侧但紧贴边界的（落在 margin 内的）点也是支持向量。$C=0.01$ 时 margin 比 $C=10\,000$ 时大。因此 $C$ 大时更关注（被正确分类的）靠近边界的点，$C$ 小时关注更远的点。无论哪种情况，误分类点都被给了权重，不管离多远。在这个例子中程序对 $C$ 的选择不很敏感，因为线性边界本身很「刚」。

<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-2-2">原文 §12.2.2</a>

---

## 12.3 支持向量机与核 {#s-12-3}

### 12.3.1 只通过内积做分类 {#s-12-3-1}

前面描述的 SVC 在**输入特征空间**中找线性边界。与其它线性方法一样，我们可以用多项式或样条等基展开（预备知识第 5 章）扩大特征空间使程序更灵活。放大空间中的线性边界通常能在训练集上分得更好，并映射回原空间的非线性边界。一旦基函数 $h_m(x)$，$m=1,\dots,M$ 选定，程序与之前相同：用输入特征 $h(x_i)=(h_1(x_i),\dots,h_M(x_i))$ 拟合 SVC，产生（非线性）函数 $\hat f(x)=h(x)^\top\hat\beta+\hat\beta_0$。分类器是 $\hat G(x)=\mathrm{sign}(\hat f(x))$。

SVM 分类器是这个思想的推广，**放大空间的维数可以非常大，甚至无穷**。我们先说明 SVM 技术如何处理这个问题，然后会看到 SVM 实际上是在用一个特定准则和特定正则形式解一个函数拟合问题，属于一个更大的问题类（包含第 5 章的光滑样条）。

我们可以把优化问题 (12.9) 及其解重新表示成**只通过内积涉及输入特征**的形式。直接对变换后的特征向量 $h(x_i)$ 这样做。然后会看到对特定的 $h$，这些内积可以极便宜地算出。

> **推导** · 把 (12.13) 与 $\beta=\sum\alpha_i y_ix_i$ 写成核形式
>
> **(a) 对偶函数。** (12.13) 中的内积 $x_i^\top x_{i'}$ 换成 $\langle h(x_i),h(x_{i'})\rangle$：
>
> $$L_D=\sum_{i=1}^N\alpha_i-\frac12\sum_{i=1}^N\sum_{i'=1}^N\alpha_i\alpha_{i'}y_iy_{i'}\langle h(x_i),h(x_{i'})\rangle \eqno{12.19}$$
>
> **(b) 解函数 $f(x)$。** 由 (12.10) $\beta=\sum\alpha_i y_ix_i$，
>
> $$f(x)=h(x)^\top\beta+\beta_0=h(x)^\top\sum_{i=1}^N\alpha_i y_i h(x_i)+\beta_0=\sum_{i=1}^N\alpha_i y_i\langle h(x),h(x_i)\rangle+\beta_0 \eqno{12.20}$$
>
> 注意 (12.20) 与 $\dim h$ 无关。给定 $\alpha_i$，$\beta_0$ 可由任一（或所有）满足 $0<\alpha_i<C$ 的 $x_i$ 解 $y_if(x_i)=1$ 得到。
>
> **(c) 定义核。** 我们**完全不必指定变换 $h(x)$ 本身**，只要求知道核函数
>
> $$K(x,x')=\langle h(x),h(x')\rangle \eqno{12.21}$$
>
> 它计算变换后空间中的内积。$K$ 应当是对称半正定函数（详见第 5 章 §5.8.1）。$\blacksquare$
>
> **关键**：$\langle h(x),h(x_i)\rangle=K(x,x_i)$，所以预测只需算 $N$ 个核值，与 $\dim h$ 无关。**SVM 从不显式构造 $h$**。

SVM 文献中最常用的三个 $K$ 是

$$
K(x,x')=\big(1+\langle x,x'\rangle\big)^d,\qquad K(x,x')=\exp\big(-\gamma\lVert x-x'\rVert^2\big),\qquad K(x,x')=\tanh\big(\kappa_1\langle x,x'\rangle+\kappa_2\big) \eqno{12.22}
$$

分别是 $d$ 阶多项式、径向基、神经网络核。

**例：二输入 + 二阶多项式核。** 展开

$$
K(X,X')=\big(1+\langle X,X'\rangle\big)^2=\big(1+X_1X_1'+X_2X_2'\big)^2=1+2X_1X_1'+2X_2X_2'+\big(X_1X_1'\big)^2+2X_1X_1'X_2X_2'+\big(X_2X_2'\big)^2
$$

这可以分解为 $h_1=1,\ h_2=\sqrt2 X_1,\ h_3=\sqrt2 X_2,\ h_4=X_1^2,\ h_5=X_2^2,\ h_6=\sqrt2 X_1X_2$，使得

$$
K(X,X')=\langle h(X),h(X')\rangle,\qquad \dim h=M=6 \eqno{12.23}
$$

由 (12.20) 可见解可以写成

$$
\hat f(x)=\sum_{i=1}^N\hat\alpha_i y_i K(x,x_i)+\hat\beta_0 \eqno{12.24}
$$

（系数取 $\hat\alpha_i=(\alpha_i/C)$ 时上式与 (12.20) 一致，见后文讨论。）

### 12.3.2 SVM 作为惩罚方法 {#s-12-3-2}

取 $f(x)=h(x)^\top\beta+\beta_0$，考虑优化问题

$$
\min_{\beta_0,\ \beta}\ \sum_{i=1}^N\lambda\big[1-y_if(x_i)\big]_+ +\lVert\beta\rVert^2 \eqno{12.25}
$$

其中下标「$+$」表示正部，即 $[t]_+=\max(t,0)$。这个形式是「损失 + 惩罚」，是函数估计中大家熟悉的范式。容易验证（Exercise 12.1）：(12.25) 在 $\lambda=1/C$ 时的解与 (12.8) 的解相同。

「hinge」损失 $L(y,f)=[1-yf]_+$ 的性质说明它对两类分类是合理的。图 12.4 把它与逻辑回归的对数似然、平方误差及其一个变体作比较。负对数似然（binomial deviance）与 SVM 损失有类似的尾部，给 margin 内侧的点零惩罚，给错误侧且远的点线性惩罚。

> **推导** · (12.8) 与 (12.25) 的等价（Exercise 12.1）
>
> **基础知识** · hinge 损失为什么「估计分类器本身」（联系预备知识 O5）
>
> 0-1 损失下 Bayes 规则是 $\arg\max_k p_k(x)$（预备知识 O5），**不估计任何概率值**。hinge 损失恰好继承这个性质：记 $p=\Pr(Y=+1\mid X=x)$，则
>
> $$\min_{f\ \text{不依赖}\ x}\ \mathbb E\big[(1-Yf(X))_+\big]=\min_f\Big\{p\,[1-f]_+^++(1-p)[1+f]_+\Big\}$$
>
> **第 1 步**（$f\ge1$）：损失 $=\ p\cdot0+(1-p)(1+f)$，对 $f$ 递增，最小在 $f=1$。
> **第 2 步**（$-1\le f\le1$）：损失 $=\ p(1-f)+(1-p)(1+f)=1+(1-2p)f$，线性，最小在 $f=-1$ 当 $p>\frac12$、$f=1$ 当 $p<\frac12$。
> **第 3 步**（$f\le-1$）：损失 $=\ p(1-f)+(1-p)\cdot0$，对 $f$ 递减。
>
> 所以最优解是 $f^*=\mathrm{sign}(p-\frac12)$，**只取后验的符号，完全不估计概率的大小**。对照：平方误差的最优解是 $f^*=2p-1$（估计概率），对数损失的最优解是 $f^*=\log\frac{p}{1-p}$（估计 log-odds，unbounded）。**这就是表 12.1 中三种损失的唯一区别，也是 hinge 损失「稀疏」性质的来源。**

> **推导** · (12.25) 与 (12.8) 等价（Exercise 12.1）
>
> **第 0 步：把 (12.8) 写成纯损失形式。** (12.8) 的目标 $\lVert\beta\rVert+C\sum_i\xi_i$ 中 $\xi_i$ 只带**正**系数 $C>0$，故在可行域上取 $\xi_i$ 的最小值，即
>
> $$\xi_i^*=\big[1-y_if(x_i)\big]_+=\big[1-a_i\big]_+,\qquad a_i:=y_i(\beta^\top x_i+\beta_0)$$
>
> 所以 (12.8) 等价于最小化
>
> $$J_2(\beta,\beta_0)=\underbrace{\lVert\beta\rVert}_{\text{一次齐次}}+\ C\sum_{i=1}^N\big[1-a_i\big]_+ \qquad\text{s.t.}\ \beta\ne0$$
>
> **第 1 步：$J_2$ 是正齐次的。** 对任意 $t>0$：$\lVert t\beta\rVert=t\lVert\beta\rVert$，且 $1-t\,a_i$ 不等于 $t(1-a_i)$（损失不是齐次的），但**符号模式** $a_i\le0\ \Leftrightarrow\ ta_i\le0$ 被保留。所以 $J_2(t\beta,t\beta_0)=t\lVert\beta\rVert+C\sum_i[1-ta_i]_+$，其极小点 $\beta$ 的**方向**只由方向决定。
>
> **第 2 步：$J_1$（= (12.25)）的一阶条件。** $J_1(\beta,\beta_0)=\lambda\sum_i[1-a_i]_++\lVert\beta\rVert^2$。分段线性损失的可导点处 $\partial[1-a_i]_+/\partial a_i=-\mathbb 1\{a_i<1\}$，而 $\partial a_i/\partial\beta_j=y_ix_{ij}$，$\partial a_i/\partial\beta_0=y_i$。故
>
> $$\frac{\partial J_1}{\partial\beta_j}=-\lambda\sum_{i\in A}y_ix_{ij}+2\beta_j=0\ \Longrightarrow\ \boxed{\ \beta=\frac{\lambda}{2}\sum_{i\in A}y_ix_i\ }\qquad(A:=\{\,i:a_i<1\,\})$$
>
> $$\frac{\partial J_1}{\partial\beta_0}=-\lambda\sum_{i\in A}y_i=0\ \Longrightarrow\ \boxed{\ \sum_{i\in A}y_i=0\ }$$
>
> 记 $\beta^{(1)}$、$\beta_0^{(1)}$ 为 (12.25) 的极小，$A^{(1)}$ 为其活动集。
>
> **第 3 步：(12.8) 的 KKT 条件。** 把 $\lVert\beta\rVert$ 写成 (12.9) 的形式后求导（$\nabla_\beta\lVert\beta\rVert=\beta/\lVert\beta\rVert$，见预备知识 C1）：
>
> $$\frac{\partial L_P}{\partial\beta}=\frac{\beta}{\lVert\beta\rVert}-\sum_i\alpha_i y_ix_i=0\ \Longrightarrow\ \frac{\beta}{\lVert\beta\rVert}=\sum_i\alpha_i y_ix_i$$
>
> $$\frac{\partial L_P}{\partial\beta_0}=-\sum_i\alpha_i y_i=0,\qquad \frac{\partial L_P}{\partial\xi_i}=C-\alpha_i-\mu_i=0\ \Longrightarrow\ \alpha_i=C-\mu_i\in[0,C]$$
>
> 再加互补松弛 $\alpha_i\,[a_i+\xi_i-1]=0$ 与 $\mu_i\xi_i=0$。
>
> **第 4 步：构造对应关系。** 取
>
> $$\alpha_i=C\cdot\mathbb 1\{i\in A^{(1)}\},\qquad C:=\frac{\lambda}{2\lVert\beta^{(1)}\rVert}$$
>
> （即取 $C$ 使两条 $\beta$-平稳性对齐。）逐条验证 KKT：
>
> 1. **原始可行**：$A^{(1)}$ 的定义给出 $a_i<1$（故取 $\xi_i=[1-a_i]_+$ 后可行）；$B:=\{a_i=1\}$ 的点取 $\xi_i=0$ 也可行。
> 2. **对偶可行**：$\alpha_i\in\{0,C\}\subset[0,C]$，可取 $\mu_i=C-\alpha_i\in\{0,C\}$。
> 3. **$\mu_i\xi_i=0$**：$i\in A^{(1)}$ 时 $\mu_i=C>0$ 且 $\xi_i=[1-a_i]_+\ge0$——**要求 $\xi_i>0$**。由第 2 步 $\sum_{i\in A^{(1)}}y_i=0$ 与 $A^{(1)}$ 的极大性可以验证 $\xi_i>0$（否则该点可从活动集移出而不破坏前两个方程，矛盾）。$i\notin A^{(1)}$ 时 $\mu_i=0$。✓
> 4. **$\beta$-平稳性**：
>    $$\sum_i\alpha_i y_ix_i=C\sum_{i\in A^{(1)}}y_ix_i=\frac{\lambda}{2\lVert\beta^{(1)}\rVert}\cdot 2\beta^{(1)}=\frac{\beta^{(1)}}{\lVert\beta^{(1)}\rVert}$$
>    由第 2 步 $\sum_{i\in A^{(1)}}y_ix_i=2\beta^{(1)}/\lambda$。✓
> 5. **$\beta_0$-平稳性**：$\sum_i\alpha_i y_i=C\sum_{i\in A^{(1)}}y_i=0$（第 2 步第二式）。✓
> 6. **互补松弛 $\alpha_i[a_i+\xi_i-1]=0$**：$i\in A^{(1)}$ 时 $a_i+\xi_i-1=a_i+(1-a_i)-1=0$ ✓；$i\notin A^{(1)}$ 时 $\alpha_i=0$ ✓。
>
> 四条 KKT 全部满足。因为 (12.8) 是**凸**问题（预备知识 O2：KKT 是充要条件），$(\beta^{(1)},\beta_0^{(1)})$ 就是 (12.8) 在该 $C$ 下的**全局最优**。$\blacksquare$
>
> **反向同理**：给定 (12.8) 的最优 $(\hat\beta,\hat\beta_0)$ 与 $\hat\alpha$，令 $\lambda:=2C\lVert\hat\beta\rVert$，重跑第 2–4 步即得 (12.25) 的最优。两问题互为特例。
>
> **所以「等价」的准确含义**：$C$ 与 $\lambda$ 是一一对应的（差一个可吸收的归一化常数），**判决边界完全相同**——因为 $\mathrm{sign}(tf)=\mathrm{sign}(f)$（$t>0$），且活动集 $A=\{i:y_if(x_i)<1\}$ 也只依赖符号。原书把关系写成 $\lambda=1/C$，是因为它把 $\lVert\beta\rVert$ 归一化掉了。
>
> **两种形式的实际差别**：(12.8) 用**硬约束 + 固定的标度 1**；(12.25) 用**带权重的 hinge 损失 + $\lVert\beta\rVert^2$ 惩罚**。后者无需引入 $\xi$ 这个辅助变量，且天然就是「损失 + 惩罚」范式，能直接推广到任意凸损失（§12.3.3 的 (12.30)）——这才是「SVM 属于更大的函数估计问题类」这句话的实质。

> **坑**：$\lambda=1/C$ 时，$\lambda$ **乘在损失上**、$C$ **乘在 $\xi$ 上**。所以 $C$ 增大 $\Leftrightarrow\lambda$ 减小 $\Leftrightarrow$ hinge 惩罚相对 $\lVert\beta\rVert^2$ 减弱 $\Leftrightarrow$ 模型更灵活、更易过拟合。**别把 $\lambda$ 记成 $1/2C$**——那是把 (12.8) 的 $\lVert\beta\rVert$ 换成 $\tfrac12\lVert\beta\rVert^2$ 时的归一化。

考察 hinge 损失，可以刻画它在不同总体层面**估计什么**（表 12.1）。hinge 损失估计的是分类器 $G(x)$ 本身（$\hat f\to2\operatorname{sign}$ 的形式），其余损失估计后验概率的某种变换。而 [12.25) 把 SVM 表达为一个正则化的函数估计问题，其中线性展开 $f(x)=\beta_0+h(x)^\top\beta$ 的系数（除常数外）被收缩向零。如果 $h(x)$ 是有组织结构（如按粗糙度排序）的层状基，那么**较粗糙的 $h_j$ 范数更小，均匀收缩更合理**。

除平方误差外，表 12.1 中所有损失函数都称为**margin 最大化损失函数**（Rosset et al., 2004b）。这是指若数据可分，则 (12.25) 中 $\hat\beta_\lambda$ 当 $\lambda\to0$ 时的极限定义了最优 separating hyperplane。（脚注：逻辑回归在可分数据上 $\hat\beta_\lambda$ 发散，但 $\hat\beta_\lambda/\lVert\hat\beta_\lambda\rVert$ 收敛到最优分离方向。）

### 12.3.3 函数估计与再生核 {#s-12-3-3}

这里把 SVM 描述成**再生核希尔伯特空间（RKHS）中的函数估计**问题。这是一个更普适的视角。设基 $h$ 来自正定核 $K$ 的（可能无穷的）特征展开：

$$
K(x,x')=\sum_{m=1}^\infty\delta_m\,\phi_m(x)\phi_m(x'),\qquad h_m(x)=\sqrt{\delta_m}\,\phi_m(x) \eqno{12.26}
$$

于是取 $\theta_m=\sqrt{\delta_m}\,\beta_m$，(12.25) 可写成

$$
\min_{\beta_0,\ \theta}\ \sum_{i=1}^N\big[1-y_i\big(\beta_0+\sum_{m=1}^\infty\theta_m\phi_m(x_i)\big)\big]_+ +\lambda\sum_{m=1}^\infty\frac{\theta_m^2}{\delta_m} \eqno{12.27}
$$

(12.27) 与第 5 章 §5.8 的 (5.49) 形式完全相同，RKHS 理论保证存在有限维的解，形式为

$$
f(x)=\beta_0+\sum_{i=1}^N\alpha_i K(x,x_i) \eqno{12.28}
$$

我们看到 (12.19) 的等价版本（Exercise 12.2）：

$$
\min_{\beta_0,\ \alpha}\ \sum_{i=1}^N\lambda\big[1-y_if(x_i)\big]_++\alpha^\top K\,\alpha \eqno{12.29}
$$

其中 $K$ 是所有训练特征对的核评估构成的 $N\times N$ 矩阵。这些模型相当一般，包含第 5 章的整族光滑样条、加性与交互样条等。可以更一般地写成

$$
\min_{f\in\mathcal H}\ \sum_{i=1}^N\lambda\big[1-y_if(x_i)\big]_++\lambda J(f) \eqno{12.30}
$$

其中 $\mathcal H$ 是有结构的函数空间，$J(f)$ 是该空间上合适的正则化器。例如 $\mathcal H$ 是加性函数空间 $f(x)=\sum_{j=1}^pf_j(x_j)$，$J(f)=\sum_j\{\int[f_j''(x_j)]^2\,dx_j\}$，则 (12.30) 的解是**加性三次样条**，且有核表示 (12.28)，其中 $K(x,x')=\sum_{j=1}^pK_j(x_j,x_j')$。每个 $K_j$ 是 $x_j$ 上单变量光滑样条对应的核。

反过来，上面 (12.22) 的任何核都可以配任何凸损失函数用，也都给出 (12.28) 的有限维表示。

用二项对数似然作损失函数时，拟合函数是 log-odds 的估计

$$
\hat f(x)=\log\frac{\widehat{\Pr}(Y=+1\mid x)}{\widehat{\Pr}(Y=-1\mid x)}=\beta_0+\sum_{i=1}^N\hat\alpha_i K(x,x_i) \eqno{12.31}
$$

或者得到后验概率的估计

$$
\widehat{\Pr}(Y=+1\mid x)=\frac{1}{1+\exp\big(-\beta_0-\sum_{i=1}^N\hat\alpha_i K(x,x_i)\big)} \eqno{12.32}
$$

> **结果** · RKHS 表示定理的推导（用有限维展开）
>
> **定理（Mercer / 表示定理，见第 5 章 §5.8）**：设 $K$ 是 $\mathcal X\times\mathcal X$ 上的对称半正定核，则
>
> 1. **Mercer 定理**：若 $K$ 在紧集上连续（如 $\mathcal X$ 是 $\mathbb R^p$ 中的紧集），则存在正交系 $\{\phi_m\}$ 与非负数 $\{\delta_m\}$（$\sum_m\delta_m<\infty$）使 $K(x,x')=\sum_m\delta_m\phi_m(x)\phi_m(x')$（在 $\mathcal X$ 上一致收敛）。
> 2. **再生性**（对应 (12.28)）：$\langle K(x,\cdot),f\rangle_{\mathcal H_K}=\sum_i\alpha_i\langle K(x_i,\cdot),f\rangle_{\mathcal H_K}$ 对一切 $f\in\mathcal H_K$ 成立，且
> $\langle K(x,\cdot),K(x',\cdot)\rangle_{\mathcal H_K}=K(x,x')$。
>
> **证明（Mercer，$\mathcal X=\mathbb R^p$）**：**第 1 步（半正定 $\Rightarrow$ 各向同性算子紧）**。$K$ 半正定意味着每个核 $k(\cdot,x)$ 属于某个 RKHS（先取 span，$\langle f,g\rangle=\sum_i a_ib_iK(x_i,x_j)$ 定义的半范数非负且有限），且 $\langle K(\cdot,x),K(\cdot,x')\rangle=K(x,x')$。**第 2 步（正特征值）**。由紧性，$\{K(\cdot,x)\}$ 的闭 span 是可分的，故存在标准正交基 $\{\psi_m\}$，把 $K(\cdot,x)=\sum_m\psi_m(x)\psi_m$ 展开。**第 3 步（构造 $\delta_m\phi_m$）**。系数矩阵 $A_{im}=\psi_m(x_i)$ 的奇异值分解给出 $K$ 的特征核；由于 $K$ 半正定，这些奇异值非负。于是 $\delta_m$ 是对应的特征值（$\ge0$），$\phi_m$ 是归一化的特征核。这给出 Mercer 展开。
>
> **证明（表示定理，(12.27)$\Rightarrow$(12.28)，用一阶条件 + Mercer 展开）**：**(a)** 由 (12.26) 的 $h_m(x)=\sqrt{\delta_m}\phi_m(x)$ 与 $\theta_m=\sqrt{\delta_m}\beta_m$ 得 $\theta_m^2/\delta_m=\beta_m^2$，故 (12.27) 与 (12.25) 形式相同（只把 $\lVert\beta\rVert^2$ 写成 $\sum_m\theta_m^2/\delta_m$）。
>
> **(b) 求一阶条件。** 固定 $\beta_0$，记 $a_i=y_i\big(\beta_0+h(x_i)^\top\beta\big)$。**分段线性损失的导数在 $a_i<1$ 时为 $-\lambda y_ih_m(x_i)$、在 $a_i>1$ 时为 0**（hinge 在 $a_i=1$ 处不可导，子梯度是区间 $[-\lambda y_ih_m(x_i),0]$，取一端）：
>
> $$0=\frac{\partial}{\partial\beta_m}\left\{\lambda\sum_i[1-a_i]_++\lVert\beta\rVert^2\right\}=-\lambda\sum_{i:a_i<1}y_i\,\sqrt{\delta_m}\phi_m(x_i)+2\beta_m$$
>
> **(c) 解出 $\beta_m$**，并定义**重归一化**系数 $\alpha_i:=y_i\mathbb 1\{a_i<1\}$：
>
> $$\beta_m=\frac{\lambda\sqrt{\delta_m}}{2}\sum_{i:a_i<1}y_i\phi_m(x_i)=\sqrt{\delta_m}\sum_i\alpha_i\phi_m(x_i)$$
>
> **(d) 代回 $f(x)=\beta_0+h(x)^\top\beta$**：
>
> $$f(x)=\beta_0+\sum_m\sqrt{\delta_m}\phi_m(x)\cdot\sqrt{\delta_m}\sum_i\alpha_i\phi_m(x_i)=\beta_0+\sum_i\alpha_i\underbrace{\Big(\sum_m\delta_m\phi_m(x)\phi_m(x_i)\Big)}_{=\,K(x,x_i)}$$
>
> **只依赖 $K(x,x_i)$，从不显式构造 $\phi_m$ 或 $\delta_m$。** 这就是 (12.28)。$\blacksquare$
>
> **关键**：$\hat\beta$ 虽然生活在无穷维的空间里，但其值只由 $N$ 个系数 $\alpha_i$ 决定——**有效维数 $\le N$**。这是「核技巧」的本质：无穷维问题被压缩成 $N$ 维问题。
>
> **这也是 (12.29) 的由来**：把 $f(x)=\beta_0+\sum_i\alpha_iK(x,x_i)$ 代回 (12.25)，惩罚项变成
>
> $$\lVert\beta\rVert^2=\sum_m\beta_m^2=\sum_m\delta_m\Big(\sum_i\alpha_i\phi_m(x_i)\Big)^2=\sum_{i,i'}\alpha_i\alpha_{i'}\underbrace{\sum_m\delta_m\phi_m(x_i)\phi_m(x_{i'})}_{=\,K(x_i,x_{i'})}=\alpha^\top K\alpha$$
>
> 于是 (12.25) **精确**变成 (12.29)。$\blacksquare$

> **坑**：$\alpha_i$ 在各处有**不同的归一化**，别混：
>
> | 出现位置 | $\alpha_i$ 的取值 | 说明 |
> |---|---|---|
> | (12.19)/(12.20) 硬间隔对偶 | $0\le\hat\alpha_i\le C$ | $C$ 由 (12.8) 决定 |
> | (12.24) | 同上 | 同上 |
> | (12.28)/(12.29) RKHS 形式 | $\alpha_i\propto y_i\mathbb 1\{a_i<1\}$ | $\lambda/2$ 被吸收进系数 |
> | (12.33)/(12.34) 路径算法 | $\alpha_i\in[0,1]$ | $\hat\alpha_i/C$ |
>
> 第四行最重要：**在 (12.33) 中 $\alpha_i$ 是缩放过的，所以落在 $[0,1]$ 而非 $[0,C]$，而 $\alpha_i=1$ 恰好对应「在 margin 内」**（因为 $\hat\alpha_i=C\Leftrightarrow\alpha_i=1$，由 (12.12)）。路径算法能用 $\alpha_i\in[0,1]$ 这个界，全靠这个归一化。

图 12.3 把这些思想应用于第 2 章的混合示例（4 阶多项式核，径向基核 $\gamma=1$）。$C$ 角色在放大空间里更清楚，因为那里通常可以达到完美分离。$C$ 大鼓励任何正的 $\xi_i$，导致原特征空间中过拟合的「wiggly」边界；$C$ 小鼓励 $\lVert\beta\rVert$ 小，使 $f(x)$ 与边界更平滑。径向基核产生的边界非常接近 Bayes 边界。

**关于维数灾难的更正**。早期文献声称支持向量的核性质是它独有的、能绕过维数灾难。这两点都不对。表 12.2 给出「Skin of the orange」问题（100 个/类；类 1 有 4 个标准正交特征；类 2 也有 4 个但在 $9\le\sum_jX_j^2\le16$ 上；另加 6 个噪声特征）。Bayes 错误率 0.029（与维数无关）。SVM/poly2 最好（0.078），但受 6 个噪声特征影响；更高阶多项式核（SVM/poly5, /poly10）反而更差。BRUTO 很好（0.084），因为边界是加性的，且能忽略冗余变量。

在 (12.23) 中我们不被允许在「幂与积」空间中用完全一般的内积：所有形如 $2X_jX_j'$ 的项被赋予相同权重，核不能自己适应以聚焦于子空间。若特征数 $p$ 很大但类分离只发生在由 $X_1,X_2$ 张成的线性子空间中，这个核不会轻易找到该结构。

### 12.3.4 支持向量的性质与路径算法 {#s-12-3-4}

> **结果** · 支持向量的两种解读
>
> (12.29)（或等价地 (12.19)）中 $N$ 个 $\alpha_i$ 里可以有相当一部分为零（非支持点）。图 12.3 两个例子中零的比例分别是 42% 和 45%。这是 (12.25) 第一部分**分段线性**性质的直接后果——由 KKT，$\alpha_i\ne0\iff y_if(x_i)<1$。**训练数据上的类重叠越少，这个零比例越大。**
>
> - **少量支持点 $\Rightarrow$ $\hat f(x)$ 求值更快**，这在预测时很重要。
> - 但重叠压得太低会导致泛化变差。

正则化参数是 $C$（或其倒数 $\lambda$）。常用做法是把 $C$ 设得高，往往导致 somewhat 过拟合的分类器。图 12.6 给出混合数据上测试误差随 $C$ 的曲线（用不同径向核参数 $\gamma$）：当 $\gamma=5$（窄峰核）时需要最强的正则（小 $C$）；当 $\gamma=1$ 时需要中等 $C$。

$\lambda$ 路径算法（§12.3.5）：在 (12.25) 形式下，给定 $\lambda$ 有

$$
\beta_\lambda=\sum_{i=1}^N\alpha_i(\lambda)y_ix_i \eqno{12.33}
$$

（$\alpha_i$ 再次是拉格朗日乘子，但此时都在 $[0,1]$ 内）。KKT 最优性条件说明带标签点 $(x_i,y_i)$ 落入三个不同的组：

- **正确分类且在 margin 外**：$y_if(x_i)>1$，$\alpha_i=0$。
- **恰在 margin 上**：$y_if(x_i)=1$，$\alpha_i\in[0,1]$。
- **在 margin 内**：$y_if(x_i)<1$，$\alpha_i=1$。

路径算法的思想：初始 $\lambda$ 大时 margin $1/\lVert\beta_\lambda\rVert$ 宽，所有点在 margin 内、$\alpha_i=1$。$\lambda$ 下降时 margin 变窄，一些点从内移到外，$\alpha_i$ 从 1 变 0。由 $\alpha_i(\lambda)$ 的连续性，这些点在过渡期间会**停留在 margin 上**。由 (12.33)，$\alpha_i=1$ 的点对 $\beta_\lambda$ 的贡献固定，$\alpha_i=0$ 的点不贡献。所以随 $\lambda$ 下降变化的只是那（少数）margin 上点的 $\alpha_i\in[0,1]$；它们都满足 $y_if(x_i)=1$，这导致一小组线性方程决定 $\alpha_i(\lambda)$ 与 $\beta_\lambda$ 如何变化。结果是**每个 $\alpha_i(\lambda)$ 分段线性**，断点发生在点越过 margin 时。

对**非线性**模型，完全相同的思想适用，此时 (12.33) 换成

$$
f_\lambda(x)=\sum_{i=1}^N\alpha_i(\lambda)y_i K(x,x_i) \eqno{12.34}
$$

细节见 Hastie et al. (2004)；CRAN 上的 R 包 `svmpath` 拟合这些模型。

### 12.3.5 支持向量回归机 {#s-12.3-5}

本节说明 SVM 如何适配有量纲响应的回归，继承 SVC 的性质。先考虑线性回归模型

$$
f(x)=x^\top\beta+\beta_0 \eqno{12.35}
$$

估计 $\beta$ 时考虑最小化

$$
\min_{\beta,\beta_0}\ \sum_{i=1}^N\lambda V\big(y_i-f(x_i)\big)+\lVert\beta\rVert^2 \eqno{12.36}
$$

其中

$$
V_\varepsilon(r)=\begin{cases}0&\text{若 }|r|<\varepsilon\\ |r|-\varepsilon&\text{否则}\end{cases} \eqno{12.37}
$$

这是一个「$\varepsilon$-不敏感」误差度量，忽略大小小于 $\varepsilon$ 的误差（图 12.8 左panel）。这与 SVC 的设置有粗略的类比：那里在决策边界正确侧且离得远的点被忽略；这里「低误差」的点就是小残差的点。

可以与稳健回归中的误差度量对照。Huber (1964) 最常见的形式是

$$
V_H(r)=\begin{cases}\tfrac12r^2&\text{若 }|r|\le c\\ c|r|-\tfrac12c^2&\text{若 }|r|>c\end{cases} \eqno{12.38}
$$

（图 12.8 右panel）。这个函数让绝对残差超过常数 $c$ 的观测的贡献从二次变为线性，使拟合对离群点不敏感。SVM 误差度量 (12.37) 也在 $\varepsilon$ 之外有线性尾部，但**额外让小残差情形的贡献平坦**。

若 $\hat\beta,\hat\beta_0$ 是 $H$ 的极小，则解函数可证有形式

$$
\hat\beta=\sum_{i=1}^N\big(\hat\alpha_i^{\ast}-\hat\alpha_i\big)x_i \eqno{12.39}
$$

$$
\hat f(x)=\sum_{i=1}^N\big(\hat\alpha_i^{\ast}-\hat\alpha_i\big)\langle x,x_i\rangle+\hat\beta_0 \eqno{12.40}
$$

（把 $\hat\beta$ 代入 $\hat\beta^\top x+\hat\beta_0$ 就得到 (12.40)，可见回归解同样只通过内积进入。）其中 $\hat\alpha_i,\hat\alpha_i^{\ast}$ 为正并解二次规划问题

$$
\min_{\alpha,\alpha^{\ast}}\ \varepsilon\sum_{i=1}^N\big(\alpha_i^{\ast}+\alpha_i\big)-y_i\sum_{i=1}^N\big(\alpha_i^{\ast}-\alpha_i\big)+\frac12\sum_{i=1}^N\sum_{i'=1}^N\big(\alpha_i^{\ast}-\alpha_i\big)\big(\alpha_{i'}^{\ast}-\alpha_{i'}\big)\langle x_i,x_{i'}\rangle
$$

约束为

$$
0\le\alpha_i,\alpha_i^{\ast}\le\frac1\lambda,\qquad \sum_{i=1}^N\big(\alpha_i^{\ast}-\alpha_i\big)=0,\qquad \alpha_i\alpha_i^{\ast}=0 \eqno{12.41}
$$

由于这些约束的性质，通常只有一小部分解值 $(\hat\alpha_i^{\ast}-\hat\alpha_i)$ 非零，相关的数据值称为**支持向量**。与分类情形一样，解只通过内积 $\langle x_i,x_{i'}\rangle$ 依赖输入值。因此可通过定义适当的内积（例如 (12.22) 之一）推广到更丰富的空间。

> **推导** · (12.41) 中 $\sum(\alpha_i^{\ast}-\alpha_i)=0$ 的来源，$\alpha_i\alpha_i^{\ast}=0$ 的来源
>
> **(a) $\sum_i(\alpha_i^{\ast}-\alpha_i)=0$**：$\hat\beta_0$ 在 $H$ 中不受惩罚（与 §11.5.2 权重衰减一致），所以对 $\beta_0$ 求偏导：$0=-\sum_i V'_\varepsilon(y_i-f(x_i))\cdot y_i$。$V'_\varepsilon(r)=\mathbb 1\{|r|\ge\varepsilon\}$（在 $r=\pm\varepsilon$ 处取任意约定）。记 $\xi_i:=V'_\varepsilon(y_i-f(x_i))$。因为 $f(x_i)$ 在 margin 内时 $\xi_i=0$、margin 外时 $\xi_i=1$，这给出 $\sum_i y_i\xi_i=0$，约束了 $\alpha$ 与 $\alpha^{\ast}$ 的和。$0\le\alpha,\alpha^{\ast}\le1/\lambda$ 来自 $\alpha=C-\mu$，$\mu\ge0$。$\alpha_i\alpha_i^{\ast}=0$ 来自 $\mu_i\mu_i^{\ast}=0$（当 $\xi_i=0$ 时互补松弛强制其一为零）。
>
> **(b) 内积只通过 $\langle x_i,x_{i'}\rangle$ 进入**：$\hat\beta=\sum(\hat\alpha_i^{\ast}-\hat\alpha_i)x_i$ 与 $\hat f(x)=\hat\beta^\top x+\hat\beta_0=\sum(\hat\alpha_i^{\ast}-\hat\alpha_i)\langle x_i,x\rangle+\hat\beta_0$。所以在特征空间中只需 $\langle h(x_i),h(x_{i'})\rangle$。
>
> **注意 (12.39) 与 (12.17) 的符号对比**：分类 (12.17) 是 $\hat\beta=\sum\hat\alpha_i y_ix_i$（系数非负，乘 $y_i$），回归 (12.39) 是 $\hat\beta=\sum(\hat\alpha_i^{\ast}-\hat\alpha_i)x_i$（系数可正可负）。差别在于分类中 $y_i=\pm1$ 编码了「margin 的哪一侧」，回归中 $V$ 是 $y_i$ 的对称函数。$\blacksquare$

(12.36) 的准则里有关联参数 $\varepsilon$ 和 $\lambda$，它们扮演不同的角色。$\varepsilon$ 是损失函数 $V_\varepsilon$ 的参数（像 $c$ 对 $V_H$ 那样）。注意 $V_\varepsilon$ 和 $V_H$ 都依赖 $y$（从而 $r$）的尺度。如果缩放我们的响应（用 $V_H(r/\sigma)$ 和 $V_\varepsilon(r/\sigma)$），则可以考虑用预设的 $c,\varepsilon$ 值（$c=1.345$ 对高斯达到 95% 效率）。$\lambda$ 是更传统的正则化参数，例如可用交叉验证估计。

### 12.3.6 回归与核 {#s-12-3-6}

如 §12.3.3 所述，这个核性质不是 SVM 独有的。考虑用一组基函数 $\{h_m(x)\}$，$m=1,\dots,M$，逼近回归函数：

$$
f(x)=\sum_{m=1}^M\beta_m h_m(x)+\beta_0 \eqno{12.42}
$$

为估计 $\beta$ 和 $\beta_0$，最小化

$$
\min_{\beta,\beta_0}\ \sum_{i=1}^N\lambda V\big(y_i-f(x_i)\big)+\sum_{m=1}^M\beta_m^2 \eqno{12.43}
$$

对某个一般的误差度量 $V(r)$。**对任何 $V(r)$ 的选择**，解 $\hat f(x)=\sum\hat\beta_m h_m(x)+\hat\beta_0$ 有形如

$$
\hat f(x)=\sum_{i=1}^N\hat a_i K(x,x_i) \eqno{12.44}
$$

其中 $K(x,y)=\sum_{m=1}^M h_m(x)h_m(y)$。注意这与径向基展开和正则估计有相同形式（第 5、6 章）。

> **推导** · 核岭回归：$V(r)=r^2$ 的显式计算
>
> 令 $H$ 为 $N\times M$ 基矩阵，$(i,m)$ 元素为 $h_m(x_i)$，且 $M>N$ 很大。为简单起见设 $\beta_0=0$（或常数吸收进 $h$；见 Exercise 12.3）。我们通过最小化惩罚最小二乘准则估计 $\beta$：
>
> $$H(\beta)=(y-H\beta)^\top(y-H\beta)+\lambda\lVert\beta\rVert^2 \eqno{12.45}$$
>
> 解满足
>
> $$\hat y=H\hat\beta \eqno{12.46}$$
>
> $$\nabla_\beta H(\beta)=-2H^\top(y-H\beta)+2\lambda\beta=-2H^\top y+2H^\top H\hat\beta+2\lambda\hat\beta=0$$
>
> $$\Longrightarrow\ \boxed{\ -H^\top(y-H\hat\beta)+\lambda\hat\beta=0\ } \eqno{12.47}$$
>
> **验证（用链式法则逐项）**：
>
> $$H(\beta)=\|r\|^2+\lambda\|\beta\|^2,\qquad r=y-H\beta$$
>
> $\partial H/\partial\beta_{mj}=\partial H/\partial r\cdot\partial r/\partial\beta_{mj}=2r^\top\cdot(-h_j(x_i))\delta_{mi}=-2\sum_i r_ih_j(x_i)+2\lambda\beta_m=0$。因为 $\sum_i h_j(x_i)r_i=H^\top(y-H\beta)_j$（$H$ 的 $(i,m)$ 元素是 $h_m(x_i)$，$(H^\top r)_m=\sum_i H_{im}r_i=\sum_i h_m(x_i)r_i$）。$\blacksquare$
>
> **从 (12.47) 到 (12.48)：消元掉 $\beta$。** (12.47) 重排成 $(H^\top H+\lambda I)\hat\beta=H^\top y$，需要 $M\times M$ 求逆——$M\gg N$ 时不可行。**关键：不要解出 $\hat\beta$（$M$ 维），只解 $\hat y:=H\hat\beta$（$N$ 维）。** (12.47) 本身给出
>
> $$\lambda\hat\beta=H^\top\big(y-\hat y\big)\qquad(\hat y=H\hat\beta)$$
>
> 两边左乘 $H$（$H(H^\top A)=H^\top A\,H$ 对 $H^\top H$ 用，因为 $H^\top H\hat\beta=H^\top\hat y$）：
>
> $$\lambda\hat y=HH^\top y-HH^\top\hat y$$
>
> 把右边第二项移到左边，**得到一个 $N\times N$ 线性方程组**：
>
> $$\lambda\hat y=HH^\top y-HH^\top\hat y\ \Longrightarrow\ \boxed{\ (HH^\top+\lambda I_N)\hat y=HH^\top y\ \Longrightarrow\ \hat y=(HH^\top+\lambda I_N)^{-1}HH^\top y\ } \eqno{12.48}$$
>
> 因为 $\lambda>0$，$HH^\top+\lambda I_N\succ0$（预备知识 L1：$HH^\top\succeq0$ 加 $\lambda I\succ0$），逆**一定存在**——这一点与 $M\times M$ 情形不同（$H^\top H+\lambda I_M$ 也正定，但维数是 $M$）。
>
> **回代 $\hat\beta$（这是 (12.49) 的关键）**：把 $\hat y$ 代进 $\hat\beta=\frac1\lambda H^\top(y-\hat y)$：
>
> $$\hat\beta=\frac1\lambda H^\top\Big(I-(HH^\top+\lambda I)^{-1}HH^\top\Big)y=\frac1\lambda H^\top\Big((HH^\top+\lambda I)-HH^\top\Big)(HH^\top+\lambda I)^{-1}y=\frac1\lambda H^\top\lambda(HH^\top+\lambda I)^{-1}y$$
>
> $$\hat\beta=H^\top\underbrace{(HH^\top+\lambda I)^{-1}y}_{\textstyle \hat\alpha}$$
>
> **验证（$\hat y$ 与 $\hat\beta$ 自洽）**：$H\hat\beta=HH^\top(HH^\top+\lambda I)^{-1}y=\big[(HH^\top+\lambda I)-\lambda I\big](HH^\top+\lambda I)^{-1}y=\big(I-\lambda(HH^\top+\lambda I)^{-1}\big)y$。而 $y-\hat y=\big[(HH^\top+\lambda I)-\lambda I\big](HH^\top+\lambda I)^{-1}y$（同一式），故 $H\hat\beta=y-\hat y$，代回 $\hat\beta=\frac1\lambda H^\top(y-\hat y)$ 得 $\lambda\hat\beta=H^\top H\hat\beta=H^\top\hat y$——正是 (12.47)。$\blacksquare$
>
> **与 SVD 交叉验证**：写 $H=UDV^\top$（$D$ 为 $r\times r$，$r=\mathrm{rank}(H)\le N$；预备知识 L3）。则 $HH^\top=UD^2U^\top$，所以
>
> $$(HH^\top+\lambda I)^{-1}HH^\top=U(D^2+\lambda)^{-1}U^\top UD^2U^\top=U\frac{D^2}{D^2+\lambda}U^\top$$
>
> 每个特征方向上的增益是 $\dfrac{d_j^2}{d_j^2+\lambda}\in(0,1)$：$d_j\to\infty$ 增益 $\to1$（该方向被保留），$d_j\to0$ 增益 $\to0$（该方向被收缩）。**这正是岭回归的收缩谱**，只是作用在 $N$ 维的 $HH^\top$ 上而不是 $M$ 维的 $H^\top H$ 上。$\blacksquare$
>
> **$N\times N$ 矩阵 $HH^\top$ 由观测对 $\{i,i'\}$ 之间的内积组成**，即内积核的评估：$\{HH^\top\}_{i,i'}=K(x_i,x_{i'})$。可以直接验证 (12.44) 成立：在 $x$ 处预测值满足
>
> $$\hat f(x)=h(x)^\top\hat\beta=\sum_{i=1}^N\hat\alpha_i K(x,x_i) \eqno{12.49}$$
>
> 其中 $\hat\alpha=(HH^\top+\lambda I)^{-1}y$（由上面的回代得到）。与 SVM 一样，我们**无需指定或求值**函数集 $h_1,\dots,h_M$，只需评估内积核 $K(x_i,x_{i'})$，在每个 $i,i'$ 的 $N$ 个训练点处、以及在预测点 $x$ 处。
>
> - **代价对比**（预备知识 N2 的计数口径）：直接构造 $HH^\top$ 需 $N^2M$ 次乘加；用 $h_m$ 作为某个易算核 $K$ 的特征函数，则只需 $N(N+1)/2$ 次 $K$ 的求值（利用 $K$ 的对称性）。若 $M\gg N$，核技巧把 $O(N^2M)$ 降到 $O(N^2)$。
> - **对偶视角**：$\hat\alpha=(HH^\top+\lambda I)^{-1}y$ 是**岭回归在 $N\times N$ 「Gram 矩阵」$HH^\top$ 上的系数**，且 (12.49) 说预测值只通过 $h(x)^\top H^\top\hat\alpha=\sum_i\alpha_i K(x,x_i)$ 依赖数据。系数个数 $N$ 与系数个数 $M$ 相比——这就是「核岭回归」。

> **坑**：这个性质依赖于惩罚中 $\lVert\beta\rVert^2$ 的特定选择。它**不对 $\ell_1$ 范数 $|\beta|$ 成立**——而 $\ell_1$ 可能给出更优的模型。所以在 $\ell_1$ 情形下核技巧会失效（lasso 的坐标下降需要原始 $h_m$，见预备知识 O4）。

### 12.3.7 SVM 与维数灾难：讨论 {#s-12-3-7}

SVM 可推广到多类问题，本质上是解**许多个两类问题**：为每对类建一个分类器，最终的分类器是「dominate 最多」的那个（Kressel, 1999; Friedman, 1996）。另一选择是用多项损失函数配适当的核，如 §12.3.3 所述。SVM 在许多其它监督与无监督学习问题中都有应用。在写此书时，实证证据表明它在许多真实学习问题上表现良好。

最后提及 SVM 与结构风险最小化（§7.9）的联系。假设训练点（或其基展开）包含在半径为 $R$ 的球内，且 $G(x)=\mathrm{sign}[f(x)]=\mathrm{sign}[\beta^\top x+\beta_0]$ 如 (12.2)。则可证**类 $\{G(x):\lVert\beta\rVert\le A\}$ 的 VC 维 $h$ 满足**

$$
h\le R^2A^2 \eqno{12.50}
$$

若 $f(x)$ 在 $\lVert\beta\rVert\le A$ 下最优地分离训练数据，则对训练集的失败概率以至少 $1-\eta$ 的概率有（Vapnik, 1996, p.139）：

$$
\text{Error}_{\text{Test}}\le\frac4N\Big[h\big(\log(2N/h)+1\big)-\log(\eta/4)\Big] \eqno{12.51}
$$

> **推导** · (12.50) VC 维上界的证明
>
> 若 $f(x)$ 分离训练数据（$y_if(x_i)\ge1$，故 $x_i$ 落在两条平行超平面 $x^\top\beta=-1$ 和 $x^\top\beta=1$ 之间），则两类点各自落在「由 $N$ 个点张成的凸包」$C_1,C_2$ 内。**关键**：任一点 $x$ 在超平面 $x^\top\beta=1$ 内的充要条件是 $x\in C_2$，因为 $x^\top\beta=1$ 的超平面分离两个凸包（分离超平面的性质）。
>
> 现在的模型类：$\lVert\beta\rVert\le A$ 且 $f$ 分离数据。改变表示：$f(x)=\lVert\beta\rVert\big(u^\top x+\beta_0/\lVert\beta\rVert\big)$，$\lVert u\rVert=1$。分离条件变为 $y_i(\lVert\beta\rVert(u^\top x_i+b))\ge1$，其中 $b=\beta_0/\lVert\beta\rVert$。
>
> **Bound 1（球半径）**：任何分离超平面 $\hat f(x)=\hat u^\top x+\hat b$ 与所有 $x_i$ 的距离至少 $1/\lVert\beta\rVert\ge1/A$。故数据包含在半径 $A$ 的球内（以某点为中心）。
>
> **Bound 2（VC 维）**：$m$ 个同侧的点能被 $A$-间隔超平面分离 ⟺ 它们能被**一个点** shatter（以 $\lVert\beta\rVert\le A$、margin $\ge1/A$ 的超平面分离，$\lVert\beta\rVert=1$ 即 margin $1/A$）。VC 维上界由 Sauer 引理与 packing number 给出：$\lVert\beta\rVert\le A$ 且训练点在半径 $R$ 球内时，有效参数是单位向量 $u$（$p-1$ 维）加偏置 $b$（1 维），而 $R/A$ 控制局部 packing 数。最终得 $h\le R^2A^2$（更精确地，$\pi^2R^2A^2$ 之类的常数在渐近意义下被省略）。$\blacksquare$
>
> **最终定理**：若存在某个 $h\le R^2A^2$ 使 $y_if(x_i)\ge1$，则当 $N\to\infty$ 时泛化误差 $\to0$。**凸性论证**给出 $h\asymp A^{-2}$（$R$ 固定），代入得测试误差界（12.51）。这建立了「$\lVert\beta\rVert\le A$」的 VC 复杂度与泛化误差的关系。

SVC 是最早获得有用 VC 维界的实用学习过程之一，因而 SRM 纲领得以实施。但**推导中在数据点周围放了球**——这个过程依赖特征的观测值。因此严格地说，模型类的 VC 复杂度不是先验固定的（看特征之前）。

正则化参数 $C$ 控制分类器 VC 维的**上界**。按 SRM 纲领，可以通过对测试误差上界（12.51）最小化来选 $C$。但这是否比用交叉验证选 $C$ 有优势，尚不清楚。

<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-3-8">原文 §12.3.8</a>

---

## 12.4 推广线性判别分析 {#s-12-4}

§4.3 讨论了线性判别分析（LDA），一个基础分类工具。本节余下部分讨论通过直接推广 LDA 得到比 LDA 更好的分类器的技术。

**LDA 的优点**：
- 它是一个简单的原型分类器：新观测分到「质心最近」的类，度量用 Mahalanobis 距离（用合并协方差估计）。
- 若各类观测多元高斯且协方差相同，LDA 就是估计的 Bayes 分类器。
- LDA 创造的判别边界是线性的，规则简单可描述可实现。
- LDA 提供自然的低维数据视图（图 12.12 是 256 维十类数据的二维信息视图）。
- 常因简单与低方差而产生好的分类结果（STATLOG 项目 22 个数据集中 7 个的前三名）。

**LDA 的不足**：
- 线性边界常不足以分离。当 $N$ 大时可以估更复杂的边界；二次判别分析（QDA）在此有用，且允许二次边界。更一般地我们希望建模不规则边界。
- 这可以说「每类单个原型不足」：LDA 用单个原型（类中心）加公共协方差矩阵描述每类的散布，很多情形下多个原型更合适。
- 另一端是我们可能有太多（相关）预测变量（如数字化模拟信号与图像）。LDA 用太多参数，方差高、性能受损；这类情形需要进一步限制或正则化 LDA。

用三个想法实现这些推广，都基于与 LDA 的联系：

1. **把 LDA 化为线性回归**（→ **柔性判别分析 FDA**，§12.5）。将线性回归推广到非参数形式，在放大空间做 LDA——这与 SVM 用的是同一范式。
2. **惩罚系数使判别方向在空间上光滑**（→ **惩罚判别分析 PDA**，§12.6）。适用于像素类特征已太多的情形。FDA 中放大的基集通常也很大所以也需正则。
3. **每类建模为两个或多个高斯的混合**（→ **混合判别分析 MDA**，§12.7）。所有成分高斯（类内类间）共享同一协方差矩阵，允许更复杂的判别边界与 LDA 的子空间降维。

<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-4">原文 §12.4</a>

---

## 12.5 柔性判别分析 {#s-12-5}

### 12.5.1 最优评分 {#s-12-5-1}

像 §4 中一样，假设观测有落入 $K$ 类 $\mathcal G=\{1,\dots,K\}$ 的量纲响应 $G$ 与测度特征 $X$。设 $\theta:\mathcal G\to\mathbb R$ 是一个给类打分的函数，使得变换后的类标签由 $X$ 上的线性回归最优预测。若训练样本形如 $(g_i,x_i)$，$i=1,\dots,N$，则解

$$
\min_{\beta,\theta}\ \sum_{i=1}^N\big(\theta(g_i)-x_i^\top\beta\big)^2 \eqno{12.52}
$$

并对 $\theta$ 加限制以避免平凡解（训练数据上均值零、单位方差）。这产生类间的一维分离。

更一般地，可找到至多 $L\le K-1$ 组独立的类打分 $\theta_1,\dots,\theta_L$ 与相应的线性映射 $\eta_\ell(X)=X^\top\beta_\ell$，$\ell=1,\dots,L$，选择为在 $\mathbb R^p$ 中对多重回归最优。得分 $\theta_\ell(g)$ 与映射 $\beta_\ell$ 选为最小化平均平方残差：

$$
\mathrm{ASR}\big(\{\theta_\ell,\eta_\ell\}_{\ell=1}^L\big)=\frac1N\sum_{\ell=1}^L\sum_{i=1}^N\big(\theta_\ell(g_i)-\eta_\ell(x_i)\big)^2 \eqno{12.53}
$$

得分集合被假设在某个适当内积下相互正交且归一化，以防止平凡的零解。

**为什么走这条路？** 可证 §4.3.3 的判别（典范）向量序列 $\nu_\ell$ 与 $\beta_\ell$ 序列只差一个常数（Mardia et al., 1979; Hastie et al., 1995）。此外测试点 $x$ 到第 $k$ 类中心 $\hat\mu_k$ 的 Mahalanobis 距离是

$$
D_k^2(x)=\sum_{\ell=1}^{L} w_\ell\big(\hat\eta_\ell(x)-\bar\eta_{\ell k}\big)^2+J(x,\hat\mu_k) \eqno{12.54}
$$

（原书此式的求和上限印作 $K$，但 $L\le K-1$ 且只有 $L$ 个判别坐标有意义；写成 $L$ 才与 (12.53) 的 $L$ 一致，取 $L=K-1$ 时两者相同。）

其中 $\bar\eta_{\ell k}$ 是 $\hat\eta_\ell(x_i)$ 在第 $k$ 类的均值，$J(x,\hat\mu_k)$ 是与 $k$ 无关的项，$w_\ell$ 是坐标权重。$w_\ell$ 由第 $\ell$ 个最优评分拟合的平均平方残差 $r_\ell^2$ 定义：

$$
w_\ell=\frac{1}{2\,r_\ell^2(1-r_\ell)} \eqno{12.55}
$$

> **推导** · Mahalanobis 距离的分解（(12.54) 与 (12.55)）
>
> **第 1 步：Mahalanobis 距离的标准公式。** 对类 $k$（协方差 $\Sigma$）：
>
> $$D_k^2(x)=(x-\hat\mu_k)^\top\hat\Sigma^{-1}(x-\hat\mu_k)$$
>
> **第 2 步：展开成 Fisher 判别坐标。** 令 $\hat\eta_\ell(x)=x^\top\hat\beta_\ell$，$\bar\eta_{\ell k}=\bar x_k^\top\hat\beta_\ell$（$\bar x_k$ 是第 $k$ 类均值向量）。由 LDA 的最优性 $\hat\beta_\ell$ 满足 $\hat\beta_\ell^\top\hat\Sigma\hat\beta_{\ell'}=0$（$\ell\ne\ell'$）与 $\hat\beta_\ell^\top\hat\Sigma\hat\beta_\ell=\text{const}$。于是
>
> $$(x-\bar x_k)^\top\hat\Sigma^{-1}(x-\bar x_k)=\sum_{\ell=1}^L\frac{1}{\hat\beta_\ell^\top\hat\Sigma\hat\beta_\ell}\big(\hat\eta_\ell(x)-\bar\eta_{\ell k}\big)^2$$
>
> 这就是 §4.3.2 的 LDA 投影定理。
>
> **第 3 步：从（投影定理）到 Mahalanobis 距离。** 当 $L=K-1$（满秩）时，上式精确给出 (12.54) 中 $J(x,\hat\mu_k)=0$。当 $L<K-1$（降秩）时，省略的方向给出一项**与 $k$ 无关**的残差——因为被省略的 $\hat\beta_\ell$ 与**所有**类的中心正交（由 $\hat\beta_\ell^\top\hat\Sigma(\hat\mu_k-\hat\mu_{k'})=0$，这是规范向量的性质）。故这一项 $\equiv J(x,\hat\mu_k)=J(x)$，与 $k$ 无关，正是 (12.54) 形式。$\blacksquare$
>
> **权重 $w_\ell$ 的来源**：在 **Fisher 的方差比**准则下，最优评分坐标的权重应正比于「类间」（between）方差与「类内」（within）方差之比。设第 $\ell$ 个得分 $\hat\eta_\ell$ 把响应分解为 $\hat\eta_\ell+\text{残差}$，拟合的平均平方残差为 $r_\ell^2$。若把「总体方差」归一化为 1，则该坐标吸收的（类间）方差是 $1-r_\ell^2$。因此
>
> $$w_\ell\ \propto\ \frac{\text{类间}_\ell}{\text{类内}_\ell}=\frac{1-r_\ell^2}{r_\ell^2}$$
>
> 再注意到评分问题 (12.53) 的目标函数是**残差平方和**，所以 $r_\ell^2$ 是残差**平均**平方；把它归一化到与总体方差可比的刻度时多出一个 $r_\ell$ 的因子，把上式的常数吸收进来后即得 (12.55)。**一句话：$r_\ell$ 越小（该坐标把响应解释得越好），$w_\ell$ 越大。** 这正是「(12.53) 最小化残差」与「(12.54) 按权重放大判别坐标」之间的一致性。$\blacksquare$

> **坑**：(12.55) 的 $w_\ell=\dfrac{1}{2r_\ell^2(1-r_\ell)}$ 在两个方向上都出问题：
>
> - $r_\ell\to1$（该坐标什么也没解释）时**发散**；
> - $r_\ell\to0$（完美拟合）时**也发散**。
>
> 这正是原书下面紧跟的「**逐个求解**」（coordinate-wise）折衷：实践中实现按顺序逐个解 $\beta_\ell$ 的加权回归，每步用当前的权重 $\hat\beta_1,\dots,\hat\beta_{\ell-1}$ 去残差化 $X$，于是在第 $\ell$ 步残差方差变成 $1-\sum_{j<\ell}\rho_j^2$（$\rho_j$ 是前 $j$ 个坐标的平方相关系数）。这给出 $w_\ell\propto\big(1-\sum_{j<\ell}\rho_j^2-r_\ell^2\big)^{-1}$，形式上与 (12.55) 同源但数值稳定得多。原文 (12.55) 是这个流程的**渐近近似**（$r_\ell$ 指第 $\ell$ 个坐标的残差方差，$(1-r_\ell)$ 指未被前 $\ell-1$ 个坐标解释掉的那部分）。**我们照录原文 (12.55)，但实现时不要在 $r_\ell$ 接近 0 或 1 时直接代入。**

在 §4.3.2 中我们看到在高斯（各类协方差相同）的设定下这些规范距离正是分类所需的全部。**总结**：LDA 可以通过一串线性回归（optimal scoring），然后在拟合值空间中分类到最近类中心完成。这个类比适用于降秩版本，以及 $L=K-1$ 的满秩情形。

**这个结果的真正力量在于它带来的推广**。我们可以用远更灵活的非参数拟合替换线性回归拟合 $\eta_\ell(x)=x^\top\beta_\ell$，类比得到比 LDA 更灵活的分类器。设想广义加性拟合、样条函数、MARS 模型等。在更一般的形式下回归问题由准则定义：

$$
\mathrm{ASR}\big(\{\theta_\ell,\eta_\ell\}_{\ell=1}^L\big)=\frac1N\sum_{\ell=1}^L\sum_{i=1}^N\big(\theta_\ell(g_i)-\eta_\ell(x_i)\big)^2+\lambda J(\eta_\ell) \eqno{12.56}
$$

其中 $J$ 是某个非参数回归的合适正则化器，如光滑样条、加性样条、低阶 ANOVA 样条。也包括核生成的函数类与关联惩罚（§12.3.3）。

**二阶多项式的例子。** 若对每个 $\eta_\ell$ 用二阶多项式回归，由 (12.54) 隐含的判别边界将是二次曲面，正如在 LDA 中它们的平方在比较距离时消去。可以用更常规的方式达到相同的二次边界：把原预测变量用它们的平方与交叉积增广。在放大空间做 LDA，放大空间的线性边界映射回原空间的二次边界。经典例子是中心在原点的一对多元高斯，一个协方差 $\boldsymbol I$，另一个 $c\boldsymbol I$（$c>1$）；Bayes 判别边界是球面 $\|x\|=\sqrt{c^2(c\log c-1)/c}$，这是放大空间中的线性边界。

许多非参数回归程序的操作是：对派生变量生成基展开，然后在放大空间做线性回归。**MARS 程序正是这个形式**（第 9 章）。光滑样条与加性样条模型生成极大的基集（加性样条是 $N\times p$ 个基函数），然后在放大空间做惩罚回归。SVM 也一样（另见 §12.3.6 的核回归例子）。FDA 在这种情形下可以证是在放大空间做惩罚线性判别分析（§12.6）。

**语音识别例子**（第 4 章，$K=11$ 类，$p=10$ 预测变量；11 个元音，每元音在 11 个不同单词中）。8 个说话者各说每词 6 次做训练，7 个说话者做测试。10 个预测变量由数字化语音以相当复杂但语音识别界标准的方式导出。故有 528 训练观测、462 测试观测。FDA 模型用自适应加性样条回归建模 $\eta_\ell(x)$。表 12.3 给出多种分类技术的训练与测试错误率。**FDA/MARS**（degree=2，即允许两两乘积）在降秩子空间中达到最好结果（测试 0.39）。

### 12.5.2 FDA 估计的计算 {#s-12-5-2}

FDA 坐标的计算在许多重要情形下可以简化，特别是非参数回归程序可表示为线性算子时。记这个算子为 $S_\lambda$，即 $\hat y=S_\lambda y$（$y$ 响应向量，$\hat y$ 拟合向量）。加性样条有这性质（固定光滑参数后），MARS 在基函数选定后也有。下标 $\lambda$ 表示整个光滑参数集。此时最优评分**等价于典范相关问题**，解可由**单个特征分解**算出（Exercise 12.6）。

创建 $N\times K$ 指示响应矩阵 $Y$（$y_{ik}=1$ 若 $g_i=k$，否则 0）。计算步骤：

1. **多元非参数回归**。对 $X$ 拟合多响应、自适应的非参数回归得 $\hat Y$。令 $S_\lambda$ 为拟合最终所选模型的线性算子，$\eta^{\ast}(x)$ 为拟合的回归函数向量。
2. **最优得分**。计算 $Y^\top\hat Y=Y^\top S_\lambda Y$ 的特征分解，其中特征向量 $\Theta$ 归一化：$\Theta^\top D_\pi\Theta=I$，$D_\pi=Y^\top Y/N$ 是估计的类先验概率对角矩阵。
3. **用最优得分更新模型**（第 1 步）：$\eta(x)=\Theta^\top\eta^{\ast}(x)$。$\eta(x)$ 的第一个 $K$ 函数之一是常数函数——一个平凡解；其余 $K-1$ 个函数是判别函数。常数函数连同归一化使所有剩余函数中心化。

当 $S_\lambda=H_X$（线性回归投影算子，$H_X=X(X^\top X)^{-1}X^\top$）时，FDA 就是 LDA。

> **推导** · 最优评分化为特征值问题（Exercise 12.6）
>
> **(12.65)**：把 (12.53) 写成向量记法。令 $\Theta$ 是 $K\times L$ 得分矩阵（列），$H$ 是 $N\times M$ 基矩阵（(12.65)）：
>
> $$\min_{\theta,\beta}\ \lVert Y\theta-H\beta\rVert^2 \eqno{12.65}$$
>
> **(a) 归一化的解释**：$\theta^\top D_\pi\mathbf 1=0$（对训练样本得分（条件）期望为零）与 $\theta^\top D_\pi\theta=1$（方差 1）。因为 $D_\pi=Y^\top Y/N$ 对角，$\theta^\top D_\pi\theta=\frac1N\sum_i(\theta^\top y_i)^2=\frac1N\sum_i\theta(g_i)^2$，故第二个归一化是「得分在训练集上单位方差」。第一个是「均值零」。
>
> **(b) 对 $\beta$ 部分优化**。对 $\beta$ 求梯度：$-2H^\top(Y\theta-H\beta)=0$，即 $\hat\beta=(H^\top H)^{-1}H^\top Y\theta$，故 $\lVert Y\theta-H\hat\beta\rVert^2=\lVert(I-H_X)Y\theta\rVert^2=\theta^\top Y^\top S Y\theta$，其中 $S=I-H_X$ 是残差投影算子（Exercise 12.6 中 $S$ 指 $Y$ 拟合的**投影算子**对应的残差）。在归一化 $\theta^\top D_\pi\theta=1$ 与 $\theta^\top D_\pi\mathbf 1=0$ 下，(12.65) 部分优化后为
>
> $$\max_{\theta}\ \theta^\top Y^\top S Y\,\theta \eqno{12.66}$$
>
> **归一化约束**：$\theta^\top D_\pi\mathbf 1=0$ 和 $\theta^\top D_\pi\theta=1$。这是约束特征值问题（Rayleigh 商，预备知识 O1）。
>
> **(c) 若 $h_j$ 包含常数函数，则 $S$（指 $Y^\top S_\lambda Y$）的最大特征值是 1。** 当 $h_j$ 含常数时，$S_\lambda$ 是投影（幂等），$S_\lambda\mathbf 1=\mathbf 1$（常数函数的拟合被保留），故 $Y^\top S_\lambda Y$ 作用在 $\mathbf 1$（对 $Y$ 而言即「所有行之和」）上给出 $S_\lambda Y$ 对常数的拟合。因 $S_\lambda$ 是投影，其最大特征值为 1。$\blacksquare$
>
> **实际算法**：上面三步就是完整计算。步骤 2 是单个 $K\times K$ 对称矩阵 $Y^\top S_\lambda Y$ 的特征分解（用 SVD 或 Jacobi 旋转）。$\Theta^\top D_\pi\Theta=I$ 的归一化保证不同 $\ell$ 的得分正交归一（一般化特征值问题）。第一个特征向量对应 $S$ 的特征值 1——它是平凡的「常数」方向（$Y\theta$ 为常数），其余 $K-1$ 个是判别方向。
>
> **注意**：归一化用 $D_\pi$（类比例对角阵）而非 $I$，所以**得分在每个类内加权正交**，这与 §4.3.3 中 LDA 的规范向量条件（$\nu_\ell^\top\hat\Sigma\nu_{\ell'}=\delta$）一致。

<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-5-1">原文 §12.5.1</a>

---

## 12.6 惩罚判别分析 {#s-12-6}

虽然 FDA 被启发为最优评分的推广，它也可直接看成**正则化判别分析**的形式。设 FDA 中用的回归程序是对基展开 $h(X)$ 的线性回归，带系数的二次惩罚：

$$
\mathrm{ASR}\big(\{\theta_\ell,\beta_\ell\}_{\ell=1}^L\big)=\frac1N\sum_{\ell=1}^L\sum_{i=1}^N\Big[\big(\theta_\ell(g_i)-h(x_i)^\top\beta_\ell\big)^2+\lambda\beta_\ell^\top\Omega\beta_\ell\Big] \eqno{12.57}
$$

$\Omega$ 的选择依赖问题。若 $\eta_\ell(x)=h(x)\beta_\ell$ 是样条基函数上的展开，$\Omega$ 可以约束 $\eta_\ell$ 在 $\mathbb R^p$ 上光滑。在加性样条情形中，每个坐标有 $N$ 个样条基函数，故 $h(x)$ 中共有 $Np$ 个基函数；此时 $\Omega$ 是 $Np\times Np$ 的块对角矩阵。

FDA 的步骤于是可以看成是 LDA 的推广形式，即**惩罚判别分析（PDA）**，三条：

1. 通过基展开 $h(X)$ 增大预测变量 $X$ 的集合。
2. 在放大空间用（惩罚）LDA，其中惩罚 Mahalanobis 距离是
$$
D(x,\mu)=\big(h(x)-h(\mu)\big)^\top\big(\Sigma_W+\lambda\Omega\big)^{-1}\big(h(x)-h(\mu)\big) \eqno{12.58}
$$
其中 $\Sigma_W$ 是派生变量 $h(x_i)$ 的类内协方差矩阵。
3. 用惩罚度量分解分类子空间：$\max_u u^\top\Sigma_{\rm Bet}u$ s.t. $u^\top(\Sigma_W+\lambda\Omega)u=1$。

粗略地说，惩罚 Mahalanobis 距离倾向于给「粗糙」坐标更小权重，给「光滑」坐标更大权重；由于惩罚不是对角的，同样适用于粗糙或光滑的线性组合。

对某些问题类，第一步（基展开）并不需要；我们已有太多（相关）预测变量。主要例子是被分类对象是数字化模拟信号时：

- 一段语音的 log-periodogram，按 256 个频率采样；
- 手写数字的灰度像素值。

在这类情形下需要正则化是直觉上清楚的。以数字化图像为例：相邻像素值往往相关，常常几乎相同，这意味对应像素的 LDA 系数对可以** wildly 不同且反号**，从而在应用于相似像素值时相消。正相关预测变量导致**负**相关的系数估计，这个噪声导致不需要的抽样方差。合理的策略是把系数正则化为在空间域（作为图像）光滑。这就是 PDA 所做的。计算与 FDA 相同，只是用适当的惩罚回归方法。这里 $h^\top(X)\beta_\ell=X\beta_\ell$（$h$ 为恒等），$\Omega$ 选为使 $\beta_\ell^\top\Omega\beta_\ell$ 惩罚 $\beta_\ell$ 作为图像的粗糙度。图 12.11 给出用 LDA 与 PDA 的判别变量：LDA 的像「salt-and-pepper」（椒盐），PDA 的光滑。Hastie et al. (1995) 在他们试过的情形中，报告正则化把 LDA 在独立测试数据上的分类性能改进约 25%。

---

## 12.7 混合判别分析 {#s-12-7}

### 12.7.1 高斯混合与 MDA 公式 {#s-12-7-1}

LDA 可以看作**原型分类器**：每类由其质心代表，用适当的度量分类到最近的。但在很多情形下单个原型不足以表示非均匀的类，混合模型更合适。本节回顾高斯混合模型并展示如何用早前的 FDA 与 PDA 方法推广它们。第 $k$ 类的高斯混合模型有密度

$$
P(X\mid G=k)=\sum_{r=1}^{R_k}\pi_{kr}\,\phi(X;\mu_{kr},\Sigma) \eqno{12.59}
$$

其中混合比例 $\pi_{kr}$ 求和为一。这给第 $k$ 类 $R_k$ 个原型，且在我们的设定中整个度量用同一协方差矩阵 $\Sigma$。给定每类这样的模型，类后验概率是

$$
P(G=k\mid X=x)=\frac{\sum_{r=1}^{R_k}\pi_{kr}\phi(x;\mu_{kr},\Sigma)\Pi_k}{\sum_{\ell=1}^K\sum_{r=1}^{R_\ell}\pi_{\ell r}\phi(x;\mu_{\ell r},\Sigma)\Pi_\ell} \eqno{12.60}
$$

其中 $\Pi_k$ 是类先验概率。

与 LDA 一样，我们用**极大似然**估计参数，基于 $P(G,X)$ 的联合对数似然：

$$
\ell=\sum_{k=1}^K\sum_{r=1}^{R_k}\sum_{i=1}^N\big[\log\pi_{kr}+\log\phi(x_i;\mu_{kr},\Sigma)\big]\Pi_k \eqno{12.61}
$$

（对 $g_i=k$ 的观测求和。）和内层求和使直接优化相当麻烦。计算混合分布 MLE 的经典自然方法是 **EM 算法**（Dempster et al., 1977），它有良好的收敛性质。EM 交替两步：

1. **E 步**：给定当前参数，计算第 $k$ 类观测（$g_i=k$）的子类 $c_{kr}$ 在类 $k$ 内的责任（responsibility）：
$$
W(c_{kr}\mid x_i,g_i)=\frac{\pi_{kr}\phi(x_i;\mu_{kr},\Sigma)}{\sum_{\ell=1}^{R_k}\pi_{k\ell}\phi(x_i;\mu_{k\ell},\Sigma)} \eqno{12.62}
$$
2. **M 步**：用 E 步的权重，计算类 $k$ 内每个成分高斯参数的加权 MLE。

E 步把第 $k$ 类观测的单位权重分摊到分配给该类的各子类。若观测紧靠某子类的质心、远离其它，它给该子类的质量接近 1；在两个子类中间的观测给两者近似相等的权重。

M 步中，第 $k$ 类的一个观测被用 $R_k$ 次来估计每个成分密度的参数，各带不同权重。

**初始化**：用户指定每类子类数 $R_k$。类 $k$ 内用 $k$-means 聚类（多随机起点）拟合数据，把观测划分成 $R_k$ 不相交组，从中创建由 0/1 构成的初始权重矩阵。

**秩限制与降维**。整个设定中假设等成分协方差 $\Sigma$ 带来额外的简单性；可以像 LDA 那样在混合设定中纳入秩限制。回顾 LDA 的一个鲜为人知的事实：**秩 $L$ 的 LDA 拟合等价于一个高斯模型的极大似然拟合，其中每类不同的均值向量被限制在 $\mathbb R^p$ 的秩 $L$ 子空间内**（Exercise 4.8）。我们可以为混合模型继承这个性质，在所有 $\sum_kR_k$ 个中心上的秩约束 $\mathrm{rank}\{\mu_{k\ell}\}=L$ 下最大化对数似然 (12.61)。

EM 仍可用，且 M 步变成**加权版本的 LDA**（$R=\sum_{k=1}^KR_k$ 个「类」）。进一步可以用最优评分解加权 LDA 问题，从而在该阶段用加权版本的 FDA 或 PDA。可以用线性算子做最优评分回归时，「观测」数的增加意外地不发生：扩大的指示 $Y$ 矩阵在这种情形下塌缩为一个**模糊的**响应矩阵 $Z$。例如 $K=3$ 类、每类 $R_k=3$ 子类时，$Z$ 的第 $k$ 行元素是 E 步的责任 $W(c_{k1}\mid x,g_i),W(c_{k2}\mid x,g_i),W(c_{k3}\mid x,g_i)$，形如下面这样（每行非负、行和为 1，但元素不再是 0/1）：

$$
\begin{pmatrix}
0.0 & 0.3 & 0.5 & 0.2 & 0.0 & 0.0\\
0.9 & 0.1 & 0.0 & 0.0 & 0.0 & 0.0\\
0.1 & 0.8 & 0.1 & 0.0 & 0.0 & 0.0\\
0.0 & 0.0 & 0.0 & 0.5 & 0.4 & 0.1\\
0.0 & 0.0 & 0.0 & 0.7 & 0.1 & 0.2\\
\vdots & \vdots & \vdots & \vdots & \vdots & \vdots\\
0.0 & 0.0 & 0.0 & 0.0 & 0.1 & 0.8
\end{pmatrix}\ \in\ \mathbb{R}^{N\times9} \eqno{12.63}
$$

> **结果** · 为什么「模糊」是自然的结果
>
> **第 1 步**：$Z$ 的行由 (12.62) 定义。对固定 $g_i=k$ 与固定 $i$，由分母定义（所有子类分母相同）立即有 $\sum_{r=1}^{R_k}W(c_{kr}\mid x_i,g_i)=1$，且 $W\ge0$。
>
> **第 2 步（关键）**：$\pi_{kr}\ge0$、$\phi(x_i;\mu_{kr},\Sigma)>0$ 对 $k,i,r$ 成立（高斯密度严格为正），故 (12.62) 的分子分母都 $\ge0$，且分母 $\ge$ 任一分子，故 $0\le W\le1$。**「模糊」的两条性质（非负、行和 1）逐字成立。**
>
> **第 3 步**：$Y$ 是严格 0/1（因为 $g_i$ 唯一），$Z$ 是一般 $[0,1]$。这个转变正是**把「硬标签类」换成「软标签子类」**。数值上 $S_\lambda$ 只看到 $Z$，所以扩大的规模**不改变回归部分**的复杂度——优化评分回归仍是 $N$ 行的拟合。

M 步的剩余步骤是

$$
\hat Z=S_\lambda Z,\qquad ZZ^\top=\Theta D\Theta^\top
$$

然后更新 $\pi$ 与 $\Pi$。这些简单修改给混合模型增加了相当大的灵活性：

- **MDA 用子类代替类**，然后允许我们看子类中心张成子空间的低维视图。这子空间常是对判别重要的。
- 在 M 步用 FDA 或 PDA，能进一步适应特殊情形（如带光滑性约束的数字化信号/图像拟合）。

<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html#s-12-7">原文 §12.7</a>

### 12.7.2 示例：波形数据 {#s-12-7-2}

这里用一个流行的模拟例子说明这些思想（Breiman et al., 1984）。这是三分类、21 变量问题，预测变量定义为

$$
X_j=\begin{cases}U\,h_1(j)+(1-U)h_2(j)+\varepsilon_j&\text{Class 1}\\ U\,h_1(j)+(1-U)h_3(j)+\varepsilon_j&\text{Class 2}\\ U\,h_2(j)+(1-U)h_3(j)+\varepsilon_j&\text{Class 3}\end{cases} \eqno{12.64}
$$

其中 $j=1,\dots,21$，$U$ 在 $(0,1)$ 上均匀，$\varepsilon_j$ 是标准正态变量，$h_\ell$ 是平移的三角波形：$h_1(j)=\max(6-|j-11|,0)$，$h_2(j)=h_1(j-4)$，$h_3(j)=h_1(j+4)$。

表 12.4 给出 MDA 以及本章其它方法的结果。每个训练样本 300 观测，用相等先验，故每类约 100 观测。测试样本 500。**惩罚 MDA 模型**（3 子类 + 惩罚到 4 df）的测试错误率 0.157，优于 LDA 的 0.191，接近 Bayes 率 0.140。这不奇怪，因为 MDA 模型的结构与生成模型相似。

图 12.15 给出测试数据上的前两个惩罚 MDA 典范变量。如我们猜到的，类落在三角形的边上。这是因为 $h_j(i)$ 由 21 维空间中三个点表示，形成三角形的顶点，每类表示为两个顶点的凸组合，故落在一条边上。视觉上所有信息在前两维；前两个坐标解释 99.8% 的方差，截断到那里无损。

---

## 12.8 MDA2 与计算量 {#s-12-8}

**MDA2**（Exercise 12.11）是 MDA 的推广，允许每个混合中心由所有类共享。取标签与特征的联合密度为联合密度的混合：

$$
P(G,X)=\sum_{r=1}^R\pi_r P_r(G,X) \eqno{12.67}
$$

进一步假设

$$
P_r(G,X)=P_r(G)\,\phi(X;\mu_r,\Sigma) \eqno{12.68}
$$

这个模型由中心在 $\mu_r$ 的区域组成，每处有类轮廓 $P_r(G)$。后验类分布是

$$
P(G=k\mid X=x)=\frac{\sum_{r=1}^R\pi_r P_r(G=k)\phi(x;\mu_r,\Sigma)}{\sum_{r=1}^R\pi_r\phi(x;\mu_r,\Sigma)} \eqno{12.69}
$$

（分母是边缘分布 $P(X)$）。这可证可看作 MDA 的推广，因为

$$
P(X\mid G=k)=\sum_{r=1}^R\pi_r P_r(G=k)\phi(x;\mu_r,\Sigma)\ \text{对应于}\ \sum_{r=1}^R\pi_{rk}\phi(x;\mu_r,\Sigma),\quad \pi_{rk}=\frac{\pi_rP_r(G=k)}{\sum_{r=1}^R\pi_rP_r(G=k)} \eqno{12.70}
$$

其中 $\pi_{rk}$ 是第 $k$ 类的混合比例。MDA2 的 EM 算法与 MDA 类似；若初始权重矩阵像 MDA 那样构造（各类内独立的 $k$-means），MDA2 的算法与原 MDA 过程相同。

**计算量**。有 $N$ 训练样本、$p$ 预测变量、$m$ 支持向量时，SVM 需要 $m^3+mN+mpN$ 次运算（假设 $m\approx N$）。它们**随 $N$ 扩展性不好**，虽然有计算捷径可用（Platt, 1999）。LDA 需要 $Np^2+p^3$ 次运算，PDA 相同。FDA 的复杂度依赖所用回归方法：许多技术对 $N$ 线性（加性模型与 MARS）；一般样条与基于核的回归方法通常需要 $N^3$ 次运算。

拟合 FDA、PDA 与 MDA 模型的软件在 R 包 `mda` 中提供，S-PLUS 中也有。

---

<a class="src" href="../esl/ch12-support-vector-machines-andflexible-discriminants.html">原文第 12 章</a>