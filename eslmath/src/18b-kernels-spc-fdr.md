## 18.5 特征不可得时的分类 {#s-18-5}

前四节的全部机器都是「给每个变量一个数」：对角 LDA 假设协方差矩阵对角，因而 $\Sigma^{-1}$ 的第 $j$ 个对角元可以单独算；二次正则化把 $XX^\top$ 的 $N$ 个特征值当作全部信息；lasso 要的是 $x_j^\top r$，仍然是「第 $j$ 个变量」的量。本节换一套思路：**当 $p$ 维数值特征根本拿不到时，只要求出一对对象之间的相似度**，其余全部从相似度里推出来。

两种可用的原始输入：

- **内积矩阵**：只能算出 $\langle x_i,x_{i'}\rangle$（典型情形是文本的词袋表示、基因组序列）。
- **配对距离矩阵**：只能算出 $\lVert x_i-x_{i'}\rVert_2$（典型情形是蛋白质序列、形状距离）。

两类输入可以互相转换，转换之后 §18.3–18.4 的大部分机器都能用。核心事实只有一句：**$p$ 维向量的内积矩阵（Gram 矩阵）是 $N\times N$ 的，与 $p$ 无关**；任何只需要 Gram 矩阵的方法都可以在 $p=10^6$ 的情况下计算。本节要做的全部工作就是逐个验证哪些方法满足这个「只用 Gram」的要求。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-5">原文 §18.5</a>

> **基础知识** · Gram 矩阵与核
>
> 设 $X\in\mathbb{R}^{N\times p}$，第 $i$ 行是 $x_i^\top$。Gram 矩阵 $K=XX^\top\in\mathbb{R}^{N\times N}$，元 $K_{ii'}=\langle x_i,x_{i'}\rangle$。
>
> - $K$ **对称半正定**：$x^\top Kx=\lVert X^\top x\rVert_2^2\ge0$。
> - $\mathrm{rank}(K)=\mathrm{rank}(X)$（见预备知识 L1、L3）。
> - **存在性**：$K$ 半正定 $\iff$ 存在 $C$ 使 $K=C^\top C$（预备知识 L1 的正定性等价说法）。所以任何半正的 $N\times N$ 矩阵都**是某个 $\mathbb{R}^p$ 里 $N$ 个点的 Gram 矩阵**。这就是「核技巧」的全部合法性来源：半正定性保证了 $p$ 维向量的存在性，于是可以只算 $N\times N$。
> - $XX^\top$ 与 $X^\top X$ 的非零特征值相同（预备知识 L3 的 SVD 推导：$X=UDV^\top\Rightarrow XX^\top=UD^2U^\top$，$X^\top X=VD^2V^\top$）。

### 18.5.1 例：字符串核与蛋白质分类 {#s-18-5-1}

**问题：给两个氨基酸字符串算一个相似度，并说明它为什么能驱动一个 $20^4=160000$ 维空间里的 SVM。** 对应 (18.25)(18.26)。

蛋白质是氨基酸的字符串，长度在 75 到 160 之间，每个位置是 20 种氨基酸之一。既然每个位置只有 20 种取值，最自然的想法是**把字符串当成长度 $p$ 的 $\{0,1\}^{20}$ 向量**：第 $j$ 位是「该位置出现了氨基酸 $j$」的指示向量

$$
X_j=\big(I(t_j=\text{A}),\,I(t_j=\text{C}),\,\dots,\,I(t_j=\text{V})\big)\in\{0,1\}^{20},\qquad j=1,\dots,p
$$

> **坑** · 这个编码把「顺序」完全丢掉了。序列 `ACGT` 与 `GTCA` 的编码逐位相同，$\lVert x-x'\rVert_2=0$，但它们显然不是同一个分子。欧氏距离在这里没有任何生物学含义，所以「用现成的 $p$ 维分类器」的路线第一步就断了。
>
> 补救办法不能是「加一个特征把顺序编码回来」——那需要 $20^p$ 个特征。唯一可行的办法是：**放弃 $p$ 维空间的坐标，直接在对象之间定义相似度**。

**做法：数「共有子串」。** 把长度为 $p$ 的字符串 $x$ 的特征映射定义为

$$
\Phi_m(x)=\big\{\varphi_a(x)\big\}_{a\in\mathcal{A}_m},\qquad \varphi_a(x)=\#\{\text{子串 } a \text{ 在 } x \text{ 中出现的次数}\} \eqno{18.25}
$$

其中 $\mathcal{A}_m$ 是全部长度为 $m$ 的氨基酸串组成的集合，$|\mathcal{A}_m|=20^m$。于是 $\Phi_m(x)$ 是一 $\mathbb{R}^{20^m}$ 的**计数向量**，而它同时把顺序编码进了特征名里：特征「LQE 出现几次」只有在顺序一致时才有贡献。

两串的内积定义为特征映射的内积：

$$
K_m(x_1,x_2)=\Big\langle\Phi_m(x_1),\ \Phi_m(x_2)\Big\rangle\eqno{18.26}
$$

> **推导** · (18.26) 可以完全展开，这是理解核方法的关键一步：
>
> $$\sum_{a\in\mathcal{A}_m}\varphi_a(x_1)\,\varphi_a(x_2)=\sum_{a\in\mathcal{A}_m}\Big(\sum_{r=1}^{p-m+1}I(x_{1,r:r+m-1}=a)\Big)\Big(\sum_{s=1}^{p-m+1}I(x_{2,s:s+m-1}=a)\Big)$$
>
> $=\sum_{r=1}^{p-m+1}\sum_{s=1}^{p-m+1}\sum_{a\in\mathcal{A}_m}I(x_{1,r}=a)I(x_{2,s}=a)$（把两个求和交换、$a$ 求和只留下共同的那些 $a$）
>
> $=\sum_{r=1}^{p-m+1}\sum_{s=1}^{p-m+1}\mathbf 1\big(x_{1,r:r+m-1}=x_{2,s:s+m-1}\big)$
>
> = 两个字符串中**相同长度-$m$ 子串的配对个数**。
>
> 每一项都是 0/1 且只依赖位置对 $(r,s)$，所以 $K_m(x_1,x_2)$ 是 $p^2$ 个 0/1 指示的平均：$K_m(x_1,x_2)=\frac{1}{(p-m+1)^2}\sum_{r,s}(\text{共有子串指示})$。这就是「共有位置数」的严格含义。

> **结果** · (18.26) 定义的是 $N\times N$ 的**字符串核矩阵** $K_m=[K_m(x_i,x_{i'})]_{i,i'=1}^N$，与 $p$ 和 $20^m$ 都无关。理论上它可以用树结构在 $O(\mathrm{poly}(N,p))$ 内算出而不构造 $\Phi_m$（Leslie et al., 2004）。数值上 $m=3$ 时 $\Phi_3$ 已是 8000 维，$m=4$ 时是 160000 维，而 $N=1708$（负类 1663、正类 45）——**$p=160000\gg N$，但要算的只是 $1708\times1708$ 的矩阵**。用 $K_m$ 驱动 SVM（对偶问题只含 $N$ 个 $\alpha_i$，见预备知识 O1）在 160000 维空间里求最大间隔，得到图 18.9 的 ROC（面积 0.84）。
>
> 为什么这与谱方法是同一件事：$K_m$ 半正定（它是 $\Phi_m(X^\top)^\top\Phi_m(X^\top)$，见预备知识 L1 的 $K=C^\top C$），所以它的特征分解 $K_m=U\Lambda U^\top$ 给出 $\Phi_m(X^\top)^\top=U\Lambda^{1/2}$。也就是说，**在 160000 维特征空间里的任何线性算法，都可以写成「投影到这 $N=1708$ 个基向量上」的 $N$ 维算法**。核 PCA（§14.5.4）就是这一句的直接使用。

### 18.5.2 用内积核与配对距离做分类 {#s-18-5-2}

**问题：哪些分类器可以只用内积矩阵 $K=XX^\top$ 计算？** 对应 (18.27)(18.28)，本节的核心。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-5-2">原文 §18.5.2</a>

> **基础知识** · 双线性型的展开
>
> $\langle a-b,c\rangle=\langle a,c\rangle-\langle b,c\rangle$，对任意三个向量成立（分配律）。因此距离可以写成内积的组合。

#### 第一步：内积 $\to$ 距离 (18.27)

$k$ 近邻（§13.3）只需要**配对距离**，不需要特征。展开：

$$
\lVert x_i-x_{i'}\rVert_2^2=\sum_{j=1}^p(x_{ij}-x_{i'j})^2=\sum_{j=1}^p x_{ij}^2-2\sum_{j=1}^p x_{ij}x_{i'j}+\sum_{j=1}^p x_{i'j}^2=\langle x_i,x_i\rangle+\langle x_{i'},x_{i'}\rangle-2\langle x_i,x_{i'}\rangle \eqno{18.27}
$$

> **推导** · 逐步说清「哪一步用了什么」：
>
> 1. 第一个等号：欧氏范数的定义（L1），$\lVert v\rVert_2^2=\sum_jv_j^2$，这里 $v=x_i-x_{i'}$。
> 2. 第二个等号：平方的分配律 $(a-b)^2=a^2-2ab+b^2$，**逐项**作用在 $\sum_j$ 内部。
> 3. 第三个等号：$\sum_jx_{ij}^2=\lVert x_i\rVert_2^2=\langle x_i,x_i\rangle$（对角元，$\lVert v\rVert_2^2=\langle v,v\rangle$）；$\sum_jx_{ij}x_{i'j}=x_i^\top x_{i'}=\langle x_i,x_{i'}\rangle$。
>
> 所以 $K_{ii}$ 给出 $\lVert x_i\rVert_2^2$，$K_{ii'}$ 给出内积，(18.27) 把三者组合成距离。**验证**：令 $i'=i$ 则右边 $=\lVert x_i\rVert^2+\lVert x_i\rVert^2-2\lVert x_i\rVert^2=0$ ✓；令 $i'=0$（原点）则 $=\lVert x_i\rVert^2$，而左边 $=\lVert x_i\rVert^2$ ✓。

> **结果** · (18.27) 说明：**整个 $N\times N$ Gram 矩阵恰好是 $k$ 近邻所需的全部信息**。所以 1-NN、距离加权的 1-NN（图 18.9 的蓝线）、最近中位点（medoid）、$k$-medoids 聚类（§14.3.10）在 $p=10^6$ 时都能算，工作量 $O(N^2)$。

#### 第二步：内积 $\to$ 类中心距离 (18.28)

最近类中心分类器要的是 $\lVert x_0-\bar x_k\rVert_2$，其中 $\bar x_k=\frac{1}{N_k}\sum_{i\in C_k}x_i$。逐步展开：

$$
\begin{aligned}
\Big\lVert x_0-\bar x_k\Big\rVert_2^2
&=\Big\langle x_0-\bar x_k,\ x_0-\bar x_k\Big\rangle\\
&=\langle x_0,x_0\rangle-2\langle x_0,\bar x_k\rangle+\lVert\bar x_k\rVert_2^2\\
&=\langle x_0,x_0\rangle-\frac{2}{N_k}\sum_{i\in C_k}\langle x_0,x_i\rangle+\frac{1}{N_k^2}\sum_{i,i'\in C_k}\langle x_i,x_{i'}\rangle
\end{aligned}
$$

每一步的依据：第 2 行是双线性型对 $\langle a-b,a-b\rangle$ 的展开（$\langle a,a\rangle-2\langle a,b\rangle+\langle b,b\rangle$，用对称性 $\langle\bar x_k,x_0\rangle=\langle x_0,\bar x_k\rangle$）；第 3 行代入 $\bar x_k=\frac{1}{N_k}\sum_{i\in C_k}x_i$，并把双线性型当双线性形式展开两次。特别地最后一项

$$
\frac{1}{N_k^2}\sum_{i,i'\in C_k}\langle x_i,x_{i'}\rangle=\frac{1}{N_k^2}\Big(\sum_{i\in C_k}x_i\Big)^\top\Big(\sum_{i'\in C_k}x_{i'}\Big)=\Big\lVert\frac{1}{N_k}\sum_{i\in C_k}x_i\Big\rVert_2^2=\lVert\bar x_k\rVert_2^2
$$

即它是 Gram 矩阵里 $C_k\times C_k$ 方块的总和。把上面三行按类汇总，得到 (18.28) 印出的分组形式：

$$
2\sum_{k=1}^{K}\frac{1}{N_k}\sum_{i\in C_k}\Big\lVert x_0-\bar x_k\Big\rVert_2^2=\Big\langle x_0,x_0\Big\rangle+\sum_{k=1}^{K}\frac{2}{KN_k}\sum_{i\in C_k}\Big\langle x_0,x_i\Big\rangle+\sum_{k=1}^{K}\frac{2}{KN_k^2}\sum_{i,i'\in C_k}\Big\langle x_i,x_{i'}\Big\rangle \eqno{18.28}
$$

> **结果** · (18.28) 是本节的核心结论。逐项读它的三个加项：
>
> | 项 | 需要什么 | 出现次数 |
> |---|---|---|
> | $\langle x_0,x_0\rangle$ | Gram 的 $(0,0)$ 元 | 1 |
> | $\langle x_0,x_i\rangle$ | Gram 的第 0 行/列 | $O(N)$ |
> | $\langle x_i,x_{i'}\rangle$ | Gram 的方块 | $O(N^2)$ |
>
> **没有一项含「第 $j$ 个变量」。** 所以最近类中心分类、$K$-means（质心更新只需 Gram）、判别分析（§18.2 的对角 LDA 也可核化，见 Exercise 12.10）、带二次正则化的逻辑与多项式回归（§12.3.3）在 $p$ 任意大时都能算——总代价只是 $N\times N$ 的存储与 $O(N^2)$ 的运算。
>
> 进一步，最近收缩质心（NSC，§18.2）**不能**这样算，因为它的阈值 $\hat\beta_j$ 是逐变量的量，必须访问 $x_{ij}$。

> **坑** · (18.28) 印出的分组方式里，第一项 $\langle x_0,x_0\rangle$ 只出现一次，而后两项带系数 2（或等价地带上 $1/K$），这是把 $\sum_k$ 归一化成「类平均」时 $K$ 与 2 的归一化系数被吸收的结果。若要自己核对，只需检查上面的三行展开：三类项分别只依赖 Gram 的对角元、第 0 行、以及方块，任何归一化方式都不会引入新信息。
>
> 另一个坑：$\bar x_k$ 是**均值**。若对象本身没有「平均」这个概念（如蛋白质），medoid（类内到其它点平均距离最小的**观测本身**）才是可用的替代量。它只需要距离矩阵，天然「只用 Gram」；但它在文摘实验里表现极差（§18.5.3 表 18.3 的 0.65），因为 medoid 的方差远大于均值。

#### 第三步：距离 $\to$ 内积（配对距离做分类）

若手里只有配对**平方**距离

$$
\Delta^2_{ii'}=\lVert x_i-x_{i'}\rVert_2^2 \eqno{18.29}
$$

先把距离变成「负二倍」的半矩阵 $B$：$B_{ii'}=-\Delta^2_{ii'}/2$，再做**双重中心化**

$$
\tilde K=(I-M)B(I-M),\qquad M=\frac{1}{N}\mathbf{1}\mathbf{1}^\top \eqno{18.31}
$$

关键是验证 $\tilde K$ 确实等于中心化后的内积。记

$$
r_i=\sum_{j=1}^N\Delta^2_{ij}\ (\text{第 } i\text{ 行和}),\qquad r=\sum_{i=1}^N r_i,\qquad d=\sum_{i=1}^N\Delta^2_{ii}\ (\text{对角和})
$$

由 (18.27) 反解出内积：$\langle x_i,x_j\rangle=\frac12\big(\Delta^2_{ii}+\Delta^2_{jj}-\Delta^2_{ij}\big)$。再用 $\bar x=\frac1N\sum_i x_i$：

- $\displaystyle \langle x_i,\bar x\rangle=\frac1N\sum_{j=1}^N\langle x_i,x_j\rangle=\frac{1}{2N}\Big(N\Delta^2_{ii}+d-r_i\Big)$
  （用 $\sum_j\Delta^2_{jj}=d$）
- $\displaystyle \lVert\bar x\rVert_2^2=\frac{1}{N^2}\sum_{i,i'}\langle x_i,x_{i'}\rangle=\frac{1}{2N^2}\big(Nd+Nd-r\big)=\frac{d}{N}-\frac{r}{2N^2}$

于是

$$
\Big\langle x_i-\bar x,\ x_{i'}-\bar x\Big\rangle=\langle x_i,x_{i'}\rangle-\langle x_i,\bar x\rangle-\langle\bar x,x_{i'}\rangle+\lVert\bar x\rVert_2^2=-\frac12\Delta^2_{ii'}+\frac{r_i+r_{i'}}{2N}-\frac{d}{N}+\frac{r}{2N^2} \eqno{18.30}
$$

验证（小例）：$N=2$，$x_1=0$，$x_2=e_1$。则 $\Delta^2_{12}=1$，$r_1=r_2=1$，$r=2$，$d=1$。右边 $=\frac{-1}{2}+\frac{2}{4}-\frac12+\frac{2}{8}=-\frac14$。而 $\bar x=\frac12e_1$，直接算 $\langle-\frac12e_1,\frac12e_1\rangle=-\frac14$ ✓。

现在直接算 (18.31) 的元。$(M)_{ij}=1/N$、$(\mathbf 1^\top B)_i=\sum_lB_{il}=-\frac12\sum_l\Delta^2_{il}=-\frac{r_i}{2}$、$(MB)_{ij}=\frac1N(\mathbf 1^\top B)_j=-\frac{r_j}{2N}$、$(BM)_{ij}=-\frac{r_i}{2N}$、$(MBM)_{ij}=\frac1{N^2}\sum_{a,b}B_{ab}=\frac1{N^2}\cdot\frac12(2Nd-r)=\frac dN-\frac{r}{2N^2}$。合起来

$$
\tilde K_{ij}=B_{ij}-(MB)_{ij}-(BM)_{ij}+(MBM)_{ij}=-\frac12\Delta^2_{ij}+\frac{r_i+r_j}{2N}-\frac dN+\frac{r}{2N^2}
$$

与 (18.30) 的右边**逐项相同**，故

$$
\tilde K=\big[\langle x_i-\bar x,\ x_{i'}-\bar x\rangle\big]_{i,i'}=\tilde X\tilde X^\top,\qquad \tilde X=(I-M)X
$$

> **结果** · 「配对距离 $\to$ 中心化 Gram」的完整链条是 (18.29) $\to$ $B=\{-\Delta^2_{ii'}/2\}$ $\to$ (18.31) 双重中心化 $\to$ 得到 $\tilde X\tilde X^\top$。做完这一步，前三小节所有「只用 Gram」的分类器全部可用。
>
> 主成分也可以这样算：$\tilde X=UDV^\top$ 则 $\tilde K=UD^2U^\top$，而 $\tilde X$ 的主成分变量矩阵是 $\tilde Z=\tilde X V=UD$。**$U$ 从 $N\times N$ 的 $\tilde K$ 的特征分解得到**，所以「找前几个主成分」在 $p\gg N$ 时是 $O(N^3)$ 的事（预备知识 L3）。若原始 $X$ 未中心化，先做 $\tilde X=(I-M)X$ 再算 $(I-M)K(I-M)$，这就是原文强调的「双重中心化核」。

#### 第四步：核方法做不到什么

原文明确列出三条限制，值得逐条对应到前面的公式：

- **不能标准化变量**。标准化要 $s_j=\sqrt{\frac{1}{N-p}\sum_i(x_{ij}-\bar x_j)^2}$，需要第 $j$ 列。Gram 矩阵里没有「第 $j$ 列」这个概念。
- **不能评估单个变量的贡献**。逐变量的 $t$ 检验、§18.2 的 NSC、以及任何带 lasso 惩罚的模型（§18.4）都要求访问 $x_{ij}$，因此无法核化。
- **不能分离好变量与噪声变量**。核方法让所有方向平等参与；若信噪比低（相关特征中真正相关的只占小部分），核方法通常不如做特征选择的方法。文摘实验（表 18.3）是这个限制的定量演示：最近类中心 0.29、1-NN 0.44、medoid 0.65，而做了逐变量标准化的 NSC 只有 0.17。

### 18.5.3 例：文摘分类 {#s-18-5-3}

**问题：用配对方法能做词袋文本分类吗？效果差多少？**

48 篇论文的摘要（Bradley Efron / Trevor Hastie 与 Rob Tibshirani / Jerome Friedman，各 16 篇），去掉引号、括号、特殊符号并全部转小写，删掉高频无信息词 `we`（否则会不公平地区分 HT 的摘要）。抽出全部 4492 个词，其中不重复词 $p=1310$，特征 $x_{ij}$ 是第 $j$ 个词在第 $i$ 篇摘要里出现的次数。这是标准的词袋表示——**特征是存在的，而且就是词频向量**，所以本例不涉及特征不可得，只涉及「用不用特征本身」。

交叉验证结果（表 18.3，10 折）：

| 方法 | CV 误差 (SE) | 是否需要原始 $X$ |
|---|---|---|
| 最近收缩质心（NSC） | 0.17 (0.05) | 需要 |
| SVM（线性核，无正则，ovo） | 0.23 (0.06) | 只需 Gram |
| 最近 medoid | 0.65 (0.07) | 只需距离 |
| 1-NN | 0.44 (0.07) | 只需 Gram |
| 最近类中心 | 0.29 (0.07) | 只需 Gram |

> **推导** · NSC 与最近类中心的关系（这是本例的数学要点）。记 $s_j$ 为特征 $j$ 的类内合并标准差，$s_0$ 为 $s_1,\dots,s_p$ 的中位数。NSC 用
>
> $$\hat\beta_j^{\star}=\big(1-\frac{s_0^2}{s_j^2}\big)_+\cdot\frac{\bar x_{kj}-\sum_{k'}\bar x_{k'j}}{\sqrt{(1-1/K)s_j^2+s_0^2}}$$
>
> 作为标准化类间均值；收缩量 $\big(1-s_0^2/s_j^2\big)_+$ 是**逐变量**的函数，所以 NSC 必须访问 $x_{ij}$。
>
> 当 $s_0\to 0$（本例 10 折 CV 选出的正是「不收缩」，即保留全部特征）时，收缩因子 $\to1$，分类分数退化为
>
> $$\frac{\bar x_{kj}}{\lVert\bar x_{\star j}\rVert_2},\qquad \bar x_{\star j}=\sqrt{\frac{1}{K}\sum_{k=1}^K\bar x_{kj}^2}$$
>
> 即**把每个变量按类中心长度 $s_j^{\rm cent}=\sqrt{\sum_k\bar x_{kj}^2}$ 标准化后再判别**。判别规则 $\arg\min_k\sum_j(x_{0j}-\bar x_{kj})^2/s_j^2$ 只用到 $x_{0j}/s_j$，等价于在标准化后的坐标里做最近类中心。

> **结果** · 表 18.3 的三行对比把「核方法的能力边界」讲清楚了：
>
> - 最近类中心（0.29）用球面度量，假设所有特征单位相同。词频里各词尺度差异极大（有的词出现几百次，有的一次也没有），球面度量因此次优。
> - NSC（0.17）即使不收缩，也因为上面的逐变量标准化而胜出。**这 0.12 的差距完全来自标准化，也就是「访问原始特征」这件事。**
> - SVM（0.23）处在中间：它估计特征的一个线性组合 $\beta=\sum_j\beta_jx_j$，对未标准化的特征有一定抵抗力，但它的对偶只用 Gram，所以拿不到标准化的好处。
> - medoid（0.65）最差：$N=16$ 且 $p=1310$，类中心的方差已经很大，medoid 的方差更大。原文的解释是「小样本 + 高维」。

> **坑** · 三个字「我们」被删掉才能公平比较，否则 HT 的摘要会因为高频使用 `we` 而被识别出来。这类看似随意的预处理在这个玩具例子里直接决定结论，说明**用词频做分类时标准化/停用词表不是可选项**。
>
> 另一个坑：表 18.3 的 NSC 一行在交叉验证里「选了不收缩」，但这只说明在这 $p=1310$ 的规模上收缩没帮助；原文补充说特征数降到 500 左右时精度损失不大，所以「不收缩」不等于「1300 个特征都有用」。

## 18.6 高维回归：监督主成分 {#s-18-6}

本节要给出一个「$p\gg N$ 时也能用的回归」构造。背景是标准 PCA 回归的两步法：先在 $X$ 的 $N$ 维行空间里找前 $m$ 个主成分 $Z=XV_m$，再对 $y$ 回归。但普通 PCA 找的是**方差大**的方向，不保证与响应相关；图 18.14 右下角显示，在淋巴瘤数据上第一个普通主成分与生存时间几乎不相关。监督主成分的做法是：**先用一元回归筛掉不相关的特征，再只在这批特征上做 PCA**。

算法 18.1 的三个步骤：

1. 对每个特征单独做标准化的一元回归，取系数（生存分析里换成 Cox 模型的 score 统计量）。
2. 对每个阈值 $\theta_1<\theta_2<\dots<\theta_K$：(2a) 取 $|\hat\beta_j|>\theta$ 的特征构成 $\tilde X$，算它的前 $m$ 个主成分；(2b) 用这些主成分建回归模型预测响应。
3. 用交叉验证选 $\theta$ 与 $m$。

数据：弥漫大 B 细胞淋巴瘤（DLBCL）240 例，$p=7399$ 个基因，160 例训练 / 80 例测试；响应是生存时间，部分右删失。CV 选出的 Cox score 阈值是 3.53，对应 27 个基因。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-6">原文 §18.6</a>

### 18.6.0 回归模型与监督投影 {#s-18-6-0}

**问题：把「找与 $y$ 相关的大方差方向」写成公式，并说明它的误差阶。** 对应 (18.32)–(18.37)。

#### 步骤 1：潜变量模型 (18.32)(18.33)

设有一个潜在变量 $U$（图 18.13 的「细胞类型」的连续版本）与响应、两组特征：

$$
Y=\beta_0+\beta_1U+\varepsilon \eqno{18.32}
$$

$$
X_j=\alpha_{0j}+\alpha_{1j}U+\varepsilon_j,\qquad j\in\mathcal{P} \eqno{18.33}
$$

其中 $\mathcal{P}$ 是「通路」（真正相关的特征）集合，其余 $X_k\ (k\notin\mathcal{P})$ 与 $U$ 独立。误差 $\varepsilon$、$\varepsilon_j$ 均值为零且与各自模型中的其它随机变量独立。

> **结果** · (18.32)(18.33) 说的是：**$Y$ 不直接依赖任何 $X_j$，它与所有 $X_j$（$j\in\mathcal P$）都是同一个潜在因子 $U$ 的函数**。这是一个单成分因子模型（§14.7）的特例，$U$ 就是「细胞类型」的连续版本。
>
> 这一步是整个方法的设计依据：既然目标是预测 $Y$，就先找出「能反映 $U$」的方向。算法 18.1 的三步因此各自有了一个明确的数学角色：
>
> | 算法步骤 | 对应的模型成分 |
> |---|---|
> | 步骤 1：逐特征一元回归系数 | 估计集合 $\mathcal P$（平均而言只有 $\alpha_{1j}\ne0$ 时系数才非零） |
> | 步骤 2a：$X_{\mathcal P}$ 的第一主成分 | 估计潜在因子 $U$（$\varepsilon_j$ 等方差高斯时它是单因子模型的极大似然估计） |
> | 步骤 2b：回归 | 估计 (18.32) 中的 $\beta_1$ |

#### 步骤 2：一元系数与监督投影 (18.34)

先把所有变量标准化（中心化、方差归一），记 $y\in\mathbb{R}^N$ 为中心化响应列，$x_j\in\mathbb{R}^N$ 为中心化的第 $j$ 个特征列。**单个特征 $j$ 的 OLS 斜率**可逐步算出：

$$
\tilde\beta_j=\frac{x_j^\top(x_j^\top y)}{x_j^\top x_j}\cdot\frac{1}{x_j^\top x_j}=\frac{\langle y,x_j\rangle}{\lVert x_j\rVert_2^2}
$$

推导：把 $x_j$ 与 $y$ 都中心化后，一元回归的正规方程是 $x_j^\top(y-\tilde\beta_jx_j)=0$，解得 $\tilde\beta_j=\frac{x_j^\top y}{x_j^\top x_j}$（预备知识 L2）。若所有特征标准化到 $\lVert x_j\rVert_2^2=N$，则 $\tilde\beta_j=\frac1N\langle y,x_j\rangle$，**所有特征的 OLS 系数都正比于同一组内积 $\langle y,x_j\rangle$**。

偏最小二乘（PLS，§3.5.2）的第一潜变量正是把这一组权重用上：

$$
z=\sum_{j\in\mathcal{P}}\langle y,x_j\rangle\, x_j \eqno{18.34}
$$

即 $z=X_{\mathcal P}(X_{\mathcal P}^\top y)$：把「$y$ 与每个特征的协方差」当作载荷，对特征取线性组合。算法 18.1 步骤 1 的筛选阈值 $\theta$ 就作用在 $|\tilde\beta_j|\propto|\langle y,x_j\rangle|$ 上。

#### 步骤 3：监督主成分方向 (18.35)

监督主成分不用 (18.34) 的权重，而用「自洽」的权重。设 $d$ 是 $X_{\mathcal P}$ 的**首奇异值**，$\hat u$ 是其左奇异向量（第一主成分方向），则

$$
\hat u=\frac{1}{d^{2}}\sum_{j\in\mathcal{P}}\langle \hat u,x_j\rangle x_j \eqno{18.35}
$$

> **推导** · (18.35) 就是主成分的定义式，逐步核验。设 $S=X_{\mathcal P}\in\mathbb{R}^{N\times p_1}$，$p_1=|\mathcal P|$。第一主成分方向 $\hat u$（$\lVert\hat u\rVert_2=1$）是 Rayleigh 商
>
> $$\max_{\lVert u\rVert_2=1}\ \frac{u^\top S^\top S u}{u^\top u}=\max_{\lVert u\rVert_2=1}\sum_{j\in\mathcal{P}}\langle u,x_j\rangle^2$$
>
> 的最大值点（预备知识 L1 的二次型极值：对固定 $u$，$u^\top(S^\top S)u=\sum_j\langle u,x_j\rangle^2$）。一阶条件：在约束切空间 $\lVert u\rVert_2=1$ 上梯度为 0，即存在 $\lambda$ 使
>
> $$2S^\top Su-2\lambda u=0\ \Rightarrow\ S^\top Su=\lambda u$$
>
> $\lambda$ 就是 $S^\top S=VSVS^\top$（$S=UDV^\top$，预备知识 L3）的首特征值，即 $d^2$。写成矩阵形式 $S^\top S\hat u=d^2\hat u$，再逐列展开：$(S^\top S)_{:j}=S^\top x_j$，而 $\hat u$ 的第 $i$ 个分量为
>
> $$d^2\hat u_i=\big(S^\top S\hat u\big)_i=\sum_{j\in\mathcal P}\langle \hat u,x_j\rangle (x_j)_i=\left(\sum_{j\in\mathcal P}\langle\hat u,x_j\rangle x_j\right)_i$$
>
> 两边对所有 $i$ 取值即得 (18.35)。**验证**：若所有 $\langle\hat u,x_j\rangle$ 同号非零，把 (18.35) 与 $\hat u$ 作内积得 $d^2=\sum_j\langle\hat u,x_j\rangle^2$，正是 Rayleigh 商的函数值 ✓。

> **结果** · (18.34) 与 (18.35) 的对比就是 PLS 与监督主成分的全部差别：
>
> - (18.34) 的权重来自**单一向量 $y$**：$\langle y,x_j\rangle$。噪声 $\varepsilon_j$ 对权重贡献 $\langle y,\varepsilon_j\rangle$，与信号 $\alpha_{1j}\langle y,U\rangle$ 量级相当（都是 $O_p(\sqrt N)$），所以权重本身噪声大。
> - (18.35) 的权重来自**数据自己的前主成分方向**，而 $\hat u$ 由 $p_1$ 个特征共同决定；噪声 $\varepsilon_j$ 在方向上被平均掉。
> - (18.35) 是**自洽**方程：$\hat u$ 既是权重来源又是待求量，这正是它比 (18.34) 稳的原因。

#### 步骤 4：误差阶 (18.36)

$$
z=u+O_p(1),\qquad \hat u=u+O_p\Big(\sqrt{\tfrac{p_1}{N}}\Big) \eqno{18.36}
$$

> **推导** · 把噪声算出来。设 $U$ 已中心化且 $\lVert U\rVert_2^2=N$、$\mathrm{Var}(\varepsilon_j)=1$，$\alpha_1=(a_1,\dots,a_{p_1})$ 且 $\lVert a_1\rVert_2^2$ 固定（信号强度）。
>
> **先算 $\hat u$ 的噪声。** 把 (18.35) 右端按 (18.33) 拆开（略去截距 $\alpha_{0j}$，因为已中心化）：
>
> $$\frac{1}{d^2}\sum_{j\in\mathcal P}\langle \hat u,x_j\rangle x_j=\frac{1}{d^2}\sum_j \big(a_j\langle\hat u,U\rangle+\langle\hat u,\varepsilon_j\rangle\big)x_j=\frac{1}{d^2}\sum_j a_j\langle\hat u,U\rangle x_j+\frac{1}{d^2}\sum_j\langle\hat u,\varepsilon_j\rangle x_j$$
>
> 噪声项记 $\eta=\frac{1}{d^2}\sum_j\langle\hat u,\varepsilon_j\rangle x_j$。取 $\hat u\approx u$（下一步验证这自洽），则 $\langle u,\varepsilon_j\rangle=\sum_{i=1}^N u_i\varepsilon_{ij}$ 是 $N$ 个独立零均值项之和，量级 $O_p(1)$。于是
>
> $$\lVert\eta\rVert_2^2=\frac{1}{d^4}\sum_{j,l}\langle u,\varepsilon_j\rangle\langle u,\varepsilon_l\rangle\langle x_j,x_l\rangle\approx\frac{1}{d^4}\sum_{j,l}\langle u,\varepsilon_j\rangle\langle u,\varepsilon_l\rangle\,a_ja_l$$
>
> 换成 $j,l$ 的和再换成 $\sum_{j,l}$ 的两重和：用 $\sum_j a_j\langle u,\varepsilon_j\rangle$ 与 $E[\langle u,\varepsilon_j\rangle\langle u,\varepsilon_l\rangle]=\delta_{jl}$，得
>
> $$\lVert\eta\rVert_2^2\approx\frac{1}{d^4}\cdot\big(\sum_j a_j\langle u,\varepsilon_j\rangle\big)^2\approx\frac{1}{d^4}\cdot\lVert a_1\rVert_2^2\,p_1=O_p\Big(\frac{p_1}{d^4}\Big)$$
>
> 而 $d^2=\lVert S^\top u\rVert_2^2$ 的主项是 $N\lVert a_1\rVert_2^2$（$\sum_j a_jx_j=a_1^\top S\approx\lVert a_1\rVert_2^2U$，长度平方 $\approx N\lVert a_1\rVert_2^4$），故 $d^2\sim N$，$d^4\sim N^2$。代入得
>
> $$\lVert\eta\rVert_2=O_p\Big(\frac{\sqrt{p_1}}{N}\Big)=O_p\Big(\sqrt{\frac{p_1}{N}}\Big)$$
>
> **这正是 (18.36) 第二项。** 因 $p_1/N\to0$，它趋于 0，$\hat u$ 对 $U$ **一致**（supervised PC 的关键性质：普通首主成分会被大量噪声特征污染，不一致）。
>
> **再算 $z$ 的噪声。** $\langle y,x_j\rangle=a_j\langle y,U\rangle+\langle y,\varepsilon_j\rangle$，其中 $\langle y,U\rangle=\beta_1\lVert U\rVert_2^2+\langle\varepsilon,U\rangle=O_p(N^{1/2})$，$\langle y,\varepsilon_j\rangle=\sum_i y_i\varepsilon_{ij}=O_p(N^{1/2})$。把 $z$ 除以 $N$（即与 $U$ 同量级）来比较：
>
> $$\frac{1}{N}\sum_j a_j\langle y,U\rangle x_j\ \big|_{\text{沿 }U\text{ 的分量}}=\frac{\langle y,U\rangle^2\lVert a_1\rVert_2^2}{N}=O_p(1)\cdot\lVert a_1\rVert_2^2$$
>
> 而噪声方向的分量是 $O_p(\sqrt{p_1/N})$ 的同一量——与 $\hat u$ 同阶。差别在于：$\hat u$ 的噪声被 $p_1$ 个特征平均掉（衰减 $\sqrt{p_1/N}$），而 $z$ 的权重只被**一个** $y$ 决定，噪声 $\langle y,\varepsilon_j\rangle$ 与信号 $\langle y,U\rangle$ 同为 $O_p(\sqrt N)$，权重里的信噪比只有 $O_p(1)$，因此在 $u$ 的归一化单位下 (18.36) 记为 $O_p(1)$。

> **坑** · (18.36) 的两个 $O_p$ 必须在**同一个归一化**下比较，且 $p_1/N\to0$ 是 (18.36) 第二项趋于零的前提。若 $p_1\asymp N$（阈值放得很松），$\sqrt{p_1/N}\asymp1$，监督主成分的优势消失；若 $p_1\gg N$，它甚至比普通主成分更差。图 18.16 中监督主成分与阈值 PLS 的误差在特征数很大时一起变差，正是这个原因。
>
> 另一处：式子里的 $\langle y,U\rangle$ 用到 $U$ 不可观测，严谨做法是用 (18.33) 的线性预测量 $\hat u$ 代替，这一步的合法性由「$p_1/N\to0$ 下 $\hat u$ 一致」提供。

#### 步骤 5：模拟数据生成模型 (18.37)

原文的对比实验 ($N=100$，$p=5000$；100 次重复) 用如下模型：

$$
x_{ij}=\begin{cases}3+\varepsilon_{ij}, & j=1,\dots,50\ \text{且}\ i\le50\\ 4+\varepsilon_{ij}, & j=1,\dots,50\ \text{且}\ i>50\\ 1.5+\varepsilon_{ij}, & j=51,\dots,250,\ (1\le i\le25\ \text{或}\ 51\le i\le75)\\ 5.5+\varepsilon_{ij}, & j=51,\dots,250,\ (26\le i\le50\ \text{或}\ 76\le i\le100)\\ \varepsilon_{ij}, & j=251,\dots,5000\end{cases} \qquad y_i=2\cdot\frac{1}{50}\sum_{j=1}^{50}x_{ij}+\varepsilon_i \eqno{18.37}
$$

其中 $\varepsilon_{ij},\varepsilon_i$ 独立、均值 0、标准差分别为 1 与 1.5。

> **结果** · 这个模型的三个部分分别对应三种失败模式：
>
> - 前 50 个基因：两组均值差 1，且与 $y$ 相关 → 监督主成分与阈值 PLS 能筛出来，lasso 也还行。
> - 第 51–250 个基因：两组均值差 4（**方差更大、逐变量信号更强**），但按 25/50/75 的模式与 $y$ **不相关** → 这一段专门惩罚「只看单个变量」的方法。它是阈值筛选（步骤 1）能成功的原因：逐变量系数小。
> - 第 251–5000 个基因：纯噪声。普通 PCA 与 PLS 都因为 $p\gg N$ 被这段污染（图 18.16 最右端两点误差最大），而监督主成分的阈值把它们先扔掉了。

### 18.6.1 与潜变量模型的联系 {#s-18-6-1}

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-6-1">原文 §18.6.1</a>

把 (18.32)(18.33) 当作真模型，逐步算出三步算法在估计什么。

1. **步骤 1 估计 $\mathcal P$。** 特征 $j$ 的总体一元回归系数（标准化后）是
   $$
   \rho_j=\mathrm{Cov}(x_j,y)=\alpha_{1j}\beta_1\mathrm{Var}(U)+\alpha_{1j}\mathrm{Cov}(U,\varepsilon)+\mathrm{Cov}(\varepsilon_j,y)=\alpha_{1j}\beta_1\mathrm{Var}(U)
   $$
   推导：$x_j=\alpha_{0j}+\alpha_{1j}U+\varepsilon_j$ 与 $y=\beta_0+\beta_1U+\varepsilon$ 的协方差只由**共同的** $U$ 项产生（其余项按假设与另一侧独立）。所以 $\rho_j=0\iff\alpha_{1j}=0$（设 $\beta_1\ne0$、$\mathrm{Var}(U)>0$），步骤 1 的阈值在平均意义上恢复 $\mathcal P$。

2. **步骤 2a 估计 $U$。** 在「$\varepsilon_j$ 独立、等方差、高斯」的假设下，$X_{\mathcal P}$ 的第一主成分方向是单因子模型的极大似然估计：其载荷 $\hat a_j=\langle\hat u,x_j\rangle$ 正是 (18.35)，而因子得分就是 $\hat u$ 本身。
   > **坑** · 这一步依赖**等方差高斯**。若 $\mathrm{Var}(\varepsilon_j)=\sigma_j^2$ 差异大，普通主成分按加权范数 $\|A\hat u\|^2=\sum_j\sigma_j^2\hat a_j^2$ 最大化，会偏向噪声大的特征；此时应该先用 §18.2 的逐变量标准化把 $s_j$ 消掉（这也正是 §18.5.3 文摘例子里 NSC 赢的原因）。

3. **步骤 2b 估计 $\beta_1$。** 把 $\hat u$ 当单一预测量做一元回归，$\hat\beta_1=\langle y,\hat u\rangle/\lVert\hat u\rVert_2^2$。

> **结果** · 监督主成分 = 「**筛选 + 因子分析 + 回归**」的三步组合。一致性来自 (18.36) 的 $O_p(\sqrt{p_1/N})\to0$：只要 $p_1/N\to0$ 且 $p_1$ 随 $p$ 增长，$\hat u\to u$，而普通首主成分会被 $p-p_1$ 个噪声特征污染，因此**不一致**。这就是原文说「$p$ 与 $p_1$ 都增长、但 $p_1$ 相对 $p$ 很小时监督主成分一致，而普通主成分可能不一致」的证明思路。
>
> 延伸：若步骤 1 的阈值留下很多特征，需要一份「稀疏近似」以便解释——那就是下一小节的预条件。

### 18.6.2 与偏最小二乘的关系 {#s-18-6-2}

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-6-2">原文 §18.6.2</a>

PLS（§3.5.2）与监督主成分的差别是**要不要扔掉噪声特征**：PLS 用 (18.34) 的权重 $\langle y,x_j\rangle$ 给每个特征一个权重（噪声特征权重小但非零），所以 $p\gg N$ 时大量噪声特征仍会污染预测。「阈值 PLS」= 先按步骤 1、2a 筛特征，再对筛出的特征做 PLS。图 18.16 显示阈值 PLS 与监督主成分接近，但监督主成分在整段范围内误差更低。

**PLS 的白化视角（把两者的差别放到同一个公式里）。** PLS 第一成分的构造等价于：先把 $X_{\mathcal P}$ 白化，再在其中挑与 $y$ 最相关的方向。取 $X_{\mathcal P}$ 的 SVD $S=UDV^\top$（$V$ 是右奇异向量，列方向即「特征组合」）。

- PLS 的权重 $w=(V^\top X^\top y)$，方向 $z=\sum_j\langle y,x_j\rangle x_j=S X^\top y$，即在**原始坐标**里用 $y$ 的投影。
- 监督主成分的权重 $w_{\star}=V_1$，即 $S^\top S$ 的首特征向量。

关键差别：$V_1$ 是 $S^\top S$ 的最大特征方向，$V^\top X^\top y$ 用的是 $y$ 直接给出的方向。两者只有在 $y\approx a_1^\top S$（即 $y$ 恰好沿主成分）时才一致。

> **推导** · 用 PLS 的对偶形式看更清楚：PLS 求解的是
>
> $$\max_w\ \frac{(y^\top Xw)^2}{\lVert Xw\rVert_2^2}\quad(\text{标准化后等价于 } \max_w \frac{\langle y,Xw\rangle^2}{\lVert Xw\rVert_2^2})$$
>
> 一阶条件：$\frac{\partial}{\partial w}\left[(X^\top y)^\top w\big/\lVert Xw\rVert^2-\lambda\right]=0$ 给出 $X^\top y=\lambda S^\top Sw$，即
>
> $$w=\frac{1}{\lambda}(S^\top S)^{-1}S^\top y=\frac{1}{\lambda}VSVS^\top X^\top y\ \text{的 $V$ 部分}=\frac1\lambda V^\top X^\top y$$
>
> （用 $S^\top S=VSVS^\top$，预备知识 L3）。代回 (18.34)：$z=Sw=\frac1\lambda\,S VV^\top X^\top y=\frac1\lambda SX^\top y$，与定义一致 ✓。
>
> 现在看 $w$ 里各分量的信噪比：$w\propto V^\top X^\top y=V(UU^\top+D\,\frac{1}{?}V^\top X^\top y)$。用 $X=UDV^\top$，$X^\top y=VDU^\top y$，故 $V^\top X^\top y=VDU^\top y$，第 $r$ 个分量为 $d_r (U^\top y)_r$。第 $r$ 个分量的信噪比由 $d_r$ 决定，而 $d_r^2$ 在 $p\gg N$ 时前几个很大、后面迅速衰减——所以 PLS 实际上把权重集中在前几个主成分上（这正是「PLS 降维」），**但它没有把噪声特征显式剔除，只是给了它们小权重**。

> **坑** · 上面这一步「前几个 $d_r$ 大、后面小」正是 PLS 的问题所在：当噪声特征数 $\gg N$ 时，$X^\top X$ 的特征向量本身已被噪声污染，前几个方向不一定对应信号。监督主成分先用逐变量阈值剔除，再在剩下的 $p_1$ 个特征里找第一主成分——**筛选与降序两件事分开做**。(18.36) 的 $O_p(\sqrt{p_1/N})$ 与 $O_p(1)$ 之差就是这个「分开做」的定量回报。
>
> 图 18.16 的最右端（普通主成分、PLS、阈值 PLS、监督主成分都用全部 5000 个基因）就是不做筛选的版本，四者误差都大。

### 18.6.3 特征选择的预条件 {#s-18-6-3}

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-6-3">原文 §18.6.3</a>

**问题：能不能兼得监督主成分的低误差与 lasso 的稀疏性？**

监督主成分的缺点：即使步骤 1 的阈值筛掉了很多特征，留下的那批里**高度相关的特征会一起被选中**（相关特征在第一主成分上的载荷必然接近），因此 $\hat u$ 往往是几百个基因的组合，没有稀疏性、难以解释。lasso（§18.4、§3.4.2）用 $\ell_1$ 惩罚自动稀疏，但直接用 lasso 会被大量噪声特征带偏——图 18.17 的绿线在远少于 50 个特征时就开始过拟合（lasso 的路径在 100 个特征处自截断，$N=100$）。

**预条件（Paul et al., 2008）的做法**：先用监督主成分对**训练集**算出预测值 $\hat y_i$（阈值由 CV 定），然后把 $\hat y$ 当作 lasso 的**响应变量**，而 lasso 仍用**全部** $p$ 个特征：

$$
\hat\beta^{\rm pre}=\arg\min_{\beta}\ \frac{1}{2N}\Big\lVert\hat y-X\beta\Big\rVert_2^2+\lambda\lVert\beta\rVert_1
$$

> **结果** · 为什么这能起作用：$X$ 的行空间是 $N$ 维的，第一主成分方向 $\hat u$ 只占一维。把响应换成 $\hat y$ 相当于先把响应投影到这个「已知有用」的一维方向上，$\hat y$ 里不再含与 $X$ 无关的响应噪声（$\hat y$ 是 $X$ 的线性函数）。于是 lasso 面对的是一个**信噪比已被提高**的回归问题，过拟合的起点被推后；等价的说法是 $\hat y$ 落在 $\mathrm{col}(X_{\mathcal P})$ 里，lasso 的最优解会自然偏向这一块。
>
> 图 18.17（$N=100$，$p=5000$）给出对照：普通 lasso（绿）在约 100 个特征处自截断（$N$ 的限制，§18.4）；监督主成分（橙）在 50 个特征处误差最低，而 50 正是模型里真实的信号特征数；预条件 lasso（紫）用约 25 个特征就达到与监督主成分相当的误差。
>
> 实践提示：预条件 lasso 的 $\lambda$ 通常按「简约性」这类主观标准选，而不是按 CV 最优——因为响应已被去噪，CV 会倾向于选过大的模型。

> **坑** · 预条件必须**只用训练集**算 $\hat y$。若用全数据算 $\hat y_i$，第 $i$ 个样本的响应里含它自己的贡献，等于把噪声回灌，$X$ 与 $\hat y$ 之间会出现人为相关，交叉验证结果偏乐观。原文没有强调这一点，但它是这类「把预测值当响应」的做法（也叫 stacked generalization）的通用陷阱。

---

## 18.7 特征评估与多重检验问题 {#s-18-7}

本章前半的四个小节都在做**预测**；本节转向**推断**：判断 $M$ 个特征里哪些真的有信号。这是传统的多重假设检验问题，因此从这一节起原文用 $M$ 代替 $p$ 表示特征数（避免与 $p$ 值混淆）。

引例是辐射敏感性微阵列数据（Rieger et al., 2004）：$M=12625$ 个基因，58 个样本（44 个正常反应，14 个严重反应）。目标不再是预测某个病人有没有癌症，而是**找出表达量在两组之间不同的基因**——评估单个特征的重要性，且不用任何多元预测模型。

<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-7">原文 §18.7</a>

### 18.7.0 从 $t$ 统计量到多重检验 {#s-18-7-0}

#### 逐基因的 $t$ 统计量 (18.38)(18.39)

对每个基因 $j$ 构造两样本 $t$ 统计量。记 $C_\ell$ 为第 $\ell$ 组样本下标集合（$|C_1|=N_1=44$，$|C_2|=N_2=14$），$\bar x_{\ell j}=\frac{1}{N_\ell}\sum_{i\in C_\ell}x_{ij}$：

$$
t_j=\frac{\bar x_{2j}-\bar x_{1j}}{s e_j} \eqno{18.38}
$$

$$
s e_j=\hat\sigma_j\sqrt{\tfrac{1}{N_1}+\tfrac{1}{N_2}},\qquad \hat\sigma_j^2=\frac{1}{N_1+N_2-2}\left(\sum_{i\in C_1}(x_{ij}-\bar x_{1j})^2+\sum_{i\in C_2}(x_{ij}-\bar x_{2j})^2\right) \eqno{18.39}
$$

> **推导** · (18.39) 的两个部分来源完全不同，必须分开算。
>
> **（a）根号里的部分**来自均值的方差。设两组各有独立同方差误差 $\sigma_j^2$，则
>
> $$\mathrm{Var}\big(\bar x_{1j}-\bar x_{2j}\big)=\mathrm{Var}(\bar x_{1j})+\mathrm{Var}(\bar x_{2j})=\frac{\sigma_j^2}{N_1}+\frac{\sigma_j^2}{N_2}$$
>
> 推导：$\bar x_{1j}=\frac{1}{N_1}\sum_{i\in C_1}x_{ij}$，故 $\mathrm{Var}(\bar x_{1j})=\frac{1}{N_1^2}\sum_{i\in C_1}\sigma_j^2=\frac{\sigma_j^2}{N_1}$（用了方差可加性与独立同分布，预备知识 P1）。而 $\bar x_{1j}$ 只由 $C_1$ 的样本组成、$\bar x_{2j}$ 只由 $C_2$ 组成，两组样本独立，故交叉项协方差为 0，$\mathrm{Var}(A-B)=\mathrm{Var}(A)+\mathrm{Var}(B)-2\mathrm{Cov}(A,B)=\mathrm{Var}(A)+\mathrm{Var}(B)$。标准化分母因此是 $\sigma_j\sqrt{1/N_1+1/N_2}$。
>
> **（b）$\hat\sigma_j^2$** 是 $\sigma_j^2$ 的无偏估计，用**合并组内平方和**：$SS=\sum_{i\in C_1}(x_{ij}-\bar x_{1j})^2+\sum_{i\in C_2}(x_{ij}-\bar x_{2j})^2$。抽掉两个均值各消耗 1 个自由度，故自由度 $N_1+N_2-2$，即 (18.39) 的分母 $N_1+N_2-2$。
>
> **（c）检验的精确分布。** 在 $H_{0j}$（两组均值相等）下 $t_j\sim t_{N_1+N_2-2}$（预备知识 P4：分子正态、分母为 $\sigma_j$ 的 $\chi^2_{\nu}/\nu$）。所以逐个检验的 $p$ 值可以直接查 $t$ 分布表，无需任何假设。

#### 逐个 5% 水平会造成什么

实际数据里 12625 个 $t_j$ 全部落在 $[-4.7,5.0]$，其中 1189 个满足 $\lvert t_j\rvert\ge2$。逐个按 5% 水平判定时：

$$
\#\{\text{假阳性}\}\ \overset{d}{=}\ \mathrm{Bin}\big(M,\alpha\big),\qquad E\big[12625\times0.05\big]=631.3,\qquad \mathrm{SD}=\sqrt{12625\times0.05\times0.95}=\sqrt{599.7}\approx24.5
$$

（这个 Bin 分布只在「各基因独立」下成立，原文明确说基因显然不独立，但这给出了一个下限性的对照。）观测到的 1189 比 631 高出 22 个标准差。

> **坑** · 上面这个对照容易被读反。631 是「**期望的**假阳性数」，1189 是「**全部的**显著基因数」。两者不等价：1189 里既有真阳性也有假阳性，数量本身无法区分。原文的论证方式是「即使分组与基因完全无关也会有 631 个」——它说明的是**逐个 5% 的阈值在大 $M$ 下完全不可用**，而不是说数据里的信号很强。

#### 置换 $p$ 值 (18.40)(18.41)

理论 $t$ 分布依赖正态性（且 (18.39) 是逐基因独立的估计）。更稳妥的做法是**置换**：把 58 个样本的标签随机重排，重算 $t$ 统计量，记 $t^k_j$ 为第 $k$ 次置换下基因 $j$ 的值。单个基因的置换 $p$ 值：

$$
p_j=\frac{1}{K}\sum_{k=1}^{K}I\big(\lvert t^k_j\rvert>\lvert t_j\rvert\big) \eqno{18.40}
$$

利用「所有基因都在同一把尺子上测量」这一事实，可以把**全部 $M$ 个基因**的置换统计值汇总成一个零分布：

$$
p_j=\frac{1}{MK}\sum_{j'=1}^{M}\sum_{k=1}^{K}I\big(\lvert t^k_{j'}\rvert>\lvert t_j\rvert\big) \eqno{18.41}
$$

> **结果** · (18.41) 相对 (18.40) 的两个好处：
>
> 1. **分辨率更细**。(18.40) 的分母是 $K$（本例 $K=1000$），可分辨的最小 $p$ 值是 $1/1000=0.001$；(18.41) 的分母是 $MK=1000\times12625$，可分辨到 $8\times10^{-8}$。这正是本例里 $p_{(11)}=0.00012$ 能被算出来的原因——若用 (18.40)，1/1000 的粗度会让排序在 $p=0.001$ 附近饱和。
> 2. **不依赖分布假设**。置换零分布对任意单变量分布都有效，只要在标签置换下误差分布保持不变（交换性）。基因表达经对数变换后大致正态，但也有例外；置换法绕开这个问题。
>
> $\binom{58}{14}\approx10^{13}$ 种置换无法枚举，实际只抽 $K=1000$ 次随机置换（图 18.18 的蓝色直方图就是这 1000 次汇总的零分布）。

要检验的假设族是

$$
H_{0j}:\ \text{治疗对基因 }j\text{ 无作用}\qquad \text{对比}\qquad H_{1j}:\ \text{治疗对基因 }j\text{ 有作用} \eqno{18.42}
$$

逐个在水平 $\alpha$ 拒绝（$p_j<\alpha$）时，第一类错误率就是 $\alpha$：$\Pr(A_j)=\alpha$，$A_j$ 是「$H_{0j}$ 被错误拒绝」的事件。

> **基础知识** · 两种总体误差率（表 18.5）
>
> | | 未显著 | 显著 | 合计 |
> |---|---|---|---|
> | $H_0$ 为真 | $U$ | $V$ | $M_0$ |
> | $H_0$ 为假 | $T$ | $S$ | $M_1$ |
> | 合计 | $M-R$ | $R$ | $M$ |
>
> - **族系误差率** $\mathrm{FWER}=\Pr(V\ge1)=\Pr\big(\cup_{j=1}^M A_j\big)$：至少犯一次错。
> - 第一类错误率 $=\frac{E[V]}{M_0}$；第二类错误率 $=\frac{E[T]}{M_1}$；功效 $=1-\frac{E[T]}{M_1}$。

#### FWER 在 $M$ 大时为什么不可用

若 $M$ 个检验独立且各自水平 $\alpha$，由独立性（概率乘法）：

$$
\mathrm{FWER}=\Pr\Big(\bigcup_{j=1}^M A_j\Big)=1-\Pr\Big(\bigcap_{j=1}^M A_j^c\Big)=1-(1-\alpha)^M
$$

$M=12625$、$\alpha=0.05$ 时这几乎等于 1。若检验正相依（$\Pr(A_j\mid A_k)>\Pr(A_j)$，基因数据里很常见），FWER 略小于它但仍随 $M$ 趋于 1。

Bonferroni 校正把每个检验压到 $\alpha/M$。由并集上界（Boole 不等式，**任意相依结构都成立**）：

$$
\mathrm{FWER}=\Pr\Big(\bigcup_{j=1}^M A_j\Big)\le\sum_{j=1}^M\Pr(A_j)=M\cdot\frac{\alpha}{M}=\alpha
$$

**代价是极其保守**：$M=12625$ 时阈值 $0.05/12625=3.96\times10^{-6}$，而 12625 个基因里最小 $p$ 值约 $8\times10^{-5}$——**一个基因也检不出来**。

> **结果** · 关键观察：Bonferroni 强行让 $E[V]=\alpha M_0\le\alpha M$ **与 $M$ 无关**。问题不在于假阳性太多（631 个相对于 12625 完全可以接受），而在于我们要求「一个都不许错」。**只要允许「报出的基因里有 15% 是错的」，$M$ 增大带来的功效损失就被抵消。** 这就是 FDR 的立足点。

> **衔接** · §18.7.1 以后（(18.43)–(18.54)：FDR 的定义、BH 规则的证明、SAM、pFDR 的后验解释）由分片 `18c-multipletesting.md` 承担；本节到此为止，只把多重检验的问题背景与逐特征 cutoff 的失败原因讲完。

