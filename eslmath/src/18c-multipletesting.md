## 18.7.1 错误发现率 {#s-18-7-1}

上一节（<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-7">原文 §18.7</a> 前半）已经交代了逐基因的 $t$ 统计量 (18.38)、合并标准误 (18.39)、置换 $p$ 值 (18.40)(18.41) 以及假设族 $H_{0j}$ 对比 $H_{1j}$ (18.42)，并算出「$M=12625$ 个检验、每个在 $\alpha=0.05$ 水平上做」时族系误差率 FWER $\approx1-(1-\alpha)^M\approx1$，Bonferroni 校正则把阈值压到 $0.05/12625=3.96\times10^{-6}$ 从而一个基因也检不出来。本节解决这个两难：换一个**整体误差指标**，使「允许报出 $R$ 个基因、其中错 $\alpha$ 份」成为可接受的目标。

### 定义与「为什么是比例」

设 $V$ 是被错误拒绝（假发现）的个数，$R$ 是全部被拒绝（发现）的个数。$R$ 是**随机**的：数据好的时候它大，数据差的时候它小。所以「错误比例」必须写成

$$
\mathrm{FDR}=E\left[\frac{V}{R}\right]\qquad \text{约定 } R=0 \text{ 时 } V/R=0 \eqno{18.43}
$$

对<u>被叫显著的那 $R$ 个基因</u>取期望（在产生数据的那次重复试验的意义上取）。

> **基础知识** · 四条要反复用的概率工具
>
> - **（预备知识 P5）** $p$ 值：$p_j$ 在 $H_{0j}$ 为真时满足 $\Pr(p_j\le c)\le c$；若精确均匀则等号成立。等价的 $t$ 统计量表述是「在水平 $c$ 上拒绝」，对偶的「每检验水平」是 $\alpha=F_0(|T|\ge c)$。
> - **（预备知识 P2）** 全期望公式 $E[X\mid g]=E[E[X\mid g']]$（对任意划分 $g'$）与条件期望的线性性 $E[\sum_j f_j]=\sum_j E[f_j]$。
> - **（预备知识 P4）** $t$、$\chi^2$、$F$ 的抽样分布：$t$ 统计量 (18.38) 在正态总体下精确服从 $t_{N_1+N_2-2}$；置换零分布 (18.40)(18.41) 是它不依赖正态性的替代品。
> - **（预备知识 P1）** 大数定律 / 依概率收敛：$M$ 个 i.i.d. 分量之和的相对波动以 $1/\sqrt{M}$ 衰减，这正是后文把 $\hat\pi_0$ 的估计误差说成 $O_p(M^{-1/2})$ 的依据。

#### 推导 · FWER 与 FDR 的量级对比（为什么必须换指标）

1. **FWER 随 $M$ 线性爆炸。** 独立时 $M_0$ 个真零假设各自以水平 $\alpha$ 被错误拒绝，$V\sim\mathrm{Binomial}(M_0,\alpha)$，故
   $$
   \mathrm{FWER}=\Pr(V\ge1)=1-(1-\alpha)^{M_0}\approx M_0\alpha\ \ (\text{小 }\alpha)
   $$
   Bonferroni 就是把这一个乘积里的 $M_0$ 强行设成 $M$，代价是功效全丢。

2. **FDR 在同一水平 $\alpha$ 上只差一个 $\pi_0$ 因子。** 由条件期望的线性性，对 $M_0$ 个真零假设逐个求和：
   $$
   E[V]=\sum_{j=1}^{M}\Pr(H_{0j}\text{ 真且被拒})=\sum_{j\in\mathcal N}\Pr(p_j\le c)=M_0\,F_0(c)\ \text{（精确均匀时）}=M_0\alpha
   $$
   而 $E[R]=\sum_{j=1}^M\Pr(p_j\le c)\ \ge\ M_0\alpha$。于是
   $$
   \mathrm{FDR}\ \overset{(18.47)}{\approx}\ \frac{E[V]}{E[R]}\ \le\ \frac{M_0\alpha}{E[R]}\ =\ \alpha\,\frac{M_0}{E[R]}\ \le\ \alpha\frac{M_0}{M}=\pi_0\alpha
   $$
   （最后一步用 $E[R]\le M$，且 $\pi_0=M_0/M$）。**注意分母 $E[R]$ 随 $M$ 一起长**：允许假阳性的「预算」$\alpha R$ 与报出的基因数成正比，而 FWER 要求预算 $\le\alpha$ 恒定。

3. **两个量的比。** 同一目标 $\alpha$ 下，$\mathrm{FWER}\approx M_0\alpha$ 而 $\mathrm{FDR}\approx\pi_0\alpha=M_0\alpha/M$，比值 $\approx M$。$M=12625$ 时，若每个检验都在 5% 水平上做，FWER 已经饱和到 1，而「报出的基因里 5% 是错的」这个目标始终可达（只要真有信号让 $E[R]$ 长大）。

4. **最坏情形：全部零假设都为真。** 此时 $V\equiv R$，所以无论怎么调截断点，$\mathrm{FDR}\equiv1$（报出的每一个都是错的，确实如此），而 $\mathrm{FWER}=1-(1-\alpha)^M\to1$ 是**渐近**地趋于 1。区别在于 FDR 给出的是**恒定可比**的刻度：全零时任何阈值都给 1，一旦有信号就会下降；FWER 在全零时已经饱和，无法区分「略多一点信号」和「全错」。

> **结果** · FDR 把「一个都不许错」（FWER）换成「十个里错一个」（FDR）。基因数据里 $\alpha$ 取到 0.15 完全合理——报 11 个基因里错 1.5 个，对探索性研究是划算的。

### Benjamini–Hochberg 步骤与它的证明

记排好序的 $p$ 值为 $p_{(1)}\le p_{(2)}\le\cdots\le p_{(M)}$。BH 规则（Algorithm 18.2）：取

$$
L=\max\Big\{j:\ p_{(j)}<\alpha\cdot\frac{j}{M}\Big\},\qquad \text{然后拒绝一切 } p_j\le p_{(L)} \eqno{18.44}
$$

即在「$p$ 值对 $j$ 作图 + 斜率 $\alpha/M$ 的直线」的下方找**最后一个**落点。

> **推导** · **先说清楚这一步哪里需要小心**。BH 的结论 $\mathrm{FDR}\le\alpha$ 在原书里是直接给出的，但它的标准证明**不能**写成「把随机分母换成常数」或「$p_j$ 与 $p_{(L)}$ 独立」——这两条都是错的：
>
> - $L=\#\{p_j\le q\}$ 本身是**随机变量**（$q$ 由数据定），不能当成常数塞进 $\mathbb{E}[\,\cdot\,]$；
> - 对任何**落在前 $L$ 名之内**的零假设，事件 $\{p_j\le p_{(L)}\}$ 是恒真的，所以 $p_j$ 与 $p_{(L)}$ **必然不独立**。
>
> 正确的做法是**换一种表述**：BH 的控制量不是 $\mathbb{E}[V/L]$，而是「前 $k$ 个发现里假发现的**期望个数**」。这一步把随机阈值固定成了 $p_{(k)}$，独立性才成立。
>
> **第 1 步：把控制目标换成期望假发现数。** BH 要证的是：对任何 $k$（只要接受集非空）
>
> $$E[V_{(k)}]\le \alpha\,\frac{M_0}{M}\,k\ \le\ \alpha k \qquad\bigl(V_{(k)}=\text{前 }k\text{ 个里的假发现数}\bigr)$$
>
> **第 2 步：逐个零假设算它进前 $k$ 名的概率。** $j\in\mathcal N$（$\mathcal N$ 是 $M_0$ 个真原假设）进入前 $k$ 名，当且仅当 $p_j\le p_{(k)}$。因为 $k$ 是**事先固定**的数，$p_{(k)}=\big(P_{(1)},\dots,P_{(k)}\big)$ 这一族与 $p_j$ 独立（独立性假设下，$p_{(k)}$ 只由原假设下的 p 值决定），所以
>
> $$\Pr\big(j\in\text{前 }k\big)=E\big[\Pr\big(p_j\le p_{(k)}\mid p_{(k)}\big)\big]=E\big[F_0\big(p_{(k)}\big)\big]=E\big[p_{(k)}\big]$$
>
> **这是唯一用到独立性的地方**，也正是它必须先「把 $k$ 固定」的原因。
>
> **第 3 步：用 $p$ 值的均匀性。** $p_{(k)}\le\alpha k/M$（BH 的选取规则：取最大的 $k$ 满足它），于是
>
> $$E\big[V_{(k)}\big]=\sum_{j\in\mathcal N}\Pr\big(j\in\text{前 }k\big)=M_0\,E\big[p_{(k)}\big]\le M_0\frac{\alpha k}{M}$$
>
> **第 4 步：回到 FDR。** 接受集非空时 $\mathrm{FDR}=\mathbb{E}[V_{(k)}]/k$，故
>
> $$\mathrm{FDR}\ \le\ \frac{M_0}{M}\,\alpha\ \le\ \alpha \qquad\blacksquare\ \eqno{18.45}$$
>
> **逐步可验**：第 2 步把随机阈值换成**事先固定的** $p_{(k)}$（这一步同时用到了独立性）；第 3 步用 $p$ 值在原假设下服从$U(0,1)$；第 4 步用 $L=k$。

第 1 步用常数 $L$ 替代随机分母，是「BH 是 FDR 意义下阈值固定」这一现象的根源：**一旦选定了 $L$，截断点就是确定的 $p_{(L)}$，而 $p_{(L)}\le\alpha L/M$ 恰好保证「报 $L$ 个、错不超过 $\alpha L$」**。把它写成局部形式（这是原书证明的核心）：对任何满足 $p_{(k)}\le\alpha k/M$ 的 $k$，

$$
E\big[V_k\big]\ \le\ M_0\,E\big[F_0(p_{(k)})\big]\ \le\ M_0\,\frac{\alpha k}{M}\ \le\ \alpha k
$$

即**「前 $k$ 个发现里的假发现期望数不超过 $\alpha k$ 乘以零假设比例」**——这就是为什么排序后「前 $k$ 个」这个说法在 FDR 意义下是自洽的。

> **坑**
> - 证明第 2 步的独立性不可去。检验正相依时 BH 仍可能超调；稳健版本（Benjamini–Yekutieli）把阈值换成 $\alpha/\sum_{i=1}^M 1/i\approx\alpha/\log M$。
> - $\pi_0$ 因子是「向上偏」还是「向下偏」取决于估计：把 $M_0$ 当成 $M$ 会高估 FDR（保守），把 $M_0=0$ 代入则完全失控。

### 直接可算的插入式估计：算法 18.3

BH 需要 $p$ 值；算法 18.3 完全不用 $p$ 值，直接在 $t$ 统计量上取截断点 $C$：

$$
R_{\mathrm{obs}}(C)=\sum_{j=1}^{M}I\big(\lvert t_j\rvert>C\big),\qquad
\hat E[V](C)=\frac{1}{K}\sum_{k=1}^{K}\sum_{j'=1}^{M}I\big(\lvert t^k_{j'}\rvert>C\big),\qquad
\widehat{\mathrm{FDR}}(C)=\frac{\hat E[V](C)}{R_{\mathrm{obs}}(C)} \eqno{18.46}
$$

数值例子（$C=4.101$）：$R_{\mathrm{obs}}=11$；1000 次置换里超过 $C$ 的值共 1518 个，即每次置换平均 $1518/1000=1.518$ 个；故 $\widehat{\mathrm{FDR}}=1.518/11\approx0.14$，与 $\alpha=0.15$ 相差来自离散性。

> **推导** · 为什么 $\hat E[V]$ 与 $E[V]$ 之间差一个 $M/M_0$，以及 BH 与插入式估计的等价
>
> 1. **两个分布不一样，这是关键。** 对基因 $j$ 打乱标签后 $t^k_j$ 的分布是 $F_{0j}$（该基因的**条件零分布**：置换消灭了分组效应，只剩下噪声），而**观测**的 $t_j$ 的边缘分布是混合分布 $G_j=\pi_0F_{0j}+(1-\pi_0)F_{1j}$。所以
>    $$
>    E\big[\hat E[V](C)\big]=\frac{1}{K}\sum_{k=1}^{K}\sum_{j'=1}^{M}F_{0j'}(C)=M\bar F_0(C),\qquad \bar F_0=\frac1M\sum_{j=1}^{M}F_{0j}
>    $$
>    而 $E[V(C)]=\sum_{j\in\mathcal N}F_{0j}(C)=M_0\bar F_0(C)$（若 $F_{0j}$ 齐次则等号直接成立，否则右侧是它的带符号平均）。相除得 $\hat E[V](C)$ 估计的是 $\frac{M}{M_0}E[V]$。
> 2. **分子分母同分布不等于 FDR 为 1。** 注意 $E[R_{\mathrm{obs}}(C)]=\sum_j G_j(C)=\pi_0M\bar F_0(C)+(1-\pi_0)M\bar F_1(C)$，用的是**混合**分布；而 $M\bar F_0(C)$ 用的是**纯零**分布。$F_1$ 的尾巴比 $F_0$ 厚得多（例子里 11 个观测值 vs 1.5 个纯零值），两个数的差正是「真信号」的量。
> 3. **BH 与算法 18.3 等价（习题 18.17）。** BH 的截断点 $c=\lvert T\rvert_{(L)}$ 是第 $L$ 大的 $\lvert t_j\rvert$，对应的合并置换 $p$ 值 (18.41) 为 $p_{(L)}=\#\{(k,j'):\lvert t^k_{j'}\rvert>c\}/(MK)$。由 $L$ 的定义 $p_{(L)}\le\alpha L/M$，代入 (18.46)
>    $$
>    \widehat{\mathrm{FDR}}(c)=\frac{\frac{1}{K}\#\{(k,j'):\lvert t^k_{j'}\rvert>c\}}{L}=\frac{M\,p_{(L)}}{L}\le\frac{M\cdot\alpha L/M}{L}=\alpha .
>    $$
>    反过来，把截断点降到第 $L+1$ 大的 $\lvert t_j\rvert=\lvert T\rvert_{(L+1)}$：此时报出的基因占 $R/M=(L+1)/M$，对应的置换 $p$ 值恰是 $p_{(L+1)}$，故 $\widehat{\mathrm{FDR}}=p_{(L+1)}/\big((L+1)/M\big)$；由 $L$ 的**最大性** $p_{(L+1)}>\alpha\frac{L+1}{M}$，得 $\widehat{\mathrm{FDR}}>\alpha$。这就是习题 18.17(b)：跨过阈值的下一步立刻破坏 $\alpha$，所以 $L$ 是唯一自洽的选择。
> 4. **理论版与估计版对接。** 由第 1 步，$\widehat{\mathrm{FDR}}\approx\frac{M}{M_0}\mathrm{FDR}$，故 $\mathrm{FDR}\approx\hat\pi_0\cdot\widehat{\mathrm{FDR}}$，其中 $\hat\pi_0=\hat M_0/M$。取 $\hat\pi_0=1$（即 $\hat M_0=M$）是最保守的向上偏估计；有了 $\pi_0$ 的估计，就能把 $\mathrm{FDR}$ 本身估得更准，并通过 (18.45) 反过来改进 BH 阈值。

### 插入式估计所依赖的近似

整个 (18.46) 只用到一步「期望之比代替期望之商」：

$$
E\left[\frac{V}{R}\right]\ \approx\ \frac{E[V]}{E[R]} \eqno{18.47}
$$

> **推导** · 什么时候这一步合法
>
> 1. **精确的偏差项。** 记 $\bar V=E[V]$、$\bar R=E[R]$，则两个「交叉项」正好抵消：
>    $$
>    E\left[\frac{V}{R}\right]-\frac{\bar V}{\bar R}=E\left[(V-\bar V)\left(\frac{1}{R}-\frac{1}{\bar R}\right)\right]
>    $$
>    因为 $E[(V-\bar V)]\cdot E[1/R]=0$ 且 $E[(V-\bar V)]/\bar R=0$，两边都只剩这一个乘积项（此处默认 $R>0$ 几乎必然；否则两式都要按 $\Pr(R>0)$ 条件化）。
> 2. **Cauchy–Schwarz 给它一个上界**：
>    $$
>    \left|E\left[\frac{V}{R}\right]-\frac{E[V]}{E[R]}\right|\ \le\ \sqrt{\mathrm{Var}(V)}\,\sqrt{\mathrm{Var}(1/R)}
>    $$
>    所以只要 $R$ 集中（$\mathrm{Var}(R)=O(\bar R^2/M)$，由 $V,R$ 是 $O(M)$ 个 i.i.d. 分量之和，预备知识 P1），$1/R$ 也集中，偏差 $\to0$。
> 3. **具体到 $p\gg N$ 的基因实验**：$R$ 是「$M$ 个基因里显著基因数」的估计，它本身是 $O(M)$ 规模的和，集中性来自 $M$ 而不是 $N$。所以插入式估计的误差量级是 $O_p(1/\sqrt{M})$；$M=12625$ 时约 $0.9\%$ 的相对误差，远小于 $\alpha$ 的选择本身带来的不确定性。
> 4. **一致性**：$\widehat{\mathrm{FDR}}(C)$ 是 $\frac{M}{M_0}\mathrm{FDR}$ 的一致估计（Storey 2002），故 (18.47) 在 $K$ 与 $M$ 同时增大时是渐近成立的，不是渐近有偏。

> **坑** · $K$ 太小会让 (18.46) 的分子粗糙：$K=1000$ 时分子的标准差约为 $\sqrt{1.518}\approx1.2$，除以 $R_{\mathrm{obs}}=11$ 就是 0.11 的相对误差，恰好和要估的 0.14 同量级。所以图 18.20 里 SAM 要画出多个 $\Delta$ 的 $\widehat{\mathrm{FDR}}$ 曲线让人挑，而不是信一个点。

### $\pi_0$ 的估计

分母里的 $M/M_0$、以及 (18.45) 里的 $\pi_0$ 因子都要求知道「有多少个零假设是真的」。原书把它放进练习 18.19(c)，这里补上推导。置换零分布的中位数与四分位数是最自然的选择，因为**零假设下 $t$ 分布对称，中心一半区间里几乎不含信号**：

$$
\hat\pi_0=\frac{\#\{j:\ t_j\in(q_{.25},q_{.75})\}}{\tfrac12 M},\qquad
\hat\pi_0\leftarrow\min(\hat\pi_0,1)
$$

> **推导** · 分位数法为什么自洽
>
> 1. 记 $F_0,F_1$ 为条件零/备择分布的 cdf，$F=\pi_0F_0+(1-\pi_0)F_1$ 为经验分布（用 $t_j$ 估）。设零分布的中位数 $0$、四分位距 $2\sigma_0$（即 $F_0^{-1}(0.75)-F_0^{-1}(0.25)$），令 $\delta=\Phi(0.75)-\Phi(0.25)=0.5$ 对高斯。
> 2. **中心区间的质量分解**（全概率公式）：
>    $$
>    F(q_{.75})-F(q_{.25})=\pi_0\big[F_0(q_{.75})-F_0(q_{.25})\big]+(1-\pi_0)\big[F_1(q_{.75})-F_1(q_{.25})\big]\ \ge\ \pi_0\cdot\delta
>    $$
>    因为第二项非负。用经验分布估左端（经验四分位数差 $\approx0.5$，因 $t$ 大致对称），得 $\hat\pi_0\cdot0.5\lesssim0.5$，即 $\hat\pi_0\lesssim1$——**这一步保证了不系统性高估**。
> 3. **信号越少越准**：若 $\#\{j:t_j\in(q_{.25},q_{.75})\}\approx\tfrac12M$（信号都跑到极端尾部），则 $\hat\pi_0\approx1$，正确。
> 4. **与 (18.47) 联动**：把 $\hat\pi_0$ 乘到 $\widehat{\mathrm{FDR}}$ 上即得 $\mathrm{FDR}$ 的更好估计；若 $\hat\pi_0<1$，还可以用关系 (18.45) 把有效水平放大——把「允许的 $M_0$ 个假阳性预算」从 $\alpha M$ 缩到 $\alpha M_0$，功效提高。
> 5. **更平滑的版本（Storey 2002）**：取分位点 $q_\lambda$，令
>    $$
>    \hat\pi_0(\lambda)=\frac{\#\{j:\ t_j>q_\lambda\}}{M(1-\lambda)},\qquad \hat\pi_0=\min_{0\le\lambda<1}\hat\pi_0(\lambda)
>    $$
>    理由同上：$F(1-\lambda)\ge\pi_0\lambda$。检验的逻辑是**若 $\hat\pi_0(\lambda)>1$ 说明这个 $\lambda$ 太小（分母被低估），该 $\lambda$ 不可信**；对 $t$ 这类对称统计量 $\lambda=0.5$ 附近最稳，所以 (18.19(c) 用四分位数。

> **结果** · 三条可操作的规则对照
>
> | 规则 | 控制什么 | 阈值 | $M=12625,\ \alpha=0.05$ 时 |
> |---|---|---|---|
> | Bonferroni | $\mathrm{FWER}\le\alpha$ | $p<\alpha/M$ | 阈值 $4\times10^{-6}$，零发现 |
> | BH | $\mathrm{FDR}\le\pi_0\alpha$ | $p_{(j)}<\alpha j/M$ | 沿斜线取交点 |
> | SAM | $\widehat{\mathrm{FDR}}$（可调） | $\lvert t_{(j)}-\tilde t_{(j)}\rvert>\Delta$ | $\Delta=0.71$ 时 11 个发现 |

## 18.7.2 非对称截断点与 SAM {#s-18-7-2}

BH 只用 $\lvert t_j\rvert$，即对 $t_j$ 与 $-t_j$ 用同一个阈值。但很多生物学实验里差异表达基因几乎全是单向的（上调），此时把阈值对称化会白白丢掉功效。原书的做法（SAM）是让阈值随位置变化，见 <a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-7-2">原文 §18.7.2</a>。

### 置换零分布与两条期望曲线

SAM 图（图 18.20）的两个坐标都来自排序统计量。记 $t^k_j$ 为第 $k$ 次置换得到的统计量，$t^k_{(1)}\le\cdots\le t^k_{(M)}$ 为它在 $M$ 个基因上的顺序统计量，则

$$
t_{(1)}\le\cdots\le t_{(M)},\qquad \tilde t_{(j)}=\frac{1}{K}\sum_{k=1}^{K}t^k_{(j)}
$$

横坐标 $\tilde t_{(j)}$ 是**零假设下第 $j$ 小的统计量的期望**（原书图中的平均水平），纵坐标 $t_{(j)}$ 是观测值。

> **推导** · 置换零分布怎么造，以及它为什么保留了基因间的相关结构
>
> 1. **构造。** 把 $N$ 个样本的分组标签 $\boldsymbol{g}=(g_1,\dots,g_N)\in\{0,1\}^N$ 随机重排 $K$ 次，第 $k$ 次得到 $\boldsymbol{g}^{(k)}$，用它重算 (18.38) 得到 $t^k_1,\dots,t^k_M$。零分布的 $K$ 个取值就是 $\lvert t^k_j\rvert$（逐基因版 (18.40)）或全部 $MK$ 个（合并版 (18.41)）。$K=1000$、从 $\binom{58}{14}\approx10^{13}$ 种置换中随机抽。
> 2. **为什么可以这样。** 检验的零假设是「$Y_i$ 与 $\boldsymbol{x}_i$ 独立」。置换零假设 $g^{(k)}\perp\boldsymbol{x}$，是在保持 $\boldsymbol{x}$ 的分布结构的前提下构造的**重抽样分布**：$Y^{(k)}$ 的联合分布与 $Y$ 在零假设下给定 $\boldsymbol{x}$ 的条件分布同分布，所以 $t^k$ 的经验分布是 $t$ 在零假设下的（渐近）抽样分布，即 $p$ 值的精确替代。
> 3. **为什么真实信号与相关结构被保留。** 置换只打乱**标签**，特征矩阵 $\boldsymbol{X}$ 一不动。基因之间的相关结构（$p\gg N$ 时必然存在的共线性）因此被**完整保留**——若换成「重抽样基因」或「假定独立」的 Bootstrap，这些结构就被破坏，而 $t$ 分布的近似恰恰是被它们破坏的。任何**与标签无关**的成分（均值平台、量纲、批次效应）都被吸收进零分布。反过来，打乱标签不会**制造**也不会**消除**信号，只是把「哪些基因真的有差异」这个信息从标签里抽掉，所以 $F_1$ 的厚尾在置换后消失——这正是 11 vs 1.5 的来源。最后，置换后 $t^k_j$ 的分布是 $F_{0j}$（该基因的条件零分布，见上节第 1 步），因为分组效应被消灭。
> 4. **精确性的极限。** 置换检验是**渐近**精确的：小 $p$ 值处有限 $K$ 的分辨率只有 $1/K=10^{-3}$。例子里 $p_{(11)}=0.00012$ 已经小于 $1/1000$，所以这个 $p$ 值实际只能由 $K\ge10^4$ 才估得准，这也是 BH 在 $j=11$ 处被卡住的原因之一。
> 5. **横坐标的可算性。** $\tilde t_{(j)}$ 是 $K$ 个顺序统计量的平均，故 $\mathrm{Var}(\tilde t_{(j)})=\frac1K\mathrm{Var}\big(t^k_{(j)}\big)$：$K$ 越大，零分布曲线越平滑，SAM 的截断点越稳定。

### 与似然比检验的类比，及 SAM 的规则

若在零假设下统计量的对数似然为 $\ell_0(t_j)$、备择下为 $\ell(t_j)$，似然比检验（预备知识 P3 指数族、P5）拒绝零假设当且仅当

$$
\ell(t_j)-\ell_0(t_j)>\Delta \eqno{18.48}
$$

由于 $\ell_0$ 与 $\ell$ 一般不对称，这个不等式在 $t_j$ 轴上给出的阈值对 $t_j$ 和 $-t_j$ **不一样**——非对称性不是人为规定的，而是似然比的自然结果。SAM 把这个想法照搬过来：

$$
\big\lvert t_{(j)}-\tilde t_{(j)}\big\rvert>\Delta \eqno{18.49}
$$

每个 $t_{(j)}$ 的阈值取决于它对应的零值 $\tilde t_{(j)}$。做法是：从原点往右移动，找**基因第一次离开带宽 $\Delta$** 的位置，该处的 $t_{(j)}$ 就是上截断点 $C_{\mathrm{hi}}$，它右边的全部基因判显著（图中标红）；同理找左下角的 $C_{\mathrm{low}}$。每个 $\Delta$ 给出上下两个截断点，$\widehat{\mathrm{FDR}}$ 按 (18.46) 照算；实际使用时扫一批 $\Delta$，画 $\Delta$–$\widehat{\mathrm{FDR}}$ 曲线再挑。例子里 $\Delta=0.71$ 给出 11 个显著基因，全部在右上角，左下角的点从不离开带，故 $C_{\mathrm{low}}=-\infty$：**SAM 不强制对称**。

> **推导** · (18.49) 的零分布标度：$\widehat{\mathrm{FDR}}$ 大约等于 $\Delta$（单侧）
>
> 1. 设全零假设成立，$t_1,\dots,t_M$ i.i.d. $F_0$，密度 $g$。大样本下 $t_{(j)}$ 与 $t^k_{(j)}$ 都集中在分位点附近：$t_{(j)}\approx F_0^{-1}\big(j/M\big)$（第 $j$ 小），而 $\tilde t_{(j)}$ 是 $K$ 个同分布顺序统计量的平均。
> 2. 两者之差的方差 $\approx\frac{2}{K}\mathrm{Var}\big(t^k_{(j)}\big)$，在 $j$ 接近端点时很大、$j\approx M/2$ 时很小。为看清结构，**固定 $j$、用正态近似**：设 $t_{(j)}-\tilde t_{(j)}\approx N(\tilde t_{(j)},\,\tau_j^2)$，则
>    $$
>    \Pr\big(t_{(j)}-\tilde t_{(j)}>\Delta\big)\approx\Delta\,g\big(\tilde t_{(j)}\big)\qquad(\Delta\ll\tau_j)
>    $$
>    因为 $\int_{\tilde t+\Delta}^\infty g = \int_{\tilde t+\Delta}^{\tilde t+\Delta+\tau}g + O(\tau^2)\approx\Delta g(\tilde t)$。
> 3. 对 $j$ 求和（只取上侧，总发现数 $R=M-j_0+1$）：
>    $$
>    \hat E[V](C_{\mathrm{hi}})\approx\sum_{j=j_0}^{M}\Delta\,g\big(\tilde t_{(j)}\big)\approx\Delta\sum_{j=j_0}^{M}g\big(t_{(j)}\big)\approx\Delta\cdot\frac{R}{M}\cdot\underbrace{\sum_{j=1}^{M}g\big(t_{(j)}\big)}_{\approx\,M\int g\,du=M}=\Delta R
>    $$
>    用了两个近似：$\sum_{j=1}^Mg(t_{(j)})\approx M\int g\,du=M$（顺序统计量密度和等于样本量的经典恒等式的离散版），以及尾部那 $R$ 个点各摊到 $\frac1M\int g=R/M$。
> 4. 于是
>    $$
>    \widehat{\mathrm{FDR}}\big(\Delta\big)=\frac{\hat E[V]}{R}\ \approx\ \Delta \quad\text{（单侧）},\qquad \approx 2\Delta\quad\text{（同时用上下两侧）}
>    $$
>    **这就是 $\Delta$ 的自然标度**：想控制 $\widehat{\mathrm{FDR}}\approx\alpha$，就取 $\Delta\approx\alpha$。原书让人「主观地」在一批 $(\Delta,\widehat{\mathrm{FDR}})$ 里选，本质上是因为第 2 步的正态近似在端点处失效——那里的 $\tau_j$ 很大，尾部概率不再线性于 $\Delta$。
> 5. **自洽性检验**：例子中 $\Delta=0.71$，若真有信号，则 $F_1$ 的厚尾让实际 $\widehat{\mathrm{FDR}}=0.14\ll\Delta$——说明**实际 FDR 低于 $\Delta$ 正是「带外有点离开了零分布」的证据**。这给出了选 $\Delta$ 的一个客观依据：找 $\widehat{\mathrm{FDR}}$ 曲线开始明显下沉的那个拐点。
> 6. **非对称性从哪来**：$C_{\mathrm{hi}}$ 与 $C_{\mathrm{low}}$ 分别是「$F_0^{-1}$ 的哪一端 + $\Delta$」，而 $\tilde t_{(j)}$ 本身就是 $F_0$ 的分位点函数，对偏斜的 $F_0$ 它在两端不等距，于是两个截断点自动不对称。与 (18.48) 完全对应：$\ell-\ell_0$ 在 $t$ 轴上的等高点集一般不是关于 0 对称的。

> **坑**
> - SAM 的 $\Delta$ 是**统计量的单位**（不是标准化的），所以换一批量纲不同的基因就必须重选 $\Delta$；BH 的阈值作用在 $p$ 值上，没有这个问题。
> - $C_{\mathrm{low}}=-\infty$ 这类「一侧完全不显著」的结果不是数值问题，而是 $\Delta$ 相对该侧零分布宽度太小的直接反映；此时应当只报告单侧结果，不要假装做了双侧检验。
> - $K$ 偏小时 $\tilde t_{(j)}$ 的抖动直接变成截断点的抖动：$\mathrm{sd}(\tilde t_{(j)})\propto1/\sqrt K$。图 18.20 里 $\Delta$ 很小的时候曲线很毛躁，就是这个原因。

## 18.7.3 FDR 的贝叶斯解释 {#s-18-7-3}

FDR 之所以难解释，是因为「$E[V/R]$」里的期望是对未知的 $t$ 分布取的。把 $t$ 看作一个两成分混合分布的抽样，$V/R$ 就有了后验概率的解释，见 <a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-18-7-3">原文 §18.7.3</a>。

### 正错误发现率

(18.43) 在 $\Pr(R=0)>0$ 时没有定义（分母可能为 0）。条件化掉这一情形得到**正**错误发现率：

$$
\mathrm{pFDR}=E\left[\frac{V}{R}\,\Bigm|\,R>0\right] \eqno{18.50}
$$

「positive」指只估计「有阳性发现时」的错误率。这是唯一有干净贝叶斯解释的版本。

### 两成分混合模型

设对每个基因 $j$ 引入隐变量 $Z_j=0$（$H_{0j}$ 为真）与 $Z_j=1$（为假），并假设 $(t_j,Z_j)$ 两两 i.i.d.，条件分布为

$$
t_j\mid Z_j\ \sim\ (1-Z_j)\cdot F_0+Z_j\cdot F_1 \eqno{18.51}
$$

即 $t_j$ 来自 $F_0$ 还是 $F_1$ 由 $Z_j$ 决定。令 $\Pr(Z_j=0)=\pi_0$，由全概率公式得边缘分布

$$
t_j\ \sim\ \pi_0\cdot F_0+(1-\pi_0)\cdot F_1 \eqno{18.52}
$$

> **推导** · (18.52) 只是全概率公式，但它是后面一切的关键
>
> $$
> \Pr(t_j\le c)=\Pr(Z_j=0)\Pr(t_j\le c\mid Z_j=0)+\Pr(Z_j=1)\Pr(t_j\le c\mid Z_j=1)=\pi_0F_0(c)+(1-\pi_0)F_1(c)
> $$
> 更一般的形式：$\pi_0f_0+(1-\pi_0)f_1$ 的**混合密度**。这个模型的可辨识性：只要 $F_0\ne F_1$，从 $t_j$ 的样本原则上可以同时估 $\pi_0,F_0,F_1$（但方差很大，所以实用中都退回去用 $\pi_0$ 的分位数估计）。

### 核心结果：pFDR 就是后验零概率

$$
\mathrm{pFDR}(\Gamma)=\Pr\big(Z_j=0\ \big|\ t_j\in\Gamma\big) \eqno{18.53}
$$

即「已知这个检验拒绝了，它其实没被拒绝的后验概率」。

> **推导** · (18.50) $\Rightarrow$ (18.53)：五步，每步都给出算式
>
> 记 $q_\Gamma=\Pr(Z_j=0\mid t_j\in\Gamma)$（**待证它就等于 $\mathrm{pFDR}$**）。固定一个拒绝域 $\Gamma$（例子里 $\Gamma=(-\infty,-4.10)\cup(4.10,\infty)$），并记
> $$
> R=\sum_{j=1}^{M}I(t_j\in\Gamma),\qquad V=\sum_{j=1}^{M}I(Z_j=0,\,t_j\in\Gamma),\qquad V_j=I(Z_j=0,\,t_j\in\Gamma)
> $$
> （故 $V=\sum_{j=1}^MV_j$）。**唯一的假设**是 (18.51) 的「各对 $(t_j,Z_j)$ i.i.d.」。
>
> **第 1 步（逐个 $j$ 拆开，纯代数）。** $V=\sum_jV_j$，所以
> $$
> \frac VR=\frac{1}{R}\sum_{j=1}^{M}V_j=\frac{1}{M}\sum_{j=1}^{M}\frac{M\,V_j}{R}\qquad (R>0)
> $$
> 取条件期望，用条件期望的线性性（预备知识 P2）：
> $$
> \mathrm{pFDR}=\mathbb E\left[\frac VR\,\Bigm|\,R>0\right]=\frac{1}{M}\sum_{j=1}^{M}\mathbb E\left[\frac{M\,V_j}{R}\,\Bigm|\,R>0\right] \qquad\text{(a)}
> $$
>
> **第 2 步（对 $R$ 条件化，$R$ 变成常数）。** 由事件 $\{R=1,\dots,M\}$ 划分 $\{R>0\}$，有
> $$
> \mathbb E\left[\frac{M V_j}{R}\,\Bigm|\,R>0\right]=\sum_{k=1}^{M}\mathbb E\left[\frac{M V_j}{R}\,\Bigm|\,R=k\right]\Pr(R=k\mid R>0)=\sum_{k=1}^{M}\frac{M}{k}\mathbb E\left[V_j\,\Bigm|\,R=k\right]\Pr(R=k\mid R>0) \qquad\text{(b)}
> $$
> 第二个等号用了：给定 $R=k$ 时 $R$ 是常数，可以提出分母。
>
> **第 3 步（交换对称性：$M$ 个位置等权）。** 给定 $R=k$，恰有 $k$ 个基因落在 $\Gamma$ 里，而 $(t_j,Z_j)$ i.i.d. 意味着这 $k$ 个位置在 $1,\dots,M$ 上**可交换**。因此每个位置的期望相等，而它们的和是 $V$：
> $$
> \sum_{j=1}^{M}\mathbb E\left[V_j\,\Bigm|\,R=k\right]=\mathbb E\left[V\,\Bigm|\,R=k\right]\ \Rightarrow\ M\,\mathbb E\left[V_j\,\Bigm|\,R=k\right]=\mathbb E\left[V\,\Bigm|\,R=k\right] \qquad(\forall k) \qquad\text{(c)}
> $$
>
> **第 4 步（二项：给定 $R=k$ 时 $V\sim\mathrm{Binomial}(k,q_\Gamma)$）。** 给定 $R=k$，被拒的 $k$ 个基因各自是零假设的概率都是同一个 $q_\Gamma$（i.i.d. 下的逐项后验），故
> $$
> V\mid R=k\ \sim\ \mathrm{Binomial}\big(k,q_\Gamma\big),\qquad \mathbb E\left[V\,\Bigm|\,R=k\right]=k\,q_\Gamma \qquad\text{(d)}
> $$
> 把 (c) 与 (d) 合起来代回 (b)：
> $$
> \mathbb E\left[\frac{M V_j}{R}\,\Bigm|\,R>0\right]=\sum_{k=1}^{M}\frac{M}{k}\cdot\frac{k\,q_\Gamma}{M}\Pr(R=k\mid R>0)=q_\Gamma\sum_{k=1}^{M}\Pr(R=k\mid R>0)=q_\Gamma \qquad\text{(e)}
> $$
> **$R$ 的整个分布被消掉了**，只因为 $q_\Gamma$ 与 $k$ 无关。
>
> **第 5 步（收尾）。** 代回 (a)：$\mathrm{pFDR}=\frac1M\sum_{j=1}^{M}q_\Gamma=q_\Gamma$。于是
> $$
> \mathrm{pFDR}=\mathbb E\left[\frac{V}{R}\,\Bigm|\,R>0\right]=\Pr\big(Z_j=0\mid t_j\in\Gamma\big),
> $$
> 即 (18.53)。∎

> **结果** · 为什么必须先条件掉 $R=0$
>
> 若目标改成 $\mathrm{FDR}=E[V/R]$（不条件化 $R=0$），第 2 步之后第 (b) 式里就多出一个 $1/k$：$\mathbb E[V_j/R\mid R=k]=\frac{1}{k}\mathbb E[V_j\mid R=k]=\frac{1}{k}\cdot\frac{kq_\Gamma}{M}=\frac{q_\Gamma}{k}$，对 $k$ 求和得不到常数，**消不掉**——这就是必须退回 (18.47)「期望之比」的根源。而 (18.50) 先把 $R=0$ 条件掉之后，每一项都塌成同一个 $q_\Gamma$，$\Pr(R>0)$ 与 $R$ 的分布一并消失。这条推导就是原书习题 18.20（编号 (18.60)(18.61)），见下节。

> **结果** · 后验形式（一步贝叶斯公式）
>
> $$
> q_\Gamma=\Pr(Z_j=0\mid t_j\in\Gamma)=\frac{\Pr(t_j\in\Gamma\mid Z_j=0)\Pr(Z_j=0)}{\Pr(t_j\in\Gamma)}=\frac{\pi_0 F_0(\Gamma)}{\pi_0 F_0(\Gamma)+(1-\pi_0)F_1(\Gamma)}
> $$
> 对**单点**区间同样成立：$\Pr(Z_j=0\mid t_j=t_0)=\frac{\pi_0f_0(t_0)}{\pi_0f_0(t_0)+(1-\pi_0)f_1(t_0)}$。这给出两个完全可算的局部指标：
>
> - **局部错误发现率**（Efron–Tibshirani 2002）：取 $\Gamma=\{t_0\}$ 的无穷小邻域，
>   $$
>   \mathrm{localFDR}(t_0)=\Pr\big(Z_j=0\mid t_j=t_0\big) \eqno{18.54}
>   $$
> - **$q$ 值**（Storey 2003）：使 $t_j$ 被拒的**所有**拒绝域中 FDR 的最小值。$\Gamma=\{\lvert T\rvert\ge2\}=\{-T\le-2\}\cup\{T\ge2\}$ 时，$t_j=5$ 的 $q$ 值小于 $t_j=2$ 的 $q$ 值，正如 $5$ 比 $2$ 更显著——FDR 是区域级指标、$q$ 值是基因级指标。

代入与 (18.45) 的对应：$F_0(\Gamma)$ 恰是「$\Gamma$ 的第一类错误率」，$\pi_0F_0(\Gamma)$ 是「总体里被错拒的比例」；$(1-\pi_0)F_1(\Gamma)$ 是「总体里被正确拒的比例」，见习题公式 (18.59)。

> **坑**
> - 模型 (18.51) 假设 $t_j$ 之间**独立**，而基因表达数据恰恰高度相关（第 18.2 节的动机）。混合模型对 $F_0,F_1$ 的单变量描述仍然可用，但「独立」这一步在 pFDR $\to$ FDR 的换算里是通过 $R$ 的求和用到的。
> - $\Pr(R=0)>0$ 时 (18.43) 无定义、(18.50) 才有定义；而 (18.45) 的界是给 (18.43) 的。两者混用会得到不同的常数。
> - $\pi_0$ 是**单个基因**为零假设的概率，与「$M_0$ 个零假设」不是同一个概念的比例化表达（虽然数值上 $\pi_0=M_0/M$）。用它做推断时必须记住每个基因的先验相同这一假设。

## 18.8 习题公式 {#s-18-exercises}

本章习题（<a class="src" href="../esl/ch18-high-dimensional-problems-p-n.html#s-exercises-17">原文 Exercises</a>）里的编号公式集中在这里。多数只需 2–5 句说明 + 一两行推导。

### 习题 18.2：lasso 化到最近收缩质心 (18.55)

原书习题公式，本手册给出其标准形式。朴素贝叶斯高斯模型 $x_{ij}\sim N(\mu_j+\mu_{jk},\sigma_j^2)$、$\sum_{k=1}^{K}\mu_{jk}=0$ 下的优化问题

$$
\min_{\{\mu_j,\mu_{jk}\}}\ \sum_{j=1}^{p}\left\{2\sigma_j^2+\lambda\sum_{k=1}^{K}\frac{1}{N_k}\sum_{i\in C_k}\frac{\big(x_{ij}-\mu_j-\mu_{jk}\big)^2}{\hat\sigma_j^2}\right\}\qquad \hat\sigma_j=s_j \eqno{18.55}
$$

（$\hat\sigma_j$ 是特征 $j$ 的类内合并标准差。）

> **推导** · 它就是最近收缩质心 (18.4)–(18.7)
>
> 1. **拆开残差平方**：$\frac{1}{N_k}\sum_{i\in C_k}(x_{ij}-\mu_j-\mu_{jk})^2=\frac{1}{N_k}\sum_{i\in C_k}(x_{ij}-\bar x_{kj})^2+(\bar x_{kj}-\mu_j-\mu_{jk})^2$。第一项只含数据。
> 2. **记 $y_{kj}=\bar x_{kj}-\bar x_j$**（类中心相对总均值的偏离，约束 $\sum_k\mu_{jk}=0$ 保证总均值项被吸收）。第二项 $=\sum_k(y_{kj}-\mu_{jk})^2$，在约束 $\sum_km_k^2d_{kj}^2=1$（即 $\sum_k\theta_k^2=1$，$\theta_k=m_kd_{kj}$）下最小化，加上 $\ell_1$ 惩罚 $\lambda\sum_k|\theta_k|$。
> 3. **一阶条件**：对 $\theta_k$ 求偏导得 $2(\theta_k-y_{kj})+\lambda\,\mathrm{sign}(\theta_k)-2\mu\theta_k=0$，即 $\theta_k=\dfrac{y_{kj}-\frac\lambda2\,\mathrm{sign}(\theta_k)}{1-\mu}$，其中 $\mu$ 是约束 $\sum_k\theta_k^2=1$ 的 Lagrange 乘子。把 $\frac{\lambda}{2(1-\mu)}$ 重新参数化为 $\tilde\lambda$（它只逐特征改变阈值大小），解就是**逐特征软阈值**
>    $$
>    \theta_{kj}=\Big(1-\frac{\tilde\lambda}{2\lvert y_{kj}\rvert}\Big)_{+}y_{kj}\ \Rightarrow\ d_{kj}=\frac{1}{m_k}\Big(1-\frac{\tilde\lambda}{2\lvert y_{kj}\rvert}\Big)_{+}y_{kj}
>    $$
>    其中 $1/s_j^2$ 只把 $\tilde\lambda$ 逐特征缩放，$2\sigma_j^2$ 只是常数（对 $\mu_j$ 求导给出 $\mu_j=\bar x_j$，与 (18.3) 的约束一致）。
> 4. 与 (18.5)(18.6)(18.7) 对照：$m_k^2=1/N_k$（原书取 $1/(N_k-1)$），$s_0=0$（不收缩）。

### 习题 18.7(b)：岭回归的 SVD 形式 (18.56)

$X=UDV^\top=RV^\top$，$R=UD$ 是 $N\times N$ 非奇异阵，$V$ 是 $p\times N$ 列正交阵。则

$$
\hat\beta_\lambda=V\big(R^\top R+\lambda I\big)^{-1}R^\top y \eqno{18.56}
$$

> **推导** · 由 (18.14) 代 $X^\top X=VR^\top RV$、$X^\top y=VR^\top y$（这一步就是预备知识 L2：$y$ 在 $\mathrm{col}(X)=\mathrm{col}(V)$ 上的投影用 $V$ 作正交基最省）：
> $X^\top X+\lambda I=V\big(R^\top R+\lambda I\big)V^\top$（因为 $V^\top V=I_N$，故 $V\big(\cdot\big)V^\top$ 的作用与 $\big(\cdot\big)$ 一致，且它是对称的幂等嵌入）。代入并把 $V^\top V$ 夹在中间消掉：$\hat\beta_\lambda=V\big(R^\top R+\lambda I\big)^{-1}V^\top V R^\top y=V\big(R^\top R+\lambda I\big)^{-1}R^\top y$。$\lambda>0$ 时括号内可逆（特征值 $\sigma_i^2+\lambda>0$），$\lambda=0$ 时给出零残差的最小范数解 $VD^{-1}U^\top y$。
>
> **意义**：$p\gg N$ 时只需做一次 $N\times N$ 的分解，10 折交叉验证（习题 18.12）就能复用它。

### 习题 18.14(a)：1-NN 作为判别函数 (18.57)

$$
\delta(x_0)=\log\frac{d_-(x_0)}{d_+(x_0)} \eqno{18.57}
$$

$d_\pm(x_0)$ 是 $x_0$ 到 $\pm1$ 类训练样本的最短距离。

> **推导** · 后验比 = 先验比 × 密度比
>
> $\hat\pi_+(x_0)=\frac{\hat\pi_+\hat f_+(x_0)}{\hat\pi_+\hat f_+(x_0)+\hat\pi_-\hat f_-(x_0)}$，判别函数取对数比 $\log\frac{\hat\pi_+\hat f_+}{\hat\pi_-\hat f_-}$。1-NN 的非参数密度估计给出 $\hat f_\pm(x_0)\propto\frac{1}{d_\pm(x_0)}$（倒距离核：最近邻越近，密度估计越大；题中给的归一化是 $\hat f_\pm(x_0)=N_\pm/d_\pm(x_0)$，其中 $N_\pm$ 与先验 $\hat\pi_\pm=N_\pm/N$ 一起约掉），因此
> $$
> \delta(x_0)=\log\frac{\pi_+}{\pi_-}+\log\frac{\hat f_+(x_0)}{\hat f_-(x_0)}=\log\frac{\pi_+}{\pi_-}+\log\frac{d_-(x_0)}{d_+(x_0)}
> $$
> 先验相等时就是 (18.57)；$d_+<d_-$ 判 $+1$，与 1-NN 一致。(b) 只需在右边加上 $\log\frac{\pi_+}{\pi_-}$；(c) 把 $d$ 换成到第 $K$ 近邻的距离。

### 习题 18.15：新样本的核主成分坐标 (18.58)

$(\mathbf I-M)\mathbf K(\mathbf I-M)=UD^2U^\top$、$\mathbf M=\mathbf 1\mathbf 1^\top/N$、$Z=UD^{-1}$，则新点 $x_0$ 的内积向量 $\mathbf k_0=\mathbf Xx_0$ 的（中心化）投影为

$$
z_0=D^{-1}U^\top(\mathbf I-M)\Big[\mathbf k_0-\mathbf K\frac{\mathbf 1}{N}\Big] \eqno{18.58}
$$

> **推导** · 逐项展开，每一项都能对上
>
> 1. 训练样本的主成分分数是对中心化内积矩阵做 $U^\top$ 投影再除以 $D$：$Z=U^\top(\mathbf I-M)\mathbf K\cdot D^{-1}$。新样本没有 $\mathbf M$ 的「行减去均值」这一步可用，因为它的行 $x_0$ 不在中心化训练集内。
> 2. 中心化的 $\mathbf k_0$：$\tilde{\mathbf k}_0=(\mathbf I-M)\big[\mathbf k_0-\mathbf K\frac{\mathbf1}{N}\big]$。第一项 $(\mathbf I-M)\mathbf k_0=\mathbf k_0-\frac{\mathbf 1\mathbf 1^\top}{N}\mathbf k_0$ 是「新内积减去它在训练集上的平均值」；第二项 $-(\mathbf I-M)\mathbf K\frac{\mathbf1}N=-\big(\mathbf K\frac{\mathbf1}N-\frac{\mathbf1\mathbf1^\top\mathbf K}{N^2}N\big)$，即「训练内积的列均值中心化」，恰好与第一项同型。
> 3. 验证均值一致性：若 $\mathbf k_0$ 本身是常数向量 $c\mathbf 1$（即新样本等于训练均值），则 $\mathbf k_0-\mathbf K\frac{\mathbf1}N=c\mathbf1-c\mathbf1=0$，故 $z_0=0$ —— 主成分分数平移到 0，符合「中心化」的定义。
> 4. 维度：$U^\top$ 是 $N\times N$，$(\mathbf I-M)[\cdot]$ 是 $N$ 维向量，$D^{-1}$ 对角，输出 $N$ 维，与训练时的 $Z$ 一致。整个算法只用内积，可推广到任意核矩阵。

### 习题 18.18：pFDR 的两类误差分解 (18.59)

$$
\mathrm{pFDR}=\frac{\pi_0\cdot\{\Gamma\text{ 的第一类错误率}\}}{\pi_0\cdot\{\Gamma\text{ 的第一类错误率}\}+\pi_1\cdot\{\Gamma\text{ 的功效}\}} \eqno{18.59}
$$

其中 $\pi_1=1-\pi_0$。

> **推导** · 由 (18.53) 一步得到
>
> $\Pr(Z=0\mid t\in\Gamma)=\frac{\Pr(t\in\Gamma\mid Z=0)\Pr(Z=0)}{\Pr(t\in\Gamma\mid Z=0)\Pr(Z=0)+\Pr(t\in\Gamma\mid Z=1)\Pr(Z=1)}$（贝叶斯公式；其渐近版本就是 $p$ 值，见预备知识 P5）。分子 $\pi_0F_0(\Gamma)$ 就是「第一类错误率 × 零假设比例」，分母第二项 $\pi_1F_1(\Gamma)$ 就是「功效 × 备择比例」。
>
> **可算性**：只要能估出 $\pi_0$，再用同一批 $t$ 算两组比例，第一类错误率用置换零分布估、功效用观测分布估，就能把 pFDR 直接算出来。这是 $q$ 值的标准实现（R 包 `qvalue`）。
>
> **推论**：把 $\Gamma$ 取遍所有包含 $t_0$ 的区间，最小值就是 $t_0$ 的 $q$ 值——「最小化 $q$ 值」等价于「最大化 (18.59) 的分式」，而分式在第一类错误率与功效之间做加权调和。

### 习题 18.20：证明 (18.53) (18.60)(18.61)

$$
\mathrm{pFDR}=\mathbb E\left[\frac{V}{R}\,\Bigm|\,R>0\right]=\frac{1}{M}\sum_{j=1}^{M}\mathbb E\left[\frac{M\,I\{Z_j=0,\ t_j\in\Gamma\}}{R}\,\Bigm|\,R>0\right] \eqno{18.60}
$$

$$
=\sum_{k=1}^{M}\mathbb E\left[\frac{V}{R}\,\Bigm|\,R=k\right]\Pr(R=k\mid R>0)=\sum_{k=1}^{M}q_\Gamma\,\Pr(R=k\mid R>0)=q_\Gamma \eqno{18.61}
$$

其中 $q_\Gamma=\Pr(Z_j=0\mid t_j\in\Gamma)$；两个等号分别来自全期望公式与「给定 $R=k$ 时 $V\sim\mathrm{Binomial}(k,q_\Gamma)$」。

> **推导** · (18.60) 只是恒等变形
>
> $V=\sum_{j=1}^M V_j$，$V_j=I\{Z_j=0,\,t_j\in\Gamma\}$，故 $\frac VR=\frac1M\sum_j\frac{M V_j}{R}$；取条件期望（线性性，预备知识 P2）即得 (18.60)。注意分式里的 $R$ 在事件为空时无定义，所以每一项都隐含 $I(R>0)$。
>
> **(18.61) 的两个等号**：
> - 第一个等号是全期望公式：$\mathbb E[\cdot\mid R>0]=\sum_k\mathbb E[\cdot\mid R=k]\Pr(R=k\mid R>0)$，事件 $\{R=1,\dots,M\}$ 划分了 $\{R>0\}$。
> - 关键一步（题中给的引理）：**给定 $R=k$，$V\sim\mathrm{Binomial}\big(k,\ \Pr(H_j=0\mid T_j\in\Gamma)\big)$**。理由：$R=k$ 意味着恰有 $k$ 个基因的 $t$ 落在 $\Gamma$ 里；由 (18.51) 的 i.i.d. 假设这 $k$ 个位置可交换，而它们每一个是零假设的后验概率都是同一个 $q_\Gamma$，故 $V\mid R=k$ 是 $k$ 次成功概率 $q_\Gamma$ 的伯努利之和。
> - 于是每个 $k$ 上 $\mathbb E[V/R\mid R=k]=\frac1k\cdot k\,q_\Gamma=q_\Gamma$。代回 (18.61)：$\sum_k q_\Gamma\Pr(R=k\mid R>0)=q_\Gamma\sum_k\Pr(R=k\mid R>0)=q_\Gamma$。
> - $R$ 的分布被完全消掉，这就是 $\mathrm{pFDR}=\Pr(Z_j=0\mid t_j\in\Gamma)$ 的完整证明，即 (18.53)。
>
> **为什么原书要引入 pFDR 而不是直接证 FDR**：如果目标是 $E[V/R]$，第 2 步换不来 —— $R$ 出现在分母上，条件化到 $R=k$ 后分母是常数 $k$，$\frac1k\sum_k$ 无法消掉 $k$；只能退回 (18.47) 的期望之比近似。pFDR 把 $R=0$ 的情形条件掉之后，每一项都变成同一个数，才有闭式。