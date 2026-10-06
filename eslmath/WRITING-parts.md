# 大章分片写法

> 第 14、18 章的原书 PDF 是**第一版**内容（本章的关联规则/自组织映射/NMF/PageRank，
> 第 18 章的 diagonal LDA/二次正则化）。这两章按第一版编号写，与本地 esl 站点逐条对得上。

有些章公式太多（130–200 个），单个 agent 读材料就会耗尽预算。所以这些章的
`src/NN-*.md` 只是外壳，正文由若干分片拼成。外壳文件负责 front matter、导语和
`<!--include ... -->` 行。

## 已有的分片安排

| 章 | 外壳 | 分片 |
|---|---|---|
| 3线性回归 | `03-linear-regression.md` | `03a-least-squares.md`(§3.2–3.4)、`03b-regularization-lars.md`(§3.5–3.6)、`03c-dimreduction-shrinkage.md`(§3.7–3.8) |
| 18 高维问题 | `18-highdim.md` | `18a-diagonal-quadratic-l1.md`(§18.2–18.4, (18.1)–(18.24))、`18b-kernels-spc-fdr.md`(§18.5–18.6, (18.25)–(18.37))、`18c-multipletesting.md`(§18.7+习题, (18.38)–(18.61)) |
| 14 无监督 | `14-unsupervised.md` | `14a-association-cluster.md`(§14.2–14.3, (14.1)–(14.46))、`14b-som-pca.md`(§14.4–14.5.1, (14.47)–(14.56))、`14c-curves-spectral.md`(§14.5.2–14.7.2, (14.57)–(14.89))、`14d-ica-mds-pagerank.md`(§14.7.3–14.10, (14.90)–(14.99)) |

## 分片约定

`src/14-unsupervised.md` 只是外壳，正文由三个分片拼成：

| 分片文件 | 覆盖内容 | 对应原书 |
|---|---|---|
| `src/14a-association-cluster.md` | 关联规则 + 聚类 | §14.2–14.3，(14.1)–(14.46) |
| `src/14b-som-pca.md` | 自组织映射 + 主成分 | §14.4–14.5.1，(14.47)–(14.56) |
| `src/14c-curves-spectral.md` | 主曲线 + 谱聚类 + 核PCA + 稀疏PCA + NMF + 因子分析 + ICA | §14.5.2–14.7.2，(14.57)–(14.89) |
| `src/14d-ica-mds-pagerank.md` | EPP + 直接 ICA + MDS + 非线性降维 + PageRank | §14.7.3–14.10，(14.90)–(14.99) |

拼装由 `build_site.py` 的 `<!--include ... -->` 完成。写分片时：

- **不要**写 front matter（`---` 开头那坨），外壳文件里已经有了。
- 一级标题不要写（`# 14 无监督学习` 已在外壳里），直接从 `## 14.2 …` 开始。
- 标题照常写 `{#id}`。
- 编号边界要衔接：A 结束在 (14.31)，B 从 (14.32) 开始，C 从 (14.57) 开始。
- 单独校验一个分片：
  ```
  python3 check.py 14a-pca-factor.md
  ```
