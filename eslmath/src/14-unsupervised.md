---
id: m14
n: "14"
title: 无监督学习
title_en: Unsupervised Learning
desc: 关联规则与聚类的误差准则、自组织映射、主成分与主曲线、谱聚类与核 PCA、非负矩阵分解与独立成分分析、MDS 与 PageRank
prev: m13
next: m15
prev_title: 第 13 章 原型方法与最近邻
next_title: 第 15 章 随机森林
---

# 14 无监督学习 {#s-14}

本章讲「只有 $X$ 没有 $Y$」时能做什么。按原书顺序走四步：

- **§14.2–14.4 把数据分组**：关联规则（把购物篮变成监督学习）、聚类（最小化簇内偏差）、自组织映射（带拓扑约束的聚类）。
- **§14.5 把数据压扁**：主成分（方差/重构误差两条准则）、主曲线（流形的低维表示）、谱聚类（把图拉普拉斯当 PCA 用）、核 PCA、稀疏 PCA。
- **§14.6–14.7 拆成独立成分**：非负矩阵分解、原型分析、因子分析、独立成分分析、探索性投影寻优。
- **§14.8–14.10 保持距离**：多维标度、非线性降维（Isomap/LTSA）、Google PageRank。

数学上集中在四个工具：二次型极值与投影（预备知识 L1–L3）、极小割/图拉普拉斯（就地补定义）、EM 算法（O3）、熵与 KL 散度（P3）。

<!--include 14a-association-cluster.md -->
<!--include 14b-som-pca.md -->
<!--include 14c-curves-spectral.md -->
<!--include 14d-ica-mds-pagerank.md -->