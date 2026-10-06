---
id: m03
n: "3"
title: 线性回归方法
title_en: Linear Methods for Regression
desc: 最小二乘的正规方程与帽子矩阵、高斯误差下 t/F/χ² 分布的来源、QR 与 SVD 视角，以及 ridge / lasso / LARS 的正则化推导
prev: m02
next: m04
prev_title: 第 2 章 监督学习概述
next_title: 第 4 章 线性分类方法
---

# 3 线性回归方法 {#s-3}

本章是全书的数学主干：几乎所有后续方法都能看成这里某个技巧的变体（正则化、判别、降维、树、提升、神经网络、集成）。

三条主线：

- **§3.2–3.4 最小二乘的完整理论**：把 $\arg\min_\beta\|y-X\beta\|^2$ 一步步推到 $\hat\beta=(X^\top X)^{-1}X^\top y$，并说明这个解为什么同时是投影、为什么系数在正态误差下服从多元正态、$t$/$F$/$\chi^2$ 统计量从哪来。
- **§3.5–3.6 正则化与计算**：用 SVD 换一套坐标看最小二乘，ridge 与 lasso 的收缩效应、KKT 条件、LARS 算法。
- **§3.7–3.8 降维与收缩的统一**：PCR / PLS / CCA / 稀疏惩罚，都写成 $\hat\beta(\lambda)=\arg\min R(\beta)+\lambda J(\beta)$ 的特例。

<!--include 03a-least-squares.md -->
<!--include 03b-regularization-lars.md -->
<!--include 03c-dimreduction-shrinkage.md -->