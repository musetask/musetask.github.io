# 项目规则

## 目标屏幕（"移动端"指的就是这块屏幕）

- 物理分辨率：2800 × 1260，452 PPI（约 14 英寸笔记本屏）
- CSS 视口：浏览器按 DPR 渲染，**布局宽度约 1400 CSS px**（常见 DPR=2；即使按 DPR=1.5 算也只有约 1870 CSS px）
- 因此本项目里说的"移动端优先"= **先按 ~1400px 宽的视口把页面排好**，横向空间按"窄"来处理：
  - 正文栏宽不要超过 ~46rem（约 736px），让它在 1400px 视口下居中，两侧留白充足；
  - 不要出现横向滚动（`document.documentElement.scrollWidth` 必须等于视口宽度）；
  - 宽内容（代码块、公式、表格）一律在自身容器内 `overflow-x:auto` 横向滚动，不许撑破页面；
  - 任何 `@media (min-width: ...)` 的桌面增强断点都要 ≥ 1400px，否则会在目标屏幕上误触发。

## 验证方式

- 只在 1400 × 630 CSS px 视口下验证，不要跑一堆设备尺寸的测试矩阵。
- 用 `python3 -m http.server` 起本地静态服务，然后检查：
  `document.documentElement.scrollWidth === window.innerWidth`
