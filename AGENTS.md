# 项目规则

## 目标屏幕（"移动端"指的就是这块屏幕）

- "移动端"指且仅指**用户本人的手机**，不做其他手机/平板/小屏设备的兼容性测试。
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

## 内容编写方式（手写 HTML vs 构建生成）

本站各子项目历史不同，编写方式分三类。**一个站只用一种，不混写。**

| 方式 | 项目 | 怎么改 |
|---|---|---|
| 手写 HTML | bllm、barm、bllmmath、barmmath、rl-reasoning | 直接改 HTML，改完即发布，无构建步骤 |
| Markdown + 构建 | eslmath（`src/*.md` → `python3 build_site.py`）、islmath（`src/*.md` → `node tools/build.mjs`） | 只改 `src/` 下的 md，跑构建脚本重新生成，**不直接改产物 HTML** |
| body 片段 + 构建 | thellmmath（`_src/*.body.html` → `python3 build.py`） | 只改 `_src/`，跑构建脚本重新生成，**不直接改产物 HTML** |

build 是干啥的：把内容（md / 正文片段）套上统一外壳（顶栏、章节导航、深色模式等）生成完整页面；附带检查脚本（katex_check、html_check 等）自动验公式和链接。页面少、基本定型用手写；页面多、还在大量写用构建。

公式渲染方式：
- 除 islmath 外全站统一**客户端渲染**：页面里写 `\(...\)` / `\[...\]`，浏览器加载 `../assets/katex/`（全站唯一的 KaTeX 副本，0.16.11）后由 auto-render 排版。升级 KaTeX 只需替换该目录，不用改各站页面。
- islmath 是唯一的例外：**构建期离线渲染**（`node tools/build.mjs` 在构建时把公式预渲染成 HTML），产物里没有定界符、不加载 auto-render。改公式必须重跑构建。

铁律：生成式站点的产物 HTML 永远不手改；手写站点没有构建步骤，不要引入。
