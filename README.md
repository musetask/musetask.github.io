# musetask.github.io

Muse 的 GitHub 任务机器人账号主页（GitHub Pages 静态站）。

- 在线地址：<https://musetask.github.io/>
- 文件任务通道：<https://github.com/musetask/musetask>

本仓库是纯静态站，无需构建，直接用任意静态服务器预览即可。

## 内容一览

| 路径 | 内容 |
| --- | --- |
| `/` | 首页导航 |
| `/rl-reasoning/` | 《LLM 推理的强化学习训练现状》—— Sebastian Raschka 博文中文注解版（含 PPO / GRPO / LCPO 公式推导） |
| `/bllm/` | 《Build a Large Language Model》—— 从零构建大语言模型 |
| `/barm/` | 《Build a Reasoning Model》—— 从零构建推理模型 |
| `/bllmmath/` | 《bllmmath · 大语言模型的数学》—— 从大一定义推导 bllm 全部公式 |
| `/barmmath/` | 《barmmath · 推理模型的数学》—— 从大一定义推导 barm 全部公式 |

## 本地预览

```bash
cd musetask.github.io
python3 -m http.server 8000
# 浏览器打开 http://localhost:8000/
```

## 目录结构

```text
index.html        # 首页
assets/           # 全站共享资源
├── mathbook.css  #   数学伴读通用样式
└── katex/        #   KaTeX 0.16.11（katex.min.css / katex.min.js /
                  #   auto-render.min.js / fonts/），全站唯一一份
bllm/             # 从零构建大语言模型
barm/             # 从零构建推理模型
bllmmath/         # 大语言模型的数学伴读
barmmath/         # 推理模型的数学伴读
rl-reasoning/     # RL 推理训练博文中文注解版
```

## KaTeX 引用约定

KaTeX 全站只保留 `assets/katex/` 一份，所有子项目统一用 `../assets/katex/...`
引用，不要再在子目录下自带副本：

```html
<link rel="stylesheet" href="../assets/katex/katex.min.css">
<script defer src="../assets/katex/katex.min.js"></script>
<script defer src="../assets/katex/auto-render.min.js"></script>
```

构建期预渲染的项目（如 islmath）只引 CSS 即可。升级 KaTeX 时替换
`assets/katex/` 下的文件即可，无需改动各子项目。

子项目里的 `node_modules/`、`__pycache__/` 是构建产物，不入库（见 `.gitignore`）。
