/* 共用：KaTeX 宏定义 + 渲染封装。build.mjs 与 texcheck.mjs 都用它。 */
import katex from "katex";

export const MACROS = {
  // 数集与黑板体
  "\\R": "\\mathbb{R}",
  "\\E": "\\mathbb{E}",
  "\\N": "\\mathbb{N}",
  "\\I": "\\mathbb{I}",
  "\\Prob": "\\mathbb{P}",
  // 统计量
  "\\Var": "\\mathrm{Var}",
  "\\Cov": "\\mathrm{Cov}",
  "\\Cor": "\\mathrm{Cor}",
  "\\sd": "\\mathrm{sd}",
  "\\se": "\\mathrm{se}",
  "\\MSE": "\\mathrm{MSE}",
  "\\RSS": "\\mathrm{RSS}",
  "\\mse": "\\mathrm{mse}",
  "\\rss": "\\mathrm{rss}",
  "\\RSE": "\\mathrm{RSE}",
  "\\VIF": "\\mathrm{VIF}",
  "\\SST": "\\mathrm{SST}",
  "\\SSE": "\\mathrm{SSE}",
  "\\SSR": "\\mathrm{SSR}",
  "\\ind": "\\mathbf{1}",
  "\\Rank": "\\mathrm{rank}",
  "\\adjR": "\\overline{R^2}",
  "\\KL": "\\mathrm{KL}",
  "\\AUC": "\\mathrm{AUC}",
  // 常用记号
  "\\eps": "\\epsilon",
  "\\Eps": "\\epsilon",
  // 注意：不要添加自引用宏（如 "\\mu": "\\mu"），KaTeX 会无限展开报
  // "Too many expansions"；KaTeX 已内置 \mu \lambda \beta 等，直接写全名即可。
  // 统计分布
  "\\Norm": "\\mathcal{N}",
  "\\Bern": "\\mathrm{Bernoulli}",
  "\\Pois": "\\mathrm{Poisson}",
  "\\Unif": "\\mathrm{Uniform}",
};

export const OPTS = {
  throwOnError: true,
  output: "html",
  strict: "ignore",
  trust: false,
  macros: MACROS,
};

export function render(tex, display) {
  return katex.renderToString(tex, { ...OPTS, displayMode: !!display });
}

export { katex };
