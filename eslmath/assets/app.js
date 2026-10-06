(function () {
  "use strict";
  var root = document.documentElement;

  /* ---------------------------------------------------------- theme */
  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }
  var saved = load("esl-theme");
  if (saved) root.setAttribute("data-theme", saved);
  else if (window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches) {
    root.setAttribute("data-theme", "dark");
  }
  var tbtn = document.getElementById("themeToggle");
  if (tbtn) {
    tbtn.addEventListener("click", function () {
      var n = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", n);
      store("esl-theme", n);
    });
  }

  /* ---------------------------------------------------------- KaTeX */
  function renderMath() {
    if (typeof renderMathInElement !== "function") return;
    renderMathInElement(document.body, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false }
      ],
      ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code", "option"],
      throwOnError: false,
      errorColor: "#b41f24",
      strict: false,
      trust: false
    });
    document.documentElement.setAttribute("data-katex", "done");
  }
  if (typeof renderMathInElement === "function") renderMath();
  else addEventListener("DOMContentLoaded", renderMath);

  /* ---------------------------------------------------------- progress */
  var bar = document.getElementById("progress"), ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      if (!bar) return;
      var max = document.body.scrollHeight - innerHeight;
      bar.style.width = (max > 0 ? Math.min(100, Math.max(0, scrollY / max * 100)) : 0) + "%";
    });
  }
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("load", onScroll);
  onScroll();

  /* ---------------------------------------------------------- current heading */
  var heads = [].slice.call(document.querySelectorAll("main h1, main h2, main h3"));
  var titleEl = document.querySelector(".bar .title");
  if (heads.length && titleEl) {
    var label = function (h) {
      var num = h.querySelector(".num");
      var rest = h.cloneNode(true);
      var rnum = rest.querySelector(".num");
      if (rnum) rnum.parentNode.removeChild(rnum);
      var body = (rest.textContent || "").replace(/\s+/g, " ").trim();
      var pre = num ? (num.textContent || "").replace(/\s+/g, " ").trim() : "";
      return (pre ? pre + " · " : "") + body;
    };
    var pick = function () {
      var y = scrollY + 120, cur = heads[0];
      for (var i = 0; i < heads.length; i++) {
        if (heads[i].offsetTop <= y) cur = heads[i]; else break;
      }
      titleEl.textContent = label(cur);
    };
    var tid = null;
    addEventListener("scroll", function () {
      if (tid) return;
      tid = setTimeout(function () { tid = null; pick(); }, 120);
    }, { passive: true });
    pick();
  }
})();