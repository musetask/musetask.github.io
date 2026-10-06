/* ISLP web edition — 交互脚本（无依赖，离线可用） */
(function () {
  "use strict";
  var root = document.documentElement;

  /* ---------------------------------------------------------- 深色模式 */
  var tbtn = document.getElementById("themeToggle");
  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function labelTheme() {
    if (!tbtn) return;
    var dark = root.getAttribute("data-theme") === "dark";
    tbtn.textContent = dark ? "☀" : "◐";
    tbtn.setAttribute("aria-label", dark ? "切换浅色模式" : "切换深色模式");
  }
  labelTheme();
  if (tbtn) tbtn.addEventListener("click", function () {
    var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    store("islp-theme", next);
    labelTheme();
  });
  document.addEventListener("keydown", function (e) {
    if (e.key === "t" && !/input|textarea|select/i.test(e.target.tagName || "")) {
      tbtn.click();
    }
  });

  /* ---------------------------------------------------------- 阅读进度 */
  var bar = document.getElementById("progress");
  var top = document.getElementById("totop");
  var ticking = false;
  function onScroll() {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(function () {
      ticking = false;
      var max = document.documentElement.scrollHeight - innerHeight;
      if (bar) bar.style.width = (max > 0 ? Math.min(100, Math.max(0, scrollY / max * 100)) : 0) + "%";
      if (top) top.classList.toggle("on", scrollY > innerHeight * 1.5);
    });
  }
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", onScroll);
  onScroll();
  if (top) top.addEventListener("click", function () {
    scrollTo({ top: 0, behavior: "smooth" });
  });

  /* ---------------------------------------------------------- 复制代码 */
  document.querySelectorAll(".cb").forEach(function (box) {
    var pre = box.querySelector("pre");
    var b = box.querySelector(".copy");
    if (!pre || !b) return;
    b.addEventListener("click", function () {
      var text = pre.innerText.replace(/\n$/, "");
      var done = function () {
        b.textContent = "已复制";
        setTimeout(function () { b.textContent = "Copy"; }, 1400);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {
          legacy(text, done);
        });
      } else legacy(text, done);
    });
  });
  function legacy(text, done) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.cssText = "position:fixed;opacity:0;top:0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); done(); } catch (e) {}
    document.body.removeChild(ta);
  }

  /* ---------------------------------------------------------- 图片灯箱 */
  var lb = document.createElement("div");
  lb.className = "lb";
  lb.innerHTML = '<button type="button">关闭</button><img alt="">';
  document.body.appendChild(lb);
  var big = lb.querySelector("img");
  lb.addEventListener("click", function (e) {
    if (e.target.tagName !== "IMG") close();
  });
  lb.querySelector("button").addEventListener("click", close);
  function open(src, alt) {
    big.src = src;
    big.alt = alt || "";
    lb.classList.add("open");
    document.body.style.overflow = "hidden";
  }
  function close() {
    lb.classList.remove("open");
    document.body.style.overflow = "";
  }
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });
  document.querySelectorAll("a.zoom").forEach(function (a) {
    var img = a.querySelector("img");
    if (!img) return;
    a.addEventListener("click", function (e) {
      if (e.metaKey || e.ctrlKey || e.shiftKey) return;   // 允许新窗口打开原图
      e.preventDefault();
      open(img.currentSrc || img.src, img.alt || "");
    });
  });

  /* ---------------------------------------------------------- 目录搜索 */
  var q = document.getElementById("q");
  var toc = document.getElementById("toc");
  if (q && toc && typeof TOC !== "undefined") {
    var empty = document.getElementById("empty");

    function esc(s) {
      return s.replace(/[&<>"]/g, function (c) {
        return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
      });
    }
    function hl(text, term) {
      if (!term) return esc(text);
      var i = text.toLowerCase().indexOf(term);
      if (i < 0) return esc(text);
      return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + term.length)) +
        "</mark>" + esc(text.slice(i + term.length));
    }
    function draw(term) {
      var html = "";
      var shown = 0;
      for (var i = 0; i < TOC.length; i++) {
        var r = TOC[i];
        if (term && r.key.indexOf(term) < 0) continue;
        /* 公式条目（level 3）只在搜索时出现，避免首页列表过长 */
        if (!term && r.level > 2) continue;
        shown++;
        var aid = r.a || r.id;
        var href = r.file + (aid ? "#" + aid : "");
        html += '<li class="l' + r.level + '"><a href="' + href + '">' +
          (r.num ? '<span class="n">' + esc(r.num) + "</span>" : '<span class="n"></span>') +
          '<span class="t">' + hl(r.t, term) + "</span>" +
          '<span class="pg">' + esc(r.pages) + "</span></a></li>";
      }
      toc.innerHTML = html;
      if (empty) empty.hidden = shown !== 0;
    }
    q.addEventListener("input", function () {
      draw(q.value.trim().toLowerCase());
    });
    q.addEventListener("keydown", function (e) {
      if (e.key === "Escape") { q.value = ""; draw(""); }
      if (e.key === "Enter") {
        var a = toc.querySelector("a");
        if (a) location.href = a.getAttribute("href");
      }
    });
    draw("");
  }

  /* ---------------------------------------------------------- 标题栏高亮 */
  var heads = [].slice.call(document.querySelectorAll("main h2, main h3"));
  if (heads.length && "IntersectionObserver" in window) {
    var seen = new Set();
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) seen.add(en.target.id);
        else seen.delete(en.target.id);
      });
      var first = null;
      for (var i = 0; i < heads.length; i++) {
        if (seen.has(heads[i].id)) { first = heads[i]; break; }
      }
      var el = document.getElementById("barTitle");
      if (el && first) {
        el.textContent = first.getAttribute("data-t") || first.textContent;
      }
    }, { rootMargin: "-10% 0px -75% 0px" });
    heads.forEach(function (h) { io.observe(h); });
  }
})();