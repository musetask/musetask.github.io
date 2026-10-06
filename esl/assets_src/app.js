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

  /* ---------------------------------------------------------- lightbox */
  var box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = '<button class="close" type="button" aria-label="关闭">关闭</button><img alt="">';
  document.body.appendChild(box);
  var big = box.querySelector("img");
  function close() { box.classList.remove("open"); }
  box.addEventListener("click", function (e) {
    if (e.target.tagName !== "IMG") close();
  });
  box.querySelector(".close").addEventListener("click", close);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
  });

  function open(src, alt) {
    big.src = src;
    big.alt = alt || "";
    box.classList.add("open");
  }
  document.querySelectorAll(".figblock img, .eq img, a.zoom img").forEach(function (img) {
    img.addEventListener("click", function () {
      open(img.currentSrc || img.src, img.alt || "");
    });
  });

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
