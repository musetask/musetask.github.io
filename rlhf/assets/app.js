// The RLHF Book — reading chrome: theme, progress, copy, lightbox.
(function () {
  'use strict';

  // ---- theme
  var root = document.documentElement;
  var btn = document.getElementById('theme');
  if (btn) {
    btn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('rlhf-theme', next); } catch (e) {}
      btn.textContent = next === 'dark' ? '☀' : '☽';
    });
    btn.textContent = root.getAttribute('data-theme') === 'dark' ? '☀' : '☽';
  }

  // ---- reading progress
  var bar = document.getElementById('progress');
  if (bar) {
    var tick = function () {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? (h.scrollTop / max) * 100 : 0) + '%';
    };
    addEventListener('scroll', tick, { passive: true });
    addEventListener('resize', tick);
    tick();
  }

  // ---- copy code
  document.querySelectorAll('.codeblock .copy').forEach(function (b) {
    b.addEventListener('click', function () {
      var pre = b.closest('.codeblock').querySelector('pre');
      if (!pre) return;
      var text = pre.innerText;
      var done = function () {
        b.textContent = '已复制';
        b.classList.add('done');
        setTimeout(function () {
          b.textContent = '复制';
          b.classList.remove('done');
        }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else {
        fallback();
      }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand('copy'); done(); } catch (e) {}
        document.body.removeChild(ta);
      }
    });
  });

  // ---- lightbox
  var lb = document.getElementById('lb');
  if (lb) {
    var img = lb.querySelector('img');
    var close = function () {
      lb.hidden = true;
      img.removeAttribute('src');
    };
    document.addEventListener('click', function (e) {
      var a = e.target.closest('a.zoom');
      if (a && a.href) {
        e.preventDefault();
        img.src = a.href;
        lb.hidden = false;
        return;
      }
      if (!lb.hidden && (e.target === lb || e.target.closest('.x'))) close();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && !lb.hidden) close();
    });
  }
})();