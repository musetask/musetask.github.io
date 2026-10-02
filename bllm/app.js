
(function(){
  "use strict";
  var root=document.documentElement;
  function store(k,v){ try{ localStorage.setItem(k,v); }catch(e){} }
  var tbtn=document.getElementById('themeToggle');
  function label(){
    if(tbtn) tbtn.title = root.getAttribute('data-theme')==='dark'?'Light mode':'Dark mode';
  }
  label();
  if(tbtn) tbtn.addEventListener('click',function(){
    var n=root.getAttribute('data-theme')==='dark'?'light':'dark';
    root.setAttribute('data-theme',n); store('pdf2html-theme',n); label();
  });
  document.addEventListener('keydown',function(e){
    if(e.key==='t' && !/input|textarea/i.test(e.target.tagName||'')) tbtn.click();
  });

  document.querySelectorAll('.codeblock').forEach(function(box){
    var pre=box.querySelector('pre'); if(!pre) return;
    var b=document.createElement('button');
    b.className='copy'; b.type='button'; b.textContent='Copy';
    b.addEventListener('click',function(){
      var text=pre.innerText.replace(/\n$/,'');
      var done=function(){ b.textContent='Copied';
        setTimeout(function(){ b.textContent='Copy'; },1500); };
      if(navigator.clipboard && navigator.clipboard.writeText){
        navigator.clipboard.writeText(text).then(done,function(){ legacy(text,done); });
      } else legacy(text,done);
    });
    box.appendChild(b);
  });
  function legacy(text,done){
    var ta=document.createElement('textarea');
    ta.value=text; ta.style.position='fixed'; ta.style.opacity='0';
    document.body.appendChild(ta); ta.select();
    try{ document.execCommand('copy'); done(); }catch(e){}
    document.body.removeChild(ta);
  }

  var bar=document.getElementById('progress'), ticking=false;
  function onScroll(){
    if(ticking) return;
    ticking=true;
    requestAnimationFrame(function(){
      ticking=false;
      var max=document.body.scrollHeight-innerHeight;
      if(bar) bar.style.width=(max>0?Math.min(100,Math.max(0,scrollY/max*100)):0)+'%';
    });
  }
  addEventListener('scroll',onScroll,{passive:true});
  addEventListener('load',onScroll);
  onScroll();

  var box=document.createElement('div');
  box.className='lightbox';
  box.innerHTML='<button class="close" aria-label="Close">Close</button><img alt="">';
  document.body.appendChild(box);
  var big=box.querySelector('img');
  box.addEventListener('click',function(e){
    if(e.target.tagName!=='IMG') box.classList.remove('open');
  });
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape') box.classList.remove('open');
  });
  document.querySelectorAll('figure img').forEach(function(img){
    img.addEventListener('click',function(){
      big.src=img.currentSrc||img.src; big.alt=img.alt||'';
      box.classList.add('open');
    });
  });
})();
