# Build the page with the neural-network AI: python3 train/az/web/build.py <in.html> <net.bin> <out.html>
# Adds the AI library and the weights as extra <script> tags in front of the game's own <script>
# (the training engine reads the first plain <script> tag, so it keeps working on the output too),
# plus small hooks in the game script.
import sys, os, base64, re
HERE = os.path.dirname(os.path.abspath(__file__))
src, wpath, out = sys.argv[1], sys.argv[2], sys.argv[3]
html = open(src, encoding='utf8').read()
lib = '\n'.join(open(os.path.join(HERE, f), encoding='utf8').read().replace("if(typeof module!=='undefined')", "if(false)")
                for f in ['nn.js', 'fastlegal.js', 'gameweb.js', 'mctsweb.js', 'threat.js', 'aiworker.js'])
assert '</script' not in lib
b64 = base64.b64encode(open(wpath, 'rb').read()).decode()

def sub(old, new, count=1):
    global html
    assert html.count(old) >= 1, old[:80]
    html = html.replace(old, new, count)

# --- difficulty level ---
sub("const LEVELS={easy:'简单',normal:'普通',hard:'困难',master:'大师'};",
    "const LEVELS={easy:'简单',normal:'普通',hard:'困难',master:'大师',neural:'宗师'};")
sub("  if(lvl==='master') return planMCTS(p,budget*2.6,4);",
    "  if(lvl==='master'||lvl==='neural') return planMCTS(p,budget*2.6,4);   // neural falls back to master without WebGPU")
for seg in ['data-opt="lvl"', 'data-opt="w0"', 'data-opt="w1"']:
    i = html.index(seg); j = html.index('</div>', i)
    html = html[:j] + '<button data-v="neural" aria-pressed="false">宗师</button>' + html[j:]
sub('<li><b>大师</b>：',
    '<li><b>宗师</b>：神经网络 AI。通过上万局自我对弈学会这个游戏，对任何地图都适用；局面判断一半来自网络、一半来自大师的威胁计算（会算对方下回合能对大本营打多少伤害，以及对方铺几个兵就能把攻城锤送到大本营背后）；用显卡（WebGPU）搜索，你思考时它也在后台思考。每步约 8 秒。浏览器不支持 WebGPU 时自动改用大师。<span id="nn-status"></span></li>\n      <li><b>大师</b>：')
# --- hooks ---
sub("function maybeAI(){\n  if(thinking||st.phase!=='deploy'||!isAI(st.active)) return;\n  if(videoJob){setTimeout(maybeAI,500);return;}   // let a replay recording finish first\n",
    "function maybeAI(){\n  if(thinking||st.phase!=='deploy'||!isAI(st.active)) return;\n  if(videoJob){setTimeout(maybeAI,500);return;}   // let a replay recording finish first\n"
    "  if(st.aiLevels[st.active]==='neural'&&nnOn()){ nnTurn(); return; }\n")
sub("function endTurn(){\n  if(st.phase!=='deploy') return;\n",
    "function endTurn(){\n  if(st.phase!=='deploy') return;\n  if(nnGame()) HexAI.onEnd(st);\n")
sub("  st=fresh(mode,opts); sel=null; previewKey=null; inspectId=null; fx=null; thinking=false;\n",
    "  st=fresh(mode,opts); sel=null; previewKey=null; inspectId=null; fx=null; thinking=false;\n  if(nnOn()) HexAI.setMap(MAP.id,MAP);\n")
sub("function render(){\n  const v=computeView();\n",
    "function render(){\n  const v=computeView();\n  nnPonder();\n")
glue = r'''
/* ---------- neural-network AI (宗师) ---------- */
const nnOn=()=>typeof HexAI!=='undefined'&&HexAI.ready&&HexAI.gpu;
const nnGame=()=>nnOn()&&st&&st.mode!=='hotseat'&&st.aiLevels.some((l,p)=>l==='neural'&&st.aiSides[p]);
let nnLast='';
function nnPonder(){   // keep the search tree in step with the player's deployment and let it think meanwhile
  if(!nnGame()||st.phase!=='deploy'||isAI(st.active)) return;
  const k=st.round+':'+st.active+':'+st.units.length+':'+st.bank[st.active];
  if(k===nnLast) return; nnLast=k; HexAI.ponder(st);
}
function nnTurn(){
  const p=st.active, game=st; thinking=true; render();
  caption(`${NAMES[p]} 思考中…`,p);
  Promise.resolve(nnUpd).then(()=>HexAI.plan(st,TYPES,{ms:8000})).then(plan=>{   // a network update finishes first
    if(st!==game) return;
    let i=0; caption(null);
    const L=HexAI.last, e=document.getElementById('nn-status'); if(L&&e) e.textContent=`（上一回合：你思考时它已预先算好 ${L.reused} 个局面，轮到它后又搜索了 ${L.sims} 个，用时 ${(L.ms/1000).toFixed(1)} 秒）`;
    if(!plan.length) caption(`${NAMES[p]} 本回合不出兵`,p);
    const next=()=>{
      if(st!==game) return;
      if(i<plan.length){
        const [tp,k]=plan[i++];
        if(st.bank[p]>=U[tp].cost){place(p,tp,k);flash={k,p};caption(`${NAMES[p]} 部署 ${U[tp].n}（${i}/${plan.length}）`,p);render();}
        setTimeout(next,SP().place); return;
      }
      setTimeout(()=>{if(st!==game)return;flash=null;caption(null);thinking=false;endTurn();},SP().after);
    };
    next();
  });
}
let nnGen=null; const NNCFG={blend:0.5,kappa:20,threat:true};   // half network, half hand-written evaluation (+ the ram-behind-the-base check)
async function nnFetchNet(){
  try{
    const info=await (await fetch('/api/net-info',{cache:'no-store'})).json();
    if(info.gen==null||info.gen===nnGen) return null;
    const r=await fetch('/api/net',{cache:'no-store'}); if(!r.ok) return null;
    return {gen:info.gen,bin:await r.arrayBuffer()};
  }catch(e){ return null; }   // not served by the training server (e.g. the published page): keep the built-in network
}
let nnUpd=null;
function nnCheckUpdate(){ nnUpd=nnCheckUpdate1(); }
async function nnCheckUpdate1(){   // at each new game: switch to a newer promoted network if there is one
  if(!nnOn()) return; const live=await nnFetchNet(); if(!live) return;
  await HexAI.reload(live.bin); nnGen=live.gen; HexAI.setMap(MAP.id,MAP);
  const e=document.getElementById('nn-status'); if(e) e.textContent=`（已更新到第 ${nnGen} 代网络）`;
  toast(`宗师已更新到第 ${nnGen} 代`,-1);
}
function nnReport(){   // send every finished game against 宗师 to the training server, as training data
  if(!st||st.phase!=='over'||st.nnReported||!st.aiLevels.some((l,p)=>l==='neural'&&st.aiSides[p])||st.aiSides.every(x=>x)) return;
  st.nnReported=true;
  const rec={game:'蜂巢兵棋',v:1,map:MAP.id,mapDef:MAPS.some(m=>m.id===MAP.id)?undefined:MAP,mode:st.mode,aiSides:st.aiSides,aiLevels:st.aiLevels,
    round:st.round,hp:st.hp,winner:st.winner,nnGen,nnCfg:NNCFG,moves:st.hist};
  fetch('/api/game',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)}).catch(()=>{});
}
if(typeof __HEXWAR_EXPORT!=='function'&&typeof HexAI!=='undefined'){   // the page itself (not the training sandbox, not a search worker)
  (async()=>{
    const say=t=>{const e=document.getElementById('nn-status'); if(e) e.textContent=t;};
    say('（正在启动显卡…）');
    setTimeout(()=>{ if(!HexAI.ready) say('（显卡启动失败，宗师会改用大师）'); },20000);
    try{
      if(!navigator.gpu){ say('（当前浏览器不支持 WebGPU，宗师会改用大师）'); return; }
      let bin=Uint8Array.from(atob(document.getElementById('nnw').textContent.trim()),c=>c.charCodeAt(0)).buffer;
      const live=await nnFetchNet(); if(live){ bin=live.bin; nnGen=live.gen; }
      const main=[...document.scripts].find(s=>s.textContent.includes('__HEXWAR_EXPORT')&&s.id!=='nnlib').textContent;
      const n=Math.max(1,Math.min(4,Math.floor((navigator.hardwareConcurrency||4)/2)));
      const ok=await HexAI.start(document.getElementById('nnlib').textContent,main,bin,n,NNCFG);   // half network, half the hand-written threat-aware evaluation: 20-4 vs 10-14 against 大师
      if(!ok||!HexAI.gpu){ say('（无法启用显卡，宗师会改用大师）'); return; }
      say(`（显卡已就绪，${n} 个搜索线程${nnGen!=null?`，第 ${nnGen} 代网络（训练中，自动更新）`:''}）`);
      if(st) HexAI.setMap(MAP.id,MAP);
      if(st&&isAI(st.active)&&!thinking) maybeAI(); else nnLast='', nnPonder();
    }catch(e){ console.warn(e); say('（宗师启动失败，改用大师）'); }
  })();
}
'''
sub("function maybeAI(){", glue + "function maybeAI(){")
sub("function showOver(){", "function showOver(){\n  nnReport();")
sub("  if(nnOn()) HexAI.setMap(MAP.id,MAP);\n", "  if(nnOn()){ HexAI.setMap(MAP.id,MAP); nnCheckUpdate(); }\n")
if '--arena' in sys.argv:   # test build: ?arena=N plays N headless games, 宗师 vs 大师, sides and maps alternating
    sub("function maybeAI(){", r"""
async function nnArena(N,ms){
  const E={U,TYPES,resolve,deployable,simEndRound,distK,TIDX,buildOcc,connect,cover,income,mines,KC,KR,TER,dims:()=>({cols:COLS,rows:ROWS}),
    get CELLS(){return CELLS;},get BASE(){return BASE;},get NB1(){return NB1;},get NB2(){return NB2;},loadMap,consts:{FINALE,BANK_CAP,BASE_HP}};
  const G=makeGameWeb(E), out=document.createElement('pre'); out.id='arena-out'; out.style.cssText='position:fixed;left:0;top:0;z-index:99;background:#000;color:#0f0;font:12px monospace;max-height:50vh;overflow:auto;margin:0;padding:6px';
  document.body.appendChild(out); let w=0,l=0,d=0;
  const live=s=>({round:s.round,first:(s.round-1)%2,active:G.toMove(s),units:s.units,roads:s.roads,bank:s.bank,hp:s.hp,rage:s.rage,pending:s.pending,nextId:s.nextId,phase:'deploy'});
  for(let g=0;g<N;g++){
    const map=['valley','plateau','canyon'][g%3], nnSide=Math.floor(g/3)%2; loadMap(map); HexAI.setMap(map,MAPS.find(m=>m.id===map));
    let s=G.newState(), sims=0, turns=0, skipped=0, tNN=0, tM=0; const tg=performance.now();
    while(!s.over){
      const p=G.toMove(s), L=live(s);
      if(p===nnSide){
        const t1=performance.now(); const plan=await HexAI.plan(L,TYPES,{ms}); tNN+=performance.now()-t1; sims+=HexAI.last.sims; turns++;
        for(const [ty,k] of plan){ const a=TIDX[ty]*10000+k; if(G.legal(s).includes(a)) s=G.apply(s,a); else skipped++; }
        HexAI.onEnd(live(s)); s=G.apply(s,G.END);
      } else {
        Object.assign(st,{round:s.round,first:(s.round-1)%2,active:p,phase:'deploy',units:s.units.slice(),roads:Object.assign({},s.roads),
          bank:s.bank.slice(),hp:s.hp.slice(),rage:s.rage.slice(),pending:s.pending.slice(),turnStack:[],hist:[],nextId:s.nextId});
        const t2=performance.now(); const gen=aiGen(p,'master',1500); let r; do{r=gen.next();}while(!r.done); tM+=performance.now()-t2;
        for(const [ty,k] of r.value){ if(!U[ty]) continue; const a=TIDX[ty]*10000+k; if(G.legal(s).includes(a)) s=G.apply(s,a); }
        HexAI.onEnd(live(s)); s=G.apply(s,G.END);
        await new Promise(r=>{const c=new MessageChannel();c.port1.onmessage=r;c.port2.postMessage(0);});   // not throttled in background tabs
      }
    }
    if(s.winner===nnSide) w++; else if(s.winner<0) d++; else l++;
    out.textContent+=`game ${g+1} ${map} nn=${nnSide?'second':'first'}: ${s.winner===nnSide?'WIN':s.winner<0?'draw':'loss'} in ${s.round} rounds, hp ${s.hp}, avg ${Math.round(sims/Math.max(1,turns))} sims/turn, skipped ${skipped}, nn ${(tNN/Math.max(1,turns)/1000).toFixed(1)}s/turn, master ${(tM/Math.max(1,turns)/1000).toFixed(1)}s/turn, game ${((performance.now()-tg)/60000).toFixed(1)} min | total ${w}-${l}-${d}\n`;
  }
  out.textContent+='DONE\n';
}
function maybeAI(){""")
    sub("      if(st&&isAI(st.active)&&!thinking) maybeAI(); else nnLast='', nnPonder();",
        "      const q=new URLSearchParams(location.search); if(q.get('arena')){ nnArena(+q.get('arena'),+(q.get('ms')||8000)); return; }\n      if(st&&isAI(st.active)&&!thinking) maybeAI(); else nnLast='', nnPonder();")
# --- library and weights before the game script ---
i = html.index('<script>')
html = html[:i] + '<script type="application/octet-stream" id="nnw">' + b64 + '</script>\n<script id="nnlib">\n' + lib + '\n</script>\n' + html[i:]
open(out, 'w', encoding='utf8').write(html)
print('wrote', out, len(html), 'bytes')
