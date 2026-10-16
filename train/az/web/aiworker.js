// Neural-network AI for the page. Search workers (each: game engine + network on WebGPU + tree search) keep
// thinking in the background, also while the player deploys (pondering). The page talks to them via HexAI.
// Worker source = nn.js + fastlegal.js + gameweb.js + mctsweb.js + this file + the page's own script.

// ================= worker side =================
function HexAIWorker(runMain){
  const stubEl=()=>({innerHTML:'',textContent:'',hidden:true,style:{},dataset:{},classList:{toggle(){},add(){},remove(){}},
    setAttribute(){},addEventListener(){},closest(){return null},querySelectorAll(){return [];},querySelector(){return null;},appendChild(){},getContext(){return null;}});
  let E=null,G=null,M=null,net=null,cpuW=null,root=null,running=false,loopOn=false,sims=0,batch=24,blend=0,kappa=20,TH=null;
  function loadEngine(){
    const els={}; let api=null;
    const doc={querySelector:s=>els[s]||(els[s]=stubEl()),querySelectorAll:()=>[],addEventListener(){},body:stubEl(),createElement:stubEl,getElementById:s=>els[s]||(els[s]=stubEl()),scripts:[]};
    runMain(doc,{},()=>({matches:true,addEventListener(){}}),f=>{f();return 0;},()=>{},{},a=>{api=a;});
    return api;
  }
  const key=s=>JSON.stringify([s.round,s.mover,s.placed,s.hp,s.bank,s.rage,
    s.units.map(u=>[u.type,u.owner,u.c,u.r,u.revived?1:0]).sort(),s.pending.map(p=>[p.type,p.owner,p.c,p.r]).sort(),
    Object.keys(s.roads).sort().map(k=>[k,s.roads[k]])]);
  // move the root forward to state t, keeping the tree when t follows from placements by the current mover
  function sync(t){
    const tk=key(t);
    let nd=root;
    for(let d=0;nd&&d<12;d++){
      if(key(nd.s)===tk){root=nd;return true;}
      if(nd.term||nd.s.round!==t.round||nd.s.mover!==t.mover) break;
      M.prepare(nd);
      const have=new Set(nd.s.units.map(u=>u.type+'@'+(u.c*100+u.r)+'#'+u.owner));
      let bi=-1,bn=-1;
      for(let i=0;i<nd.acts.length;i++){
        const a=nd.acts[i]; if(a===G.END) continue;
        const ty=E.TYPES[Math.floor(a/10000)], k=a%10000;
        if(have.has(ty+'@'+k+'#'+nd.p)) continue;
        if(!t.units.some(u=>u.owner===nd.p&&u.type===ty&&u.c*100+u.r===k&&!u.revived&&u.born===t.round)) continue;
        const n=nd.N?nd.N[i]:0; if(n>bn){bn=n;bi=i;}
      }
      if(bi<0) break;
      nd=M.child(nd,bi);
    }
    root=M.node(t); return false;
  }
  async function evalBatch(list){
    const T=G.tables(), HW=T.HW, NT=G.NT, B=list.length, C=G.C, x=new Uint8Array(B*C*HW);
    list.forEach((l,b)=>x.set(G.encode(l.s).x,b*C*HW));
    const r=net?await net.forward(x,B,T.H,T.W):HexNN.forwardCPU(cpuW,x,B,T.H,T.W);
    if(!net) await new Promise(res=>setTimeout(res,0));
    return list.map((l,b)=>{
      const lg=new Float32Array(l.idx.length), o=b*NT*HW;
      for(let i=0;i<l.idx.length;i++){const j=l.idx[i]; lg[i]=j===65535?r.e[b]:r.lg[o+j];}
      let v=r.v[b];
      if(blend>0){ const pm=G.toMove(l.s), h=E.evalBoard(l.s,pm,true)+(TH?TH.correction(l.s,pm):0); v=(1-blend)*v+blend*Math.tanh(h/kappa); }   // hand-written threat-aware evaluation mixed in
      return {v,logits:lg};
    });
  }
  let think=null;   // {qid, t0, ms}: answer with the root's visit counts when time is up or the leader is safe
  function visitsMsg(qid){
    const r=root, out={type:'visits',qid,sims};
    if(r&&r.N){ out.acts=Array.from(r.acts); out.N=Array.from(r.N); out.Q=Array.from(r.N,(n,i)=>n?r.W[i]/n:0); out.v=r.v; }
    return out;
  }
  function checkThink(){
    if(!think) return;
    const el=performance.now()-think.t0; let done=el>=think.ms||!root||root.term;
    if(!done&&root&&root.N&&el>400){
      let n1=0,n2=0,tot=0; for(const n of root.N){tot+=n; if(n>n1){n2=n1;n1=n;} else if(n>n2) n2=n;}
      if(tot>150&&n1-n2>tot*(think.ms-el)/el*1.2) done=true;
    }
    if(done){ const q=think.qid; think=null; postMessage(visitsMsg(q)); }
  }
  async function loop(){
    if(loopOn) return; loopOn=true;
    try{
      while(running&&root&&!root.term){
        const n=await M.step(root,evalBatch,net?batch:4);
        sims+=n; checkThink(); if(!n) await new Promise(res=>setTimeout(res,5));
      }
    }catch(e){
      postMessage({type:'error',msg:String(e&&e.stack||e)});
      // drop the half-finished batch and get a fresh GPU device; the search goes on from the same tree
      try{ net=await HexNN.gpu(cpuW); }catch(err){ net=null; }
      loopOn=false; if(running) setTimeout(loop,50); checkThink(); return;
    }
    loopOn=false; checkThink();
  }
  onmessage=async e=>{
    const m=e.data;
    try{
      if(m.type==='init'){
        E=loadEngine(); G=makeGameWeb(E); M=makeMCTSWeb(G,{}); if(m.threat) TH=makeThreat(E);
        const W=HexNN.parse(m.weights); cpuW=W; batch=m.batch||24; blend=m.blend||0; kappa=m.kappa||20;
        try{ net=await HexNN.gpu(W); }catch(err){ net=null; }
        postMessage({type:'ready',gpu:!!net});
      } else if(m.type==='map'){ E.loadMap(m.map); root=null; }
      else if(m.type==='weights'){ const W=HexNN.parse(m.weights); cpuW=W; root=null; try{ net=await HexNN.gpu(W); }catch(err){ net=null; } postMessage({type:'reloaded',qid:m.qid,gpu:!!net}); }
      else if(m.type==='sync'){ const kept=sync(m.s); if(m.qid) postMessage({type:'synced',qid:m.qid,kept}); if(running) loop(); }
      else if(m.type==='syncEnd'){   // the player in m.s ends (or, when pondering, is assumed to end) the turn: sync to it, then step past END
        const tk=key(m.s);
        if(!(root&&root.preKey===tk)){ sync(m.s); M.prepare(root); const i=root.acts.indexOf(G.END); root=M.child(root,i); root.preKey=tk; }
        if(running) loop(); }
      else if(m.type==='go'){ running=true; loop(); }
      else if(m.type==='stop'){ running=false; }
      else if(m.type==='advance'){
        if(root){ M.prepare(root); const i=root.acts.indexOf(m.a); root=i>=0?M.child(root,i):M.node(G.apply(root.s,m.a)); }
        if(running) loop();
      }
      else if(m.type==='query'){ postMessage(visitsMsg(m.qid)); }
      else if(m.type==='think'){ think={qid:m.qid,t0:performance.now(),ms:m.ms}; running=true; loop(); checkThink(); if(think&&!loopOn) checkThink(); }
    }catch(err){ postMessage({type:'error',msg:String(err&&err.stack||err)}); if(m.qid) postMessage({type:'failed',qid:m.qid,sims}); }
  };
}

// ================= page side =================
const HexAI=(()=>{
  let last=null, workers=[], ready=false, gpu=false, failed=null, qid=0, waiters=new Map(), END=-1, mapId=null, aiEnded=false;
  const call=(w,msg,ms=15000)=>new Promise(res=>{const id=++qid;waiters.set(id,res);w.postMessage(Object.assign(msg,{qid:id}));
    setTimeout(()=>{ if(waiters.has(id)){ waiters.delete(id); res({sims:0}); } },ms);});
  const all=msg=>workers.forEach(w=>w.postMessage(msg));
  function start(libSrc,mainSrc,weights,nWorkers,opts={}){
    const src=libSrc+'\nHexAIWorker(function(document,window,matchMedia,setTimeout,clearTimeout,navigator,__HEXWAR_EXPORT){\n'+mainSrc+'\n});\n';
    const url=URL.createObjectURL(new Blob([src],{type:'text/javascript'}));
    let pending=nWorkers;
    return new Promise(resolve=>{
      for(let i=0;i<nWorkers;i++){
        let w; try{ w=new Worker(url); }catch(e){ failed='worker: '+e.message; resolve(false); return; }
        w.onmessage=e=>{
          const m=e.data;
          if(m.type==='ready'){ gpu=gpu||m.gpu; if(--pending===0){ready=true;resolve(true);} }
          else if(m.type==='error'){ console.warn('HexAI worker',m.msg); }
          else if(m.qid&&waiters.has(m.qid)){ const f=waiters.get(m.qid); waiters.delete(m.qid); f(m); }
        };
        w.onerror=e=>{ failed='worker error: '+(e.message||''); console.warn(failed); };
        w.postMessage({type:'init',weights:weights.slice(0),batch:24,blend:opts.blend||0,kappa:opts.kappa||20,threat:!!opts.threat});
        workers.push(w);
      }
    });
  }
  // live game state -> search state
  function toAZ(st){
    return {round:st.round,mover:st.active===st.first?0:1,units:st.units.map(u=>({id:u.id,type:u.type,owner:u.owner,c:u.c,r:u.r,born:u.born,revived:u.revived})),
      roads:Object.assign({},st.roads),bank:st.bank.slice(),hp:st.hp.slice(),rage:st.rage.slice(),pending:st.pending.map(p=>Object.assign({},p)),
      nextId:st.nextId,placed:st.units.filter(u=>u.owner===st.active&&u.born===st.round&&!u.revived).length,over:false,winner:null};
  }
  function setMap(id,mapObj){ if(!ready) return; mapId=id; aiEnded=false; all({type:"map",map:mapObj||id}); }
  async function syncAll(st){ const s=toAZ(st); await Promise.all(workers.map(w=>call(w,{type:'sync',s}))); }
  // while the player deploys: think about the position after "the player ends the turn now". After their last
  // placement, all of this thinking is about the position the AI will really face.
  function ponder(st){ if(!ready||st.phase!=='deploy') return; all({type:'syncEnd',s:toAZ(st)}); all({type:'go'}); }
  // the live game is about to end the current player's turn
  function onEnd(st){
    if(!ready) return;
    if(aiEnded){ aiEnded=false; return; }   // the AI already moved its own tree past this END
    all({type:'syncEnd',s:toAZ(st)}); all({type:'go'});
  }
  async function merged(msg){
    const rs=await Promise.all(workers.map(w=>call(w,Object.assign({type:'query'},msg||{}),(msg&&msg.ms||0)+5000)));
    const tot=new Map(); let sims=0;
    for(const r of rs){ sims+=r.sims; if(!r.acts) continue; r.acts.forEach((a,i)=>{const t=tot.get(a)||{n:0,w:0}; t.n+=r.N[i]; t.w+=r.N[i]*r.Q[i]; tot.set(a,t);}); }
    return {tot,sims};
  }
  // plan the whole turn for player p: one placement per decision, until the search prefers to end the turn.
  // The workers keep time themselves (page timers are throttled in background tabs).
  async function plan(st,TYPES,opts={}){
    const turnMs=opts.ms||8000, t0=performance.now(), out=[];
    await syncAll(st); all({type:'go'});
    const m0=await merged(), sims0=m0.sims; let reused=0; for(const [,t] of m0.tot) reused+=t.n;   // already searched while the opponent was thinking
    for(let d=0;d<9;d++){
      const left=turnMs-(performance.now()-t0), budget=Math.max(700,d===0?left*0.35:left*0.3);
      const {tot}=await merged({type:'think',ms:budget});
      const arr=[...tot].sort((a,b)=>b[1].n-a[1].n);
      const pick=arr.length?arr[0][0]:END;
      all({type:'advance',a:pick});
      if(pick===END){ aiEnded=true; break; }
      out.push([TYPES[Math.floor(pick/10000)],pick%10000]);
    }
    const {sims}=await merged(); last={sims:sims-sims0,reused:Math.round(reused),ms:performance.now()-t0,placed:out.length};
    return out;
  }
  // switch every worker to a newer network
  async function reload(weights){ if(!ready) return; await Promise.all(workers.map(w=>call(w,{type:'weights',weights:weights.slice(0)},30000))); aiEnded=false; }
  return {start,reload,setMap,ponder,onEnd,plan,get ready(){return ready;},get last(){return last;},get gpu(){return gpu;},get failed(){return failed;},stop(){all({type:'stop'});}};
})();
