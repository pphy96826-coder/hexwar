// Self-play / arena / imitation games. Started by train.py, which serves the network over this process's
// stdin/stdout with binary frames: [type:1][len:u32][payload]. Logs go to stderr.
//   node selfplay.js '<json config>'
const {Worker,isMainThread,parentPort,workerData}=require('worker_threads');
const path=require('path');
const HTML=path.join(__dirname,'..','..','hexwar.html');
if(isMainThread){
  const cfg=JSON.parse(process.argv[2]);
  const NW=cfg.workers||8;
  const out=process.stdout;
  const send=(type,buf)=>{const h=Buffer.alloc(5);h.write(type,0,'latin1');h.writeUInt32LE(buf.length,1);out.write(h);out.write(buf);};
  const log=m=>process.stderr.write(m+'\n');
  // --- batching inference requests from all workers ---
  // up to MAXQ batches in flight so the GPU works while the workers run their tree searches; the Python side
  // answers in order, so replies are matched first-in first-out
  const MAXQ=cfg.inflight||2;
  let queue=[], pending=[];
  const flush=()=>{
    if(!queue.length||pending.length>=MAXQ) return;
    if(pending.length&&queue.length<Math.ceil(NW/2)) return;   // keep batches reasonably large
    const batch=queue; queue=[]; pending.push(batch);
    let n=0, size=4; for(const b of batch) for(const r of b.reqs){n++;size+=5+r.x.length+r.idx.length*2;}
    const buf=Buffer.alloc(size); let o=0; buf.writeUInt32LE(n,o); o+=4;
    for(const b of batch) for(const r of b.reqs){
      buf.writeUInt8(r.slot,o++); buf.writeUInt8(r.H,o++); buf.writeUInt8(r.W,o++); buf.writeUInt16LE(r.idx.length,o); o+=2;
      Buffer.from(r.x.buffer,r.x.byteOffset,r.x.length).copy(buf,o); o+=r.x.length;
      Buffer.from(r.idx.buffer,r.idx.byteOffset,r.idx.length*2).copy(buf,o); o+=r.idx.length*2;
    }
    send('Q',buf);
  };
  let inbuf=Buffer.alloc(0);
  process.stdin.on('data',d=>{
    inbuf=Buffer.concat([inbuf,d]);
    while(inbuf.length>=5){
      const len=inbuf.readUInt32LE(1); if(inbuf.length<5+len) break;
      const type=inbuf.toString('latin1',0,1), pl=inbuf.subarray(5,5+len); inbuf=inbuf.subarray(5+len);
      if(type==='A'){
        let o=0; const batch=pending.shift();
        for(const b of batch){
          const res=[];
          for(const r of b.reqs){
            const v=pl.readFloatLE(o); o+=4;
            const lg=new Float32Array(r.idx.length); for(let i=0;i<lg.length;i++){lg[i]=pl.readFloatLE(o);o+=4;}
            res.push({v,logits:lg});
          }
          b.w.postMessage({type:'res',res},[]);
        }
        flush();
      }
    }
  });
  // --- workers ---
  const per=[...Array(NW)].map((_,i)=>Math.floor(cfg.games/NW)+(i<cfg.games%NW?1:0));
  let alive=0; const results=[];
  per.forEach((g,i)=>{ if(!g) return; alive++;
    const w=new Worker(__filename,{workerData:{cfg,games:g,wid:i,html:HTML,nw:NW}});
    w.on('message',m=>{
      if(m.type==='req'){queue.push({w,reqs:m.reqs});flush();}
      else if(m.type==='data') send('D',Buffer.from(m.buf));
      else if(m.type==='result') results.push(m.r);
      else if(m.type==='log') log(m.msg);
      else if(m.type==='done'){ if(--alive===0){ send('R',Buffer.from(JSON.stringify(results))); send('X',Buffer.alloc(0)); setTimeout(()=>process.exit(0),200);} }
    });
    w.on('error',e=>{log('worker error '+e.stack);process.exit(1);});
  });
} else {
  // ================= worker =================
  const {cfg,games,wid,html,nw:NWK}=workerData;
  const E=require('../engine')(html);
  const {randomMap}=require('./maps');
  const G=require('./game')(E);
  const M=require('./mcts')(G,{cpuct:cfg.cpuct});
  const TH=require('./web/threat.js')(E);
  const FIXED=['valley','plateau','canyon'];
  let resolveRes=null;
  parentPort.on('message',m=>{if(m.type==='res'&&resolveRes){const f=resolveRes;resolveRes=null;f(m.res);}});
  const ask=reqs=>new Promise(r=>{resolveRes=r;parentPort.postMessage({type:'req',reqs});});
  // a handcrafted AI turn: let the old engine plan on a copy of the state, then turn the plan into actions
  function handPlan(s,level,budget){
    const st=E.st, p=G.toMove(s);
    Object.assign(st,{round:s.round,first:(s.round-1)%2,active:p,phase:'deploy',units:s.units.slice(),roads:Object.assign({},s.roads),
      bank:s.bank.slice(),hp:s.hp.slice(),rage:s.rage.slice(),pending:s.pending.slice(),turnStack:[],hist:[],nextId:s.nextId});
    const g=E.aiGen(p,level,budget); let r; do{r=g.next();}while(!r.done);
    const acts=[]; let t=s;
    for(const [ty,k] of r.value){const a=E.TIDX[ty]*10000+k; if(G.legal(t).includes(a)){acts.push(a);t=G.apply(t,a);}}
    acts.push(G.END); return acts;
  }
  // scripted attacker for training: each pick takes the placement that adds the most damage to the enemy base, gets
  // closest to it and kills the most, minus what it loses; ends the turn when nothing scores. Some randomness.
  // flanker: the players' winning plan. Lay a chain of engineers to a cell next to the enemy base and drop a ram there
  // (in one turn if the money allows, else the chain now and the ram next turn); otherwise play like the striker.
  function flankerPlan(s,fallback){
    const p=G.toMove(s), rr=TH.ramReach(s,p);
    if(rr.n<Infinity&&rr.n<=s.bank[p]){
      const acts=[]; let cur=s;
      const put=(ty,k)=>{const a=E.TIDX[ty]*10000+k; if(G.legal(cur).includes(a)){acts.push(a);cur=G.apply(cur,a);return true;} return false;};
      for(let i=0;i<rr.n;i++) if(!put('engineer',rr.path[i])) break;
      put('ram',rr.path[rr.n]);
      if(acts.length){ acts.push(G.END); return acts; }
    }
    return fallback?fallback(s):strikerPlan(s);
  }
  function strikerPlan(s){
    const p=G.toMove(s), e=1-p, eb=E.BASE[e], acts=[];
    const head=t=>{let m=99;for(const u of t.units)if(u.owner===p)m=Math.min(m,E.distK(u.c*100+u.r,eb));return m;};
    let cur=s;
    for(let pick=0;pick<8;pick++){
      const r0=E.resolve(cur), h0=head(cur); let best=null;
      const L=G.legal(cur).filter(a=>a!==G.END);
      for(const a of L.length>120?L.filter(()=>Math.random()<120/L.length):L){
        const n=G.apply(cur,a), r=E.resolve(n), u=n.units[n.units.length-1], d=E.U[u.type];
        let kills=0; for(const v of n.units) if(v.owner===e&&r.dead.has(v.id)&&!r0.dead.has(v.id)) kills+=E.U[v.type].cost;
        const sc=4*(r.baseDmg[e]-r0.baseDmg[e])+2*(h0-head(n))+1.5*kills-(r.dead.has(u.id)?1.2*d.cost:0)-0.3*d.cost+Math.random()*0.6;
        if(!best||sc>best.sc) best={sc,a,n};
      }
      if(!best||best.sc<=0) break;
      acts.push(best.a); cur=best.n;
    }
    acts.push(G.END); return acts;
  }
  // one game as a generator yielding network requests {s,idx,slot}
  function* game(sides){   // sides[p] = {slot} for the network, or {hand:level}
    let s=G.newState(); const roots=[null,null]; const samples=[]; let decisions=0;
    while(!s.over){
      const p=G.toMove(s), side=sides[p];
      if(side.hand){
        const record=cfg.mode==='imitate';
        for(const a of (side.hand==='striker'?strikerPlan(s):side.hand==='flanker'?flankerPlan(s):side.hand==='flankmaster'?flankerPlan(s,x=>handPlan(x,'master',side.budget||1500)):handPlan(s,side.hand,side.budget||300))){
          if(record){ const L=G.legal(s), idx=L.map(x=>{const j=G.actIndex(s,x);return j<0?65535:j;}); const e=G.encode(s);
            samples.push({x:e.x,H:e.H,W:e.W,p,idx,pi:L.map(x=>x===a?1:0)}); }
          s=G.apply(s,a); if(s.over) break;
        }
        roots[0]=roots[1]=null; continue;
      }
      let root=roots[p]&&roots[p].s===s?roots[p]:M.node(s);
      const sims=side.sims||cfg.sims;
      const gen=M.think(root,sims,!!cfg.noise);
      let r=gen.next();
      while(!r.done){
        const resp=yield {s:r.value.s,idx:r.value.idx,slot:side.slot};
        if(side.blend){   // mix in the hand-written evaluation (it sees base threats such as a ram behind the base)
          const ls=r.value.s, pm=G.toMove(ls), h=E.evalBoard(ls,pm,true)+(side.threat?TH.correction(ls,pm):0);
          resp.v=(1-side.blend)*resp.v+side.blend*Math.tanh(h/(side.kappa||20));
        }
        r=gen.next(resp);
      }
      if(cfg.mode==='selfplay'){ const pol=M.policy(root), e=G.encode(s); let n=0,w=0; for(let i=0;i<root.N.length;i++){n+=root.N[i];w+=root.W[i];}
        samples.push({x:e.x,H:e.H,W:e.W,p,idx:pol.idx,pi:pol.p,q:n?w/n:0}); }
      const temp=(cfg.mode==='selfplay'&&s.round<=(cfg.tempRounds||3))?1:(cfg.mode==='arena'&&decisions<(cfg.arenaRandom||2)?1:0);
      const i=M.choose(root,temp); decisions++;
      const nxt=M.child(root,i);
      // keep the subtree for whoever moves next if it is the same player; otherwise start fresh next time
      roots[p]=nxt; roots[1-p]=null; s=nxt.s;
    }
    return {winner:s.winner,rounds:s.round,hp:s.hp,samples};
  }
  function packSamples(samples,winner){
    const meta=[], parts=[];
    for(const sm of samples){ meta.push({H:sm.H,W:sm.W,z:winner<0?0:(winner===sm.p?1:-1),q:sm.q===undefined?undefined:+sm.q.toFixed(4),idx:sm.idx,pi:sm.pi.map(x=>+x.toFixed(4))}); parts.push(Buffer.from(sm.x.buffer,sm.x.byteOffset,sm.x.length)); }
    const js=Buffer.from(JSON.stringify(meta)), h=Buffer.alloc(4); h.writeUInt32LE(js.length,0);
    return Buffer.concat([h,js,...parts]);
  }
  // puzzle mode: from a given position, let sides[0] play one whole turn; report its placements and the state after
  function* oneTurn(s0,side){
    let s=s0; const p=G.toMove(s0), acts=[]; let root=M.node(s);
    while(!s.over&&G.toMove(s)===p&&s.mover===s0.mover&&s.round===s0.round){
      const gen=M.think(root,side.sims,false); let r=gen.next();
      while(!r.done){
        const resp=yield {s:r.value.s,idx:r.value.idx,slot:side.slot};
        if(side.blend){ const ls=r.value.s, pm=G.toMove(ls), h=E.evalBoard(ls,pm,true)+(side.threat?TH.correction(ls,pm):0);
          resp.v=(1-side.blend)*resp.v+side.blend*Math.tanh(h/(side.kappa||20)); }
        r=gen.next(resp);
      }
      const i=M.choose(root,0), a=root.acts[i]; acts.push(a);
      root=M.child(root,i); s=root.s; if(a===G.END) break;
    }
    return {acts,s};
  }
  if(cfg.mode==='puzzle'){
    (async()=>{
      for(let i=wid;i<cfg.puzzles.length;i+=NWK){
        const pz=cfg.puzzles[i]; E.loadMap(pz.map);
        const g=oneTurn(pz.s,cfg.sides[0]); let x=g.next();
        while(!x.done){ const e=G.encode(x.value.s); const [res]=await ask([{slot:x.value.slot,H:e.H,W:e.W,x:e.x,idx:x.value.idx}]); x=g.next(res); }
        const {acts,s}=x.value, p=G.toMove(pz.s), q=1-p;
        // after the AI's turn: can the opponent still get a ram next to the AI base (now, or after this round resolves)?
        const res=E.resolve(s), post=Object.assign({},s,{units:s.units.filter(u=>!res.dead.has(u.id)),round:s.round+1});
        const now=TH.ramReach(s,q), later=TH.ramReach(post,q);
        parentPort.postMessage({type:'result',r:{id:pz.id,placed:acts.filter(a=>a!==G.END).map(a=>E.TYPES[Math.floor(a/10000)]+'@'+(a%10000)),
          chainNow:now.n,chainNext:later.n,oppBank:s.bank[q],corr:+TH.correction(s,p).toFixed(1)}});
      }
      parentPort.postMessage({type:'done'});
    })();
  } else
  (async()=>{
    let left=games, gi=0;
    // cfg.mapList: play cfg.perMap games on each listed map (map ids or map objects), maps shared out over the workers
    const myMaps=cfg.mapList?cfg.mapList.filter((_,i)=>i%NWK===wid):null;
    if(myMaps) left=myMaps.length*cfg.perMap;
    while(left>0){
      // one wave: up to cfg.parallel games sharing one map (the engine holds one map per worker)
      const n=myMaps?cfg.perMap:Math.min(left,cfg.parallel||16); left-=n;
      const mp=myMaps?myMaps.shift():cfg.maps==='fixed'?FIXED[(wid+gi)%3]:(Math.random()<(cfg.fixedShare??0.35)?FIXED[Math.floor(Math.random()*3)]:randomMap());
      E.loadMap(mp); const mapName=typeof mp==='string'?mp:mp.id;
      const runs=[];
      for(let j=0;j<n;j++,gi++){
        const swap=(wid+gi)%2===1, sides=cfg.sides.map(x=>({...x})); if(swap) sides.reverse();
        runs.push({g:game(sides),swap,req:null,done:false});
      }
      for(const r of runs){const x=r.g.next(); if(x.done){r.done=true;r.res=x.value;} else r.req=x.value;}
      while(runs.some(r=>!r.done)){
        const act=runs.filter(r=>!r.done);
        const reqs=act.map(r=>{const e=G.encode(r.req.s);return {slot:r.req.slot,H:e.H,W:e.W,x:e.x,idx:r.req.idx};});
        const res=await ask(reqs);
        act.forEach((r,j)=>{const x=r.g.next(res[j]); if(x.done){r.done=true;r.res=x.value;} else r.req=x.value;});
      }
      for(const r of runs){
        const res=r.res, aWinner=res.winner<0?-1:((res.winner===0)!==r.swap?0:1);   // 0 = sides[0] won
        parentPort.postMessage({type:'result',r:{map:mapName,winner:aWinner,rounds:res.rounds,pw:res.winner}});   // pw: winning player (0 = 青, first in round 1)
        if(res.samples.length){ const b=packSamples(res.samples,res.winner); parentPort.postMessage({type:'data',buf:b},[b.buffer]); }
      }
    }
    parentPort.postMessage({type:'done'});
  })().catch(e=>{parentPort.postMessage({type:'log',msg:'worker crash '+e.stack});process.exit(1);});
}
