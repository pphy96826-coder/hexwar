const path=require('path');
const E=require('../../engine')(path.join(__dirname,'..','..','..','hexwar.html'));
const G=require('../game')(E), GW=require('./gameweb.js')(E), M=require('../mcts')(G), MW=require('./mctsweb.js')(GW);
E.loadMap('valley');
const fake=(s,idx)=>{ // deterministic pseudo network
  let h=s.round*31+s.units.length*7+s.placed*3+s.bank[0]*5+s.bank[1]*11; for(const u of s.units) h=(h*33+u.c*7+u.r+u.owner)%100003;
  return {v:((h%200)-100)/150, logits:Float32Array.from(idx,(x,i)=>((h*(i+3)+x)%97)/30)};
};
(async()=>{
  let s=GW.newState(); for(let i=0;i<14;i++){ const L=GW.legal(s); s=GW.apply(s,L[(i*7)%L.length]); }
  const r1=M.node(s); const g=M.think(r1,300,false); let x=g.next(); while(!x.done) x=g.next(fake(x.value.s,x.value.idx));
  const r2=MW.node(s); let sims=0; while(sims<301) sims+=await MW.step(r2,async L=>L.map(l=>fake(l.s,l.idx)),1);
  const a=Array.from(r1.N), b=MW.visits(r2);
  console.log('same visit counts (batch 1):', a.length===b.length&&a.every((v,i)=>v===b[i]), a.slice(0,10), b.slice(0,10));
  const r3=MW.node(s); sims=0; while(sims<301) sims+=await MW.step(r3,async L=>L.map(l=>fake(l.s,l.idx)),16);
  console.log('batch 16 best', MW.best(r3), 'batch 1 best', MW.best(r2), 'top visits', MW.visits(r3).sort((p,q)=>q-p).slice(0,5));
})();
