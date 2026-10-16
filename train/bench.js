// Quick benchmark: weights vs the scripted rusher and vs each other.
//   node train/bench.js            (uses hexwar.html's current weights, train/weights.json if present, and the originals)
const {Worker}=require('worker_threads'), path=require('path'), fs=require('fs');
const HTML=path.join(__dirname,'..','hexwar.html'), NW=8;
const ORIG={iso:0.25,adv:0.6,mine:1.6,bank:0.7,over:0.6,die:0.4,statBase:0.5,pot:0.6,exp:0.5,hpLin:1.6,hpLow:0.9};
const workers=[...Array(NW)].map(()=>new Worker(path.join(__dirname,'worker.js'),{workerData:{html:HTML}}));
function match(A,B,games,budget,levels,map){
  const per=[...Array(NW)].map((_,i)=>Math.floor(games/NW)+(i<games%NW?1:0));
  return Promise.all(per.map((g,i)=>g?new Promise(res=>{const w=workers[i];w.once('message',res);w.postMessage({A,B,games:g,offset:i,level:'hard',levels,budget,map});}):{a:0,b:0,d:0}))
    .then(ps=>ps.reduce((s,p)=>({a:s.a+p.a,b:s.b+p.b,d:s.d+p.d}),{a:0,b:0,d:0}));
}
module.exports={match,ORIG,workers};
if(require.main===module)(async()=>{
  const cur=require('./engine')(HTML).getEW(), G=+(process.argv[2]||16), B=+(process.argv[3]||600);
  for(const [name,W] of [['original',ORIG],['current',cur]]){
    const r=await match(W,W,G,B,['hard','rush']);
    console.log(`${name.padEnd(9)} hard vs rusher: AI ${r.a} - rusher ${r.b} - draw ${r.d}`);
  }
  workers.forEach(w=>w.terminate());
})();
