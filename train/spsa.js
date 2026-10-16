// Self-play tuning of the AI's evaluation weights with SPSA.
//   node train/spsa.js [--iters 100] [--games 16] [--budget 300] [--workers 8]
//   node train/spsa.js --validate [--games 40] [--budget 800]     (tuned weights vs the original ones)
// Each iteration nudges every weight up or down at random (in log space), plays the "+" version against the
// "-" version, and moves the weights toward whichever side won more. Results: train/weights.json, train/log.txt.
// Afterwards: node train/spsa.js --embed   writes train/weights.json into hexwar.html as the new defaults.
const {Worker}=require('worker_threads'), fs=require('fs'), path=require('path');
const arg=(k,d)=>{const i=process.argv.indexOf('--'+k);return i>0?+process.argv[i+1]:d;};
const flag=k=>process.argv.includes('--'+k);
const HTML=path.join(__dirname,'..','hexwar.html'), DIR=__dirname, OUT=path.join(DIR,'weights.json');
const NW=arg('workers',8), LEVEL='hard';
const logf=path.join(DIR,'log.txt');
const log=m=>{const l=`[${new Date().toLocaleTimeString()}] ${m}`;console.log(l);fs.appendFileSync(logf,l+'\n');};
const fmt=w=>Object.entries(w).map(([k,v])=>`${k}=${v.toFixed(2)}`).join(' ');

const E0=require('./engine')(HTML);
const BASE=E0.getEW();                         // weights currently in hexwar.html
if(flag('embed')){
  const w=JSON.parse(fs.readFileSync(OUT,'utf8')), r={};
  for(const k of Object.keys(BASE)) r[k]=+w[k].toFixed(3);
  const s=fs.readFileSync(HTML,'utf8'), out=s.replace(/\/\*EW\*\/[\s\S]*?\/\*END\*\//,'/*EW*/'+JSON.stringify(r).replace(/"/g,'')+'/*END*/');
  if(out===s) throw new Error('marker not found'); fs.writeFileSync(HTML,out); console.log('embedded',fmt(r)); process.exit(0);
}
const workers=[...Array(NW)].map(()=>new Worker(path.join(DIR,'worker.js'),{workerData:{html:HTML}}));
function match(A,B,games,budget){
  const per=[...Array(NW)].map((_,i)=>Math.floor(games/NW)+(i<games%NW?1:0));
  return Promise.all(per.map((g,i)=>g?new Promise(res=>{const w=workers[i];w.once('message',res);w.postMessage({A,B,games:g,offset:i,level:LEVEL,budget});}):{a:0,b:0,d:0}))
    .then(parts=>parts.reduce((s,p)=>({a:s.a+p.a,b:s.b+p.b,d:s.d+p.d}),{a:0,b:0,d:0}));
}
const keys=Object.keys(BASE);
const toW=phi=>Object.fromEntries(keys.map((k,i)=>[k,BASE[k]*Math.exp(phi[i])]));
(async()=>{
  if(flag('validate')){
    const W=JSON.parse(fs.readFileSync(OUT,'utf8')), G=arg('games',40), B=arg('budget',800);
    log(`validate: tuned vs original, ${G} games, budget ${B}ms`);
    const r=await match(W,BASE,G,B);
    log(`  tuned ${r.a} - original ${r.b} - draws ${r.d}  (${((r.a+r.d/2)/G*100).toFixed(0)}%)`);
    workers.forEach(w=>w.terminate()); return;
  }
  const ITERS=arg('iters',100), G=arg('games',16), B=arg('budget',300);
  let phi=keys.map(()=>0);
  if(fs.existsSync(OUT)&&flag('resume')){const w=JSON.parse(fs.readFileSync(OUT,'utf8'));phi=keys.map(k=>Math.log(w[k]/BASE[k]));}
  log(`SPSA: ${ITERS} iterations x ${G} games, budget ${B}ms, ${NW} workers`);
  log(`  start ${fmt(toW(phi))}`);
  const a0=0.6, c0=0.25, A=10;
  for(let k=0;k<ITERS;k++){
    const ak=a0/Math.pow(k+1+A,0.602), ck=c0/Math.pow(k+1,0.101);
    const delta=keys.map(()=>Math.random()<0.5?-1:1);
    const plus=toW(phi.map((x,i)=>x+ck*delta[i])), minus=toW(phi.map((x,i)=>x-ck*delta[i]));
    const r=await match(plus,minus,G,B);
    const score=(r.a-r.b)/G;                   // +1: "+" won everything, -1: "-" won everything
    phi=phi.map((x,i)=>Math.max(-1.6,Math.min(1.6,x+ak*score/(2*ck)*delta[i])));   // stay within ~5x of the start
    fs.writeFileSync(OUT,JSON.stringify(toW(phi)));
    log(`iter ${k+1}/${ITERS}: plus ${r.a} minus ${r.b} draw ${r.d} -> ${fmt(toW(phi))}`);
  }
  log('done; weights saved to train/weights.json');
  workers.forEach(w=>w.terminate());
})();
