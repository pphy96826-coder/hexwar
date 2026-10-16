// Worker: plays matches between two evaluation-weight sets (each side's AI scores positions with its own weights).
const {parentPort,workerData}=require('worker_threads');
const E=require('./engine')(workerData.html);
const MAX_ROUNDS=40;
function playGame(W,L,budget,map){
  E.newGame('hotseat',{map:map||'valley'}); let st=E.st;
  while(st.phase!=='over'&&st.round<=MAX_ROUNDS){
    const p=st.active;
    E.setEW(W[p]);
    const g=E.aiGen(p,L[p],budget); let r; do{r=g.next();}while(!r.done);
    for(const [t,k] of r.value) if(st.bank[p]>=E.U[t].cost) E.place(p,t,k);
    E.endTurn(); st=E.st;
  }
  return st.phase==='over'?st.winner:-1;
}
parentPort.on('message',job=>{
  let a=0,b=0,d=0;
  for(let g=0;g<job.games;g++){
    const aSide=(job.offset+g)%2;               // alternate who plays blue (moves first in round 1)
    const W=aSide===0?[job.A,job.B]:[job.B,job.A];
    const L=job.levels?(aSide===0?job.levels:[job.levels[1],job.levels[0]]):[job.level,job.level];
    const w=playGame(W,L,job.budget,job.map);
    if(w<0) d++; else if(w===aSide) a++; else b++;
  }
  parentPort.postMessage({a,b,d});
});
