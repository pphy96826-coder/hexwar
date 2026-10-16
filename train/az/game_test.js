const path=require('path'); const E=require('../engine')(path.join(__dirname,'..','..','hexwar.html'));
const {randomMap}=require('./maps'); const G=require('./game')(E);
// 1) random playouts on random maps
let games=0,steps=0,wins=[0,0,0]; const t0=performance.now();
for(let g=0;g<60;g++){
  E.loadMap(g%4===3?'valley':randomMap()); let s=G.newState();
  while(!s.over){const L=G.legal(s);const a=Math.random()<0.25?G.END:L[Math.floor(Math.random()*L.length)];s=G.apply(s,a);steps++;if(steps%7===0)G.encode(s);}
  games++; wins[s.winner<0?2:s.winner]++;
}
console.log(`random playouts: ${games} games, ${steps} actions, wins ${wins.join('/')}, ${((performance.now()-t0)/steps).toFixed(3)} ms/action`);
// 2) symmetry: a mirrored position (owners swapped) seen by the other player must encode identically
E.loadMap('plateau'); let s=G.newState(); for(let i=0;i<30&&!s.over;i++){const L=G.legal(s);s=G.apply(s,L[Math.floor(Math.random()*L.length)]);}
const D=G.dims(), mir=u=>({...u,owner:1-u.owner,c:D.cols-1-u.c,r:D.rows+1-u.r});
const m={...s,units:s.units.map(mir),bank:s.bank.slice().reverse(),hp:s.hp.slice().reverse(),rage:s.rage.slice().reverse(),
  roads:Object.fromEntries(Object.entries(s.roads).map(([k,v])=>{const c=Math.floor(k/100),r=k%100;return [(D.cols-1-c)*100+(D.rows+1-r),1-v];})),
  pending:s.pending.map(mir)};
// the other player must be the one to move: same mover slot, round parity flipped (so first mover flips).
// Planes that depend on the round number itself (79 round, 80 finale, 82/83 income) are excluded.
for(const sh of [0]){
  const m2={...m,round:s.round+1,units:m.units.map(u=>({...u,born:u.born+1}))};
  const A=G.encode(s), B=G.encode(m2), HW=D.H*D.W, skip=new Set([79,80,82,83]);
  let diff=0, total=0, bad=new Map();
  for(let ch=0;ch<G.C;ch++){ if(skip.has(ch)) continue; for(let i=0;i<HW;i++){ total++; if(A.x[ch*HW+i]!==B.x[ch*HW+i]){diff++;bad.set(ch,(bad.get(ch)||0)+1);} } }
  console.log(`symmetry test: movers ${G.toMove(s)} vs ${G.toMove(m2)}, differing entries ${diff} of ${total}`, diff?[...bad]:'');
}
