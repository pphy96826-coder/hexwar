// Base threats the network is blind to, used to correct the AI's position evaluation (browser and training tests).
// The key one: a battering ram next to a base, behind it, where the defender's forward-facing units cannot hit it.
// A player can lay a chain of cheap units in one turn and drop the ram at its end, so "can a ram be placed there
// right now" is not enough: we count how many chain units it would take.
function makeThreat(E){
  const RAM=5, CHAIN=1;   // costs: ram, cheapest chain unit (engineer: may also stand on rivers)
  // fewest chain placements for player q to get a ram next to the other base in state s (Infinity if impossible),
  // and the path (chain cells, then the ram cell)
  const D=new Int16Array(10000), P=new Int16Array(10000), Q=new Int16Array(40000);
  function ramReach(s,q,occ){
    occ=occ||E.buildOcc(s);
    const p=1-q, conn=E.connect(s,q,occ), B=E.BASE, TER=E.TER, U=E.U, cells=E.CELLS, NB=E.NB1;
    const free=k=>!occ.has(k)&&k!==B[0]&&k!==B[1]&&TER(k)!=='rock';
    // q's own units cut off from supply: reconnecting one makes its whole group a stepping stone for free
    const own=k=>{const v=occ.get(k); return !!v&&v.owner===q&&!U[v.type].boom&&!conn.has(k);};
    for(const c of cells){D[c.k]=32767;P[c.k]=-1;}
    let h=0,t=0;
    // 0-1 search: entering an empty cell costs one placement, entering one of q's cut-off units costs nothing
    for(const k of conn){D[k]=0;Q[t++]=k;}
    for(const c of cells) if(D[c.k]>1&&free(c.k)&&E.distK(c.k,B[q])<=2){D[c.k]=1;Q[t++]=c.k;}
    while(h<t&&t<39990){
      const k=Q[h++], d=D[k];
      for(const n of NB.get(k)){
        if(free(n)){ if(d+1<D[n]){D[n]=d+1;P[n]=k;Q[t++]=n;} }
        else if(d<D[n]&&own(n)){D[n]=d;P[n]=k;Q[t++]=n;}
      }
    }
    let best=Infinity, tgt=-1;
    for(const x of NB.get(B[p])){ const d=D[x]; if(d<32767&&free(x)&&TER(x)!=='river'&&d-1<best){best=d-1;tgt=x;} }
    if(tgt<0) return {n:Infinity,path:[]};
    const path=[tgt]; let k=tgt; while(P[k]>=0){k=P[k]; if(free(k)) path.unshift(k);}
    return {n:best,path};
  }
  // correction (in the hand evaluation's units) for player p in state s, after this turn resolves.
  // Each side's ram threat gets an arrival time (0 = hitting from the coming resolution, 1 = next round, ...) and a
  // finish time = arrival + rounds the ram needs to bring the other base down (4 a round). Only the faster threat
  // counts in full: a slower counter-attack does not cancel it (the AI used to race when it had to defend).
  function correction(s,p){
    const e=1-p, res=E.resolve(s), alive=s.units.filter(u=>!res.dead.has(u.id));
    const post=Object.assign({},s,{round:s.round+1,units:alive,placed:0}), occPost=E.buildOcc(post), occNow=res.occ;
    const hp=[s.hp[0]-res.baseDmg[0],s.hp[1]-res.baseDmg[1]];
    const first=(s.round-1)%2, toMove=s.mover?1-first:first;
    const T={};   // per attacker q: {t: threat strength, at: arrival time}
    for(const q of [p,e]){
      const o=1-q; T[q]={t:0,at:Infinity};
      if(hp[o]<=0) continue;
      const adj=E.NB1.get(E.BASE[o]);
      if(alive.some(u=>u.owner===q&&u.type==='ram'&&adj.includes(u.c*100+u.r))){ T[q]={t:1.2,at:0}; continue; }
      const bank1=Math.min(E.consts.BANK_CAP,s.bank[q]+E.income(s.round+1)+E.mines(post,q));
      const bank2=Math.min(E.consts.BANK_CAP,bank1+E.income(s.round+2));
      if(bank2<RAM) continue;
      const need=ramReach(post,q,occPost).n*CHAIN+RAM;
      if(need<=bank1) T[q]={t:1,at:1}; else if(need<=bank2) T[q]={t:0.45,at:2};
      // q may still deploy in this round, before it resolves: chain now from the current board, ram now or next turn
      if(q===toMove||(!s.mover&&q===1-first)){
        const now=s.bank[q], chain=ramReach(s,q,occNow).n*CHAIN;
        if(chain+RAM<=now) T[q]={t:1,at:0};
        else if(chain<=now&&RAM<=Math.min(E.consts.BANK_CAP,now-chain+E.income(s.round+1))&&T[q].at>1) T[q]={t:0.8,at:1};
      }
    }
    // damage per round once the ram is in: 4 from the ram plus what q's units already deal to that base
    const fin=q=>T[q].at+Math.ceil(Math.max(1,hp[1-q])/(4+res.baseDmg[1-q]));
    let c=0;
    for(const q of [p,e]){
      if(!T[q].t) continue;
      const o=1-q, sg=q===p?1:-1, other=1-q;
      // slower than the other side's threat: it will not land in time to matter much
      const slow=T[other].t&&fin(q)>fin(other)?0.25:1;
      // a ram behind a base keeps hitting it every round: worth several rounds of base damage
      c+=sg*slow*T[q].t*Math.min(hp[o],16)*1.1;
    }
    return c;
  }
  return {ramReach,correction};
}
if(typeof module!=='undefined') module.exports=makeThreat;
