// Browser copy of game.js (same states, actions, encoding), with per-map tables precomputed for speed.
// Verified byte-identical to game.js by test_gameweb.js.
function makeGameWeb(E){
  const END=-1, MAX_PLACE=8, MAX_ROUND=40, C=87;
  const U=E.U, TYPES=E.TYPES, NT=TYPES.length, {FINALE}=E.consts;
  const first=s=>(s.round-1)%2, toMove=s=>s.mover?1-first(s):first(s);
  function newState(){ return {round:1,mover:0,units:[],roads:{},bank:[E.income(1),E.income(1)],hp:[20,20],rage:[0,0],pending:[],nextId:1,placed:0,over:false,winner:null}; }
  function apply(s,a){
    if(a===END){
      if(!s.mover) return Object.assign({},s,{mover:1,placed:0});
      const n=E.simEndRound(s); n.placed=0;
      if(!n.over&&n.round>MAX_ROUND){n.over=true;n.winner=-1;}
      return n;
    }
    const p=toMove(s), t=Math.floor(a/10000), k=a%10000, ty=TYPES[t];
    const n=Object.assign({},s,{units:s.units.slice(),bank:s.bank.slice(),placed:s.placed+1,nextId:s.nextId+1});
    n.bank[p]-=U[ty].cost;
    if(s.roads[k]!==undefined){n.roads=Object.assign({},s.roads);delete n.roads[k];}
    n.units.push({id:s.nextId,type:ty,owner:p,c:E.KC(k),r:E.KR(k),born:s.round});
    return n;
  }
  function result(s,p){ return !s.over?0:(s.winner<0?0:(s.winner===p?1:-1)); }
  const legal=makeFastLegal(E,{toMove,END});
  // ---- per-map tables (rebuilt when the map changes) ----
  const TERR=['plain','forest','hill','river','bridge','mine','rock'];
  let M=null;
  function tables(){
    const {cols,rows}=E.dims(); const sig=cols+'x'+rows+':'+E.CELLS.length+':'+E.BASE[0]+':'+E.BASE[1]+':'+E.CELLS.map(c=>E.TER(c.k)[0]).join('');
    if(M&&M.sig===sig) return M;
    const W=cols, H=(rows+1)/2, HW=H*W, B=E.BASE;
    const pos=[new Int32Array(10000).fill(-1),new Int32Array(10000).fill(-1)];   // key -> cell index in the mover's frame
    for(const {c,r,k} of E.CELLS) for(const p of [0,1]){ let cc=c,rr=r; if(p===1){cc=cols-1-c;rr=rows+1-r;} pos[p][k]=Math.floor(rr/2)*W+cc; }
    const stat=[0,1].map(p=>{const x=new Uint8Array(C*HW), e=1-p;
      for(const {c,k} of E.CELLS){const i=pos[p][k];
        x[i]=1; x[HW+i]=c%2; x[(2+Math.max(0,TERR.indexOf(E.TER(k))))*HW+i]=1;
        if(k===B[p]) x[9*HW+i]=1; if(k===B[e]) x[10*HW+i]=1;
        x[73*HW+i]=Math.min(255,E.distK(k,B[p])); x[74*HW+i]=Math.min(255,E.distK(k,B[e]));}
      return x;});
    const idx=[0,1].map(p=>Int32Array.from(E.CELLS,c=>pos[p][c.k]));
    M={sig,W,H,HW,pos,stat,idx}; return M;
  }
  function encode(s){
    const T=tables(), {W,H,HW}=T, p=toMove(s), e=1-p, P=T.pos[p], x=T.stat[p].slice();
    const res=E.resolve(s);
    for(const k of res.covers[p]) if(P[k]>=0) x[65*HW+P[k]]=1;
    for(const k of res.covers[e]) if(P[k]>=0) x[66*HW+P[k]]=1;
    for(const k of E.deployable(s,p,'militia')) x[67*HW+P[k]]=1;
    for(const k of E.deployable(s,e,'militia')) x[68*HW+P[k]]=1;
    for(const kk in s.roads){const k=+kk, rd=s.roads[kk]; if(P[k]<0) continue; if(rd===p) x[69*HW+P[k]]=1; else if(rd===e) x[70*HW+P[k]]=1;}
    for(const pr of s.pending) x[(pr.owner===p?71:72)*HW+P[pr.c*100+pr.r]]=1;
    for(const u of s.units){
      const st=res.stats.get(u.id), d=U[u.type], i=P[u.c*100+u.r];
      x[((u.owner===p?11:34)+E.TIDX[u.type])*HW+i]=1;
      x[57*HW+i]=st.hp; x[58*HW+i]=st.atk; x[59*HW+i]=st.armor;
      if(st.isolated&&!d.drop) x[60*HW+i]=1; if(st.noAtk) x[61*HW+i]=1;
      if(d.revive&&!u.revived) x[62*HW+i]=1;
      x[63*HW+i]=Math.min(20,res.dmg.get(u.id)||0); if(res.dead.has(u.id)) x[64*HW+i]=1;
    }
    const nr=s.round+1, inc=q=>E.income(nr)+E.mines(s,q), uniq=q=>s.units.some(u=>u.owner===q&&U[u.type].unique)?1:0;
    const glob=[[75,s.bank[p]],[76,s.bank[e]],[77,Math.max(0,s.hp[p])],[78,Math.max(0,s.hp[e])],[79,Math.min(40,s.round)],
      [80,s.round>=FINALE?1:0],[81,s.mover?0:1],[82,inc(p)],[83,inc(e)],[84,uniq(p)],[85,uniq(e)],[86,s.placed]];
    const I=T.idx[p];
    for(const [ch,v] of glob){ if(!v) continue; const o=ch*HW, vv=v&255; for(let j=0;j<I.length;j++) x[o+I[j]]=vv; }
    return {x,H,W};
  }
  function actIndex(s,a){
    if(a===END) return -1;
    const T=tables(), t=Math.floor(a/10000), k=a%10000;
    return t*T.HW+T.pos[toMove(s)][k];
  }
  return {newState,legal,apply,encode,actIndex,toMove,result,END,C,NT,tables};
}
if(typeof module!=='undefined'){ module.exports=makeGameWeb; global.makeFastLegal=require('./fastlegal.js'); }
